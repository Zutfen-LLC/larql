//! Intrinsic attention-window policy and shared causal-range helper.
//!
//! Single source of truth for Gemma 4's per-layer local/global attention
//! semantics on the F32 CPU reference path.
//!
//! - [`intrinsic_attention_window`] derives the per-layer window from the
//!   model architecture (sliding → `Some(sliding_window)`, global → `None`).
//! - [`causal_attention_range`] turns a window into a concrete `[start, end)`
//!   K/V range for one query position.
//! - [`effective_window`] combines an architecture intrinsic window with a
//!   caller-supplied bounded window (the stricter of the two wins).
//!
//! These helpers are used by the F32 GQA prefill kernel, the dense
//! attention block, the prefill/decode dispatch path, and the diagnostic
//! attention captures so every code path observes the same semantics.

use larql_models::ModelArchitecture;

/// Half-open `[start, end_exclusive)` K/V range that a single query may
/// attend to under a (possibly windowed) causal policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionRange {
    pub start: usize,
    pub end_exclusive: usize,
}

impl AttentionRange {
    /// Number of keys in the range.
    pub fn len(&self) -> usize {
        self.end_exclusive - self.start
    }

    /// True when the range covers no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Compute the legal causal K/V range for `query_position`.
///
/// `available_keys` is the number of K/V rows currently available (cache
/// rows + the current token's own row, or the full prompt length during
/// prefill). `window` is `None` for global attention or `Some(W)` for a
/// sliding-window layer.
///
/// Semantics:
/// ```text
/// causal_end = min(query_position + 1, available_keys)
/// global:        start = 0
/// windowed W:    start = causal_end.saturating_sub(W)
/// range:         start .. causal_end
/// ```
///
/// `Some(0)` is rejected (a zero-width window is malformed); this is the
/// caller's responsibility — [`intrinsic_attention_window`] never returns
/// `Some(0)` and [`effective_window`] clamps caller-supplied zeros away.
pub fn causal_attention_range(
    query_position: usize,
    available_keys: usize,
    window: Option<usize>,
) -> AttentionRange {
    let causal_end = (query_position + 1).min(available_keys);
    let start = match window {
        Some(w) => {
            debug_assert!(
                w > 0,
                "causal_attention_range: Some(0) window is malformed — reject upstream"
            );
            let w_nonzero = w.max(1);
            causal_end.saturating_sub(w_nonzero)
        }
        None => 0,
    };
    AttentionRange {
        start,
        end_exclusive: causal_end,
    }
}

/// The architecture-driven attention window for a layer, if any.
///
/// Returns `Some(sliding_window)` for a layer flagged as sliding-window
/// by the architecture, or `None` for a global/full-attention layer or a
/// conventional architecture that has no intrinsic window.
///
/// Fails loudly (panics) when an architecture reports
/// `is_sliding_window_layer(layer) == true` but has no positive
/// `sliding_window` value — a malformed configuration must not silently
/// fall back to a guessed window.
pub fn intrinsic_attention_window(arch: &dyn ModelArchitecture, layer: usize) -> Option<usize> {
    if !arch.is_sliding_window_layer(layer) {
        return None;
    }
    match arch.sliding_window_size() {
        Some(w) if w > 0 => Some(w),
        other => panic!(
            "intrinsic_attention_window: layer {layer} is_sliding_window_layer=true but \
             sliding_window_size={other:?}; refusing to substitute a guessed window"
        ),
    }
}

/// Combine an architecture intrinsic window with a caller-supplied
/// bounded window.
///
/// ```text
/// intrinsic local window + no caller window:  intrinsic
/// no intrinsic window + caller window:        caller window
/// intrinsic window + caller window:           min(intrinsic, caller window)
/// neither:                                     unbounded (None)
/// ```
///
/// A caller may impose a *stricter* memory/attention bound than the
/// architecture, but a caller-supplied larger window never weakens
/// Gemma's intrinsic local limit. Global Gemma layers remain global when
/// no caller window is supplied.
pub fn effective_window(intrinsic: Option<usize>, caller: Option<usize>) -> Option<usize> {
    match (intrinsic, caller) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Effective window for a layer, combining the architecture intrinsic
/// window with an optional caller-supplied bounded window. Convenience
/// wrapper over [`intrinsic_attention_window`] + [`effective_window`].
pub fn effective_window_for_layer(
    arch: &dyn ModelArchitecture,
    layer: usize,
    caller_window: Option<usize>,
) -> Option<usize> {
    let intrinsic = intrinsic_attention_window(arch, layer);
    effective_window(intrinsic, caller_window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_range_covers_full_causal_prefix() {
        assert_eq!(
            causal_attention_range(0, 10, None),
            AttentionRange {
                start: 0,
                end_exclusive: 1
            }
        );
        assert_eq!(
            causal_attention_range(1024, 1025, None),
            AttentionRange {
                start: 0,
                end_exclusive: 1025
            }
        );
    }

    #[test]
    fn windowed_range_examples_for_w_512() {
        let w = Some(512usize);
        assert_eq!(
            causal_attention_range(0, 1025, w),
            AttentionRange {
                start: 0,
                end_exclusive: 1
            }
        );
        assert_eq!(
            causal_attention_range(510, 1025, w),
            AttentionRange {
                start: 0,
                end_exclusive: 511
            }
        );
        assert_eq!(
            causal_attention_range(511, 1025, w),
            AttentionRange {
                start: 0,
                end_exclusive: 512
            }
        );
        assert_eq!(
            causal_attention_range(512, 1025, w),
            AttentionRange {
                start: 1,
                end_exclusive: 513
            }
        );
        assert_eq!(
            causal_attention_range(513, 1025, w),
            AttentionRange {
                start: 2,
                end_exclusive: 514
            }
        );
        assert_eq!(
            causal_attention_range(1024, 1025, w),
            AttentionRange {
                start: 513,
                end_exclusive: 1025
            }
        );
    }

    #[test]
    fn window_one_attends_only_to_self() {
        let w = Some(1usize);
        for qi in 0..5 {
            let r = causal_attention_range(qi, 10, w);
            assert_eq!(
                r,
                AttentionRange {
                    start: qi,
                    end_exclusive: qi + 1
                }
            );
        }
    }

    #[test]
    fn range_clamps_to_available_keys() {
        // available_keys < query_position+1 → end clamps.
        let r = causal_attention_range(10, 3, None);
        assert_eq!(
            r,
            AttentionRange {
                start: 0,
                end_exclusive: 3
            }
        );
    }

    #[test]
    fn effective_window_combinations() {
        // intrinsic + no caller → intrinsic
        assert_eq!(effective_window(Some(512), None), Some(512));
        // no intrinsic + caller → caller
        assert_eq!(effective_window(None, Some(256)), Some(256));
        // both → min
        assert_eq!(effective_window(Some(512), Some(256)), Some(256));
        assert_eq!(effective_window(Some(512), Some(1024)), Some(512));
        // neither → None
        assert_eq!(effective_window(None, None), None);
    }

    #[test]
    fn range_len_and_empty() {
        assert_eq!(
            AttentionRange {
                start: 3,
                end_exclusive: 7
            }
            .len(),
            4
        );
        assert!(AttentionRange {
            start: 5,
            end_exclusive: 5
        }
        .is_empty());
        assert!(!AttentionRange {
            start: 5,
            end_exclusive: 6
        }
        .is_empty());
    }
}

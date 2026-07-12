//! Raw-logits forward passes used by target-delta optimisation and Apollo.

use std::ops::Range;

use super::super::embed::embed_tokens_pub;
use super::super::hooks::LayerHook;
use super::super::layer::{run_layer_with_capture_hooked, run_layer_with_ffn};
use super::super::ple::precompute_per_layer_inputs;
use super::super::{apply_norm, dot_proj};
use crate::attention::SharedKV;
use larql_models::ModelWeights;
use ndarray::Array2;

/// Return type for [`forward_raw_logits`]. `h_pre_norm` is the residual
/// at the last transformer block's output (pre-final-norm), `h_final`
/// is after final-norm, and `logits` are the raw logits at the final
/// token position (pre-softmax).
pub struct RawForward {
    pub h_pre_norm: Array2<f32>,
    pub h_final: Array2<f32>,
    pub logits: ndarray::Array1<f32>,
}

/// Tail stages of a traced forward pass (last token position). Returned by
/// [`forward_raw_logits_traced`] so the ST5 comparator can compare the
/// final-norm and logits boundaries alongside the per-layer stages a
/// [`LayerHook`] captured during the same forward.
#[derive(Debug, Clone)]
pub struct TracedTail {
    /// Last-token hidden state before the final norm (`h_pre_norm[-1]`).
    pub pre_final_norm: Vec<f32>,
    /// Last-token hidden state after the final norm (`h_final[-1]`).
    pub final_norm: Vec<f32>,
    /// Last-token raw lm-head output before logits scaling/softcap.
    pub lm_head_raw: Vec<f32>,
    /// Last-token final logits after the model's logits transform.
    pub final_logits: Vec<f32>,
}

/// Canonical F32 forward pass that fires a [`LayerHook`] at every semantic
/// boundary (pre-layer, post-attention, post-FFN, post-PLE, post-layer) and
/// returns the tail stages (pre-final-norm, final-norm, lm-head raw, final
/// logits) at the last token.
///
/// This is the **same math** as [`forward_raw_logits`] — it embeds with the
/// production `embed_tokens_pub`, runs each layer through
/// `run_layer_with_capture_hooked` (the hooked sibling of
/// `run_layer_with_ffn`, identical attention + FFN + PLE + `layer_scalar`),
/// then applies the production final-norm + lm-head + logits transform. It
/// is a thin capture harness over the canonical primitives, NOT a second
/// implementation of the forward pass.
///
/// The caller owns `hook` (typically a `RecordHook::for_layers(0..n)`) and
/// reads the per-layer stages out of it after the call; this function only
/// returns the non-layer tail stages.
pub fn forward_raw_logits_traced(
    weights: larql_models::WeightsView,
    token_ids: &[u32],
    hook: &mut dyn LayerHook,
) -> TracedTail {
    assert!(!token_ids.is_empty(), "traced forward needs >= 1 token");
    let total_len = token_ids.len();
    let norm_offset = weights.arch.norm_weight_offset();

    let mut h = embed_tokens_pub(&weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(&weights, &h, token_ids);
    let ffn = crate::ffn::ViewFfn { view: weights };
    let mut kv_cache: std::collections::HashMap<usize, SharedKV> = std::collections::HashMap::new();

    for layer in 0..weights.num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _act, _attn, kv_out)) = run_layer_with_capture_hooked(
            weights,
            &h,
            layer,
            &ffn,
            /*capture_activation=*/ false,
            /*capture_attention=*/ false,
            ple_inputs.get(layer),
            shared_kv,
            hook,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }
    }

    let h_final = apply_norm(&weights, &h, weights.arch.final_norm_key(), norm_offset);
    let last = total_len - 1;
    let last_2d = h_final.slice(ndarray::s![last..total_len, ..]);
    let logits_raw = dot_proj(&last_2d, &weights.lm_head);
    let logits = apply_logits_transform(&weights, logits_raw.row(0).as_slice().unwrap());

    TracedTail {
        pre_final_norm: h.row(last).to_vec(),
        final_norm: h_final.row(last).to_vec(),
        lm_head_raw: logits_raw.row(0).to_vec(),
        final_logits: logits.to_vec(),
    }
}

/// Compute the ST5/ST6 traced tail stages — pre-final-norm, final-norm,
/// lm-head raw, final logits — from a post-layer residual `h` using the
/// production final-norm + lm-head + logits transform.
///
/// This is the shared tail computation for both the F32 traced forward
/// ([`forward_raw_logits_traced`]) and the Q4_K hooked forward
/// (`predict_kquant_hidden_hooked` + this function), so the two capture
/// paths use byte-identical tail math. `h` must have at least one row.
pub fn traced_tail_from_hidden(weights: larql_models::WeightsView, h: &Array2<f32>) -> TracedTail {
    traced_tail_with_lm_head(weights, h, &weights.lm_head)
}

/// Like [`traced_tail_from_hidden`] but projects through an explicit
/// `lm_head` matrix instead of `weights.lm_head`. The final norm still uses
/// `weights`' norm weights (F32 in both F32 and Q4_K vindexes, so identical).
///
/// Used by the ST6 §8 body-vs-lm-head error decomposition:
///   - route B (Q4_K body + F32 lm-head) passes the F32 reference's lm-head,
///   - route C (Q4_K body + production Q4_K lm-head) passes `weights.lm_head`.
pub fn traced_tail_with_lm_head(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    lm_head: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
) -> TracedTail {
    let total_len = h.nrows();
    assert!(total_len > 0, "traced tail needs >= 1 row");
    let last = total_len - 1;
    let norm_offset = weights.arch.norm_weight_offset();
    let h_final = apply_norm(&weights, h, weights.arch.final_norm_key(), norm_offset);
    let last_2d = h_final.slice(ndarray::s![last..total_len, ..]);
    let logits_raw = dot_proj(&last_2d, lm_head);
    let logits = apply_logits_transform(weights.canonical(), logits_raw.row(0).as_slice().unwrap());
    TracedTail {
        pre_final_norm: h.row(last).to_vec(),
        final_norm: h_final.row(last).to_vec(),
        lm_head_raw: logits_raw.row(0).to_vec(),
        final_logits: logits.to_vec(),
    }
}

/// Apply the model's final logits transform: divide by `logits_scaling`
/// then apply the optional `final_logit_softcapping` tanh.
fn apply_logits_transform(weights: &ModelWeights, raw_row: &[f32]) -> ndarray::Array1<f32> {
    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    raw_row
        .iter()
        .map(|&v| {
            let mut logit = v * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            logit
        })
        .collect()
}

/// Project a single hidden state row to raw logits (pre-softmax, pre-temperature).
///
/// Used by constrained generation: the caller masks the returned vector (e.g. sets
/// disallowed token positions to `f32::NEG_INFINITY`) before applying argmax.
pub fn hidden_to_raw_logits(weights: &ModelWeights, h_single: &Array2<f32>) -> Vec<f32> {
    let norm_offset = weights.arch.norm_weight_offset();
    let h_final = apply_norm(
        weights,
        h_single,
        weights.arch.final_norm_key(),
        norm_offset,
    );
    let logits_raw = dot_proj(&h_final.slice(ndarray::s![0..1, ..]), &weights.lm_head);
    apply_logits_transform(weights, logits_raw.row(0).as_slice().unwrap()).to_vec()
}

/// Raw-logits forward pass used by target-delta optimisation.
///
/// Returns (pre-final-norm residual, final-norm residual, logits) at
/// the LAST token position. If `perturb_at_layer` is Some, adds `delta`
/// to the residual's last position after that layer's block runs —
/// matching the Python reference `ffn_out[0, -1, :] += delta; h = h + ffn_out`
/// (since `run_layer_with_ffn` already collapses the block's output +
/// skip, perturbing the post-block `h[-1]` is algebraically the same).
pub fn forward_raw_logits(
    weights: larql_models::WeightsView,
    token_ids: &[u32],
    perturb: Option<(usize, ndarray::ArrayView1<f32>)>,
) -> RawForward {
    forward_raw_logits_with_prefix(weights, token_ids, None, perturb)
}

/// Forward pass with an optional `initial_residual` prepended as a virtual
/// position-0 token before layer 0.
///
/// Mirrors the Python `prefill_to_layer(initial_residual=...)` API used by
/// `UnlimitedContextEngine`/Apollo. The prefix flows through every layer
/// along with the query tokens and participates in attention at each
/// position — it's *not* a per-layer K/V injection, it's a residual
/// prepend.
///
/// Correctness caveat: the prefix is processed at RoPE position 0 here
/// regardless of where in the original sequence it was captured. For
/// Apollo's stored boundaries (captured at window-end positions ~N×512),
/// this is a variant (ii)-style position shift — lossy but survivable
/// when combined with `vec_inject` amplification, which is the whole
/// point of the architecture.
///
/// `initial_residual`, when `Some`, must be a slice of exactly
/// `weights.hidden_size` floats. `token_ids` may not be empty.
pub fn forward_raw_logits_with_prefix(
    weights: larql_models::WeightsView,
    token_ids: &[u32],
    initial_residual: Option<&[f32]>,
    perturb: Option<(usize, ndarray::ArrayView1<f32>)>,
) -> RawForward {
    forward_layer_range(
        weights,
        token_ids,
        initial_residual,
        0..weights.num_layers,
        perturb,
    )
}

/// Forward pass starting at `from_layer` using a pre-computed boundary
/// residual as position-0.
///
/// Skips layers `0..from_layer` entirely — the `boundary_residual` is
/// treated as the output of layer `from_layer - 1` for the stored context.
/// Only `from_layer..num_layers` are computed, which for Apollo with
/// `crystal_layer=30` means 4 layers (30-33) instead of 34.
///
/// Layout: `h[0] = boundary`, `h[1..]` = query embeddings.
/// The perturbation is applied at `target_layer` to the last row.
pub fn forward_from_layer(
    weights: larql_models::WeightsView,
    token_ids: &[u32],
    boundary_residual: &[f32],
    from_layer: usize,
    perturb: Option<(usize, ndarray::ArrayView1<f32>)>,
) -> RawForward {
    forward_layer_range(
        weights,
        token_ids,
        Some(boundary_residual),
        from_layer..weights.num_layers,
        perturb,
    )
}

/// Shared implementation. Runs `layer_range` of the transformer with an
/// optional position-0 residual prefix, perturbs the last row at the
/// requested target layer, and projects the last position to logits.
///
/// Layout when `prefix` is `Some`: row 0 = prefix, rows 1..=q = query
/// embeddings, total_len = q + 1. PLE token ids prepend a `0` placeholder
/// for the virtual prefix row.
///
/// Layout when `prefix` is `None`: rows 0..q = query embeddings,
/// total_len = q.
fn forward_layer_range(
    weights: larql_models::WeightsView,
    token_ids: &[u32],
    prefix: Option<&[f32]>,
    layer_range: Range<usize>,
    perturb: Option<(usize, ndarray::ArrayView1<f32>)>,
) -> RawForward {
    let hidden = weights.hidden_size;
    let q_len = token_ids.len();

    let q_embed = embed_tokens_pub(&weights, token_ids);
    let (mut h, total_len) = if let Some(prefix) = prefix {
        assert_eq!(
            prefix.len(),
            hidden,
            "prefix len {} does not match hidden size {}",
            prefix.len(),
            hidden,
        );
        let mut h = ndarray::Array2::<f32>::zeros((q_len + 1, hidden));
        for (i, &v) in prefix.iter().enumerate() {
            h[[0, i]] = v;
        }
        for r in 0..q_len {
            for c in 0..hidden {
                h[[r + 1, c]] = q_embed[[r, c]];
            }
        }
        (h, q_len + 1)
    } else {
        (q_embed, q_len)
    };

    // PLE: only used by Gemma 4 E2B. With a prefix prepended there's no
    // token id for the virtual row, so we pass a placeholder 0. For models
    // where PLE is active this is a known approximation; for Gemma 3 4B
    // (the Apollo target) PLE is disabled and this branch is a no-op.
    let ple_token_ids: Vec<u32> = if prefix.is_some() {
        let mut v = Vec::with_capacity(q_len + 1);
        v.push(0);
        v.extend_from_slice(token_ids);
        v
    } else {
        token_ids.to_vec()
    };
    let ple_inputs = precompute_per_layer_inputs(&weights, &h, &ple_token_ids);
    let ffn = crate::ffn::ViewFfn { view: weights };

    let mut kv_cache: std::collections::HashMap<usize, SharedKV> = std::collections::HashMap::new();

    for layer in layer_range {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));

        if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
            // Perturb the LAST row (the query's last token) after this
            // layer's block runs. With a prefix present the last row is
            // total_len - 1 = q_len (not q_len - 1).
            if let Some((target_layer, delta)) = perturb {
                if layer == target_layer {
                    let last = total_len - 1;
                    let mut row = h.row_mut(last);
                    for (i, d) in delta.iter().enumerate() {
                        if i < row.len() {
                            row[i] += *d;
                        }
                    }
                }
            }
        }
    }

    let h_pre_norm = h.clone();
    let norm_offset = weights.arch.norm_weight_offset();
    let h_final = apply_norm(&weights, &h, weights.arch.final_norm_key(), norm_offset);

    let last_2d = h_final.slice(ndarray::s![total_len - 1..total_len, ..]);
    let logits_raw = dot_proj(&last_2d, &weights.lm_head);
    let logits = apply_logits_transform(&weights, logits_raw.row(0).as_slice().unwrap());

    RawForward {
        h_pre_norm,
        h_final,
        logits,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod forward_from_layer_tests {
    use super::*;
    use crate::forward::RecordHook;
    use larql_models::test_fixtures::make_test_weights;

    #[test]
    fn forward_raw_logits_returns_vocab_logits() {
        let weights = make_test_weights();
        let raw = forward_raw_logits(
            larql_models::WeightsView::dense(&weights),
            &[0u32, 1, 2],
            None,
        );
        assert_eq!(
            raw.logits.len(),
            weights.vocab_size,
            "logits length should be vocab_size"
        );
        assert_eq!(
            raw.h_pre_norm.shape(),
            &[3, weights.hidden_size],
            "h_pre_norm shape"
        );
    }

    #[test]
    fn forward_raw_logits_traced_captures_boundaries_and_tail() {
        // The ST5 traced forward must run the same math as forward_raw_logits
        // and expose every coarse boundary via the hook + TracedTail.
        let weights = make_test_weights();
        let n = weights.num_layers;
        let view = larql_models::WeightsView::dense(&weights);
        let ids = [0u32, 1, 2];

        // Traced path.
        let mut hook = RecordHook::for_layers(0..n);
        let tail = forward_raw_logits_traced(view, &ids, &mut hook);

        // Reference path (same math, no capture).
        let raw = forward_raw_logits(view, &ids, None);

        // Per-layer boundary capture: every layer recorded pre/post stages.
        for layer in 0..n {
            assert!(hook.pre_layer.contains_key(&layer), "pre_layer {layer}");
            assert!(
                hook.post_attention.contains_key(&layer),
                "post_attention {layer}"
            );
            assert!(hook.post_ffn.contains_key(&layer), "post_ffn {layer}");
            assert!(hook.post_ple.contains_key(&layer), "post_ple {layer}");
            assert!(hook.post_layer.contains_key(&layer), "post_layer {layer}");
        }

        // Tail stages: last-token vectors of the expected width.
        let hidden = weights.hidden_size;
        assert_eq!(tail.pre_final_norm.len(), hidden);
        assert_eq!(tail.final_norm.len(), hidden);
        assert_eq!(tail.lm_head_raw.len(), weights.vocab_size);
        assert_eq!(tail.final_logits.len(), weights.vocab_size);

        // pre_final_norm == last row of the reference pre-norm residual.
        let last = ids.len() - 1;
        for (i, v) in tail.pre_final_norm.iter().enumerate() {
            assert!(
                (v - raw.h_pre_norm[[last, i]]).abs() < 1e-4,
                "pre_final_norm[{i}] mismatch"
            );
        }
        // final logits match the reference logits (same transform).
        for (i, v) in tail.final_logits.iter().enumerate() {
            assert!(
                (v - raw.logits[i]).abs() < 1e-4,
                "final_logits[{i}] mismatch"
            );
        }
        assert!(
            tail.final_logits.iter().all(|v| v.is_finite()),
            "final logits must be finite"
        );
    }

    #[test]
    fn forward_raw_logits_single_token() {
        let weights = make_test_weights();
        let raw = forward_raw_logits(larql_models::WeightsView::dense(&weights), &[5u32], None);
        assert_eq!(raw.logits.len(), weights.vocab_size);
        assert!(
            raw.logits.iter().all(|v| v.is_finite()),
            "all logits should be finite"
        );
    }

    #[test]
    fn forward_from_layer_zero_equals_full_forward() {
        // forward_from_layer with from_layer=0 should be equivalent to
        // forward_raw_logits_with_prefix when the boundary is the zero vector.
        // They won't be identical (boundary passes through all layers as a real position)
        // but output shape must match.
        let weights = make_test_weights();
        let token_ids = &[1u32, 2];
        let boundary = vec![0.0f32; weights.hidden_size];

        let from_layer = forward_from_layer(
            larql_models::WeightsView::dense(&weights),
            token_ids,
            &boundary,
            0,
            None,
        );
        // from_layer=0 with zero boundary: should have (1 boundary + 2 query) positions
        assert_eq!(from_layer.h_pre_norm.shape(), &[3, weights.hidden_size]);
        assert_eq!(from_layer.logits.len(), weights.vocab_size);
        assert!(from_layer.logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_from_layer_skips_early_layers() {
        // Starting from layer 1 (of 2) should give a DIFFERENT result than
        // starting from layer 0, proving layers are actually being skipped.
        let weights = make_test_weights();
        let token_ids = &[3u32];
        let boundary = vec![0.1f32; weights.hidden_size];

        let from_0 = forward_from_layer(
            larql_models::WeightsView::dense(&weights),
            token_ids,
            &boundary,
            0,
            None,
        );
        let from_1 = forward_from_layer(
            larql_models::WeightsView::dense(&weights),
            token_ids,
            &boundary,
            1,
            None,
        );

        // Outputs should differ (layer 0's transform changes the residual)
        let differ = from_0
            .logits
            .iter()
            .zip(from_1.logits.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "from_layer=0 and from_layer=1 should produce different logits"
        );
    }

    #[test]
    fn forward_from_layer_output_shape() {
        let weights = make_test_weights();
        // 3 query tokens, from_layer=1: h has 4 rows (1 boundary + 3 query)
        let raw = forward_from_layer(
            larql_models::WeightsView::dense(&weights),
            &[0u32, 1, 2],
            &vec![0.0; weights.hidden_size],
            1,
            None,
        );
        assert_eq!(raw.h_pre_norm.shape(), &[4, weights.hidden_size]);
        assert_eq!(raw.logits.len(), weights.vocab_size);
    }

    #[test]
    fn forward_raw_logits_with_prefix_shape() {
        let weights = make_test_weights();
        let prefix = vec![0.5f32; weights.hidden_size];
        let raw = forward_raw_logits_with_prefix(
            larql_models::WeightsView::dense(&weights),
            &[0u32, 1],
            Some(&prefix),
            None,
        );
        // prefix + 2 tokens = 3 positions
        assert_eq!(raw.h_pre_norm.shape(), &[3, weights.hidden_size]);
        assert_eq!(raw.logits.len(), weights.vocab_size);
    }

    /// `traced_tail_from_hidden` computes the same tail stages (pre-final-norm,
    /// final-norm, lm-head raw, final logits) from a post-layer residual that
    /// `forward_raw_logits` produces inline. They must agree at the last token.
    #[test]
    fn traced_tail_from_hidden_matches_forward_raw_logits() {
        let weights = make_test_weights();
        let view = larql_models::WeightsView::dense(&weights);
        let ids = [0u32, 1, 2];
        let raw = forward_raw_logits(view, &ids, None);
        let tail = traced_tail_from_hidden(view, &raw.h_pre_norm);
        let last = ids.len() - 1;
        let hidden = weights.hidden_size;
        let vocab = weights.vocab_size;
        assert_eq!(tail.pre_final_norm.len(), hidden);
        assert_eq!(tail.final_norm.len(), hidden);
        assert_eq!(tail.lm_head_raw.len(), vocab);
        assert_eq!(tail.final_logits.len(), vocab);
        for i in 0..hidden {
            assert!(
                (tail.pre_final_norm[i] - raw.h_pre_norm[[last, i]]).abs() < 1e-5,
                "pre_final_norm[{i}]"
            );
            assert!(
                (tail.final_norm[i] - raw.h_final[[last, i]]).abs() < 1e-5,
                "final_norm[{i}]"
            );
        }
        for i in 0..vocab {
            assert!(
                (tail.final_logits[i] - raw.logits[i]).abs() < 1e-5,
                "final_logits[{i}]"
            );
        }
    }

    /// `traced_tail_with_lm_head` projects through an explicit lm-head. Passing
    /// the model's own lm-head must match `traced_tail_from_hidden`; passing a
    /// zeroed lm-head must produce zero logits.
    #[test]
    fn traced_tail_with_lm_head_uses_explicit_matrix() {
        let weights = make_test_weights();
        let view = larql_models::WeightsView::dense(&weights);
        let ids = [0u32, 1, 2];
        let raw = forward_raw_logits(view, &ids, None);
        // Own lm-head → same result as traced_tail_from_hidden.
        let tail_own = traced_tail_with_lm_head(view, &raw.h_pre_norm, &weights.lm_head);
        for i in 0..weights.vocab_size {
            assert!(
                (tail_own.final_logits[i] - raw.logits[i]).abs() < 1e-5,
                "own lm-head final_logits[{i}]"
            );
        }
        // Zeroed lm-head → zero raw logits (before the transform; with
        // softcap absent on the test fixture the final logits are also zero).
        let zero_lm = ndarray::Array2::zeros((weights.vocab_size, weights.hidden_size));
        let tail_zero = traced_tail_with_lm_head(view, &raw.h_pre_norm, &zero_lm);
        assert!(
            tail_zero.lm_head_raw.iter().all(|v| v.abs() < 1e-6),
            "zeroed lm-head must produce zero raw logits"
        );
    }
}

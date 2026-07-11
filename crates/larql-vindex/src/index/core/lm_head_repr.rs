//! LARQL-GPU-B4 narrow lm-head descriptor.
//!
//! Exposes the active lm-head representation through a read-only accessor
//! on [`crate::index::core::VectorIndex`] so the inference decode loop can
//! build a [`larql_compute::backend::greedy::GreedyQ4kHeadSpec`] without
//! duplicating the loader assumptions about which format the vindex
//! actually carries.
//!
//! The descriptor is intentionally minimal: it reports the *format* and
//! the *verified bytes + dims* for the Q4_K case, and a non-Q4_K variant
//! for f16 / f32 / absent heads so the caller can reject them cleanly and
//! use the existing host path. It does **not** expose tokenizer,
//! final-norm, or inference-layer types — those stay on the inference
//! side.

/// The active lm-head representation, as discovered by the narrow vindex
/// accessor. Only the Q4_K variant carries the bytes + dims the device
/// greedy path needs; the other variants exist so the caller can fall
/// back cleanly.
#[derive(Debug, Clone, Copy)]
pub enum LmHeadRepresentation<'a> {
    /// A verified Q4_K lm-head: `bytes` is the packed Q4_K matrix
    /// (`[physical_vocab, hidden]`, 144 bytes per 256-element super-block).
    /// `physical_vocab` is the row count implied by `bytes.len()` and
    /// `hidden`; `logical_vocab` is `min(index.logical_vocab_size.unwrap_or(physical), physical)`.
    Q4K {
        bytes: &'a [u8],
        hidden: usize,
        physical_vocab: usize,
        logical_vocab: usize,
    },
    /// An f16 lm-head (tied-embedding) — not eligible for the device Q4_K
    /// greedy path. Carries no payload; the caller keeps the host path.
    F16,
    /// An f32 lm-head — not eligible. The caller keeps the host path.
    F32,
    /// No lm-head bytes loaded at all (or the Q4_K bytes are malformed:
    /// zero-length, or not a whole number of Q4_K super-blocks).
    Absent,
}

impl<'a> LmHeadRepresentation<'a> {
    /// `true` only for the [`LmHeadRepresentation::Q4K`] variant.
    pub fn is_q4k(&self) -> bool {
        matches!(self, Self::Q4K { .. })
    }
}

/// Compute the Q4_K [`LmHeadRepresentation`] from the raw bytes + dims,
/// shared by the vindex accessor and its tests. Returns `Absent` when the
/// bytes don't form a whole number of Q4_K super-blocks for the given
/// `(hidden, physical_vocab)`.
pub(crate) fn classify_q4k(
    bytes: &[u8],
    hidden: usize,
    physical_vocab: usize,
    logical_vocab: Option<usize>,
) -> LmHeadRepresentation<'_> {
    if hidden == 0 || !hidden.is_multiple_of(256) || physical_vocab == 0 {
        return LmHeadRepresentation::Absent;
    }
    // Q4_K: 144 bytes per 256-element super-block.
    let row_bytes = (hidden / 256) * 144;
    let needed = match physical_vocab.checked_mul(row_bytes) {
        Some(n) => n,
        None => return LmHeadRepresentation::Absent,
    };
    if bytes.len() < needed {
        return LmHeadRepresentation::Absent;
    }
    let logical = logical_vocab.unwrap_or(physical_vocab).min(physical_vocab);
    LmHeadRepresentation::Q4K {
        bytes,
        hidden,
        physical_vocab,
        logical_vocab: logical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::VectorIndex;
    use larql_compute::cpu::ops::q4_common;
    use std::sync::Arc;

    /// Q4_K round-trip: a small vindex with a synthesized Q4_K lm-head
    /// reports the Q4K variant with the right bytes + dims. Hidden must be
    /// a multiple of 256 for Q4_K; vocab must make the byte count exact.
    #[test]
    fn lm_head_representation_reports_q4k_for_kquant_vindex() {
        let vocab = 256usize;
        let hidden = 256usize;
        let f32_rows = vec![0.1f32; vocab * hidden];
        let q4k = q4_common::quantize_q4_k(&f32_rows);

        let mut index = VectorIndex::empty(1, hidden);
        index.vocab_size = vocab;
        Arc::make_mut(&mut index.storage).set_lm_head_kquant_synth(Arc::new(q4k.clone()));

        let repr = index.lm_head_representation();
        match repr {
            LmHeadRepresentation::Q4K {
                bytes,
                hidden: h,
                physical_vocab: p,
                logical_vocab: l,
            } => {
                assert_eq!(bytes.len(), q4k.len());
                assert_eq!(h, hidden);
                assert_eq!(p, vocab);
                assert_eq!(l, vocab); // logical == physical when unset
            }
            other => panic!("expected Q4K, got {other:?}"),
        }
        assert!(repr.is_q4k());
    }

    /// `logical_vocab_size` smaller than physical clamps the descriptor's
    /// `logical_vocab` to the logical value — the device greedy path must
    /// never select a padded row.
    #[test]
    fn lm_head_representation_clamps_logical_below_physical() {
        let vocab = 512usize;
        let hidden = 256usize;
        let f32_rows = vec![0.1f32; vocab * hidden];
        let q4k = q4_common::quantize_q4_k(&f32_rows);

        let mut index = VectorIndex::empty(1, hidden);
        index.vocab_size = vocab;
        index.logical_vocab_size = Some(400);
        Arc::make_mut(&mut index.storage).set_lm_head_kquant_synth(Arc::new(q4k));

        match index.lm_head_representation() {
            LmHeadRepresentation::Q4K {
                physical_vocab,
                logical_vocab,
                ..
            } => {
                assert_eq!(physical_vocab, vocab);
                assert_eq!(logical_vocab, 400);
            }
            other => panic!("expected Q4K, got {other:?}"),
        }
    }

    /// `classify_q4k` is the pure shape-check helper. A short byte buffer
    /// returns `Absent` rather than a truncated Q4K descriptor.
    #[test]
    fn classify_q4k_rejects_short_byte_buffer() {
        let r = classify_q4k(&[0u8; 50], 256, 10, None);
        assert!(matches!(r, LmHeadRepresentation::Absent));
    }

    /// `classify_q4k` honours the logical-vocab clamp.
    #[test]
    fn classify_q4k_clamps_logical() {
        let hidden = 256;
        let vocab = 10;
        let bytes = vec![0u8; vocab * (hidden / 256) * 144];
        let r = classify_q4k(&bytes, hidden, vocab, Some(7));
        match r {
            LmHeadRepresentation::Q4K { logical_vocab, .. } => assert_eq!(logical_vocab, 7),
            other => panic!("expected Q4K, got {other:?}"),
        }
    }

    /// A vindex with no lm-head bytes reports `Absent`.
    #[test]
    fn lm_head_representation_absent_when_no_bytes() {
        let index = VectorIndex::empty(1, 256);
        assert!(matches!(
            index.lm_head_representation(),
            LmHeadRepresentation::Absent
        ));
    }

    /// Malformed Q4_K bytes (not a whole number of super-blocks) report
    /// `Absent` rather than a bogus Q4K descriptor.
    #[test]
    fn lm_head_representation_absent_for_malformed_q4k_bytes() {
        let mut index = VectorIndex::empty(1, 256);
        index.vocab_size = 10;
        // 1 byte short of a full super-block per row.
        Arc::make_mut(&mut index.storage).set_lm_head_kquant_synth(Arc::new(vec![0u8; 143]));
        assert!(matches!(
            index.lm_head_representation(),
            LmHeadRepresentation::Absent
        ));
    }
}

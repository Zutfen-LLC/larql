//! LARQL-GPU-B4 substrate types for device-side greedy lm-head selection.
//!
//! For **plain unpenalised greedy** decode, the final transformer hidden
//! state can stay resident on the accelerator through the final norm, the
//! Q4_K lm-head GEMV, and the top-K candidate reduction — collapsing the
//! full hidden-state DtoH, the normalised-query HtoD, and the full
//! score-vector DtoH into one fixed-size candidate readback (~40 bytes).
//!
//! These types are the backend-neutral result/descriptor shape the
//! [`crate::DecodeBackend`] additive method speaks. They live in the
//! substrate (`larql-compute`) so both the CUDA backend and the
//! inference decode loop can reference them without a dep cycle. They
//! carry **no** tokenizer or inference-layer types — the contract is
//! `Q4_K bytes + dims + final-norm weight + candidate width`.
//!
//! Non-greedy sampling, repetition penalties, constrained decoding, and
//! non-Q4_K head formats are deliberately out of scope: the additive
//! [`crate::DecodeBackend::decode_token_greedy_q4k`] method returns
//! `HostHidden` (or `None`) for every ineligible case, and the inference
//! loop falls back to the exact existing host norm → lm-head → sample
//! path.

/// Outcome of one [`crate::DecodeBackend::decode_token_greedy_q4k`] call.
///
/// `DevicePick` is the B4 fast path: the backend kept the final hidden
/// state resident, ran the final norm + Q4_K lm-head + top-K reduction
/// on-device, and is returning only a fixed-size candidate result.
/// `HostHidden` is the fallback: the backend ran the transformer layers
/// (the KV was advanced exactly once) but could not finalise the lm-head
/// on-device — the inference loop runs the existing host final norm +
/// lm-head path on the returned hidden vector.
#[derive(Debug)]
pub enum GreedyDecodeOutput {
    /// The device-side greedy path succeeded. The winning token, its raw
    /// score, and the fixed-size candidate set used for the callback
    /// probability are already selected; the host must NOT re-sample.
    DevicePick(DeviceGreedyPick),
    /// The backend computed the final hidden state but could not complete
    /// the device-side lm-head. The caller runs the existing host final
    /// norm → lm_head_topk → sample path on this vector.
    HostHidden(Vec<f32>),
}

/// The fixed-size result of a device-side greedy lm-head selection.
///
/// `probability_hits` is the sorted (descending by score) candidate set
/// the existing host path would have produced for the configured greedy
/// candidate width (currently five). The callback probability is computed
/// by softmaxing these scores exactly as `softmax_prob` does today, so
/// the reported probability is unchanged when B4 engages.
///
/// `token_id` is always `< logical_vocab_size` (the device reduction
/// operates only over `[0, min(logical, physical))` — a padded physical
/// row can never win).
#[derive(Debug, Clone)]
pub struct DeviceGreedyPick {
    /// The preselected winning token id. Guaranteed `< logical_vocab_size`.
    pub token_id: u32,
    /// The winning token's raw (post-lm-head, pre-softmax) score.
    pub score: f32,
    /// The fixed-size sorted `(token_id, score)` candidate set used for
    /// the callback probability. Length ≤ the configured candidate width.
    /// Descending by score; non-finite scores are excluded, matching the
    /// host top-K contract.
    pub probability_hits: Vec<(u32, f32)>,
}

/// Narrow read-only descriptor of a Q4_K lm-head plus its final norm,
/// consumed by the device-side greedy path. Built by the inference loop
/// from the active vindex lm-head representation and the model's final
/// norm weight — the descriptor is the single place where the loader
/// assumptions about "the active lm-head is Q4_K" are verified, so the
/// inference loop never duplicates them.
///
/// `final_norm_weight` is `None` when the architecture has no learned
/// final-norm weight (the device applies the `1.0` identity scaling, as
/// the host `rms_norm_eps` reference does). Architectures requiring
/// LayerNorm, bias, or any unsupported final norm must NOT build this
/// descriptor — the inference loop keeps them on the host path.
#[derive(Debug, Clone, Copy)]
pub struct GreedyQ4kHeadSpec<'a> {
    /// The packed Q4_K lm-head bytes, row-major `[physical_vocab_size,
    /// hidden_size]`. Only the first `min(logical, physical)` rows are
    /// read by the device GEMV; padding rows at the tail are excluded.
    pub lm_head_bytes: &'a [u8],
    /// The hidden dimension (columns of the lm-head). Must be a multiple
    /// of 256 (Q4_K super-block size).
    pub hidden_size: usize,
    /// The physical row count of `lm_head_bytes` (may include padding).
    pub physical_vocab_size: usize,
    /// The logical (tokenizer) vocabulary size. Token ids at or above
    /// this value must never be selected. `min(logical, physical)` bounds
    /// the device GEMV + reduction.
    pub logical_vocab_size: usize,
    /// The final-norm learned weight (`final_norm_key`), `hidden_size`
    /// f32 elements. `None` for the parameter-free final norm.
    pub final_norm_weight: Option<&'a [f32]>,
    /// The RMSNorm epsilon, exactly matching the host `apply_norm` path
    /// (`arch.norm_eps()` or the override).
    pub final_norm_eps: f64,
    /// The final-norm weight offset (`arch.norm_weight_offset()`).
    pub final_norm_offset: f32,
    /// The greedy candidate width (currently five). The device reduction
    /// returns at most this many `(id, score)` pairs for the callback
    /// probability.
    pub candidate_width: usize,
}

impl<'a> GreedyQ4kHeadSpec<'a> {
    /// The number of rows the device GEMV + reduction may consider:
    /// `min(logical_vocab_size, physical_vocab_size)`. Padded physical
    /// rows above this are excluded so an invalid padded row can never win.
    pub fn logical_rows(&self) -> usize {
        self.logical_vocab_size.min(self.physical_vocab_size)
    }

    /// Validate the descriptor's dimensions and byte size against the Q4_K
    /// super-block geometry. Returns the validated `logical_rows` count on
    /// success. Used by both the host fallback decision and the device
    /// launch guard so the two cannot drift.
    ///
    /// Fails (`None`) when: hidden isn't a multiple of 256; the physical
    /// row count doesn't match the byte length; `logical_rows` is zero;
    /// or the byte/row arithmetic overflows. All checked — never panics.
    pub fn validate(&self) -> Option<usize> {
        if self.hidden_size == 0 || !self.hidden_size.is_multiple_of(256) {
            return None;
        }
        if self.physical_vocab_size == 0 {
            return None;
        }
        let logical = self.logical_rows();
        if logical == 0 {
            return None;
        }
        // Q4_K: 144 bytes per 256-element super-block.
        let row_bytes = (self.hidden_size / 256).checked_mul(144)?;
        let physical_bytes = self.physical_vocab_size.checked_mul(row_bytes)?;
        if self.lm_head_bytes.len() < physical_bytes {
            return None;
        }
        if let Some(w) = self.final_norm_weight {
            if w.len() != self.hidden_size {
                return None;
            }
        }
        Some(logical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(
        bytes_len: usize,
        hidden: usize,
        physical: usize,
        logical: usize,
        norm: Option<usize>,
    ) -> GreedyQ4kHeadSpec<'static> {
        static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
        static NORM: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
        let bytes = BYTES.get_or_init(|| vec![0u8; 1 << 24]);
        let norm_buf = NORM.get_or_init(|| vec![1.0f32; 1 << 16]);
        let slice = &bytes[..bytes_len.min(bytes.len())];
        let norm_slice = norm.map(|n| &norm_buf[..n.min(norm_buf.len())]);
        GreedyQ4kHeadSpec {
            lm_head_bytes: slice,
            hidden_size: hidden,
            physical_vocab_size: physical,
            logical_vocab_size: logical,
            final_norm_weight: norm_slice,
            final_norm_eps: 1e-6,
            final_norm_offset: 1.0,
            candidate_width: 5,
        }
    }

    #[test]
    fn logical_rows_clamps_logical_to_physical() {
        // logical > physical → clamped to physical.
        let s = spec(0, 256, 100, 200, Some(256));
        assert_eq!(s.logical_rows(), 100);
        // logical < physical → logical wins.
        let s = spec(0, 256, 200, 100, Some(256));
        assert_eq!(s.logical_rows(), 100);
        // equal → no-op.
        let s = spec(0, 256, 100, 100, Some(256));
        assert_eq!(s.logical_rows(), 100);
    }

    #[test]
    fn validate_rejects_non_multiple_of_256_hidden() {
        let row_bytes = 144; // hidden=256
        let s = spec(100 * row_bytes, 255, 100, 100, None);
        assert!(s.validate().is_none());
    }

    #[test]
    fn validate_rejects_zero_physical_vocab() {
        let s = spec(0, 256, 0, 0, None);
        assert!(s.validate().is_none());
    }

    #[test]
    fn validate_rejects_zero_logical_rows() {
        // physical > 0 but logical == 0.
        let row_bytes = 144;
        let s = spec(10 * row_bytes, 256, 10, 0, None);
        assert!(s.validate().is_none());
    }

    #[test]
    fn validate_rejects_short_byte_buffer() {
        let row_bytes = 144; // hidden=256
                             // physical=100 needs 100*144 bytes; provide only 50.
        let s = spec(50 * row_bytes, 256, 100, 100, None);
        assert!(s.validate().is_none());
    }

    #[test]
    fn validate_rejects_wrong_norm_weight_length() {
        let row_bytes = 144;
        let s = spec(100 * row_bytes, 256, 100, 100, Some(128));
        assert!(s.validate().is_none(), "norm weight must equal hidden_size");
    }

    #[test]
    fn validate_accepts_well_formed_descriptor() {
        let row_bytes = 144;
        let s = spec(100 * row_bytes, 256, 100, 80, Some(256));
        assert_eq!(s.validate(), Some(80));
    }

    #[test]
    fn validate_accepts_none_norm_weight() {
        let row_bytes = 144;
        let s = spec(100 * row_bytes, 256, 100, 100, None);
        assert_eq!(s.validate(), Some(100));
    }

    #[test]
    fn validate_accepts_extra_trailing_bytes() {
        // A vindex may over-allocate; only the physical prefix is required.
        let row_bytes = 144;
        let s = spec(200 * row_bytes, 256, 100, 100, None);
        assert_eq!(s.validate(), Some(100));
    }
}

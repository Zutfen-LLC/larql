//! LARQL-GPU-B4 generation-scoped workspace for the device-side greedy
//! lm-head path.
//!
//! Holds the device-resident scratch buffers the resident terminal path
//! (final RMSNorm output, logical-vocabulary score buffer, partial
//! candidate buffer, fixed-size result buffer) so they aren't allocated
//! and dropped every token — that would force allocator/stream
//! synchronization on devices without memory-pool support.
//!
//! The workspace is rebuilt when its dimensions or the vindex generation
//! change (a different lm-head shape, or a `reset_kv_cache` boundary).
//! Reset/teardown follows the existing CUDA generation lifecycle
//! (`reset_kv_cache` drops the arena + graph cache).

use crate::backend::RuntimeError;
use crate::ops::{GREEDY_BLOCK_SIZE, GREEDY_MAX_K};
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// One-element placeholder for the parameter-free final-RMSNorm path
/// (B4-CORRECTION C). A `static` so its address is stable across tokens —
/// the weight-cache key `(ptr, len)` hits after the first upload, and the
/// workspace's `norm_weight_src_ptr` stays constant for the generation. The
/// kernel's `has_weight=0` flag makes it ignore the contents; cudarc just
/// requires a non-empty device buffer.
pub(crate) static PARAM_FREE_NORM_PLACEHOLDER: [f32; 1] = [0.0f32];

/// Device-resident scratch for one greedy-head shape. Allocated once per
/// `(logical_rows, candidate_width)` and reused across tokens. Held behind
/// a `Mutex<Option<…>>` on `CudaBackend` so the single-threaded decode
/// path can borrow it without contention.
pub(crate) struct GreedyHeadWorkspace {
    /// The normalised-hidden output of the final RMSNorm (`hidden` f32).
    /// Written by `launch_rms_norm_dev`, consumed by the Q4_K lm-head GEMV.
    pub(crate) normed_hidden: CudaSlice<f32>,
    /// The logical-vocabulary score buffer (`logical_rows` f32). Written by
    /// the Q4_K lm-head GEMV, consumed by the partial-reduction kernel.
    pub(crate) scores: CudaSlice<f32>,
    /// Per-block partial candidates: `num_blocks * k` scores + ids. Written
    /// by the partial kernel, consumed by the final kernel.
    pub(crate) partial_scores: CudaSlice<f32>,
    pub(crate) partial_ids: CudaSlice<u32>,
    /// Fixed-size final result: `k` scores + `k` ids (B4-CORRECTION A:
    /// sized to the actual candidate width, not `GREEDY_MAX_K`, so the
    /// readback transfer is exactly `k` f32 + `k` u32 = `8k` bytes). Written
    /// by the final kernel, read back to host (the only DtoH on the B4 path).
    pub(crate) result_scores: CudaSlice<f32>,
    pub(crate) result_ids: CudaSlice<u32>,
    /// Persistent final-RMSNorm weight device handle (B4-CORRECTION C).
    /// Resolved once on the first eligible decode token (cold upload) and
    /// reused for every subsequent token, eliminating the per-token
    /// `clone_htod` the pre-correction `launch_rms_norm_into_dev` paid every
    /// token. `None` until first resolution; the parameter-free path stores
    /// the one-element placeholder handle here.
    pub(crate) norm_weight_dev: Option<Arc<CudaSlice<f32>>>,
    /// Source host pointer the cached weight was resolved from (`None` while
    /// unresolved). Tracked so a head-spec change that swaps the norm weight
    /// without a workspace rebuild re-resolves instead of serving stale data.
    /// Parameter-free (`None` weight) is represented as `Some(0)`.
    pub(crate) norm_weight_src_ptr: Option<usize>,
    /// `has_weight` flag for the launcher: `1` for a learned weight, `0` for
    /// the parameter-free placeholder.
    pub(crate) norm_has_weight: i32,
    /// The shape this workspace was built for.
    pub(crate) logical_rows: usize,
    pub(crate) hidden: usize,
    pub(crate) candidate_width: usize,
    /// Number of partial-reduction blocks for `logical_rows`.
    pub(crate) num_blocks: usize,
}

impl GreedyHeadWorkspace {
    /// Build (or rebuild) the workspace for `(logical_rows, hidden, k)`.
    /// All buffers are zero-initialised device allocations with NO
    /// host→device transfer. Errors map to `RuntimeError` so the caller
    /// falls back to `HostHidden`.
    pub(crate) fn build(
        stream: &Arc<CudaStream>,
        logical_rows: usize,
        hidden: usize,
        candidate_width: usize,
    ) -> Result<Self, RuntimeError> {
        let k = candidate_width.min(GREEDY_MAX_K);
        let num_blocks = logical_rows.div_ceil(GREEDY_BLOCK_SIZE);
        let partial_len = num_blocks.saturating_mul(k).max(GREEDY_MAX_K);
        let normed_hidden = stream
            .alloc_zeros::<f32>(hidden)
            .map_err(|e| RuntimeError::context("allocating greedy normed_hidden", e))?;
        let scores = stream
            .alloc_zeros::<f32>(logical_rows)
            .map_err(|e| RuntimeError::context("allocating greedy scores", e))?;
        let partial_scores = stream
            .alloc_zeros::<f32>(partial_len)
            .map_err(|e| RuntimeError::context("allocating greedy partial_scores", e))?;
        let partial_ids = stream
            .alloc_zeros::<u32>(partial_len)
            .map_err(|e| RuntimeError::context("allocating greedy partial_ids", e))?;
        // B4-CORRECTION A: size the result buffers to the actual candidate
        // width so the terminal DtoH transfers exactly `k` f32 + `k` u32,
        // not `GREEDY_MAX_K` of each. The launcher checks `len() >= k`, so an
        // exactly-`k` buffer is the minimal correct allocation.
        let result_scores = stream
            .alloc_zeros::<f32>(k)
            .map_err(|e| RuntimeError::context("allocating greedy result_scores", e))?;
        let result_ids = stream
            .alloc_zeros::<u32>(k)
            .map_err(|e| RuntimeError::context("allocating greedy result_ids", e))?;
        Ok(Self {
            normed_hidden,
            scores,
            partial_scores,
            partial_ids,
            result_scores,
            result_ids,
            // B4-CORRECTION C: the final-norm weight is resolved lazily on
            // first use (see `run_greedy_chain_on_device`), not at build —
            // `build` does not see the head spec.
            norm_weight_dev: None,
            norm_weight_src_ptr: None,
            norm_has_weight: 0,
            logical_rows,
            hidden,
            candidate_width: k,
            num_blocks,
        })
    }

    /// `true` when the workspace matches `(logical_rows, hidden, k)` and
    /// can be reused for the next token.
    pub(crate) fn matches(
        &self,
        logical_rows: usize,
        hidden: usize,
        candidate_width: usize,
    ) -> bool {
        let k = candidate_width.min(GREEDY_MAX_K);
        self.logical_rows == logical_rows && self.hidden == hidden && self.candidate_width == k
    }

    /// Length of each final result buffer (B4-CORRECTION A). Equal to the
    /// configured candidate width `k`; the terminal DtoH reads back exactly
    /// `result_len()` f32 scores + `result_len()` u32 ids.
    #[cfg(test)]
    pub(crate) fn result_len(&self) -> usize {
        self.candidate_width
    }
}

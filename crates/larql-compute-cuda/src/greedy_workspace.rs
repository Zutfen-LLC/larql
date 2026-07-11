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
    /// Fixed-size final result: `k` scores + `k` ids. Written by the final
    /// kernel, read back to host (the only DtoH on the B4 path).
    pub(crate) result_scores: CudaSlice<f32>,
    pub(crate) result_ids: CudaSlice<u32>,
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
        let result_scores = stream
            .alloc_zeros::<f32>(GREEDY_MAX_K)
            .map_err(|e| RuntimeError::context("allocating greedy result_scores", e))?;
        let result_ids = stream
            .alloc_zeros::<u32>(GREEDY_MAX_K)
            .map_err(|e| RuntimeError::context("allocating greedy result_ids", e))?;
        Ok(Self {
            normed_hidden,
            scores,
            partial_scores,
            partial_ids,
            result_scores,
            result_ids,
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
}

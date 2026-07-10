//! LARQL-GPU-B3A: resident decode arena + FFN graph state (B3A-3/5/6).
//!
//! This module owns the **device side** of the CUDA Graph replay for the
//! resident decode FFN. The pure plan/identity contract lives in
//! [`crate::ffn_graph`]; this module holds the graph-captured state, the
//! stable scratch buffers, and the generation-scoped cache.
//!
//! NOTE: the structures here are constructed/referenced by the B3A-5 pipeline
//! integration (build + replay) which lands in the next commit. Until that
//! wiring is in place the `dead_code` allows below silence the "never used"
//! warnings. They are removed once the pipeline integration lands.
#![allow(dead_code)]
//!
//! ## Architecture (B3A review points 2, 3, 6, 7)
//!
//! ### Why a dedicated capture stream (B3A-SMOKE finding)
//!
//! The LARQL runtime's default stream is the NULL stream (`cudarc`'s
//! `default_stream()` returns `cu_stream = null_mut`), and CUDA forbids
//! capturing it (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). Graph capture and
//! replay therefore happen on a **dedicated non-NULL stream** created via
//! `CudaContext::new_stream()`.
//!
//! Additionally, `cudarc` enables per-slice `CudaEvent` tracking by default.
//! During capture, `launch_builder.arg(&CudaSlice)` injects `cuStreamWaitEvent`
//! for prior write events → `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`. Event
//! tracking is therefore **disabled** for graph buffers (the stream orders
//! captured work explicitly; cross-stream ordering is managed by this module
//! via explicit synchronization).
//!
//! ### Cross-stream synchronization
//!
//! The resident decode loop runs attention + KV mirror on the runtime's NULL
//! stream, but the FFN graph replays on the dedicated capture stream. The
//! arena manages the handoff:
//!
//! 1. Attention writes the post-attn hidden state into the arena's **input**
//!    buffer (via `launch_residual_add_into` on the NULL stream).
//! 2. Before graph replay, the capture stream **waits** on the NULL stream
//!    (an event recorded after the attention write), so the graph sees the
//!    fresh input.
//! 3. The graph replays on the capture stream, writing the post-FFN state
//!    into the arena's **output** buffer.
//! 4. The NULL stream **waits** on the capture stream before the next layer's
//!    attention reads the output.
//!
//! This ping-pong keeps hidden state device-resident across layers with **zero
//! host crossings and zero per-token allocations** (the two stable buffers are
//! allocated once per generation and reused by address).

use std::sync::Arc;

use crate::backend::CudaRuntime;
use crate::backend::RuntimeError;
use crate::ffn_graph::GraphGenerationId;
use cudarc::driver::{CudaGraph, CudaSlice, CudaStream, DriverError};

/// One layer's captured resident-FFN graph + its referenced buffers.
///
/// Owns the executable graph and every device buffer whose address the graph
/// captured. The explicit [`Drop`] impl destroys the graph **before** the
/// buffers (B3A review point 7: exec graph → captured graph → scratch →
/// weights), so the graph never references freed memory during teardown.
pub(crate) struct ResidentFfnGraph {
    /// The executable graph (CUgraphExec + CUgraph). `Option` so [`Drop`] can
    /// `take()` it before the scratch/weights.
    pub(crate) graph: Option<CudaGraph>,
    /// Internal scratch buffers the graph reads/writes (normed input, gate/up
    /// outputs, activation, down output, post-norm output). `Option` for
    /// ordered teardown.
    pub(crate) scratch: Option<ResidentFfnGraphScratch>,
    /// Retained weight-buffer handles (gate/up/down `Arc<CudaSlice<u8>>` +
    /// norm-weight buffers). Keeping the `Arc`s alive pins the device
    /// addresses the graph captured.
    pub(crate) weights: Option<RetainedWeights>,
}

/// The per-layer internal scratch buffers a captured FFN graph references.
///
/// Allocated once at graph build time (after event tracking is disabled, so
/// the slices carry no `CudaEvent` handles) and held for the graph's lifetime.
/// Their device addresses are what the captured kernel nodes bind; they must
/// not move or be reallocated until the graph is destroyed.
pub(crate) struct ResidentFfnGraphScratch {
    /// Normalized input `[hidden]` (pre-FFN RMSNorm output).
    pub(crate) normed_input: CudaSlice<f32>,
    /// Gate projection output `[inter]`.
    pub(crate) gate_out: CudaSlice<f32>,
    /// Up projection output `[inter]`.
    pub(crate) up_out: CudaSlice<f32>,
    /// Activation output `[inter]` (gate×up for Gated, silu/gelu(up) for Standard).
    pub(crate) act: CudaSlice<f32>,
    /// Down projection output `[hidden]`.
    pub(crate) down_out: CudaSlice<f32>,
    /// Post-FFN norm output `[hidden]` (only allocated when `has_post_norms`).
    pub(crate) post_norm_out: Option<CudaSlice<f32>>,
}

/// Retained weight-buffer handles that pin the device addresses the graph
/// captured. Dropping these would free the weights while the graph still
/// references them.
pub(crate) struct RetainedWeights {
    pub(crate) gate: Arc<CudaSlice<u8>>,
    pub(crate) up: Arc<CudaSlice<u8>>,
    pub(crate) down: Arc<CudaSlice<u8>>,
    /// Pre-FFN norm weight (device-resident, stable address).
    pub(crate) pre_norm_weight: CudaSlice<f32>,
    /// Post-FFN norm weight (only when `has_post_norms`).
    pub(crate) post_norm_weight: Option<CudaSlice<f32>>,
}

impl ResidentFfnGraph {
    /// Replay the captured graph on its capture stream. Returns `Err` on
    /// replay failure (the caller falls back to the existing resident device
    /// FFN chain for this layer — never directly to CPU).
    pub(crate) fn replay(&self) -> Result<(), DriverError> {
        let Some(ref graph) = self.graph else {
            return Err(DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
            ));
        };
        graph.launch()
    }
}

impl Drop for ResidentFfnGraph {
    fn drop(&mut self) {
        // Explicit teardown order (B3A review point 7):
        // 1. destroy executable graph (CudaGraph::drop → execDestroy + destroy).
        //    The captured CUgraph is owned inside CudaGraph, so it's destroyed
        //    with it (step 2 is implicit).
        self.graph.take();
        // 3. release graph scratch + norm buffers.
        self.scratch.take();
        // 4. release retained weight-buffer handles.
        self.weights.take();
        // 5. stream/context destroyed later by CudaRuntime (outlives the cache).
    }
}

// SAFETY (Send + Sync for the graph-state types):
//
// `CudaGraph` is documented by cudarc as NOT internally synchronized — "API
// calls accessing the same graph object must be serialized externally." LARQL's
// resident decode is single-threaded per token (the existing `CudaBackend`
// contract: one decode stream, no concurrent graph access), and every graph-
// state field is reachable only through a `Mutex` on `CudaBackend`. The decode
// loop holds the mutex briefly (to look up or insert a graph entry) and drops
// the guard before launching; no two threads ever touch the same graph object
// concurrently. Asserting `Send + Sync` here is sound under that contract —
// the `Mutex` provides the external serialization cudarc requires.
unsafe impl Send for ResidentFfnGraph {}
unsafe impl Sync for ResidentFfnGraph {}
unsafe impl Send for ResidentFfnGraphScratch {}
unsafe impl Sync for ResidentFfnGraphScratch {}
unsafe impl Send for RetainedWeights {}
unsafe impl Sync for RetainedWeights {}
unsafe impl Send for ResidentFfnGraphCache {}
unsafe impl Sync for ResidentFfnGraphCache {}
unsafe impl Send for ResidentDecodeArena {}
unsafe impl Sync for ResidentDecodeArena {}

/// Generation-scoped cache of per-layer FFN graphs.
///
/// Keyed by `(generation, layer_index)` — the plan-level identity
/// ([`crate::ffn_graph::ResidentFfnPlanKey`]) is checked at build time but
/// not stored in the key, because each layer captures distinct weight
/// pointers and is never deduplicated across layers (B3A review point 6).
///
/// On [`Self::reset`]: every executable graph is destroyed (via `Drop`),
/// scratch + retained weights released, then the caller flushes the weight
/// cache. The generation advances so a stale graph can never be replayed
/// under a new generation.
pub(crate) struct ResidentFfnGraphCache {
    pub(crate) generation: GraphGenerationId,
    /// One optional graph entry per layer. `None` = not yet built or ineligible.
    pub(crate) layers: Vec<Option<ResidentFfnGraph>>,
}

impl ResidentFfnGraphCache {
    pub(crate) fn new() -> Self {
        Self {
            generation: GraphGenerationId::INITIAL,
            layers: Vec::new(),
        }
    }

    /// Ensure the layer vector has `num_layers` slots, resetting the cache if
    /// the layer count changed (a new model/vindex). Does NOT destroy existing
    /// graphs unless the count changed — the generation reset handles that.
    pub(crate) fn ensure_capacity(&mut self, num_layers: usize) {
        if self.layers.len() != num_layers {
            self.layers.clear();
            self.layers.resize_with(num_layers, || None);
        }
    }

    /// Reset for a new generation: destroy all graphs, clear slots, advance
    /// the generation. Called at `reset_kv_cache` BEFORE the weight cache is
    /// flushed (so graph teardown references valid buffers).
    pub(crate) fn reset(&mut self) {
        // Dropping each entry runs ResidentFfnGraph::drop (ordered teardown).
        for slot in &mut self.layers {
            slot.take();
        }
        self.generation = self.generation.next();
    }

    /// Get the graph entry for a layer (for replay), or `None` if not built.
    pub(crate) fn get(&self, layer_index: usize) -> Option<&ResidentFfnGraph> {
        self.layers.get(layer_index).and_then(|s| s.as_ref())
    }
}

impl Default for ResidentFfnGraphCache {
    fn default() -> Self {
        Self::new()
    }
}

/// The resident decode arena: two stable hidden-state ping-pong buffers +
/// the dedicated capture/replay stream.
///
/// Allocated once per generation (lazily, at the first decode token where the
/// graph path is eligible). The two `[hidden]` buffers are the I/O slots the
/// attention block and the FFN graph write into by stable address; their
/// contents change every token but their addresses are fixed for the arena's
/// lifetime (B3A review point 2 — the graph never owns these; the arena does).
///
/// **Ownership**:
/// - `hidden_a` / `hidden_b`: owned here. Attention writes one; the FFN graph
///   writes the other. The `flip` flag tracks which is "current input".
/// - `cap_stream`: the dedicated non-NULL stream for capture + replay.
///
/// **Cross-stream sync**: attention runs on the runtime's NULL stream; the
/// graph replays on `cap_stream`. The decode loop inserts a sync between them
/// at each layer boundary (see the pipeline integration in B3A-5).
pub(crate) struct ResidentDecodeArena {
    /// Stable hidden-state buffer A `[hidden]`.
    pub(crate) hidden_a: CudaSlice<f32>,
    /// Stable hidden-state buffer B `[hidden]`.
    pub(crate) hidden_b: CudaSlice<f32>,
    /// The dedicated capture/replay stream (non-NULL, event-tracking disabled).
    pub(crate) cap_stream: Arc<CudaStream>,
    /// The generation this arena was allocated for. Stale after a reset.
    pub(crate) generation: GraphGenerationId,
}

impl ResidentDecodeArena {
    /// Allocate the arena for a generation: two `[hidden]` buffers + a
    /// dedicated capture stream. Disables event tracking on the context so
    /// graph buffers don't carry `CudaEvent` handles (which would break
    /// capture — see B3A-SMOKE finding #2).
    ///
    /// SAFETY: `disable_event_tracking` is unsafe because slices created
    /// before the call won't be tracked. The arena creates all its buffers
    /// AFTER this call, and the decode path is single-threaded, so all
    /// cross-stream synchronization is explicit (managed by the pipeline's
    /// layer-boundary syncs). This is the documented configuration for graph
    /// capture.
    pub(crate) fn new(
        runtime: &CudaRuntime,
        hidden: usize,
        generation: GraphGenerationId,
    ) -> Result<Self, RuntimeError> {
        let ctx = runtime.stream().context().clone();
        let cap_stream = ctx
            .new_stream()
            .map_err(|err| RuntimeError::context("creating graph capture stream", err))?;
        // Disable event tracking so graph buffers carry no CudaEvent handles.
        // SAFETY: the arena manages all synchronization explicitly on cap_stream
        // + the runtime stream; no graph buffer is used on a third stream.
        // Created BEFORE the buffers below so they carry no events.
        unsafe {
            cap_stream.context().disable_event_tracking();
        }
        let hidden_a = cap_stream
            .alloc_zeros::<f32>(hidden)
            .map_err(|err| RuntimeError::context("allocating arena hidden_a", err))?;
        let hidden_b = cap_stream
            .alloc_zeros::<f32>(hidden)
            .map_err(|err| RuntimeError::context("allocating arena hidden_b", err))?;
        Ok(Self {
            hidden_a,
            hidden_b,
            cap_stream,
            generation,
        })
    }

    /// The "input" buffer for the current layer (the one attention just wrote).
    /// Layers alternate: even layers read A, odd layers read B (flip by index).
    pub(crate) fn input(&self, flip: bool) -> &CudaSlice<f32> {
        if flip {
            &self.hidden_b
        } else {
            &self.hidden_a
        }
    }

    /// The "output" buffer for the current layer (the one the FFN graph writes).
    /// Opposite of input — after the graph writes, the next layer reads it.
    pub(crate) fn output(&self, flip: bool) -> &CudaSlice<f32> {
        if flip {
            &self.hidden_a
        } else {
            &self.hidden_b
        }
    }

    /// Synchronize the capture stream with the runtime stream: the capture
    /// stream waits for the runtime stream to finish writing the arena input.
    /// Called before graph replay when the input was produced on the runtime
    /// stream.
    fn _doc_anchor(&self) {
        // Method exists to anchor the cross-stream sync documentation; the
        // actual sync is performed in the pipeline where both streams are
        // accessible. See `CudaBackend::sync_graph_input` in pipeline.rs.
    }
}

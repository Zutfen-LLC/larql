//! LARQL-GPU-B3B: resident decode arena + FFN graph state.
//!
//! This module owns the **device side** of the CUDA Graph replay for the
//! resident decode FFN. The pure plan/identity contract lives in
//! [`crate::ffn_graph`]; this module holds the graph-captured state, the
//! stable scratch buffers, and the generation-scoped cache.
//!
//! ## Architecture (B3B single non-NULL decode stream)
//!
//! ### One stream for everything
//!
//! B3B replaced the NULL/default runtime stream with a single dedicated
//! non-NULL stream (see `CudaRuntime::initialize_impl`) and removed B3A's
//! separate `cap_stream`. Graph capture AND replay now run on the runtime
//! stream — the same stream attention, KV append, residual, and every other
//! decode kernel use. Layer-to-layer ordering is therefore by stream
//! submission alone: zero per-layer D2D and zero per-layer cross-stream syncs.
//!
//! ### Arena slot flip/ownership (the ping-pong that powers zero-copy replay)
//!
//! The arena owns two stable `[hidden]` buffers (`hidden_a`, `hidden_b`),
//! allocated once per generation on the runtime stream and reused by address
//! for the whole generation (graph capture binds their device addresses).
//! Per layer `flip = li % 2 == 1`:
//!
//! - **`input(flip)`** is the layer's hidden input AND the FFN graph's input
//!   slot. Attention reads it for the input norm, then writes its post-attn
//!   residual **in place** into it (`launch_residual_add_inplace_into`) — so
//!   the graph reads the post-attn residual from the exact address it
//!   captured, with no D2D seed copy.
//! - **`output(flip)`** is the FFN graph's output slot (= the next layer's
//!   `input(¬flip)`). The graph writes the post-FFN state there; the next
//!   layer's attention reads it directly — no D2D output copy.
//!
//! Because `output(flip) == input(¬flip)`, layer N's input is layer N-1's
//! graph output, carried by flip alone. Layer 0 re-uploads each token's
//! embedding into `input(false)` (a normal HtoD, not a D2D). The in-place
//! residual is sound: `residual_add` is element-wise independent, and a single
//! stream guarantees no concurrent access to the slot.
//!
//! ### Capture safety
//!
//! CudaEvent tracking is disabled context-wide at runtime init, so
//! `launch_builder.arg(&CudaSlice)` injects no `cuStreamWaitEvent` during
//! capture (which would yield `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`). On one
//! stream, submission order is the only ordering needed.

use crate::backend::CudaRuntime;
use crate::backend::RuntimeError;
use crate::ffn_graph::GraphGenerationId;
use cudarc::driver::{CudaGraph, CudaSlice, DriverError};
use std::sync::Arc;

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
// NOTE: the fields are never *read* via Rust — they exist to pin the device
// addresses the captured graph binds for the graph's lifetime (their `Drop`
// frees the device memory after the graph is destroyed).
#[allow(dead_code)]
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
// NOTE: fields pin device addresses for the graph's lifetime (see
// `ResidentFfnGraphScratch`); they are dropped, not read.
#[allow(dead_code)]
pub(crate) struct RetainedWeights {
    #[allow(dead_code)]
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

/// The resident decode arena: two stable hidden-state ping-pong buffers.
///
/// Allocated once per generation (lazily, at the first decode token where the
/// graph path is eligible) on the runtime stream. The two `[hidden]` buffers
/// are the I/O slots the attention block and the FFN graph write into by stable
/// address; their contents change every token but their addresses are fixed
/// for the arena's lifetime (graph capture binds these addresses).
///
/// **Ownership / flip semantics** (B3B single stream):
/// - `hidden_a` / `hidden_b`: owned here. Per layer `flip = li % 2 == 1`:
///   `input(flip)` is the layer's hidden input AND the FFN graph's input slot
///   (attention writes its post-attn residual into it in place);
///   `output(flip)` is the FFN graph's output slot and equals
///   `input(¬flip)` (the next layer's input). Layer 0 re-uploads each
///   token's embedding into `input(false)` (an HtoD, not a D2D).
/// - There is no longer a dedicated capture stream: capture + replay run on
///   the runtime stream (`CudaRuntime::stream`), so layer-to-layer ordering is
///   by stream submission alone (zero per-layer D2D, zero cross-stream syncs).
pub(crate) struct ResidentDecodeArena {
    /// Stable hidden-state buffer A `[hidden]` = `input(false)` / `output(true)`.
    pub(crate) hidden_a: CudaSlice<f32>,
    /// Stable hidden-state buffer B `[hidden]` = `input(true)` / `output(false)`.
    pub(crate) hidden_b: CudaSlice<f32>,
    /// The generation this arena was allocated for. Stale after a reset.
    pub(crate) generation: GraphGenerationId,
}

impl ResidentDecodeArena {
    /// Allocate the arena for a generation: two `[hidden]` buffers on the
    /// runtime stream. Event tracking is already disabled context-wide at
    /// runtime init (B3B), so the buffers carry no `CudaEvent` handles —
    /// capture-safe with no further action.
    pub(crate) fn new(
        runtime: &CudaRuntime,
        hidden: usize,
        generation: GraphGenerationId,
    ) -> Result<Self, RuntimeError> {
        let stream = runtime.stream();
        let hidden_a = stream
            .alloc_zeros::<f32>(hidden)
            .map_err(|err| RuntimeError::context("allocating arena hidden_a", err))?;
        let hidden_b = stream
            .alloc_zeros::<f32>(hidden)
            .map_err(|err| RuntimeError::context("allocating arena hidden_b", err))?;
        Ok(Self {
            hidden_a,
            hidden_b,
            generation,
        })
    }

    /// Borrow the input buffer immutably. The layer's hidden input AND the FFN
    /// graph's input slot: attention reads it for the input norm, then writes
    /// its post-attn residual into it in place (`launch_residual_add_inplace_
    /// into`), and the graph reads the post-attn residual from this address.
    pub(crate) fn input(&self, flip: bool) -> &CudaSlice<f32> {
        if flip {
            &self.hidden_b
        } else {
            &self.hidden_a
        }
    }

    /// Borrow the input buffer mutably — for the layer-0 HtoD upload of the
    /// embedding and for re-entry placement of a carried device buffer.
    pub(crate) fn input_mut(&mut self, flip: bool) -> &mut CudaSlice<f32> {
        if flip {
            &mut self.hidden_b
        } else {
            &mut self.hidden_a
        }
    }

    /// Borrow the output buffer immutably. The FFN graph writes the post-FFN
    /// state here; it equals `input(¬flip)` (the next layer's input). The
    /// final layer's output is read back here for the lm-head input.
    pub(crate) fn output(&self, flip: bool) -> &CudaSlice<f32> {
        if flip {
            &self.hidden_a
        } else {
            &self.hidden_b
        }
    }

    /// Borrow both the input and output buffers mutably simultaneously. They
    /// are disjoint fields (`hidden_a` / `hidden_b`), so this is safe and lets
    /// the graph-capture path hold both a `&CudaSlice` input (read) and a
    /// `&mut CudaSlice` output (write) across the captured kernel launches.
    pub(crate) fn input_output_mut(
        &mut self,
        flip: bool,
    ) -> (&CudaSlice<f32>, &mut CudaSlice<f32>) {
        if flip {
            (&self.hidden_b, &mut self.hidden_a)
        } else {
            (&self.hidden_a, &mut self.hidden_b)
        }
    }
}

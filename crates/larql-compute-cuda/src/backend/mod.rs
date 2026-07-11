mod runtime;

use crate::kv_cache::{CudaKVCache, KvCacheError};
use crate::options::BackendOptions;
use crate::pipeline::HostKvType;
use cudarc::driver::CudaSlice;
use std::sync::{Arc, Mutex};

pub(crate) use runtime::{CudaRuntime, RuntimeError};

pub struct CudaBackend {
    options: BackendOptions,
    runtime: Option<Arc<CudaRuntime>>,
    runtime_status: Option<String>,
    /// Device-side KV cache, allocated lazily via
    /// `preallocate_kv_cache_per_layer`. `None` when no runtime is present
    /// (scaffold path) or before the first prefill. Mirrors Metal's
    /// `kv_cache: Mutex<Option<KVCache>>`.
    kv_cache: Mutex<Option<CudaKVCache>>,
    /// Host-side KV mirror used by the host-orchestrated decode/prefill
    /// pipeline (`pipeline.rs`). One `[len, kv_dim]` `(K, V)` pair per layer.
    /// Reset by `reset_host_kv` at the start of every prefill; grown by one
    /// row per decode step. Attention reads from this mirror (the device
    /// `kv_cache` stays populated for the `DecodeBackend` lifecycle contract
    /// but device-side attention kernels aren't implemented yet).
    pub(crate) host_kv: Mutex<HostKvType>,
    /// GPU-006 diagnostics: how many decode attention launches used the
    /// resident-KV device path (`true`) vs. fell back to the full-KV-upload
    /// path (`false`). Surfaced via `device_info()` when `LARQL_GPU_DIAG=1`.
    /// Thread-safe (atomic) so concurrent decode steps don't contend.
    resident_kv_decode_uses: std::sync::atomic::AtomicU64,
    resident_kv_decode_fallbacks: std::sync::atomic::AtomicU64,
    /// GPU-007 diagnostics: how many decode layers carried the hidden state
    /// device-resident across the attention→FFN→next-layer chain (`true`) vs.
    /// fell back to the host-orchestrated per-block path (`false`). Surfaced
    /// via `device_info()` when `LARQL_GPU_DIAG=1`. Thread-safe (atomic).
    resident_hidden_uses: std::sync::atomic::AtomicU64,
    resident_hidden_fallbacks: std::sync::atomic::AtomicU64,
    /// LARQL-GPU-PROFILE-001 gated decomposition counters. All zeroed at
    /// construction; only mutated when `LARQL_GPU_PROFILE=1` is set (the
    /// recorder methods short-circuit otherwise). Surfaced via
    /// `profile_diag()` / `take_profile_counters()`.
    profile: ProfileCounters,
    /// B3A: the resident-FFN graph replay mode, resolved ONCE at backend
    /// construction from `LARQL_CUDA_GRAPHS` (B3A review point 9 — the hot
    /// path reads this field, never process-global env). `Auto` = use the
    /// graph path when a layer is eligible; `Disabled` = never (kill switch).
    /// Default is `Disabled` until the B3A-11 performance gate flips it.
    #[allow(dead_code)]
    graph_mode: crate::ffn_graph::GraphMode,
    /// B3A: capture-aware graph profiling counters (point 8). All zeroed at
    /// construction; only mutated when `LARQL_GPU_PROFILE=1`. Surfaced via
    /// `graph_diag()` / `take_profile_counters()`.
    #[allow(dead_code)]
    graph_profile: GraphProfileCounters,
    /// LARQL-GPU-B4 device-greedy profiling counters.
    greedy_profile: GreedyProfileCounters,
    /// B3A: generation-scoped cache of per-layer resident-FFN executable
    /// graphs. Allocated lazily; reset (graphs destroyed, generation
    /// advanced) at every `reset_kv_cache` BEFORE the weight cache flushes.
    #[allow(dead_code)]
    pub(crate) graph_cache: std::sync::Mutex<crate::ffn_graph_state::ResidentFfnGraphCache>,
    /// B3A: the resident decode arena (ping-pong hidden buffers + capture
    /// stream). `None` until the first graph-eligible decode token allocates
    /// it; dropped (with the graphs) at generation reset.
    #[allow(dead_code)]
    pub(crate) arena: std::sync::Mutex<Option<crate::ffn_graph_state::ResidentDecodeArena>>,
    /// LARQL-GPU-B4: generation-scoped device-side greedy-head workspace.
    /// `None` until the first eligible greedy decode token allocates it;
    /// rebuilt when `(logical_rows, hidden, candidate_width)` changes;
    /// dropped at generation reset (`reset_kv_cache`).
    pub(crate) greedy_workspace:
        std::sync::Mutex<Option<crate::greedy_workspace::GreedyHeadWorkspace>>,
    /// Test-only eligibility hook for exercising a resident → fallback →
    /// resident transition without changing any layer math.
    #[cfg(test)]
    resident_hidden_forced_fallback_layer: std::sync::atomic::AtomicUsize,
}

/// Internal atomic storage for the LARQL-GPU-PROFILE-001 decomposition
/// counters (see [`CudaBackend::profile`]). All fields are `AtomicU64` with
/// `Ordering::Relaxed` — the decode hot path is single-threaded per token, so
/// relaxed ordering is sufficient and avoids any fence cost. The recorder
/// methods on `CudaBackend` gate on `gpu_profile_enabled()` before touching
/// any of these, so normal decode never increments them.
#[derive(Default)]
struct ProfileCounters {
    mirror_append_ns: std::sync::atomic::AtomicU64,
    mirror_rows_copied: std::sync::atomic::AtomicU64,
    hidden_readback_ns: std::sync::atomic::AtomicU64,
}

/// B3A capture-aware graph profiling counters (review point 8). All
/// `AtomicU64` with `Ordering::Relaxed`; only mutated when
/// `LARQL_GPU_PROFILE=1` (the recorder methods short-circuit otherwise).
///
/// The distinction between `direct_kernel_submissions` (individual `*_dev`
/// launches, counted by `note_launch`), `graph_submissions` (one per
/// `CudaGraph::launch()`), and `captured_kernel_nodes` (the physical nodes a
/// graph contains, counted once at build) is what makes the submission-
/// reduction accounting honest. `logical_graph_kernel_executions` =
/// `captured_kernel_nodes` × replays.
//
// The fields are written by the `note_graph_*` recorder methods which land
// with the B3A-5 pipeline integration. Until then the `dead_code` allow
// silences the "never read" warning.
#[derive(Default)]
#[allow(dead_code)]
struct GraphProfileCounters {
    /// One per `CudaGraph::launch()` (incl. token 1 — review point 1).
    graph_submissions: std::sync::atomic::AtomicU64,
    /// One per `ResidentFfnGraph::build()` (one per eligible layer per gen).
    graph_builds: std::sync::atomic::AtomicU64,
    /// += kernel_node_count() at build time.
    captured_kernel_nodes: std::sync::atomic::AtomicU64,
    /// += kernel_node_count() per replay.
    logical_graph_kernel_executions: std::sync::atomic::AtomicU64,
    /// Build `Err` count.
    graph_failures: std::sync::atomic::AtomicU64,
    /// Replay `Err` → resident device chain fallback count.
    graph_fallbacks: std::sync::atomic::AtomicU64,
    /// Any D2D copy on the graph path (the cross-stream seed + output copies
    /// the separate cap_stream design requires; counted honestly per B3A
    /// review point 3 — never claimed zero unless the design achieves zero).
    d2d_submissions: std::sync::atomic::AtomicU64,
    /// Explicit cross-stream `synchronize()` calls the graph path issues
    /// (runtime-stream↔cap_stream handoff, outside any capture window). The
    /// previous accounting hid these because raw `stream.synchronize()`
    /// bypasses `note_sync`; counting them here keeps the sync total honest
    /// (B3A review point 3).
    graph_cross_stream_syncs: std::sync::atomic::AtomicU64,
}

/// LARQL-GPU-B4 device-greedy profiling counters. All `AtomicU64` with
/// `Ordering::Relaxed`; only mutated when `LARQL_GPU_PROFILE=1` (the
/// recorder methods short-circuit otherwise). Surfaced via
/// `take_profile_counters()`.
#[derive(Default)]
pub(crate) struct GreedyProfileCounters {
    /// Calls to `decode_token_greedy_q4k` on an eligible plain-greedy step.
    pub(crate) attempts: std::sync::atomic::AtomicU64,
    /// Device returned a `DevicePick`.
    pub(crate) engaged: std::sync::atomic::AtomicU64,
    /// Device computed the hidden state but returned `HostHidden`.
    pub(crate) fallbacks: std::sync::atomic::AtomicU64,
    /// Device could not even read back the hidden state.
    pub(crate) failures: std::sync::atomic::AtomicU64,
    /// Full lm-head score-vector DtoH copies on the host path.
    pub(crate) lm_head_full_score_dtoh_copies: std::sync::atomic::AtomicU64,
    pub(crate) lm_head_full_score_dtoh_bytes: std::sync::atomic::AtomicU64,
    /// Fixed-size candidate-result DtoH copies on the B4 path.
    pub(crate) lm_head_result_dtoh_copies: std::sync::atomic::AtomicU64,
    pub(crate) lm_head_result_dtoh_bytes: std::sync::atomic::AtomicU64,
    /// Final-hidden-state DtoH readbacks on the host path.
    pub(crate) final_hidden_readbacks: std::sync::atomic::AtomicU64,
    pub(crate) final_hidden_readback_bytes: std::sync::atomic::AtomicU64,
}

/// A consumed snapshot of the profile counters, returned by
/// [`CudaBackend::take_profile_counters`] (the inherent method). Plain values
/// (the atomics have been reset to zero). Distinct from the trait-level
/// [`larql_compute::ProfileCountersSnapshot`] so the CUDA crate doesn't
/// re-export a type into the shared substrate; the trait impl converts.
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaProfileSnapshot {
    pub launches: u64,
    pub htod_copies: u64,
    pub htod_bytes: u64,
    pub dtoh_copies: u64,
    pub dtoh_bytes: u64,
    pub syncs: u64,
    pub mirror_append_ns: u64,
    pub mirror_rows_copied: u64,
    pub hidden_readback_ns: u64,
    // ── B3A capture-aware graph counters (review points 3, 8) ──
    // These make submission-reduction accounting honest: total host CUDA
    // submissions/token = launches + graph_submissions + d2d_submissions.
    // The previous accounting reported only `launches`, excluding the cap_stream
    // graph launches and D2D copies the graph path adds.
    /// One `CudaGraph::launch()` per eligible layer per token (incl. token 1).
    pub graph_submissions: u64,
    /// One per `ResidentFfnGraph::build()` (one per eligible layer per gen).
    pub graph_builds: u64,
    /// Physical kernel nodes a captured graph contains (+= at build).
    pub captured_kernel_nodes: u64,
    /// `captured_kernel_nodes` × replays.
    pub logical_graph_kernel_executions: u64,
    /// Build `Err` count.
    pub graph_failures: u64,
    /// Replay `Err` → resident device chain fallback count.
    pub graph_fallbacks: u64,
    /// D2D copies on the graph path (seed + output; the cross-stream cost).
    pub d2d_submissions: u64,
    /// Explicit cross-stream `synchronize()` calls the graph path issues.
    pub graph_cross_stream_syncs: u64,
    // ── LARQL-GPU-B4 device-greedy counters ─────────────────────────────
    pub device_greedy_attempts: u64,
    pub device_greedy_engaged: u64,
    pub device_greedy_fallbacks: u64,
    pub device_greedy_failures: u64,
    pub lm_head_full_score_dtoh_copies: u64,
    pub lm_head_full_score_dtoh_bytes: u64,
    pub lm_head_result_dtoh_copies: u64,
    pub lm_head_result_dtoh_bytes: u64,
    pub final_hidden_readbacks: u64,
    pub final_hidden_readback_bytes: u64,
}

impl From<CudaProfileSnapshot> for larql_compute::ProfileCountersSnapshot {
    fn from(c: CudaProfileSnapshot) -> Self {
        Self {
            launches: c.launches,
            htod_copies: c.htod_copies,
            htod_bytes: c.htod_bytes,
            dtoh_copies: c.dtoh_copies,
            dtoh_bytes: c.dtoh_bytes,
            syncs: c.syncs,
            mirror_append_ns: c.mirror_append_ns,
            mirror_rows_copied: c.mirror_rows_copied,
            hidden_readback_ns: c.hidden_readback_ns,
            graph_submissions: c.graph_submissions,
            graph_builds: c.graph_builds,
            captured_kernel_nodes: c.captured_kernel_nodes,
            logical_graph_kernel_executions: c.logical_graph_kernel_executions,
            graph_failures: c.graph_failures,
            graph_fallbacks: c.graph_fallbacks,
            d2d_submissions: c.d2d_submissions,
            graph_cross_stream_syncs: c.graph_cross_stream_syncs,
            device_greedy_attempts: c.device_greedy_attempts,
            device_greedy_engaged: c.device_greedy_engaged,
            device_greedy_fallbacks: c.device_greedy_fallbacks,
            device_greedy_failures: c.device_greedy_failures,
            lm_head_full_score_dtoh_copies: c.lm_head_full_score_dtoh_copies,
            lm_head_full_score_dtoh_bytes: c.lm_head_full_score_dtoh_bytes,
            lm_head_result_dtoh_copies: c.lm_head_result_dtoh_copies,
            lm_head_result_dtoh_bytes: c.lm_head_result_dtoh_bytes,
            final_hidden_readbacks: c.final_hidden_readbacks,
            final_hidden_readback_bytes: c.final_hidden_readback_bytes,
        }
    }
}

/// Atomic launch/copy/sync counters held by [`CudaRuntime`] (every kernel
/// launch and every host↔device copy flows through the runtime's single
/// stream). Mirrors [`ProfileCounters`] but lives on the `Arc`-shared runtime
/// rather than the per-request `CudaBackend`, so the `*_dev` device-resident
/// chains (which call `launch_*_dev` directly, bypassing the `native_*`
/// wrappers) are counted too.
#[derive(Default, Debug)]
pub(crate) struct RuntimeProfile {
    launches: std::sync::atomic::AtomicU64,
    htod_copies: std::sync::atomic::AtomicU64,
    htod_bytes: std::sync::atomic::AtomicU64,
    dtoh_copies: std::sync::atomic::AtomicU64,
    dtoh_bytes: std::sync::atomic::AtomicU64,
    syncs: std::sync::atomic::AtomicU64,
}

/// Consumed snapshot of [`RuntimeProfile`], returned by
/// [`CudaRuntime::take_profile_counters`].
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct RuntimeProfileSnapshot {
    pub launches: u64,
    pub htod_copies: u64,
    pub htod_bytes: u64,
    pub dtoh_copies: u64,
    pub dtoh_bytes: u64,
    pub syncs: u64,
}

impl CudaBackend {
    pub fn new() -> Result<Self, BackendInitError> {
        Self::with_options(BackendOptions::default())
    }

    pub fn with_options(options: BackendOptions) -> Result<Self, BackendInitError> {
        let graph_mode = crate::ffn_graph::graph_mode_from_env();
        match CudaRuntime::initialize(options.device_ordinal) {
            Ok(runtime) => Ok(Self {
                options,
                runtime: Some(Arc::new(runtime)),
                runtime_status: None,
                kv_cache: Mutex::new(None),
                host_kv: Mutex::new(Vec::new()),
                resident_kv_decode_uses: std::sync::atomic::AtomicU64::new(0),
                resident_kv_decode_fallbacks: std::sync::atomic::AtomicU64::new(0),
                resident_hidden_uses: std::sync::atomic::AtomicU64::new(0),
                resident_hidden_fallbacks: std::sync::atomic::AtomicU64::new(0),
                profile: ProfileCounters::default(),
                graph_mode,
                graph_profile: GraphProfileCounters::default(),
                greedy_profile: GreedyProfileCounters::default(),
                graph_cache: std::sync::Mutex::new(
                    crate::ffn_graph_state::ResidentFfnGraphCache::new(),
                ),
                arena: std::sync::Mutex::new(None),
                greedy_workspace: std::sync::Mutex::new(None),
                #[cfg(test)]
                resident_hidden_forced_fallback_layer: std::sync::atomic::AtomicUsize::new(
                    usize::MAX,
                ),
            }),
            Err(err) if options.allow_cpu_delegate => Ok(Self {
                options,
                runtime: None,
                runtime_status: Some(err.to_string()),
                kv_cache: Mutex::new(None),
                host_kv: Mutex::new(Vec::new()),
                resident_kv_decode_uses: std::sync::atomic::AtomicU64::new(0),
                resident_kv_decode_fallbacks: std::sync::atomic::AtomicU64::new(0),
                resident_hidden_uses: std::sync::atomic::AtomicU64::new(0),
                resident_hidden_fallbacks: std::sync::atomic::AtomicU64::new(0),
                profile: ProfileCounters::default(),
                graph_mode,
                graph_profile: GraphProfileCounters::default(),
                greedy_profile: GreedyProfileCounters::default(),
                graph_cache: std::sync::Mutex::new(
                    crate::ffn_graph_state::ResidentFfnGraphCache::new(),
                ),
                arena: std::sync::Mutex::new(None),
                greedy_workspace: std::sync::Mutex::new(None),
                #[cfg(test)]
                resident_hidden_forced_fallback_layer: std::sync::atomic::AtomicUsize::new(
                    usize::MAX,
                ),
            }),
            Err(err) => Err(BackendInitError::Unavailable(err.to_string())),
        }
    }

    pub fn options(&self) -> &BackendOptions {
        &self.options
    }

    pub(crate) fn native_q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => runtime
                .launch_q4k_matvec(q4k_data, x, num_rows, hidden)
                .map(Some),
            None => Ok(None),
        }
    }

    pub(crate) fn native_q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => runtime
                .launch_q6k_matvec(q6k_data, x, num_rows, hidden)
                .map(Some),
            None => Ok(None),
        }
    }

    pub(crate) fn native_q4k_matmul(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => runtime
                .launch_q4k_matmul(q4k_data, x, num_rows, hidden, seq_len)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Native Q6_K amortised matmul, routed through the
    /// `QuantMatVec::q6k_matmul` trait method (native-then-CPU fallback).
    /// Parity-verified when a CUDA runtime is present.
    pub(crate) fn native_q6k_matmul(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => runtime
                .launch_q6k_matmul(q6k_data, x, num_rows, hidden, seq_len)
                .map(Some),
            None => Ok(None),
        }
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn native_q4k_dual_matvec(
        &self,
        q4k_a: &[u8],
        q4k_b: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Option<(Vec<f32>, Vec<f32>)>, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => runtime
                .launch_q4k_dual_matvec(q4k_a, q4k_b, x, num_rows, hidden)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Native dense f32 GEMV. `w` is the row-major `[n, k]` slice behind an
    /// `ArrayView2` — the launcher checks the slice length against `n*k`.
    pub(crate) fn native_f32_gemv(
        &self,
        w: &[f32],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => runtime.launch_f32_gemv(w, x, num_rows, hidden).map(Some),
            None => Ok(None),
        }
    }

    /// Native dense f16 GEMV. `w_f16` is a row-major `[n, k]` little-endian
    /// f16 byte slice.
    pub(crate) fn native_f16_gemv(
        &self,
        w_f16: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => runtime
                .launch_f16_gemv(w_f16, x, num_rows, hidden)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Native Q4_0 × Q8 matvec. `q4_data` is packed Q4_0 (18 bytes per
    /// 32-element block); `q8_x` / `q8_scales` are the pre-quantised Q8
    /// input. Routed through `QuantMatVec::q4_matvec` (native-then-CPU
    /// fallback).
    pub(crate) fn native_q4_matvec(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => runtime
                .launch_q4_matvec(q4_data, q8_x, q8_scales, num_rows, hidden)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Native Q4_0 vector-matrix. Routed through `QuantMatVec::q4_vecmat`
    /// (native-then-CPU fallback).
    pub(crate) fn native_q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => runtime
                .launch_q4_vecmat(activation, q4_data, intermediate, hidden)
                .map(Some),
            None => Ok(None),
        }
    }

    pub(crate) fn native_runtime_available(&self) -> bool {
        self.runtime.is_some()
    }

    /// Borrow the device runtime (`None` on the scaffold path). Used by the
    /// device-resident activation chain in `pipeline.rs` to drive several
    /// kernels on the same stream and read back once.
    pub(crate) fn runtime(&self) -> Option<&Arc<CudaRuntime>> {
        self.runtime.as_ref()
    }

    /// B3A: the resident-FFN graph replay mode (B3A review point 9). Resolved
    /// once at construction from `LARQL_CUDA_GRAPHS`; the decode hot path reads
    /// this field, never process-global env.
    #[allow(dead_code)]
    pub(crate) fn graph_mode(&self) -> crate::ffn_graph::GraphMode {
        self.graph_mode
    }

    /// Snapshot the persistent weight-cache hit/miss counters and resident-byte
    /// footprint. `None` on the scaffold path (no runtime / no device); the
    /// counts are always zero there since no weights are ever uploaded.
    /// Release-visible so the `LARQL_GPU_DIAG` diagnostic surface can read it
    /// from a release build.
    pub(crate) fn weight_cache_stats(&self) -> Option<crate::weight_cache::CacheStats> {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.weight_cache_stats())
    }

    /// A one-line diagnostic string for the current weight-cache state (hit /
    /// miss counters + resident byte footprint), or `None` on the scaffold
    /// path (no runtime). Surfaced through `device_info()` when
    /// `LARQL_GPU_DIAG` is set; otherwise this method is still callable but
    /// produces no default output.
    pub fn weight_cache_diag(&self) -> Option<String> {
        let s = self.weight_cache_stats()?;
        // Hit rate over the union of byte + float lookups (cumulative, so a
        // fresh backend reads 0/0 → reported as "n/a" rather than 100%).
        let total = s.bytes_hits + s.bytes_misses + s.float_hits + s.float_misses;
        let hit_rate = if total == 0 {
            "n/a".to_string()
        } else {
            let hits = s.bytes_hits + s.float_hits;
            format!("{:.1}%", (hits as f64 / total as f64) * 100.0)
        };
        Some(format!(
            "cuda weight cache: bytes hit/miss={}/{}, float hit/miss={}/{}, hit_rate={}, resident={} bytes ({} u8 + {} f32 buffers)",
            s.bytes_hits,
            s.bytes_misses,
            s.float_hits,
            s.float_misses,
            hit_rate,
            s.cached_bytes,
            s.cached_u8_buffers,
            s.cached_f32_buffers,
        ))
    }

    /// Record one resident-KV decode attention outcome for the `LARQL_GPU_DIAG`
    /// surface (GPU-006). `resident = true` → the device-resident KV path was
    /// used; `false` → fell back to the full-KV-upload path. Thread-safe.
    pub(crate) fn note_resident_kv_decode(&self, resident: bool) {
        use std::sync::atomic::Ordering;
        if resident {
            self.resident_kv_decode_uses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.resident_kv_decode_fallbacks
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A one-line diagnostic string for the resident-KV decode attention
    /// usage, or `None` when neither path has ever run (counters at zero).
    /// Surfaced through `device_info()` when `LARQL_GPU_DIAG` is set.
    pub fn resident_kv_diag(&self) -> Option<String> {
        use std::sync::atomic::Ordering;
        let uses = self.resident_kv_decode_uses.load(Ordering::Relaxed);
        let fallbacks = self.resident_kv_decode_fallbacks.load(Ordering::Relaxed);
        if uses == 0 && fallbacks == 0 {
            return None;
        }
        let total = uses + fallbacks;
        let rate = if total == 0 {
            "n/a".to_string()
        } else {
            format!("{:.1}%", (uses as f64 / total as f64) * 100.0)
        };
        Some(format!(
            "cuda resident-KV decode: uses={uses}, fallbacks={fallbacks}, resident_rate={rate}"
        ))
    }

    /// Record one resident-hidden decode layer outcome for the `LARQL_GPU_DIAG`
    /// surface (GPU-007). `resident = true` → the layer carried the hidden
    /// state device-resident across attention→FFN→next-layer (no inter-block
    /// / inter-layer hidden-state readback); `false` → fell back to the
    /// host-orchestrated per-block path (the hidden state was read back to
    /// host between the attention and FFN blocks, or between layers).
    /// Thread-safe.
    pub(crate) fn note_resident_hidden(&self, resident: bool) {
        use std::sync::atomic::Ordering;
        if resident {
            self.resident_hidden_uses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.resident_hidden_fallbacks
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Force one layer through the resident-hidden host fallback in tests.
    /// This is intentionally backend-local and does not mutate the pipeline
    /// layer, so the fallback and CPU reference retain identical semantics.
    #[cfg(test)]
    pub(crate) fn force_resident_hidden_fallback_layer_for_test(&self, layer: Option<usize>) {
        use std::sync::atomic::Ordering;
        self.resident_hidden_forced_fallback_layer
            .store(layer.unwrap_or(usize::MAX), Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn resident_hidden_fallback_forced_for_test(&self, layer: usize) -> bool {
        use std::sync::atomic::Ordering;
        self.resident_hidden_forced_fallback_layer
            .load(Ordering::Relaxed)
            == layer
    }

    /// A one-line diagnostic string for the resident-hidden decode usage, or
    /// `None` when neither path has ever run (counters at zero). Surfaced
    /// through `device_info()` when `LARQL_GPU_DIAG` is set. Mirrors
    /// [`resident_kv_diag`] (GPU-006).
    pub fn resident_hidden_diag(&self) -> Option<String> {
        use std::sync::atomic::Ordering;
        let uses = self.resident_hidden_uses.load(Ordering::Relaxed);
        let fallbacks = self.resident_hidden_fallbacks.load(Ordering::Relaxed);
        if uses == 0 && fallbacks == 0 {
            return None;
        }
        let total = uses + fallbacks;
        let rate = if total == 0 {
            "n/a".to_string()
        } else {
            format!("{:.1}%", (uses as f64 / total as f64) * 100.0)
        };
        Some(format!(
            "cuda resident-hidden decode: uses={uses}, fallbacks={fallbacks}, resident_rate={rate}"
        ))
    }

    /// LARQL-GPU-B4 diagnostic: device-side greedy lm-head engagement, or
    /// `None` when no device-greedy step has run. Surfaced through
    /// `device_info()` when `LARQL_GPU_DIAG` is set.
    pub fn device_greedy_diag(&self) -> Option<String> {
        use std::sync::atomic::Ordering;
        let attempts = self.greedy_profile.attempts.load(Ordering::Relaxed);
        let engaged = self.greedy_profile.engaged.load(Ordering::Relaxed);
        let fallbacks = self.greedy_profile.fallbacks.load(Ordering::Relaxed);
        let failures = self.greedy_profile.failures.load(Ordering::Relaxed);
        if attempts == 0 {
            return None;
        }
        let rate = format!("{:.1}%", (engaged as f64 / attempts as f64) * 100.0);
        Some(format!(
            "cuda device-greedy lm-head: attempts={attempts}, engaged={engaged}, fallbacks={fallbacks}, failures={failures}, engage_rate={rate}"
        ))
    }

    // ── B3A graph profiling recorders (capture-aware, point 8) ──────────
    //
    // The structural counters (builds, submissions, failures, fallbacks) are
    // UNCONDITIONAL — they're the diagnostic surface (`graph_diag`), not the
    // profile decomposition, and they're cheap relaxed atomics. The node-count
    // + d2d decomposition counters gate on `gpu_profile_enabled()` so normal
    // decode pays nothing for the detailed breakdown.

    pub(crate) fn note_graph_build(&self) {
        self.graph_profile
            .graph_builds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn note_graph_submission(&self) {
        self.graph_profile
            .graph_submissions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn note_graph_captured_nodes(&self, nodes: u32) {
        if crate::options::gpu_profile_enabled() {
            self.graph_profile
                .captured_kernel_nodes
                .fetch_add(nodes as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[allow(dead_code)]
    pub(crate) fn note_graph_logical_exec(&self, nodes: u32) {
        if crate::options::gpu_profile_enabled() {
            self.graph_profile
                .logical_graph_kernel_executions
                .fetch_add(nodes as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub(crate) fn note_graph_failure(&self) {
        self.graph_profile
            .graph_failures
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn note_graph_fallback(&self) {
        self.graph_profile
            .graph_fallbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn note_graph_d2d(&self) {
        if crate::options::gpu_profile_enabled() {
            self.graph_profile
                .d2d_submissions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Record one explicit cross-stream `synchronize()` from the graph path
    /// (B3A review point 3). Always counted (not profile-gated): it is a real
    /// host CUDA submission the graph path introduces, and hiding it would
    /// make the sync accounting dishonest.
    /// Explicit cross-stream `synchronize()` calls the graph path issues.
    /// B3B made this always zero: the single non-NULL stream removed every
    /// runtime↔cap_stream handoff, so there are no cross-stream syncs to count.
    /// Kept (reporting 0) so the benchmark JSON's `graph_cross_stream_syncs`
    /// category is still measured independently (B3B section 7).
    #[allow(dead_code)]
    pub(crate) fn note_graph_cross_stream_sync(&self) {
        self.graph_profile
            .graph_cross_stream_syncs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// B3A graph diagnostic: one-liner for the graph replay counters, or `None`
    /// when no graph activity has occurred. Surfaced through `device_info()`
    /// when `LARQL_GPU_DIAG` is set.
    #[allow(dead_code)]
    pub fn graph_diag(&self) -> Option<String> {
        use std::sync::atomic::Ordering;
        let builds = self.graph_profile.graph_builds.load(Ordering::Relaxed);
        let submissions = self.graph_profile.graph_submissions.load(Ordering::Relaxed);
        let failures = self.graph_profile.graph_failures.load(Ordering::Relaxed);
        let fallbacks = self.graph_profile.graph_fallbacks.load(Ordering::Relaxed);
        if builds == 0 && submissions == 0 && failures == 0 && fallbacks == 0 {
            return None;
        }
        let warm = submissions.saturating_sub(builds);
        let replay_rate = if submissions == 0 {
            "n/a".to_string()
        } else {
            format!("{:.1}%", (warm as f64 / submissions as f64) * 100.0)
        };
        // Node + D2D + sync decomposition (review point 3 honest accounting).
        // Read non-destructively (load) so the cumulative diagnostic survives
        // across reads; `cuda_profile_counters` swaps them for window deltas.
        let captured_nodes = self
            .graph_profile
            .captured_kernel_nodes
            .load(Ordering::Relaxed);
        let logical_execs = self
            .graph_profile
            .logical_graph_kernel_executions
            .load(Ordering::Relaxed);
        let d2d = self.graph_profile.d2d_submissions.load(Ordering::Relaxed);
        let cross_syncs = self
            .graph_profile
            .graph_cross_stream_syncs
            .load(Ordering::Relaxed);
        Some(format!(
            "cuda resident-FFN graph: builds={builds}, submissions={submissions}, replays={warm}, failures={failures}, fallbacks={fallbacks}, replay_rate={replay_rate}, captured_nodes={captured_nodes}, logical_execs={logical_execs}, d2d={d2d}, cross_stream_syncs={cross_syncs}"
        ))
    }

    // ── LARQL-GPU-PROFILE-001 gated decomposition counters ───────────────
    //
    // The backend owns the two host-side counters that the runtime cannot see:
    // the host KV mirror append cost (pipeline.rs) and the final hidden-state
    // readback. The launch/copy/sync counters live on `CudaRuntime` (every
    // device op flows through its single stream). Each recorder is a no-op
    // when `LARQL_GPU_PROFILE` is unset, so normal decode is untouched.

    /// Record host KV mirror append cost: `nanos` spent reallocating + copying
    /// the `[len, kv_dim]` Array2 pair for one layer, and `rows_copied` prior
    /// rows copied (the O(seq_len) term). No-op unless `LARQL_GPU_PROFILE=1`.
    pub(crate) fn note_mirror_append(&self, nanos: u64, rows_copied: usize) {
        if crate::options::gpu_profile_enabled() {
            self.profile
                .mirror_append_ns
                .fetch_add(nanos, std::sync::atomic::Ordering::Relaxed);
            self.profile
                .mirror_rows_copied
                .fetch_add(rows_copied as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Record final hidden-state readback cost (`nanos`). No-op unless
    /// `LARQL_GPU_PROFILE=1`.
    pub(crate) fn note_hidden_readback(&self, nanos: u64) {
        if crate::options::gpu_profile_enabled() {
            self.profile
                .hidden_readback_ns
                .fetch_add(nanos, std::sync::atomic::Ordering::Relaxed);
        }
    }

    // ── LARQL-GPU-B4 device-greedy recorders (profile-gated) ────────────
    //
    // All gate on `gpu_profile_enabled()` so normal decode is a no-op branch.
    // The engagement counters (attempts/engaged/fallbacks/failures) are
    // UNCONDITIONAL when device-greedy is in use so the diagnostic surface
    // always reports engagement rate; only the byte/copy decomposition is
    // profile-gated.

    /// Record one device-greedy attempt (a call to the additive greedy
    /// method on an eligible plain-greedy step). Unconditional.
    pub(crate) fn note_device_greedy_attempt(&self) {
        self.greedy_profile
            .attempts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record one device-greedy engagement (device returned `DevicePick`).
    /// Unconditional.
    pub(crate) fn note_device_greedy_engaged(&self) {
        self.greedy_profile
            .engaged
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record one device-greedy fallback (device returned `HostHidden`
    /// after computing the hidden state). Unconditional.
    pub(crate) fn note_device_greedy_fallback(&self) {
        self.greedy_profile
            .fallbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record one device-greedy failure (device could not even read back the
    /// hidden state). Unconditional.
    pub(crate) fn note_device_greedy_failure(&self) {
        self.greedy_profile
            .failures
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a full lm-head score-vector DtoH copy on the host path (the
    /// transfer B4 eliminates). Profile-gated.
    pub(crate) fn note_lm_head_full_score_dtoh(&self, bytes: usize) {
        if crate::options::gpu_profile_enabled() {
            self.greedy_profile
                .lm_head_full_score_dtoh_copies
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.greedy_profile
                .lm_head_full_score_dtoh_bytes
                .fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Record a fixed-size candidate-result DtoH copy on the B4 path.
    /// Profile-gated.
    pub(crate) fn note_lm_head_result_dtoh(&self, bytes: usize) {
        if crate::options::gpu_profile_enabled() {
            self.greedy_profile
                .lm_head_result_dtoh_copies
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.greedy_profile
                .lm_head_result_dtoh_bytes
                .fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Record a final-hidden-state DtoH readback on the host path (the
    /// transfer B4 eliminates). Profile-gated.
    pub(crate) fn note_final_hidden_readback_bytes(&self, bytes: usize) {
        if crate::options::gpu_profile_enabled() {
            self.greedy_profile
                .final_hidden_readbacks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.greedy_profile
                .final_hidden_readback_bytes
                .fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// A multi-line diagnostic string summarising the profile counters, or
    /// `None` when nothing has been recorded (counters at zero). Surfaced
    /// through `device_info()` when `LARQL_GPU_DIAG` is set. Reads the
    /// runtime-side atomics directly (non-destructive); use
    /// [`take_profile_counters`] for a consuming snapshot.
    pub fn profile_diag(&self) -> Option<String> {
        use std::sync::atomic::Ordering;
        // Runtime-side (relaxed loads — diagnostic only, never on the hot path).
        let (launches, htod_copies, htod_bytes, dtoh_copies, dtoh_bytes, syncs) = self
            .runtime
            .as_ref()
            .map(|r| {
                (
                    r.profile.launches.load(Ordering::Relaxed),
                    r.profile.htod_copies.load(Ordering::Relaxed),
                    r.profile.htod_bytes.load(Ordering::Relaxed),
                    r.profile.dtoh_copies.load(Ordering::Relaxed),
                    r.profile.dtoh_bytes.load(Ordering::Relaxed),
                    r.profile.syncs.load(Ordering::Relaxed),
                )
            })
            .unwrap_or((0, 0, 0, 0, 0, 0));
        let mirror_ns = self.profile.mirror_append_ns.load(Ordering::Relaxed);
        let mirror_rows = self.profile.mirror_rows_copied.load(Ordering::Relaxed);
        let readback_ns = self.profile.hidden_readback_ns.load(Ordering::Relaxed);
        if launches == 0 && mirror_ns == 0 && readback_ns == 0 {
            return None;
        }
        Some(format!(
            "cuda profile: launches={launches}, htod={htod_copies}/{htod_bytes}B, \
             dtoh={dtoh_copies}/{dtoh_bytes}B, syncs={syncs}, \
             mirror_append={:.3}ms/{}rows, hidden_readback={:.3}ms",
            mirror_ns as f64 / 1e6,
            mirror_rows,
            readback_ns as f64 / 1e6,
        ))
    }

    /// Consume and reset all profile counters, returning the snapshot. Used
    /// by the bench harness to read per-run deltas. Returns `None` when no
    /// profiling has been enabled (counters at zero). Modeled on Metal's
    /// `take_split_timings`. Merges the runtime-side launch/copy/sync
    /// counters with the backend-side mirror/readback counters.
    pub fn cuda_profile_counters(&self) -> Option<CudaProfileSnapshot> {
        use std::sync::atomic::Ordering;
        // Runtime-side (launches/copies/syncs). May be None when the backend
        // has no runtime (CPU-delegate scaffold) or when profiling is off.
        let rt = self
            .runtime
            .as_ref()
            .and_then(|r| r.take_profile_counters());
        // Backend-side (mirror/readback). These are zero unless pipeline.rs
        // recorded them during a decode_token call.
        let mirror_append_ns = self.profile.mirror_append_ns.swap(0, Ordering::Relaxed);
        let mirror_rows_copied = self.profile.mirror_rows_copied.swap(0, Ordering::Relaxed);
        let hidden_readback_ns = self.profile.hidden_readback_ns.swap(0, Ordering::Relaxed);
        // B3A graph counters — consumed (swap) so consecutive reads give
        // per-window deltas, matching the mirror/readback swap above.
        let graph_submissions = self
            .graph_profile
            .graph_submissions
            .swap(0, Ordering::Relaxed);
        let graph_builds = self.graph_profile.graph_builds.swap(0, Ordering::Relaxed);
        let captured_kernel_nodes = self
            .graph_profile
            .captured_kernel_nodes
            .swap(0, Ordering::Relaxed);
        let logical_graph_kernel_executions = self
            .graph_profile
            .logical_graph_kernel_executions
            .swap(0, Ordering::Relaxed);
        let graph_failures = self.graph_profile.graph_failures.swap(0, Ordering::Relaxed);
        let graph_fallbacks = self
            .graph_profile
            .graph_fallbacks
            .swap(0, Ordering::Relaxed);
        let d2d_submissions = self
            .graph_profile
            .d2d_submissions
            .swap(0, Ordering::Relaxed);
        let graph_cross_stream_syncs = self
            .graph_profile
            .graph_cross_stream_syncs
            .swap(0, Ordering::Relaxed);
        // ── LARQL-GPU-B4 device-greedy counters (consumed + reset) ──
        let device_greedy_attempts = self.greedy_profile.attempts.swap(0, Ordering::Relaxed);
        let device_greedy_engaged = self.greedy_profile.engaged.swap(0, Ordering::Relaxed);
        let device_greedy_fallbacks = self.greedy_profile.fallbacks.swap(0, Ordering::Relaxed);
        let device_greedy_failures = self.greedy_profile.failures.swap(0, Ordering::Relaxed);
        let lm_head_full_score_dtoh_copies = self
            .greedy_profile
            .lm_head_full_score_dtoh_copies
            .swap(0, Ordering::Relaxed);
        let lm_head_full_score_dtoh_bytes = self
            .greedy_profile
            .lm_head_full_score_dtoh_bytes
            .swap(0, Ordering::Relaxed);
        let lm_head_result_dtoh_copies = self
            .greedy_profile
            .lm_head_result_dtoh_copies
            .swap(0, Ordering::Relaxed);
        let lm_head_result_dtoh_bytes = self
            .greedy_profile
            .lm_head_result_dtoh_bytes
            .swap(0, Ordering::Relaxed);
        let final_hidden_readbacks = self
            .greedy_profile
            .final_hidden_readbacks
            .swap(0, Ordering::Relaxed);
        let final_hidden_readback_bytes = self
            .greedy_profile
            .final_hidden_readback_bytes
            .swap(0, Ordering::Relaxed);
        let greedy_active = device_greedy_attempts != 0
            || device_greedy_engaged != 0
            || device_greedy_fallbacks != 0
            || device_greedy_failures != 0
            || lm_head_full_score_dtoh_copies != 0
            || final_hidden_readbacks != 0;
        let graph_active = graph_submissions != 0
            || graph_builds != 0
            || graph_failures != 0
            || graph_fallbacks != 0;
        if rt.is_none()
            && mirror_append_ns == 0
            && hidden_readback_ns == 0
            && !graph_active
            && !greedy_active
        {
            return None;
        }
        let rt = rt.unwrap_or_default();
        Some(CudaProfileSnapshot {
            launches: rt.launches,
            htod_copies: rt.htod_copies,
            htod_bytes: rt.htod_bytes,
            dtoh_copies: rt.dtoh_copies,
            dtoh_bytes: rt.dtoh_bytes,
            syncs: rt.syncs,
            mirror_append_ns,
            mirror_rows_copied,
            hidden_readback_ns,
            graph_submissions,
            graph_builds,
            captured_kernel_nodes,
            logical_graph_kernel_executions,
            graph_failures,
            graph_fallbacks,
            d2d_submissions,
            graph_cross_stream_syncs,
            device_greedy_attempts,
            device_greedy_engaged,
            device_greedy_fallbacks,
            device_greedy_failures,
            lm_head_full_score_dtoh_copies,
            lm_head_full_score_dtoh_bytes,
            lm_head_result_dtoh_copies,
            lm_head_result_dtoh_bytes,
            final_hidden_readbacks,
            final_hidden_readback_bytes,
        })
    }

    /// Native RMSNorm (body norm) — the device twin of
    /// `larql_compute::residual::rms_norm_eps`. Computes `out` in place into
    /// the caller's buffer. Returns `Ok(true)` on a native launch, `Ok(false)`
    /// when there's no runtime (caller falls back to the host reference), or
    /// `Err` on a launch failure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn native_rms_norm(
        &self,
        x: &[f32],
        weight: Option<&[f32]>,
        out: &mut [f32],
        rows: usize,
        cols: usize,
        eps: f64,
        offset: f32,
    ) -> Result<bool, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => {
                runtime.launch_rms_norm(x, weight, out, rows, cols, eps, offset)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Native per-head RMSNorm — the device twin of
    /// `larql_compute::residual::rms_norm_heads` / `rms_norm_heads_no_weight`.
    /// `weight = None` selects the parameter-free path. Returns `Ok(true)` on
    /// a native launch, `Ok(false)` when there's no runtime, or `Err` on a
    /// launch failure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn native_rms_norm_heads(
        &self,
        x: &[f32],
        weight: Option<&[f32]>,
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f64,
        offset: f32,
    ) -> Result<bool, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => {
                runtime.launch_rms_norm_heads(
                    x, weight, out, seq_len, num_heads, head_dim, eps, offset,
                )?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Native GEGLU-SiLU: `out[i] = silu(gate[i]) * up[i]`. Returns `Ok(true)`
    /// on a native launch, `Ok(false)` when there's no runtime (caller falls
    /// back to the host reference), or `Err` on a launch failure.
    pub(crate) fn native_geglu_silu(
        &self,
        gate: &[f32],
        up: &[f32],
        out: &mut [f32],
    ) -> Result<bool, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => {
                runtime.launch_geglu_silu(gate, up, out, gate.len())?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Native GEGLU-GELU-tanh: `out[i] = gelu_tanh(gate[i]) * up[i]`.
    /// Returns `Ok(true)` on a native launch, `Ok(false)` when there's no
    /// runtime, or `Err` on a launch failure.
    pub(crate) fn native_geglu_gelu_tanh(
        &self,
        gate: &[f32],
        up: &[f32],
        out: &mut [f32],
    ) -> Result<bool, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => {
                runtime.launch_geglu_gelu_tanh(gate, up, out, gate.len())?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Native standard SiLU: `out[i] = silu(x[i])`. Returns `Ok(true)` on a
    /// native launch, `Ok(false)` when there's no runtime, or `Err` on a
    /// launch failure.
    pub(crate) fn native_activation_silu(
        &self,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<bool, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => {
                runtime.launch_activation_silu(input, out, input.len())?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Native standard GELU-tanh: `out[i] = gelu_tanh(x[i])`. Returns
    /// `Ok(true)` on a native launch, `Ok(false)` when there's no runtime,
    /// or `Err` on a launch failure.
    pub(crate) fn native_activation_gelu_tanh(
        &self,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<bool, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => {
                runtime.launch_activation_gelu_tanh(input, out, input.len())?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Native scaled residual add: `out[i] = h[i] + b_scale * x[i]`. Returns
    /// `Ok(true)` on a native launch, `Ok(false)` when there's no runtime
    /// (caller falls back to the host reference), or `Err` on a launch
    /// failure. `h`, `x`, `out` must each be `n` elements.
    pub(crate) fn native_residual_add(
        &self,
        h: &[f32],
        x: &[f32],
        out: &mut [f32],
        b_scale: f32,
        n: usize,
    ) -> Result<bool, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => {
                runtime.launch_residual_add(h, x, out, b_scale, n)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Native RoPE — the device twin of
    /// `larql_compute::attention::rope::apply_rope_partial_at_full`. `inv_freq`
    /// is the caller-precomputed `half_rotary`-length frequency array (built
    /// identically to the reference, including the `llama3` variant). Returns
    /// `Ok(true)` on a native launch, `Ok(false)` when there's no runtime
    /// (caller falls back to the host reference), or `Err` on a launch failure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn native_rope(
        &self,
        x: &[f32],
        inv_freq: &[f64],
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        half_rotary: usize,
        position_offset: usize,
        position_divisor: f64,
    ) -> Result<bool, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => {
                runtime.launch_rope(
                    x,
                    inv_freq,
                    out,
                    seq_len,
                    num_heads,
                    head_dim,
                    half_rotary,
                    position_offset,
                    position_divisor,
                )?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Native fused decode-step GQA attention — the device twin of
    /// `gqa_attention_decode_step`. `q` is `[num_q * head_dim]` (the new
    /// token's RoPE'd Q); `k_cache`/`v_cache` are `[total_len * kv_dim]`.
    /// On success `out` is filled with `[num_q * head_dim]` and `Ok(true)` is
    /// returned; `Ok(false)` when there's no runtime (caller falls back to the
    /// host reference); `Err` on a launch failure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn native_decode_attention(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        out: &mut Vec<f32>,
        scale: f32,
        softcap: Option<f32>,
        num_q: usize,
        head_dim: usize,
        kv_dim: usize,
        reps: usize,
        total_len: usize,
    ) -> Result<bool, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => {
                runtime.launch_decode_attention(
                    q, k_cache, v_cache, out, scale, softcap, num_q, head_dim, kv_dim, reps,
                    total_len,
                )?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Native fused prefill (seq×seq) causal GQA attention — the device twin
    /// of `gqa_attention_with_weights` (the symmetric `gqa_attention_capture`
    /// path). `q` is `[seq, num_q*head_dim]`; `k`/`v` are `[seq, kv_dim]`. On
    /// success `out` is filled with `[seq, num_q*head_dim]` (row-major) and
    /// `Ok(true)` is returned; `Ok(false)` when there's no runtime (caller
    /// falls back to the host reference); `Err` on a launch failure or when
    /// the shape exceeds the device shared-mem/index budget (caller falls
    /// back to the host reference).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn native_prefill_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut Vec<f32>,
        scale: f32,
        softcap: Option<f32>,
        num_q: usize,
        head_dim: usize,
        kv_dim: usize,
        reps: usize,
        seq_len: usize,
    ) -> Result<bool, RuntimeError> {
        match self.runtime.as_ref() {
            Some(runtime) => {
                runtime.launch_prefill_attention(
                    q, k, v, out, scale, softcap, num_q, head_dim, kv_dim, reps, seq_len,
                )?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// inner value (a poisoned mutex only means a prior holder panicked; the
    /// cache itself is still usable). This keeps the documented "fall back
    /// to the host store on failure" contract instead of aborting the
    /// process on a poisoned lock.
    fn lock_kv_cache(&self) -> std::sync::MutexGuard<'_, Option<CudaKVCache>> {
        self.kv_cache.lock().unwrap_or_else(|poisoned| {
            // Recover: a poisoned lock indicates a panic in a prior critical
            // section, not corrupt cache data. Take the guard so subsequent
            // ops keep working (and `has_kv_cache` can still report state).
            poisoned.into_inner()
        })
    }

    /// True when a device KV cache has been allocated. Used by
    /// `DecodeBackend::has_kv_cache`.
    pub(crate) fn kv_cache_allocated(&self) -> bool {
        self.lock_kv_cache().is_some()
    }

    /// Pre-allocate (or grow) the device KV cache to the per-layer shapes.
    /// Returns an error if allocation fails (the caller falls back to the
    /// CPU-reference store). When the cache already covers `shapes` with
    /// matching geometry *and* `max_seq` this is a no-op; when it has fewer
    /// layers it grows; when an existing layer's geometry disagrees or is
    /// undersized for `max_seq` it is reallocated from scratch (a shape /
    /// size mismatch implies a different model/prompt context).
    pub(crate) fn preallocate_kv_cache(
        &self,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> Result<(), KvCacheError> {
        let runtime = match self.runtime.as_ref() {
            Some(rt) => rt,
            None => return Err(KvCacheError::Alloc("no CUDA runtime".to_string())),
        };
        let stream = runtime.stream();
        let mut guard = self.lock_kv_cache();
        match guard.as_mut() {
            // Existing cache covers the shapes with matching geometry AND
            // every layer's `max_seq` is >= the request → no-op.
            Some(cache)
                if !cache.has_shape_mismatch(shapes)
                    && cache.layers.len() >= shapes.len()
                    && cache.layers.iter().all(|l| l.max_seq >= max_seq) =>
            {
                Ok(())
            }
            // Matching geometry, enough layers, but some layer is undersized
            // for `max_seq` → reallocate from scratch (grow_to_shapes only
            // appends; it can't resize existing layers).
            Some(cache)
                if !cache.has_shape_mismatch(shapes) && cache.layers.len() >= shapes.len() =>
            {
                let new_cache = CudaKVCache::new_per_layer(stream, shapes, max_seq)?;
                *cache = new_cache;
                Ok(())
            }
            // Matching geometry but fewer layers → grow (new layers get the
            // requested `max_seq`).
            Some(cache) if !cache.has_shape_mismatch(shapes) => {
                cache.grow_to_shapes(stream, shapes, max_seq)
            }
            // Shape mismatch or no cache → allocate fresh.
            _ => {
                let new_cache = CudaKVCache::new_per_layer(stream, shapes, max_seq)?;
                *guard = Some(new_cache);
                Ok(())
            }
        }
    }

    /// Native KV append: write a contiguous block of `seq_len`
    /// freshly-projected K/V rows for `layer` into the device cache starting
    /// at slot `pos`. One host→device upload + one kernel launch + one sync
    /// per call (no per-row stalls). Returns `Ok(true)` on a native launch,
    /// `Ok(false)` when there's no runtime / no cache / the layer is out of
    /// range (caller falls back to the host reference store), or `Err` on a
    /// launch failure.
    pub(crate) fn native_kv_append(
        &self,
        layer: usize,
        new_k: &[f32],
        new_v: &[f32],
        pos: usize,
        seq_len: usize,
    ) -> Result<bool, RuntimeError> {
        let runtime = match self.runtime.as_ref() {
            Some(rt) => rt,
            None => return Ok(false),
        };
        let mut guard = self.lock_kv_cache();
        let cache = match guard.as_mut() {
            Some(c) => c,
            None => return Ok(false),
        };
        let layer_cache = match cache.layers.get_mut(layer) {
            Some(l) => l,
            None => return Ok(false),
        };
        runtime.launch_kv_append(
            new_k,
            new_v,
            &mut layer_cache.k_cache,
            &mut layer_cache.v_cache,
            pos,
            seq_len,
            layer_cache.num_kv_heads,
            layer_cache.head_dim,
        )?;
        // Lockstep: every layer advances its cursor to `pos + seq_len`.
        layer_cache.current_len = pos + seq_len;
        Ok(true)
    }

    /// Resident-KV decode attention (GPU-006): append the current token's
    /// post-RoPE K row + post-V-norm V row to layer `li`'s device cache, then
    /// launch the `decode_attention` kernel over the device-resident K/V —
    /// eliminating the per-token full-KV host readback + re-upload the full-
    /// upload path pays. The host KV mirror stays the parity oracle and the
    /// source for truncate/state-dump; this path keeps it in lockstep by
    /// leaving the host append to the caller (which needs the rows anyway for
    /// the state dump).
    ///
    /// Returns `Ok(Some(out_dev))` (the resident `[num_q * head_dim]` attention
    /// output, no sync/dtoh — the O projection consumes it on the same stream)
    /// on a successful append + launch; `Ok(None)` when the resident path is
    /// ineligible (no runtime, no cache, layer out of range, shape mismatch, or
    /// the device cursor disagrees with `prev` — the lockstep invariant the
    /// append relies on); `Err` on a launch failure. The cache lock spans the
    /// append + attention launch; it is released once the attention kernel is
    /// *launched* (CUDA stream-ordered, so the kernel reads the resident buffer
    /// before any later append on the same stream writes past `total_len`).
    ///
    /// `prev` is the prior host-mirror length (== the device cursor when in
    /// lockstep); `total_len = prev + 1` is the post-append attention length.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resident_kv_decode_attention(
        &self,
        runtime: &Arc<CudaRuntime>,
        li: usize,
        q_dev: &CudaSlice<f32>,
        new_k: &[f32],
        new_v: &[f32],
        prev: usize,
        scale: f32,
        softcap: Option<f32>,
        num_q: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        reps: usize,
    ) -> Result<Option<CudaSlice<f32>>, RuntimeError> {
        let kv_dim_ok = num_kv_heads.checked_mul(head_dim) == Some(kv_dim) && kv_dim > 0;
        let mut guard = self.lock_kv_cache();
        let cache = match guard.as_mut() {
            Some(c) => c,
            None => return Ok(None),
        };
        let layer_cache = match cache.layers.get_mut(li) {
            Some(l) => l,
            None => return Ok(None),
        };
        // Shape + lockstep guards. The layer's per-layer geometry must match
        // the caller's (num_kv_heads, head_dim), and the device cursor must
        // equal `prev` so the append lands at exactly the slot the host path
        // would have concatenated the new row into.
        if !kv_dim_ok
            || layer_cache.num_kv_heads != num_kv_heads
            || layer_cache.head_dim != head_dim
        {
            return Ok(None);
        }
        if layer_cache.current_len != prev {
            return Ok(None);
        }
        // Capacity: the appended row at slot `prev` must fit `max_seq`.
        if prev >= layer_cache.max_seq {
            return Ok(None);
        }
        // Append the new K/V row at slot `prev` (cursor → prev+1). Uses the
        // same `launch_kv_append` prefill uses; one htod + one launch + sync.
        runtime.launch_kv_append(
            new_k,
            new_v,
            &mut layer_cache.k_cache,
            &mut layer_cache.v_cache,
            prev,
            1,
            layer_cache.num_kv_heads,
            layer_cache.head_dim,
        )?;
        layer_cache.current_len = prev + 1;
        let total_len = layer_cache.current_len;
        // Borrow the resident K/V for the attention launch (same lock scope).
        let k_ref = &layer_cache.k_cache;
        let v_ref = &layer_cache.v_cache;
        let score_len = match num_q.checked_mul(total_len) {
            Some(s) => s,
            None => return Ok(None),
        };
        let mut scores_dev = runtime.alloc_zeros_f32(score_len).map_err(|err| {
            RuntimeError::usage(format!(
                "allocating resident decode_attention scores: {err}"
            ))
        })?;
        let out_dev = runtime.launch_decode_attention_resident_dev(
            q_dev,
            k_ref,
            v_ref,
            &mut scores_dev,
            scale,
            softcap,
            num_q,
            head_dim,
            kv_dim,
            reps,
            total_len,
        )?;
        // The attention kernel is launched on the stream; the output `out_dev`
        // is an owned allocation. The cache lock can drop now (stream ordering
        // guarantees the kernel reads the resident K/V before a later append).
        Ok(Some(out_dev))
    }

    /// Reset every layer's `current_len` cursor to 0 (new prompt). The device
    /// buffers are not zeroed — subsequent appends overwrite slots in order.
    /// Also flushes the persistent weight cache: `reset_kv_cache` is the
    /// "start of a new prompt/generation" boundary for decode/prefill, so
    /// dropping the cached weight buffers here bounds staleness to one
    /// generation on that path (the cache is repopulated during the new
    /// generation's prefill projections and reused across every decode token).
    ///
    /// This does **not** flush on the browse path (gate/lm-head KNN never
    /// resets the KV cache) — see [`CudaBackend::flush_weight_cache`] for
    /// cross-vindex reuse.
    pub(crate) fn reset_kv_cache_native(&self) {
        // B3A-6: destroy executable graphs + release scratch + retained weights
        // BEFORE flushing the weight cache (so graph teardown references valid
        // buffers), then advance the generation (the primary ABA defense).
        if let Ok(mut cache) = self.graph_cache.lock() {
            cache.reset();
        }
        // The arena (ping-pong buffers + capture stream) is generation-scoped;
        // drop it so the next generation allocates fresh stable-address buffers.
        if let Ok(mut arena) = self.arena.lock() {
            arena.take();
        }
        // LARQL-GPU-B4: the greedy-head workspace is generation-scoped too;
        // drop it so the next generation (possibly a different lm-head shape)
        // rebuilds it. The weight-cache flush below also drops the cached
        // logical-row lm-head prefix, so a stale buffer can't be served.
        if let Ok(mut ws) = self.greedy_workspace.lock() {
            ws.take();
        }
        if let Some(runtime) = self.runtime.as_ref() {
            runtime.flush_weight_cache();
        }
        if let Ok(mut guard) = self.kv_cache.lock() {
            if let Some(cache) = guard.as_mut() {
                cache.clear();
            }
        }
    }

    /// Drop every cached device weight buffer immediately. The cache is
    /// otherwise resident for the backend's lifetime and flushed
    /// per-generation by [`reset_kv_cache_native`] on the decode/prefill path.
    ///
    /// Call this when a backend is **reused across vindex loads**: weight-cache
    /// keying is pointer identity (`(ptr, len)`), and a recycled mmap address
    /// could otherwise make a later vindex read an earlier model's cached
    /// weights. The normal case (one backend per vindex, or decode/prefill
    /// which resets per generation) needs no explicit call. The gap this closes
    /// is the **browse** path — gate/lm-head KNN (`f16_gemv`/`f32_gemv` via
    /// `larql-vindex` DESCRIBE/WALK/SELECT) never resets the KV cache, so a
    /// backend handed a second vindex for browse must flush at the rebind
    /// boundary.
    ///
    /// Reachable through `ComputeBackend::flush_weight_cache` (the trait's
    /// default no-op dispatches here for CUDA; CPU/Metal/Vulkan scaffold are
    /// no-op), so a long-lived browse path can flush via `&dyn ComputeBackend`
    /// without downcasting. As of this writing no code path reuses a backend
    /// across vindex loads, so this is a forward-looking escape hatch.
    pub fn flush_weight_cache(&self) {
        if let Some(runtime) = self.runtime.as_ref() {
            runtime.flush_weight_cache();
        }
    }

    /// Current committed length of the KV cache (reads the first layer's
    /// cursor; all layers progress in lockstep). 0 when no cache is
    /// allocated.
    pub(crate) fn kv_cache_len_native(&self) -> usize {
        self.lock_kv_cache()
            .as_ref()
            .map(|c| c.current_len())
            .unwrap_or(0)
    }

    /// Roll back the cursor to `len` on every layer, unconditionally (mirrors
    /// Metal's `truncate_kv_cache`), restoring the lockstep invariant even
    /// when cursors had diverged (e.g. after a partial populate failure). The
    /// physical K/V data below `len` is preserved.
    pub(crate) fn truncate_kv_cache_native(&self, len: usize) {
        if let Ok(mut guard) = self.kv_cache.lock() {
            if let Some(cache) = guard.as_mut() {
                for layer in &mut cache.layers {
                    layer.current_len = len;
                }
            }
        }
    }

    pub(crate) fn runtime_summary(&self) -> &str {
        match (&self.runtime, &self.runtime_status) {
            (Some(runtime), _) => runtime.summary(),
            (None, Some(status)) => status.as_str(),
            (None, None) => "CUDA runtime unavailable; using CPU delegate scaffold",
        }
    }
}

#[derive(Debug, Clone)]
pub enum BackendInitError {
    Unavailable(String),
}

impl std::fmt::Display for BackendInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for BackendInitError {}

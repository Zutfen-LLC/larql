mod runtime;

use crate::kv_cache::{CudaKVCache, KvCacheError};
use crate::options::BackendOptions;
use crate::pipeline::HostKvType;
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
}

impl CudaBackend {
    pub fn new() -> Result<Self, BackendInitError> {
        Self::with_options(BackendOptions::default())
    }

    pub fn with_options(options: BackendOptions) -> Result<Self, BackendInitError> {
        match CudaRuntime::initialize(options.device_ordinal) {
            Ok(runtime) => Ok(Self {
                options,
                runtime: Some(Arc::new(runtime)),
                runtime_status: None,
                kv_cache: Mutex::new(None),
                host_kv: Mutex::new(Vec::new()),
            }),
            Err(err) if options.allow_cpu_delegate => Ok(Self {
                options,
                runtime: None,
                runtime_status: Some(err.to_string()),
                kv_cache: Mutex::new(None),
                host_kv: Mutex::new(Vec::new()),
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

    /// Lock the KV-cache mutex, recovering from poisoning by taking the
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

    /// Reset every layer's `current_len` cursor to 0 (new prompt). The device
    /// buffers are not zeroed — subsequent appends overwrite slots in order.
    pub(crate) fn reset_kv_cache_native(&self) {
        if let Ok(mut guard) = self.kv_cache.lock() {
            if let Some(cache) = guard.as_mut() {
                cache.clear();
            }
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

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileError, CompileOptions, Ptx};

use crate::ops::{
    ACTIVATION_GELU_TANH_CUDA_SRC, ACTIVATION_GELU_TANH_KERNEL, ACTIVATION_SILU_CUDA_SRC,
    ACTIVATION_SILU_KERNEL, DECODE_ATTENTION_CUDA_SRC, DECODE_ATTENTION_KERNEL, F16_GEMV_CUDA_SRC,
    F16_GEMV_KERNEL, F32_GEMV_CUDA_SRC, F32_GEMV_KERNEL, GEGGLU_GELU_TANH_CUDA_SRC,
    GEGGLU_GELU_TANH_KERNEL, GEGGLU_SILU_CUDA_SRC, GEGGLU_SILU_KERNEL, GREEDY_TOPK_FINAL_CUDA_SRC,
    GREEDY_TOPK_FINAL_KERNEL, GREEDY_TOPK_PARTIAL_CUDA_SRC, GREEDY_TOPK_PARTIAL_KERNEL,
    KV_APPEND_CUDA_SRC, KV_APPEND_KERNEL, PREFILL_ATTENTION_CUDA_SRC, PREFILL_ATTENTION_KERNEL,
    Q4K_DUAL_MATVEC_CUDA_SRC, Q4K_DUAL_MATVEC_KERNEL, Q4K_MATMUL_CUDA_SRC, Q4K_MATMUL_KERNEL,
    Q4K_MATVEC_CUDA_SRC, Q4K_MATVEC_KERNEL, Q4_MATVEC_CUDA_SRC, Q4_MATVEC_KERNEL,
    Q4_VECMAT_CUDA_SRC, Q4_VECMAT_KERNEL, Q6K_MATMUL_CUDA_SRC, Q6K_MATMUL_KERNEL,
    Q6K_MATVEC_CUDA_SRC, Q6K_MATVEC_KERNEL, RESIDUAL_ADD_CUDA_SRC, RESIDUAL_ADD_KERNEL,
    RMS_NORM_CUDA_SRC, RMS_NORM_HEADS_CUDA_SRC, RMS_NORM_HEADS_KERNEL, RMS_NORM_KERNEL,
    ROPE_CUDA_SRC, ROPE_KERNEL,
};
use crate::ptx_cache;
use crate::weight_cache::{CacheStats, WeightCache};

#[derive(Debug)]
pub(crate) struct CudaRuntime {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    q4k_matvec: CudaFunction,
    q6k_matvec: CudaFunction,
    q4k_matmul: CudaFunction,
    /// Loaded but unused until the prefill-kquant slice routes the amortised
    /// Q6_K matmul through the backend (see `native_q6k_matmul`).
    #[allow(dead_code)]
    q6k_matmul: CudaFunction,
    q4k_dual_matvec: CudaFunction,
    f32_gemv: CudaFunction,
    f16_gemv: CudaFunction,
    q4_matvec: CudaFunction,
    q4_vecmat: CudaFunction,
    kv_append: CudaFunction,
    rms_norm: CudaFunction,
    rms_norm_heads: CudaFunction,
    geglu_silu: CudaFunction,
    geglu_gelu_tanh: CudaFunction,
    activation_silu: CudaFunction,
    activation_gelu_tanh: CudaFunction,
    residual_add: CudaFunction,
    rope: CudaFunction,
    decode_attention: CudaFunction,
    prefill_attention: CudaFunction,
    /// LARQL-GPU-B4 greedy top-K reduction kernels.
    greedy_topk_partial: CudaFunction,
    greedy_topk_final: CudaFunction,
    /// Persistent device-resident weight cache (see `weight_cache.rs`).
    /// Uploads each immutable weight matrix once and reuses the device buffer
    /// across calls — the first slice of the per-projection htod round-trip
    /// collapse. Activations stay on the fresh `clone_htod` path.
    weight_cache: WeightCache,
    summary: String,
    /// LARQL-GPU-PROFILE-001 launch/copy/sync counters. All relaxed atomics;
    /// only mutated when `LARQL_GPU_PROFILE=1` (the `note_*` helpers gate on
    /// `gpu_profile_enabled()`).
    pub(crate) profile: super::RuntimeProfile,
    /// Stream-capture depth (B3A review point 8). When > 0 the runtime is
    /// mid-`begin_capture`/`end_capture`, so `note_launch`/`note_htod`/
    /// `note_dtoh`/`note_sync` are **suppressed** — captured kernel launches
    /// become graph NODES (counted once at build via `note_graph_captured_nodes`
    /// on `CudaBackend`), not physical direct submissions. Single-threaded
    /// decode guarantees at most one capture is in flight, so a single depth
    /// is sound. Incremented by [`Self::enter_capture`], decremented by
    /// [`Self::exit_capture`]; the build path wraps the capture window in a
    /// guard so every return path balances.
    capture_depth: std::sync::atomic::AtomicUsize,
}

impl CudaRuntime {
    pub(crate) fn initialize(ordinal: usize) -> Result<Self, RuntimeError> {
        match std::panic::catch_unwind(|| Self::initialize_impl(ordinal)) {
            Ok(result) => result,
            Err(payload) => Err(RuntimeError::usage(format!(
                "probing CUDA runtime panicked: {}",
                panic_payload_to_string(payload)
            ))),
        }
    }

    fn initialize_impl(ordinal: usize) -> Result<Self, RuntimeError> {
        let context = CudaContext::new(ordinal)
            .map_err(|err| RuntimeError::context("initializing CUDA context", err))?;
        let device_name = context
            .name()
            .map_err(|err| RuntimeError::context("querying CUDA device name", err))?;
        let (cc_major, cc_minor) = context
            .compute_capability()
            .map_err(|err| RuntimeError::context("querying CUDA compute capability", err))?;

        // Concatenate the kernel sources into one NVRTC translation unit so
        // a single module load exposes all entry points (each kernel is
        // `extern "C"` with a distinct name).
        let combined_src = format!(
            "{Q4K_MATVEC_CUDA_SRC}\n{Q6K_MATVEC_CUDA_SRC}\n{Q4K_MATMUL_CUDA_SRC}\n{Q6K_MATMUL_CUDA_SRC}\n{Q4K_DUAL_MATVEC_CUDA_SRC}\n{F32_GEMV_CUDA_SRC}\n{F16_GEMV_CUDA_SRC}\n{Q4_MATVEC_CUDA_SRC}\n{Q4_VECMAT_CUDA_SRC}\n{KV_APPEND_CUDA_SRC}\n{RMS_NORM_CUDA_SRC}\n{RMS_NORM_HEADS_CUDA_SRC}\n{GEGGLU_SILU_CUDA_SRC}\n{GEGGLU_GELU_TANH_CUDA_SRC}\n{ACTIVATION_SILU_CUDA_SRC}\n{ACTIVATION_GELU_TANH_CUDA_SRC}\n{RESIDUAL_ADD_CUDA_SRC}\n{ROPE_CUDA_SRC}\n{DECODE_ATTENTION_CUDA_SRC}\n{PREFILL_ATTENTION_CUDA_SRC}\n{GREEDY_TOPK_PARTIAL_CUDA_SRC}\n{GREEDY_TOPK_FINAL_CUDA_SRC}"
        );
        // Target the device's real compute capability (e.g. `compute_89`)
        // instead of NVRTC's default virtual arch — better SASS once the
        // driver JITs, and "kernel uses features your GPU lacks" surfaces at
        // compile time rather than launch time. `CompileOptions::arch` is
        // `Option<&'static str>`; the device arch is only known at runtime, so
        // leak the small string once at backend init (a backend is constructed
        // rarely and the string is a handful of bytes).
        let fmad = false;
        let arch: &'static str =
            Box::leak(format!("compute_{cc_major}{cc_minor}").into_boxed_str());
        let module = compile_or_load_module(&context, &combined_src, arch, fmad)?;
        let q4k_matvec = module
            .load_function(Q4K_MATVEC_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading q4k_matvec CUDA function", err))?;
        let q6k_matvec = module
            .load_function(Q6K_MATVEC_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading q6k_matvec CUDA function", err))?;
        let q4k_matmul = module
            .load_function(Q4K_MATMUL_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading q4k_matmul CUDA function", err))?;
        let q6k_matmul = module
            .load_function(Q6K_MATMUL_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading q6k_matmul CUDA function", err))?;
        let q4k_dual_matvec = module
            .load_function(Q4K_DUAL_MATVEC_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading q4k_dual_matvec CUDA function", err))?;
        let f32_gemv = module
            .load_function(F32_GEMV_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading f32_gemv CUDA function", err))?;
        let f16_gemv = module
            .load_function(F16_GEMV_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading f16_gemv CUDA function", err))?;
        let q4_matvec = module
            .load_function(Q4_MATVEC_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading q4_matvec CUDA function", err))?;
        let q4_vecmat = module
            .load_function(Q4_VECMAT_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading q4_vecmat CUDA function", err))?;
        let kv_append = module
            .load_function(KV_APPEND_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading kv_append CUDA function", err))?;
        let rms_norm = module
            .load_function(RMS_NORM_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading rms_norm CUDA function", err))?;
        let rms_norm_heads = module
            .load_function(RMS_NORM_HEADS_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading rms_norm_heads CUDA function", err))?;
        let geglu_silu = module
            .load_function(GEGGLU_SILU_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading geglu_silu CUDA function", err))?;
        let geglu_gelu_tanh = module
            .load_function(GEGGLU_GELU_TANH_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading geglu_gelu_tanh CUDA function", err))?;
        let activation_silu = module
            .load_function(ACTIVATION_SILU_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading activation_silu CUDA function", err))?;
        let activation_gelu_tanh = module
            .load_function(ACTIVATION_GELU_TANH_KERNEL.identifier)
            .map_err(|err| {
                RuntimeError::context("loading activation_gelu_tanh CUDA function", err)
            })?;
        let residual_add = module
            .load_function(RESIDUAL_ADD_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading residual_add CUDA function", err))?;
        let rope = module
            .load_function(ROPE_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading rope CUDA function", err))?;
        let decode_attention = module
            .load_function(DECODE_ATTENTION_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading decode_attention CUDA function", err))?;
        let prefill_attention = module
            .load_function(PREFILL_ATTENTION_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading prefill_attention CUDA function", err))?;
        let greedy_topk_partial = module
            .load_function(GREEDY_TOPK_PARTIAL_KERNEL.identifier)
            .map_err(|err| {
                RuntimeError::context("loading greedy_topk_partial CUDA function", err)
            })?;
        let greedy_topk_final = module
            .load_function(GREEDY_TOPK_FINAL_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading greedy_topk_final CUDA function", err))?;
        // LARQL-GPU-B3B: the canonical runtime stream is a dedicated NON-NULL
        // stream. The NULL/default stream (`default_stream()` returns
        // `cu_stream = null_mut`) cannot be captured
        // (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`), which forced B3A onto a
        // *separate* capture stream and the per-layer cross-stream handoff it
        // requires. A single non-NULL stream lets graph capture AND replay run
        // on the same stream as every attention/KV/residual/kernel, so layer
        // ordering is by stream submission alone — zero per-layer D2D and zero
        // per-layer cross-stream syncs. Every consumer takes the stream as a
        // `&Arc<CudaStream>` parameter (weight cache, KV cache, every launch),
        // so all CUDA work transparently moves onto this stream.
        let stream = context
            .new_stream()
            .map_err(|err| RuntimeError::context("creating non-NULL CUDA runtime stream", err))?;
        // Disable per-slice `CudaEvent` tracking context-wide. cudarc enables it
        // by default; during stream capture `launch_builder.arg(&CudaSlice)` would
        // inject `cuStreamWaitEvent` → `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`.
        // On a single stream CUDA executes work in submission order, so explicit
        // event ordering is redundant for correctness AND incompatible with
        // capture — disabling it is the documented configuration. This MUST run
        // before any `CudaSlice` is created (slices minted before this keep their
        // events); the runtime stream is the only stream and this runs at init
        // before any allocation. SAFETY: every CUDA operation runs on this one
        // stream, so cross-stream ordering never arises — the sole effect is to
        // stop injecting redundant cuStreamWaitEvent/Record calls.
        unsafe {
            stream.context().disable_event_tracking();
        }

        Ok(Self {
            _context: context,
            stream,
            _module: module,
            q4k_matvec,
            q6k_matvec,
            q4k_matmul,
            q6k_matmul,
            q4k_dual_matvec,
            f32_gemv,
            f16_gemv,
            q4_matvec,
            q4_vecmat,
            kv_append,
            rms_norm,
            rms_norm_heads,
            geglu_silu,
            geglu_gelu_tanh,
            activation_silu,
            activation_gelu_tanh,
            residual_add,
            rope,
            decode_attention,
            prefill_attention,
            greedy_topk_partial,
            greedy_topk_final,
            weight_cache: WeightCache::default(),
            profile: super::RuntimeProfile::default(),
            capture_depth: std::sync::atomic::AtomicUsize::new(0),
            summary: format!(
                "CUDA device {device_name} (ordinal {ordinal}, sm_{cc_major}{cc_minor}, NVRTC target {arch}); native q4k_matvec/q6k_matvec/q4k_matmul/q6k_matmul/q4k_dual_matvec/f32_gemv/f16_gemv/q4_matvec/q4_vecmat/kv_append/rms_norm/rms_norm_heads/geglu_silu/geglu_gelu_tanh/activation_silu/activation_gelu_tanh/residual_add/rope/decode_attention/prefill_attention/greedy_topk_partial/greedy_topk_final loaded, remaining ops use CPU fallback; single non-NULL decode stream (B3B), CudaEvent tracking disabled"
            ),
        })
    }

    pub(crate) fn summary(&self) -> &str {
        &self.summary
    }

    /// Borrow the runtime's stream. Used by the KV cache to allocate
    /// per-layer device buffers (`CudaSlice<f32>` holds an `Arc<CudaStream>`
    /// so the cache outlives this borrow).
    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Snapshot the weight-cache hit/miss counters and resident-byte footprint
    /// (diagnostic). Surfaced to the `LARQL_GPU_DIAG` surface via
    /// `CudaBackend::weight_cache_diag`.
    pub(crate) fn weight_cache_stats(&self) -> CacheStats {
        self.weight_cache.stats()
    }

    /// Drop every cached device weight buffer. The backend calls this at each
    /// generation boundary (`reset_kv_cache`) so a backend reused across
    /// vindex loads can't serve a stale buffer mapped at a recycled address.
    pub(crate) fn flush_weight_cache(&self) {
        self.weight_cache.flush();
    }

    /// KV append: copy a contiguous block of `seq_len` freshly-projected
    /// K/V rows into the cache starting at slot `pos`. `new_k` / `new_v`
    /// are `seq_len * row_elems` elements each, uploaded in a single
    /// host→device transfer and written by one kernel launch over all rows
    /// (no per-row sync). `row_elems = num_kv_heads * head_dim` is
    /// precomputed here with overflow checks and passed to the kernel as a
    /// 32-bit value, so the device-side multiplication cannot wrap. The
    /// caller is responsible for bumping the layer's `current_len` after a
    /// successful launch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_kv_append(
        &self,
        new_k: &[f32],
        new_v: &[f32],
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
        pos: usize,
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), RuntimeError> {
        // `row_elems = num_kv_heads * head_dim`. Guard the product against
        // `usize` overflow and against exceeding the 32-bit kernel index
        // (the kernel takes `row_elems` as `unsigned int`).
        let row_elems = num_kv_heads.checked_mul(head_dim).ok_or_else(|| {
            RuntimeError::usage(format!(
                "kv_append row_elems overflow: num_kv_heads={num_kv_heads} head_dim={head_dim}"
            ))
        })?;
        if row_elems > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "kv_append row_elems {row_elems} exceeds the 32-bit kernel index limit"
            )));
        }
        let block_len = seq_len.checked_mul(row_elems).ok_or_else(|| {
            RuntimeError::usage(format!(
                "kv_append block overflow: seq_len={seq_len} row_elems={row_elems}"
            ))
        })?;
        if new_k.len() != block_len || new_v.len() != block_len {
            return Err(RuntimeError::usage(format!(
                "kv_append expected new_k/new_v of length {block_len} (seq_len={seq_len} row_elems={row_elems}), got {} / {}",
                new_k.len(),
                new_v.len()
            )));
        }
        // `pos` and `seq_len` are passed as 32-bit args; the slot offset is
        // computed in 64-bit on the device, so the only remaining truncation
        // surface is the args themselves.
        if pos > u32::MAX as usize || seq_len > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "kv_append shape (pos={pos}, seq_len={seq_len}) exceeds the 32-bit kernel index limit"
            )));
        }
        // Capacity check in checked arithmetic: the last written slot is
        // `(pos + seq_len) * row_elems + (row_elems - 1)`, i.e. the block
        // occupies `(pos + seq_len) * row_elems` elements.
        let end_slot = pos
            .checked_add(seq_len)
            .and_then(|last_row| last_row.checked_mul(row_elems))
            .ok_or_else(|| {
                RuntimeError::usage(
                    "kv_append slot offset overflow (pos + seq_len) * row_elems".to_string(),
                )
            })?;
        let max_seq_elems = k_cache.len();
        if end_slot > max_seq_elems {
            return Err(RuntimeError::usage(format!(
                "kv_append slots {pos}..{end_slot} exceed cache capacity {max_seq_elems}"
            )));
        }

        let k_in = self
            .stream
            .clone_htod(new_k)
            .map_err(|err| RuntimeError::context("uploading K block to CUDA", err))?;
        let v_in = self
            .stream
            .clone_htod(new_v)
            .map_err(|err| RuntimeError::context("uploading V block to CUDA", err))?;
        let threads_x = KV_APPEND_KERNEL.geometry.threads_per_group[0];
        let pos_u = pos as u32;
        let seq_u = seq_len as u32;
        let row_elems_u = row_elems as u32;
        let total = block_len as u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.kv_append);
        launch_args
            .arg(&k_in)
            .arg(&v_in)
            .arg(k_cache)
            .arg(v_cache)
            .arg(&pos_u)
            .arg(&seq_u)
            .arg(&row_elems_u);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA kv_append kernel", err))?;
        self.note_launch();
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA kv_append stream", err))
    }

    pub(crate) fn launch_q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        // Delegate to the device-resident variant so the shape validation,
        // weight-cache upload, and kernel-arg layout live in exactly one
        // place — the fused device-resident decode FFN chain and this
        // host-readback path share a single launch implementation and can't
        // drift on arg order (a drift would be silent UB in the unsafe
        // launch). Mirrors `launch_q4k_matmul`'s delegation pattern.
        //
        // Short-circuit the degenerate shape before touching the device:
        // cudarc rejects empty `clone_htod`/`alloc_zeros` (see the rms_norm
        // placeholder workaround), and the original host launcher returned
        // `Ok(vec![])` here with no device work — preserve that contract.
        if num_rows == 0 {
            return Ok(Vec::new());
        }
        let x_dev = self.upload_f32(x)?;
        let out_dev = self.launch_q4k_matvec_dev(q4k_data, &x_dev, num_rows, hidden)?;
        self.sync_dtoh_f32(&out_dev)
    }

    pub(crate) fn launch_q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        // Delegate to the device-resident variant — see `launch_q4k_matvec`
        // for the single-source rationale + the zero-shape short-circuit.
        if num_rows == 0 {
            return Ok(Vec::new());
        }
        let x_dev = self.upload_f32(x)?;
        let out_dev = self.launch_q6k_matvec_dev(q6k_data, &x_dev, num_rows, hidden)?;
        self.sync_dtoh_f32(&out_dev)
    }

    pub(crate) fn launch_q4k_matmul(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        // Delegate to the device-resident variant so the shape validation,
        // weight-cache upload, and kernel-arg layout live in exactly one
        // place. The fused device-resident chain and this host-readback path
        // therefore share a single launch implementation and can't drift on
        // arg order (a drift would be silent UB in the unsafe launch).
        //
        // Short-circuit the degenerate shape before touching the device:
        // cudarc rejects empty `clone_htod`/`alloc_zeros` (see the rms_norm
        // placeholder workaround), and the original host launcher returned
        // `Ok(vec![])` here with no device work — preserve that contract.
        if num_rows == 0 || seq_len == 0 {
            return Ok(Vec::new());
        }
        let x_dev = self.upload_f32(x)?;
        let out_dev = self.launch_q4k_matmul_dev(q4k_data, &x_dev, num_rows, hidden, seq_len)?;
        self.sync_dtoh_f32(&out_dev)
    }

    /// Amortised Q6_K × f32 matmul launch. Routed live through
    /// `QuantMatVec::q6k_matmul` (native-then-CPU fallback) and
    /// `ffn/weight.rs::quant_matmul`'s Q6_K arm.
    pub(crate) fn launch_q6k_matmul(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        // Delegate to the device-resident variant — see `launch_q4k_matmul`
        // for the single-source rationale + the zero-shape short-circuit.
        if num_rows == 0 || seq_len == 0 {
            return Ok(Vec::new());
        }
        let x_dev = self.upload_f32(x)?;
        let out_dev = self.launch_q6k_matmul_dev(q6k_data, &x_dev, num_rows, hidden, seq_len)?;
        self.sync_dtoh_f32(&out_dev)
    }

    pub(crate) fn launch_q4k_dual_matvec(
        &self,
        q4k_a: &[u8],
        q4k_b: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), RuntimeError> {
        if num_rows == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if x.len() != hidden {
            return Err(RuntimeError::usage(format!(
                "q4k_dual_matvec expected x.len() == hidden ({hidden}), got {}",
                x.len()
            )));
        }
        if !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q4k_dual_matvec hidden size must be a multiple of 256, got {hidden}"
            )));
        }

        let expected_bytes = num_rows
            .checked_mul(hidden / 256)
            .and_then(|blocks| blocks.checked_mul(144))
            .ok_or_else(|| RuntimeError::usage("q4k_dual_matvec byte-size overflow".to_string()))?;
        if q4k_a.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q4k_dual_matvec expected {expected_bytes} Q4_K bytes for w_a shape ({num_rows}, {hidden}), got {}",
                q4k_a.len()
            )));
        }
        if q4k_b.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q4k_dual_matvec expected {expected_bytes} Q4_K bytes for w_b shape ({num_rows}, {hidden}), got {}",
                q4k_b.len()
            )));
        }

        let wa_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q4k_a)
            .map_err(|err| RuntimeError::context("uploading Q4_K w_a weights to CUDA", err))?;
        let wb_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q4k_b)
            .map_err(|err| RuntimeError::context("uploading Q4_K w_b weights to CUDA", err))?;
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading activation vector to CUDA", err))?;
        let mut out_a_dev = self.stream.alloc_zeros::<f32>(num_rows).map_err(|err| {
            RuntimeError::context("allocating CUDA q4k_dual_matvec out_a buffer", err)
        })?;
        let mut out_b_dev = self.stream.alloc_zeros::<f32>(num_rows).map_err(|err| {
            RuntimeError::context("allocating CUDA q4k_dual_matvec out_b buffer", err)
        })?;
        let threads_x = Q4K_DUAL_MATVEC_KERNEL.geometry.threads_per_group[0];
        let n = num_rows as u32;
        let k = hidden as u32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.q4k_dual_matvec);
        launch_args
            .arg(&*wa_dev)
            .arg(&*wb_dev)
            .arg(&x_dev)
            .arg(&mut out_a_dev)
            .arg(&mut out_b_dev)
            .arg(&n)
            .arg(&k);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q4k_dual_matvec kernel", err))?;
        self.stream.synchronize().map_err(|err| {
            RuntimeError::context("synchronizing CUDA q4k_dual_matvec stream", err)
        })?;
        let out_a = self
            .stream
            .clone_dtoh(&out_a_dev)
            .map_err(|err| RuntimeError::context("reading CUDA q4k_dual_matvec out_a", err))?;
        let out_b = self
            .stream
            .clone_dtoh(&out_b_dev)
            .map_err(|err| RuntimeError::context("reading CUDA q4k_dual_matvec out_b", err))?;
        Ok((out_a, out_b))
    }

    /// Dense f32 GEMV launch. `w` is a row-major `[n, k]` slice (the
    /// flattened `ArrayView2` from `MatMul::f32_gemv`). Returns `out[n]`.
    pub(crate) fn launch_f32_gemv(
        &self,
        w: &[f32],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        if num_rows == 0 {
            return Ok(Vec::new());
        }
        // The device indexes in 64-bit, but `n`/`k` are passed as 32-bit
        // kernel args, so a dim exceeding u32::MAX would truncate. Reject
        // before the length check / upload (falling back to CPU) rather than
        // launching a truncated dispatch.
        if num_rows > u32::MAX as usize || hidden > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "f32_gemv shape ({num_rows}, {hidden}) exceeds the 32-bit kernel index limit"
            )));
        }
        if x.len() != hidden {
            return Err(RuntimeError::usage(format!(
                "f32_gemv expected x.len() == hidden ({hidden}), got {}",
                x.len()
            )));
        }
        let expected = num_rows
            .checked_mul(hidden)
            .ok_or_else(|| RuntimeError::usage("f32_gemv size overflow".to_string()))?;
        if w.len() != expected {
            return Err(RuntimeError::usage(format!(
                "f32_gemv expected {expected} weight elements for shape ({num_rows}, {hidden}), got {}",
                w.len()
            )));
        }

        let w_dev = self
            .weight_cache
            .get_or_upload_f32(&self.stream, w)
            .map_err(|err| RuntimeError::context("uploading f32 weights to CUDA", err))?;
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading f32 activation to CUDA", err))?;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(num_rows)
            .map_err(|err| RuntimeError::context("allocating CUDA f32_gemv output buffer", err))?;
        let threads_x = F32_GEMV_KERNEL.geometry.threads_per_group[0];
        let n = num_rows as u32;
        let k = hidden as u32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.f32_gemv);
        launch_args
            .arg(&*w_dev)
            .arg(&x_dev)
            .arg(&mut out_dev)
            .arg(&n)
            .arg(&k);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA f32_gemv kernel", err))?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA f32_gemv stream", err))?;
        self.stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA f32_gemv output", err))
    }

    /// Dense f16 GEMV launch. `w_f16` is a row-major `[n, k]` little-endian
    /// f16 byte slice (same layout as `MatMul::f16_gemv`). Returns `out[n]`.
    pub(crate) fn launch_f16_gemv(
        &self,
        w_f16: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        if num_rows == 0 {
            return Ok(Vec::new());
        }
        // The device computes the per-row byte offset as `2 * row * k` and
        // passes `n`/`k` as 32-bit. Guard against shapes that would exceed
        // the 32-bit kernel argument range before the length check / upload.
        // (A future kernel taking 64-bit grid dims would lift this.)
        if num_rows > u32::MAX as usize || hidden > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "f16_gemv shape ({num_rows}, {hidden}) exceeds the 32-bit kernel index limit"
            )));
        }
        if x.len() != hidden {
            return Err(RuntimeError::usage(format!(
                "f16_gemv expected x.len() == hidden ({hidden}), got {}",
                x.len()
            )));
        }
        let expected = num_rows
            .checked_mul(hidden)
            .and_then(|elems| elems.checked_mul(2))
            .ok_or_else(|| RuntimeError::usage("f16_gemv byte-size overflow".to_string()))?;
        if w_f16.len() != expected {
            return Err(RuntimeError::usage(format!(
                "f16_gemv expected {expected} f16 bytes for shape ({num_rows}, {hidden}), got {}",
                w_f16.len()
            )));
        }

        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, w_f16)
            .map_err(|err| RuntimeError::context("uploading f16 weights to CUDA", err))?;
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading f16 activation to CUDA", err))?;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(num_rows)
            .map_err(|err| RuntimeError::context("allocating CUDA f16_gemv output buffer", err))?;
        let threads_x = F16_GEMV_KERNEL.geometry.threads_per_group[0];
        let n = num_rows as u32;
        let k = hidden as u32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.f16_gemv);
        launch_args
            .arg(&*w_dev)
            .arg(&x_dev)
            .arg(&mut out_dev)
            .arg(&n)
            .arg(&k);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA f16_gemv kernel", err))?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA f16_gemv stream", err))?;
        self.stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA f16_gemv output", err))
    }

    /// Q4_0 × Q8 matvec launch. `q4_data` is packed Q4_0 (18 bytes per
    /// 32-element block), `q8_x` / `q8_scales` are the pre-quantised Q8
    /// input. Returns `out[num_rows]`.
    pub(crate) fn launch_q4_matvec(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        if num_rows == 0 {
            return Ok(Vec::new());
        }
        // The device indexes the row offset in 64-bit, but `n`/`k` are
        // passed as 32-bit kernel args, so a dim exceeding u32::MAX would
        // truncate. Reject before the length check / upload (falling back
        // to CPU) rather than launching a truncated dispatch.
        if num_rows > u32::MAX as usize || hidden > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "q4_matvec shape ({num_rows}, {hidden}) exceeds the 32-bit kernel index limit"
            )));
        }
        if !hidden.is_multiple_of(32) {
            return Err(RuntimeError::usage(format!(
                "q4_matvec hidden size must be a multiple of 32, got {hidden}"
            )));
        }
        let blocks = hidden / 32;
        if q8_x.len() != hidden {
            return Err(RuntimeError::usage(format!(
                "q4_matvec expected q8_x.len() == hidden ({hidden}), got {}",
                q8_x.len()
            )));
        }
        if q8_scales.len() != blocks {
            return Err(RuntimeError::usage(format!(
                "q4_matvec expected q8_scales.len() == hidden/32 ({blocks}), got {}",
                q8_scales.len()
            )));
        }
        let expected_bytes = num_rows
            .checked_mul(blocks)
            .and_then(|b| b.checked_mul(18))
            .ok_or_else(|| RuntimeError::usage("q4_matvec byte-size overflow".to_string()))?;
        if q4_data.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q4_matvec expected {expected_bytes} Q4_0 bytes for shape ({num_rows}, {hidden}), got {}",
                q4_data.len()
            )));
        }

        // `q4_data` is the immutable Q4_0 weight → cached. `q8_x` /
        // `q8_scales` are the per-token Q8 quantization of the input →
        // uploaded fresh every call (they change per token).
        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q4_data)
            .map_err(|err| RuntimeError::context("uploading Q4_0 weights to CUDA", err))?;
        let q8_dev = self
            .stream
            .clone_htod(q8_x)
            .map_err(|err| RuntimeError::context("uploading Q8 input to CUDA", err))?;
        let scales_dev = self
            .stream
            .clone_htod(q8_scales)
            .map_err(|err| RuntimeError::context("uploading Q8 scales to CUDA", err))?;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(num_rows)
            .map_err(|err| RuntimeError::context("allocating CUDA q4_matvec output buffer", err))?;
        let threads_x = Q4_MATVEC_KERNEL.geometry.threads_per_group[0];
        let n = num_rows as u32;
        let k = hidden as u32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.q4_matvec);
        launch_args
            .arg(&*w_dev)
            .arg(&q8_dev)
            .arg(&scales_dev)
            .arg(&mut out_dev)
            .arg(&n)
            .arg(&k);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q4_matvec kernel", err))?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA q4_matvec stream", err))?;
        self.stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA q4_matvec output", err))
    }

    /// Q4_0 vector-matrix launch. `out[hidden] = activation[intermediate] @
    /// Q4[intermediate, hidden]`. One output column per thread.
    pub(crate) fn launch_q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        if hidden == 0 {
            return Ok(Vec::new());
        }
        if intermediate == 0 {
            return Ok(vec![0.0f32; hidden]);
        }
        if intermediate > u32::MAX as usize || hidden > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "q4_vecmat shape ({intermediate}, {hidden}) exceeds the 32-bit kernel index limit"
            )));
        }
        if activation.len() != intermediate {
            return Err(RuntimeError::usage(format!(
                "q4_vecmat expected activation.len() == intermediate ({intermediate}), got {}",
                activation.len()
            )));
        }
        if !hidden.is_multiple_of(32) {
            return Err(RuntimeError::usage(format!(
                "q4_vecmat hidden size must be a multiple of 32, got {hidden}"
            )));
        }
        let blocks_per_row = hidden / 32;
        let expected_bytes = intermediate
            .checked_mul(blocks_per_row)
            .and_then(|b| b.checked_mul(18))
            .ok_or_else(|| RuntimeError::usage("q4_vecmat byte-size overflow".to_string()))?;
        if q4_data.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q4_vecmat expected {expected_bytes} Q4_0 bytes for shape ({intermediate}, {hidden}), got {}",
                q4_data.len()
            )));
        }

        let act_dev = self
            .stream
            .clone_htod(activation)
            .map_err(|err| RuntimeError::context("uploading activation to CUDA", err))?;
        // `q4_data` is the immutable Q4_0 weight → cached.
        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q4_data)
            .map_err(|err| RuntimeError::context("uploading Q4_0 weights to CUDA", err))?;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(hidden)
            .map_err(|err| RuntimeError::context("allocating CUDA q4_vecmat output buffer", err))?;
        let threads_x = Q4_VECMAT_KERNEL.geometry.threads_per_group[0];
        let n = intermediate as u32;
        let k = hidden as u32;
        let cfg = LaunchConfig {
            grid_dim: (k.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.q4_vecmat);
        launch_args
            .arg(&act_dev)
            .arg(&*w_dev)
            .arg(&mut out_dev)
            .arg(&n)
            .arg(&k);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q4_vecmat kernel", err))?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA q4_vecmat stream", err))?;
        self.stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA q4_vecmat output", err))
    }

    /// Native RMSNorm over each row of a `[rows, cols]` matrix — the device
    /// twin of `larql_compute::residual::rms_norm_eps`. `x` is the row-major
    /// `[rows*cols]` flattened input. `weight` is `Some` for the learned-
    /// weight case (the `None` arm of the CPU reference passes a zero-length
    /// placeholder; `has_weight` distinguishes the two on the device). Returns
    /// `out[rows*cols]`. One thread-block per row; the block size is capped
    /// at 1024 threads (32 warps) since the device reduction uses a fixed
    /// 32-slot shared array.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_rms_norm(
        &self,
        x: &[f32],
        weight: Option<&[f32]>,
        out: &mut [f32],
        rows: usize,
        cols: usize,
        eps: f64,
        offset: f32,
    ) -> Result<(), RuntimeError> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        if x.len() != rows * cols || out.len() != rows * cols {
            return Err(RuntimeError::usage(format!(
                "rms_norm expected x/out of length {} (rows={rows} cols={cols}), got {} / {}",
                rows * cols,
                x.len(),
                out.len()
            )));
        }
        if let Some(w) = weight {
            if w.len() != cols {
                return Err(RuntimeError::usage(format!(
                    "rms_norm expected weight of length {cols}, got {}",
                    w.len()
                )));
            }
        }
        // 32-bit kernel args guards (the device indexes in 64-bit, but
        // `rows`/`cols` are passed as `unsigned int`).
        if rows > u32::MAX as usize || cols > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "rms_norm shape ({rows}, {cols}) exceeds the 32-bit kernel index limit"
            )));
        }

        // Delegate the kernel-arg layout + launch to `launch_rms_norm_dev`
        // (the single source of truth) so this host-readback path and the
        // device-resident attention chain share one launch implementation and
        // can't drift on arg order.
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading rms_norm input to CUDA", err))?;
        let out_dev = self.launch_rms_norm_dev(&x_dev, weight, rows, cols, eps, offset)?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA rms_norm stream", err))?;
        let host_out = self
            .stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA rms_norm output", err))?;
        out.copy_from_slice(&host_out);
        Ok(())
    }

    /// Native per-head RMSNorm — the device twin of
    /// `larql_compute::residual::rms_norm_heads` / `rms_norm_heads_no_weight`.
    /// `x` is the row-major `[seq_len * num_heads * head_dim]` flattened
    /// input. One thread-block per (position, head). `has_weight = 0`
    /// selects the parameter-free path (mirrors the `None`-weight CPU
    /// reference `rms_norm_heads_no_weight`). The block size is capped at
    /// 1024 threads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_rms_norm_heads(
        &self,
        x: &[f32],
        weight: Option<&[f32]>,
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f64,
        offset: f32,
    ) -> Result<(), RuntimeError> {
        let total = seq_len
            .checked_mul(num_heads)
            .and_then(|p| p.checked_mul(head_dim));
        let total = match total {
            Some(t) if x.len() == t && out.len() == t => t,
            _ => {
                return Err(RuntimeError::usage(format!(
                    "rms_norm_heads expected x/out of length {} (seq={seq_len} heads={num_heads} dim={head_dim}), got {} / {}",
                    seq_len * num_heads * head_dim,
                    x.len(),
                    out.len()
                )))
            }
        };
        if total == 0 {
            return Ok(());
        }
        // The CPU reference `rms_norm_heads_eps` indexes the weight as
        // `weight[d]` (a single `head_dim`-length slice broadcast across all
        // heads — Gemma3/4 `q_norm.weight`/`k_norm.weight` are shape
        // `[head_dim]`). The device kernel matches that broadcast indexing,
        // so accept only `head_dim`-length weights.
        if let Some(w) = weight {
            if w.len() != head_dim {
                return Err(RuntimeError::usage(format!(
                    "rms_norm_heads expected weight of length {head_dim} (broadcast across heads), got {}",
                    w.len()
                )));
            }
        }
        if seq_len > u32::MAX as usize
            || num_heads > u32::MAX as usize
            || head_dim > u32::MAX as usize
            || (num_heads as u64) * (head_dim as u64) > u32::MAX as u64
        {
            return Err(RuntimeError::usage(format!(
                "rms_norm_heads shape (seq={seq_len}, heads={num_heads}, dim={head_dim}) exceeds the 32-bit kernel index limit"
            )));
        }

        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading rms_norm_heads input to CUDA", err))?;
        // Delegate the kernel-arg layout + launch to `launch_rms_norm_heads_dev`
        // (single source of truth) — see `launch_rms_norm`.
        let out_dev = self
            .launch_rms_norm_heads_dev(&x_dev, weight, seq_len, num_heads, head_dim, eps, offset)?;
        self.stream.synchronize().map_err(|err| {
            RuntimeError::context("synchronizing CUDA rms_norm_heads stream", err)
        })?;
        let host_out = self
            .stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA rms_norm_heads output", err))?;
        out.copy_from_slice(&host_out);
        Ok(())
    }

    /// Native GEGLU-SiLU launch: `out[i] = silu(gate[i]) * up[i]`, one
    /// thread per element. `gate`, `up`, `out` are each `n` elements.
    pub(crate) fn launch_geglu_silu(
        &self,
        gate: &[f32],
        up: &[f32],
        out: &mut [f32],
        n: usize,
    ) -> Result<(), RuntimeError> {
        self.launch_elementwise_binary(&self.geglu_silu, gate, up, out, None, n, "geglu_silu")
    }

    /// Native GEGLU-GELU-tanh launch: `out[i] = gelu_tanh(gate[i]) * up[i]`.
    pub(crate) fn launch_geglu_gelu_tanh(
        &self,
        gate: &[f32],
        up: &[f32],
        out: &mut [f32],
        n: usize,
    ) -> Result<(), RuntimeError> {
        self.launch_elementwise_binary(
            &self.geglu_gelu_tanh,
            gate,
            up,
            out,
            None,
            n,
            "geglu_gelu_tanh",
        )
    }

    /// Native standard SiLU launch: `out[i] = silu(x[i])`. `up` is unused
    /// (the binary kernel signature is shared via a no-op second buffer is
    /// avoided here — the standalone `activation_silu` kernel takes a single
    /// input).
    pub(crate) fn launch_activation_silu(
        &self,
        input: &[f32],
        out: &mut [f32],
        n: usize,
    ) -> Result<(), RuntimeError> {
        self.launch_elementwise_unary(&self.activation_silu, input, out, n, "activation_silu")
    }

    /// Native standard GELU-tanh launch: `out[i] = gelu_tanh(x[i])`.
    pub(crate) fn launch_activation_gelu_tanh(
        &self,
        input: &[f32],
        out: &mut [f32],
        n: usize,
    ) -> Result<(), RuntimeError> {
        self.launch_elementwise_unary(
            &self.activation_gelu_tanh,
            input,
            out,
            n,
            "activation_gelu_tanh",
        )
    }

    /// Native scaled residual add: `out[i] = h[i] + b_scale * x[i]`, one
    /// thread per element. `h`, `x`, `out` are each `n` elements. The
    /// device form fuses the `b_scale == 1.0` / `b_scale != 1.0` arms of the
    /// host `add_residual` (the two are numerically identical, so no branch
    /// is needed). Routed through the shared binary launcher with `b_scale`
    /// as the extra scalar arg (the activation kernels pass `None`).
    pub(crate) fn launch_residual_add(
        &self,
        h: &[f32],
        x: &[f32],
        out: &mut [f32],
        b_scale: f32,
        n: usize,
    ) -> Result<(), RuntimeError> {
        self.launch_elementwise_binary(
            &self.residual_add,
            h,
            x,
            out,
            Some(b_scale),
            n,
            "residual_add",
        )
    }

    /// Native RoPE launch over a `[seq_len, num_heads * head_dim]` Q/K tensor.
    /// `inv_freq` is `half_rotary = rotary_dim/2` precomputed `f64`
    /// frequencies (the host builds them identically to the reference, so the
    /// `llama3` wavelength-band variant is handled before upload). `n =
    /// seq_len * num_heads * head_dim` is the element count; the kernel is one
    /// thread per element.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_rope(
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
    ) -> Result<(), RuntimeError> {
        let total = seq_len
            .checked_mul(num_heads)
            .and_then(|p| p.checked_mul(head_dim));
        let total = match total {
            Some(t) if x.len() == t && out.len() == t => t,
            _ => {
                return Err(RuntimeError::usage(format!(
                    "rope expected x/out of length {} (seq={seq_len} heads={num_heads} dim={head_dim}), got {} / {}",
                    seq_len * num_heads * head_dim,
                    x.len(),
                    out.len()
                )))
            }
        };
        if total == 0 {
            return Ok(());
        }
        if inv_freq.len() != half_rotary {
            return Err(RuntimeError::usage(format!(
                "rope expected inv_freq of length {half_rotary}, got {}",
                inv_freq.len()
            )));
        }
        if half_rotary == 0 {
            return Err(RuntimeError::usage(
                "rope requires half_rotary >= 1 (rotary_dim >= 2)".to_string(),
            ));
        }
        // Guard the grid + flat-index against the 32-bit limit. The kernel
        // indexes in 64-bit, but the element count drives a 1D grid whose
        // width must fit a `u32` (matching the other elementwise launchers).
        if total > u32::MAX as usize
            || seq_len > u32::MAX as usize
            || num_heads > u32::MAX as usize
            || head_dim > u32::MAX as usize
        {
            return Err(RuntimeError::usage(format!(
                "rope shape (seq={seq_len}, heads={num_heads}, dim={head_dim}) exceeds the 32-bit kernel index limit"
            )));
        }
        // The `position_divisor > 0` clamp and the kernel-arg layout live in
        // `launch_rope_dev` (single source of truth); the host guards above
        // only cheap-reject obviously-bad host slices before the upload.

        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading rope input to CUDA", err))?;
        // Delegate the kernel-arg layout + launch to `launch_rope_dev` (single
        // source of truth) — see `launch_rms_norm`.
        let out_dev = self.launch_rope_dev(
            &x_dev,
            inv_freq,
            seq_len,
            num_heads,
            head_dim,
            half_rotary,
            position_offset,
            position_divisor,
        )?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA rope stream", err))?;
        let host_out = self
            .stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA rope output", err))?;
        out.copy_from_slice(&host_out);
        Ok(())
    }

    /// Fused decode-step GQA attention — the device twin of
    /// `gqa_attention_decode_step`. `q` is `[num_q * head_dim]` (the new
    /// token's Q, post-RoPE); `k_cache`/`v_cache` are `[total_len *
    /// kv_dim]` (`kv_dim = num_kv * head_dim`, row-major). `out` receives
    /// `[num_q * head_dim]` and is resized to exactly that length.
    ///
    /// One thread-block per query head fuses QKᵀ → scale (+ optional
    /// `softcap`) → softmax → weighted-V. The `scores` scratch buffer
    /// (`num_q * total_len` f32) is allocated and freed per call (the
    /// fully-fused single-command-buffer pipeline will fold it into a
    /// persistent device buffer).
    ///
    /// `softcap` is applied when `Some(cap)`; `None` skips the cap.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_decode_attention(
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
    ) -> Result<(), RuntimeError> {
        let q_len = num_q.checked_mul(head_dim);
        let kv_len = total_len.checked_mul(kv_dim);
        let score_len = num_q.checked_mul(total_len);
        let (q_len, kv_len, score_len) = match (q_len, kv_len, score_len) {
            (Some(ql), Some(kvl), Some(sl))
                if q.len() == ql && k_cache.len() == kvl && v_cache.len() == kvl =>
            {
                (ql, kvl, sl)
            }
            _ => {
                return Err(RuntimeError::usage(format!(
                    "decode_attention shape mismatch: q={} (expected {q_len:?}), k/v={} (expected {kv_len:?}, kv_dim={kv_dim}, total={total_len}), num_q={num_q}, head_dim={head_dim}",
                    q.len(),
                    k_cache.len(),
                )))
            }
        };
        if total_len == 0 {
            out.clear();
            out.resize(num_q * head_dim, 0.0);
            return Ok(());
        }
        if reps == 0 {
            return Err(RuntimeError::usage(
                "decode_attention requires reps >= 1 (num_kv heads > 0)".to_string(),
            ));
        }
        // Guard the grid + 64-bit device indices against the 32-bit limit.
        if q_len > u32::MAX as usize
            || kv_len > u32::MAX as usize
            || score_len > u32::MAX as usize
            || num_q > u32::MAX as usize
            || head_dim > u32::MAX as usize
            || kv_dim > u32::MAX as usize
            || total_len > u32::MAX as usize
        {
            return Err(RuntimeError::usage(format!(
                "decode_attention shape (num_q={num_q}, head_dim={head_dim}, kv_dim={kv_dim}, total_len={total_len}) exceeds the 32-bit kernel index limit"
            )));
        }

        let q_dev = self
            .stream
            .clone_htod(q)
            .map_err(|err| RuntimeError::context("uploading decode_attention q to CUDA", err))?;
        let k_dev = self.stream.clone_htod(k_cache).map_err(|err| {
            RuntimeError::context("uploading decode_attention K cache to CUDA", err)
        })?;
        let v_dev = self.stream.clone_htod(v_cache).map_err(|err| {
            RuntimeError::context("uploading decode_attention V cache to CUDA", err)
        })?;
        let mut scores_dev = self.stream.alloc_zeros::<f32>(score_len).map_err(|err| {
            RuntimeError::context("allocating decode_attention scores scratch", err)
        })?;
        let out_dev = self.launch_decode_attention_dev(
            &q_dev,
            &k_dev,
            &v_dev,
            &mut scores_dev,
            scale,
            softcap,
            num_q,
            head_dim,
            kv_dim,
            reps,
            total_len,
        )?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing decode_attention stream", err))?;
        let host_out = self
            .stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading decode_attention output", err))?;
        out.clear();
        out.extend_from_slice(&host_out);
        Ok(())
    }

    /// Native fused decode-step GQA attention, device-resident variant — the
    /// twin of [`launch_decode_attention`] used by the device-resident decode
    /// attention chain. `q_dev`/`k_dev`/`v_dev` are already on the device
    /// (the chain's resident Q after RoPE + the full uploaded KV cache);
    /// `scores_dev` is a caller-owned scratch of length `num_q * total_len`
    /// (owned by the chain scope so its drop happens after the final readback,
    /// not mid-chain — on pool-less devices a mid-chain `CudaSlice::drop`
    /// forces a stream sync). The returned `CudaSlice<f32>` stays resident
    /// with NO internal sync/dtoh, so the O projection can consume it on the
    /// same stream without a round trip. All shape/u32-index guards live here
    /// (single source of truth — the host-readback
    /// [`launch_decode_attention`] delegates to this).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_decode_attention_dev(
        &self,
        q_dev: &CudaSlice<f32>,
        k_dev: &CudaSlice<f32>,
        v_dev: &CudaSlice<f32>,
        scores_dev: &mut CudaSlice<f32>,
        scale: f32,
        softcap: Option<f32>,
        num_q: usize,
        head_dim: usize,
        kv_dim: usize,
        reps: usize,
        total_len: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        let q_len = num_q.checked_mul(head_dim);
        let kv_len = total_len.checked_mul(kv_dim);
        let score_len = num_q.checked_mul(total_len);
        let (q_len, kv_len, score_len) = match (q_len, kv_len, score_len) {
            (Some(ql), Some(kvl), Some(sl))
                if q_dev.len() == ql
                    && k_dev.len() == kvl
                    && v_dev.len() == kvl
                    && scores_dev.len() == sl =>
            {
                (ql, kvl, sl)
            }
            _ => {
                return Err(RuntimeError::usage(format!(
                    "decode_attention_dev shape mismatch: q={} (expected {q_len:?}), k={} / v={} (expected {kv_len:?}, kv_dim={kv_dim}, total={total_len}), scores={} (expected {score_len:?}), num_q={num_q}, head_dim={head_dim}",
                    q_dev.len(),
                    k_dev.len(),
                    v_dev.len(),
                    scores_dev.len(),
                )))
            }
        };
        if total_len == 0 {
            // Empty context: attention output is zero (no keys to attend to).
            return self.stream.alloc_zeros::<f32>(q_len).map_err(|err| {
                RuntimeError::context("allocating decode_attention_dev output", err)
            });
        }
        if reps == 0 {
            return Err(RuntimeError::usage(
                "decode_attention_dev requires reps >= 1 (num_kv heads > 0)".to_string(),
            ));
        }
        if q_len > u32::MAX as usize
            || kv_len > u32::MAX as usize
            || score_len > u32::MAX as usize
            || num_q > u32::MAX as usize
            || head_dim > u32::MAX as usize
            || kv_dim > u32::MAX as usize
            || total_len > u32::MAX as usize
        {
            return Err(RuntimeError::usage(format!(
                "decode_attention_dev shape (num_q={num_q}, head_dim={head_dim}, kv_dim={kv_dim}, total_len={total_len}) exceeds the 32-bit kernel index limit"
            )));
        }

        self.launch_decode_attention_kernel(
            q_dev, k_dev, v_dev, scores_dev, scale, softcap, num_q, head_dim, kv_dim, reps,
            total_len,
        )
    }

    /// Native fused decode-step GQA attention over the device-resident
    /// `CudaKVCache` — the resident-KV twin of
    /// [`launch_decode_attention_dev`] (GPU-006). `q_dev` is the new token's
    /// post-RoPE Q (resident from the attention device chain); `k_dev`/`v_dev`
    /// are the **resident** per-layer K/V buffers from `CudaKVCache.layers[li]`
    /// (`[max_seq, num_kv_heads, head_dim]` f32 = `[max_seq, kv_dim]` flattened
    /// — exactly the `i * kv_dim + kv_off` layout the `decode_attention` kernel
    /// reads, so no new kernel is needed). `scores_dev` is caller-owned scratch
    /// of length `num_q * total_len`. `total_len` is the post-append cursor
    /// (the kernel attends over exactly `0..total_len`, never the uninitialized
    /// capacity rows beyond it). Returns the `[num_q * head_dim]` attention
    /// output resident on the device (no sync, no dtoh) so the O projection
    /// consumes it on the same stream — collapsing the per-token full-KV
    /// host readback + re-upload the full-upload path pays.
    ///
    /// Unlike [`launch_decode_attention_dev`], the K/V buffers are borrowed
    /// (resident in the cache), and `total_len` is trusted to be the valid
    /// prefix length (the cache cursor) rather than the full buffer length —
    /// the caller (the resident-KV decode path) guarantees the cursor is
    /// advanced by exactly one per decode token and the append lands at the
    /// right slot. The shape validation here checks `kv_dim`-strided capacity
    /// (`max_seq * kv_dim`) rather than `total_len * kv_dim`: the device buffer
    /// is pre-allocated to `max_seq` rows, and only the `0..total_len` prefix
    /// is valid.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_decode_attention_resident_dev(
        &self,
        q_dev: &CudaSlice<f32>,
        k_dev: &CudaSlice<f32>,
        v_dev: &CudaSlice<f32>,
        scores_dev: &mut CudaSlice<f32>,
        scale: f32,
        softcap: Option<f32>,
        num_q: usize,
        head_dim: usize,
        kv_dim: usize,
        reps: usize,
        total_len: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        let q_len = num_q.checked_mul(head_dim);
        let score_len = num_q.checked_mul(total_len);
        let (q_len, score_len) = match (q_len, score_len) {
            (Some(ql), Some(sl)) if q_dev.len() == ql && scores_dev.len() == sl => (ql, sl),
            _ => {
                return Err(RuntimeError::usage(format!(
                    "decode_attention_resident_dev shape mismatch: q={} (expected {q_len:?}), scores={} (expected {score_len:?}), num_q={num_q}, head_dim={head_dim}, total_len={total_len}",
                    q_dev.len(),
                    scores_dev.len(),
                )))
            }
        };
        // The resident K/V buffer is `max_seq * kv_dim` elements; `total_len`
        // is the valid cursor (≤ max_seq). Validate the capacity holds the
        // valid prefix, and that K and V match (same per-layer geometry).
        if total_len == 0 {
            return self.stream.alloc_zeros::<f32>(q_len).map_err(|err| {
                RuntimeError::context("allocating decode_attention_resident_dev output", err)
            });
        }
        if kv_dim == 0 || reps == 0 {
            return Err(RuntimeError::usage(format!(
                "decode_attention_resident_dev requires kv_dim >= 1 and reps >= 1 (got kv_dim={kv_dim}, reps={reps})"
            )));
        }
        // `total_len <= max_seq` implied by the cursor invariant; check the
        // buffer holds at least `total_len * kv_dim` valid elements.
        let need_kv = match total_len.checked_mul(kv_dim) {
            Some(n) if k_dev.len() >= n && v_dev.len() >= n => n,
            _ => {
                return Err(RuntimeError::usage(format!(
                    "decode_attention_resident_dev resident K/V capacity {} / {} < total_len*kv_dim ({total_len}*{kv_dim}={})",
                    k_dev.len(),
                    v_dev.len(),
                    total_len.saturating_mul(kv_dim),
                )))
            }
        };
        let _ = need_kv;
        if q_len > u32::MAX as usize
            || score_len > u32::MAX as usize
            || num_q > u32::MAX as usize
            || head_dim > u32::MAX as usize
            || kv_dim > u32::MAX as usize
            || total_len > u32::MAX as usize
        {
            return Err(RuntimeError::usage(format!(
                "decode_attention_resident_dev shape (num_q={num_q}, head_dim={head_dim}, kv_dim={kv_dim}, total_len={total_len}) exceeds the 32-bit kernel index limit"
            )));
        }

        self.launch_decode_attention_kernel(
            q_dev, k_dev, v_dev, scores_dev, scale, softcap, num_q, head_dim, kv_dim, reps,
            total_len,
        )
    }

    /// Shared kernel launch for [`launch_decode_attention_dev`] and
    /// [`launch_decode_attention_resident_dev`] — single source of the
    /// `decode_attention` kernel-arg layout so the two callers can't drift on
    /// arg order (a drift would be silent UB in the unsafe launch). Allocates
    /// the `[num_q * head_dim]` output, binds all args, and launches one block
    /// per query head (the kernel collaboratively fuses QKᵀ → scale → softmax →
    /// weighted-V within the block). No sync/dtoh — the output stays resident
    /// for the O projection to consume on the same stream. All shape/u32 guards
    /// live in the two callers; this helper assumes validated shapes.
    #[allow(clippy::too_many_arguments)]
    fn launch_decode_attention_kernel(
        &self,
        q_dev: &CudaSlice<f32>,
        k_dev: &CudaSlice<f32>,
        v_dev: &CudaSlice<f32>,
        scores_dev: &mut CudaSlice<f32>,
        scale: f32,
        softcap: Option<f32>,
        num_q: usize,
        head_dim: usize,
        kv_dim: usize,
        reps: usize,
        total_len: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        let q_len = num_q * head_dim;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(q_len)
            .map_err(|err| RuntimeError::context("allocating decode_attention_dev output", err))?;

        let threads = DECODE_ATTENTION_KERNEL.geometry.threads_per_group[0];
        let cfg = LaunchConfig {
            grid_dim: (num_q as u32, 1, 1),
            block_dim: (threads, 1, 1),
            // shm_max[256] f32 + shm_sum[256] f64.
            shared_mem_bytes: threads * 4 + threads * 8,
        };
        let (softcap_val, has_softcap) = match softcap {
            Some(cap) => (cap, 1u32),
            None => (0.0f32, 0u32),
        };
        let num_q_u = num_q as u32;
        let head_dim_u = head_dim as u32;
        let kv_dim_u = kv_dim as u32;
        let reps_u = reps as u32;
        let total_len_u = total_len as u32;
        let mut launch_args = self.stream.launch_builder(&self.decode_attention);
        launch_args
            .arg(q_dev)
            .arg(k_dev)
            .arg(v_dev)
            .arg(scores_dev)
            .arg(&mut out_dev)
            .arg(&scale)
            .arg(&softcap_val)
            .arg(&has_softcap)
            .arg(&num_q_u)
            .arg(&head_dim_u)
            .arg(&kv_dim_u)
            .arg(&reps_u)
            .arg(&total_len_u);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA decode_attention kernel", err))?;
        self.note_launch();
        Ok(out_dev)
    }

    /// Native fused prefill (seq×seq) causal GQA attention — the device twin
    /// of `gqa_attention_with_weights` (the symmetric `gqa_attention_capture`
    /// path). `q` is `[seq, num_q*head_dim]`; `k`/`v` are `[seq, kv_dim]`; on
    /// success `out` is filled with `[seq, num_q*head_dim]` (row-major). One
    /// thread-block per `(query head, query position)`; dynamic shared memory
    /// (`3072 + seq_len*4` bytes) holds the per-block causal `scores` scratch
    /// + the fixed 256-slot max/sum reduction arrays.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_prefill_attention(
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
    ) -> Result<(), RuntimeError> {
        // Cheap host-slice length validation (the deep shape/u32/shared-mem
        // guards live in `launch_prefill_attention_dev` — single source of
        // truth — so this host-readback path and the device-resident attention
        // chain share one launch implementation).
        let q_dim = num_q.saturating_mul(head_dim);
        let q_len = seq_len.saturating_mul(q_dim);
        let kv_len = seq_len.saturating_mul(kv_dim);
        if q.len() != q_len || k.len() != kv_len || v.len() != kv_len {
            return Err(RuntimeError::usage(format!(
                "prefill_attention shape mismatch: q={} (expected {q_len}), k={} / v={} (expected {kv_len}, kv_dim={kv_dim}, seq={seq_len}), num_q={num_q}, head_dim={head_dim}",
                q.len(),
                k.len(),
                v.len(),
            )));
        }
        if seq_len == 0 {
            out.clear();
            out.resize(num_q * head_dim, 0.0);
            return Ok(());
        }

        let q_dev = self
            .stream
            .clone_htod(q)
            .map_err(|err| RuntimeError::context("uploading prefill_attention q to CUDA", err))?;
        let k_dev = self
            .stream
            .clone_htod(k)
            .map_err(|err| RuntimeError::context("uploading prefill_attention K to CUDA", err))?;
        let v_dev = self
            .stream
            .clone_htod(v)
            .map_err(|err| RuntimeError::context("uploading prefill_attention V to CUDA", err))?;
        let out_dev = self.launch_prefill_attention_dev(
            &q_dev, &k_dev, &v_dev, scale, softcap, num_q, head_dim, kv_dim, reps, seq_len,
        )?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing prefill_attention stream", err))?;
        let host_out = self
            .stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading prefill_attention output", err))?;
        out.clear();
        out.extend_from_slice(&host_out);
        Ok(())
    }

    /// Shared binary elementwise dispatch (in_a, in_b → out). The GEGLU
    /// activation kernels take `(a, b, out, n)`; the residual-add kernel
    /// additionally takes a scalar `b_scale` between `out` and `n`, supplied
    /// via `extra_scalar = Some(b_scale)` (the activation kernels pass
    /// `None`). The upload/launch/sync/readback + overflow guard is identical
    /// for all five callers — only the function handle, the optional scalar,
    /// and the context string differ.
    #[allow(clippy::too_many_arguments)]
    fn launch_elementwise_binary(
        &self,
        func: &CudaFunction,
        in_a: &[f32],
        in_b: &[f32],
        out: &mut [f32],
        extra_scalar: Option<f32>,
        n: usize,
        ctx: &'static str,
    ) -> Result<(), RuntimeError> {
        if n == 0 {
            return Ok(());
        }
        if in_a.len() != n || in_b.len() != n || out.len() != n {
            return Err(RuntimeError::usage(format!(
                "{ctx} expected in_a/in_b/out of length {n}, got {} / {} / {}",
                in_a.len(),
                in_b.len(),
                out.len()
            )));
        }
        // Validate the host slices before paying for the upload. The kernel-arg
        // layout + LaunchConfig live in exactly one place
        // (`launch_elementwise_binary_dev`) so this host-readback path and the
        // device-resident chain share one launch implementation and can't
        // drift on arg order (a drift would be silent UB in the unsafe launch).
        let a_dev = self
            .stream
            .clone_htod(in_a)
            .map_err(|err| RuntimeError::context_concat("uploading ", ctx, " in_a to CUDA", err))?;
        let b_dev = self
            .stream
            .clone_htod(in_b)
            .map_err(|err| RuntimeError::context_concat("uploading ", ctx, " in_b to CUDA", err))?;
        let out_dev =
            self.launch_elementwise_binary_dev(func, &a_dev, &b_dev, extra_scalar, n, ctx)?;
        self.stream.synchronize().map_err(|err| {
            RuntimeError::context_concat("synchronizing CUDA ", ctx, " stream", err)
        })?;
        let host_out = self
            .stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context_concat("reading CUDA ", ctx, " output", err))?;
        out.copy_from_slice(&host_out);
        Ok(())
    }

    /// Shared unary elementwise dispatch (input → out). The standalone
    /// `activation_silu`/`activation_gelu_tanh` kernels take a single input.
    fn launch_elementwise_unary(
        &self,
        func: &CudaFunction,
        input: &[f32],
        out: &mut [f32],
        n: usize,
        ctx: &'static str,
    ) -> Result<(), RuntimeError> {
        if n == 0 {
            return Ok(());
        }
        if input.len() != n || out.len() != n {
            return Err(RuntimeError::usage(format!(
                "{ctx} expected input/out of length {n}, got {} / {}",
                input.len(),
                out.len()
            )));
        }
        // Delegate the launch to `launch_elementwise_unary_dev` so the
        // kernel-arg layout is single-source — see
        // `launch_elementwise_binary` for the rationale.
        let input_dev = self.stream.clone_htod(input).map_err(|err| {
            RuntimeError::context_concat("uploading ", ctx, " input to CUDA", err)
        })?;
        let out_dev = self.launch_elementwise_unary_dev(func, &input_dev, n, ctx)?;
        self.stream.synchronize().map_err(|err| {
            RuntimeError::context_concat("synchronizing CUDA ", ctx, " stream", err)
        })?;
        let host_out = self
            .stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context_concat("reading CUDA ", ctx, " output", err))?;
        out.copy_from_slice(&host_out);
        Ok(())
    }

    // ── device-resident launch variants ──────────────────────────────────
    //
    // These are the per-projection round-trip-collapse primitives. Unlike the
    // launchers above, they take an input that is **already on the device**
    // (`&CudaSlice<f32>`) and return a **device-resident** output
    // (`CudaSlice<f32>`) — they do NOT upload the input, do NOT synchronize,
    // and do NOT read the output back to the host. A caller chains several of
    // these on the same stream (CUDA stream-ordered, so a kernel reading a
    // buffer written by an earlier kernel on the same stream sees the data
    // without a sync) and performs a single `sync_dtoh_f32` at the end of the
    // chain. This collapses the per-projection htod(input) + dtoh(output)
    // round-trips that the host-orchestrated pipeline pays between every
    // sequential kernel. Weights still go through the persistent
    // `weight_cache`; only transient activations stay resident across the
    // chain (they are dropped once the chain's final `sync_dtoh_f32` returns).

    /// Upload an f32 activation to the device once, returning a
    /// device-resident handle. The chain caller holds this for the lifetime of
    /// the chain (until the final `sync_dtoh_f32`).
    pub(crate) fn upload_f32(&self, x: &[f32]) -> Result<CudaSlice<f32>, RuntimeError> {
        // LARQL-GPU-PROFILE-001: count the per-token activation host→device
        // upload. Weights go through the (cached) weight cache; activations
        // are fresh every call, so this is the dominant per-token htod cost.
        if crate::options::gpu_profile_enabled() {
            self.note_htod(x.len() * 4);
        }
        self.stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading activation to CUDA", err))
    }

    /// Allocate a zero-initialised device-resident f32 buffer of length `n`
    /// with NO host→device transfer. Used for kernel-write-only scratch (e.g.
    /// the decode-attention `scores` buffer, which the kernel fills during the
    /// softmax pass) so the chain pays a device-local allocation instead of a
    /// host `Vec` alloc + htod of zeros on every call. Mirrors the per-launch
    /// `self.stream.alloc_zeros::<f32>(n)` discipline; exposed so the pipeline
    /// (which can't reach the private `stream`) can use it from a device chain.
    pub(crate) fn alloc_zeros_f32(&self, n: usize) -> Result<CudaSlice<f32>, RuntimeError> {
        self.stream.alloc_zeros::<f32>(n).map_err(|err| {
            RuntimeError::context("allocating zeroed CUDA device scratch buffer", err)
        })
    }

    /// Upload an f64 array (e.g. a RoPE `inv_freq` table) to the device once,
    /// returning a device-resident handle. The attention device chain uploads
    /// `inv_freq` once and shares the device buffer across the Q and K RoPE
    /// launches (the host slice is identical), avoiding a redundant
    /// per-launch htod of the frequency table.
    pub(crate) fn upload_f64(&self, x: &[f64]) -> Result<CudaSlice<f64>, RuntimeError> {
        self.stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading f64 buffer to CUDA", err))
    }

    /// Synchronize the stream and read a device-resident f32 buffer back to the
    /// host. The single sync + readback at the end of a device-resident chain
    // — collapses one sync+dtoh per chained kernel into one for the whole
    // chain.
    pub(crate) fn sync_dtoh_f32(&self, dev: &CudaSlice<f32>) -> Result<Vec<f32>, RuntimeError> {
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA device chain stream", err))?;
        // LARQL-GPU-PROFILE-001: count the sync (the synchronize above) +
        // the dtoh readback of `dev` (the device→host copy that returns the
        // chain result). No-op unless LARQL_GPU_PROFILE=1.
        if crate::options::gpu_profile_enabled() {
            self.note_sync();
            self.note_dtoh(dev.len() * 4);
        }
        self.stream
            .clone_dtoh(dev)
            .map_err(|err| RuntimeError::context("reading CUDA device chain output", err))
    }

    // ── LARQL-GPU-PROFILE-001 runtime-side recorder helpers ─────────────
    //
    // All gate on `gpu_profile_enabled()` so normal decode is a no-op branch.

    /// Record one direct kernel launch. **Suppressed during stream capture**
    /// (B3A review point 8): a launch issued inside `begin_capture`/
    /// `end_capture` becomes a graph node, not a physical execution, so it
    /// must NOT inflate `direct_kernel_submissions` — the build path counts
    /// it once via [`super::CudaBackend::note_graph_captured_nodes`].
    fn note_launch(&self) {
        use std::sync::atomic::Ordering;
        if self.capture_depth.load(Ordering::Relaxed) != 0 {
            return;
        }
        self.profile.launches.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn note_htod(&self, bytes: usize) {
        use std::sync::atomic::Ordering;
        if self.capture_depth.load(Ordering::Relaxed) != 0 {
            return;
        }
        self.profile.htod_copies.fetch_add(1, Ordering::Relaxed);
        self.profile
            .htod_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn note_dtoh(&self, bytes: usize) {
        use std::sync::atomic::Ordering;
        if self.capture_depth.load(Ordering::Relaxed) != 0 {
            return;
        }
        self.profile.dtoh_copies.fetch_add(1, Ordering::Relaxed);
        self.profile
            .dtoh_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// LARQL-GPU-B4 calls this directly when it syncs once for the
    /// fixed-size result readback (it reads two small buffers with one
    /// sync, so it can't use `sync_dtoh_f32`).
    pub(crate) fn note_sync(&self) {
        use std::sync::atomic::Ordering;
        if self.capture_depth.load(Ordering::Relaxed) != 0 {
            return;
        }
        self.profile.syncs.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark the runtime as mid-stream-capture (B3A review point 8). While in
    /// effect, `note_launch`/`note_htod`/`note_dtoh`/`note_sync` are no-ops.
    /// Balanced by [`Self::exit_capture`]; the build path uses [`CaptureGuard`]
    /// so every return path decrements.
    pub(crate) fn enter_capture(&self) {
        self.capture_depth
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// End a capture window started by [`Self::enter_capture`]. Saturating so
    /// an unbalanced call can't underflow.
    pub(crate) fn exit_capture(&self) {
        // fetch_sub with saturating semantics: loop to avoid underflow past 0.
        use std::sync::atomic::Ordering;
        let _ = self
            .capture_depth
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
    }

    /// Consume and reset the runtime profile counters, returning the snapshot
    /// or `None` when nothing was recorded.
    pub(crate) fn take_profile_counters(&self) -> Option<super::RuntimeProfileSnapshot> {
        use std::sync::atomic::Ordering;
        let launches = self.profile.launches.swap(0, Ordering::Relaxed);
        if launches == 0 {
            return None;
        }
        Some(super::RuntimeProfileSnapshot {
            launches,
            htod_copies: self.profile.htod_copies.swap(0, Ordering::Relaxed),
            htod_bytes: self.profile.htod_bytes.swap(0, Ordering::Relaxed),
            dtoh_copies: self.profile.dtoh_copies.swap(0, Ordering::Relaxed),
            dtoh_bytes: self.profile.dtoh_bytes.swap(0, Ordering::Relaxed),
            syncs: self.profile.syncs.swap(0, Ordering::Relaxed),
        })
    }

    /// Amortised Q4_K × f32 matmul, device-resident: `x_dev` is the already-
    /// uploaded `[seq, hidden]` input; returns the `[seq, num_rows]` output as
    /// a device buffer (no sync, no readback). Mirrors `launch_q4k_matmul`'s
    /// shape validation (the weight byte layout must be checked both for safety
    /// and because the weight-cache key is `(ptr, len)`), but skips the input
    /// upload + output readback.
    pub(crate) fn launch_q4k_matmul_dev(
        &self,
        q4k_data: &[u8],
        x_dev: &CudaSlice<f32>,
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        if num_rows == 0 || seq_len == 0 {
            return self
                .stream
                .alloc_zeros::<f32>(seq_len * num_rows)
                .map_err(|err| {
                    RuntimeError::context("allocating CUDA q4k_matmul_dev output", err)
                });
        }
        if x_dev.len() != seq_len * hidden {
            return Err(RuntimeError::usage(format!(
                "q4k_matmul_dev expected x_dev.len() == seq*hidden ({}, {}), got {}",
                seq_len * hidden,
                hidden,
                x_dev.len()
            )));
        }
        if !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q4k_matmul_dev hidden size must be a multiple of 256, got {hidden}"
            )));
        }
        let expected_bytes = num_rows
            .checked_mul(hidden / 256)
            .and_then(|blocks| blocks.checked_mul(144))
            .ok_or_else(|| RuntimeError::usage("q4k_matmul_dev byte-size overflow".to_string()))?;
        if q4k_data.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q4k_matmul_dev expected {expected_bytes} Q4_K bytes for shape ({num_rows}, {hidden}), got {}",
                q4k_data.len()
            )));
        }
        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q4k_data)
            .map_err(|err| RuntimeError::context("uploading Q4_K weights to CUDA", err))?;
        let out_len = seq_len * num_rows;
        let mut out_dev = self.stream.alloc_zeros::<f32>(out_len).map_err(|err| {
            RuntimeError::context("allocating CUDA q4k_matmul_dev output buffer", err)
        })?;
        let threads_x = Q4K_MATMUL_KERNEL.geometry.threads_per_group[0];
        let tiles = (num_rows * seq_len) as u32;
        let cfg = LaunchConfig {
            grid_dim: (tiles.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = num_rows as u32;
        let k = hidden as u32;
        let seq = seq_len as u32;
        let mut launch_args = self.stream.launch_builder(&self.q4k_matmul);
        launch_args
            .arg(&*w_dev)
            .arg(x_dev)
            .arg(&mut out_dev)
            .arg(&n)
            .arg(&k)
            .arg(&seq);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q4k_matmul_dev kernel", err))?;
        self.note_launch();
        Ok(out_dev)
    }

    /// Amortised Q6_K × f32 matmul, device-resident twin of
    /// [`launch_q4k_matmul_dev`].
    pub(crate) fn launch_q6k_matmul_dev(
        &self,
        q6k_data: &[u8],
        x_dev: &CudaSlice<f32>,
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        if num_rows == 0 || seq_len == 0 {
            return self
                .stream
                .alloc_zeros::<f32>(seq_len * num_rows)
                .map_err(|err| {
                    RuntimeError::context("allocating CUDA q6k_matmul_dev output", err)
                });
        }
        if x_dev.len() != seq_len * hidden {
            return Err(RuntimeError::usage(format!(
                "q6k_matmul_dev expected x_dev.len() == seq*hidden ({}, {}), got {}",
                seq_len * hidden,
                hidden,
                x_dev.len()
            )));
        }
        if !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q6k_matmul_dev hidden size must be a multiple of 256, got {hidden}"
            )));
        }
        let expected_bytes = num_rows
            .checked_mul(hidden / 256)
            .and_then(|blocks| blocks.checked_mul(210))
            .ok_or_else(|| RuntimeError::usage("q6k_matmul_dev byte-size overflow".to_string()))?;
        if q6k_data.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q6k_matmul_dev expected {expected_bytes} Q6_K bytes for shape ({num_rows}, {hidden}), got {}",
                q6k_data.len()
            )));
        }
        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q6k_data)
            .map_err(|err| RuntimeError::context("uploading Q6_K weights to CUDA", err))?;
        let out_len = seq_len * num_rows;
        let mut out_dev = self.stream.alloc_zeros::<f32>(out_len).map_err(|err| {
            RuntimeError::context("allocating CUDA q6k_matmul_dev output buffer", err)
        })?;
        let threads_x = Q6K_MATMUL_KERNEL.geometry.threads_per_group[0];
        let tiles = (num_rows * seq_len) as u32;
        let cfg = LaunchConfig {
            grid_dim: (tiles.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = num_rows as u32;
        let k = hidden as u32;
        let seq = seq_len as u32;
        let mut launch_args = self.stream.launch_builder(&self.q6k_matmul);
        launch_args
            .arg(&*w_dev)
            .arg(x_dev)
            .arg(&mut out_dev)
            .arg(&n)
            .arg(&k)
            .arg(&seq);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q6k_matmul_dev kernel", err))?;
        Ok(out_dev)
    }

    /// Q4_K × f32 matvec, device-resident: `x_dev` is the already-uploaded
    /// `[hidden]` input; returns the `[num_rows]` output as a device buffer
    /// (no sync, no readback). Mirrors `launch_q4k_matvec`'s shape validation
    /// (the weight byte layout must be checked both for safety and because the
    /// weight-cache key is `(ptr, len)`), but skips the input upload + output
    /// readback. The decode FFN device chain chains gate/up/down matvecs
    /// through this so an N-matvec chain pays one upload + one readback.
    pub(crate) fn launch_q4k_matvec_dev(
        &self,
        q4k_data: &[u8],
        x_dev: &CudaSlice<f32>,
        num_rows: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        if num_rows == 0 {
            return self.stream.alloc_zeros::<f32>(0).map_err(|err| {
                RuntimeError::context("allocating CUDA q4k_matvec_dev output", err)
            });
        }
        if x_dev.len() != hidden {
            return Err(RuntimeError::usage(format!(
                "q4k_matvec_dev expected x_dev.len() == hidden ({hidden}), got {}",
                x_dev.len()
            )));
        }
        if !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q4k_matvec_dev hidden size must be a multiple of 256, got {hidden}"
            )));
        }
        let expected_bytes = num_rows
            .checked_mul(hidden / 256)
            .and_then(|blocks| blocks.checked_mul(144))
            .ok_or_else(|| RuntimeError::usage("q4k_matvec_dev byte-size overflow".to_string()))?;
        if q4k_data.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q4k_matvec_dev expected {expected_bytes} Q4_K bytes for shape ({num_rows}, {hidden}), got {}",
                q4k_data.len()
            )));
        }
        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q4k_data)
            .map_err(|err| RuntimeError::context("uploading Q4_K weights to CUDA", err))?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(num_rows).map_err(|err| {
            RuntimeError::context("allocating CUDA q4k_matvec_dev output buffer", err)
        })?;
        self.launch_q4k_matvec_into(&self.stream, &w_dev, x_dev, &mut out_dev, num_rows, hidden)?;
        Ok(out_dev)
    }

    /// Q4_K × f32 matvec into a pre-allocated stable output buffer (B3A-4).
    ///
    /// The graph-capture twin of [`launch_q4k_matvec_dev`]: takes an
    /// already-resolved `w_dev` (from the weight cache — resolved before capture
    /// so its device address is stable) and writes into `out_dev` (a stable
    /// graph-owned buffer) instead of allocating a fresh `CudaSlice`. Launches
    /// on `stream` (the capture stream for graph capture, `self.stream` for the
    /// non-graph `_dev` path). Shares the kernel-arg layout + `LaunchConfig`
    /// with `_dev` so the two cannot drift; the `_dev` launcher delegates here
    /// after its `alloc_zeros`.
    pub(crate) fn launch_q4k_matvec_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        w_dev: &std::sync::Arc<CudaSlice<u8>>,
        x_dev: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        num_rows: usize,
        hidden: usize,
    ) -> Result<(), RuntimeError> {
        if num_rows == 0 {
            return Ok(());
        }
        debug_assert_eq!(x_dev.len(), hidden);
        debug_assert!(hidden.is_multiple_of(256));
        debug_assert_eq!(out_dev.len(), num_rows);
        let threads_x = Q4K_MATVEC_KERNEL.geometry.threads_per_group[0];
        let n = num_rows as u32;
        let k = hidden as u32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = stream.launch_builder(&self.q4k_matvec);
        launch_args
            .arg(&**w_dev)
            .arg(x_dev)
            .arg(out_dev)
            .arg(&n)
            .arg(&k);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q4k_matvec_into kernel", err))?;
        self.note_launch();
        Ok(())
    }

    /// Resolve a Q4_K weight matrix through the persistent weight cache,
    /// returning the device-resident handle. Exposed so the graph-build path
    /// (B3A-5) can warm the cache + hold the `Arc` before stream capture begins
    /// (the captured kernel node binds the then-stable device address).
    #[allow(dead_code)]
    pub(crate) fn resolve_q4k_weight(
        &self,
        q4k_data: &[u8],
    ) -> Result<std::sync::Arc<CudaSlice<u8>>, RuntimeError> {
        self.weight_cache
            .get_or_upload_bytes(&self.stream, q4k_data)
            .map_err(|err| RuntimeError::context("uploading Q4_K weights to CUDA", err))
    }

    /// Resolve a dense f32 weight (e.g. the final RMSNorm weight) through
    /// the persistent weight cache (B4-CORRECTION C). Uploads the slice once
    /// on the first call with a given `(ptr, len)` and returns the cached
    /// `Arc<CudaSlice<f32>>` on every subsequent call, so the immutable
    /// final-norm weight is uploaded at most once per generation/vindex
    /// binding instead of every decode token. The cache is flushed at
    /// `reset_kv_cache`, matching the greedy workspace lifetime.
    #[allow(dead_code)]
    pub(crate) fn resolve_f32_weight(
        &self,
        weight: &[f32],
    ) -> Result<std::sync::Arc<CudaSlice<f32>>, RuntimeError> {
        self.weight_cache
            .get_or_upload_f32(&self.stream, weight)
            .map_err(|err| RuntimeError::context("uploading f32 weight to CUDA", err))
    }

    /// Q6_K × f32 matvec, device-resident twin of [`launch_q4k_matvec_dev`].
    pub(crate) fn launch_q6k_matvec_dev(
        &self,
        q6k_data: &[u8],
        x_dev: &CudaSlice<f32>,
        num_rows: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        if num_rows == 0 {
            return self.stream.alloc_zeros::<f32>(0).map_err(|err| {
                RuntimeError::context("allocating CUDA q6k_matvec_dev output", err)
            });
        }
        if x_dev.len() != hidden {
            return Err(RuntimeError::usage(format!(
                "q6k_matvec_dev expected x_dev.len() == hidden ({hidden}), got {}",
                x_dev.len()
            )));
        }
        if !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q6k_matvec_dev hidden size must be a multiple of 256, got {hidden}"
            )));
        }
        // Q6_K super-block = 210 bytes / 256 elements.
        let expected_bytes = num_rows
            .checked_mul(hidden / 256)
            .and_then(|blocks| blocks.checked_mul(210))
            .ok_or_else(|| RuntimeError::usage("q6k_matvec_dev byte-size overflow".to_string()))?;
        if q6k_data.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q6k_matvec_dev expected {expected_bytes} Q6_K bytes for shape ({num_rows}, {hidden}), got {}",
                q6k_data.len()
            )));
        }
        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q6k_data)
            .map_err(|err| RuntimeError::context("uploading Q6_K weights to CUDA", err))?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(num_rows).map_err(|err| {
            RuntimeError::context("allocating CUDA q6k_matvec_dev output buffer", err)
        })?;
        self.launch_q6k_matvec_into(&self.stream, &w_dev, x_dev, &mut out_dev, num_rows, hidden)?;
        Ok(out_dev)
    }

    /// Q6_K × f32 matvec into a pre-allocated stable output buffer (B3A-4).
    /// Graph-capture twin of [`launch_q6k_matvec_dev`]; see
    /// [`launch_q4k_matvec_into`] for the contract.
    pub(crate) fn launch_q6k_matvec_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        w_dev: &std::sync::Arc<CudaSlice<u8>>,
        x_dev: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        num_rows: usize,
        hidden: usize,
    ) -> Result<(), RuntimeError> {
        if num_rows == 0 {
            return Ok(());
        }
        debug_assert_eq!(x_dev.len(), hidden);
        debug_assert!(hidden.is_multiple_of(256));
        debug_assert_eq!(out_dev.len(), num_rows);
        let threads_x = Q6K_MATVEC_KERNEL.geometry.threads_per_group[0];
        let n = num_rows as u32;
        let k = hidden as u32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = stream.launch_builder(&self.q6k_matvec);
        launch_args
            .arg(&**w_dev)
            .arg(x_dev)
            .arg(out_dev)
            .arg(&n)
            .arg(&k);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q6k_matvec_into kernel", err))?;
        self.note_launch();
        Ok(())
    }

    /// Resolve a Q6_K weight matrix through the persistent weight cache (B3A-5).
    #[allow(dead_code)]
    pub(crate) fn resolve_q6k_weight(
        &self,
        q6k_data: &[u8],
    ) -> Result<std::sync::Arc<CudaSlice<u8>>, RuntimeError> {
        self.weight_cache
            .get_or_upload_bytes(&self.stream, q6k_data)
            .map_err(|err| RuntimeError::context("uploading Q6_K weights to CUDA", err))
    }

    /// Device-resident GEGLU-SiLu: `out[i] = silu(gate[i]) * up[i]`. `gate_dev`
    /// / `up_dev` are already-resident `n`-element buffers; returns the
    /// `n`-element output as a device buffer (no sync, no readback).
    pub(crate) fn launch_geglu_silu_dev(
        &self,
        gate_dev: &CudaSlice<f32>,
        up_dev: &CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        self.launch_elementwise_binary_dev(
            &self.geglu_silu,
            gate_dev,
            up_dev,
            None,
            n,
            "geglu_silu",
        )
    }

    /// Device-resident GEGLU-GELU-tanh twin of [`launch_geglu_silu_dev`].
    pub(crate) fn launch_geglu_gelu_tanh_dev(
        &self,
        gate_dev: &CudaSlice<f32>,
        up_dev: &CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        self.launch_elementwise_binary_dev(
            &self.geglu_gelu_tanh,
            gate_dev,
            up_dev,
            None,
            n,
            "geglu_gelu_tanh",
        )
    }

    /// Device-resident standard SiLu: `out[i] = silu(x[i])`.
    pub(crate) fn launch_activation_silu_dev(
        &self,
        input_dev: &CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        self.launch_elementwise_unary_dev(&self.activation_silu, input_dev, n, "activation_silu")
    }

    /// Device-resident standard GELU-tanh twin of
    /// [`launch_activation_silu_dev`].
    pub(crate) fn launch_activation_gelu_tanh_dev(
        &self,
        input_dev: &CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        self.launch_elementwise_unary_dev(
            &self.activation_gelu_tanh,
            input_dev,
            n,
            "activation_gelu_tanh",
        )
    }

    /// Device-resident residual add: `out[i] = h_dev[i] + b_scale * x_dev[i]`.
    /// The device twin of [`launch_residual_add`]: `h_dev` is the residual base
    /// (the pre-block hidden state carried across blocks/layers by GPU-007),
    /// `x_dev` is the new projection output (post-attention O or post-FFN down),
    /// and `b_scale` is the layer's `residual_multiplier`. No upload, no sync,
    /// no readback — the caller chains it on the same stream and reads back
    /// once. Routes through the same `residual_add` kernel + arg layout
    /// (`extra_scalar = Some(b_scale)`) the host-readback launcher uses, so the
    /// two can't drift on the kernel contract. Used by the cross-layer
    /// hidden-state residency chain (GPU-007C/D).
    pub(crate) fn launch_residual_add_dev(
        &self,
        h_dev: &CudaSlice<f32>,
        x_dev: &CudaSlice<f32>,
        n: usize,
        b_scale: f32,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        self.launch_elementwise_binary_dev(
            &self.residual_add,
            h_dev,
            x_dev,
            Some(b_scale),
            n,
            "residual_add",
        )
    }

    // ── B3A-4: into-buffer graph-capture twins of the elementwise launchers ──
    //
    // Each writes into a pre-allocated `&mut CudaSlice<f32>` (a stable
    // graph-owned buffer) instead of allocating a fresh output. The graph
    // build path (B3A-5) calls these during stream capture so the captured
    // kernel nodes bind stable buffer addresses. They delegate to the shared
    // `launch_elementwise_*_into` cores, so the arg layout + LaunchConfig
    // cannot drift from the `_dev` twins.

    /// GEGLU-SiLu into a stable buffer: `out[i] = silu(gate[i]) * up[i]`.
    #[allow(dead_code)]
    pub(crate) fn launch_geglu_silu_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        gate_dev: &CudaSlice<f32>,
        up_dev: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), RuntimeError> {
        self.launch_elementwise_binary_into(
            stream,
            &self.geglu_silu,
            gate_dev,
            up_dev,
            out_dev,
            None,
            n,
            "geglu_silu",
        )
    }

    /// GEGLU-GELU-tanh into a stable buffer.
    #[allow(dead_code)]
    pub(crate) fn launch_geglu_gelu_tanh_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        gate_dev: &CudaSlice<f32>,
        up_dev: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), RuntimeError> {
        self.launch_elementwise_binary_into(
            stream,
            &self.geglu_gelu_tanh,
            gate_dev,
            up_dev,
            out_dev,
            None,
            n,
            "geglu_gelu_tanh",
        )
    }

    /// Standard SiLU into a stable buffer: `out[i] = silu(x[i])`.
    #[allow(dead_code)]
    pub(crate) fn launch_activation_silu_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        input_dev: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), RuntimeError> {
        self.launch_elementwise_unary_into(
            stream,
            &self.activation_silu,
            input_dev,
            out_dev,
            n,
            "activation_silu",
        )
    }

    /// Standard GELU-tanh into a stable buffer.
    #[allow(dead_code)]
    pub(crate) fn launch_activation_gelu_tanh_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        input_dev: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), RuntimeError> {
        self.launch_elementwise_unary_into(
            stream,
            &self.activation_gelu_tanh,
            input_dev,
            out_dev,
            n,
            "activation_gelu_tanh",
        )
    }

    /// Residual add into a stable buffer: `out[i] = h_dev[i] + b_scale * x_dev[i]`.
    /// Used by both the FFN graph (final residual into the arena's stable slot)
    /// and — when the arena is active — the resident-attention block's final
    /// residual into the stable post-attn buffer (B3A-3 ping-pong write-point).
    #[allow(dead_code)]
    pub(crate) fn launch_residual_add_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        h_dev: &CudaSlice<f32>,
        x_dev: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        n: usize,
        b_scale: f32,
    ) -> Result<(), RuntimeError> {
        self.launch_elementwise_binary_into(
            stream,
            &self.residual_add,
            h_dev,
            x_dev,
            out_dev,
            Some(b_scale),
            n,
            "residual_add",
        )
    }

    /// In-place scaled residual add on `buf`: `buf[i] = buf[i] + b_scale * x[i]`,
    /// one thread per element. LARQL-GPU-B3B single-stream: writes attention's
    /// post-attn residual directly into the arena input slot the FFN graph
    /// reads, removing the per-layer D2D seed copy the two-stream B3A design
    /// paid (`cap_stream.memcpy_dtod`).
    ///
    /// `buf` is bound as BOTH the residual base (kernel arg 0, read) and the
    /// output (arg 2, write) via two shared `&CudaSlice` borrows. This is
    /// numerically sound: the `residual_add` kernel is element-wise independent
    /// (`out[i] = h[i] + b_scale*x[i]`), so `h == out` is a well-defined
    /// in-place add. It is also sound at the API level: cudarc's
    /// `PushKernelArg for &CudaSlice` pushes only the `cu_device_ptr` (and,
    /// when event tracking is on, records an event); with event tracking
    /// disabled context-wide (B3B init) no `cuStreamWaitEvent` is injected,
    /// and single-stream execution guarantees no concurrent access to `buf`.
    ///
    /// The caller holds the arena slot by a shared borrow for the duration of
    /// the in-place add (the device write happens through the `unsafe` kernel
    /// launch, not through a Rust `&mut`), so the post-add contents are visible
    /// to the next same-stream operation (the FFN graph capture/replay).
    pub(crate) fn launch_residual_add_inplace_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        buf: &CudaSlice<f32>,
        x_dev: &CudaSlice<f32>,
        n: usize,
        b_scale: f32,
    ) -> Result<(), RuntimeError> {
        if n == 0 {
            return Ok(());
        }
        debug_assert_eq!(buf.len(), n);
        debug_assert_eq!(x_dev.len(), n);
        let threads_x = GEGGLU_SILU_KERNEL.geometry.threads_per_group[0];
        let n_u = n as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        // Bind `buf` twice (as `h` and as `out`); `x_dev` once; then the scalar
        // + length — the exact arg layout of `launch_elementwise_binary_into`
        // (the `_dev`/host-readback path) so the kernel sees identical args.
        let mut launch_args = stream.launch_builder(&self.residual_add);
        launch_args
            .arg(buf)
            .arg(x_dev)
            .arg(buf)
            .arg(&b_scale)
            .arg(&n_u);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA residual_add (in-place)", err))?;
        self.note_launch();
        Ok(())
    }
    /// already on the device; the output is allocated and returned as a device
    /// buffer. No upload, no sync, no readback — the caller drives the chain
    /// and reads back once via [`sync_dtoh_f32`]. Mirrors the arg layout of
    /// [`launch_elementwise_binary`] so the device kernel sees identical args.
    fn launch_elementwise_binary_dev(
        &self,
        func: &CudaFunction,
        in_a: &CudaSlice<f32>,
        in_b: &CudaSlice<f32>,
        extra_scalar: Option<f32>,
        n: usize,
        ctx: &'static str,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        if n == 0 {
            return self.stream.alloc_zeros::<f32>(0).map_err(|err| {
                RuntimeError::context_concat("allocating CUDA ", ctx, " output", err)
            });
        }
        if in_a.len() != n || in_b.len() != n {
            return Err(RuntimeError::usage(format!(
                "{ctx}_dev expected in_a/in_b of length {n}, got {} / {}",
                in_a.len(),
                in_b.len()
            )));
        }
        if n > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "{ctx}_dev length {n} exceeds the 32-bit kernel index limit"
            )));
        }
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|err| RuntimeError::context_concat("allocating CUDA ", ctx, " output", err))?;
        self.launch_elementwise_binary_into(
            &self.stream,
            func,
            in_a,
            in_b,
            &mut out_dev,
            extra_scalar,
            n,
            ctx,
        )?;
        Ok(out_dev)
    }

    /// Binary elementwise into a pre-allocated stable output buffer (B3A-4).
    ///
    /// The graph-capture core: writes into `out_dev` (a stable graph-owned
    /// buffer) instead of allocating. Launches on `stream`. Shares the kernel-
    /// arg layout + `LaunchConfig` with the `_dev` twin so the two cannot drift;
    /// the `_dev` launcher delegates here after its `alloc_zeros`.
    #[allow(clippy::too_many_arguments)]
    fn launch_elementwise_binary_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        func: &CudaFunction,
        in_a: &CudaSlice<f32>,
        in_b: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        extra_scalar: Option<f32>,
        n: usize,
        ctx: &'static str,
    ) -> Result<(), RuntimeError> {
        if n == 0 {
            return Ok(());
        }
        debug_assert_eq!(in_a.len(), n);
        debug_assert_eq!(in_b.len(), n);
        debug_assert_eq!(out_dev.len(), n);
        let threads_x = GEGGLU_SILU_KERNEL.geometry.threads_per_group[0];
        let n_u = n as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = stream.launch_builder(func);
        launch_args.arg(in_a).arg(in_b).arg(out_dev);
        if let Some(ref scalar) = extra_scalar {
            launch_args.arg(scalar);
        }
        launch_args.arg(&n_u);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context_concat("launching CUDA ", ctx, " kernel", err))?;
        self.note_launch();
        Ok(())
    }

    /// Shared device-resident unary elementwise dispatch. Mirrors the arg layout
    /// of [`launch_elementwise_unary`] (single input → output), but the input is
    /// already resident and the output stays resident (no upload/sync/readback).
    fn launch_elementwise_unary_dev(
        &self,
        func: &CudaFunction,
        input: &CudaSlice<f32>,
        n: usize,
        ctx: &'static str,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        if n == 0 {
            return self.stream.alloc_zeros::<f32>(0).map_err(|err| {
                RuntimeError::context_concat("allocating CUDA ", ctx, " output", err)
            });
        }
        if input.len() != n {
            return Err(RuntimeError::usage(format!(
                "{ctx}_dev expected input of length {n}, got {}",
                input.len()
            )));
        }
        if n > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "{ctx}_dev length {n} exceeds the 32-bit kernel index limit"
            )));
        }
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|err| RuntimeError::context_concat("allocating CUDA ", ctx, " output", err))?;
        self.launch_elementwise_unary_into(&self.stream, func, input, &mut out_dev, n, ctx)?;
        Ok(out_dev)
    }

    /// Unary elementwise into a pre-allocated stable output buffer (B3A-4).
    /// Graph-capture core; see [`launch_elementwise_binary_into`].
    fn launch_elementwise_unary_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        func: &CudaFunction,
        input: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        n: usize,
        ctx: &'static str,
    ) -> Result<(), RuntimeError> {
        if n == 0 {
            return Ok(());
        }
        debug_assert_eq!(input.len(), n);
        debug_assert_eq!(out_dev.len(), n);
        let threads_x = ACTIVATION_SILU_KERNEL.geometry.threads_per_group[0];
        let n_u = n as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = stream.launch_builder(func);
        launch_args.arg(input).arg(out_dev).arg(&n_u);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context_concat("launching CUDA ", ctx, " kernel", err))?;
        self.note_launch();
        Ok(())
    }

    // ── device-resident norm / RoPE / attention launch variants ──────────
    //
    // These extend the per-projection round-trip collapse (Sessions 20/21
    // covered the FFN block) into the **attention** block: the norm →
    // Q/K/V → QK-norm → RoPE → attention → O chain. Like the matmul/matvec
    // /activation `_dev` twins, they take an input already on the device and
    // return a device-resident output with no internal upload/sync/readback,
    // so a caller chains several on one stream and reads back once.

    /// Device-resident body RMSNorm: `x_dev` is the already-uploaded
    /// `[rows*cols]` row-major input; `weight` is a host `[cols]` slice (or
    /// `None` for the parameter-free `w = 1.0` path, uploaded as a one-element
    /// placeholder). Returns the `[rows*cols]` output as a device buffer (no
    /// sync, no readback). Owns the kernel-arg layout + `LaunchConfig` so the
    /// host-readback [`launch_rms_norm`] delegates here and the two can't
    /// drift on arg order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_rms_norm_dev(
        &self,
        x_dev: &CudaSlice<f32>,
        weight: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f64,
        offset: f32,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        if rows == 0 || cols == 0 {
            return self
                .stream
                .alloc_zeros::<f32>(0)
                .map_err(|err| RuntimeError::context("allocating CUDA rms_norm_dev output", err));
        }
        if x_dev.len() != rows * cols {
            return Err(RuntimeError::usage(format!(
                "rms_norm_dev expected x_dev.len() == rows*cols ({rows}*{cols}={}), got {}",
                rows * cols,
                x_dev.len()
            )));
        }
        if let Some(w) = weight {
            if w.len() != cols {
                return Err(RuntimeError::usage(format!(
                    "rms_norm_dev expected weight of length {cols}, got {}",
                    w.len()
                )));
            }
        }
        if rows > u32::MAX as usize || cols > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "rms_norm_dev shape ({rows}, {cols}) exceeds the 32-bit kernel index limit"
            )));
        }
        // Upload the weight without a per-call heap alloc: the `Some` arm
        // uploads the caller's slice directly (clone_htod takes &[f32]); the
        // `None` arm uploads a one-element placeholder and flags
        // `has_weight = 0` so the device ignores the pointer (cudarc requires
        // a non-empty host slice for clone_htod).
        let placeholder = [0.0f32];
        let (weight_slice, has_weight): (&[f32], i32) = match weight {
            Some(w) => (w, 1),
            None => (&placeholder[..], 0),
        };
        let weight_dev = self
            .stream
            .clone_htod(weight_slice)
            .map_err(|err| RuntimeError::context("uploading rms_norm_dev weight to CUDA", err))?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(rows * cols).map_err(|err| {
            RuntimeError::context("allocating CUDA rms_norm_dev output buffer", err)
        })?;
        let block_dim = 1024u32;
        let rows_u = rows as u32;
        let cols_u = cols as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows_u, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.rms_norm);
        launch_args
            .arg(x_dev)
            .arg(&weight_dev)
            .arg(&mut out_dev)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&eps)
            .arg(&offset)
            .arg(&has_weight);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA rms_norm_dev kernel", err))?;
        self.note_launch();
        Ok(out_dev)
    }

    /// LARQL-GPU-B4: partial top-K reduction over the logical-vocabulary
    /// score buffer. Grid: `num_blocks` blocks of `GREEDY_BLOCK_SIZE`
    /// threads. Each block writes `k` `(score, id)` pairs into
    /// `partial_scores/partial_ids` at offset `block_idx * k`.
    pub(crate) fn launch_greedy_topk_partial(
        &self,
        scores_dev: &CudaSlice<f32>,
        partial_scores_dev: &mut CudaSlice<f32>,
        partial_ids_dev: &mut CudaSlice<u32>,
        logical_rows: usize,
        num_blocks: usize,
        k: usize,
    ) -> Result<(), RuntimeError> {
        if logical_rows == 0 {
            return Ok(());
        }
        let k = k.min(crate::ops::GREEDY_MAX_K);
        let partial_len = num_blocks.checked_mul(k).ok_or_else(|| {
            RuntimeError::usage("greedy_topk_partial partial_len overflow".to_string())
        })?;
        if scores_dev.len() < logical_rows
            || partial_scores_dev.len() < partial_len
            || partial_ids_dev.len() < partial_len
        {
            return Err(RuntimeError::usage(format!(
                "greedy_topk_partial buffer too small: scores>={logical_rows} got {}, partials>={partial_len} got {}/{}",
                scores_dev.len(),
                partial_scores_dev.len(),
                partial_ids_dev.len()
            )));
        }
        let logical_u = logical_rows as u32;
        let k_u = k as u32;
        let threads = crate::ops::GREEDY_BLOCK_SIZE as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.greedy_topk_partial);
        launch_args
            .arg(scores_dev)
            .arg(partial_scores_dev)
            .arg(partial_ids_dev)
            .arg(&logical_u)
            .arg(&k_u);
        unsafe { launch_args.launch(cfg) }.map_err(|err| {
            RuntimeError::context("launching CUDA greedy_topk_partial kernel", err)
        })?;
        self.note_launch();
        Ok(())
    }

    /// LARQL-GPU-B4: final top-K reduction over the per-block partial
    /// candidates. Single block, `GREEDY_BLOCK_SIZE` threads. Writes the
    /// final sorted (descending) top-`k` into `result_scores/result_ids`.
    pub(crate) fn launch_greedy_topk_final(
        &self,
        partial_scores_dev: &mut CudaSlice<f32>,
        partial_ids_dev: &CudaSlice<u32>,
        result_scores_dev: &mut CudaSlice<f32>,
        result_ids_dev: &mut CudaSlice<u32>,
        num_partials: usize,
        k: usize,
    ) -> Result<(), RuntimeError> {
        let k = k.min(crate::ops::GREEDY_MAX_K);
        if partial_scores_dev.len() < num_partials
            || partial_ids_dev.len() < num_partials
            || result_scores_dev.len() < k
            || result_ids_dev.len() < k
        {
            return Err(RuntimeError::usage(format!(
                "greedy_topk_final buffer too small: partials>={num_partials} got {}/{}, result>={k} got {}/{}",
                partial_scores_dev.len(),
                partial_ids_dev.len(),
                result_scores_dev.len(),
                result_ids_dev.len()
            )));
        }
        let num_partials_u = num_partials as u32;
        let k_u = k as u32;
        let threads = crate::ops::GREEDY_BLOCK_SIZE as u32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.greedy_topk_final);
        launch_args
            .arg(partial_scores_dev)
            .arg(partial_ids_dev)
            .arg(result_scores_dev)
            .arg(result_ids_dev)
            .arg(&num_partials_u)
            .arg(&k_u);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA greedy_topk_final kernel", err))?;
        self.note_launch();
        Ok(())
    }

    /// Upload an RMSNorm weight vector once, returning a stable device handle
    /// (B3A-4/B3A-5). The `_dev` launcher re-uploads the weight per call via
    /// `clone_htod`; the graph path must hold a stable-address weight buffer for
    /// the captured graph's lifetime, so the graph-build path uploads once and
    /// passes the handle to [`launch_rms_norm_into`].
    ///
    /// `Some(w)` uploads the `[cols]` weight; `None` uploads the one-element
    /// placeholder used by the parameter-free path (the kernel's `has_weight=0`
    /// flag makes it ignore the pointer, but cudarc requires a non-empty slice).
    #[allow(dead_code)]
    pub(crate) fn upload_rms_norm_weight(
        &self,
        weight: Option<&[f32]>,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        let placeholder = [0.0f32];
        let slice: &[f32] = match weight {
            Some(w) => w,
            None => &placeholder[..],
        };
        self.stream
            .clone_htod(slice)
            .map_err(|err| RuntimeError::context("uploading rms_norm weight to CUDA", err))
    }

    /// RMSNorm into a pre-allocated stable output buffer with a device-resident
    /// weight (B3A-4 / B4-CORRECTION C). `weight_dev` is a pre-uploaded stable
    /// weight buffer (resolved through the f32 weight cache — see
    /// [`resolve_f32_weight`] — or [`upload_rms_norm_weight`]), and the output
    /// writes into `out_dev` (a stable graph-owned buffer, or the greedy-head
    /// workspace's `normed_hidden`) instead of allocating. `has_weight` is `1`
    /// when a real weight was uploaded, `0` for the parameter-free placeholder
    /// path. Launches on `stream`. Shares the kernel-arg layout + `LaunchConfig`
    /// with [`launch_rms_norm_dev`]. Used by both the B3A graph-capture FFN path
    /// and the B4 device-greedy final-norm step.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_rms_norm_into(
        &self,
        stream: &cudarc::driver::CudaStream,
        x_dev: &CudaSlice<f32>,
        weight_dev: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        eps: f64,
        offset: f32,
        has_weight: i32,
    ) -> Result<(), RuntimeError> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        debug_assert_eq!(x_dev.len(), rows * cols);
        debug_assert_eq!(out_dev.len(), rows * cols);
        let block_dim = 1024u32;
        let rows_u = rows as u32;
        let cols_u = cols as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows_u, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = stream.launch_builder(&self.rms_norm);
        launch_args
            .arg(x_dev)
            .arg(weight_dev)
            .arg(out_dev)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&eps)
            .arg(&offset)
            .arg(&has_weight);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA rms_norm_into kernel", err))?;
        self.note_launch();
        Ok(())
    }

    /// Device-resident per-head RMSNorm twin of [`launch_rms_norm_dev`].
    /// `x_dev` is the `[seq*num_heads*head_dim]` row-major input; `weight` is a
    /// host `[head_dim]` slice broadcast across heads (or `None` for the
    /// parameter-free path). Returns the output as a device buffer. Owns the
    /// kernel-arg layout so [`launch_rms_norm_heads`] delegates here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_rms_norm_heads_dev(
        &self,
        x_dev: &CudaSlice<f32>,
        weight: Option<&[f32]>,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f64,
        offset: f32,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        let total = seq_len
            .checked_mul(num_heads)
            .and_then(|p| p.checked_mul(head_dim));
        let total = match total {
            Some(t) if x_dev.len() == t => t,
            _ => {
                return Err(RuntimeError::usage(format!(
                    "rms_norm_heads_dev expected x_dev of length {} (seq={seq_len} heads={num_heads} dim={head_dim}), got {}",
                    seq_len * num_heads * head_dim,
                    x_dev.len()
                )))
            }
        };
        if total == 0 {
            return self.stream.alloc_zeros::<f32>(0).map_err(|err| {
                RuntimeError::context("allocating CUDA rms_norm_heads_dev output", err)
            });
        }
        if let Some(w) = weight {
            if w.len() != head_dim {
                return Err(RuntimeError::usage(format!(
                    "rms_norm_heads_dev expected weight of length {head_dim} (broadcast across heads), got {}",
                    w.len()
                )));
            }
        }
        if seq_len > u32::MAX as usize
            || num_heads > u32::MAX as usize
            || head_dim > u32::MAX as usize
            || (num_heads as u64) * (head_dim as u64) > u32::MAX as u64
        {
            return Err(RuntimeError::usage(format!(
                "rms_norm_heads_dev shape (seq={seq_len}, heads={num_heads}, dim={head_dim}) exceeds the 32-bit kernel index limit"
            )));
        }
        let placeholder = [0.0f32];
        let (weight_slice, has_weight): (&[f32], i32) = match weight {
            Some(w) => (w, 1),
            None => (&placeholder[..], 0),
        };
        let weight_dev = self.stream.clone_htod(weight_slice).map_err(|err| {
            RuntimeError::context("uploading rms_norm_heads_dev weight to CUDA", err)
        })?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(total).map_err(|err| {
            RuntimeError::context("allocating CUDA rms_norm_heads_dev output buffer", err)
        })?;
        let block_dim = 1024u32;
        let seq_u = seq_len as u32;
        let heads_u = num_heads as u32;
        let dim_u = head_dim as u32;
        let blocks = (seq_u.checked_mul(heads_u)).ok_or_else(|| {
            RuntimeError::usage("rms_norm_heads_dev grid dim overflow seq*heads".to_string())
        })?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.rms_norm_heads);
        launch_args
            .arg(x_dev)
            .arg(&weight_dev)
            .arg(&mut out_dev)
            .arg(&seq_u)
            .arg(&heads_u)
            .arg(&dim_u)
            .arg(&eps)
            .arg(&offset)
            .arg(&has_weight);
        unsafe { launch_args.launch(cfg) }.map_err(|err| {
            RuntimeError::context("launching CUDA rms_norm_heads_dev kernel", err)
        })?;
        self.note_launch();
        Ok(out_dev)
    }

    /// Device-resident RoPE: `x_dev` is the already-uploaded
    /// `[seq*num_heads*head_dim]` Q/K tensor; `inv_freq` is the host
    /// `half_rotary`-length frequency array (built identically to the
    /// reference). Uploads `inv_freq` once then delegates to
    /// [`launch_rope_dev_with_invfreq`] (single source of truth for the
    /// kernel-arg layout). Callers that share `inv_freq` across multiple RoPE
    /// launches (the attention device chain's Q + K) should upload it once via
    /// [`upload_f64`] and call [`launch_rope_dev_with_invfreq`] directly.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_rope_dev(
        &self,
        x_dev: &CudaSlice<f32>,
        inv_freq: &[f64],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        half_rotary: usize,
        position_offset: usize,
        position_divisor: f64,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        let inv_freq_dev = self
            .stream
            .clone_htod(inv_freq)
            .map_err(|err| RuntimeError::context("uploading rope_dev inv_freq to CUDA", err))?;
        self.launch_rope_dev_with_invfreq(
            &inv_freq_dev,
            x_dev,
            seq_len,
            num_heads,
            head_dim,
            half_rotary,
            position_offset,
            position_divisor,
        )
    }

    /// Device-resident RoPE with an already-uploaded `inv_freq` device buffer
    /// — the twin of [`launch_rope_dev`] for chains that reuse the frequency
    /// table across multiple RoPE launches (the prefill attention chain's Q +
    /// K share one `inv_freq`). Owns the kernel-arg layout + `LaunchConfig` so
    /// [`launch_rope`] and [`launch_rope_dev`] delegate here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_rope_dev_with_invfreq(
        &self,
        inv_freq_dev: &CudaSlice<f64>,
        x_dev: &CudaSlice<f32>,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        half_rotary: usize,
        position_offset: usize,
        position_divisor: f64,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        let total = seq_len
            .checked_mul(num_heads)
            .and_then(|p| p.checked_mul(head_dim));
        let total = match total {
            Some(t) if x_dev.len() == t => t,
            _ => {
                return Err(RuntimeError::usage(format!(
                    "rope_dev expected x_dev of length {} (seq={seq_len} heads={num_heads} dim={head_dim}), got {}",
                    seq_len * num_heads * head_dim,
                    x_dev.len()
                )))
            }
        };
        if total == 0 {
            return self
                .stream
                .alloc_zeros::<f32>(0)
                .map_err(|err| RuntimeError::context("allocating CUDA rope_dev output", err));
        }
        if inv_freq_dev.len() != half_rotary {
            return Err(RuntimeError::usage(format!(
                "rope_dev expected inv_freq of length {half_rotary}, got {}",
                inv_freq_dev.len()
            )));
        }
        if half_rotary == 0 {
            return Err(RuntimeError::usage(
                "rope_dev requires half_rotary >= 1 (rotary_dim >= 2)".to_string(),
            ));
        }
        if total > u32::MAX as usize
            || seq_len > u32::MAX as usize
            || num_heads > u32::MAX as usize
            || head_dim > u32::MAX as usize
        {
            return Err(RuntimeError::usage(format!(
                "rope_dev shape (seq={seq_len}, heads={num_heads}, dim={head_dim}) exceeds the 32-bit kernel index limit"
            )));
        }
        let divisor = if position_divisor > 0.0 {
            position_divisor
        } else {
            1.0
        };
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(total)
            .map_err(|err| RuntimeError::context("allocating CUDA rope_dev output buffer", err))?;
        let threads_x = ROPE_KERNEL.geometry.threads_per_group[0];
        let total_u = total as u32;
        let cfg = LaunchConfig {
            grid_dim: (total_u.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_u = seq_len as u32;
        let heads_u = num_heads as u32;
        let dim_u = head_dim as u32;
        let half_u = half_rotary as u32;
        let pos_off = position_offset as f64;
        let n_u64 = total as u64;
        let mut launch_args = self.stream.launch_builder(&self.rope);
        launch_args
            .arg(x_dev)
            .arg(inv_freq_dev)
            .arg(&mut out_dev)
            .arg(&seq_u)
            .arg(&heads_u)
            .arg(&dim_u)
            .arg(&half_u)
            .arg(&pos_off)
            .arg(&divisor)
            .arg(&n_u64);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA rope_dev kernel", err))?;
        self.note_launch();
        Ok(out_dev)
    }

    /// Device-resident fused prefill (seq×seq) causal GQA attention.
    /// `q_dev`/`k_dev`/`v_dev` are already-resident: `q` is
    /// `[seq, num_q*head_dim]`, `k`/`v` are `[seq, kv_dim]`. Returns the
    /// `[seq, num_q*head_dim]` output as a device buffer. Owns the kernel-arg
    /// layout + shared-mem budget guard so [`launch_prefill_attention`]
    /// delegates here. `Err` (mapped to `None` by the pipeline) when the shape
    /// exceeds the device shared-mem/index budget.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_prefill_attention_dev(
        &self,
        q_dev: &CudaSlice<f32>,
        k_dev: &CudaSlice<f32>,
        v_dev: &CudaSlice<f32>,
        scale: f32,
        softcap: Option<f32>,
        num_q: usize,
        head_dim: usize,
        kv_dim: usize,
        reps: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f32>, RuntimeError> {
        let q_dim = num_q.checked_mul(head_dim);
        let q_len = seq_len.checked_mul(q_dim.unwrap_or(usize::MAX));
        let kv_len = seq_len.checked_mul(kv_dim);
        let (q_len, kv_len) = match (q_len, kv_len) {
            (Some(ql), Some(kvl)) if q_dev.len() == ql && k_dev.len() == kvl && v_dev.len() == kvl => {
                (ql, kvl)
            }
            _ => {
                return Err(RuntimeError::usage(format!(
                    "prefill_attention_dev shape mismatch: q={} (expected {q_len:?}), k={} / v={} (expected {kv_len:?}, kv_dim={kv_dim}, seq={seq_len}), num_q={num_q}, head_dim={head_dim}",
                    q_dev.len(),
                    k_dev.len(),
                    v_dev.len(),
                )))
            }
        };
        if seq_len == 0 {
            return self.stream.alloc_zeros::<f32>(0).map_err(|err| {
                RuntimeError::context("allocating CUDA prefill_attention_dev output", err)
            });
        }
        if reps == 0 {
            return Err(RuntimeError::usage(
                "prefill_attention_dev requires reps >= 1 (num_kv heads > 0)".to_string(),
            ));
        }
        if q_len > u32::MAX as usize
            || kv_len > u32::MAX as usize
            || num_q > u32::MAX as usize
            || head_dim > u32::MAX as usize
            || kv_dim > u32::MAX as usize
            || seq_len > u32::MAX as usize
        {
            return Err(RuntimeError::usage(format!(
                "prefill_attention_dev shape (num_q={num_q}, head_dim={head_dim}, kv_dim={kv_dim}, seq_len={seq_len}) exceeds the 32-bit kernel index limit"
            )));
        }
        let shm_fixed = 256 * 8 + 256 * 4;
        let shm_scores = seq_len.checked_mul(4);
        let shared_mem_bytes = match shm_scores {
            Some(ss) if shm_fixed + ss <= 48 * 1024 => shm_fixed + ss,
            _ => {
                return Err(RuntimeError::usage(format!(
                    "prefill_attention_dev seq_len={seq_len} exceeds the 48 KB dynamic shared-mem budget; fall back to host"
                )));
            }
        };
        let mut out_dev = self.stream.alloc_zeros::<f32>(q_len).map_err(|err| {
            RuntimeError::context("allocating CUDA prefill_attention_dev output", err)
        })?;
        let threads = PREFILL_ATTENTION_KERNEL.geometry.threads_per_group[0];
        let cfg = LaunchConfig {
            grid_dim: (num_q as u32, seq_len as u32, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };
        let (softcap_val, has_softcap) = match softcap {
            Some(cap) => (cap, 1u32),
            None => (0.0f32, 0u32),
        };
        let num_q_u = num_q as u32;
        let head_dim_u = head_dim as u32;
        let kv_dim_u = kv_dim as u32;
        let reps_u = reps as u32;
        let seq_len_u = seq_len as u32;
        let mut launch_args = self.stream.launch_builder(&self.prefill_attention);
        launch_args
            .arg(q_dev)
            .arg(k_dev)
            .arg(v_dev)
            .arg(&mut out_dev)
            .arg(&scale)
            .arg(&softcap_val)
            .arg(&has_softcap)
            .arg(&num_q_u)
            .arg(&head_dim_u)
            .arg(&kv_dim_u)
            .arg(&reps_u)
            .arg(&seq_len_u);
        unsafe { launch_args.launch(cfg) }.map_err(|err| {
            RuntimeError::context("launching CUDA prefill_attention_dev kernel", err)
        })?;
        Ok(out_dev)
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Compile the combined CUDA kernel module via NVRTC, caching the resulting
/// PTX on disk so a warm process start skips the (hundreds-of-ms) NVRTC
/// round-trip.
///
/// Cache reads/writes are best-effort: a miss, corrupt entry, or I/O error
/// transparently falls back to a fresh compile, so the cache can never make
/// CUDA unavailable. The PTX is keyed on the source text, the target arch, the
/// `fmad` policy, and a cache-format/cudarc version component (see
/// Discover directories NVRTC should search for CUDA headers (e.g.
/// `cuda_fp16.h`). Checks `$CUDA_HOME/include`, the Debian/Ubuntu
/// `/usr/include` (where `nvidia-cuda-dev` places headers), and the
/// conventional `/usr/local/cuda/include`. Returns only directories that
/// actually exist and contain `cuda_fp16.h`.
fn cuda_include_paths() -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        candidates.push(format!("{cuda_home}/include"));
    }
    candidates.push("/usr/local/cuda/include".to_string());
    candidates.push("/usr/include".to_string());
    candidates
        .into_iter()
        .filter(|p| std::path::Path::new(&format!("{p}/cuda_fp16.h")).exists())
        .collect()
}

/// [`crate::ptx_cache`]). After a successful compile the PTX text is written
/// atomically (temp file + rename) so a crash can't leave a corrupt entry.
fn compile_or_load_module(
    context: &Arc<CudaContext>,
    src: &str,
    arch: &'static str,
    fmad: bool,
) -> Result<Arc<CudaModule>, RuntimeError> {
    let key = ptx_cache::cache_key(src, arch, fmad);
    // Cache hit: load the cached PTX text directly (the driver JITs PTX→SASS).
    // A load failure on a cached entry (corrupt/empty `.ptx`) falls through to
    // a fresh compile rather than failing CUDA init.
    if let Some(cached) = ptx_cache::try_read(&key) {
        match context.load_module(Ptx::from_src(cached)) {
            Ok(module) => return Ok(module),
            Err(_) => { /* fall through to recompile */ }
        }
    }
    let include_paths = cuda_include_paths();
    let opts = CompileOptions {
        fmad: Some(fmad),
        arch: Some(arch),
        include_paths,
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(src, opts)
        .map_err(|err| RuntimeError::compile("compiling CUDA k-quant NVRTC module", err))?;
    // `compile_ptx_with_opts` returns a PTX image; serialise it to text for
    // the cache and reload from that text so the on-disk format and the live
    // load path are identical.
    let ptx_src = ptx.to_src();
    ptx_cache::try_write(&key, &ptx_src);
    context
        .load_module(Ptx::from_src(ptx_src))
        .map_err(|err| RuntimeError::context("loading CUDA k-quant module", err))
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeError {
    message: String,
}

impl RuntimeError {
    pub(crate) fn context(action: &'static str, source: DriverError) -> Self {
        Self {
            message: format!("{action}: {source}"),
        }
    }

    /// Compose a context string from a prefix + a kernel/operation name + a
    /// suffix, then attach a `DriverError`. Used by the shared elementwise
    /// launchers so each kernel gets a distinct, actionable error string
    /// without a per-kernel `context` method.
    fn context_concat(
        prefix: &'static str,
        name: &'static str,
        suffix: &'static str,
        source: DriverError,
    ) -> Self {
        Self {
            message: format!("{prefix}{name}{suffix}: {source}"),
        }
    }

    fn compile(action: &'static str, source: CompileError) -> Self {
        Self {
            message: format!("{action}: {source}"),
        }
    }

    pub(crate) fn usage(message: String) -> Self {
        Self { message }
    }
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RuntimeError {}

#[cfg(test)]
mod b3b_single_stream_tests {
    //! LARQL-GPU-B3B: single-stream CUDA-Graph capture/replay lifecycle tests.
    //!
    //! B3B moved graph capture + replay onto the runtime's single non-NULL
    //! stream (replacing B3A's separate `cap_stream`). These tests validate the
    //! full lifecycle on **that one stream** — the exact configuration the
    //! production resident-FFN graph path now uses:
    //!
    //! - stream capture on the runtime's non-NULL stream (`rt.stream`)
    //! - `CudaGraph::launch()` replay landing on the same stream
    //! - **stable-pointer replay**: mutating a captured buffer's *contents*
    //!   (same device address) changes the replay output — the core invariant
    //!   the resident-FFN ping-pong arena depends on
    //! - the **in-place residual add** primitive (`launch_residual_add_inplace_into`)
    //!   that binds one buffer as both read and write — numerically sound on a
    //!   single stream and the mechanism that removes the per-layer D2D seed copy
    //! - graph instantiate with the cudarc-forced `AUTO_FREE_ON_LAUNCH` flag
    //! - clean teardown (exec graph → captured graph → buffers) with no driver error
    //! - repeated create → replay → drop → rebuild → replay → drop on the SAME
    //!   runtime stream (the generation reset/rebuild lifecycle)
    //!
    //! Runtime-gated: no-op on hosts without CUDA. Runs inline here (not in
    //! `tests/`) because `CudaRuntime` and its stream/function fields are
    //! `pub(crate)`.
    use super::*;

    /// Build a `CudaRuntime` if a CUDA device is available; otherwise return
    /// `None` so the test no-ops (mirrors `CudaBackend::native_runtime_available`).
    fn try_runtime() -> Option<CudaRuntime> {
        CudaRuntime::initialize(0).ok()
    }

    /// One full single-stream lifecycle on the runtime stream: capture →
    /// instantiate → launch (token 1) → mutate input → replay (token 2+) →
    /// drop graph → rebuild → replay → drop. Reusable for the repeated-lifecycle
    /// stress loop.
    ///
    /// The runtime stream is created non-NULL + event-tracking-disabled at
    /// `CudaRuntime::initialize` (B3B), so capture works on it directly — no
    /// dedicated capture stream, no per-cycle `disable_event_tracking` call.
    fn run_one_single_stream_cycle(rt: &CudaRuntime) {
        let n = 256usize;
        let b_scale = 1.0f32;
        let stream = &rt.stream; // the single non-NULL runtime stream

        // Stable buffers — addresses persist for the graph's lifetime.
        let h_vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let x_vals: Vec<f32> = vec![0.5f32; n];
        let h_dev = stream.clone_htod(&h_vals).expect("upload h_dev (b3b)");
        let mut x_dev = stream.clone_htod(&x_vals).expect("upload x_dev (b3b)");
        // Stable output — pre-allocated, zeroed. The graph writes here.
        let mut out_dev = stream.alloc_zeros::<f32>(n).expect("alloc out_dev (b3b)");

        // ── Capture on the runtime stream ──
        // RELAXED mode (see `ffn_graph::graph_capture_mode`): production-
        // equivalent for B3B (no syncs during the capture window) and required
        // for the parallel test harness (allows concurrent cuStreamSynchronize
        // on the shared primary context).
        stream
            .begin_capture(crate::ffn_graph::graph_capture_mode())
            .expect("begin_capture on runtime stream (b3b)");
        {
            let threads_x = GEGGLU_SILU_KERNEL.geometry.threads_per_group[0];
            let n_u = n as u32;
            let cfg = LaunchConfig {
                grid_dim: (n_u.div_ceil(threads_x), 1, 1),
                block_dim: (threads_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch_args = stream.launch_builder(&rt.residual_add);
            launch_args.arg(&h_dev).arg(&x_dev).arg(&mut out_dev);
            launch_args.arg(&b_scale);
            launch_args.arg(&n_u);
            unsafe { launch_args.launch(cfg) }
                .expect("launch residual_add into stable buffer during capture (b3b)");
        }
        // Instantiate with the cudarc-forced AUTO_FREE_ON_LAUNCH flag (see
        // `ffn_graph::graph_instantiate_flags` — cudarc 0.19.8's typed flags
        // enum has no constructible zero variant; AUTO_FREE is a no-op here
        // because the graph references only externally-owned buffers).
        let graph = stream
            .end_capture(crate::ffn_graph::graph_instantiate_flags())
            .expect("end_capture (b3b)")
            .expect("end_capture returned a non-null graph (b3b)");

        // ── Token 1: launch the just-built graph (capture does not execute) ──
        graph.launch().expect("graph.launch #1 / token 1 (b3b)");
        stream.synchronize().expect("sync after token 1 (b3b)");
        let out1: Vec<f32> = stream
            .clone_dtoh(&out_dev)
            .expect("read back token 1 (b3b)");
        let expected1: Vec<f32> = h_vals
            .iter()
            .zip(&x_vals)
            .map(|(h, x)| h + b_scale * x)
            .collect();
        let max_abs1 = out1
            .iter()
            .zip(&expected1)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs1 < 1e-5,
            "(b3b) token-1 graph output diverged: max_abs={max_abs1:.6e}"
        );

        // ── Token 2: stable-pointer replay after mutating x_dev in place ──
        // The arena depends on this: the graph reads the buffer's *current*
        // contents at the same fixed device address on every replay.
        let new_x_val = 0.25f32;
        let new_x_vals = vec![new_x_val; n];
        stream
            .memcpy_htod(&new_x_vals[..], &mut x_dev)
            .expect("in-place memcpy_htod to mutate x_dev (b3b)");
        graph
            .launch()
            .expect("graph.launch #2 / token 2 replay (b3b)");
        stream.synchronize().expect("sync after token 2 (b3b)");
        let out2: Vec<f32> = stream
            .clone_dtoh(&out_dev)
            .expect("read back token 2 (b3b)");
        let expected2: Vec<f32> = h_vals.iter().map(|h| h + b_scale * new_x_val).collect();
        let max_abs2 = out2
            .iter()
            .zip(&expected2)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs2 < 1e-5,
            "(b3b) token-2 stable-pointer replay diverged: max_abs={max_abs2:.6e} (replay must read the buffer's current contents at the stable address)"
        );
        // And it must differ from token 1 — proves the in-place memcpy changed
        // what the graph reads (the captured address is live).
        let replay_diff = out1
            .iter()
            .zip(&out2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            replay_diff > 0.1,
            "(b3b) replay after in-place mutation produced nearly-identical output (max_diff={replay_diff:.6e})"
        );

        // ── Destroy the graph, then rebuild on the SAME runtime stream ──
        // (mirrors reset_kv_cache → next-generation rebuild.)
        drop(graph);
        stream
            .begin_capture(crate::ffn_graph::graph_capture_mode())
            .expect("begin_capture (rebuild, b3b)");
        {
            let threads_x = GEGGLU_SILU_KERNEL.geometry.threads_per_group[0];
            let n_u = n as u32;
            let cfg = LaunchConfig {
                grid_dim: (n_u.div_ceil(threads_x), 1, 1),
                block_dim: (threads_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch_args = stream.launch_builder(&rt.residual_add);
            launch_args.arg(&h_dev).arg(&x_dev).arg(&mut out_dev);
            launch_args.arg(&b_scale);
            launch_args.arg(&n_u);
            unsafe { launch_args.launch(cfg) }
                .expect("launch residual_add during rebuild capture (b3b)");
        }
        let rebuilt = stream
            .end_capture(crate::ffn_graph::graph_instantiate_flags())
            .expect("end_capture (rebuild, b3b)")
            .expect("end_capture returned a non-null graph (rebuild, b3b)");
        rebuilt.launch().expect("rebuilt graph.launch (b3b)");
        stream
            .synchronize()
            .expect("sync after rebuilt launch (b3b)");
        let out3: Vec<f32> = stream
            .clone_dtoh(&out_dev)
            .expect("read back rebuilt (b3b)");
        let max_abs3 = out3
            .iter()
            .zip(&expected2)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs3 < 1e-5,
            "(b3b) rebuilt-graph replay diverged: max_abs={max_abs3:.6e}"
        );

        // Teardown order (B3A-6 / point 7): exec graph → buffers → inputs.
        drop(rebuilt);
        drop(out_dev);
        drop(x_dev);
        drop(h_dev);
    }

    /// Single-stream capture → instantiate → token-1 launch → token-2 replay →
    /// rebuild, on the runtime's non-NULL stream. Runtime-gated.
    #[test]
    fn b3b_single_stream_graph_capture_replay_teardown() {
        let Some(rt) = try_runtime() else {
            eprintln!("b3b_single_stream: no CUDA runtime — skipping (no GPU on this host)");
            return;
        };
        // Serialize vs other capture tests (concurrent stream capture on the
        // shared primary context is not supported — see CUDA_CAPTURE_TEST_LOCK).
        let _g = crate::CUDA_CAPTURE_TEST_LOCK.lock().unwrap();
        run_one_single_stream_cycle(&rt);
        eprintln!(
            "b3b_single_stream: PASS — capture/replay/rebuild verified on the runtime stream"
        );
    }

    /// Repeated create → replay → drop → rebuild → drop lifecycle on the SAME
    /// runtime stream (mirrors a backend serving multiple generations through
    /// reset_kv_cache boundaries). This is the stronger single-stream
    /// replacement for the old two-stream `b3a_smoke_repeated_capture_teardown_
    /// lifecycle`: it proves the installed driver/runtime handles repeated
    /// graph construction/teardown on the one stream the whole decode path now
    /// shares, with no capture invalidation across cycles. Runtime-gated.
    #[test]
    fn b3b_single_stream_repeated_capture_reset_rebuild_lifecycle() {
        let Some(rt) = try_runtime() else {
            eprintln!("b3b_single_stream_lifecycle: no CUDA runtime — skipping");
            return;
        };
        let _g = crate::CUDA_CAPTURE_TEST_LOCK.lock().unwrap();
        // Five independent cycles on the SAME runtime (each cycle = a
        // reset_kv_cache generation boundary: capture → replay → rebuild).
        for i in 0..5 {
            run_one_single_stream_cycle(&rt);
            eprintln!("b3b_single_stream_lifecycle: cycle {} complete", i + 1);
        }
    }

    /// Validate the in-place residual-add primitive (`launch_residual_add_
    /// inplace_into`) that B3B uses to write attention's post-attn residual
    /// directly into the arena input slot. The primitive binds one buffer as
    /// BOTH the residual base (read) and the output (write); this test proves
    /// the result equals a fresh-buffer residual add on the same stream.
    /// Runtime-gated.
    #[test]
    fn b3b_inplace_residual_add_matches_fresh_buffer_add() {
        let Some(rt) = try_runtime() else {
            eprintln!("b3b_inplace_residual_add: no CUDA runtime — skipping");
            return;
        };
        // Acquire the capture lock: this test issues tight back-to-back
        // `cuStreamSynchronize` calls. Under RELAXED capture mode these are
        // permitted during a concurrent capture, but the lock keeps this test
        // off the shared primary context entirely while a capture test runs
        // (belt-and-suspenders; see `CUDA_CAPTURE_TEST_LOCK`).
        let _g = crate::CUDA_CAPTURE_TEST_LOCK.lock().unwrap();
        let n = 1024usize;
        let b_scale = 0.75f32;
        let stream = &rt.stream;
        let h_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let x_vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();

        // Fresh-buffer reference: out = h + b_scale * x (allocating variant).
        let h_ref = stream.clone_htod(&h_vals).expect("upload h_ref");
        let x_ref = stream.clone_htod(&x_vals).expect("upload x_ref");
        let mut out_ref = stream.alloc_zeros::<f32>(n).expect("alloc out_ref");
        rt.launch_residual_add_into(stream, &h_ref, &x_ref, &mut out_ref, n, b_scale)
            .expect("launch_residual_add_into (reference)");
        let ref_out = rt.sync_dtoh_f32(&out_ref).expect("read back reference");

        // In-place variant: buf = h; buf += b_scale * x (buf bound as h AND out).
        let buf_in = stream.clone_htod(&h_vals).expect("upload buf_in");
        let x_in = stream.clone_htod(&x_vals).expect("upload x_in");
        rt.launch_residual_add_inplace_into(stream, &buf_in, &x_in, n, b_scale)
            .expect("launch_residual_add_inplace_into");
        let inplace_out = rt.sync_dtoh_f32(&buf_in).expect("read back in-place");

        let max_abs = ref_out
            .iter()
            .zip(&inplace_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-6,
            "(b3b) in-place residual add diverged from fresh-buffer add: max_abs={max_abs:.6e}"
        );
        // Also confirm it actually applied the add (not a no-op).
        let applied = h_vals
            .iter()
            .zip(&inplace_out)
            .map(|(h, o)| (h - o).abs())
            .fold(0.0f32, f32::max);
        assert!(
            applied > 0.1,
            "(b3b) in-place residual add produced no change"
        );
    }
}

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileError, CompileOptions};

use crate::ops::{
    ACTIVATION_GELU_TANH_CUDA_SRC, ACTIVATION_GELU_TANH_KERNEL, ACTIVATION_SILU_CUDA_SRC,
    ACTIVATION_SILU_KERNEL, DECODE_ATTENTION_CUDA_SRC, DECODE_ATTENTION_KERNEL, F16_GEMV_CUDA_SRC,
    F16_GEMV_KERNEL, F32_GEMV_CUDA_SRC, F32_GEMV_KERNEL, GEGGLU_GELU_TANH_CUDA_SRC,
    GEGGLU_GELU_TANH_KERNEL, GEGGLU_SILU_CUDA_SRC, GEGGLU_SILU_KERNEL, KV_APPEND_CUDA_SRC,
    KV_APPEND_KERNEL, PREFILL_ATTENTION_CUDA_SRC, PREFILL_ATTENTION_KERNEL,
    Q4K_DUAL_MATVEC_CUDA_SRC, Q4K_DUAL_MATVEC_KERNEL, Q4K_MATMUL_CUDA_SRC, Q4K_MATMUL_KERNEL,
    Q4K_MATVEC_CUDA_SRC, Q4K_MATVEC_KERNEL, Q4_MATVEC_CUDA_SRC, Q4_MATVEC_KERNEL,
    Q4_VECMAT_CUDA_SRC, Q4_VECMAT_KERNEL, Q6K_MATMUL_CUDA_SRC, Q6K_MATMUL_KERNEL,
    Q6K_MATVEC_CUDA_SRC, Q6K_MATVEC_KERNEL, RESIDUAL_ADD_CUDA_SRC, RESIDUAL_ADD_KERNEL,
    RMS_NORM_CUDA_SRC, RMS_NORM_HEADS_CUDA_SRC, RMS_NORM_HEADS_KERNEL, RMS_NORM_KERNEL,
    ROPE_CUDA_SRC, ROPE_KERNEL,
};
#[cfg(test)]
use crate::weight_cache::CacheStats;
use crate::weight_cache::WeightCache;

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
    /// Persistent device-resident weight cache (see `weight_cache.rs`).
    /// Uploads each immutable weight matrix once and reuses the device buffer
    /// across calls — the first slice of the per-projection htod round-trip
    /// collapse. Activations stay on the fresh `clone_htod` path.
    weight_cache: WeightCache,
    summary: String,
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
            "{Q4K_MATVEC_CUDA_SRC}\n{Q6K_MATVEC_CUDA_SRC}\n{Q4K_MATMUL_CUDA_SRC}\n{Q6K_MATMUL_CUDA_SRC}\n{Q4K_DUAL_MATVEC_CUDA_SRC}\n{F32_GEMV_CUDA_SRC}\n{F16_GEMV_CUDA_SRC}\n{Q4_MATVEC_CUDA_SRC}\n{Q4_VECMAT_CUDA_SRC}\n{KV_APPEND_CUDA_SRC}\n{RMS_NORM_CUDA_SRC}\n{RMS_NORM_HEADS_CUDA_SRC}\n{GEGGLU_SILU_CUDA_SRC}\n{GEGGLU_GELU_TANH_CUDA_SRC}\n{ACTIVATION_SILU_CUDA_SRC}\n{ACTIVATION_GELU_TANH_CUDA_SRC}\n{RESIDUAL_ADD_CUDA_SRC}\n{ROPE_CUDA_SRC}\n{DECODE_ATTENTION_CUDA_SRC}\n{PREFILL_ATTENTION_CUDA_SRC}"
        );
        let ptx = compile_ptx_with_opts(
            &combined_src,
            CompileOptions {
                fmad: Some(false),
                ..Default::default()
            },
        )
        .map_err(|err| RuntimeError::compile("compiling CUDA k-quant NVRTC module", err))?;
        let module = context
            .load_module(ptx)
            .map_err(|err| RuntimeError::context("loading CUDA k-quant module", err))?;
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
        let stream = context.default_stream();

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
            weight_cache: WeightCache::default(),
            summary: format!(
                "CUDA device {device_name} (ordinal {ordinal}, sm_{cc_major}{cc_minor}); native q4k_matvec/q6k_matvec/q4k_matmul/q6k_matmul/q4k_dual_matvec/f32_gemv/f16_gemv/q4_matvec/q4_vecmat/kv_append/rms_norm/rms_norm_heads/geglu_silu/geglu_gelu_tanh/activation_silu/activation_gelu_tanh/residual_add/rope/decode_attention/prefill_attention loaded, remaining ops use CPU fallback"
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

    /// Snapshot the weight-cache hit/miss counters (test diagnostic).
    #[cfg(test)]
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
        if num_rows == 0 {
            return Ok(Vec::new());
        }
        if x.len() != hidden {
            return Err(RuntimeError::usage(format!(
                "q4k_matvec expected x.len() == hidden ({hidden}), got {}",
                x.len()
            )));
        }
        if !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q4k_matvec hidden size must be a multiple of 256, got {hidden}"
            )));
        }

        let expected_bytes = num_rows
            .checked_mul(hidden / 256)
            .and_then(|blocks| blocks.checked_mul(144))
            .ok_or_else(|| RuntimeError::usage("q4k_matvec byte-size overflow".to_string()))?;
        if q4k_data.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q4k_matvec expected {expected_bytes} Q4_K bytes for shape ({num_rows}, {hidden}), got {}",
                q4k_data.len()
            )));
        }

        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q4k_data)
            .map_err(|err| RuntimeError::context("uploading Q4_K weights to CUDA", err))?;
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading activation vector to CUDA", err))?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(num_rows).map_err(|err| {
            RuntimeError::context("allocating CUDA q4k_matvec output buffer", err)
        })?;
        let threads_x = Q4K_MATVEC_KERNEL.geometry.threads_per_group[0];
        let n = num_rows as u32;
        let k = hidden as u32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.q4k_matvec);
        launch_args
            .arg(&*w_dev)
            .arg(&x_dev)
            .arg(&mut out_dev)
            .arg(&n)
            .arg(&k);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q4k_matvec kernel", err))?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA q4k_matvec stream", err))?;
        self.stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA q4k_matvec output", err))
    }

    pub(crate) fn launch_q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        if num_rows == 0 {
            return Ok(Vec::new());
        }
        if x.len() != hidden {
            return Err(RuntimeError::usage(format!(
                "q6k_matvec expected x.len() == hidden ({hidden}), got {}",
                x.len()
            )));
        }
        if !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q6k_matvec hidden size must be a multiple of 256, got {hidden}"
            )));
        }

        // Q6_K super-block = 210 bytes / 256 elements.
        let expected_bytes = num_rows
            .checked_mul(hidden / 256)
            .and_then(|blocks| blocks.checked_mul(210))
            .ok_or_else(|| RuntimeError::usage("q6k_matvec byte-size overflow".to_string()))?;
        if q6k_data.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q6k_matvec expected {expected_bytes} Q6_K bytes for shape ({num_rows}, {hidden}), got {}",
                q6k_data.len()
            )));
        }

        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q6k_data)
            .map_err(|err| RuntimeError::context("uploading Q6_K weights to CUDA", err))?;
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading activation vector to CUDA", err))?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(num_rows).map_err(|err| {
            RuntimeError::context("allocating CUDA q6k_matvec output buffer", err)
        })?;
        let threads_x = Q6K_MATVEC_KERNEL.geometry.threads_per_group[0];
        let n = num_rows as u32;
        let k = hidden as u32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.q6k_matvec);
        launch_args
            .arg(&*w_dev)
            .arg(&x_dev)
            .arg(&mut out_dev)
            .arg(&n)
            .arg(&k);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q6k_matvec kernel", err))?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA q6k_matvec stream", err))?;
        self.stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA q6k_matvec output", err))
    }

    pub(crate) fn launch_q4k_matmul(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, RuntimeError> {
        if num_rows == 0 || seq_len == 0 {
            return Ok(vec![0.0f32; seq_len * num_rows]);
        }
        if x.len() != seq_len * hidden {
            return Err(RuntimeError::usage(format!(
                "q4k_matmul expected x.len() == seq*hidden ({}, {}), got {}",
                seq_len * hidden,
                hidden,
                x.len()
            )));
        }
        if !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q4k_matmul hidden size must be a multiple of 256, got {hidden}"
            )));
        }

        let expected_bytes = num_rows
            .checked_mul(hidden / 256)
            .and_then(|blocks| blocks.checked_mul(144))
            .ok_or_else(|| RuntimeError::usage("q4k_matmul byte-size overflow".to_string()))?;
        if q4k_data.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q4k_matmul expected {expected_bytes} Q4_K bytes for shape ({num_rows}, {hidden}), got {}",
                q4k_data.len()
            )));
        }

        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q4k_data)
            .map_err(|err| RuntimeError::context("uploading Q4_K weights to CUDA", err))?;
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading activations to CUDA", err))?;
        // Output is [seq, rows] row-major.
        let out_len = seq_len * num_rows;
        let mut out_dev = self.stream.alloc_zeros::<f32>(out_len).map_err(|err| {
            RuntimeError::context("allocating CUDA q4k_matmul output buffer", err)
        })?;
        let threads_x = Q4K_MATMUL_KERNEL.geometry.threads_per_group[0];
        // One thread per (row, seq) pair.
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
            .arg(&x_dev)
            .arg(&mut out_dev)
            .arg(&n)
            .arg(&k)
            .arg(&seq);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q4k_matmul kernel", err))?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA q4k_matmul stream", err))?;
        self.stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA q4k_matmul output", err))
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
        if num_rows == 0 || seq_len == 0 {
            return Ok(vec![0.0f32; seq_len * num_rows]);
        }
        if x.len() != seq_len * hidden {
            return Err(RuntimeError::usage(format!(
                "q6k_matmul expected x.len() == seq*hidden ({}, {}), got {}",
                seq_len * hidden,
                hidden,
                x.len()
            )));
        }
        if !hidden.is_multiple_of(256) {
            return Err(RuntimeError::usage(format!(
                "q6k_matmul hidden size must be a multiple of 256, got {hidden}"
            )));
        }

        let expected_bytes = num_rows
            .checked_mul(hidden / 256)
            .and_then(|blocks| blocks.checked_mul(210))
            .ok_or_else(|| RuntimeError::usage("q6k_matmul byte-size overflow".to_string()))?;
        if q6k_data.len() != expected_bytes {
            return Err(RuntimeError::usage(format!(
                "q6k_matmul expected {expected_bytes} Q6_K bytes for shape ({num_rows}, {hidden}), got {}",
                q6k_data.len()
            )));
        }

        let w_dev = self
            .weight_cache
            .get_or_upload_bytes(&self.stream, q6k_data)
            .map_err(|err| RuntimeError::context("uploading Q6_K weights to CUDA", err))?;
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading activations to CUDA", err))?;
        // Output is [seq, rows] row-major.
        let out_len = seq_len * num_rows;
        let mut out_dev = self.stream.alloc_zeros::<f32>(out_len).map_err(|err| {
            RuntimeError::context("allocating CUDA q6k_matmul output buffer", err)
        })?;
        let threads_x = Q6K_MATMUL_KERNEL.geometry.threads_per_group[0];
        // One thread per (row, seq) pair.
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
            .arg(&x_dev)
            .arg(&mut out_dev)
            .arg(&n)
            .arg(&k)
            .arg(&seq);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA q6k_matmul kernel", err))?;
        self.stream
            .synchronize()
            .map_err(|err| RuntimeError::context("synchronizing CUDA q6k_matmul stream", err))?;
        self.stream
            .clone_dtoh(&out_dev)
            .map_err(|err| RuntimeError::context("reading CUDA q6k_matmul output", err))
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

        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading rms_norm input to CUDA", err))?;
        // `cudarc` requires a non-empty host slice for `clone_htod`; pass a
        // one-element placeholder when the weight is `None` and flag
        // `has_weight = 0` so the device ignores the pointer.
        let (weight_vec, has_weight) = match weight {
            Some(w) => (w.to_vec(), 1i32),
            None => (vec![0.0f32], 0i32),
        };
        let weight_dev = self
            .stream
            .clone_htod(&weight_vec)
            .map_err(|err| RuntimeError::context("uploading rms_norm weight to CUDA", err))?;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(rows * cols)
            .map_err(|err| RuntimeError::context("allocating CUDA rms_norm output buffer", err))?;
        // Cap the block size at 1024 (32 warps); the device reduction uses a
        // fixed 32-slot shared array. Larger `cols` are handled by the
        // strided accumulation loop.
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
            .arg(&x_dev)
            .arg(&weight_dev)
            .arg(&mut out_dev)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&eps)
            .arg(&offset)
            .arg(&has_weight);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA rms_norm kernel", err))?;
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
        let (weight_vec, has_weight) = match weight {
            Some(w) => (w.to_vec(), 1i32),
            None => (vec![0.0f32], 0i32),
        };
        let weight_dev = self
            .stream
            .clone_htod(&weight_vec)
            .map_err(|err| RuntimeError::context("uploading rms_norm_heads weight to CUDA", err))?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(total).map_err(|err| {
            RuntimeError::context("allocating CUDA rms_norm_heads output buffer", err)
        })?;
        let block_dim = 1024u32;
        let seq_u = seq_len as u32;
        let heads_u = num_heads as u32;
        let dim_u = head_dim as u32;
        let blocks = (seq_u.checked_mul(heads_u)).ok_or_else(|| {
            RuntimeError::usage("rms_norm_heads grid dim overflow seq*heads".to_string())
        })?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(&self.rms_norm_heads);
        launch_args
            .arg(&x_dev)
            .arg(&weight_dev)
            .arg(&mut out_dev)
            .arg(&seq_u)
            .arg(&heads_u)
            .arg(&dim_u)
            .arg(&eps)
            .arg(&offset)
            .arg(&has_weight);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA rms_norm_heads kernel", err))?;
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
        let divisor = if position_divisor > 0.0 {
            position_divisor
        } else {
            1.0
        };

        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|err| RuntimeError::context("uploading rope input to CUDA", err))?;
        let inv_freq_dev = self
            .stream
            .clone_htod(inv_freq)
            .map_err(|err| RuntimeError::context("uploading rope inv_freq to CUDA", err))?;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(total)
            .map_err(|err| RuntimeError::context("allocating CUDA rope output buffer", err))?;
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
            .arg(&x_dev)
            .arg(&inv_freq_dev)
            .arg(&mut out_dev)
            .arg(&seq_u)
            .arg(&heads_u)
            .arg(&dim_u)
            .arg(&half_u)
            .arg(&pos_off)
            .arg(&divisor)
            .arg(&n_u64);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA rope kernel", err))?;
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
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(q_len)
            .map_err(|err| RuntimeError::context("allocating decode_attention output", err))?;

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
            .arg(&q_dev)
            .arg(&k_dev)
            .arg(&v_dev)
            .arg(&mut scores_dev)
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
        let q_dim = num_q.checked_mul(head_dim);
        let q_len = seq_len.checked_mul(q_dim.unwrap_or(usize::MAX));
        let kv_len = seq_len.checked_mul(kv_dim);
        let (q_len, kv_len) = match (q_len, kv_len) {
            (Some(ql), Some(kvl)) if q.len() == ql && k.len() == kvl && v.len() == kvl => (ql, kvl),
            _ => {
                return Err(RuntimeError::usage(format!(
                    "prefill_attention shape mismatch: q={} (expected {q_len:?}), k={} / v={} (expected {kv_len:?}, kv_dim={kv_dim}, seq={seq_len}), num_q={num_q}, head_dim={head_dim}",
                    q.len(),
                    k.len(),
                    v.len(),
                )))
            }
        };
        if seq_len == 0 {
            out.clear();
            out.resize(num_q * head_dim, 0.0);
            return Ok(());
        }
        if reps == 0 {
            return Err(RuntimeError::usage(
                "prefill_attention requires reps >= 1 (num_kv heads > 0)".to_string(),
            ));
        }
        // Guard the grid + 64-bit device indices against the 32-bit limit.
        if q_len > u32::MAX as usize
            || kv_len > u32::MAX as usize
            || num_q > u32::MAX as usize
            || head_dim > u32::MAX as usize
            || kv_dim > u32::MAX as usize
            || seq_len > u32::MAX as usize
        {
            return Err(RuntimeError::usage(format!(
                "prefill_attention shape (num_q={num_q}, head_dim={head_dim}, kv_dim={kv_dim}, seq_len={seq_len}) exceeds the 32-bit kernel index limit"
            )));
        }
        // Dynamic shared budget: 256 doubles (sum) + 256 floats (max) +
        // seq_len floats (scores). Stay within the default 48 KB dynamic
        // shared-mem ceiling (larger needs `cudaFuncSetAttribute`, which the
        // single-command-buffer follow-on will revisit); fall back to the host
        // reference otherwise.
        let shm_fixed = 256 * 8 + 256 * 4;
        let shm_scores = seq_len.checked_mul(4);
        let shared_mem_bytes = match shm_scores {
            Some(ss) if shm_fixed + ss <= 48 * 1024 => shm_fixed + ss,
            _ => {
                return Err(RuntimeError::usage(format!(
                    "prefill_attention seq_len={seq_len} exceeds the 48 KB dynamic shared-mem budget; fall back to host"
                )));
            }
        };

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
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(q_len)
            .map_err(|err| RuntimeError::context("allocating prefill_attention output", err))?;

        let threads = PREFILL_ATTENTION_KERNEL.geometry.threads_per_group[0];
        let cfg = LaunchConfig {
            // grid = (num_q query heads, seq_len query positions).
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
            .arg(&q_dev)
            .arg(&k_dev)
            .arg(&v_dev)
            .arg(&mut out_dev)
            .arg(&scale)
            .arg(&softcap_val)
            .arg(&has_softcap)
            .arg(&num_q_u)
            .arg(&head_dim_u)
            .arg(&kv_dim_u)
            .arg(&reps_u)
            .arg(&seq_len_u);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context("launching CUDA prefill_attention kernel", err))?;
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
        // The element count is the only index; guard it against the 32-bit
        // grid-dim limit (a single `unsigned int` `n` arg).
        if n > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "{ctx} length {n} exceeds the 32-bit kernel index limit"
            )));
        }

        let a_dev = self
            .stream
            .clone_htod(in_a)
            .map_err(|err| RuntimeError::context_concat("uploading ", ctx, " in_a to CUDA", err))?;
        let b_dev = self
            .stream
            .clone_htod(in_b)
            .map_err(|err| RuntimeError::context_concat("uploading ", ctx, " in_b to CUDA", err))?;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|err| RuntimeError::context_concat("allocating CUDA ", ctx, " output", err))?;
        let threads_x = GEGGLU_SILU_KERNEL.geometry.threads_per_group[0];
        let n_u = n as u32;
        // Hold the optional scalar for the lifetime of `launch_args` — the
        // builder borrows it across the (unsafe) launch, so it must outlive
        // the launch rather than the `if let` block.
        let extra = extra_scalar;
        let cfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(func);
        launch_args.arg(&a_dev).arg(&b_dev).arg(&mut out_dev);
        if let Some(ref scalar) = extra {
            launch_args.arg(scalar);
        }
        launch_args.arg(&n_u);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context_concat("launching CUDA ", ctx, " kernel", err))?;
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
        if n > u32::MAX as usize {
            return Err(RuntimeError::usage(format!(
                "{ctx} length {n} exceeds the 32-bit kernel index limit"
            )));
        }

        let input_dev = self.stream.clone_htod(input).map_err(|err| {
            RuntimeError::context_concat("uploading ", ctx, " input to CUDA", err)
        })?;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|err| RuntimeError::context_concat("allocating CUDA ", ctx, " output", err))?;
        let threads_x = ACTIVATION_SILU_KERNEL.geometry.threads_per_group[0];
        let n_u = n as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(threads_x), 1, 1),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch_args = self.stream.launch_builder(func);
        launch_args.arg(&input_dev).arg(&mut out_dev).arg(&n_u);
        unsafe { launch_args.launch(cfg) }
            .map_err(|err| RuntimeError::context_concat("launching CUDA ", ctx, " kernel", err))?;
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

#[derive(Debug, Clone)]
pub(crate) struct RuntimeError {
    message: String,
}

impl RuntimeError {
    fn context(action: &'static str, source: DriverError) -> Self {
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

    fn usage(message: String) -> Self {
        Self { message }
    }
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RuntimeError {}

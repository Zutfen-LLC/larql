use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileError, CompileOptions};

use crate::ops::{
    F16_GEMV_CUDA_SRC, F16_GEMV_KERNEL, F32_GEMV_CUDA_SRC, F32_GEMV_KERNEL, KV_APPEND_CUDA_SRC,
    KV_APPEND_KERNEL, Q4K_DUAL_MATVEC_CUDA_SRC, Q4K_DUAL_MATVEC_KERNEL, Q4K_MATMUL_CUDA_SRC,
    Q4K_MATMUL_KERNEL, Q4K_MATVEC_CUDA_SRC, Q4K_MATVEC_KERNEL, Q4_MATVEC_CUDA_SRC,
    Q4_MATVEC_KERNEL, Q4_VECMAT_CUDA_SRC, Q4_VECMAT_KERNEL, Q6K_MATMUL_CUDA_SRC, Q6K_MATMUL_KERNEL,
    Q6K_MATVEC_CUDA_SRC, Q6K_MATVEC_KERNEL,
};

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
            "{Q4K_MATVEC_CUDA_SRC}\n{Q6K_MATVEC_CUDA_SRC}\n{Q4K_MATMUL_CUDA_SRC}\n{Q6K_MATMUL_CUDA_SRC}\n{Q4K_DUAL_MATVEC_CUDA_SRC}\n{F32_GEMV_CUDA_SRC}\n{F16_GEMV_CUDA_SRC}\n{Q4_MATVEC_CUDA_SRC}\n{Q4_VECMAT_CUDA_SRC}\n{KV_APPEND_CUDA_SRC}"
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
            summary: format!(
                "CUDA device {device_name} (ordinal {ordinal}, sm_{cc_major}{cc_minor}); native q4k_matvec/q6k_matvec/q4k_matmul/q6k_matmul/q4k_dual_matvec/f32_gemv/f16_gemv/q4_matvec/q4_vecmat/kv_append loaded, remaining ops use CPU fallback"
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
            .stream
            .clone_htod(q4k_data)
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
            .arg(&w_dev)
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
            .stream
            .clone_htod(q6k_data)
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
            .arg(&w_dev)
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
            .stream
            .clone_htod(q4k_data)
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
            .arg(&w_dev)
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
            .stream
            .clone_htod(q6k_data)
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
            .arg(&w_dev)
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
            .stream
            .clone_htod(q4k_a)
            .map_err(|err| RuntimeError::context("uploading Q4_K w_a weights to CUDA", err))?;
        let wb_dev = self
            .stream
            .clone_htod(q4k_b)
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
            .arg(&wa_dev)
            .arg(&wb_dev)
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
            .stream
            .clone_htod(w)
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
            .arg(&w_dev)
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
            .stream
            .clone_htod(w_f16)
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
            .arg(&w_dev)
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

        let w_dev = self
            .stream
            .clone_htod(q4_data)
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
            .arg(&w_dev)
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
        let w_dev = self
            .stream
            .clone_htod(q4_data)
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
            .arg(&w_dev)
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

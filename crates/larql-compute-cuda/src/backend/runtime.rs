use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileError, CompileOptions};

use crate::ops::{Q4K_MATVEC_CUDA_SRC, Q4K_MATVEC_KERNEL};

#[derive(Debug)]
pub(crate) struct CudaRuntime {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    q4k_matvec: CudaFunction,
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

        let ptx = compile_ptx_with_opts(
            Q4K_MATVEC_CUDA_SRC,
            CompileOptions {
                fmad: Some(false),
                ..Default::default()
            },
        )
        .map_err(|err| RuntimeError::compile("compiling q4k_matvec NVRTC module", err))?;
        let module = context
            .load_module(ptx)
            .map_err(|err| RuntimeError::context("loading q4k_matvec CUDA module", err))?;
        let q4k_matvec = module
            .load_function(Q4K_MATVEC_KERNEL.identifier)
            .map_err(|err| RuntimeError::context("loading q4k_matvec CUDA function", err))?;
        let stream = context.default_stream();

        Ok(Self {
            _context: context,
            stream,
            _module: module,
            q4k_matvec,
            summary: format!(
                "CUDA device {device_name} (ordinal {ordinal}, sm_{cc_major}{cc_minor}); native q4k_matvec loaded, remaining ops use CPU fallback"
            ),
        })
    }

    pub(crate) fn summary(&self) -> &str {
        &self.summary
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

mod runtime;

use crate::options::BackendOptions;
use std::sync::Arc;

pub(crate) use runtime::{CudaRuntime, RuntimeError};

#[derive(Debug, Clone)]
pub struct CudaBackend {
    options: BackendOptions,
    runtime: Option<Arc<CudaRuntime>>,
    runtime_status: Option<String>,
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
            }),
            Err(err) if options.allow_cpu_delegate => Ok(Self {
                options,
                runtime: None,
                runtime_status: Some(err.to_string()),
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

    /// Native Q6_K amortised matmul. Not yet routed through the
    /// `QuantMatVec` trait (no `q6k_matmul` trait method exists — the
    /// amortised Q6_K matmul is currently a CPU-only free function in
    /// `ffn/weight.rs::quant_matmul`). Exposed here so the prefill-kquant
    /// backend-routing slice can dispatch through it without re-adding the
    /// kernel. Parity-verified when a CUDA runtime is present.
    #[allow(dead_code)]
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

    pub(crate) fn native_runtime_available(&self) -> bool {
        self.runtime.is_some()
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

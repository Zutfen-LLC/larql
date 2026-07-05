use crate::backend::runtime::{try_init, VulkanRuntime};
use crate::options::BackendOptions;

pub mod runtime;
mod device_chain;

#[derive(Debug)]
pub struct VulkanBackend {
    options: BackendOptions,
    /// Lazily-initialised Vulkan runtime (`None` on a GPU-less / loader-less
    /// host). `Option` rather than `Result` so a missing device is a silent
    /// "use the CPU delegate" state, mirroring CUDA's scaffold path.
    runtime: Option<VulkanRuntime>,
    runtime_status: Option<String>,
}

impl VulkanBackend {
    pub fn new() -> Result<Self, BackendInitError> {
        Self::with_options(BackendOptions::default())
    }

    pub fn with_options(options: BackendOptions) -> Result<Self, BackendInitError> {
        // Probe for a Vulkan device once at construction. A missing device or
        // loader is not fatal: the backend still serves correct results via
        // the CPU delegate and `supports_quant` reports `false`.
        let (runtime, status) = match try_init(&options) {
            Some(rt) => {
                let summary = rt.summary().to_string();
                (Some(rt), Some(summary))
            }
            None => (
                None,
                Some("no Vulkan device/loader; using CPU delegate".to_string()),
            ),
        };
        Ok(Self {
            options,
            runtime,
            runtime_status: status,
        })
    }

    pub fn options(&self) -> &BackendOptions {
        &self.options
    }

    /// True when a native Vulkan device runtime is present. Drives the
    /// honest `supports()` / `supports_quant()` gating (GPU-4001 §5).
    pub fn native_runtime_available(&self) -> bool {
        self.runtime.is_some()
    }

    /// Borrow the device runtime (`None` on the CPU-delegate path).
    pub(crate) fn runtime(&self) -> Option<&VulkanRuntime> {
        self.runtime.as_ref()
    }

    /// One-line device summary (name + loaded kernels) when native, or the
    /// scaffold-fallback reason when not. Used by `device_info()` and the
    /// test runtime gate.
    pub(crate) fn runtime_summary(&self) -> &str {
        match (&self.runtime, &self.runtime_status) {
            (Some(rt), _) => rt.summary(),
            (None, Some(status)) => status.as_str(),
            (None, None) => "Vulkan runtime unavailable; using CPU delegate scaffold",
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

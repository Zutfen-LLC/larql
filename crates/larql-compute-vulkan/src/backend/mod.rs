use crate::options::BackendOptions;

#[derive(Debug, Clone)]
pub struct VulkanBackend {
    options: BackendOptions,
}

impl VulkanBackend {
    pub fn new() -> Result<Self, BackendInitError> {
        Self::with_options(BackendOptions::default())
    }

    pub fn with_options(options: BackendOptions) -> Result<Self, BackendInitError> {
        if !options.allow_cpu_delegate {
            return Err(BackendInitError::Unavailable(
                "Vulkan scaffold requires `allow_cpu_delegate=true` until native kernels land",
            ));
        }
        Ok(Self { options })
    }

    pub fn options(&self) -> &BackendOptions {
        &self.options
    }
}

#[derive(Debug, Clone)]
pub enum BackendInitError {
    Unavailable(&'static str),
}

impl std::fmt::Display for BackendInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for BackendInitError {}

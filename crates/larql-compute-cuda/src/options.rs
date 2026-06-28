#[derive(Debug, Clone, Copy)]
pub struct BackendOptions {
    pub allow_cpu_delegate: bool,
    pub device_ordinal: usize,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            allow_cpu_delegate: true,
            device_ordinal: 0,
        }
    }
}

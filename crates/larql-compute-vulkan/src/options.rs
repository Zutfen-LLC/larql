#[derive(Debug, Clone, Copy)]
pub struct BackendOptions {
    pub allow_cpu_delegate: bool,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            allow_cpu_delegate: true,
        }
    }
}

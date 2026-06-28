#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferHandle {
    pub label: &'static str,
    pub bytes: usize,
}

impl BufferHandle {
    pub const fn new(label: &'static str, bytes: usize) -> Self {
        Self { label, bytes }
    }
}

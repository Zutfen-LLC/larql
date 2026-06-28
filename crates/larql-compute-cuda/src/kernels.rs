#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchGeometry {
    pub workgroups: [u32; 3],
    pub threads_per_group: [u32; 3],
}

impl Default for DispatchGeometry {
    fn default() -> Self {
        Self {
            workgroups: [1, 1, 1],
            threads_per_group: [1, 1, 1],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelHandle {
    pub identifier: &'static str,
    pub geometry: DispatchGeometry,
}

impl KernelHandle {
    pub const fn new(identifier: &'static str, geometry: DispatchGeometry) -> Self {
        Self {
            identifier,
            geometry,
        }
    }
}

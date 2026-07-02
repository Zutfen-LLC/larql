use crate::kernels::{DispatchGeometry, KernelHandle};

pub const PREFILL_KERNEL: KernelHandle = KernelHandle::new(
    "prefill_kquant",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const DECODE_KERNEL: KernelHandle = KernelHandle::new(
    "decode_token",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

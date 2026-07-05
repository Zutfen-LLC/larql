use crate::kernels::{DispatchGeometry, KernelHandle};

/// Placeholder dispatch geometry for a future fused `prefill_kquant` kernel.
/// **Not** currently loaded by `CudaRuntime` — see `backend/runtime.rs` for
/// the ~20 NVRTC kernels that are wired up. Prefill instead runs through the
/// host-orchestrated pipeline (`pipeline.rs`) and the device-resident
/// activation/attention chain (`decode_attention` / `prefill_attention`).
pub const PREFILL_KERNEL: KernelHandle = KernelHandle::new(
    "prefill_kquant",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

/// Placeholder dispatch geometry for a future fused `decode_token` kernel.
/// **Not** currently loaded by `CudaRuntime` — see `backend/runtime.rs` for
/// the ~20 NVRTC kernels that are wired up. Decode instead runs through the
/// host-orchestrated pipeline (`pipeline.rs`) and the device-resident
/// activation/attention chain (`decode_attention` / `prefill_attention`).
pub const DECODE_KERNEL: KernelHandle = KernelHandle::new(
    "decode_token",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

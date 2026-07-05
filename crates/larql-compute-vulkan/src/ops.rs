use crate::kernels::{DispatchGeometry, KernelHandle};

pub const Q4K_MATVEC_KERNEL: KernelHandle = KernelHandle::new(
    "q4k_matvec",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [256, 1, 1],
    },
);

/// Seq-general Q4_K matmul (FFN gate/up/down projections).
pub const Q4K_MATMUL_KERNEL: KernelHandle = KernelHandle::new(
    "q4k_matmul",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [64, 1, 1],
    },
);

/// RMSNorm (post-embedding + per-FFN-input).
pub const RMS_NORM_KERNEL: KernelHandle = KernelHandle::new(
    "rms_norm",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [64, 1, 1],
    },
);

/// GEGLU (SiLU-gated) activation.
pub const GEGLU_SILU_KERNEL: KernelHandle = KernelHandle::new(
    "geglu_silu",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [256, 1, 1],
    },
);

/// Residual add.
pub const RESIDUAL_ADD_KERNEL: KernelHandle = KernelHandle::new(
    "residual_add",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [256, 1, 1],
    },
);

pub const Q6K_MATVEC_KERNEL: KernelHandle = KernelHandle::new(
    "q6k_matvec",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [256, 1, 1],
    },
);

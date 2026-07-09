//! GPU-004B hardware probe: proves native CUDA path was exercised.

use larql_compute::backend::{Capability, ComputeBackend};

#[test]
fn hardware_probe_native_runtime_is_active() {
    let backend =
        larql_compute_cuda::CudaBackend::new().expect("CudaBackend::new() failed entirely");

    // supports() returns false for everything on the scaffold path.
    let supports_quant = backend.supports(Capability::QuantMatVec);
    let supports_decode = backend.supports(Capability::DecodeToken);
    let device_info = backend.device_info();

    println!(
        "HARDWARE_PROBE: supports(QuantMatVec) = {}",
        supports_quant
    );
    println!(
        "HARDWARE_PROBE: supports(DecodeToken) = {}",
        supports_decode
    );
    println!("HARDWARE_PROBE: device_info =");
    println!("{}", device_info);

    assert!(
        supports_quant,
        "QuantMatVec capability is false — CUDA runtime was NOT active. \
         All runtime-gated tests silently used the CPU scaffold fallback."
    );
    assert!(
        supports_decode,
        "DecodeToken capability is false — CUDA runtime was NOT active."
    );
}

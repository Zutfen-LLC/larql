//! `larql-compute-vulkan`
//!
//! Vulkan backend for `larql-compute`.
//!
//! As of GPU-4002 the first native kernel — a Q4_K matvec — runs on the
//! device when a Vulkan 1.1+ compute device is present, mirroring CUDA's
//! init → load → launch shape but with the SPIR-V compiled at build time
//! (see `shaders/` + `spv/` + `build.rs`). Every other op still routes
//! through the CPU reference. `supports_quant(Q4_K)` is honest: `true` only
//! when a device is present.

pub mod async_compute_backend_impl;
pub mod backend;
pub mod buffers;
pub mod decode;
pub mod kernels;
pub mod kv_dispatch_impl;
pub mod ops;
pub mod options;
pub mod trait_impl;

pub use backend::{BackendInitError, VulkanBackend};
pub use kernels::{DispatchGeometry, KernelHandle};
pub use options::BackendOptions;

pub fn vulkan_backend() -> Result<VulkanBackend, BackendInitError> {
    VulkanBackend::new()
}

pub fn vulkan_backend_with_options(
    options: BackendOptions,
) -> Result<VulkanBackend, BackendInitError> {
    VulkanBackend::with_options(options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::prelude::*;
    use larql_compute::KvIndex;
    use larql_compute::{CpuBackend, QuantFormat};
    use larql_models::test_fixtures::make_test_q4k_weights;
    use ndarray::Array2;

    fn backend() -> VulkanBackend {
        vulkan_backend().expect("vulkan backend init")
    }

    /// Returns `true` when the backend has a native Vulkan runtime. Tests
    /// that need on-device behaviour call this and `return` early on a
    /// GPU-less host (the default compile-only CI path). Set
    /// `LARQL_REQUIRE_VULKAN=1` to fail loudly instead of skipping, so a
    /// misconfigured self-hosted Vulkan runner doesn't pass vacuously.
    fn native_gate(b: &VulkanBackend) -> bool {
        if b.native_runtime_available() {
            return true;
        }
        let require = matches!(
            std::env::var("LARQL_REQUIRE_VULKAN").as_deref(),
            Ok("1" | "true" | "TRUE")
        );
        if require {
            panic!(
                "LARQL_REQUIRE_VULKAN=1 is set but no native Vulkan runtime is \
                 available; runtime_summary: {}",
                b.runtime_summary()
            );
        }
        false
    }

    #[test]
    fn constructor_returns_backend() {
        let backend = backend();
        if backend.native_runtime_available() {
            assert!(backend.name().contains("vulkan"));
        } else {
            assert!(backend.name().contains("vulkan"));
            assert!(backend.name().contains("scaffold"));
        }
    }

    // ── CPU-delegate parity (always runs, no device needed) ────────────
    // These exercise the CPU fallback path and so pass on any host.

    #[test]
    fn q4k_matvec_matches_cpu_delegate() {
        let weights = make_test_q4k_weights();
        let index = larql_compute::test_fixtures::make_q4k_fixture_index(&weights);
        let [(gate, _), _, _] = index.interleaved_kquant_layer_data(0).unwrap();
        let x = vec![0.01f32; weights.hidden_size];
        let rows = index.num_features(0);

        let got = backend()
            .q4k_matvec(gate, &x, rows, weights.hidden_size)
            .unwrap();
        let want = CpuBackend
            .q4k_matvec(gate, &x, rows, weights.hidden_size)
            .unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn q4k_matmul_matches_cpu_delegate() {
        let weights = make_test_q4k_weights();
        let index = larql_compute::test_fixtures::make_q4k_fixture_index(&weights);
        let [(gate, _), _, _] = index.interleaved_kquant_layer_data(0).unwrap();
        let seq_len = 2usize;
        let x = vec![0.05f32; seq_len * weights.hidden_size];
        let rows = index.num_features(0);

        let got = backend()
            .q4k_matmul(gate, &x, rows, weights.hidden_size, seq_len)
            .unwrap();
        let want = CpuBackend
            .q4k_matmul(gate, &x, rows, weights.hidden_size, seq_len)
            .unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn q6k_matvec_matches_cpu_delegate() {
        let rows = 3usize;
        let cols = 256usize;
        let matrix: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.004).cos()).collect();
        let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.02).sin()).collect();

        let got = backend().q6k_matvec(&q6k, &x, rows, cols).unwrap();
        let want = CpuBackend.q6k_matvec(&q6k, &x, rows, cols).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn q6k_matmul_matches_cpu_delegate() {
        let rows = 3usize;
        let cols = 256usize;
        let seq = 2usize;
        let matrix: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.004).cos()).collect();
        let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..seq * cols).map(|i| (i as f32 * 0.02).sin()).collect();

        let got = backend().q6k_matmul(&q6k, &x, rows, cols, seq).unwrap();
        let want = CpuBackend.q6k_matmul(&q6k, &x, rows, cols, seq).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn f32_gemv_topk1_matches_manual_reference() {
        let w = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.5, 0.5, 0.5],
        )
        .unwrap();
        let x = vec![0.25, 0.75, 1.0];
        let got = backend().f32_gemv_topk1(w.view(), &x).unwrap();
        assert_eq!(got.0, 2);
        assert_eq!(got.1, 4.0);
    }

    // ── Capability honesty ────────────────────────────────────────────

    #[test]
    fn supports_is_honest_without_a_device() {
        // Without a native runtime, NOTHING is supported — the contract
        // must hold so callers reliably fall back to CPU (GPU-4001 §5).
        let backend = backend();
        if backend.native_runtime_available() {
            return; // device present — covered by the native test below
        }
        assert!(!backend.supports(larql_compute::Capability::QuantMatVec));
        assert!(!backend.supports(larql_compute::Capability::F32Gemv));
        assert!(!backend.supports(larql_compute::Capability::F16Gemv));
        assert!(!backend.supports(larql_compute::Capability::DecodeToken));
        assert!(!backend.supports_quant(QuantFormat::Q4_K));
        assert!(!backend.supports_quant(QuantFormat::Q6_K));
    }

    #[test]
    fn supports_q4k_only_when_native() {
        let backend = backend();
        if !native_gate(&backend) {
            return;
        }
        // With a device: QuantMatVec + Q4_K only, nothing else.
        assert!(backend.supports(larql_compute::Capability::QuantMatVec));
        assert!(!backend.supports(larql_compute::Capability::F32Gemv));
        assert!(!backend.supports(larql_compute::Capability::F16Gemv));
        assert!(!backend.supports(larql_compute::Capability::DecodeToken));
        assert!(backend.supports_quant(QuantFormat::Q4_K));
        assert!(!backend.supports_quant(QuantFormat::Q6_K));
    }

    // ── On-device parity (GPU-4002) — runtime-gated ──────────────────

    #[test]
    fn q4k_matvec_native_matches_cpu_reference() {
        // GPU-4002 success criterion: the Vulkan q4k_matvec matches the CPU
        // reference on a device. Skipped on a GPU-less host; set
        // LARQL_REQUIRE_VULKAN=1 to force it.
        let backend = backend();
        if !native_gate(&backend) {
            eprintln!("skip: no native Vulkan runtime");
            return;
        }
        let weights = make_test_q4k_weights();
        let index = larql_compute::test_fixtures::make_q4k_fixture_index(&weights);
        let [(gate, _), _, _] = index.interleaved_kquant_layer_data(0).unwrap();
        let x = vec![0.01f32; weights.hidden_size];
        let rows = index.num_features(0);

        let got = backend
            .q4k_matvec(gate, &x, rows, weights.hidden_size)
            .unwrap();
        let want = CpuBackend
            .q4k_matvec(gate, &x, rows, weights.hidden_size)
            .unwrap();
        assert_eq!(got, want, "Vulkan q4k_matvec must match the CPU reference");
    }
}

//! `larql-compute-vulkan`
//!
//! Vulkan backend scaffold for `larql-compute`.
//!
//! Like the CUDA sibling, this crate lands the shared trait contracts and
//! backend-selection hooks first. Heavy kernel work still routes through CPU
//! reference implementations until the Vulkan compute path is filled in.

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
    use larql_compute::cpu::ops::q4_common::quantize_q6_k;
    use larql_compute::prelude::*;
    use larql_compute::KvIndex;
    use larql_compute::{CpuBackend, QuantFormat};
    use larql_models::test_fixtures::make_test_q4k_weights;
    use ndarray::Array2;

    fn backend() -> VulkanBackend {
        vulkan_backend().expect("vulkan scaffold backend")
    }

    #[test]
    fn constructor_returns_backend() {
        let backend = backend();
        assert!(backend.name().contains("vulkan"));
    }

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
        let q6k = quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.02).sin()).collect();

        let got = backend().q6k_matvec(&q6k, &x, rows, cols).unwrap();
        let want = CpuBackend.q6k_matvec(&q6k, &x, rows, cols).unwrap();
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

    #[test]
    fn supports_reports_scaffold_capabilities_honestly() {
        let backend = backend();
        assert!(!backend.supports(larql_compute::Capability::QuantMatVec));
        assert!(!backend.supports(larql_compute::Capability::F32Gemv));
        assert!(!backend.supports(larql_compute::Capability::F16Gemv));
        assert!(!backend.supports(larql_compute::Capability::DecodeToken));
        assert!(!backend.supports_quant(QuantFormat::Q6_K));
    }
}

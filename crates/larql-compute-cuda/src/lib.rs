//! `larql-compute-cuda`
//!
//! CUDA backend scaffold for `larql-compute`.
//!
//! This crate deliberately lands the control-plane shape first: backend
//! construction, shared trait conformance, dispatch metadata, and parity
//! tests. The current implementation delegates compute and K/V intents to
//! CPU/reference paths where necessary so the wider backend-selection work
//! can compile and run before CUDA kernels are brought up.

pub mod async_compute_backend_impl;
pub mod backend;
pub mod buffers;
pub mod decode;
pub mod kernels;
pub mod kv_dispatch_impl;
pub mod ops;
pub mod options;
pub mod trait_impl;

pub use backend::{BackendInitError, CudaBackend};
pub use kernels::{DispatchGeometry, KernelHandle};
pub use options::BackendOptions;

pub fn cuda_backend() -> Result<CudaBackend, BackendInitError> {
    CudaBackend::new()
}

pub fn cuda_backend_with_options(options: BackendOptions) -> Result<CudaBackend, BackendInitError> {
    CudaBackend::with_options(options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    use larql_compute::prelude::*;
    use larql_compute::KvIndex;
    use larql_compute::{CpuBackend, QuantFormat};
    use larql_models::test_fixtures::make_test_q4k_weights;
    use ndarray::Array2;

    fn backend() -> CudaBackend {
        cuda_backend().expect("cuda scaffold backend")
    }

    #[test]
    fn constructor_returns_backend() {
        let backend = backend();
        assert!(backend.name().contains("cuda"));
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
        let seq_len = 3usize;
        let x = vec![0.02f32; seq_len * weights.hidden_size];
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
        let rows = 4usize;
        let cols = 256usize;
        let matrix: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.003).sin()).collect();
        let q6k = quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).cos()).collect();

        let got = backend().q6k_matvec(&q6k, &x, rows, cols).unwrap();
        let want = CpuBackend.q6k_matvec(&q6k, &x, rows, cols).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn f32_gemv_topk1_matches_manual_reference() {
        let w = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        )
        .unwrap();
        let x = vec![0.5, 0.75, 1.0, -2.0];
        let got = backend().f32_gemv_topk1(w.view(), &x).unwrap();
        assert_eq!(got.0, 2);
        assert_eq!(got.1, 3.0);
    }

    #[test]
    fn supports_reports_scaffold_capabilities_honestly() {
        let backend = backend();
        assert!(!backend.supports(larql_compute::Capability::QuantMatVec));
        assert!(!backend.supports(larql_compute::Capability::F32Gemv));
        assert!(!backend.supports(larql_compute::Capability::F16Gemv));
        assert!(!backend.supports(larql_compute::Capability::DecodeToken));
        assert!(!backend.supports_quant(QuantFormat::Q4_K));
    }

    #[test]
    fn device_info_reports_native_or_fallback_status() {
        let info = backend().device_info();
        assert!(
            info.contains("CUDA") || info.contains("cuda"),
            "device_info should mention CUDA status: {info}"
        );
    }

    #[test]
    fn q4_input_format_routes_like_cpu() {
        // `quantize_q4_k` requires the weight element count to be a multiple
        // of 256 (one Q4_K super-block per 256 elements), so pick dimensions
        // whose product satisfies that contract.
        let cols = 128usize;
        let rows = 2usize;
        let weights: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.01).collect();
        let q4 = quantize_q4_k(&weights);
        let x = vec![0.1f32; cols];
        assert!(backend().q4k_matvec(&q4, &x, rows, cols).is_some());
    }

    #[test]
    fn native_q4k_matvec_matches_cpu_when_runtime_is_available() {
        let backend = backend();
        if !backend.native_runtime_available() {
            return;
        }

        let weights = make_test_q4k_weights();
        let index = larql_compute::test_fixtures::make_q4k_fixture_index(&weights);
        let [(gate, _), _, _] = index.interleaved_kquant_layer_data(0).unwrap();
        let x = vec![0.01f32; weights.hidden_size];
        let rows = index.num_features(0);

        let got = backend
            .native_q4k_matvec(gate, &x, rows, weights.hidden_size)
            .expect("native q4k_matvec should launch when runtime is available")
            .expect("runtime available should expose native q4k_matvec");
        let want = CpuBackend
            .q4k_matvec(gate, &x, rows, weights.hidden_size)
            .unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn native_q6k_matvec_matches_cpu_when_runtime_is_available() {
        let backend = backend();
        if !backend.native_runtime_available() {
            return;
        }

        let rows = 4usize;
        let cols = 256usize;
        let matrix: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.003).sin()).collect();
        let q6k = quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).cos()).collect();

        let got = backend
            .native_q6k_matvec(&q6k, &x, rows, cols)
            .expect("native q6k_matvec should launch when runtime is available")
            .expect("runtime available should expose native q6k_matvec");
        let want = CpuBackend.q6k_matvec(&q6k, &x, rows, cols).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn q4k_dual_matvec_matches_cpu_delegate() {
        let weights = make_test_q4k_weights();
        let index = larql_compute::test_fixtures::make_q4k_fixture_index(&weights);
        let [(gate, _), (up, _), _] = index.interleaved_kquant_layer_data(0).unwrap();
        let x = vec![0.01f32; weights.hidden_size];
        let rows = index.num_features(0);

        let (got_a, got_b) = backend()
            .q4k_dual_matvec(gate, up, &x, rows, weights.hidden_size)
            .unwrap();
        let (want_a, want_b) = CpuBackend
            .q4k_dual_matvec(gate, up, &x, rows, weights.hidden_size)
            .unwrap();
        assert_eq!(got_a, want_a);
        assert_eq!(got_b, want_b);
    }

    #[test]
    fn native_q6k_matmul_matches_cpu_when_runtime_is_available() {
        let backend = backend();
        if !backend.native_runtime_available() {
            return;
        }

        let rows = 4usize;
        let cols = 256usize;
        let seq = 3usize;
        let matrix: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.003).sin()).collect();
        let q6k = quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..seq * cols).map(|i| (i as f32 * 0.01).cos()).collect();

        let got = backend
            .native_q6k_matmul(&q6k, &x, rows, cols, seq)
            .expect("native q6k_matmul should launch when runtime is available")
            .expect("runtime available should expose native q6k_matmul");
        // CPU reference is the free `q6k_matmul_into` function; replicate it
        // via the CPU trait's matvec-per-row path through the same kernel.
        let mut want = vec![0.0f32; seq * rows];
        larql_compute::cpu::ops::q4_common::q6k_matmul_into(&mut want, &x, &q6k, rows, cols, seq);
        assert_eq!(got, want);
    }

    /// The trait-routed `QuantMatVec::q6k_matmul` must agree with the CPU
    /// free function on every host — when no CUDA runtime is present it
    /// delegates to `CpuBackend::q6k_matmul` (the amortised CPU kernel),
    /// so this always runs and pins the fallback contract.
    #[test]
    fn q6k_matmul_trait_matches_cpu_free_function() {
        use larql_compute::backend::QuantMatVec;
        let backend = backend();

        let rows = 4usize;
        let cols = 256usize;
        let seq = 3usize;
        let matrix: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.003).sin()).collect();
        let q6k = quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..seq * cols).map(|i| (i as f32 * 0.01).cos()).collect();

        let got = backend
            .q6k_matmul(&q6k, &x, rows, cols, seq)
            .expect("q6k_matmul trait method must not return None");
        let mut want = vec![0.0f32; seq * rows];
        larql_compute::cpu::ops::q4_common::q6k_matmul_into(&mut want, &x, &q6k, rows, cols, seq);
        assert_eq!(got, want);
    }

    /// When a CUDA runtime is present, the trait-routed `q6k_matmul` must
    /// pick the native kernel and match the CPU reference. Runtime-gated:
    /// no-op on hosts without CUDA (like this CI host).
    #[test]
    fn q6k_matmul_trait_native_matches_cpu_when_runtime_is_available() {
        use larql_compute::backend::QuantMatVec;
        let backend = backend();
        if !backend.native_runtime_available() {
            return;
        }

        let rows = 4usize;
        let cols = 256usize;
        let seq = 3usize;
        let matrix: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.003).sin()).collect();
        let q6k = quantize_q6_k(&matrix);
        let x: Vec<f32> = (0..seq * cols).map(|i| (i as f32 * 0.01).cos()).collect();

        let got = backend
            .q6k_matmul(&q6k, &x, rows, cols, seq)
            .expect("q6k_matmul trait method must not return None");
        let mut want = vec![0.0f32; seq * rows];
        larql_compute::cpu::ops::q4_common::q6k_matmul_into(&mut want, &x, &q6k, rows, cols, seq);
        assert_eq!(got, want);
    }

    #[test]
    fn native_q4k_dual_matvec_matches_cpu_when_runtime_is_available() {
        let backend = backend();
        if !backend.native_runtime_available() {
            return;
        }

        let weights = make_test_q4k_weights();
        let index = larql_compute::test_fixtures::make_q4k_fixture_index(&weights);
        let [(gate, _), (up, _), _] = index.interleaved_kquant_layer_data(0).unwrap();
        let x = vec![0.01f32; weights.hidden_size];
        let rows = index.num_features(0);

        let (got_a, got_b) = backend
            .native_q4k_dual_matvec(gate, up, &x, rows, weights.hidden_size)
            .expect("native q4k_dual_matvec should launch when runtime is available")
            .expect("runtime available should expose native q4k_dual_matvec");
        let (want_a, want_b) = CpuBackend
            .q4k_dual_matvec(gate, up, &x, rows, weights.hidden_size)
            .unwrap();
        assert_eq!(got_a, want_a);
        assert_eq!(got_b, want_b);
    }

    #[test]
    fn native_q4k_matmul_matches_cpu_when_runtime_is_available() {
        let backend = backend();
        if !backend.native_runtime_available() {
            return;
        }

        let weights = make_test_q4k_weights();
        let index = larql_compute::test_fixtures::make_q4k_fixture_index(&weights);
        let [(gate, _), _, _] = index.interleaved_kquant_layer_data(0).unwrap();
        let seq_len = 3usize;
        let x = vec![0.02f32; seq_len * weights.hidden_size];
        let rows = index.num_features(0);

        let got = backend
            .native_q4k_matmul(gate, &x, rows, weights.hidden_size, seq_len)
            .expect("native q4k_matmul should launch when runtime is available")
            .expect("runtime available should expose native q4k_matmul");
        let want = CpuBackend
            .q4k_matmul(gate, &x, rows, weights.hidden_size, seq_len)
            .unwrap();
        assert_eq!(got, want);
    }
}

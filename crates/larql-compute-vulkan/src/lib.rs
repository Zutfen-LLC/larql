//! `larql-compute-vulkan`
//!
//! Vulkan backend for `larql-compute`.
//!
//! As of GPU-4003 the device-resident FFN side chain — RMSNorm →
//! gate/up Q4_K matmul → GEGLU → down Q4_K matmul → residual add — runs
//! on the device when a Vulkan 1.1+ compute device is present, mirroring
//! CUDA's init → load → launch shape but with the SPIR-V compiled at
//! build time (see `shaders/` + `spv/` + `build.rs`). Attention, MoE, and
//! dense GEMV stay on the CPU reference (deferred per `docs/vulkan-mvp.md`
//! §7). `supports_quant(Q4_K)` is honest: `true` only when a device is
//! present; only `QuantMatVec` is advertised until a full decode path
//! lands.

pub mod async_compute_backend_impl;
pub mod backend;
pub mod buffers;
pub mod chain_impl;
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

    // ── FFN chain parity (GPU-4003) — runtime-gated ──────────────────
    //
    // These exercise the device-resident chain on a Vulkan device and the
    // CPU-fallback path otherwise. On-device parity is cross-precision
    // (f32 GPU vs f64 CPU rms_norm), so a relative tolerance is used.

    use larql_compute::cpu::ops::q4_common::quantize_q4_k;

    /// CPU reference for the full FFN chain, flat-input variant.
    #[allow(clippy::too_many_arguments)]
    fn cpu_ffn_chain(
        x: &[f32],
        norm_w: &[f32],
        gate_w: &[u8],
        up_w: &[u8],
        down_w: &[u8],
        hidden: usize,
        inter: usize,
        seq: usize,
        offset: f32,
        eps: f32,
    ) -> Vec<f32> {
        use larql_compute::CpuBackend;
        use larql_compute::QuantMatVec;
        let cpu = CpuBackend;
        // rms_norm
        let mut h_norm = vec![0.0f32; seq * hidden];
        for s in 0..seq {
            let base = s * hidden;
            let sqsum: f64 = x[base..base + hidden].iter().map(|&v| (v as f64) * (v as f64)).sum();
            let rms = (sqsum / hidden as f64 + eps as f64).sqrt() as f32;
            for j in 0..hidden {
                h_norm[base + j] = x[base + j] / rms * (offset + norm_w[j]);
            }
        }
        let gate_out = cpu.q4k_matmul(gate_w, &h_norm, inter, hidden, seq).unwrap();
        let up_out = cpu.q4k_matmul(up_w, &h_norm, inter, hidden, seq).unwrap();
        let act = larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate_out, &up_out);
        let down_out = cpu.q4k_matmul(down_w, &act, hidden, inter, seq).unwrap();
        let mut out = vec![0.0f32; seq * hidden];
        for i in 0..seq * hidden {
            out[i] = x[i] + down_out[i];
        }
        out
    }

    #[test]
    fn rms_norm_chain_matches_cpu_reference() {
        let backend = backend();
        if !native_gate(&backend) {
            eprintln!("skip: no native Vulkan runtime");
            return;
        }
        let seq = 2usize;
        let hidden = 256usize;
        let x: Vec<f32> = (0..seq * hidden).map(|i| ((i as f32) * 0.01).sin()).collect();
        let w = vec![1.0f32; hidden];
        let got = backend.rms_norm_chain(&x, &w, hidden, seq, 0.0, 1e-6);
        // CPU reference via the shared residual fn (reshape to Array2).
        let x_arr = ndarray::Array2::from_shape_vec((seq, hidden), x.clone()).unwrap();
        let want_arr = larql_compute::residual::rms_norm_eps(&x_arr, Some(&w), 0.0, 1e-6);
        let want = want_arr.as_slice().unwrap();
        let max_diff = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "rms_norm max diff {} exceeds tolerance", max_diff);
    }

    #[test]
    fn geglu_chain_matches_cpu_reference() {
        let backend = backend();
        if !native_gate(&backend) {
            eprintln!("skip: no native Vulkan runtime");
            return;
        }
        let len = 512usize;
        let gate: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.03).sin()).collect();
        let up: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.05).cos()).collect();
        let got = backend.geglu_silu_chain(&gate, &up, len);
        let want = larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate, &up);
        let max_diff = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "geglu max diff {} exceeds tolerance", max_diff);
    }

    #[test]
    fn residual_add_chain_matches_cpu_reference() {
        let backend = backend();
        if !native_gate(&backend) {
            eprintln!("skip: no native Vulkan runtime");
            return;
        }
        let len = 512usize;
        let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32) * -0.07).collect();
        let got = backend.residual_add_chain(&a, &b, len);
        let want: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let max_diff = got.iter().zip(want.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "residual_add max diff {} exceeds tolerance", max_diff);
    }

    #[test]
    fn ffn_chain_native_matches_cpu_reference() {
        // GPU-4003 success criterion: the device-resident FFN chain matches
        // the CPU reference. Skipped on a GPU-less host; set
        // LARQL_REQUIRE_VULKAN=1 to force it.
        let backend = backend();
        if !native_gate(&backend) {
            eprintln!("skip: no native Vulkan runtime");
            return;
        }
        let hidden = 256usize; // multiple of 256 for Q4_K blocks
        let inter = 512usize;
        let seq = 2usize;

        let x: Vec<f32> = (0..seq * hidden).map(|i| ((i as f32) * 0.01).sin()).collect();
        let norm_w = vec![1.0f32; hidden];
        let gate_f: Vec<f32> = (0..inter * hidden).map(|i| ((i as f32) * 0.001).sin()).collect();
        let up_f: Vec<f32> = (0..inter * hidden).map(|i| ((i as f32) * 0.001).cos()).collect();
        let down_f: Vec<f32> = (0..hidden * inter).map(|i| ((i as f32) * 0.001).sin()).collect();
        let gate_w = quantize_q4_k(&gate_f);
        let up_w = quantize_q4_k(&up_f);
        let down_w = quantize_q4_k(&down_f);

        let got = backend.ffn_chain(
            &x, &norm_w, &gate_w, &up_w, &down_w, hidden, inter, seq, 0.0, 1e-6,
        );
        let want = cpu_ffn_chain(
            &x, &norm_w, &gate_w, &up_w, &down_w, hidden, inter, seq, 0.0, 1e-6,
        );
        // Cross-precision + Q4_K quant noise: allow a few percent relative
        // error on the dominant magnitude.
        let max_abs = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let max_rel = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| ((a - b).abs()) / max_abs)
            .fold(0.0f32, f32::max);
        assert!(
            max_rel < 0.05,
            "FFN chain max relative error {:.4} exceeds 5% tolerance", max_rel
        );
    }

    #[test]
    fn ffn_chain_cpu_fallback_matches_reference_when_no_device() {
        // The fallback path runs on every host and must be bit-identical to
        // the CPU reference (no quant on the norm path; Q4_K on matmuls).
        let backend = backend();
        if backend.native_runtime_available() {
            return; // device present — covered by the native test above
        }
        let hidden = 256usize;
        let inter = 512usize;
        let seq = 2usize;
        let x: Vec<f32> = (0..seq * hidden).map(|i| ((i as f32) * 0.01).sin()).collect();
        let norm_w = vec![1.0f32; hidden];
        let gate_f: Vec<f32> = (0..inter * hidden).map(|i| ((i as f32) * 0.001).sin()).collect();
        let up_f: Vec<f32> = (0..inter * hidden).map(|i| ((i as f32) * 0.001).cos()).collect();
        let down_f: Vec<f32> = (0..hidden * inter).map(|i| ((i as f32) * 0.001).sin()).collect();
        let gate_w = quantize_q4_k(&gate_f);
        let up_w = quantize_q4_k(&up_f);
        let down_w = quantize_q4_k(&down_f);

        let got = backend.ffn_chain(
            &x, &norm_w, &gate_w, &up_w, &down_w, hidden, inter, seq, 0.0, 1e-6,
        );
        let want = cpu_ffn_chain(
            &x, &norm_w, &gate_w, &up_w, &down_w, hidden, inter, seq, 0.0, 1e-6,
        );
        assert_eq!(got, want, "CPU fallback must be bit-identical to the reference");
    }
}

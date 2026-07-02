use half::f16;
use larql_compute::backend::{Capability, ComputeBackend, DecodeBackend, MatMul, QuantMatVec};
use larql_compute::CpuBackend;
use ndarray::{Array2, ArrayView2};

use crate::CudaBackend;

const CPU: CpuBackend = CpuBackend;

/// FLOP threshold mirroring Metal's `calibration::DEFAULT_FLOP_THRESHOLD`.
/// Below this the htod-upload + sync + dtoh-download round-trip costs more
/// than the CPU reference loop, so the native kernel is skipped and the CPU
/// fallback runs. This keeps the per-token dense lm_head decode path on the
/// zero-copy mmap CPU path instead of re-uploading gigabytes every token.
/// (The CPU fallback is reached via the `MatMul` trait's default
/// `matmul_transb`-based path in the caller; `f32_gemv` returning `None`
/// there is the intended fast-path-fallback signal.)
const GEMV_FLOP_THRESHOLD: usize = 500_000_000;

impl MatMul for CudaBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        CPU.matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        CPU.matmul_transb(a, b)
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let n = w.nrows();
        let k = w.ncols();
        // Below the flop threshold the htod + sync + dtoh round-trip costs
        // more than the CPU loop, so skip the native kernel and let the
        // caller fall back to the `matmul_transb` CPU path.
        if 2 * n * k < GEMV_FLOP_THRESHOLD {
            return None;
        }
        // Try the native path only when the view is row-major contiguous —
        // the launcher uploads a flat `[n, k]` slice. Non-contiguous views
        // (e.g. transposed) fall through to the CPU reference.
        if let Some(w_slice) = w.as_slice() {
            if let Ok(Some(native)) = self.native_f32_gemv(w_slice, x, n, k) {
                return Some(native);
            }
        }
        let mut out = Vec::with_capacity(n);
        for row in w.rows() {
            out.push(row.iter().zip(x).map(|(a, b)| *a * *b).sum());
        }
        Some(out)
    }

    /// Not specialised on CUDA yet. Returning `None` (the trait default)
    /// keeps the greedy-decode top-1 path on the CPU reference / fused path
    /// instead of the un-fused full-upload + full-readback + CPU argmax that
    /// `f32_gemv` would otherwise force. A native fused-argmax kernel is the
    /// Phase 4 follow-up that will override this.
    fn f32_gemv_topk1(&self, _w: ArrayView2<f32>, _x: &[f32]) -> Option<(u32, f32)> {
        None
    }

    fn f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        // Below the flop threshold the htod + sync + dtoh round-trip costs
        // more than the CPU loop; skip the native kernel (see `f32_gemv`).
        if 2 * n * k < GEMV_FLOP_THRESHOLD {
            return None;
        }
        if let Ok(Some(native)) = self.native_f16_gemv(w_f16, x, n, k) {
            return Some(native);
        }
        let mut out = vec![0.0f32; n];
        for (row, out_row) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            let row_off = 2 * (row * k);
            for (col, &xv) in x.iter().enumerate().take(k) {
                let off = row_off + 2 * col;
                let bits = u16::from_le_bytes([w_f16[off], w_f16[off + 1]]);
                acc += f16::from_bits(bits).to_f32() * xv;
            }
            *out_row = acc;
        }
        Some(out)
    }

    /// Not specialised on CUDA yet — see `f32_gemv_topk1`.
    fn f16_gemv_topk1(
        &self,
        _w_f16: &[u8],
        _x: &[f32],
        _n: usize,
        _k: usize,
    ) -> Option<(u32, f32)> {
        None
    }

    fn f16_gemv_topk(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        // Gated by the same flop threshold as `f16_gemv`; below it the
        // `f16_gemv` call returns `None` and this returns `None` too.
        let mut pairs: Vec<_> = self
            .f16_gemv(w_f16, x, n, k)?
            .into_iter()
            .enumerate()
            .map(|(idx, score)| (idx as u32, score))
            .collect();
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
        pairs.truncate(top_k);
        Some(pairs)
    }
}

impl QuantMatVec for CudaBackend {
    fn q4_matvec(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        CPU.q4_matvec(q4_data, q8_x, q8_scales, num_rows, hidden)
    }

    fn q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        CPU.q4_vecmat(activation, q4_data, intermediate, hidden)
    }

    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if let Ok(Some(native)) = self.native_q4k_matvec(q4k_data, x, num_rows, hidden) {
            return Some(native);
        }
        CPU.q4k_matvec(q4k_data, x, num_rows, hidden)
    }

    fn q4k_dual_matvec(
        &self,
        q4k_a: &[u8],
        q4k_b: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        if let Ok(Some(native)) = self.native_q4k_dual_matvec(q4k_a, q4k_b, x, num_rows, hidden) {
            return Some(native);
        }
        CPU.q4k_dual_matvec(q4k_a, q4k_b, x, num_rows, hidden)
    }

    fn q4k_matmul(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        if let Ok(Some(native)) = self.native_q4k_matmul(q4k_data, x, num_rows, hidden, seq_len) {
            return Some(native);
        }
        CPU.q4k_matmul(q4k_data, x, num_rows, hidden, seq_len)
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if let Ok(Some(native)) = self.native_q6k_matvec(q6k_data, x, num_rows, hidden) {
            return Some(native);
        }
        CPU.q6k_matvec(q6k_data, x, num_rows, hidden)
    }

    fn q6k_matmul(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        if let Ok(Some(native)) = self.native_q6k_matmul(q6k_data, x, num_rows, hidden, seq_len) {
            return Some(native);
        }
        CPU.q6k_matmul(q6k_data, x, num_rows, hidden, seq_len)
    }

    fn supports_quant(&self, format: larql_compute::QuantFormat) -> bool {
        let _ = format;
        false
    }
}

impl DecodeBackend for CudaBackend {}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        if self.native_runtime_available() {
            "cuda (native k-quant + gemv + cpu fallback)"
        } else {
            "cuda (cpu-delegate scaffold)"
        }
    }

    fn device_info(&self) -> String {
        self.runtime_summary().to_string()
    }

    fn supports(&self, cap: Capability) -> bool {
        let _ = cap;
        false
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

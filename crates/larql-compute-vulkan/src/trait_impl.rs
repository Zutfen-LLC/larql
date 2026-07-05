use half::f16;
use larql_compute::backend::{Capability, ComputeBackend, DecodeBackend, MatMul, QuantMatVec};
use larql_compute::CpuBackend;
use larql_compute::QuantFormat;
use ndarray::{Array2, ArrayView2};

use crate::VulkanBackend;

const CPU: CpuBackend = CpuBackend;

impl MatMul for VulkanBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        CPU.matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        CPU.matmul_transb(a, b)
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let mut out = Vec::with_capacity(w.nrows());
        for row in w.rows() {
            out.push(row.iter().zip(x).map(|(a, b)| *a * *b).sum());
        }
        Some(out)
    }

    fn f32_gemv_topk1(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<(u32, f32)> {
        self.f32_gemv(w, x).and_then(|scores| {
            scores
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(idx, score)| (idx as u32, score))
        })
    }

    fn f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
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

    fn f16_gemv_topk1(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<(u32, f32)> {
        self.f16_gemv(w_f16, x, n, k).and_then(|scores| {
            scores
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(idx, score)| (idx as u32, score))
        })
    }

    fn f16_gemv_topk(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
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

impl QuantMatVec for VulkanBackend {
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
        // Native Vulkan path: only when a device is present and the shape is
        // a valid Q4_K block multiple (256). On any failure (launch error,
        // no device, non-multiple hidden) fall through to the CPU reference
        // — correctness always wins (GPU-4001 §5 rule 4).
        if let Some(rt) = self.runtime() {
            if hidden > 0 && hidden.is_multiple_of(256) {
                if let Ok(out) = rt.launch_q4k_matvec(q4k_data, x, num_rows, hidden) {
                    return Some(out);
                }
            }
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
        // seq == 1 collapses to the matvec path; the dedicated matmul shader
        // lands in a follow-up task.
        if seq_len == 1 {
            return self.q4k_matvec(q4k_data, x, num_rows, hidden);
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
        CPU.q6k_matmul(q6k_data, x, num_rows, hidden, seq_len)
    }

    fn supports_quant(&self, format: QuantFormat) -> bool {
        // Honest: only Q4_K is native, and only when a device is present
        // (GPU-4001 §5 rule 1-2). Every other format stays CPU.
        self.native_runtime_available() && matches!(format, QuantFormat::Q4_K)
    }
}

impl DecodeBackend for VulkanBackend {}

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str {
        if self.native_runtime_available() {
            "vulkan (native q4_k matvec; remaining ops CPU fallback)"
        } else {
            "vulkan (cpu-delegate scaffold)"
        }
    }

    fn device_info(&self) -> String {
        self.runtime_summary().to_string()
    }

    fn supports(&self, cap: Capability) -> bool {
        // Only QuantMatVec is advertised, native-gated (GPU-4001 §5 rule 1,3).
        self.native_runtime_available() && matches!(cap, Capability::QuantMatVec)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

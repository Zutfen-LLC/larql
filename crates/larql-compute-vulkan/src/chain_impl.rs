//! Public FFN-chain surface on [`crate::VulkanBackend`] (GPU-4003).
//!
//! These inherent methods are the host-facing entry points for the
//! device-resident FFN chain. Each routes to the native Vulkan runtime
//! when one is present and falls back to the CPU reference (reusing
//! `larql-compute`'s shared elementwise / quantisation references)
//! otherwise — the "correct path always exists" rule from the MVP
//! capability contract (`docs/vulkan-mvp.md` §5 rule 4).
//!
//! Attention and MoE are intentionally NOT exposed here: the MVP
//! defers them (§7), and [`crate::VulkanBackend`] advertises only
//! `QuantMatVec` via `supports()` until a full decode path lands.

use crate::VulkanBackend;

impl VulkanBackend {
    /// Device-resident RMSNorm when a runtime is present; CPU reference
    /// otherwise. Mirrors `larql_compute::residual::rms_norm_eps` on the
    /// fallback path so results are bit-identical to the CPU backend off-device.
    ///
    /// `x` is flat `[seq * hidden]`; returns flat `[seq * hidden]`.
    pub fn rms_norm_chain(
        &self,
        x: &[f32],
        w: &[f32],
        hidden: usize,
        seq: usize,
        offset: f32,
        eps: f32,
    ) -> Vec<f32> {
        if let Some(rt) = self.runtime() {
            if let Ok(out) = rt.launch_rms_norm(x, w, hidden, seq, offset, eps) {
                return out;
            }
        }
        // CPU fallback — reuse the shared larql-compute reference so the
        // off-device path is bit-identical to CpuBackend's RMSNorm.
        cpu_rms_norm(x, w, hidden, seq, offset, eps)
    }

    /// Device-residual GEGLU (SiLU-gated) activation, `silu(gate) * up`.
    pub fn geglu_silu_chain(&self, gate: &[f32], up: &[f32], len: usize) -> Vec<f32> {
        if let Some(rt) = self.runtime() {
            if let Ok(out) = rt.launch_geglu_silu(gate, up, len) {
                return out;
            }
        }
        // CPU fallback: `larql_compute::cpu::ops::geglu::geglu_silu_alloc`.
        larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate[..len], &up[..len])
    }

    /// Device-residual add, `a + b`. Element-wise over `len`.
    pub fn residual_add_chain(&self, a: &[f32], b: &[f32], len: usize) -> Vec<f32> {
        if let Some(rt) = self.runtime() {
            if let Ok(out) = rt.launch_residual_add(a, b, len) {
                return out;
            }
        }
        let mut out = vec![0.0f32; len];
        for i in 0..len {
            out[i] = a[i] + b[i];
        }
        out
    }

    /// Device-resident full FFN side chain:
    /// `rms_norm -> gate_matmul -> up_matmul -> geglu -> down_matmul -> residual`.
    ///
    /// `x` is flat `[seq * hidden]` residual input; returns flat `[seq * hidden]`.
    /// `gate_w` / `up_w` are Q4_K blobs of shape `[inter, hidden]`;
    /// `down_w` is Q4_K `[hidden, inter]`.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_chain(
        &self,
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
        if let Some(rt) = self.runtime() {
            if let Ok(out) =
                rt.launch_ffn_chain(x, norm_w, gate_w, up_w, down_w, hidden, inter, seq, offset, eps)
            {
                return out;
            }
        }
        // CPU fallback: compose the shared references step by step.
        let h_norm = cpu_rms_norm(x, norm_w, hidden, seq, offset, eps);
        use larql_compute::CpuBackend;
        use larql_compute::QuantMatVec;
        let cpu = CpuBackend;
        let gate_out = cpu
            .q4k_matmul(gate_w, &h_norm, inter, hidden, seq)
            .unwrap_or_default();
        let up_out = cpu
            .q4k_matmul(up_w, &h_norm, inter, hidden, seq)
            .unwrap_or_default();
        let act = larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate_out, &up_out);
        let down_out = cpu
            .q4k_matmul(down_w, &act, hidden, inter, seq)
            .unwrap_or_default();
        let mut out = vec![0.0f32; seq * hidden];
        for i in 0..seq * hidden {
            out[i] = x[i] + down_out[i];
        }
        out
    }
}

/// Shared CPU RMSNorm reference for the fallback path. Reimplemented here
/// as a slice/flat-input variant because `larql_compute::residual::rms_norm_eps`
/// takes an `ArrayView2<f32>`; the chain works in flat `&[f32]` to match
/// the device launch signatures. The math is identical (f64 accumulation).
#[allow(clippy::too_many_arguments)]
fn cpu_rms_norm(
    x: &[f32],
    w: &[f32],
    hidden: usize,
    seq: usize,
    offset: f32,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * hidden];
    for s in 0..seq {
        let base = s * hidden;
        let sqsum: f64 = x[base..base + hidden]
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum();
        let rms = (sqsum / hidden as f64 + eps as f64).sqrt() as f32;
        for j in 0..hidden {
            out[base + j] = x[base + j] / rms * (offset + w[j]);
        }
    }
    out
}

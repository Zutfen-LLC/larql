use half::f16;
use larql_compute::backend::{Capability, ComputeBackend, DecodeBackend, MatMul, QuantMatVec};
use larql_compute::CpuBackend;
use larql_compute::{DecodeStateDump, StateDumpMask};
use ndarray::{Array2, ArrayView2};

use crate::options::native_thresholds;
use crate::CudaBackend;

const CPU: CpuBackend = CpuBackend;

// Below the GEMV flop threshold the htod-upload + sync + dtoh-download
// round-trip costs more than the CPU reference loop, so the native kernel is
// skipped and the CPU fallback runs. This keeps the per-token dense lm_head
// decode path on the zero-copy mmap CPU path instead of re-uploading
// gigabytes every token. The threshold lives in `options::native_thresholds()`
// (env `LARQL_CUDA_GEMV_FLOP_THRESHOLD`, default `500_000_000`, mirroring
// Metal's `DEFAULT_FLOP_THRESHOLD`); `f32_gemv` returning `None` below it is
// the intended fast-path-fallback signal.

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
        if 2 * n * k < native_thresholds().gemv_flop_threshold {
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
        if 2 * n * k < native_thresholds().gemv_flop_threshold {
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
        if let Ok(Some(native)) = self.native_q4_matvec(q4_data, q8_x, q8_scales, num_rows, hidden)
        {
            return Some(native);
        }
        CPU.q4_matvec(q4_data, q8_x, q8_scales, num_rows, hidden)
    }

    fn q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if let Ok(Some(native)) = self.native_q4_vecmat(activation, q4_data, intermediate, hidden) {
            return Some(native);
        }
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
        // The host-orchestrated decode/prefill pipeline (Session 11) routes
        // every Q4_K / Q6_K projection through the native CUDA matvec / matmul
        // kernels (with CPU fallback when no runtime is present). Advertise
        // only the formats `build_pipeline_layers` actually produces for the
        // fused path — Q4_K (gate/up/O) and Q6_K (down/V in a default
        // vindex). Q4_KF is intentionally NOT advertised: the builders reject
        // it (`resolve_*_weights` panic on the tag), and the native CUDA
        // q4k kernels decode the 144-byte Q4_K super-block, not Q4_KF's
        // 160-byte layout — so advertising it would be both undeliverable and
        // numerically wrong. The scaffold path keeps `false` so
        // `fused_prefill` / `fused_decode_step` bail to the CPU reference.
        if !self.native_runtime_available() {
            return false;
        }
        matches!(
            format,
            larql_compute::QuantFormat::Q4_K | larql_compute::QuantFormat::Q6_K
        )
    }
}

impl DecodeBackend for CudaBackend {
    /// CUDA reports KV-cache support only when a native runtime is present
    /// *and* a cache has been pre-allocated. The scaffold path (no device)
    /// returns `false` so engines route through the CPU reference store.
    fn has_kv_cache(&self) -> bool {
        self.native_runtime_available() && self.kv_cache_allocated()
    }

    fn reset_kv_cache(&self) {
        self.reset_kv_cache_native();
        // Also clear the host KV mirror so the next prefill starts clean.
        // (The host mirror is the source of truth for the host-orchestrated
        // attention path; the device cache is kept in lockstep for the
        // `DecodeBackend` lifecycle contract.)
        self.reset_host_kv(0);
    }

    fn kv_cache_len(&self) -> usize {
        // The host KV mirror is the source of truth for position/residual: it
        // is read by the host-orchestrated attention path and kept in lockstep
        // with the device-resident activation/attention chain (which drives
        // its own decode_attention / prefill_attention kernels against the
        // device K/V buffers). Return the host mirror's committed length so
        // callers tracking decode position stay consistent with what attention
        // actually sees. Falls back to the device cursor when the host mirror
        // is empty (e.g. a caller queries length before any prefill on a
        // backend that populated the device cache through another path).
        let host_len = self.host_kv_len();
        if host_len > 0 {
            host_len
        } else {
            self.kv_cache_len_native()
        }
    }

    fn truncate_kv_cache(&self, len: usize) {
        self.truncate_kv_cache_native(len);
        // Mirror the truncation on the host KV so the host attention path and
        // `kv_cache_len` see the rolled-back length. Rows below `len` are
        // preserved (the host mirror keeps its `[len, kv_dim]` prefix).
        let mut kv = self.lock_host_kv();
        for (k_cache, v_cache) in kv.iter_mut() {
            let prev = k_cache.shape()[0];
            if len < prev {
                // Take ownership of the prefix slice in one allocation. Array2
                // has no in-place row truncate; the per-step decode rebuild
                // already pays this shape, and truncate is rare (only iterative
                // predispatch). `to_owned()` does one copy instead of the old
                // zero-init + slice-assign double write.
                *k_cache = k_cache.slice(ndarray::s![..len, ..]).to_owned();
                *v_cache = v_cache.slice(ndarray::s![..len, ..]).to_owned();
            }
        }
    }

    fn preallocate_kv_cache_per_layer(&self, shapes: &[(usize, usize)], max_seq: usize) {
        // Only attempt a device allocation when a runtime is present; the
        // scaffold path leaves the cache `None` and `has_kv_cache` reports
        // false, so engines route KV through the CPU reference store.
        if !self.native_runtime_available() {
            return;
        }
        // A failed allocation is non-fatal: leave the cache unallocated and
        // let `has_kv_cache` report false so the caller's KV path falls back
        // to the host reference store.
        let _ = self.preallocate_kv_cache(shapes, max_seq);
    }

    /// Populate the device KV cache for one layer with prefill K/V data.
    /// All `seq_len` rows are uploaded in a single host→device transfer and
    /// written by one kernel launch (no per-row stalls). When no runtime /
    /// cache is present this is a no-op (the host reference store remains
    /// the source of truth). A launch failure leaves the layer's cursor
    /// unchanged (the caller's KV path falls back to the host store).
    fn populate_kv_layer(
        &self,
        layer: usize,
        k_data: &[f32],
        v_data: &[f32],
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        // Checked geometry: a dim overflow would otherwise bypass the length
        // guard and panic on a host slice index. On `None` (overflow), treat
        // as short data and return early (no-op / host-store fallback).
        let row_elems = match num_kv_heads.checked_mul(head_dim) {
            Some(r) => r,
            None => return,
        };
        let block_len = match seq_len.checked_mul(row_elems) {
            Some(b) => b,
            None => return,
        };
        if k_data.len() < block_len || v_data.len() < block_len {
            return;
        }
        let k_block = &k_data[..block_len];
        let v_block = &v_data[..block_len];
        let _ = self.native_kv_append(layer, k_block, v_block, 0, seq_len);
    }

    /// Single-token fused decode. Runs the host-orchestrated pipeline
    /// (`pipeline::host_decode_token`): every Q/K/V/O + gate/up/down
    /// projection dispatches through the native CUDA q4k/q6k matvec kernels
    /// (CPU fallback when no runtime), with elementwise ops + GQA attention
    /// on host. `None` when no runtime is present (scaffold path — the caller
    /// routes through the CPU reference) or when a layer uses a feature this
    /// path doesn't handle (MoE / PLE / remote FFN / non-k-quant formats).
    fn decode_token(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
    ) -> Option<Vec<f32>> {
        if !self.native_runtime_available() {
            return None;
        }
        // RoPE position from the host KV mirror (the source of truth for the
        // host-orchestrated attention). The device cursor
        // (`kv_cache_len_native`) is only advanced by `prefill_kquant`, NOT
        // by decode — so using it here would freeze the position at the
        // post-prefill length and garble multi-token RoPE. The host mirror
        // is grown by one row per decode step inside `host_decode_token` /
        // `host_decode_token_resident`.
        //
        // GPU-007: the non-state-dump path threads the hidden state
        // device-resident through eligible layers via
        // `host_decode_token_resident` (which falls back per-layer to
        // `host_decode_token`'s host-orchestrated path). The state-dump
        // variants below keep the host path (exact dump points).
        let pos = self.host_kv_len();
        self.host_decode_token_resident(layers, x, hidden, inter, pos)
    }

    fn decode_token_with_state_dump(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        state: Option<&mut DecodeStateDump>,
    ) -> Option<Vec<f32>> {
        self.decode_token_with_state_dump_masked(
            layers,
            x,
            hidden,
            inter,
            state,
            StateDumpMask::Full,
        )
    }

    fn decode_token_with_state_dump_masked(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        state: Option<&mut DecodeStateDump>,
        mask: StateDumpMask,
    ) -> Option<Vec<f32>> {
        if !self.native_runtime_available() {
            return None;
        }
        let pos = self.host_kv_len();
        self.host_decode_token(layers, x, hidden, inter, pos, state, mask)
    }

    /// Multi-position prefill. Runs the host-orchestrated pipeline
    /// (`pipeline::host_prefill_kquant`): projections use the amortised
    /// native CUDA q4k/q6k matmul kernels across all `seq_len` positions;
    /// causal attention + elementwise ops on host. Populates the host KV
    /// mirror (and the device cache via `populate_kv_layer` in lockstep).
    /// `None` on the scaffold path or for unsupported layer features.
    fn prefill_kquant(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        _use_qk_norm: bool,
        softcap: f32,
    ) -> Option<Vec<f32>> {
        if !self.native_runtime_available() {
            return None;
        }
        let h = self.host_prefill_kquant(layers, x, hidden, inter, seq_len, softcap)?;
        // Populate the device KV cache in lockstep with the host mirror so
        // `has_kv_cache` / `kv_cache_len` reflect the post-prefill state and
        // subsequent `decode_token` calls see the right `pos`.
        {
            let kv = self.lock_host_kv();
            for (li, slot) in kv.iter().enumerate() {
                let (k_cache, v_cache) = slot;
                let k_flat: Vec<f32> = k_cache.iter().cloned().collect();
                let v_flat: Vec<f32> = v_cache.iter().cloned().collect();
                if let Some(layer) = layers.get(li) {
                    self.populate_kv_layer(
                        li,
                        &k_flat,
                        &v_flat,
                        k_cache.shape()[0],
                        layer.num_kv_heads,
                        layer.head_dim,
                    );
                }
            }
        }
        Some(h)
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        if self.native_runtime_available() {
            "cuda (native k-quant + gemv + host-orchestrated decode/prefill + cpu fallback)"
        } else {
            "cuda (cpu-delegate scaffold)"
        }
    }

    fn device_info(&self) -> String {
        let mut info = self.runtime_summary().to_string();
        // Surface the weight-cache hit rate + resident footprint when the
        // diagnostics opt-in is set (presence-as-truth). Default stays quiet.
        if crate::options::gpu_diag_enabled() {
            if let Some(diag) = self.weight_cache_diag() {
                info.push('\n');
                info.push_str(&diag);
            }
            if let Some(diag) = self.resident_kv_diag() {
                info.push('\n');
                info.push_str(&diag);
            }
            if let Some(diag) = self.resident_hidden_diag() {
                info.push('\n');
                info.push_str(&diag);
            }
        }
        info
    }

    fn supports(&self, cap: Capability) -> bool {
        // The host-orchestrated decode/prefill pipeline (Session 11) makes
        // `decode_token` / `prefill_kquant` live when a native runtime is
        // present, so advertise the corresponding capabilities. The scaffold
        // path keeps everything `false` so callers route through the CPU
        // reference. `QuantMatVec` is advertised because the per-format
        // `supports_quant` reports Q4_K / Q6_K under the same gate.
        if !self.native_runtime_available() {
            return false;
        }
        matches!(
            cap,
            Capability::QuantMatVec | Capability::DecodeToken | Capability::PrefillQ4
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Reachable through `&dyn ComputeBackend` / `Box<dyn ComputeBackend>` so a
    /// long-lived browse path can clear the address-keyed weight cache at a
    /// vindex-rebind boundary without downcasting to `CudaBackend`. Delegates to
    /// the inherent `CudaBackend::flush_weight_cache` (fully-qualified so there
    /// is no ambiguity with this trait method). Safe on the no-runtime scaffold
    /// path, where it is a no-op (`runtime` is `None`).
    fn flush_weight_cache(&self) {
        CudaBackend::flush_weight_cache(self);
    }
}

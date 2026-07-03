//! Host-orchestrated fused decode/prefill pipeline for `CudaBackend`.
//!
//! This is the Phase 4 slice that unblocks capability advertisement
//! (`supports_quant(Q4_K)` / `supports(DecodeToken)` / `supports(PrefillQ4)`)
//! and routes `kquant_forward::cached::fused_prefill` / `fused_decode_step`
//! through the CUDA backend.
//!
//! Design — "host-orchestrated pipeline":
//!
//! - The heavy compute (every Q/K/V/O + gate/up/down projection) runs on the
//!   GPU via the existing native CUDA q4k/q6k matvec kernels, reached through
//!   the `QuantMatVec` trait (`self.quant_matvec(...)` → native-then-CPU
//!   fallback, exactly the path Session 4-9 wired).
//! - Elementwise ops (RMSNorm / QK-norm / RoPE / GQA softmax / V-norm /
//!   GEGLU activation / residual adds / per-layer scalar) run on the host
//!   using the shared `larql_compute` primitives — the same math the CPU
//!   reference uses, so the output is numerically identical to the CPU
//!   `predict_kquant_decode_step_direct` path when the native kernels match
//!   their CPU twins (parity-tested).
//! - A host-side KV mirror (`host_kv` on `CudaBackend`) holds the per-layer
//!   `(K_cache, V_cache)` `[len, kv_dim]` arrays the host attention reads.
//!   The device `CudaKVCache` from Session 10 is populated in parallel via
//!   `populate_kv_layer` so the `DecodeBackend` lifecycle contract stays
//!   consistent, but attention itself reads the host mirror — the simplest
//!   correct path before device-side attention kernels land.
//!
//! What this is NOT: a single-command-buffer fused pipeline (Metal's shape).
//! Each matvec is a separate htod/launch/dtoh round-trip. That is the
//! intentional Session 11 scope — it makes `fused_prefill` / `fused_decode_step`
//! route through CUDA *today* (unblocking the `walk`/`bench` fast paths and
//! the `auto` policy on Linux) and gives a parity oracle for the future
//! fully-fused pipeline. Folding the elementwise ops into device kernels +
//! collapsing the round-trips is the follow-on work.

use larql_compute::attention::build_rope_inv_freq;
use larql_compute::attention::decode::gqa_attention_decode_step;
use larql_compute::attention::rope::apply_rope_partial_at_full;
use larql_compute::cpu::ops::moe::cpu_moe_forward;
use larql_compute::cpu::ops::outer_combine::outer_post_norm_residual;
use larql_compute::residual::{
    layer_norm_eps, rms_norm_eps, rms_norm_heads, rms_norm_heads_no_weight,
};
use larql_compute::{
    Activation, DecodeStateDump, FullPipelineLayer, NormType, QuantFormat, QuantMatVec,
    StateDumpMask,
};
use ndarray::Array2;

use crate::CudaBackend;

/// Per-layer host-side KV mirror. `(K_cache, V_cache)` each `[len, kv_dim]`.
/// Grown by `push_kv_row` during prefill/decode; reset by `clear`.
type HostKv = Vec<(Array2<f32>, Array2<f32>)>;

impl CudaBackend {
    /// Borrow the host KV mirror (allocate-empty on first access).
    pub(crate) fn lock_host_kv(&self) -> std::sync::MutexGuard<'_, HostKv> {
        self.host_kv
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Reset the host KV mirror to `num_layers` empty layers.
    pub(crate) fn reset_host_kv(&self, num_layers: usize) {
        let mut kv = self.lock_host_kv();
        kv.clear();
        kv.resize_with(num_layers, || {
            (Array2::zeros((0, 0)), Array2::zeros((0, 0)))
        });
    }

    /// The committed sequence length per the host KV mirror (the source of
    /// truth for the host-orchestrated attention). Reads the first layer's
    /// K-cache row count; all layers progress in lockstep. This is the
    /// correct RoPE position for the next decode token — unlike the device
    /// cursor (`kv_cache_len_native`), which `decode_token` does NOT advance
    /// (only `prefill_kquant` populates the device cache). 0 when no mirror
    /// is allocated.
    pub(crate) fn host_kv_len(&self) -> usize {
        self.lock_host_kv()
            .first()
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0)
    }
}

impl CudaBackend {
    /// Decode one token through all layers with the host KV mirror.
    /// Returns the `[hidden]` post-layer residual. `None` if any layer's
    /// format isn't Q4_K/Q6_K (the only formats with native matvec today) or
    /// a projection returns `None` (caller falls back to the CPU path).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn host_decode_token(
        &self,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        abs_position: usize,
        mut state: Option<&mut DecodeStateDump>,
        mask: StateDumpMask,
    ) -> Option<Vec<f32>> {
        let num_layers = layers.len();
        if x.len() != hidden {
            return None;
        }
        // Ensure the host KV mirror has one slot per layer.
        {
            let mut kv = self.lock_host_kv();
            if kv.len() != num_layers {
                kv.clear();
                kv.resize_with(num_layers, || {
                    (Array2::zeros((0, 0)), Array2::zeros((0, 0)))
                });
            }
        }

        let want_h = !matches!(mask, StateDumpMask::None);
        let want_kv = matches!(mask, StateDumpMask::Full);

        let mut h = Array2::from_shape_vec((1, hidden), x.to_vec()).ok()?;
        for (li, layer) in layers.iter().enumerate() {
            // MoE layers are handled by the host-orchestrated expert path
            // below (`host_ffn_block_moe_decode`). PLE / remote-FFN still
            // bail: PLE needs the precomputed per-layer embedding input
            // (token-embedding-derived, not carried on `FullPipelineLayer`),
            // and remote-FFN needs a dispatch callback (only
            // `decode_token_with_moe` carries one).
            if layer.ple_input_gate.is_some() || layer.ffn_is_remote {
                return None;
            }
            if let Some(dump) = state.as_deref_mut() {
                if want_h {
                    dump.h_in_per_layer.push(h.row(0).to_vec());
                }
            }

            let (h_post_attn, k_new_row, v_new_row) =
                self.host_attention_block(layer, &h, li, abs_position)?;

            if let Some(dump) = state.as_deref_mut() {
                if want_kv {
                    dump.k_new_per_layer.push(k_new_row.to_vec());
                    dump.v_new_per_layer.push(v_new_row.to_vec());
                }
            }

            // Append the new K/V row to the host mirror.
            {
                let mut kv = self.lock_host_kv();
                if let Some((k_cache, v_cache)) = kv.get_mut(li) {
                    let kv_dim = layer.num_kv_heads * layer.head_dim;
                    let prev = k_cache.shape()[0];
                    let mut k_new = Array2::zeros((prev + 1, kv_dim));
                    let mut v_new = Array2::zeros((prev + 1, kv_dim));
                    if prev > 0 {
                        k_new.slice_mut(ndarray::s![..prev, ..]).assign(k_cache);
                        v_new.slice_mut(ndarray::s![..prev, ..]).assign(v_cache);
                    }
                    k_new
                        .slice_mut(ndarray::s![prev..prev + 1, ..])
                        .assign(&Array2::from_shape_vec((1, kv_dim), k_new_row.to_vec()).ok()?);
                    v_new
                        .slice_mut(ndarray::s![prev..prev + 1, ..])
                        .assign(&Array2::from_shape_vec((1, kv_dim), v_new_row.to_vec()).ok()?);
                    *k_cache = k_new;
                    *v_cache = v_new;
                }
            }

            let mut h_post_ffn = if layer.moe.is_some() {
                self.host_ffn_block_moe_decode(layer, &h_post_attn, hidden, inter)?
            } else {
                self.host_ffn_block(layer, &h_post_attn, hidden, inter)?
            };

            // Per-layer scalar (Gemma 4). Skip 0.0 (absent) and 1.0 (identity).
            // Applied uniformly for dense + MoE: the MoE block returns the
            // outer-combined residual (no PLE on the supported 26B-A4B path),
            // so the scalar is the final step, matching `moe_ffn_block_cpu`.
            let scalar = layer.layer_scalar;
            if scalar != 0.0 && scalar != 1.0 {
                h_post_ffn.mapv_inplace(|v| v * scalar);
            }
            h = h_post_ffn;
        }

        h.row(0).to_vec().into()
    }

    /// Multi-position prefill with KV-cache population. Runs the prompt
    /// through all layers, storing post-RoPE K/V into the host mirror, and
    /// returns the `[seq_len * hidden]` final hidden state for all positions.
    ///
    /// Projections use the amortised `q4k_matmul` / `q6k_matmul` native
    /// kernels (one launch per projection across all `seq_len` positions);
    /// elementwise ops + causal attention run on host. `softcap` is applied
    /// to the attention logits when `> 0.0`.
    pub(crate) fn host_prefill_kquant(
        &self,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        softcap: f32,
    ) -> Option<Vec<f32>> {
        let num_layers = layers.len();
        if x.len() != seq_len * hidden || seq_len == 0 {
            return None;
        }
        self.reset_host_kv(num_layers);

        let softcap_opt = if softcap > 0.0 { Some(softcap) } else { None };

        let mut h = Array2::from_shape_vec((seq_len, hidden), x.to_vec()).ok()?;
        for (li, layer) in layers.iter().enumerate() {
            // See the decode loop for the PLE / remote-FFN bail rationale.
            if layer.ple_input_gate.is_some() || layer.ffn_is_remote {
                return None;
            }
            let h_post_attn = self.host_prefill_attention_block(layer, &h, li, softcap_opt)?;
            let mut h_post_ffn = if layer.moe.is_some() {
                self.host_prefill_ffn_block_moe(layer, &h_post_attn, hidden, inter)?
            } else {
                self.host_prefill_ffn_block(layer, &h_post_attn, hidden, inter)?
            };

            // Per-layer scalar (Gemma 4). Skip 0.0 / 1.0. Applied for dense
            // + MoE (the MoE block returns the outer-combined residual).
            let scalar = layer.layer_scalar;
            if scalar != 0.0 && scalar != 1.0 {
                h_post_ffn.mapv_inplace(|v| v * scalar);
            }
            h = h_post_ffn;
        }

        let out: Vec<f32> = h.iter().cloned().collect();
        Some(out)
    }

    /// Prefill attention block for one layer: norm → Q/K/V matmul (all seq
    /// positions) → QK-norm → RoPE (position_offset=0) → causal GQA attend →
    /// O matmul → residual. Stores `[seq, kv_dim]` K/V into the host mirror.
    fn host_prefill_attention_block(
        &self,
        layer: &FullPipelineLayer<'_>,
        h: &Array2<f32>,
        li: usize,
        softcap: Option<f32>,
    ) -> Option<Array2<f32>> {
        let seq_len = h.shape()[0];
        let head_dim = layer.head_dim;
        let num_q = layer.num_q_heads;
        let num_kv = layer.num_kv_heads;
        let reps = num_q.checked_div(num_kv)?;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let scale = layer.attn_scale as f64;
        let hidden = h.shape()[1];

        // Input norm over [seq, hidden].
        let h_norm = self.norm_2d(
            layer.norm_type,
            h,
            layer.input_norm,
            layer.norm_offset,
            layer.eps,
        );
        let h_norm_flat: Vec<f32> = h_norm.iter().cloned().collect();
        let h_norm_slice = h_norm_flat.as_slice();

        // Q/K/V via amortised matmul: out is [seq, rows] row-major.
        let q_vec = self.quant_matmul(
            layer.wq.format,
            layer.wq.data,
            h_norm_slice,
            q_dim,
            hidden,
            seq_len,
        )?;
        let k_vec = self.quant_matmul(
            layer.wk.format,
            layer.wk.data,
            h_norm_slice,
            kv_dim,
            hidden,
            seq_len,
        )?;
        let v_vec = self.quant_matmul(
            layer.wv.format,
            layer.wv.data,
            h_norm_slice,
            kv_dim,
            hidden,
            seq_len,
        )?;

        let mut q = Array2::from_shape_vec((seq_len, q_dim), q_vec).ok()?;
        let mut k = Array2::from_shape_vec((seq_len, kv_dim), k_vec).ok()?;
        let mut v = Array2::from_shape_vec((seq_len, kv_dim), v_vec).ok()?;

        // QK-norm.
        let qk_off = layer.qk_norm_offset;
        if let Some(w) = layer.q_norm_weight {
            q = self.rms_norm_heads_array(&q, Some(w), num_q, head_dim, qk_off);
        }
        if let Some(w) = layer.k_norm_weight {
            k = self.rms_norm_heads_array(&k, Some(w), num_kv, head_dim, qk_off);
        }
        if layer.has_v_norm {
            v = self.rms_norm_heads_array(&v, None, num_kv, head_dim, 0.0);
        }

        // RoPE at position_offset=0 (positions 0..seq_len handled inside).
        // Thread the per-layer position divisor + llama3 scaling so
        // scaled-RoPE models (Gemma 3 global layers, llama3-rope) match the
        // CPU reference (`run_attention_with_kv_backend`).
        let frac = rope_fraction(layer);
        let pos_div = layer.rope_position_divisor as f64;
        let llama3 = layer.rope_llama3_scaling;
        let q_r = self.rope_native(
            &q,
            num_q,
            head_dim,
            layer.rope_base as f64,
            frac,
            0,
            pos_div,
            llama3,
        );
        let k_r = self.rope_native(
            &k,
            num_kv,
            head_dim,
            layer.rope_base as f64,
            frac,
            0,
            pos_div,
            layer.rope_llama3_scaling,
        );

        // Causal GQA attention over the full seq (with optional logit
        // softcap, mirroring `run_attention_with_kv_backend`).
        let attn_out = self.prefill_attention_native(
            &q_r, &k_r, &v, num_q, head_dim, kv_dim, reps, scale, seq_len, softcap,
        );

        // O projection via amortised matmul: [seq, hidden].
        let attn_flat: Vec<f32> = attn_out.iter().cloned().collect();
        let attn_slice = attn_flat.as_slice();
        let o_vec = self.quant_matmul(
            layer.wo.format,
            layer.wo.data,
            attn_slice,
            hidden,
            q_dim,
            seq_len,
        )?;
        let o = Array2::from_shape_vec((seq_len, hidden), o_vec).ok()?;

        // Store [seq, kv_dim] K/V into the host mirror.
        {
            let mut kv = self.lock_host_kv();
            if let Some(slot) = kv.get_mut(li) {
                *slot = (k_r.clone(), v.clone());
            }
        }

        // Post-attention residual (+ optional post-attn norm).
        let res_mult = layer.residual_multiplier;
        let h_post_attn = if layer.has_post_norms {
            let normed = self.norm_2d(
                layer.norm_type,
                &o,
                layer.post_attn_norm,
                layer.norm_offset,
                layer.eps,
            );
            self.add_residual_native(h, &normed, res_mult)
        } else {
            self.add_residual_native(h, &o, res_mult)
        };
        Some(h_post_attn)
    }

    /// Prefill FFN block for one layer: pre-ffn norm → gate/up matmul (all
    /// seq) → activation → down matmul → post-ffn residual.
    pub(crate) fn host_prefill_ffn_block(
        &self,
        layer: &FullPipelineLayer<'_>,
        h_post_attn: &Array2<f32>,
        hidden: usize,
        inter: usize,
    ) -> Option<Array2<f32>> {
        let seq_len = h_post_attn.shape()[0];

        let pre_norm_w = if layer.has_post_norms {
            layer.pre_ffn_norm
        } else {
            Some(layer.post_attn_norm)
        };
        let h_in = match pre_norm_w {
            Some(w) => self.norm_2d(
                layer.norm_type,
                h_post_attn,
                w,
                layer.norm_offset,
                layer.eps,
            ),
            None => self.norm_2d_no_weight(h_post_attn, layer.norm_offset, layer.eps),
        };
        let h_in_flat: Vec<f32> = h_in.iter().cloned().collect();
        let h_in_slice = h_in_flat.as_slice();

        // gate / up amortised matmul: out [seq, inter].
        let gate_vec = self.quant_matmul(
            layer.gate.format,
            layer.gate.data,
            h_in_slice,
            inter,
            hidden,
            seq_len,
        )?;
        let up_vec = self.quant_matmul(
            layer.up.format,
            layer.up.data,
            h_in_slice,
            inter,
            hidden,
            seq_len,
        )?;

        // Activation across all seq positions. Validate the projection
        // outputs before indexing — a short/overflowed `quant_matmul` return
        // must bail to `None` (panic-safety), not panic on an OOB slice.
        let activated_len = seq_len.checked_mul(inter)?;
        if gate_vec.len() != activated_len || up_vec.len() != activated_len {
            return None;
        }
        let mut activated = vec![0.0f32; activated_len];
        match layer.ffn_type {
            larql_compute::FfnType::Gated => {
                for s in 0..seq_len {
                    let off = s * inter;
                    self.apply_activation_gated_native(
                        layer.activation,
                        &gate_vec[off..off + inter],
                        &up_vec[off..off + inter],
                        &mut activated[off..off + inter],
                    );
                }
            }
            larql_compute::FfnType::Standard => {
                for s in 0..seq_len {
                    let off = s * inter;
                    self.apply_activation_std_native(
                        layer.activation,
                        &up_vec[off..off + inter],
                        &mut activated[off..off + inter],
                    );
                }
            }
        }

        // Down projection with optional padded stored_cols. The padded width
        // is uniform across rows, so one pad fits all seq positions. The
        // matmul contraction is `stored_cols`; `num_rows = hidden`, so the
        // output is `[seq, hidden]`.
        let (stored_cols, act_padded) =
            down_padded_activation(layer, &activated, hidden, inter, seq_len)?;
        let down_vec = self.quant_matmul(
            layer.down.format,
            layer.down.data,
            &act_padded,
            hidden,
            stored_cols,
            seq_len,
        )?;
        let out = Array2::from_shape_vec((seq_len, hidden), down_vec).ok()?;

        let res_mult = layer.residual_multiplier;
        let h_post_ffn = if layer.has_post_norms {
            let norm_w = layer.post_ffn_norm;
            let normed = match norm_w {
                Some(w) => self.norm_2d(layer.norm_type, &out, w, layer.norm_offset, layer.eps),
                None => self.norm_2d_no_weight(&out, layer.norm_offset, layer.eps),
            };
            self.add_residual_native(h_post_attn, &normed, res_mult)
        } else {
            self.add_residual_native(h_post_attn, &out, res_mult)
        };
        Some(h_post_ffn)
    }

    /// One attention block for a single decode step:
    /// norm → Q/K/V proj → QK-norm → RoPE → GQA attend → O proj → residual.
    /// Returns `(h_post_attn, k_new_row, v_new_row)`.
    fn host_attention_block(
        &self,
        layer: &FullPipelineLayer<'_>,
        h: &Array2<f32>,
        li: usize,
        abs_position: usize,
    ) -> Option<(Array2<f32>, Vec<f32>, Vec<f32>)> {
        let head_dim = layer.head_dim;
        let num_q = layer.num_q_heads;
        let num_kv = layer.num_kv_heads;
        let reps = num_q.checked_div(num_kv)?;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let scale = layer.attn_scale as f64;
        // The `DecodeBackend::decode_token` trait signature does not carry an
        // attention logit softcap (Gemma 4's attention softcap is 0; Metal's
        // decode path applies none either). `prefill_kquant`'s `softcap` arg
        // is threaded through the prefill entry point below.
        let softcap_opt: Option<f32> = None;

        // Input norm.
        let h_norm = self.norm_1d(
            layer.norm_type,
            h,
            layer.input_norm,
            layer.norm_offset,
            layer.eps,
        );

        // Q/K/V projections via the backend's native quant matvec.
        let q_vec = self.quant_matvec(
            layer.wq.format,
            layer.wq.data,
            h_norm_row(&h_norm),
            q_dim,
            h_norm.shape()[1],
        )?;
        let k_vec = self.quant_matvec(
            layer.wk.format,
            layer.wk.data,
            h_norm_row(&h_norm),
            kv_dim,
            h_norm.shape()[1],
        )?;
        let v_vec = self.quant_matvec(
            layer.wv.format,
            layer.wv.data,
            h_norm_row(&h_norm),
            kv_dim,
            h_norm.shape()[1],
        )?;

        // Validate projection lengths (panic-safety: short matvec → None).
        if q_vec.len() != q_dim || k_vec.len() != kv_dim || v_vec.len() != kv_dim {
            return None;
        }

        let q_full = vec_to_2d_row(q_vec);
        let k_full = vec_to_2d_row(k_vec);
        let mut v_full = vec_to_2d_row(v_vec);

        // QK-norm (per-head RMSNorm) — Gemma 3/4.
        let qk_off = layer.qk_norm_offset;
        let q_normed = match layer.q_norm_weight {
            Some(w) => self.rms_norm_heads_array(&q_full, Some(w), num_q, head_dim, qk_off),
            None => q_full,
        };
        let k_normed = match layer.k_norm_weight {
            Some(w) => self.rms_norm_heads_array(&k_full, Some(w), num_kv, head_dim, qk_off),
            None => k_full,
        };
        // V-norm (parameter-free, Gemma 4).
        if layer.has_v_norm {
            v_full = self.rms_norm_heads_array(&v_full, None, num_kv, head_dim, 0.0);
        }

        // RoPE on Q and K at `abs_position`. Thread the per-layer position
        // divisor + llama3 scaling (see the prefill block for rationale).
        let frac = rope_fraction(layer);
        let pos_div = layer.rope_position_divisor as f64;
        let llama3 = layer.rope_llama3_scaling;
        let q_rope = self.rope_native(
            &q_normed,
            num_q,
            head_dim,
            layer.rope_base as f64,
            frac,
            abs_position,
            pos_div,
            llama3,
        );
        let k_rope = self.rope_native(
            &k_normed,
            num_kv,
            head_dim,
            layer.rope_base as f64,
            frac,
            abs_position,
            pos_div,
            layer.rope_llama3_scaling,
        );

        let k_new_row: Vec<f32> = k_rope.row(0).to_vec();
        let v_new_row: Vec<f32> = v_full.row(0).to_vec();

        // Concatenate the host cache + the new row, then attend.
        let (k_concat, v_concat) = {
            let kv = self.lock_host_kv();
            let (k_cache, v_cache) = kv.get(li)?;
            let prev = k_cache.shape()[0];
            if prev == 0 {
                (k_rope.clone(), v_full.clone())
            } else {
                let mut k_out = Array2::zeros((prev + 1, kv_dim));
                let mut v_out = Array2::zeros((prev + 1, kv_dim));
                k_out.slice_mut(ndarray::s![..prev, ..]).assign(k_cache);
                v_out.slice_mut(ndarray::s![..prev, ..]).assign(v_cache);
                k_out
                    .slice_mut(ndarray::s![prev..prev + 1, ..])
                    .assign(&k_rope);
                v_out
                    .slice_mut(ndarray::s![prev..prev + 1, ..])
                    .assign(&v_full);
                (k_out, v_out)
            }
        };

        let attn_out = self.decode_attention_native(
            &q_rope,
            &k_concat,
            &v_concat,
            num_q,
            head_dim,
            kv_dim,
            reps,
            scale,
            softcap_opt,
        );

        // O projection. Output dim is `hidden` (the residual width); the
        // contraction is `q_dim` (the attention output width).
        let hidden = h.shape()[1];
        let o_vec = self.quant_matvec(
            layer.wo.format,
            layer.wo.data,
            attn_out_row(&attn_out),
            hidden,
            attn_out.shape()[1],
        )?;
        if o_vec.len() != hidden {
            return None;
        }
        let attn_projected = vec_to_2d_row(o_vec);

        // Post-attention residual (+ optional post-attn norm).
        let res_mult = layer.residual_multiplier;
        let h_post_attn = if layer.has_post_norms {
            let normed = self.norm_1d(
                layer.norm_type,
                &attn_projected,
                layer.post_attn_norm,
                layer.norm_offset,
                layer.eps,
            );
            self.add_residual_native(h, &normed, res_mult)
        } else {
            self.add_residual_native(h, &attn_projected, res_mult)
        };

        Some((h_post_attn, k_new_row, v_new_row))
    }

    /// One FFN block: pre-ffn norm → gate/up proj → activation → down proj →
    /// post-ffn residual.
    pub(crate) fn host_ffn_block(
        &self,
        layer: &FullPipelineLayer<'_>,
        h_post_attn: &Array2<f32>,
        hidden: usize,
        inter: usize,
    ) -> Option<Array2<f32>> {
        // Pre-FFN norm: when `has_post_norms`, use `pre_ffn_norm`; otherwise
        // reuse `post_attn_norm` as the FFN input norm (matches `run_ffn`).
        let pre_norm_w = if layer.has_post_norms {
            layer.pre_ffn_norm
        } else {
            Some(layer.post_attn_norm)
        };
        let h_in = match pre_norm_w {
            Some(w) => self.norm_1d(
                layer.norm_type,
                h_post_attn,
                w,
                layer.norm_offset,
                layer.eps,
            ),
            None => self.norm_2d_no_weight(h_post_attn, layer.norm_offset, layer.eps),
        };
        let h_in_row = h_norm_row(&h_in);

        // gate / up projections.
        let gate_vec =
            self.quant_matvec(layer.gate.format, layer.gate.data, h_in_row, inter, hidden)?;
        let up_vec = self.quant_matvec(layer.up.format, layer.up.data, h_in_row, inter, hidden)?;

        // Validate projection lengths before the activation helpers index
        // them (panic-safety: a short matvec return bails to `None`).
        if gate_vec.len() != inter || up_vec.len() != inter {
            return None;
        }

        // Activation: activation(gate) * up  (Gated) or activation(up) (Standard).
        let activated = match layer.ffn_type {
            larql_compute::FfnType::Gated => {
                let mut a = vec![0.0f32; inter];
                self.apply_activation_gated_native(layer.activation, &gate_vec, &up_vec, &mut a);
                a
            }
            larql_compute::FfnType::Standard => {
                let mut a = vec![0.0f32; inter];
                self.apply_activation_std_native(layer.activation, &up_vec, &mut a);
                a
            }
        };

        // Down projection. The stored down row width may be padded up to a
        // 256-multiple (e.g. 26B-A4B dense slab). Derive stored_cols from the
        // byte length and zero-pad the activation to match — pad columns
        // multiply zero activations, so the result is exact. Mirrors
        // `run_ffn_decode_step_q4k_direct`.
        let (stored_cols, act_padded) =
            down_padded_activation(layer, &activated, hidden, inter, 1)?;
        let down_vec = self.quant_matvec(
            layer.down.format,
            layer.down.data,
            &act_padded,
            hidden,
            stored_cols,
        )?;
        let out = vec_to_2d_row(down_vec);

        // Post-FFN residual (+ optional post-ffn norm).
        let res_mult = layer.residual_multiplier;
        let h_post_ffn = if layer.has_post_norms {
            let norm_w = layer.post_ffn_norm;
            let normed = match norm_w {
                Some(w) => self.norm_1d(layer.norm_type, &out, w, layer.norm_offset, layer.eps),
                None => self.norm_2d_no_weight(&out, layer.norm_offset, layer.eps),
            };
            self.add_residual_native(h_post_attn, &normed, res_mult)
        } else {
            self.add_residual_native(h_post_attn, &out, res_mult)
        };
        Some(h_post_ffn)
    }

    /// Hybrid-MoE FFN block for a single decode token (Gemma 4 26B-A4B
    /// shape). Runs the dense slab via the existing [`host_ffn_block`]
    /// (native quant matvec projections + host elementwise + post-FFN
    /// norm + residual), the expert block via the substrate reference
    /// [`cpu_moe_forward`], then combines the two with the Gemma-4 outer
    /// post-norm + residual — the structure of
    /// `larql-inference::moe_ffn_block_cpu_with_index`:
    ///
    ///   h1 = dense_slab - h_post_attn   (the dense delta; the slab
    ///                                    already carries the residual)
    ///   h2 = cpu_moe_forward(h_post_attn)  (the expert contribution)
    ///   out = h_post_attn + outer_norm(h1 + h2)
    ///
    /// `None` if the dense slab or expert block can't be computed — the
    /// caller falls back to the CPU engine path. Per-layer `layer_scalar`
    /// is NOT applied here; the decode loop applies it uniformly for dense
    /// and MoE (PLE is a no-op on the 26B-A4B target, so the scalar is the
    /// final step in both cases).
    pub(crate) fn host_ffn_block_moe_decode(
        &self,
        layer: &FullPipelineLayer<'_>,
        h_post_attn: &Array2<f32>,
        hidden: usize,
        inter: usize,
    ) -> Option<Array2<f32>> {
        let moe = layer.moe.as_ref()?;
        let h_post_ffn_dense = self.host_ffn_block(layer, h_post_attn, hidden, inter)?;
        let ha = h_post_attn.as_slice()?;
        let dense = h_post_ffn_dense.as_slice()?;
        let outer_w = moe_outer_norm(layer);
        let mut combined = vec![0.0f32; hidden];
        let out = moe_combine_row(ha, dense, moe, layer, outer_w, &mut combined);
        Some(vec_to_2d_row(out))
    }

    /// Multi-position hybrid-MoE FFN block (prefill). Same structure as
    /// [`host_ffn_block_moe_decode`] but the dense slab uses the amortised
    /// native quant matmul across all `seq_len` positions
    /// ([`host_prefill_ffn_block`]), and the expert block + outer combine
    /// run per position via [`moe_combine_row`] (the substrate
    /// `cpu_moe_forward` is single-token). The `combined` scratch is
    /// hoisted out of the per-position loop to avoid per-token allocation.
    pub(crate) fn host_prefill_ffn_block_moe(
        &self,
        layer: &FullPipelineLayer<'_>,
        h_post_attn: &Array2<f32>,
        hidden: usize,
        inter: usize,
    ) -> Option<Array2<f32>> {
        let moe = layer.moe.as_ref()?;
        let seq_len = h_post_attn.shape()[0];
        let h_post_ffn_dense = self.host_prefill_ffn_block(layer, h_post_attn, hidden, inter)?;
        let ha = h_post_attn.as_slice()?;
        let dense = h_post_ffn_dense.as_slice()?;
        let outer_w = moe_outer_norm(layer);
        let mut combined = vec![0.0f32; hidden];
        let mut out = vec![0.0f32; seq_len * hidden];
        for pos in 0..seq_len {
            let off = pos * hidden;
            let row = moe_combine_row(
                &ha[off..off + hidden],
                &dense[off..off + hidden],
                moe,
                layer,
                outer_w,
                &mut combined,
            );
            out[off..off + hidden].copy_from_slice(&row);
        }
        Array2::from_shape_vec((seq_len, hidden), out).ok()
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

impl CudaBackend {
    /// Format-dispatched amortised matmul: `out[seq, rows] = W[rows, k] @ x[seq, k]`.
    /// Routes Q4_K → `q4k_matmul`, Q6_K → `q6k_matmul` (native CUDA
    /// kernels with CPU fallback). `None` for other formats.
    fn quant_matmul(
        &self,
        format: QuantFormat,
        weights: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        match format {
            // Q4_KF is not produced by `build_pipeline_layers` and the native
            // CUDA q4k matmul kernel decodes 144-byte Q4_K super-blocks (not
            // Q4_KF's 160-byte layout), so don't route it here — `None`
            // makes a future Q4_KF builder path loud rather than silently
            // mis-decoding.
            QuantFormat::Q4_K => self.q4k_matmul(weights, x, num_rows, hidden, seq_len),
            QuantFormat::Q6_K => self.q6k_matmul(weights, x, num_rows, hidden, seq_len),
            _ => None,
        }
    }
}

impl CudaBackend {
    /// Minimum element count for a native norm dispatch to be worth the
    /// host→device upload + kernel launch + sync + device→host readback
    /// round-trip. Below this, the host `rms_norm_eps` reference is faster
    /// (the norm output is read straight back to host and re-uploaded by the
    /// next op, so there's no fusion benefit — only transfer+sync overhead).
    /// Mirrors the rationale of Metal's `calibration::DEFAULT_FLOP_THRESHOLD`
    /// for the dense GEMVs: only pay the device round-trip when the op is big
    /// enough to amortise it. Tuned conservatively for correctness-first
    /// (the host-orchestrated path is the parity oracle); the fully-fused
    /// single-command-buffer pipeline lifts this gate.
    const NORM_NATIVE_MIN_ELEMS: usize = 8192;

    /// True when a norm of `elems` elements is large enough that the native
    /// CUDA kernel is likely to beat the host reference after the per-call
    /// device round-trip.
    fn native_norm_worthwhile(elems: usize) -> bool {
        elems >= Self::NORM_NATIVE_MIN_ELEMS
    }

    /// Minimum element count for a native activation dispatch to be worth the
    /// host→device upload + kernel launch + sync + device→host readback
    /// round-trip. Below this, the host `apply_activation_*` reference is
    /// faster (the activation output is read straight back to host and
    /// re-uploaded by the down projection, so there's no fusion benefit —
    /// only transfer+sync overhead). Mirrors `NORM_NATIVE_MIN_ELEMS`.
    const ACTIVATION_NATIVE_MIN_ELEMS: usize = 8192;

    /// True when an activation of `elems` elements is large enough that the
    /// native CUDA kernel is likely to beat the host reference after the
    /// per-call device round-trip.
    fn native_activation_worthwhile(elems: usize) -> bool {
        elems >= Self::ACTIVATION_NATIVE_MIN_ELEMS
    }

    /// Minimum element count for a native residual add to be worth the
    /// host→device upload + kernel launch + sync + device→host readback
    /// round-trip. Below this, the host `h + b_scale * x` reference is faster
    /// (the residual output is read straight back to host and re-uploaded by
    /// the next op, so there's no fusion benefit — only transfer+sync
    /// overhead). Mirrors `NORM_NATIVE_MIN_ELEMS` / `ACTIVATION_NATIVE_MIN_ELEMS`.
    /// Tuned conservatively for correctness-first (the host-orchestrated path
    /// is the parity oracle); the fully-fused single-command-buffer pipeline
    /// lifts this gate. The decode residual (`[1, hidden]`, ~3-9k elements)
    /// typically stays below this gate and keeps the host path; the prefill
    /// residual (`[seq, hidden]`) clears it once `seq * hidden >= 8192`.
    const RESIDUAL_NATIVE_MIN_ELEMS: usize = 8192;

    /// True when a residual add of `elems` elements is large enough that the
    /// native CUDA kernel is likely to beat the host reference after the
    /// per-call device round-trip.
    fn native_residual_worthwhile(elems: usize) -> bool {
        elems >= Self::RESIDUAL_NATIVE_MIN_ELEMS
    }

    /// Minimum element count for a native RoPE dispatch to be worth the
    /// host→device upload + kernel launch + sync + device→host readback
    /// round-trip. Below this, the host `apply_rope_partial_at_full`
    /// reference is faster (the RoPE output is read straight back to host and
    /// re-uploaded by the attention dispatch, so there's no fusion benefit —
    /// only transfer+sync overhead). Mirrors `NORM_NATIVE_MIN_ELEMS` /
    /// `ACTIVATION_NATIVE_MIN_ELEMS` / `RESIDUAL_NATIVE_MIN_ELEMS`. Tuned
    /// conservatively for correctness-first (the host-orchestrated path is
    /// the parity oracle); the fully-fused single-command-buffer pipeline
    /// lifts this gate. The decode Q/K tensor (`[1, q_dim]`, typically a few
    /// thousand elements) often stays below this gate and keeps the host path;
    /// the prefill Q/K tensor (`[seq, q_dim]`) clears it once
    /// `seq * q_dim >= 8192`.
    const ROPE_NATIVE_MIN_ELEMS: usize = 8192;

    /// True when a RoPE over `elems` elements is large enough that the native
    /// CUDA kernel is likely to beat the host reference after the per-call
    /// device round-trip.
    fn native_rope_worthwhile(elems: usize) -> bool {
        elems >= Self::ROPE_NATIVE_MIN_ELEMS
    }

    /// Minimum attention work (num_q × total_len × head_dim) for a native
    /// decode-attention dispatch to be worth the host→device upload + kernel
    /// launch + sync + device→host readback round-trip. Below this the host
    /// `gqa_attention_decode_step` reference (rayon/spin-pool parallel over
    /// heads) is faster — there's no fusion benefit, only transfer+sync
    /// overhead, when the context is short. Mirrors the other
    /// `*_NATIVE_MIN_ELEMS` gates; tuned conservatively for correctness-first
    /// (the host-orchestrated path is the parity oracle); the fully-fused
    /// single-command-buffer pipeline lifts this gate.
    const DECODE_ATTN_NATIVE_MIN_WORK: usize = 8192;

    /// True when a decode-attention over `work = num_q × total_len × head_dim`
    /// is large enough that the native CUDA kernel is likely to beat the host
    /// reference after the per-call device round-trip.
    fn native_decode_attention_worthwhile(work: usize) -> bool {
        work >= Self::DECODE_ATTN_NATIVE_MIN_WORK
    }

    /// RoPE with split-half pairing — the device twin of
    /// `larql_compute::attention::rope::apply_rope_partial_at_full`. Routes
    /// through the native CUDA `rope` kernel when a runtime is present AND
    /// the tensor is large enough to amortise the device round-trip (see
    /// `ROPE_NATIVE_MIN_ELEMS`); falls back to the host reference on
    /// `Ok(false)`/`Err`, non-contiguous views, or small inputs.
    ///
    /// The `inv_freq[half_rotary]` frequency array is built via the shared
    /// substrate [`build_rope_inv_freq`] — the single source of truth also
    /// used by the host reference — so the device computes `theta`/`cos`/
    /// `sin` identically and the f32 rotation arithmetic matches the host
    /// path with `fmad` disabled at NVRTC compile time (the two paths can't
    /// drift on the frequency construction).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_native(
        &self,
        x: &Array2<f32>,
        num_heads: usize,
        head_dim: usize,
        rope_base: f64,
        fraction: f64,
        position_offset: usize,
        position_divisor: f64,
        llama3_scaling: Option<larql_models::Llama3RopeScaling>,
    ) -> Array2<f32> {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        let n = rows * cols;
        let x_flat = x.as_slice().unwrap_or(&[]);
        if !x_flat.is_empty() && Self::native_rope_worthwhile(n) {
            // Use the shared `build_rope_inv_freq` so the uploaded
            // `inv_freq`/`half_rotary` are bit-identical to the reference's
            // (including the `llama3` wavelength-band variant). `rotary_dim`
            // is floored at 2 so `half_rotary >= 1`.
            let (_rotary_dim, half_rotary, inv_freq) =
                build_rope_inv_freq(rope_base, head_dim, fraction, llama3_scaling);
            let divisor = if position_divisor > 0.0 {
                position_divisor
            } else {
                1.0
            };
            let mut out = vec![0.0f32; n];
            if let Ok(true) = self.native_rope(
                x_flat,
                &inv_freq,
                &mut out,
                rows,
                num_heads,
                head_dim,
                half_rotary,
                position_offset,
                divisor,
            ) {
                return Array2::from_shape_vec((rows, cols), out)
                    .expect("native rope output shape");
            }
        }
        apply_rope_partial_at_full(
            x,
            num_heads,
            head_dim,
            rope_base,
            fraction,
            position_offset,
            position_divisor,
            llama3_scaling,
        )
    }

    /// Fused decode-step GQA attention — the device twin of
    /// `gqa_attention_decode_step`. Routes through the native CUDA
    /// `decode_attention` kernel when a runtime is present AND the attention
    /// work (`num_q × total_len × head_dim`) is large enough to amortise the
    /// device round-trip (see `DECODE_ATTN_NATIVE_MIN_WORK`); falls back to the
    /// host reference on `Ok(false)`/`Err`, non-contiguous Q/K/V views, or
    /// short contexts. `q` is `[1, num_q * head_dim]`; `k_cache`/`v_cache` are
    /// `[total_len, kv_dim]`. Returns `[1, num_q * head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_attention_native(
        &self,
        q: &Array2<f32>,
        k_cache: &Array2<f32>,
        v_cache: &Array2<f32>,
        num_q: usize,
        head_dim: usize,
        kv_dim: usize,
        reps: usize,
        scale: f64,
        softcap: Option<f32>,
    ) -> Array2<f32> {
        let total_len = k_cache.shape()[0];
        let work = num_q.saturating_mul(total_len).saturating_mul(head_dim);
        let q_dim = num_q * head_dim;
        if total_len >= 1
            && Self::native_decode_attention_worthwhile(work)
            && q.shape() == [1, q_dim]
            && k_cache.shape() == [total_len, kv_dim]
            && v_cache.shape() == [total_len, kv_dim]
        {
            if let (Some(qs), Some(ks), Some(vs)) =
                (q.as_slice(), k_cache.as_slice(), v_cache.as_slice())
            {
                let mut out = Vec::with_capacity(q_dim);
                if let Ok(true) = self.native_decode_attention(
                    qs,
                    ks,
                    vs,
                    &mut out,
                    scale as f32,
                    softcap,
                    num_q,
                    head_dim,
                    kv_dim,
                    reps,
                    total_len,
                ) {
                    if out.len() == q_dim {
                        return Array2::from_shape_vec((1, q_dim), out)
                            .expect("native decode_attention output shape");
                    }
                }
            }
        }
        gqa_attention_decode_step(q, k_cache, v_cache, num_q, head_dim, reps, scale, softcap)
    }

    /// `PREFILL_ATTN_NATIVE_MIN_WORK` — the prefill attention is worth a
    /// device round-trip once the work (over all query positions × heads) is
    /// large enough to amortise the htod/launch/dtoh cost. Tuned
    /// conservatively for correctness-first (the host-orchestrated path is
    /// the parity oracle); the fully-fused single-command-buffer pipeline
    /// lifts this gate.
    const PREFILL_ATTN_NATIVE_MIN_WORK: usize = 8192;

    /// True when a prefill attention over
    /// `work = seq_len × num_q × seq_len × head_dim` (causal, so ~half the
    /// QKᵀ + all the weighted-V) is large enough that the native CUDA kernel
    /// is likely to beat the host reference after the per-call device
    /// round-trip.
    fn native_prefill_attention_worthwhile(work: usize) -> bool {
        work >= Self::PREFILL_ATTN_NATIVE_MIN_WORK
    }

    /// Fused prefill (seq×seq) causal GQA attention — the device twin of
    /// `gqa_attention_with_weights` (the symmetric `gqa_attention_capture`
    /// path). Routes through the native CUDA `prefill_attention` kernel when a
    /// runtime is present AND the attention work is large enough to amortise
    /// the device round-trip (see `PREFILL_ATTN_NATIVE_MIN_WORK`); falls back
    /// to the host reference on `Ok(false)`/`Err`, non-contiguous Q/K/V views,
    /// or short prompts. `q` is `[seq, num_q * head_dim]`; `k`/`v` are
    /// `[seq, kv_dim]`. Returns `[seq, num_q * head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_attention_native(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
        num_q: usize,
        head_dim: usize,
        kv_dim: usize,
        reps: usize,
        scale: f64,
        seq_len: usize,
        softcap: Option<f32>,
    ) -> Array2<f32> {
        let q_dim = num_q * head_dim;
        // Work proxy: causal QKᵀ is ~seq_len*(seq_len/2)*num_q*head_dim flops;
        // use the full `seq_len*num_q*seq_len*head_dim` upper bound for the gate.
        let work = seq_len
            .saturating_mul(num_q)
            .saturating_mul(seq_len)
            .saturating_mul(head_dim);
        if seq_len >= 1
            && Self::native_prefill_attention_worthwhile(work)
            && q.shape() == [seq_len, q_dim]
            && k.shape() == [seq_len, kv_dim]
            && v.shape() == [seq_len, kv_dim]
        {
            if let (Some(qs), Some(ks), Some(vs)) = (q.as_slice(), k.as_slice(), v.as_slice()) {
                let mut out = Vec::with_capacity(seq_len * q_dim);
                if let Ok(true) = self.native_prefill_attention(
                    qs,
                    ks,
                    vs,
                    &mut out,
                    scale as f32,
                    softcap,
                    num_q,
                    head_dim,
                    kv_dim,
                    reps,
                    seq_len,
                ) {
                    if out.len() == seq_len * q_dim {
                        return Array2::from_shape_vec((seq_len, q_dim), out)
                            .expect("native prefill_attention output shape");
                    }
                }
            }
        }
        larql_compute::attention::gqa::gqa_attention_with_weights(
            q, k, v, num_q, head_dim, reps, scale, seq_len, false, softcap,
        )
        .0
    }

    /// Gated activation (`out[i] = act(gate[i]) * up[i]`) routed through the
    /// native CUDA kernel when a runtime is present AND the element count is
    /// large enough to amortise the device round-trip; falls back to the host
    /// `apply_activation_gated` reference on `Ok(false)`/`Err` or small
    /// inputs. Only `Silu` / `GeluTanh` are reachable via `FullPipelineLayer`
    /// (see `apply_activation_gated`); the other arms fail loud via
    /// `unreachable!`.
    pub(crate) fn apply_activation_gated_native(
        &self,
        act: Activation,
        gate: &[f32],
        up: &[f32],
        out: &mut [f32],
    ) {
        if Self::native_activation_worthwhile(out.len()) {
            let launched = match act {
                Activation::Silu => self.native_geglu_silu(gate, up, out),
                Activation::GeluTanh => self.native_geglu_gelu_tanh(gate, up, out),
                _ => unreachable!(
                    "apply_activation_gated_native: FullPipelineLayer only emits Silu/GeluTanh (got {act:?})"
                ),
            };
            if matches!(launched, Ok(true)) {
                return;
            }
        }
        apply_activation_gated(act, gate, up, out);
    }

    /// Standard (non-gated) activation (`out[i] = act(x[i])`) routed through
    /// the native CUDA kernel when a runtime is present AND the element
    /// count is large enough; falls back to the host reference otherwise.
    pub(crate) fn apply_activation_std_native(&self, act: Activation, x: &[f32], out: &mut [f32]) {
        if Self::native_activation_worthwhile(out.len()) {
            let launched = match act {
                Activation::Silu => self.native_activation_silu(x, out),
                Activation::GeluTanh => self.native_activation_gelu_tanh(x, out),
                _ => unreachable!(
                    "apply_activation_std_native: FullPipelineLayer only emits Silu/GeluTanh (got {act:?})"
                ),
            };
            if matches!(launched, Ok(true)) {
                return;
            }
        }
        apply_activation_std(act, x, out);
    }

    /// Scaled residual add (`out = h + b_scale * x`) for same-shaped `[rows,
    /// cols]` arrays — the device twin of the host `add_residual` helper.
    /// Routes through the native CUDA `residual_add` kernel when a runtime is
    /// present AND the element count is large enough to amortise the device
    /// round-trip; falls back to the host reference on `Ok(false)`/`Err`,
    /// non-contiguous views, or small inputs. The device kernel fuses the
    /// `b_scale == 1.0` / `b_scale != 1.0` arms of the host helper (the two
    /// are numerically identical, so no branch is needed on the device).
    pub(crate) fn add_residual_native(
        &self,
        h: &Array2<f32>,
        x: &Array2<f32>,
        b_scale: f32,
    ) -> Array2<f32> {
        let (rows, cols) = (h.shape()[0], h.shape()[1]);
        let n = rows * cols;
        if let (Some(hs), Some(xs)) = (h.as_slice(), x.as_slice()) {
            if n > 0 && Self::native_residual_worthwhile(n) {
                let mut out = vec![0.0f32; n];
                if let Ok(true) = self.native_residual_add(hs, xs, &mut out, b_scale, n) {
                    return Array2::from_shape_vec((rows, cols), out)
                        .expect("native residual_add output shape");
                }
            }
        }
        add_residual(h, x, b_scale)
    }

    /// RMSNorm or LayerNorm for a `[rows, cols]` array using a `&[f32]`
    /// weight. Routes the RmsNorm arm through the native CUDA `rms_norm`
    /// kernel when a runtime is present AND the norm is large enough to
    /// amortise the device round-trip (see `NORM_NATIVE_MIN_ELEMS`); falls
    /// back to the host reference on `Ok(false)`/`Err`, non-contiguous views,
    /// or small norms. `weight` must be `Some` (the `None`-weight pre-ffn
    /// path uses `norm_2d_no_weight`).
    pub(crate) fn norm_2d(
        &self,
        norm_type: NormType,
        x: &Array2<f32>,
        weight: &[f32],
        offset: f32,
        eps: f32,
    ) -> Array2<f32> {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        match norm_type {
            NormType::RmsNorm => {
                // Non-contiguous views are rare here (the pipeline builds
                // contiguous arrays); fall through to the host reference
                // rather than packing into a staging buffer.
                let x_flat = x.as_slice().unwrap_or(&[]);
                if !x_flat.is_empty() && Self::native_norm_worthwhile(rows * cols) {
                    let mut out = vec![0.0f32; rows * cols];
                    if let Ok(true) = self.native_rms_norm(
                        x_flat,
                        Some(weight),
                        &mut out,
                        rows,
                        cols,
                        eps as f64,
                        offset,
                    ) {
                        return Array2::from_shape_vec((rows, cols), out)
                            .expect("native rms_norm output shape");
                    }
                }
                let w_vec: Vec<f32> = weight.to_vec();
                rms_norm_eps(x, Some(&w_vec), offset, eps as f64)
            }
            NormType::LayerNorm => {
                let w_vec: Vec<f32> = weight.to_vec();
                layer_norm_eps(x, Some(&w_vec), None, eps as f64)
            }
        }
    }

    /// `None`-weight RMSNorm (the pre-ffn norm path when `has_post_norms` is
    /// false: `rms_norm_eps` with `weight = None`, which uses `w = 1.0`).
    /// Routes through the native `rms_norm` kernel with `has_weight = 0`
    /// when a runtime is present AND the norm is large enough to amortise
    /// the device round-trip; falls back to the host reference otherwise.
    pub(crate) fn norm_2d_no_weight(&self, x: &Array2<f32>, offset: f32, eps: f32) -> Array2<f32> {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        let x_flat = x.as_slice().unwrap_or(&[]);
        if !x_flat.is_empty() && Self::native_norm_worthwhile(rows * cols) {
            let mut out = vec![0.0f32; rows * cols];
            if let Ok(true) =
                self.native_rms_norm(x_flat, None, &mut out, rows, cols, eps as f64, offset)
            {
                return Array2::from_shape_vec((rows, cols), out)
                    .expect("native rms_norm output shape");
            }
        }
        rms_norm_eps(x, None, offset, eps as f64)
    }

    /// RMSNorm or LayerNorm for a `[1, cols]` row using a `&[f32]` weight
    /// (the `FullPipelineLayer` carries norm weights as `&[f32]`, not
    /// `&Vec<f32>`). Routes the RmsNorm arm through the native kernel.
    pub(crate) fn norm_1d(
        &self,
        norm_type: NormType,
        x: &Array2<f32>,
        weight: &[f32],
        offset: f32,
        eps: f32,
    ) -> Array2<f32> {
        self.norm_2d(norm_type, x, weight, offset, eps)
    }

    /// Per-head RMSNorm over a `[seq_len, num_heads*head_dim]` array — the
    /// device twin of `larql_compute::residual::rms_norm_heads` (weighted)
    /// / `rms_norm_heads_no_weight` (`weight = None`). Uses the substrate
    /// `DEFAULT_EPS = 1e-6` (the per-head CPU references hard-code it, so the
    /// native path must too for parity). Routes through the native
    /// `rms_norm_heads` kernel when a runtime is present AND the norm is
    /// large enough to amortise the device round-trip; falls back to the
    /// host reference on `Ok(false)`/`Err`, non-contiguous views, or small
    /// norms. The weighted kernel indexes `weight[d]` (broadcast across
    /// heads), matching the CPU `rms_norm_heads` reference and the real
    /// Gemma3/4 `[head_dim]`-shaped q_norm/k_norm weights.
    pub(crate) fn rms_norm_heads_array(
        &self,
        x: &Array2<f32>,
        weight: Option<&[f32]>,
        num_heads: usize,
        head_dim: usize,
        offset: f32,
    ) -> Array2<f32> {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        let x_flat = x.as_slice().unwrap_or(&[]);
        if !x_flat.is_empty() && Self::native_norm_worthwhile(rows * cols) {
            let mut out = vec![0.0f32; rows * cols];
            let eps = larql_compute::residual::DEFAULT_EPS;
            if let Ok(true) = self.native_rms_norm_heads(
                x_flat, weight, &mut out, rows, num_heads, head_dim, eps, offset,
            ) {
                return Array2::from_shape_vec((rows, cols), out)
                    .expect("native rms_norm_heads output shape");
            }
        }
        match weight {
            Some(w) => rms_norm_heads(x, w, num_heads, head_dim, offset),
            None => rms_norm_heads_no_weight(x, num_heads, head_dim),
        }
    }
}

fn h_norm_row(arr: &Array2<f32>) -> &[f32] {
    arr.as_slice()
        .unwrap_or_else(|| arr.row(0).to_slice().unwrap_or(&[]))
}

fn attn_out_row(arr: &Array2<f32>) -> &[f32] {
    arr.as_slice()
        .unwrap_or_else(|| arr.row(0).to_slice().unwrap_or(&[]))
}

fn vec_to_2d_row(v: Vec<f32>) -> Array2<f32> {
    let n = v.len();
    Array2::from_shape_vec((1, n), v).expect("matvec output shape")
}

fn rope_fraction(layer: &FullPipelineLayer<'_>) -> f64 {
    if layer.rotary_dim == 0 {
        1.0
    } else {
        layer.rotary_dim as f64 / layer.head_dim as f64
    }
}

/// Select the outer post-combine norm weight for a hybrid-MoE layer.
/// When the arch ships a combined-output norm (Gemma 4 26B-A4B:
/// `moe_combined_output_norm == true`), the outer norm is the dedicated
/// `moe_outer_post_norm` (`post_feedforward_layernorm`, un-suffixed),
/// falling back to `post_ffn_norm` (`_1`) when the dedicated key is
/// absent. Mirrors `moe_ffn_block_cpu_with_index`'s outer-norm
/// selection. `None` when the arch has no combined-output norm.
pub(crate) fn moe_outer_norm<'a>(layer: &'a FullPipelineLayer<'a>) -> Option<&'a [f32]> {
    if layer.moe_combined_output_norm {
        layer.moe_outer_post_norm.or(layer.post_ffn_norm)
    } else {
        None
    }
}

/// Per-row MoE combine, shared by the decode (single-row) and prefill
/// (per-position) MoE FFN blocks so the Gemma-4 combine formula lives in
/// exactly one place:
///
///   combined = (dense_row - ha_row) + cpu_moe_forward(ha_row)
///   out      = outer_post_norm_residual(ha_row, combined, outer_w, …)
///
/// The dense slab already carries the residual, so subtracting `ha_row`
/// recovers the dense delta — the outer combine re-adds the residual once
/// (no double count). `combined` is caller-owned scratch of length
/// `hidden`, reused across positions in prefill to avoid per-token
/// allocation (mirrors the substrate MoE code's TLS-scratch discipline).
pub(crate) fn moe_combine_row(
    ha_row: &[f32],
    dense_row: &[f32],
    moe: &larql_compute::MoeLayerWeights<'_>,
    layer: &FullPipelineLayer<'_>,
    outer_w: Option<&[f32]>,
    combined: &mut [f32],
) -> Vec<f32> {
    let h2 = cpu_moe_forward(ha_row, moe, layer.norm_offset, layer.eps);
    for (i, c) in combined.iter_mut().enumerate() {
        *c = (dense_row[i] - ha_row[i]) + h2[i];
    }
    outer_post_norm_residual(ha_row, combined, outer_w, layer.norm_offset, layer.eps)
}

/// `h + b_scale * x` for `[1, hidden]` arrays.
pub(crate) fn add_residual(h: &Array2<f32>, x: &Array2<f32>, b_scale: f32) -> Array2<f32> {
    if b_scale == 1.0 {
        h + x
    } else {
        h + &(x * b_scale)
    }
}

/// Apply a gated activation: `out[i] = act(gate[i]) * up[i]`.
///
/// Only `Silu` and `GeluTanh` are reachable via `FullPipelineLayer` —
/// `build_arch_params`/`build_pipeline_layers` map `larql_models::Activation`
/// to exactly those two (mirrors Metal's `stages/ffn.rs`). The other arms
/// fail loud via `unreachable!` so a future builder change that emits a new
/// variant surfaces here instead of silently miscomputing.
pub(crate) fn apply_activation_gated(act: Activation, gate: &[f32], up: &[f32], out: &mut [f32]) {
    match act {
        Activation::Silu => {
            for i in 0..out.len() {
                let x = gate[i];
                out[i] = (x / (1.0 + (-x).exp())) * up[i];
            }
        }
        Activation::GeluTanh => {
            let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
            for i in 0..out.len() {
                let x = gate[i];
                let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                out[i] = 0.5 * x * (1.0 + inner.tanh()) * up[i];
            }
        }
        _ => unreachable!(
            "apply_activation_gated: FullPipelineLayer only emits Silu/GeluTanh (got {act:?})"
        ),
    }
}

/// Apply a standard (non-gated) activation: `out[i] = act(x[i])`.
pub(crate) fn apply_activation_std(act: Activation, x: &[f32], out: &mut [f32]) {
    match act {
        Activation::Silu => {
            for i in 0..out.len() {
                let v = x[i];
                out[i] = v / (1.0 + (-v).exp());
            }
        }
        Activation::GeluTanh => {
            let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
            for i in 0..out.len() {
                let v = x[i];
                let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
                out[i] = 0.5 * v * (1.0 + inner.tanh());
            }
        }
        _ => unreachable!(
            "apply_activation_std: FullPipelineLayer only emits Silu/GeluTanh (got {act:?})"
        ),
    }
}

/// Derive the down projection's stored column count from the byte length and
/// zero-pad the activation to match. `activated` must be `seq_len * inter`
/// long; the returned `padded_act` is `seq_len * stored_cols` long (one
/// zero-padded row per sequence position). Returns `(stored_cols, padded_act)`.
/// Mirrors `run_ffn_decode_step_q4k_direct`'s padding handling — pad columns
/// multiply zero activations, so the result is exact.
fn down_padded_activation(
    layer: &FullPipelineLayer<'_>,
    activated: &[f32],
    hidden: usize,
    inter: usize,
    seq_len: usize,
) -> Option<(usize, Vec<f32>)> {
    let down = &layer.down;
    let bytes_per_sb = super_block_bytes(down.format)?;
    // Guard `hidden == 0` before the division (would otherwise panic on
    // integer divide-by-zero instead of returning `None`).
    if hidden == 0 || down.data.is_empty() {
        return None;
    }
    let down_bytes_per_row = down.data.len() / hidden;
    if down_bytes_per_row == 0 || !down_bytes_per_row.is_multiple_of(bytes_per_sb) {
        return None;
    }
    let stored_cols = down_bytes_per_row / bytes_per_sb * 256;
    if stored_cols < inter {
        return None;
    }
    let act_len = seq_len.checked_mul(inter)?;
    if activated.len() != act_len {
        return None;
    }
    if stored_cols == inter {
        Some((stored_cols, activated.to_vec()))
    } else {
        let mut padded = vec![0.0f32; seq_len * stored_cols];
        for s in 0..seq_len {
            let src_off = s * inter;
            let dst_off = s * stored_cols;
            padded[dst_off..dst_off + inter].copy_from_slice(&activated[src_off..src_off + inter]);
        }
        Some((stored_cols, padded))
    }
}

/// Bytes per 256-element super-block for the k-quant formats the down
/// projection uses. `None` for non-block formats (the host-orchestrated path
/// only handles Q4_K / Q6_K down today).
fn super_block_bytes(fmt: QuantFormat) -> Option<usize> {
    match fmt {
        // Q4_KF (160 B/super-block) is never produced by `build_pipeline_layers`
        // for the down projection, so it's intentionally absent here —
        // advertising it with the wrong constant (144) would silently
        // mis-derive `stored_cols`. Add it with `Q4_KF_BLOCK_BYTES` if a
        // builder path ever emits a Q4_KF down matrix.
        QuantFormat::Q4_K => Some(144),
        QuantFormat::Q6_K => Some(210),
        _ => None,
    }
}

// `HostKv` is held under a `Mutex` on `CudaBackend`; re-export the type alias
// for the backend module.
pub(crate) type HostKvType = HostKv;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_fraction_full_when_rotary_dim_zero() {
        let mut layer = make_minimal_layer();
        layer.rotary_dim = 0;
        layer.head_dim = 256;
        assert!((rope_fraction(&layer) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rope_fraction_partial() {
        let mut layer = make_minimal_layer();
        layer.rotary_dim = 64;
        layer.head_dim = 256;
        assert!((rope_fraction(&layer) - 0.25).abs() < 1e-12);
    }

    /// Build a `FullPipelineLayer` with just the fields `rope_fraction` /
    /// `down_stored_cols` read. The full struct is large; `Default`-ish via
    /// zeroed weights is enough for these unit tests.
    fn make_minimal_layer() -> FullPipelineLayer<'static> {
        use larql_compute::*;
        let qw = QuantWeight {
            data: &[],
            scales: None,
            format: QuantFormat::Q4_K,
        };
        FullPipelineLayer {
            wq: qw,
            wk: qw,
            wv: qw,
            wo: qw,
            gate: qw,
            up: qw,
            down: qw,
            input_norm: &[],
            post_attn_norm: &[],
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 0.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.0,
            head_dim: 256,
            num_q_heads: 0,
            num_kv_heads: 0,
            rope_base: 10000.0,
            rope_position_divisor: 1.0,
            rope_llama3_scaling: None,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
        }
    }
}

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
use larql_compute::cpu::ops::moe::{
    cpu_moe_forward, moe_expert_input, moe_post_expert_output, moe_route_from_router_input,
    moe_router_input,
};
use larql_compute::cpu::ops::outer_combine::outer_post_norm_residual;
use larql_compute::residual::{
    layer_norm_eps, rms_norm_eps, rms_norm_heads, rms_norm_heads_no_weight,
};
use larql_compute::{
    Activation, DecodeStateDump, FfnType, FullPipelineLayer, NormType, QuantFormat, QuantMatVec,
    StateDumpMask,
};
use larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
use ndarray::Array2;

use crate::options::native_thresholds;
use crate::CudaBackend;
use cudarc::driver::CudaSlice;

/// Per-layer host-side KV mirror. `(K_cache, V_cache)` each `[len, kv_dim]`.
/// Grown by `push_kv_row` during prefill/decode; reset by `clear`.
type HostKv = Vec<(Array2<f32>, Array2<f32>)>;

/// The decode hidden state for one token, in one of two residency states.
/// GPU-007 (cross-layer hidden-state residency): threaded through the
/// decode layer loop so eligible layers carry `h` device-resident across
/// the attention→FFN→next-layer chain, collapsing the per-block hidden-state
/// readback/upload boundaries the host-orchestrated loop pays.
///
/// Ownership + synchronization discipline (mirrors the existing
/// device-resident chains): the `Device` variant holds a `CudaSlice` produced
/// by a chain kernel (a fresh allocation — no aliasing); any prior host copy
/// is **stale** and must not be read without a fresh [`Self::ensure_host`].
/// The host path (state dump / fallback / final logits handoff) calls
/// `ensure_host` to materialize the host copy exactly once.
enum DecodeHiddenState {
    /// Host-resident `[1, hidden]`. The reference/fallback/state-dump path —
    /// also the entry state (the input embedding is host-side).
    Host(Array2<f32>),
    /// Device-resident `[hidden]` (flattened; decode is single-token). The
    /// host copy is absent/stale; `ensure_host` materializes it via a single
    /// `sync_dtoh_f32`.
    Device { dev: CudaSlice<f32>, hidden: usize },
}

impl DecodeHiddenState {
    /// The hidden dim (the column count of the `[1, hidden]` state).
    fn hidden(&self) -> usize {
        match self {
            Self::Host(arr) => arr.shape()[1],
            Self::Device { hidden, .. } => *hidden,
        }
    }

    /// Materialize the device copy if currently host-resident, transitioning
    /// to `Device`. The host state is replaced only after `upload_f32`
    /// succeeds, so a failed upload leaves the current host data intact.
    /// Already-device-resident state is an idempotent no-op.
    fn ensure_device(&mut self, runtime: &crate::backend::CudaRuntime) -> bool {
        let uploaded = match self {
            Self::Host(arr) => {
                let hidden = arr.shape()[1];
                let Some(slice) = arr.as_slice() else {
                    return false;
                };
                match runtime.upload_f32(slice) {
                    Ok(dev) => Some((dev, hidden)),
                    Err(_) => None,
                }
            }
            Self::Device { .. } => return true,
        };

        if let Some((dev, hidden)) = uploaded {
            *self = Self::Device { dev, hidden };
            true
        } else {
            false
        }
    }

    /// Materialize the host copy if currently device-resident, transitioning
    /// to `Host`. One `sync_dtoh_f32` per transition (idempotent thereafter).
    /// Returns `false` if the device readback fails (caller should bail to
    /// `None` so the engine re-runs on CPU, mirroring every other native
    /// dispatch's `Err → None` contract).
    fn ensure_host(&mut self, runtime: &crate::backend::CudaRuntime) -> bool {
        match self {
            Self::Host(_) => true,
            Self::Device { dev, hidden } => match runtime.sync_dtoh_f32(dev) {
                Ok(v) if v.len() == *hidden => {
                    *self = Self::Host(
                        Array2::from_shape_vec((1, *hidden), v).expect("decode hidden state shape"),
                    );
                    true
                }
                _ => false,
            },
        }
    }

    /// Borrow the host `[1, hidden]` array (the caller has already called
    /// `ensure_host`). Panics if currently `Device` — the resident loop must
    /// materialize the host copy before any host-only branch reads it.
    fn as_host(&self) -> &Array2<f32> {
        match self {
            Self::Host(arr) => arr,
            Self::Device { .. } => {
                panic!("DecodeHiddenState::as_host called on a Device variant — ensure_host first")
            }
        }
    }
}

/// Outcome of one B3B single-stream graph-path layer attempt. The decode loop
/// branches on this to carry the hidden state correctly across the arena
/// boundary (see [`CudaBackend::host_graph_decode_layer`]).
enum GraphLayerOutcome {
    /// The graph succeeded; the post-FFN hidden now lives in `arena.output(flip)`
    /// (carried by the flip — no owned buffer to clone between graph layers).
    ArenaOut { flip: bool },
    /// The graph was attempted but failed AFTER attention already appended
    /// K/V; the resident device FFN ran on the cloned post-attn state and
    /// produced this owned buffer. The caller carries it as a
    /// `DecodeHiddenState::Device`.
    DeviceFallback(CudaSlice<f32>),
    /// The graph path was not eligible/disabled, OR attention bailed before
    /// appending K/V (the input hidden was restored). The caller runs the
    /// existing non-graph attention + FFN path for this layer.
    NotAttempted,
}

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
                let prof = crate::options::gpu_profile_enabled();
                let mt0 = if prof {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let mut rows_copied = 0usize;
                {
                    let mut kv = self.lock_host_kv();
                    if let Some((k_cache, v_cache)) = kv.get_mut(li) {
                        let kv_dim = layer.num_kv_heads * layer.head_dim;
                        let prev = k_cache.shape()[0];
                        rows_copied = prev;
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
                if let Some(t0) = mt0 {
                    self.note_mirror_append(t0.elapsed().as_nanos() as u64, rows_copied);
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

    /// Decode one token through all layers, threading the hidden state
    /// device-resident across eligible layers (GPU-007: cross-layer
    /// hidden-state residency). The resident-hidden twin of
    /// [`host_decode_token`].
    ///
    /// For each eligible dense Q4_K/Q6_K decode layer, the post-attention
    /// residual output of the attention chain and the post-FFN residual output
    /// of the FFN chain stay device-resident, so the layer→next-layer handoff
    /// doesn't pay a hidden-state readback+re-upload. The per-block
    /// readback/upload the host-orchestrated loop pays between (a) the O
    /// projection and the post-attn residual, (b) the down projection and the
    /// post-FFN residual, and (c) the layer output and the next layer's
    /// attention input — are collapsed into a single readback at the final
    /// decode output.
    ///
    /// The host path stays the parity oracle + the state-dump path: a state
    /// dump (`StateDumpMask != None`) forces the host-orchestrated
    /// [`host_decode_token`] (explicit host sync at every dump point). When a
    /// layer is ineligible (MoE, PLE, remote-FFN, LayerNorm, sub-gate work,
    /// non-Q4_K/Q6_K, padded down, or any chained launch returns `Err`), the
    /// resident loop converts the device hidden state back to host exactly
    /// once (`ensure_host`) and continues through the existing
    /// `host_attention_block` + `host_ffn_block` for the rest of that layer —
    /// the next eligible layer can re-enter the resident path. Eligibility is
    /// counted (`note_resident_hidden`) for the `LARQL_GPU_DIAG` surface.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn host_decode_token_resident(
        &self,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        abs_position: usize,
    ) -> Option<Vec<f32>> {
        // State dump forces the host path (explicit host sync at dump points).
        // No runtime → defer to the host path (which itself returns `None` on
        // the scaffold, so `decode_token` falls back to the CPU reference).
        if !self.native_runtime_available() {
            return self.host_decode_token(
                layers,
                x,
                hidden,
                inter,
                abs_position,
                None,
                StateDumpMask::None,
            );
        }

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

        let runtime = self.runtime()?;
        let mut h = DecodeHiddenState::Host(Array2::from_shape_vec((1, hidden), x.to_vec()).ok()?);
        // B3B: when the graph path is active across consecutive layers, the
        // hidden state lives in the arena (carried by flip) rather than in `h`.
        // `Some(flip)` means the current hidden is `arena.output(flip)`.
        let mut arena_out_flip: Option<bool> = None;

        for (li, layer) in layers.iter().enumerate() {
            // PLE / remote-FFN bail (same rationale as `host_decode_token`):
            // these paths are host-shaped and have no resident-hidden twin.
            if layer.ple_input_gate.is_some() || layer.ffn_is_remote {
                return None;
            }

            // ── B3B single-stream graph path (opt-in via LARQL_CUDA_GRAPHS=1) ──
            // The whole layer — attention writing its post-attn residual in
            // place into the arena input slot, the K/V host-mirror append, and
            // the FFN graph build/replay — runs on the one runtime stream with
            // zero per-layer D2D and zero cross-stream syncs. Falls back to the
            // non-graph resident/host path when not eligible or on a bail.
            let graph_eligible = layer.moe.is_none()
                && h.hidden() == hidden
                && self.resident_hidden_layer_eligible(layer, hidden, inter, li)
                && self.graph_mode().enabled();
            if graph_eligible {
                match self.host_graph_decode_layer(
                    runtime,
                    layer,
                    &mut h,
                    &mut arena_out_flip,
                    li,
                    abs_position,
                    hidden,
                    inter,
                    layers.len(),
                ) {
                    GraphLayerOutcome::ArenaOut { flip } => {
                        arena_out_flip = Some(flip);
                        let scalar = layer.layer_scalar;
                        if scalar != 0.0 && scalar != 1.0 {
                            // Rare non-identity per-layer scalar: read the
                            // arena output back once, scale, carry host-side.
                            let Some(v) = self.read_arena_output_to_host(runtime, flip, hidden)
                            else {
                                self.note_resident_hidden(false);
                                return None;
                            };
                            let mut scaled = v;
                            scaled.iter_mut().for_each(|x| *x *= scalar);
                            h = DecodeHiddenState::Host(
                                Array2::from_shape_vec((1, hidden), scaled).ok()?,
                            );
                            arena_out_flip = None;
                        }
                        self.note_resident_hidden(true);
                        continue;
                    }
                    GraphLayerOutcome::DeviceFallback(dev) => {
                        arena_out_flip = None;
                        let scalar = layer.layer_scalar;
                        if scalar != 0.0 && scalar != 1.0 {
                            let mut hh = DecodeHiddenState::Device { dev, hidden };
                            if !hh.ensure_host(runtime) {
                                self.note_resident_hidden(false);
                                return None;
                            }
                            let mut scaled = hh.as_host().clone();
                            scaled.mapv_inplace(|v| v * scalar);
                            h = DecodeHiddenState::Host(scaled);
                        } else {
                            h = DecodeHiddenState::Device { dev, hidden };
                        }
                        self.note_resident_hidden(true);
                        continue;
                    }
                    GraphLayerOutcome::NotAttempted => { /* fall through below */ }
                }
            }

            // Exiting the arena (a non-graph layer follows, or the graph path
            // bailed): read the arena output back to host once (a boundary D2D,
            // not per-layer steady state).
            if let Some(f) = arena_out_flip.take() {
                let v = match self.read_arena_output_to_host(runtime, f, hidden) {
                    Some(v) => v,
                    None => {
                        self.note_resident_hidden(false);
                        return None;
                    }
                };
                h = DecodeHiddenState::Host(Array2::from_shape_vec((1, hidden), v).ok()?);
            }

            // Non-graph resident path: device-resident hidden state through
            // attention → (K/V host mirror append) → resident device FFN.
            let resident_ok = layer.moe.is_none()
                && h.hidden() == hidden
                && self.resident_hidden_layer_eligible(layer, hidden, inter, li);

            if resident_ok && h.ensure_device(runtime) {
                if let Some((h_post_attn_dev, k_new_row, v_new_row)) =
                    self.host_attention_block_device_resident(runtime, layer, &h, li, abs_position)
                {
                    // Append the new K/V row to the host mirror (the parity
                    // oracle + truncate/state-dump source — GPU-006 invariant:
                    // the host mirror append happens between attention and FFN,
                    // exactly as in `host_decode_token`).
                    {
                        let prof = crate::options::gpu_profile_enabled();
                        let mt0 = if prof {
                            Some(std::time::Instant::now())
                        } else {
                            None
                        };
                        let mut rows_copied = 0usize;
                        {
                            let mut kv = self.lock_host_kv();
                            if let Some((k_cache, v_cache)) = kv.get_mut(li) {
                                let kv_dim = layer.num_kv_heads * layer.head_dim;
                                let prev = k_cache.shape()[0];
                                rows_copied = prev;
                                let mut k_new = Array2::zeros((prev + 1, kv_dim));
                                let mut v_new = Array2::zeros((prev + 1, kv_dim));
                                if prev > 0 {
                                    k_new.slice_mut(ndarray::s![..prev, ..]).assign(k_cache);
                                    v_new.slice_mut(ndarray::s![..prev, ..]).assign(v_cache);
                                }
                                k_new.slice_mut(ndarray::s![prev..prev + 1, ..]).assign(
                                    &Array2::from_shape_vec((1, kv_dim), k_new_row.to_vec())
                                        .expect("k_new_row shape"),
                                );
                                v_new.slice_mut(ndarray::s![prev..prev + 1, ..]).assign(
                                    &Array2::from_shape_vec((1, kv_dim), v_new_row.to_vec())
                                        .expect("v_new_row shape"),
                                );
                                *k_cache = k_new;
                                *v_cache = v_new;
                            }
                        }
                        if let Some(t0) = mt0 {
                            self.note_mirror_append(t0.elapsed().as_nanos() as u64, rows_copied);
                        }
                    }

                    // FFN chain consumes the device-resident post-attn state
                    // via the resident device FFN (the CUDA-graph path is
                    // handled above in `host_graph_decode_layer`; this non-graph
                    // branch never uses graphs).
                    let h_post_ffn_dev_opt = self.host_ffn_block_device_resident(
                        runtime,
                        layer,
                        &h_post_attn_dev,
                        hidden,
                        inter,
                    );
                    if let Some(h_post_ffn_dev) = h_post_ffn_dev_opt {
                        // Per-layer scalar (Gemma 4). Skip 0.0 (absent) and
                        // 1.0 (identity). When present and non-identity
                        // (rare), apply on host: read back the device post-FFN
                        // state once, scale, and carry host-side. The next
                        // eligible layer re-uploads via its input norm (one
                        // extra boundary in the rare scalar case — the common
                        // `scalar == 1.0` path stays fully device-resident).
                        let scalar = layer.layer_scalar;
                        if scalar != 0.0 && scalar != 1.0 {
                            let mut hh = DecodeHiddenState::Device {
                                dev: h_post_ffn_dev,
                                hidden,
                            };
                            if !hh.ensure_host(runtime) {
                                self.note_resident_hidden(false);
                                return None;
                            }
                            let mut scaled = hh.as_host().clone();
                            scaled.mapv_inplace(|v| v * scalar);
                            self.note_resident_hidden(true);
                            h = DecodeHiddenState::Host(scaled);
                            continue;
                        }
                        self.note_resident_hidden(true);
                        h = DecodeHiddenState::Device {
                            dev: h_post_ffn_dev,
                            hidden,
                        };
                        continue;
                    }
                    // FFN resident path bailed AFTER the resident attention
                    // already appended K/V to the host mirror. Re-running the
                    // full host attention (the generic fallback below) would
                    // double-append K/V, so instead run ONLY the host FFN
                    // block on the device post-attn output (read back once).
                    // The host FFN path is the parity oracle + handles MoE /
                    // padded-down / sub-gate activation.
                    let mut h_post_attn_hs = DecodeHiddenState::Device {
                        dev: h_post_attn_dev,
                        hidden,
                    };
                    if !h_post_attn_hs.ensure_host(runtime) {
                        self.note_resident_hidden(false);
                        return None;
                    }
                    let h_post_attn = h_post_attn_hs.as_host().clone();
                    let mut h_post_ffn = if layer.moe.is_some() {
                        self.host_ffn_block_moe_decode(layer, &h_post_attn, hidden, inter)?
                    } else {
                        self.host_ffn_block(layer, &h_post_attn, hidden, inter)?
                    };
                    let scalar = layer.layer_scalar;
                    if scalar != 0.0 && scalar != 1.0 {
                        h_post_ffn.mapv_inplace(|v| v * scalar);
                    }
                    // The attention ran resident (counted as a use); only the
                    // FFN fell back. Count the layer as a fallback so the diag
                    // reflects the partial residency.
                    self.note_resident_hidden(false);
                    h = DecodeHiddenState::Host(h_post_ffn);
                    continue;
                }
                // Resident attention bailed → fall through to the host path
                // (which runs attention + FFN + appends K/V exactly once).
            }

            // Fallback: ensure host, run the existing host-orchestrated
            // attention + FFN blocks for this layer. The device hidden state
            // (if any) is read back exactly once here.
            if !h.ensure_host(runtime) {
                self.note_resident_hidden(false);
                return None;
            }
            let h_host = h.as_host().clone();

            let (h_post_attn, k_new_row, v_new_row) =
                self.host_attention_block(layer, &h_host, li, abs_position)?;
            // Append the new K/V row to the host mirror (same as above).
            {
                let prof = crate::options::gpu_profile_enabled();
                let mt0 = if prof {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let mut rows_copied = 0usize;
                {
                    let mut kv = self.lock_host_kv();
                    if let Some((k_cache, v_cache)) = kv.get_mut(li) {
                        let kv_dim = layer.num_kv_heads * layer.head_dim;
                        let prev = k_cache.shape()[0];
                        rows_copied = prev;
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
                if let Some(t0) = mt0 {
                    self.note_mirror_append(t0.elapsed().as_nanos() as u64, rows_copied);
                }
            }

            let mut h_post_ffn = if layer.moe.is_some() {
                self.host_ffn_block_moe_decode(layer, &h_post_attn, hidden, inter)?
            } else {
                self.host_ffn_block(layer, &h_post_attn, hidden, inter)?
            };
            let scalar = layer.layer_scalar;
            if scalar != 0.0 && scalar != 1.0 {
                h_post_ffn.mapv_inplace(|v| v * scalar);
            }
            self.note_resident_hidden(false);
            h = DecodeHiddenState::Host(h_post_ffn);
        }

        // Final decode output: ensure host, return the `[hidden]` vector.
        // B3B: if the last layer left the hidden in the arena, read it back
        // here (the single end-of-token hidden readback).
        if let Some(f) = arena_out_flip.take() {
            let v = self.read_arena_output_to_host(runtime, f, hidden)?;
            h = DecodeHiddenState::Host(Array2::from_shape_vec((1, hidden), v).ok()?);
        }
        // LARQL-GPU-PROFILE-001: time the single end-of-token hidden-state
        // readback (the device→host copy that returns the decode output). This
        // is the cost B4 (device lm-head) would eliminate.
        let rt0 = if crate::options::gpu_profile_enabled() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        if !h.ensure_host(runtime) {
            return None;
        }
        if let Some(t0) = rt0 {
            self.note_hidden_readback(t0.elapsed().as_nanos() as u64);
        }
        Some(h.as_host().row(0).to_vec())
    }

    /// Eligibility gate for the resident-hidden decode path (GPU-007). Pure
    /// (no device touch) so the eligibility logic is testable on every host.
    /// Returns `true` when the layer can run its attention + FFN blocks with
    /// the hidden state carried device-resident across both. The gates mirror
    /// the per-block device-chain gates ([`host_attention_block_device`] and
    /// [`host_ffn_block_device`]) plus the resident-specific gates:
    /// RmsNorm-only (no LayerNorm device kernel) and `NormType` uniformity.
    fn resident_hidden_layer_eligible(
        &self,
        layer: &FullPipelineLayer<'_>,
        hidden: usize,
        inter: usize,
        li: usize,
    ) -> bool {
        #[cfg(test)]
        if self.resident_hidden_fallback_forced_for_test(li) {
            return false;
        }
        let head_dim = layer.head_dim;
        let num_q = layer.num_q_heads;
        let num_kv = layer.num_kv_heads;
        if num_kv == 0 || head_dim == 0 {
            return false;
        }
        // Decode-attention work gate (same proxy as the attention device chain).
        let prev = self
            .lock_host_kv()
            .get(li)
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0);
        let total_len = prev + 1;
        let attn_work = num_q.saturating_mul(total_len).saturating_mul(head_dim);
        if !Self::native_decode_attention_worthwhile(attn_work) {
            return false;
        }
        // Activation work gate (same proxy as the FFN device chain).
        if !Self::native_activation_worthwhile(inter) {
            return false;
        }
        // All seven projections must be a k-quant the device matvec/matmul
        // handles (Q/K/V/O + gate/up/down).
        let k_quant = |f: QuantFormat| matches!(f, QuantFormat::Q4_K | QuantFormat::Q6_K);
        if !(k_quant(layer.wq.format)
            && k_quant(layer.wk.format)
            && k_quant(layer.wv.format)
            && k_quant(layer.wo.format)
            && k_quant(layer.gate.format)
            && k_quant(layer.up.format)
            && k_quant(layer.down.format))
        {
            return false;
        }
        // The gate/up/down triple must be a supported resident-hidden FFN
        // layout (see [`supported_resident_ffn_triple`]). The shared helper
        // is also used by `host_ffn_block_device_resident` so the two gates
        // cannot drift.
        if !supported_resident_ffn_triple(layer.gate.format, layer.up.format, layer.down.format) {
            return false;
        }
        // Down stored width must equal inter (no device-side zero-pad step).
        match down_stored_cols(layer, hidden, inter) {
            Some(stored_cols) if stored_cols == inter => {}
            _ => return false,
        }
        // Only RmsNorm has a device kernel (LayerNorm falls back to host).
        // This gates both the input norm and the post-attn / post-ffn norms.
        if layer.norm_type != NormType::RmsNorm {
            return false;
        }
        // Only Gated/Standard SiLu + GeluTanh activations have device chains.
        if !matches!(
            (layer.ffn_type, layer.activation),
            (larql_compute::FfnType::Gated, Activation::Silu)
                | (larql_compute::FfnType::Gated, Activation::GeluTanh)
                | (larql_compute::FfnType::Standard, Activation::Silu)
                | (larql_compute::FfnType::Standard, Activation::GeluTanh)
        ) {
            return false;
        }
        true
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
    /// O matmul → residual. Tries the device-resident chain first
    /// ([`host_prefill_attention_block_device`], which keeps the Q/K/V →
    /// QK-norm → RoPE → attention → O activations on the device across the
    /// whole chain and reads back exactly once); falls back to the
    /// host-orchestrated path ([`host_prefill_attention_block_hostonly`]) when
    /// the device path is unavailable or the layer features aren't supported.
    pub(crate) fn host_prefill_attention_block(
        &self,
        layer: &FullPipelineLayer<'_>,
        h: &Array2<f32>,
        li: usize,
        softcap: Option<f32>,
    ) -> Option<Array2<f32>> {
        if let Some(out) = self.host_prefill_attention_block_device(layer, h, li, softcap) {
            return Some(out);
        }
        self.host_prefill_attention_block_hostonly(layer, h, li, softcap)
    }

    /// Prefill attention block, host-orchestrated reference path: norm → Q/K/V
    /// matmul (all seq positions) → QK-norm → RoPE (position_offset=0) →
    /// causal GQA attend → O matmul → residual. Each projection is a separate
    /// htod/launch/dtoh round-trip (the elementwise ops route through the
    /// per-op native helpers with their min-elems gates). This is the parity
    /// oracle for the device-resident chain and the fallback when the device
    /// path bails. Stores `[seq, kv_dim]` K/V into the host mirror.
    pub(crate) fn host_prefill_attention_block_hostonly(
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

    /// Device-resident prefill attention chain: input norm (host) → upload once
    /// → Q/K/V matmul → QK-norm → V-norm → RoPE → causal attention → O matmul
    /// (all on the device, chained via device-resident buffers) → single
    /// `sync_dtoh` readback of O (+ the post-RoPE K / post-V-norm V for the
    /// host KV mirror) → post-attn norm + residual (host).
    ///
    /// This collapses the per-projection round-trips the host-orchestrated
    /// path pays between the Q/K/V matmuls, the QK-norm/V-norm, the RoPE, the
    /// attention, and the O matmul: the three projections share one upload of
    /// the normed input, and every downstream op reads its input resident
    /// instead of dtoh-then-htod. The Q/K/V stay resident all the way through
    /// RoPE + attention (so QK-norm, RoPE, and the attention reduction never
    /// touch the host); only the final O, K, V are read back. The input norm +
    /// post-attn norm + residual stay on the host (the residual needs `h`,
    /// which is host-resident from the previous layer; threading it onto the
    /// device too is the next collapse slice).
    ///
    /// Returns `None` (caller falls back to [`host_prefill_attention_block_hostonly`])
    /// when: no runtime; the attention work is below the gate; any of the
    /// Q/K/V/O projections isn't a Q4_K/Q6_K the device matmul handles; the
    /// input isn't a contiguous `[seq, hidden]` row; or the prefill attention
    /// shape exceeds the device shared-mem/index budget (`Err` from the
    /// attention launcher maps to `None`).
    pub(crate) fn host_prefill_attention_block_device(
        &self,
        layer: &FullPipelineLayer<'_>,
        h: &Array2<f32>,
        li: usize,
        softcap: Option<f32>,
    ) -> Option<Array2<f32>> {
        let runtime = self.runtime()?;
        let seq_len = h.shape()[0];
        let head_dim = layer.head_dim;
        let num_q = layer.num_q_heads;
        let num_kv = layer.num_kv_heads;
        let reps = num_q.checked_div(num_kv)?;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let scale = layer.attn_scale as f64;
        let hidden = h.shape()[1];

        // Gate the whole chain on the attention work (same proxy as
        // `prefill_attention_native`). Below it the host path is faster.
        let work = seq_len
            .saturating_mul(num_q)
            .saturating_mul(seq_len)
            .saturating_mul(head_dim);
        if seq_len < 1 || !Self::native_prefill_attention_worthwhile(work) {
            return None;
        }
        // All four attention projections must be a k-quant the device matmul
        // handles (`matmul_dev_by_fmt` only routes Q4_K/Q6_K).
        let qf = layer.wq.format;
        let kf = layer.wk.format;
        let vf = layer.wv.format;
        let of = layer.wo.format;
        let k_quant = |f: QuantFormat| matches!(f, QuantFormat::Q4_K | QuantFormat::Q6_K);
        if !(k_quant(qf) && k_quant(kf) && k_quant(vf) && k_quant(of)) {
            return None;
        }
        // The input must be contiguous to slice for the norm + upload.
        let h_slice = h.as_slice()?;
        if h_slice.len() != seq_len * hidden {
            return None;
        }

        // Input norm on host (matches the FFN device chain's pre-norm-on-host:
        // the norm weight is f32 and the residual it feeds is host-resident;
        // threading the norm onto the device + chaining norm→Q/K/V is the next
        // collapse slice). QK-norm/V-norm/RoPE run inside the device chain
        // below because they consume device-resident projection outputs.
        let h_norm = self.norm_2d(
            layer.norm_type,
            h,
            layer.input_norm,
            layer.norm_offset,
            layer.eps,
        );
        let h_norm_slice = h_norm.as_slice()?;
        if h_norm_slice.len() != seq_len * hidden {
            return None;
        }

        // RoPE inv_freq built via the shared substrate helper — the single
        // source of truth also used by the host reference, so the uploaded
        // frequencies are bit-identical and the device/host can't drift.
        let frac = rope_fraction(layer);
        let pos_div = layer.rope_position_divisor as f64;
        let llama3 = layer.rope_llama3_scaling;
        let (_rotary_dim, half_rotary, inv_freq) =
            build_rope_inv_freq(layer.rope_base as f64, head_dim, frac, llama3);

        // Per-head RMSNorm eps matches the CPU reference hard-coding.
        let qk_eps = larql_compute::residual::DEFAULT_EPS;
        let qk_off = layer.qk_norm_offset;

        // Drive the device chain. Any launch error maps to `None` so the
        // caller falls back to the host-orchestrated path (the documented
        // contract for every native dispatch in this backend). All launches
        // run on the same stream (stream-ordered — a kernel reading a buffer
        // written by an earlier kernel on the same stream sees the data
        // without an inter-kernel sync).
        let (o_vec, k_vec, v_vec): (Vec<f32>, Vec<f32>, Vec<f32>) = {
            let h_dev = runtime.upload_f32(h_norm_slice).ok()?;
            // Q/K/V projections share the normed input resident. Each
            // intermediate is bound to its own name and kept alive until the
            // block-end readback, so the per-step `CudaSlice` drops happen
            // AFTER the single `sync_dtoh_f32` (which has already synchronised
            // the stream). Rebinding mid-chain would drop the prior buffer
            // immediately, and on devices without memory-pool support
            // (`CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED == 0`) cudarc's
            // `CudaSlice::drop` forces a stream `synchronize()` — turning the
            // single sync into one per rebind. Distinct bindings (the same
            // discipline the FFN chains use) keep the single-sync guarantee
            // unconditional.
            let q_proj =
                matmul_dev_by_fmt(runtime, qf, layer.wq.data, &h_dev, q_dim, hidden, seq_len)
                    .ok()?;
            let k_proj =
                matmul_dev_by_fmt(runtime, kf, layer.wk.data, &h_dev, kv_dim, hidden, seq_len)
                    .ok()?;
            let v_proj =
                matmul_dev_by_fmt(runtime, vf, layer.wv.data, &h_dev, kv_dim, hidden, seq_len)
                    .ok()?;
            // QK-norm (per-head RMSNorm; Gemma 3/4). Optional; falls through to
            // the projection output when the norm weight is absent.
            let q_normed = match layer.q_norm_weight {
                Some(w) => runtime
                    .launch_rms_norm_heads_dev(
                        &q_proj,
                        Some(w),
                        seq_len,
                        num_q,
                        head_dim,
                        qk_eps,
                        qk_off,
                    )
                    .ok()?,
                None => q_proj,
            };
            let k_normed = match layer.k_norm_weight {
                Some(w) => runtime
                    .launch_rms_norm_heads_dev(
                        &k_proj,
                        Some(w),
                        seq_len,
                        num_kv,
                        head_dim,
                        qk_eps,
                        qk_off,
                    )
                    .ok()?,
                None => k_proj,
            };
            // V-norm (parameter-free, Gemma 4). Optional.
            let v_normed = if layer.has_v_norm {
                runtime
                    .launch_rms_norm_heads_dev(
                        &v_proj, None, seq_len, num_kv, head_dim, qk_eps, 0.0,
                    )
                    .ok()?
            } else {
                v_proj
            };
            // RoPE on Q and K at position_offset=0 (positions 0..seq_len handled
            // inside the kernel via the row index). The `inv_freq` table is
            // uploaded once and shared by both RoPE launches (the host slice is
            // identical), avoiding a redundant per-launch htod.
            let inv_freq_dev = runtime.upload_f64(&inv_freq).ok()?;
            let q_rope = runtime
                .launch_rope_dev_with_invfreq(
                    &inv_freq_dev,
                    &q_normed,
                    seq_len,
                    num_q,
                    head_dim,
                    half_rotary,
                    0,
                    pos_div,
                )
                .ok()?;
            let k_rope = runtime
                .launch_rope_dev_with_invfreq(
                    &inv_freq_dev,
                    &k_normed,
                    seq_len,
                    num_kv,
                    head_dim,
                    half_rotary,
                    0,
                    pos_div,
                )
                .ok()?;
            // Causal GQA attention over the full seq (resident q/k/v).
            let attn_dev = runtime
                .launch_prefill_attention_dev(
                    &q_rope,
                    &k_rope,
                    &v_normed,
                    scale as f32,
                    softcap,
                    num_q,
                    head_dim,
                    kv_dim,
                    reps,
                    seq_len,
                )
                .ok()?;
            // O projection: resident attention output → [seq, hidden].
            let o_dev = matmul_dev_by_fmt(
                runtime,
                of,
                layer.wo.data,
                &attn_dev,
                hidden,
                q_dim,
                seq_len,
            )
            .ok()?;
            // One sync at the end of the chain; the K/V readbacks that follow
            // are idle-stream copies (the first `sync_dtoh_f32` already
            // synchronised, so the kernels are done).
            let o_vec = runtime.sync_dtoh_f32(&o_dev).ok()?;
            let k_vec = runtime.sync_dtoh_f32(&k_rope).ok()?;
            let v_vec = runtime.sync_dtoh_f32(&v_normed).ok()?;
            if o_vec.len() != seq_len * hidden
                || k_vec.len() != seq_len * kv_dim
                || v_vec.len() != seq_len * kv_dim
            {
                return None;
            }
            (o_vec, k_vec, v_vec)
        };

        let o = Array2::from_shape_vec((seq_len, hidden), o_vec).ok()?;
        // Store the post-RoPE K / post-V-norm V into the host mirror (matches
        // the host-orchestrated path: attention later reads this mirror).
        {
            let k_r = Array2::from_shape_vec((seq_len, kv_dim), k_vec).ok()?;
            let v = Array2::from_shape_vec((seq_len, kv_dim), v_vec).ok()?;
            let mut kv = self.lock_host_kv();
            if let Some(slot) = kv.get_mut(li) {
                *slot = (k_r, v);
            }
        }

        // Post-attention residual (+ optional post-attn norm) on host.
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
    /// seq) → activation → down matmul → post-ffn residual. Tries the
    /// device-resident chain first ([`host_prefill_ffn_block_device`], which
    /// keeps the gate/up/activation/down activations on the device across the
    /// whole chain and reads back exactly once); falls back to the
    /// host-orchestrated path ([`host_prefill_ffn_block_hostonly`]) when the
    /// device path is unavailable or the layer features aren't supported.
    pub(crate) fn host_prefill_ffn_block(
        &self,
        layer: &FullPipelineLayer<'_>,
        h_post_attn: &Array2<f32>,
        hidden: usize,
        inter: usize,
    ) -> Option<Array2<f32>> {
        if let Some(out) = self.host_prefill_ffn_block_device(layer, h_post_attn, hidden, inter) {
            return Some(out);
        }
        self.host_prefill_ffn_block_hostonly(layer, h_post_attn, hidden, inter)
    }

    /// Prefill FFN block, host-orchestrated reference path: pre-ffn norm →
    /// gate/up matmul (all seq) → activation → down matmul → post-ffn
    /// residual. Each projection is a separate htod/launch/dtoh round-trip
    /// (the elementwise ops run on host). This is the parity oracle for the
    /// device-resident chain and the fallback when the device path bails.
    pub(crate) fn host_prefill_ffn_block_hostonly(
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

        Some(self.apply_post_ffn_residual(layer, h_post_attn, &out))
    }

    /// Device-resident prefill FFN chain: pre-ffn norm (host) → gate/up matmul
    /// → activation → down matmul (all on the device, chained via
    /// device-resident buffers) → single `sync_dtoh` readback → post-ffn norm
    /// + residual (host).
    ///
    /// This collapses the per-projection round-trips the host-orchestrated
    /// path pays between the gate/up matmuls, the activation, and the down
    /// matmul: instead of 3 separate htod(input)+dtoh(output) cycles the chain
    /// uploads the normed input once, chains four kernels on the same
    /// CUDA stream (stream-ordered, so no inter-kernel sync), and reads the
    /// `[seq, hidden]` down output back exactly once. The norm + post-ffn
    /// residual stay on the host (the residual needs `h_post_attn`, which is
    /// host-resident from the previous layer; threading it onto the device too
    /// is the next collapse slice).
    ///
    /// Returns `None` (caller falls back to [`host_prefill_ffn_block_hostonly`])
    /// when: no runtime; the work is below the activation gate; the gate/up/down
    /// formats aren't Q4_K/Q6_K; the down matrix's stored width is padded
    /// beyond `inter` (the device chain assumes a contiguous `[seq, inter]`
    /// activation feeds the down matmul directly); or the activation/ffn-type
    /// combination isn't one of the native kernels. `Err` from any chained
    /// launch also maps to `None` (the host path is the documented fallback).
    pub(crate) fn host_prefill_ffn_block_device(
        &self,
        layer: &FullPipelineLayer<'_>,
        h_post_attn: &Array2<f32>,
        hidden: usize,
        inter: usize,
    ) -> Option<Array2<f32>> {
        let runtime = self.runtime()?;
        let seq_len = h_post_attn.shape()[0];

        // Below the activation gate the host path is faster (no fusion
        // benefit, only transfer+sync overhead) — mirrors the other
        // `*_NATIVE_MIN_ELEMS` gates.
        if !Self::native_activation_worthwhile(seq_len.saturating_mul(inter)) {
            return None;
        }
        // All three FFN projections must be a k-quant the device matmul handles.
        let gate_fmt = layer.gate.format;
        let up_fmt = layer.up.format;
        let down_fmt = layer.down.format;
        if !matches!(
            (gate_fmt, up_fmt, down_fmt),
            (QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q4_K)
                | (QuantFormat::Q6_K, QuantFormat::Q6_K, QuantFormat::Q6_K)
        ) {
            return None;
        }
        // The down matmul must contract exactly `inter` columns — a padded
        // stored width would need a device-side zero-pad step the chain
        // doesn't perform, so bail to the host path (which pads on host).
        let stored_cols = down_stored_cols(layer, hidden, inter)?;
        if stored_cols != inter {
            return None;
        }

        // Pre-FFN norm on the host (the norm weight is f32; threading it onto
        // the device + chaining norm→gate/up is the next collapse slice).
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
        let h_in_slice = h_in.as_slice()?;

        // Drive the device chain. Any launch error maps to `None` so the
        // caller falls back to the host-orchestrated path (the documented
        // contract for every native dispatch in this backend).
        let down_vec: Vec<f32> = {
            let x_dev = runtime.upload_f32(h_in_slice).ok()?;
            let gate_dev = matmul_dev_by_fmt(
                runtime,
                gate_fmt,
                layer.gate.data,
                &x_dev,
                inter,
                hidden,
                seq_len,
            )
            .ok()?;
            let up_dev = matmul_dev_by_fmt(
                runtime,
                up_fmt,
                layer.up.data,
                &x_dev,
                inter,
                hidden,
                seq_len,
            )
            .ok()?;
            let act_n = seq_len.checked_mul(inter)?;
            let act_dev = match layer.ffn_type {
                larql_compute::FfnType::Gated => match layer.activation {
                    Activation::Silu => runtime
                        .launch_geglu_silu_dev(&gate_dev, &up_dev, act_n)
                        .ok()?,
                    Activation::GeluTanh => runtime
                        .launch_geglu_gelu_tanh_dev(&gate_dev, &up_dev, act_n)
                        .ok()?,
                    _ => return None,
                },
                larql_compute::FfnType::Standard => match layer.activation {
                    Activation::Silu => runtime.launch_activation_silu_dev(&up_dev, act_n).ok()?,
                    Activation::GeluTanh => runtime
                        .launch_activation_gelu_tanh_dev(&up_dev, act_n)
                        .ok()?,
                    _ => return None,
                },
            };
            let down_dev = matmul_dev_by_fmt(
                runtime,
                down_fmt,
                layer.down.data,
                &act_dev,
                hidden,
                stored_cols,
                seq_len,
            )
            .ok()?;
            runtime.sync_dtoh_f32(&down_dev).ok()?
        };
        if down_vec.len() != seq_len * hidden {
            return None;
        }
        let out = Array2::from_shape_vec((seq_len, hidden), down_vec).ok()?;
        Some(self.apply_post_ffn_residual(layer, h_post_attn, &out))
    }

    /// Post-FFN norm + residual, shared by every decode/prefill
    /// host-orchestrated and device-resident FFN block so they can't drift.
    /// `out` may be `[1, hidden]` (decode) or `[seq, hidden]` (prefill) — both
    /// route through `norm_2d` (the per-row `norm_1d` is itself a delegate to
    /// `norm_2d`, so a single entry point is correct for both shapes). When
    /// `has_post_norms`, normalise `out` with `post_ffn_norm` (or the
    /// no-weight path when absent) before the scaled residual add; otherwise
    /// add `out` straight onto `h_post_attn`.
    fn apply_post_ffn_residual(
        &self,
        layer: &FullPipelineLayer<'_>,
        h_post_attn: &Array2<f32>,
        out: &Array2<f32>,
    ) -> Array2<f32> {
        let res_mult = layer.residual_multiplier;
        if layer.has_post_norms {
            let norm_w = layer.post_ffn_norm;
            let normed = match norm_w {
                Some(w) => self.norm_2d(layer.norm_type, out, w, layer.norm_offset, layer.eps),
                None => self.norm_2d_no_weight(out, layer.norm_offset, layer.eps),
            };
            self.add_residual_native(h_post_attn, &normed, res_mult)
        } else {
            self.add_residual_native(h_post_attn, out, res_mult)
        }
    }

    /// Device-resident decode-step attention chain — the decode twin of
    /// [`host_prefill_attention_block_device`]. Keeps the Q/K/V matvec →
    /// QK-norm/V-norm/RoPE → decode-attention → O matvec chain resident on
    /// the device with a single final readback, collapsing the per-op
    /// upload/launch/sync/readback round trips the host-orchestrated path
    /// ([`host_attention_block_hostonly`]) pays for each native kernel. The
    /// three Q/K/V projections share one upload of the normed input; the
    /// elementwise ops + the decode attention read resident buffers; the O
    /// projection consumes the resident attention output.
    ///
    /// Two device syncs remain (vs ~8 per-op round trips on the host path):
    /// one to read back the new post-RoPE K / post-V-norm V row (needed to
    /// build the full `[prev+1, kv_dim]` KV the decode-attention kernel
    /// attends over, and also the row the caller appends to the host mirror),
    /// and one final readback of the O projection. Bails to `None` (caller
    /// falls back to [`host_attention_block_hostonly`]) when there's no
    /// runtime, the attention work is below the gate, the projections aren't
    /// all Q4_K/Q6_K, or the input isn't a contiguous `[1, hidden]` slice.
    pub(crate) fn host_attention_block_device(
        &self,
        layer: &FullPipelineLayer<'_>,
        h: &Array2<f32>,
        li: usize,
        abs_position: usize,
    ) -> Option<(Array2<f32>, Vec<f32>, Vec<f32>)> {
        let runtime = self.runtime()?;
        let head_dim = layer.head_dim;
        let num_q = layer.num_q_heads;
        let num_kv = layer.num_kv_heads;
        let reps = num_q.checked_div(num_kv)?;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let scale = layer.attn_scale as f64;
        let hidden = h.shape()[1];
        // The `DecodeBackend::decode_token` trait signature carries no
        // attention logit softcap (Gemma 4's attention softcap is 0; the host
        // path applies none either) — mirrors `host_attention_block_hostonly`.
        let softcap_opt: Option<f32> = None;

        // Single decode token.
        if h.shape() != [1, hidden] {
            return None;
        }
        // Prior KV length from the host mirror (the source of truth for the
        // host-orchestrated attention); total_len = prev + 1 (the new row).
        let prev = self
            .lock_host_kv()
            .get(li)
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0);
        let total_len = prev + 1;
        // Gate the whole chain on the decode-attention work (same proxy as
        // `decode_attention_native`). Below it the host path is faster.
        let work = num_q.saturating_mul(total_len).saturating_mul(head_dim);
        if !Self::native_decode_attention_worthwhile(work) {
            return None;
        } // All four attention projections must be a k-quant the device matvec
          // handles (`matvec_dev_by_fmt` only routes Q4_K/Q6_K).
        let qf = layer.wq.format;
        let kf = layer.wk.format;
        let vf = layer.wv.format;
        let of = layer.wo.format;
        let k_quant = |f: QuantFormat| matches!(f, QuantFormat::Q4_K | QuantFormat::Q6_K);
        if !(k_quant(qf) && k_quant(kf) && k_quant(vf) && k_quant(of)) {
            return None;
        }
        // The input must be contiguous to slice for the norm + upload.
        let h_slice = h.as_slice()?;
        if h_slice.len() != hidden {
            return None;
        }

        // Input norm on host (matches the prefill attention device chain:
        // the norm weight is f32 and the residual it feeds is host-resident).
        let h_norm = self.norm_1d(
            layer.norm_type,
            h,
            layer.input_norm,
            layer.norm_offset,
            layer.eps,
        );
        let h_norm_slice = h_norm.as_slice()?;
        if h_norm_slice.len() != hidden {
            return None;
        }

        // RoPE inv_freq via the shared substrate helper — single source of
        // truth, bit-identical to the host reference.
        let frac = rope_fraction(layer);
        let pos_div = layer.rope_position_divisor as f64;
        let llama3 = layer.rope_llama3_scaling;
        let (_rotary_dim, half_rotary, inv_freq) =
            build_rope_inv_freq(layer.rope_base as f64, head_dim, frac, llama3);
        // Per-head RMSNorm eps matches the CPU reference hard-coding.
        let qk_eps = larql_compute::residual::DEFAULT_EPS;
        let qk_off = layer.qk_norm_offset;

        // Drive the device chain. Any launch error maps to `None` so the
        // caller falls back to the host-orchestrated path. All launches run
        // on the same stream (stream-ordered). Each intermediate is bound to
        // its own name and kept alive until the block-end readback so per-step
        // `CudaSlice` drops happen AFTER the final sync (on pool-less devices
        // a `CudaSlice::drop` forces a stream sync — distinct bindings keep
        // the sync count minimal, the same discipline the other chains use).
        let (o_vec, k_new_row, v_new_row): (Vec<f32>, Vec<f32>, Vec<f32>) = {
            let h_dev = runtime.upload_f32(h_norm_slice).ok()?;
            // Q/K/V projections share the normed input resident.
            let q_proj =
                matvec_dev_by_fmt(runtime, qf, layer.wq.data, &h_dev, q_dim, hidden).ok()?;
            let k_proj =
                matvec_dev_by_fmt(runtime, kf, layer.wk.data, &h_dev, kv_dim, hidden).ok()?;
            let v_proj =
                matvec_dev_by_fmt(runtime, vf, layer.wv.data, &h_dev, kv_dim, hidden).ok()?;
            // QK-norm (per-head RMSNorm; Gemma 3/4). seq_len = 1 (decode).
            let q_normed = match layer.q_norm_weight {
                Some(w) => runtime
                    .launch_rms_norm_heads_dev(&q_proj, Some(w), 1, num_q, head_dim, qk_eps, qk_off)
                    .ok()?,
                None => q_proj,
            };
            let k_normed = match layer.k_norm_weight {
                Some(w) => runtime
                    .launch_rms_norm_heads_dev(
                        &k_proj,
                        Some(w),
                        1,
                        num_kv,
                        head_dim,
                        qk_eps,
                        qk_off,
                    )
                    .ok()?,
                None => k_proj,
            };
            // V-norm (parameter-free, Gemma 4). Optional.
            let v_normed = if layer.has_v_norm {
                runtime
                    .launch_rms_norm_heads_dev(&v_proj, None, 1, num_kv, head_dim, qk_eps, 0.0)
                    .ok()?
            } else {
                v_proj
            };
            // RoPE on Q and K at `abs_position`. The `inv_freq` table is
            // uploaded once and shared by both RoPE launches.
            let inv_freq_dev = runtime.upload_f64(&inv_freq).ok()?;
            let q_rope = runtime
                .launch_rope_dev_with_invfreq(
                    &inv_freq_dev,
                    &q_normed,
                    1,
                    num_q,
                    head_dim,
                    half_rotary,
                    abs_position,
                    pos_div,
                )
                .ok()?;
            let k_rope = runtime
                .launch_rope_dev_with_invfreq(
                    &inv_freq_dev,
                    &k_normed,
                    1,
                    num_kv,
                    head_dim,
                    half_rotary,
                    abs_position,
                    pos_div,
                )
                .ok()?;
            // Read back the new post-RoPE K / post-V-norm V row (one sync).
            // Still needed under resident-KV: the caller appends this row to
            // the host KV mirror (the parity oracle + truncate/state-dump
            // source), and the state dump captures it. Under the resident path
            // the device attention no longer needs the full host prefix, but
            // the host mirror append + state-dump still consume these rows.
            let k_new_row = runtime.sync_dtoh_f32(&k_rope).ok()?;
            let v_new_row = runtime.sync_dtoh_f32(&v_normed).ok()?;
            if k_new_row.len() != kv_dim || v_new_row.len() != kv_dim {
                return None;
            }
            // GPU-006: prefer the resident-KV device path. Append the new row
            // to layer `li`'s device CudaKVCache and attend over the resident
            // K/V (no per-token full-KV host readback + re-upload). Falls back
            // to the full-upload path below on `Ok(None)` (ineligible:
            // no/undersized cache, shape mismatch, or device cursor out of
            // lockstep) or `Err` (launch failure). The eligibility is explicit
            // and testable; a deterministic ineligibility routes to full-upload
            // rather than silently degrading.
            let resident = self.resident_kv_decode_attention(
                runtime,
                li,
                &q_rope,
                &k_new_row,
                &v_new_row,
                prev,
                scale as f32,
                softcap_opt,
                num_q,
                num_kv,
                head_dim,
                kv_dim,
                reps,
            );
            let attn_dev = match resident {
                Ok(Some(out)) => {
                    self.note_resident_kv_decode(true);
                    out
                }
                // Resident path ineligible → full-upload fallback (today's
                // behavior). Counted as a fallback for diagnostics.
                Ok(None) => {
                    self.note_resident_kv_decode(false);
                    self.decode_attention_full_upload(
                        runtime,
                        li,
                        &q_rope,
                        &k_new_row,
                        &v_new_row,
                        prev,
                        total_len,
                        kv_dim,
                        num_q,
                        head_dim,
                        reps,
                        scale as f32,
                        softcap_opt,
                    )
                    .ok()?
                }
                // Launch failure on the resident path: fall back to full-upload
                // too, but record it as a fallback (not a clean ineligibility).
                Err(_) => {
                    self.note_resident_kv_decode(false);
                    self.decode_attention_full_upload(
                        runtime,
                        li,
                        &q_rope,
                        &k_new_row,
                        &v_new_row,
                        prev,
                        total_len,
                        kv_dim,
                        num_q,
                        head_dim,
                        reps,
                        scale as f32,
                        softcap_opt,
                    )
                    .ok()?
                }
            };
            // O projection: resident attention output → [1, hidden].
            let o_dev =
                matvec_dev_by_fmt(runtime, of, layer.wo.data, &attn_dev, hidden, q_dim).ok()?;
            // Single final readback of O.
            let o_vec = runtime.sync_dtoh_f32(&o_dev).ok()?;
            if o_vec.len() != hidden {
                return None;
            }
            (o_vec, k_new_row, v_new_row)
        };

        let attn_projected = vec_to_2d_row(o_vec);

        // Post-attention residual (+ optional post-attn norm) on host.
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

    /// Device-resident decode attention block — the cross-layer residency twin
    /// of [`host_attention_block_device`] (GPU-007C). Consumes a
    /// device-resident input hidden state and produces a device-resident
    /// post-attention residual, so the FFN block (and the next layer) can
    /// consume it without an inter-block / inter-layer hidden-state readback.
    ///
    /// The chain mirrors [`host_attention_block_device`] up to the O projection
    /// (input norm → Q/K/V → QK-norm/V-norm → RoPE → resident-KV attention →
    /// O, all on the device), but:
    /// - The input norm runs **on device** via [`CudaRuntime::launch_rms_norm_dev`]
    ///   (the input is already resident; the norm weight is a small f32 slice
    ///   uploaded per-call). This is parity-safe: the device `rms_norm` kernel
    ///   is the same kernel the host-readback path uses, parity-tested. Gated
    ///   to RmsNorm only (no LayerNorm device kernel).
    /// - The O projection output stays resident (no `sync_dtoh_f32` readback).
    /// - The post-attention norm + residual run **on device** via
    ///   [`CudaRuntime::launch_rms_norm_dev`] + [`CudaRuntime::launch_residual_add_dev`]
    ///   (the residual base is the resident input `h`).
    ///
    /// `h` must be a `Device` variant (the caller uploads the input once at
    /// the first eligible layer). The K/V row readback remains (the host mirror
    /// append + GPU-006 invariant — see `host_attention_block_device`). Returns
    /// `None` (caller falls back to the host path) when `h` isn't
    /// device-resident, the norm path bails, or any chained launch returns `Err`.
    #[allow(clippy::too_many_arguments)]
    /// The device-resident attention chain **up to the O projection** (B3B
    /// extraction): input norm → Q/K/V proj → QK-norm/V-norm → RoPE → resident-
    /// KV decode attention → O projection. Returns `(o_dev, k_new_row,
    /// v_new_row)` with everything device-resident (the only host crossings are
    /// the K/V row readbacks that maintain the host mirror — GPU-006 invariant).
    ///
    /// Shared by [`host_attention_block_device_resident`] (non-graph path:
    /// fresh-buffer residual add) and [`attention_into_arena`] (B3B graph path:
    /// in-place residual add into the arena input slot) so the Q/K/V → attention
    /// → O chain has a single source and cannot drift between them. Any launch
    /// error maps to `None` (the host path is the fallback).
    #[allow(clippy::too_many_arguments)]
    fn resident_attention_chain_to_o_dev(
        &self,
        runtime: &std::sync::Arc<crate::backend::CudaRuntime>,
        layer: &FullPipelineLayer<'_>,
        h_dev: &CudaSlice<f32>,
        hidden: usize,
        li: usize,
        abs_position: usize,
    ) -> Option<(CudaSlice<f32>, Vec<f32>, Vec<f32>)> {
        // RmsNorm-only (no LayerNorm device kernel) — the eligibility gate
        // already enforces this, but keep the defensive check.
        if layer.norm_type != NormType::RmsNorm {
            return None;
        }
        let head_dim = layer.head_dim;
        let num_q = layer.num_q_heads;
        let num_kv = layer.num_kv_heads;
        let reps = num_q.checked_div(num_kv)?;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let scale = layer.attn_scale as f64;
        let softcap_opt: Option<f32> = None;

        let prev = self
            .lock_host_kv()
            .get(li)
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0);
        let total_len = prev + 1;
        let work = num_q.saturating_mul(total_len).saturating_mul(head_dim);
        if !Self::native_decode_attention_worthwhile(work) {
            return None;
        }
        let qf = layer.wq.format;
        let kf = layer.wk.format;
        let vf = layer.wv.format;
        let of = layer.wo.format;
        let k_quant = |f: QuantFormat| matches!(f, QuantFormat::Q4_K | QuantFormat::Q6_K);
        if !(k_quant(qf) && k_quant(kf) && k_quant(vf) && k_quant(of)) {
            return None;
        }

        // Input norm on device (the input is already resident; the norm weight
        // is uploaded per-call). The device `rms_norm` kernel is parity-tested
        // against the host `rms_norm_eps` reference.
        let h_norm_dev = runtime
            .launch_rms_norm_dev(
                h_dev,
                Some(layer.input_norm),
                1,
                hidden,
                layer.eps as f64,
                layer.norm_offset,
            )
            .ok()?;

        // RoPE inv_freq (shared substrate helper — single source of truth).
        let frac = rope_fraction(layer);
        let pos_div = layer.rope_position_divisor as f64;
        let llama3 = layer.rope_llama3_scaling;
        let (_rotary_dim, half_rotary, inv_freq) =
            build_rope_inv_freq(layer.rope_base as f64, head_dim, frac, llama3);
        let qk_eps = larql_compute::residual::DEFAULT_EPS;
        let qk_off = layer.qk_norm_offset;

        // Drive the device chain (mirrors `host_attention_block_device`). All
        // launches run on the same stream (stream-ordered); intermediates are
        // distinct bindings kept alive until the block-end K/V readback.
        // Q/K/V projections share the normed input resident.
        let q_proj =
            matvec_dev_by_fmt(runtime, qf, layer.wq.data, &h_norm_dev, q_dim, hidden).ok()?;
        let k_proj =
            matvec_dev_by_fmt(runtime, kf, layer.wk.data, &h_norm_dev, kv_dim, hidden).ok()?;
        let v_proj =
            matvec_dev_by_fmt(runtime, vf, layer.wv.data, &h_norm_dev, kv_dim, hidden).ok()?;
        // QK-norm / V-norm (seq_len = 1 decode).
        let q_normed = match layer.q_norm_weight {
            Some(w) => runtime
                .launch_rms_norm_heads_dev(&q_proj, Some(w), 1, num_q, head_dim, qk_eps, qk_off)
                .ok()?,
            None => q_proj,
        };
        let k_normed = match layer.k_norm_weight {
            Some(w) => runtime
                .launch_rms_norm_heads_dev(&k_proj, Some(w), 1, num_kv, head_dim, qk_eps, qk_off)
                .ok()?,
            None => k_proj,
        };
        let v_normed = if layer.has_v_norm {
            runtime
                .launch_rms_norm_heads_dev(&v_proj, None, 1, num_kv, head_dim, qk_eps, 0.0)
                .ok()?
        } else {
            v_proj
        };
        // RoPE on Q and K at `abs_position`.
        let inv_freq_dev = runtime.upload_f64(&inv_freq).ok()?;
        let q_rope = runtime
            .launch_rope_dev_with_invfreq(
                &inv_freq_dev,
                &q_normed,
                1,
                num_q,
                head_dim,
                half_rotary,
                abs_position,
                pos_div,
            )
            .ok()?;
        let k_rope = runtime
            .launch_rope_dev_with_invfreq(
                &inv_freq_dev,
                &k_normed,
                1,
                num_kv,
                head_dim,
                half_rotary,
                abs_position,
                pos_div,
            )
            .ok()?;
        // Read back the new K/V row (host mirror append + state dump +
        // GPU-006 invariant — unchanged from `host_attention_block_device`).
        let k_new_row = runtime.sync_dtoh_f32(&k_rope).ok()?;
        let v_new_row = runtime.sync_dtoh_f32(&v_normed).ok()?;
        if k_new_row.len() != kv_dim || v_new_row.len() != kv_dim {
            return None;
        }
        // GPU-006: prefer the resident-KV device path (unchanged).
        let resident = self.resident_kv_decode_attention(
            runtime,
            li,
            &q_rope,
            &k_new_row,
            &v_new_row,
            prev,
            scale as f32,
            softcap_opt,
            num_q,
            num_kv,
            head_dim,
            kv_dim,
            reps,
        );
        let attn_dev = match resident {
            Ok(Some(out)) => {
                self.note_resident_kv_decode(true);
                out
            }
            Ok(None) => {
                self.note_resident_kv_decode(false);
                self.decode_attention_full_upload(
                    runtime,
                    li,
                    &q_rope,
                    &k_new_row,
                    &v_new_row,
                    prev,
                    total_len,
                    kv_dim,
                    num_q,
                    head_dim,
                    reps,
                    scale as f32,
                    softcap_opt,
                )
                .ok()?
            }
            Err(_) => {
                self.note_resident_kv_decode(false);
                self.decode_attention_full_upload(
                    runtime,
                    li,
                    &q_rope,
                    &k_new_row,
                    &v_new_row,
                    prev,
                    total_len,
                    kv_dim,
                    num_q,
                    head_dim,
                    reps,
                    scale as f32,
                    softcap_opt,
                )
                .ok()?
            }
        };
        // O projection: resident attention output → [hidden], stays resident.
        let o_dev = matvec_dev_by_fmt(runtime, of, layer.wo.data, &attn_dev, hidden, q_dim).ok()?;
        Some((o_dev, k_new_row, v_new_row))
    }

    /// Resident attention block for a single decode step (non-graph path):
    /// runs [`resident_attention_chain_to_o_dev`] then the post-attn norm +
    /// residual into a **fresh** device buffer. This is the resident-hidden
    /// parity path + the fallback for the B3B graph path. Returns
    /// `(h_post_attn_dev, k_new_row, v_new_row)`.
    fn host_attention_block_device_resident(
        &self,
        runtime: &std::sync::Arc<crate::backend::CudaRuntime>,
        layer: &FullPipelineLayer<'_>,
        h: &DecodeHiddenState,
        li: usize,
        abs_position: usize,
    ) -> Option<(CudaSlice<f32>, Vec<f32>, Vec<f32>)> {
        // The resident path needs the device-resident input hidden state.
        let hidden = h.hidden();
        let h_dev = match h {
            DecodeHiddenState::Device { dev, .. } => dev,
            DecodeHiddenState::Host(_) => return None,
        };
        if h_dev.len() != hidden {
            return None;
        }
        let (o_dev, k_new_row, v_new_row) = self.resident_attention_chain_to_o_dev(
            runtime,
            layer,
            h_dev,
            hidden,
            li,
            abs_position,
        )?;

        // Post-attention norm + residual on device (the resident-hidden
        // collapse: the residual base `h_dev` is resident, so no readback).
        let res_mult = layer.residual_multiplier;
        let h_post_attn_dev = if layer.has_post_norms {
            let normed_dev = runtime
                .launch_rms_norm_dev(
                    &o_dev,
                    Some(layer.post_attn_norm),
                    1,
                    hidden,
                    layer.eps as f64,
                    layer.norm_offset,
                )
                .ok()?;
            runtime
                .launch_residual_add_dev(h_dev, &normed_dev, hidden, res_mult)
                .ok()?
        } else {
            runtime
                .launch_residual_add_dev(h_dev, &o_dev, hidden, res_mult)
                .ok()?
        };
        Some((h_post_attn_dev, k_new_row, v_new_row))
    }

    /// B3B single-stream graph path: resident attention block whose post-attn
    /// residual is written **in place** into the arena input slot
    /// (`arena.input(flip)`) via [`CudaRuntime::launch_residual_add_inplace_into`],
    /// so the FFN graph reads the post-attn residual from the exact stable
    /// address it captured — zero per-layer D2D seed copy. Runs the same
    /// [`resident_attention_chain_to_o_dev`] as the non-graph path, then writes
    /// `arena_input += res_mult * (post_attn_norm(o_dev) | o_dev)`. Returns the
    /// new K/V rows for the host-mirror append (which happens between attention
    /// and FFN, exactly as in the non-graph path).
    ///
    /// `arena_input` is borrowed by shared reference for the in-place add; the
    /// device write happens through the `unsafe` kernel launch (element-wise
    /// independent), and single-stream execution guarantees no concurrent
    /// access. Returns `None` on any chain launch error (caller falls back to
    /// the non-graph path before appending K/V).
    fn attention_into_arena(
        &self,
        runtime: &std::sync::Arc<crate::backend::CudaRuntime>,
        layer: &FullPipelineLayer<'_>,
        arena_input: &CudaSlice<f32>,
        hidden: usize,
        li: usize,
        abs_position: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let (o_dev, k_new_row, v_new_row) = self.resident_attention_chain_to_o_dev(
            runtime,
            layer,
            arena_input,
            hidden,
            li,
            abs_position,
        )?;
        // Post-attn norm + in-place residual into the arena input slot. The
        // input norm earlier in the chain already consumed `arena_input` into a
        // separate buffer, so reading it again here as the residual base is the
        // original layer input; the in-place add leaves `arena_input` holding
        // the post-attn residual that the FFN graph reads.
        let res_mult = layer.residual_multiplier;
        let stream = runtime.stream();
        if layer.has_post_norms {
            let normed_dev = runtime
                .launch_rms_norm_dev(
                    &o_dev,
                    Some(layer.post_attn_norm),
                    1,
                    hidden,
                    layer.eps as f64,
                    layer.norm_offset,
                )
                .ok()?;
            runtime
                .launch_residual_add_inplace_into(
                    stream,
                    arena_input,
                    &normed_dev,
                    hidden,
                    res_mult,
                )
                .ok()?;
        } else {
            runtime
                .launch_residual_add_inplace_into(stream, arena_input, &o_dev, hidden, res_mult)
                .ok()?;
        }
        Some((k_new_row, v_new_row))
    }

    /// One attention block for a single decode step:
    /// norm → Q/K/V proj → QK-norm → RoPE → GQA attend → O proj → residual.
    /// Returns `(h_post_attn, k_new_row, v_new_row)`. Tries the device-resident
    /// chain first ([`host_attention_block_device`], which keeps the Q/K/V →
    /// QK-norm/V-norm/RoPE → decode-attention → O chain resident on the device
    /// with a single readback); falls back to the host-orchestrated path
    /// ([`host_attention_block_hostonly`]) when the device path is unavailable
    /// or the layer features aren't supported.
    pub(crate) fn host_attention_block(
        &self,
        layer: &FullPipelineLayer<'_>,
        h: &Array2<f32>,
        li: usize,
        abs_position: usize,
    ) -> Option<(Array2<f32>, Vec<f32>, Vec<f32>)> {
        if let Some(out) = self.host_attention_block_device(layer, h, li, abs_position) {
            return Some(out);
        }
        self.host_attention_block_hostonly(layer, h, li, abs_position)
    }

    /// Full-KV-upload decode attention — the fallback for the resident-KV
    /// path (GPU-006) and the pre-GPU-006 behavior. Builds the full
    /// `[total_len, kv_dim]` K/V from the host mirror prefix + the new row,
    /// uploads it fresh, and attends. Used when
    /// [`resident_kv_decode_attention`] returns `Ok(None)` (ineligible) or
    /// `Err` (launch failure). `q_dev` is the resident post-RoPE Q (reused
    /// from the device chain); only the full K/V is re-uploaded. Returns the
    /// resident `[num_q * head_dim]` attention output (no sync/dtoh) so the O
    /// projection consumes it on the same stream.
    #[allow(clippy::too_many_arguments)]
    fn decode_attention_full_upload(
        &self,
        runtime: &std::sync::Arc<crate::backend::CudaRuntime>,
        li: usize,
        q_dev: &CudaSlice<f32>,
        k_new_row: &[f32],
        v_new_row: &[f32],
        prev: usize,
        total_len: usize,
        kv_dim: usize,
        num_q: usize,
        head_dim: usize,
        reps: usize,
        scale: f32,
        softcap: Option<f32>,
    ) -> Result<CudaSlice<f32>, ()> {
        // Build the full KV [total_len, kv_dim] from the prior host mirror
        // prefix + the new row (the host reference concatenates the same way).
        // The mirror holds post-RoPE K / post-V-norm V.
        let (k_full, v_full): (Vec<f32>, Vec<f32>) = {
            let kv = self.lock_host_kv();
            match kv.get(li) {
                Some((k_cache, v_cache)) if prev > 0 => {
                    let kc = k_cache.as_slice().unwrap_or(&[]);
                    let vc = v_cache.as_slice().unwrap_or(&[]);
                    let need = prev * kv_dim;
                    if kc.len() < need || vc.len() < need {
                        return Err(());
                    }
                    let mut k_full = Vec::with_capacity(total_len * kv_dim);
                    let mut v_full = Vec::with_capacity(total_len * kv_dim);
                    k_full.extend_from_slice(&kc[..need]);
                    v_full.extend_from_slice(&vc[..need]);
                    k_full.extend_from_slice(k_new_row);
                    v_full.extend_from_slice(v_new_row);
                    (k_full, v_full)
                }
                _ => (k_new_row.to_vec(), v_new_row.to_vec()),
            }
        };
        let score_len = num_q.checked_mul(total_len).ok_or(())?;
        let k_dev = runtime.upload_f32(&k_full).map_err(|_| ())?;
        let v_dev = runtime.upload_f32(&v_full).map_err(|_| ())?;
        // Kernel-write-only scratch: device-local zero alloc, no htod.
        let mut scores_dev = runtime.alloc_zeros_f32(score_len).map_err(|_| ())?;
        runtime
            .launch_decode_attention_dev(
                q_dev,
                &k_dev,
                &v_dev,
                &mut scores_dev,
                scale,
                softcap,
                num_q,
                head_dim,
                kv_dim,
                reps,
                total_len,
            )
            .map_err(|_| ())
    }

    /// Host-orchestrated decode-step attention block
    /// Host-orchestrated decode-step attention block — the parity oracle +
    /// fallback for [`host_attention_block_device`]. Each native kernel
    /// (quant matvec, QK-norm/V-norm, RoPE, decode-attention, residual) runs
    /// its own upload/launch/sync/readback round trip; the device-resident
    /// chain collapses these into a single readback.
    pub(crate) fn host_attention_block_hostonly(
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
    /// post-ffn residual. Tries the device-resident chain first
    /// ([`host_ffn_block_device`], which keeps the gate/up/activation/down
    /// activations on the device across the whole chain and reads back exactly
    /// once); falls back to the host-orchestrated path
    /// ([`host_ffn_block_hostonly`]) when the device path is unavailable or
    /// the layer features aren't supported.
    pub(crate) fn host_ffn_block(
        &self,
        layer: &FullPipelineLayer<'_>,
        h_post_attn: &Array2<f32>,
        hidden: usize,
        inter: usize,
    ) -> Option<Array2<f32>> {
        if let Some(out) = self.host_ffn_block_device(layer, h_post_attn, hidden, inter) {
            return Some(out);
        }
        self.host_ffn_block_hostonly(layer, h_post_attn, hidden, inter)
    }

    /// Decode FFN block, host-orchestrated reference path: pre-ffn norm →
    /// gate/up proj → activation → down proj → post-ffn residual. Each
    /// projection is a separate htod/launch/dtoh round-trip (the elementwise
    /// ops run on host). This is the parity oracle for the device-resident
    /// chain ([`host_ffn_block_device`]) and the fallback when the device path
    /// bails.
    pub(crate) fn host_ffn_block_hostonly(
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

        // Post-FFN residual (+ optional post-ffn norm) — shared with the
        // device-resident path via `apply_post_ffn_residual`.
        Some(self.apply_post_ffn_residual(layer, h_post_attn, &out))
    }

    /// Device-resident decode FFN chain: pre-ffn norm (host) → gate/up matvec
    /// → activation → down matvec (all on the device, chained via
    /// device-resident buffers) → single `sync_dtoh` readback → post-ffn norm
    /// + residual (host).
    ///
    /// The decode twin of [`host_prefill_ffn_block_device`]: it collapses the
    /// per-projection round-trips the host-orchestrated path pays between the
    /// gate/up matvecs, the activation, and the down matvec. Instead of 3
    /// separate htod(input)+dtoh(output) cycles the chain uploads the normed
    /// input once, chains four kernels on the same CUDA stream (stream-ordered,
    /// so no inter-kernel sync), and reads the `[hidden]` down output back
    /// exactly once. The norm + post-ffn residual stay on the host (the
    /// residual needs `h_post_attn`, which is host-resident from the attention
    /// block; threading it onto the device too is a later collapse slice).
    ///
    /// Returns `None` (caller falls back to the host-only path) when: no
    /// runtime; the work is below the activation gate (`inter <
    /// LARQL_CUDA_ACTIVATION_NATIVE_MIN_ELEMS`); the gate/up/down formats aren't
    /// Q4_K/Q6_K; the down matrix's stored width is padded beyond `inter; the
    /// input isn't a contiguous `[1, hidden]` row; or the activation/ffn-type
    /// combination isn't one of the native kernels. `Err` from any chained
    /// launch also maps to `None` (the host path is the documented fallback).
    pub(crate) fn host_ffn_block_device(
        &self,
        layer: &FullPipelineLayer<'_>,
        h_post_attn: &Array2<f32>,
        hidden: usize,
        inter: usize,
    ) -> Option<Array2<f32>> {
        let runtime = self.runtime()?;
        // Decode is a single token: the work is `inter` (the gate/up/down
        // contraction width). Below the activation gate the host path is
        // faster (no fusion benefit, only transfer+sync overhead) — mirrors
        // the other `*_NATIVE_MIN_ELEMS` gates and the prefill device chain.
        if !Self::native_activation_worthwhile(inter) {
            return None;
        }
        // All three FFN projections must be a k-quant the device matvec handles.
        let gate_fmt = layer.gate.format;
        let up_fmt = layer.up.format;
        let down_fmt = layer.down.format;
        if !matches!(
            (gate_fmt, up_fmt, down_fmt),
            (QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q4_K)
                | (QuantFormat::Q6_K, QuantFormat::Q6_K, QuantFormat::Q6_K)
        ) {
            return None;
        }
        // The down matvec must contract exactly `inter` columns — a padded
        // stored width would need a device-side zero-pad step the chain
        // doesn't perform, so bail to the host path (which pads on host).
        let stored_cols = down_stored_cols(layer, hidden, inter)?;
        if stored_cols != inter {
            return None;
        }
        // The norm path needs a contiguous `[1, hidden]` input to slice.
        if h_post_attn.shape() != [1, hidden] {
            return None;
        }

        // Pre-FFN norm on the host (the norm weight is f32; threading it onto
        // the device + chaining norm→gate/up is the next collapse slice).
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
        let h_in_slice = h_in.as_slice()?;

        // Drive the device chain. Any launch error maps to `None` so the
        // caller falls back to the host-orchestrated path (the documented
        // contract for every native dispatch in this backend).
        let down_vec: Vec<f32> = {
            let x_dev = runtime.upload_f32(h_in_slice).ok()?;
            let gate_dev =
                matvec_dev_by_fmt(runtime, gate_fmt, layer.gate.data, &x_dev, inter, hidden)
                    .ok()?;
            let up_dev =
                matvec_dev_by_fmt(runtime, up_fmt, layer.up.data, &x_dev, inter, hidden).ok()?;
            let act_dev = match layer.ffn_type {
                larql_compute::FfnType::Gated => match layer.activation {
                    Activation::Silu => runtime
                        .launch_geglu_silu_dev(&gate_dev, &up_dev, inter)
                        .ok()?,
                    Activation::GeluTanh => runtime
                        .launch_geglu_gelu_tanh_dev(&gate_dev, &up_dev, inter)
                        .ok()?,
                    _ => return None,
                },
                larql_compute::FfnType::Standard => match layer.activation {
                    Activation::Silu => runtime.launch_activation_silu_dev(&up_dev, inter).ok()?,
                    Activation::GeluTanh => runtime
                        .launch_activation_gelu_tanh_dev(&up_dev, inter)
                        .ok()?,
                    _ => return None,
                },
            };
            let down_dev = matvec_dev_by_fmt(
                runtime,
                down_fmt,
                layer.down.data,
                &act_dev,
                hidden,
                stored_cols,
            )
            .ok()?;
            runtime.sync_dtoh_f32(&down_dev).ok()?
        };
        if down_vec.len() != hidden {
            return None;
        }
        let out = vec_to_2d_row(down_vec);

        // Post-FFN norm + residual on the host — shared with the host-only
        // path via `apply_post_ffn_residual` so the two can't drift.
        Some(self.apply_post_ffn_residual(layer, h_post_attn, &out))
    }

    /// Device-resident decode FFN block — the cross-layer residency twin of
    /// [`host_ffn_block_device`] (GPU-007D). Consumes a device-resident
    /// post-attention hidden state and produces a device-resident post-FFN
    /// hidden state, so the next layer's attention block can consume it
    /// without an inter-layer hidden-state readback/upload.
    ///
    /// The chain mirrors [`host_ffn_block_device`] up to the down projection
    /// (pre-ffn norm → gate/up → activation → down, all on device), but:
    /// - The pre-FFN norm runs **on device** via [`CudaRuntime::launch_rms_norm_dev`]
    ///   (the input is already resident). Parity-safe: same kernel as the
    ///   host-readback path. Gated to RmsNorm-only (no LayerNorm device kernel).
    /// - The down projection output stays resident (no `sync_dtoh_f32`).
    /// - The post-FFN norm + residual run **on device** via
    ///   [`CudaRuntime::launch_rms_norm_dev`] + [`CudaRuntime::launch_residual_add_dev`]
    ///   — the residual base is `h_post_attn_dev` (the same resident input),
    ///   matching `apply_post_ffn_residual`'s `h + res_mult * normed(out)`.
    ///
    /// Returns `None` (caller falls back to the host path) when the norm path
    /// bails (non-RmsNorm, `None`-weight pre-ffn norm under `!has_post_norms`
    /// without a weight to upload) or any chained launch returns `Err`. The
    /// eligibility gates (a supported resident-hidden FFN triple — see
    /// [`supported_resident_ffn_triple`]; `inter` ≥ activation gate;
    /// `stored_cols == inter`) are enforced by [`resident_hidden_layer_eligible`]
    /// before this is called, but the defensive checks stay for direct-call
    /// safety.
    fn host_ffn_block_device_resident(
        &self,
        runtime: &std::sync::Arc<crate::backend::CudaRuntime>,
        layer: &FullPipelineLayer<'_>,
        h_post_attn_dev: &CudaSlice<f32>,
        hidden: usize,
        inter: usize,
    ) -> Option<CudaSlice<f32>> {
        if h_post_attn_dev.len() != hidden {
            return None;
        }
        if !Self::native_activation_worthwhile(inter) {
            return None;
        }
        let gate_fmt = layer.gate.format;
        let up_fmt = layer.up.format;
        let down_fmt = layer.down.format;
        if !supported_resident_ffn_triple(gate_fmt, up_fmt, down_fmt) {
            return None;
        }
        let stored_cols = down_stored_cols(layer, hidden, inter)?;
        if stored_cols != inter {
            return None;
        }
        // RmsNorm-only (the resident pre-ffn / post-ffn norms use the device
        // `rms_norm` kernel; LayerNorm has no device twin). The eligibility
        // gate enforces this, but keep the defensive check for direct calls.
        if layer.norm_type != NormType::RmsNorm {
            return None;
        }
        // Pre-FFN norm on device. When `has_post_norms`, use `pre_ffn_norm`;
        // otherwise reuse `post_attn_norm` (matches `host_ffn_block_device`).
        // The `None`-weight pre-ffn path (no `pre_ffn_norm` + `!has_post_norms`
        // + no `post_attn_norm`) is rare on Gemma; bail to the host path when
        // there's no weight to upload (the device `launch_rms_norm_dev`
        // `None`-weight arm exists, but the resident eligibility gate didn't
        // account for it — bail rather than risk a parity drift).
        let pre_norm_w = if layer.has_post_norms {
            layer.pre_ffn_norm
        } else {
            Some(layer.post_attn_norm)
        };
        let h_norm_dev = match pre_norm_w {
            Some(w) => runtime
                .launch_rms_norm_dev(
                    h_post_attn_dev,
                    Some(w),
                    1,
                    hidden,
                    layer.eps as f64,
                    layer.norm_offset,
                )
                .ok()?,
            None => {
                // The no-weight RMSNorm device path (`has_weight = 0`, w = 1.0).
                runtime
                    .launch_rms_norm_dev(
                        h_post_attn_dev,
                        None,
                        1,
                        hidden,
                        layer.eps as f64,
                        layer.norm_offset,
                    )
                    .ok()?
            }
        };

        // Drive the device chain (mirrors `host_ffn_block_device`). The down
        // output stays resident — no `sync_dtoh_f32` readback.
        let down_dev = {
            let gate_dev = matvec_dev_by_fmt(
                runtime,
                gate_fmt,
                layer.gate.data,
                &h_norm_dev,
                inter,
                hidden,
            )
            .ok()?;
            let up_dev =
                matvec_dev_by_fmt(runtime, up_fmt, layer.up.data, &h_norm_dev, inter, hidden)
                    .ok()?;
            let act_dev = match layer.ffn_type {
                larql_compute::FfnType::Gated => match layer.activation {
                    Activation::Silu => runtime
                        .launch_geglu_silu_dev(&gate_dev, &up_dev, inter)
                        .ok()?,
                    Activation::GeluTanh => runtime
                        .launch_geglu_gelu_tanh_dev(&gate_dev, &up_dev, inter)
                        .ok()?,
                    _ => return None,
                },
                larql_compute::FfnType::Standard => match layer.activation {
                    Activation::Silu => runtime.launch_activation_silu_dev(&up_dev, inter).ok()?,
                    Activation::GeluTanh => runtime
                        .launch_activation_gelu_tanh_dev(&up_dev, inter)
                        .ok()?,
                    _ => return None,
                },
            };
            matvec_dev_by_fmt(
                runtime,
                down_fmt,
                layer.down.data,
                &act_dev,
                hidden,
                stored_cols,
            )
            .ok()?
        };

        // Post-FFN norm + residual on device — the resident-hidden collapse.
        // Matches `apply_post_ffn_residual`: `out = h_post_attn + res_mult *
        // normed(down)` when `has_post_norms`, else `out = h_post_attn +
        // res_mult * down`. The residual base is the resident input.
        let res_mult = layer.residual_multiplier;
        let h_post_ffn_dev = if layer.has_post_norms {
            let norm_w = layer.post_ffn_norm;
            let normed_dev = match norm_w {
                Some(w) => runtime
                    .launch_rms_norm_dev(
                        &down_dev,
                        Some(w),
                        1,
                        hidden,
                        layer.eps as f64,
                        layer.norm_offset,
                    )
                    .ok()?,
                None => runtime
                    .launch_rms_norm_dev(
                        &down_dev,
                        None,
                        1,
                        hidden,
                        layer.eps as f64,
                        layer.norm_offset,
                    )
                    .ok()?,
            };
            runtime
                .launch_residual_add_dev(h_post_attn_dev, &normed_dev, hidden, res_mult)
                .ok()?
        } else {
            runtime
                .launch_residual_add_dev(h_post_attn_dev, &down_dev, hidden, res_mult)
                .ok()?
        };
        Some(h_post_ffn_dev)
    }

    // ───────────────────────────────────────────────────────────────────
    // LARQL-GPU-B3B: single-stream CUDA-Graph decode layer. The whole graph
    // path — attention into the arena input slot + K/V append + graph build/
    // replay — runs on the one non-NULL runtime stream, replacing B3A's
    // separate-cap_stream design and its per-layer D2D seed/output copies +
    // cross-stream syncs with zero per-layer D2D and zero per-layer syncs.
    // ───────────────────────────────────────────────────────────────────

    /// B3B: drive one graph-eligible decode layer on the single runtime stream.
    ///
    /// Runs the entire layer — place the hidden into the arena input slot,
    /// resident attention writing its post-attn residual **in place** into that
    /// slot, the K/V host-mirror append, then the FFN graph build (token 1) or
    /// replay (token 2+) — and reports the outcome. The graph reads
    /// `arena.input(flip)` (the in-place post-attn residual) and writes
    /// `arena.output(flip)` (the next layer's input), so consecutive graph
    /// layers carry hidden by flip alone with **zero per-layer D2D and zero
    /// cross-stream syncs**.
    ///
    /// `arena_out_flip` is the carry: `Some(prev)` means the incoming hidden
    /// already lives in `arena.output(prev)` (= `arena.input(this flip)`) from
    /// the previous graph layer (no placement copy); `None` means the hidden is
    /// in `h` (Host at layer 0, or a fresh Device after a non-graph/scalar
    /// layer) and is placed into the arena (one HtoD upload at layer 0, one D2D
    /// at a re-entry boundary — never per-layer in steady state).
    #[allow(clippy::too_many_arguments)]
    fn host_graph_decode_layer(
        &self,
        runtime: &std::sync::Arc<crate::backend::CudaRuntime>,
        layer: &FullPipelineLayer<'_>,
        h: &mut DecodeHiddenState,
        arena_out_flip: &mut Option<bool>,
        li: usize,
        abs_position: usize,
        hidden: usize,
        inter: usize,
        num_layers: usize,
    ) -> GraphLayerOutcome {
        use GraphLayerOutcome as O;
        // Gate 1: graph mode must be enabled (per-backend field, not env).
        if !self.graph_mode().enabled() {
            return O::NotAttempted;
        }
        // Gate 2: plan eligibility (a strict subset of the resident path so the
        // two cannot diverge).
        if !Self::native_activation_worthwhile(inter) {
            return O::NotAttempted;
        }
        if !supported_resident_ffn_triple(layer.gate.format, layer.up.format, layer.down.format) {
            return O::NotAttempted;
        }
        if down_stored_cols(layer, hidden, inter) != Some(inter)
            || layer.norm_type != NormType::RmsNorm
        {
            return O::NotAttempted;
        }

        // ── Ensure the arena + graph cache are ready for this generation ──
        let gen = match self.graph_cache.lock() {
            Ok(c) => c.generation,
            Err(_) => return O::NotAttempted,
        };
        {
            let mut arena_guard = match self.arena.lock() {
                Ok(g) => g,
                Err(_) => return O::NotAttempted,
            };
            let need_alloc = arena_guard
                .as_ref()
                .map(|a| a.generation != gen)
                .unwrap_or(true);
            if need_alloc {
                match crate::ffn_graph_state::ResidentDecodeArena::new(runtime, hidden, gen) {
                    Ok(a) => *arena_guard = Some(a),
                    Err(_) => return O::NotAttempted,
                }
            }
        }
        if let Ok(mut cache) = self.graph_cache.lock() {
            cache.ensure_capacity(num_layers);
        }

        let flip = li % 2 == 1;

        // ── Place the layer input into arena.input(flip) if not already there ──
        // When continuing from a previous graph layer, the hidden already lives
        // in arena.output(prev) = arena.input(flip); no copy. Otherwise place it
        // (HtoD upload at layer 0 / a fresh-token entry; D2D only at a re-entry
        // boundary after a non-graph layer).
        if arena_out_flip.is_none() {
            let placed = {
                let mut arena_guard = match self.arena.lock() {
                    Ok(g) => g,
                    Err(_) => return O::NotAttempted,
                };
                let arena = match arena_guard.as_mut() {
                    Some(a) => a,
                    None => return O::NotAttempted,
                };
                let target = arena.input_mut(flip);
                match &*h {
                    DecodeHiddenState::Host(arr) => {
                        let row: Vec<f32> = arr.row(0).to_vec();
                        if row.len() != hidden {
                            return O::NotAttempted;
                        }
                        // HtoD upload of the layer-0 embedding into the arena
                        // slot (the normal token-boundary upload — NOT a D2D).
                        if crate::options::gpu_profile_enabled() {
                            runtime.note_htod(row.len() * 4);
                        }
                        runtime.stream().memcpy_htod(&row, target).is_ok()
                    }
                    DecodeHiddenState::Device { dev, .. } => {
                        // Re-entry from a non-graph layer: one D2D to place the
                        // carried device buffer into the arena (boundary only).
                        if runtime.stream().memcpy_dtod(dev, target).is_ok() {
                            self.note_graph_d2d();
                            true
                        } else {
                            false
                        }
                    }
                }
            };
            if !placed {
                return O::NotAttempted;
            }
        }

        // ── Attention: read arena.input(flip), write post-attn residual in place ──
        let attn = {
            let arena_guard = match self.arena.lock() {
                Ok(g) => g,
                Err(_) => return O::NotAttempted,
            };
            let arena = match arena_guard.as_ref() {
                Some(a) => a,
                None => return O::NotAttempted,
            };
            let input = arena.input(flip);
            self.attention_into_arena(runtime, layer, input, hidden, li, abs_position)
        };
        let (k_new_row, v_new_row) = match attn {
            Some(kv) => kv,
            None => {
                // Attention bailed before the K/V host-mirror append. Restore
                // the hidden out of the arena so the non-graph path runs on it.
                self.restore_hidden_from_arena(runtime, h, flip, hidden);
                *arena_out_flip = None;
                return O::NotAttempted;
            }
        };

        // Append the new K/V row to the host mirror (GPU-006 invariant: between
        // attention and FFN, exactly once).
        self.append_kv_row_to_host_mirror(layer, li, &k_new_row, &v_new_row);

        // ── FFN graph: build (token 1) or replay (token 2+) on the runtime stream ──
        let already_built = self
            .graph_cache
            .lock()
            .ok()
            .map(|c| c.get(li).is_some())
            .unwrap_or(false);
        let graph_ok = if already_built {
            self.replay_ffn_graph_single_stream(layer, li)
        } else {
            self.build_ffn_graph_single_stream(runtime, layer, flip, hidden, inter, li, gen)
        };
        if graph_ok {
            return O::ArenaOut { flip };
        }

        // Graph failed AFTER attention already appended K/V. Clone the post-attn
        // residual out of the arena input and run the resident device FFN on it
        // (never re-attend — that would double-append K/V).
        self.note_graph_fallback();
        let h_post_attn_dev = {
            let arena_guard = match self.arena.lock() {
                Ok(g) => g,
                Err(_) => return O::NotAttempted,
            };
            let arena = match arena_guard.as_ref() {
                Some(a) => a,
                None => return O::NotAttempted,
            };
            match runtime.stream().clone_dtod(arena.input(flip)) {
                Ok(d) => d,
                Err(_) => return O::NotAttempted,
            }
        };
        self.note_graph_d2d(); // honest: failure-path clone (not steady-state)
        *arena_out_flip = None;
        match self.host_ffn_block_device_resident(runtime, layer, &h_post_attn_dev, hidden, inter) {
            Some(dev) => O::DeviceFallback(dev),
            None => O::NotAttempted,
        }
    }

    /// Build (capture) the FFN graph for one layer on the runtime stream and
    /// launch it for token 1 (B3B single stream). The graph reads
    /// `arena.input(flip)` (the in-place post-attn residual) and writes
    /// `arena.output(flip)`. No seed D2D, no cross-stream sync.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_arguments)]
    fn build_ffn_graph_single_stream(
        &self,
        runtime: &std::sync::Arc<crate::backend::CudaRuntime>,
        layer: &FullPipelineLayer<'_>,
        flip: bool,
        hidden: usize,
        inter: usize,
        li: usize,
        gen: crate::ffn_graph::GraphGenerationId,
    ) -> bool {
        let gate_fmt = layer.gate.format;
        let up_fmt = layer.up.format;
        let down_fmt = layer.down.format;
        let stream = runtime.stream();

        // Warm the weight cache so captured nodes bind stable addresses.
        let gate_w = match self.resolve_weight_by_fmt(runtime, gate_fmt, layer.gate.data) {
            Ok(w) => w,
            Err(_) => return false,
        };
        let up_w = match self.resolve_weight_by_fmt(runtime, up_fmt, layer.up.data) {
            Ok(w) => w,
            Err(_) => return false,
        };
        let down_w = match self.resolve_weight_by_fmt(runtime, down_fmt, layer.down.data) {
            Ok(w) => w,
            Err(_) => return false,
        };

        // Upload norm weights once (stable device addresses).
        let pre_norm_slice = if layer.has_post_norms {
            layer.pre_ffn_norm
        } else {
            Some(layer.post_attn_norm)
        };
        let pre_norm_dev = match runtime.upload_rms_norm_weight(pre_norm_slice) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let post_norm_dev = if layer.has_post_norms {
            match runtime.upload_rms_norm_weight(layer.post_ffn_norm) {
                Ok(d) => Some(d),
                Err(_) => return false,
            }
        } else {
            None
        };

        // Scratch buffers allocated on the runtime stream (event tracking is
        // disabled context-wide at init, so they carry no CudaEvent handles).
        let mut normed_input = match stream.alloc_zeros::<f32>(hidden) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let mut gate_out = match stream.alloc_zeros::<f32>(inter) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let mut up_out = match stream.alloc_zeros::<f32>(inter) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let mut act = match stream.alloc_zeros::<f32>(inter) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let mut down_out = match stream.alloc_zeros::<f32>(hidden) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let mut post_norm_out = if layer.has_post_norms {
            match stream.alloc_zeros::<f32>(hidden) {
                Ok(d) => Some(d),
                Err(_) => return false,
            }
        } else {
            None
        };

        // The arena buffers' device addresses are what the graph captures, so
        // they must stay live (in `self.arena`) for the whole generation. Hold
        // the arena lock through the capture so the `input_buf`/`output_buf_slot`
        // borrows are valid for the captured `*_into` launches; the borrows end
        // after the capture closure, and the guard drops at function return.
        let mut arena_guard = match self.arena.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let (input_buf, output_buf_slot) = match arena_guard.as_mut() {
            Some(a) => a.input_output_mut(flip),
            None => return false,
        };

        // ── Capture on the runtime stream ──
        // GLOBAL: the strictest mode; a forbidden sync inside capture is a
        // defect. The CaptureExitGuard suppresses note_launch/note_htod/
        // note_dtoh/note_sync for the captured 7 launches (they become graph
        // NODES, counted once via note_graph_captured_nodes at build).
        let _capture_guard = CaptureExitGuard::enter(runtime.as_ref());
        if stream
            .begin_capture(crate::ffn_graph::graph_capture_mode())
            .is_err()
        {
            return false;
        }
        let eps = layer.eps as f64;
        let offset = layer.norm_offset;
        let res_mult = layer.residual_multiplier;
        let capture_ok =
            (|| {
                // 1. Pre-FFN RMSNorm → normed_input.
                runtime.launch_rms_norm_into(
                    stream,
                    input_buf,
                    &pre_norm_dev,
                    &mut normed_input,
                    1,
                    hidden,
                    eps,
                    offset,
                    if pre_norm_slice.is_some() { 1 } else { 0 },
                )?;
                // 2. Gate matvec → gate_out.
                match gate_fmt {
                    QuantFormat::Q4_K => runtime.launch_q4k_matvec_into(
                        stream,
                        &gate_w,
                        &normed_input,
                        &mut gate_out,
                        inter,
                        hidden,
                    )?,
                    QuantFormat::Q6_K => runtime.launch_q6k_matvec_into(
                        stream,
                        &gate_w,
                        &normed_input,
                        &mut gate_out,
                        inter,
                        hidden,
                    )?,
                    _ => unreachable!("gate fmt checked above"),
                }
                // 3. Up matvec → up_out.
                match up_fmt {
                    QuantFormat::Q4_K => runtime.launch_q4k_matvec_into(
                        stream,
                        &up_w,
                        &normed_input,
                        &mut up_out,
                        inter,
                        hidden,
                    )?,
                    QuantFormat::Q6_K => runtime.launch_q6k_matvec_into(
                        stream,
                        &up_w,
                        &normed_input,
                        &mut up_out,
                        inter,
                        hidden,
                    )?,
                    _ => unreachable!("up fmt checked above"),
                }
                // 4. Activation → act.
                match (layer.ffn_type, layer.activation) {
                    (FfnType::Gated, Activation::Silu) => runtime
                        .launch_geglu_silu_into(stream, &gate_out, &up_out, &mut act, inter)?,
                    (FfnType::Gated, Activation::GeluTanh) => runtime
                        .launch_geglu_gelu_tanh_into(stream, &gate_out, &up_out, &mut act, inter)?,
                    (FfnType::Standard, Activation::Silu) => {
                        runtime.launch_activation_silu_into(stream, &up_out, &mut act, inter)?
                    }
                    (FfnType::Standard, Activation::GeluTanh) => runtime
                        .launch_activation_gelu_tanh_into(stream, &up_out, &mut act, inter)?,
                    _ => unreachable!("activation checked by plan_graph_eligible"),
                }
                // 5. Down matvec → down_out.
                match down_fmt {
                    QuantFormat::Q4_K => runtime.launch_q4k_matvec_into(
                        stream,
                        &down_w,
                        &act,
                        &mut down_out,
                        hidden,
                        inter,
                    )?,
                    QuantFormat::Q6_K => runtime.launch_q6k_matvec_into(
                        stream,
                        &down_w,
                        &act,
                        &mut down_out,
                        hidden,
                        inter,
                    )?,
                    _ => unreachable!("down fmt checked above"),
                }
                // 6/7. Optional post-FFN RMSNorm + residual add → output_buf.
                if layer.has_post_norms {
                    let pno = post_norm_out.as_mut().unwrap();
                    runtime.launch_rms_norm_into(
                        stream,
                        &down_out,
                        post_norm_dev.as_ref().unwrap(),
                        pno,
                        1,
                        hidden,
                        eps,
                        offset,
                        if layer.post_ffn_norm.is_some() { 1 } else { 0 },
                    )?;
                    runtime.launch_residual_add_into(
                        stream,
                        input_buf,
                        pno,
                        output_buf_slot,
                        hidden,
                        res_mult,
                    )
                } else {
                    runtime.launch_residual_add_into(
                        stream,
                        input_buf,
                        &down_out,
                        output_buf_slot,
                        hidden,
                        res_mult,
                    )
                }
            })();

        // Instantiate with the cudarc-forced AUTO_FREE_ON_LAUNCH flag (see
        // `ffn_graph::graph_instantiate_flags`).
        let graph = match stream.end_capture(crate::ffn_graph::graph_instantiate_flags()) {
            Ok(Some(g)) => g,
            _ => {
                self.note_graph_failure();
                return false;
            }
        };
        if capture_ok.is_err() {
            self.note_graph_failure();
            return false;
        }
        let _ = graph.upload(); // pre-stage the first launch

        let entry = crate::ffn_graph_state::ResidentFfnGraph {
            graph: Some(graph),
            scratch: Some(crate::ffn_graph_state::ResidentFfnGraphScratch {
                normed_input,
                gate_out,
                up_out,
                act,
                down_out,
                post_norm_out,
            }),
            weights: Some(crate::ffn_graph_state::RetainedWeights {
                gate: gate_w,
                up: up_w,
                down: down_w,
                pre_norm_weight: pre_norm_dev,
                post_norm_weight: post_norm_dev,
            }),
        };
        // Token 1: launch the just-built graph (capture does not execute).
        match entry.replay() {
            Ok(()) => {
                self.note_graph_build();
                self.note_graph_submission();
                let node_count = resident_ffn_node_count(layer.has_post_norms);
                self.note_graph_captured_nodes(node_count);
                self.note_graph_logical_exec(node_count);
                if let Ok(mut cache) = self.graph_cache.lock() {
                    if cache.generation == gen && li < cache.layers.len() {
                        cache.layers[li] = Some(entry);
                    }
                }
                true
            }
            Err(_) => {
                self.note_graph_failure();
                false
            }
        }
    }

    /// Replay an already-built FFN graph for one layer on the runtime stream
    /// (B3B, token 2+). The graph reads the in-place post-attn residual from
    /// `arena.input(flip)` and writes `arena.output(flip)` — one submission,
    /// no D2D, no cross-stream sync. (The graph stores the stream it was
    /// captured on — the runtime stream — so `launch` lands there.)
    fn replay_ffn_graph_single_stream(&self, layer: &FullPipelineLayer<'_>, li: usize) -> bool {
        let replay_result = {
            let cache = match self.graph_cache.lock() {
                Ok(c) => c,
                Err(_) => return false,
            };
            match cache.get(li) {
                Some(graph) => graph.replay(),
                None => return false,
            }
        };
        match replay_result {
            Ok(()) => {
                self.note_graph_submission();
                self.note_graph_logical_exec(resident_ffn_node_count(layer.has_post_norms));
                true
            }
            Err(_) => {
                self.note_graph_fallback();
                false
            }
        }
    }

    /// Append one new K/V row to the host mirror for layer `li` (the GPU-006
    /// host-mirror invariant: between attention and FFN, exactly once). Shared
    /// by the B3B graph path; the resident/host fallbacks inline the same logic.
    fn append_kv_row_to_host_mirror(
        &self,
        layer: &FullPipelineLayer<'_>,
        li: usize,
        k_new_row: &[f32],
        v_new_row: &[f32],
    ) {
        let prof = crate::options::gpu_profile_enabled();
        let mt0 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut rows_copied = 0usize;
        {
            let mut kv = self.lock_host_kv();
            if let Some((k_cache, v_cache)) = kv.get_mut(li) {
                let kv_dim = layer.num_kv_heads * layer.head_dim;
                let prev = k_cache.shape()[0];
                rows_copied = prev;
                let mut k_new = Array2::zeros((prev + 1, kv_dim));
                let mut v_new = Array2::zeros((prev + 1, kv_dim));
                if prev > 0 {
                    k_new.slice_mut(ndarray::s![..prev, ..]).assign(k_cache);
                    v_new.slice_mut(ndarray::s![..prev, ..]).assign(v_cache);
                }
                k_new.slice_mut(ndarray::s![prev..prev + 1, ..]).assign(
                    &Array2::from_shape_vec((1, kv_dim), k_new_row.to_vec())
                        .expect("k_new_row shape"),
                );
                v_new.slice_mut(ndarray::s![prev..prev + 1, ..]).assign(
                    &Array2::from_shape_vec((1, kv_dim), v_new_row.to_vec())
                        .expect("v_new_row shape"),
                );
                *k_cache = k_new;
                *v_cache = v_new;
            }
        }
        if let Some(t0) = mt0 {
            self.note_mirror_append(t0.elapsed().as_nanos() as u64, rows_copied);
        }
    }

    /// Restore the hidden state out of `arena.input(flip)` into `h` as a device
    /// buffer, after a graph-path attention bail (so the non-graph path runs on
    /// the correct layer input). The arena slot holds the original layer input
    /// when attention bails (the in-place residual is the last step of
    /// `attention_into_arena` and only runs on success).
    fn restore_hidden_from_arena(
        &self,
        runtime: &std::sync::Arc<crate::backend::CudaRuntime>,
        h: &mut DecodeHiddenState,
        flip: bool,
        hidden: usize,
    ) {
        let dev = {
            let arena_guard = match self.arena.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            let Some(arena) = arena_guard.as_ref() else {
                return;
            };
            match runtime.stream().clone_dtod(arena.input(flip)) {
                Ok(d) => d,
                Err(_) => return,
            }
        };
        self.note_graph_d2d(); // honest: bail-restore clone (failure path only)
        *h = DecodeHiddenState::Device { dev, hidden };
    }

    /// Read `arena.output(flip)` back to the host (a DtoH readback — no D2D:
    /// `sync_dtoh_f32` reads the arena buffer directly via the stream). Used when
    /// exiting the arena (a non-graph layer follows, or the per-layer scalar is
    /// non-identity, or the final token output).
    fn read_arena_output_to_host(
        &self,
        runtime: &std::sync::Arc<crate::backend::CudaRuntime>,
        flip: bool,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        let v = {
            let arena_guard = self.arena.lock().ok()?;
            let arena = arena_guard.as_ref()?;
            runtime.sync_dtoh_f32(arena.output(flip)).ok()?
        };
        if v.len() == hidden {
            Some(v)
        } else {
            None
        }
    }
    /// Resolve a quant weight through the cache by format (B3A-5 helper).
    fn resolve_weight_by_fmt(
        &self,
        runtime: &std::sync::Arc<crate::backend::CudaRuntime>,
        fmt: QuantFormat,
        data: &[u8],
    ) -> Result<std::sync::Arc<CudaSlice<u8>>, crate::backend::RuntimeError> {
        match fmt {
            QuantFormat::Q4_K => runtime.resolve_q4k_weight(data),
            QuantFormat::Q6_K => runtime.resolve_q6k_weight(data),
            _ => Err(crate::backend::RuntimeError::usage(format!(
                "resolve_weight_by_fmt: unsupported format {fmt:?}"
            ))),
        }
    }

    /// Hybrid-MoE FFN block for a single decode token (Gemma 4 26B-A4B
    /// shape). Runs the dense slab via the existing [`host_ffn_block`]
    /// (native quant matvec projections + host elementwise + post-FFN
    /// norm + residual), the expert block via the device-routed
    /// [`Self::moe_combine_row_device`] (native per-expert gate/up/down
    /// Q4_K matvecs; falls back to the substrate [`cpu_moe_forward`] when
    /// there is no runtime or the experts aren't Q4_K × f32), then
    /// combines the two with the Gemma-4 outer post-norm + residual — the
    /// structure of `larql-inference::moe_ffn_block_cpu_with_index`:
    ///
    ///   h1 = dense_slab - h_post_attn   (the dense delta; the slab
    ///                                    already carries the residual)
    ///   h2 = expert_contribution(h_post_attn)  (the expert block)
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
        // Single-token decode: allocate the expert scratch once (no loop to hoist out of).
        let mut expert_out = vec![0.0f32; hidden];
        let mut act = vec![0.0f32; moe.inter_padded()];
        let out = self
            .moe_combine_row_device(
                ha,
                dense,
                moe,
                layer,
                outer_w,
                &mut combined,
                &mut expert_out,
                &mut act,
            )
            .unwrap_or_else(|| moe_combine_row(ha, dense, moe, layer, outer_w, &mut combined));
        Some(vec_to_2d_row(out))
    }

    /// Multi-position hybrid-MoE FFN block (prefill). Same structure as
    /// [`host_ffn_block_moe_decode`] but the dense slab uses the amortised
    /// native quant matmul across all `seq_len` positions
    /// ([`host_prefill_ffn_block`]), and the expert block + outer combine
    /// run per position (the expert contribution is single-token). Each
    /// position tries the device-routed [`Self::moe_combine_row_device`]
    /// first (native Q4_K expert matvecs) and falls back to the substrate
    /// [`cpu_moe_forward`] when the device path bails. The `combined`
    /// scratch is hoisted out of the per-position loop to avoid per-token
    /// allocation.
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
        // Hoist the expert scratch out of the per-position loop (mirrors
        // `combined`): `expert_out` is `[hidden]` (re-zeroed per token inside
        // `moe_expert_contribution_q4k` before accumulation) and `act` is
        // `[inter_padded]` with padding columns that stay zero across
        // positions — so both reuse cleanly across the prefill loop.
        let mut expert_out = vec![0.0f32; hidden];
        let mut act = vec![0.0f32; moe.inter_padded()];
        let mut out = vec![0.0f32; seq_len * hidden];
        for pos in 0..seq_len {
            let off = pos * hidden;
            let row = self
                .moe_combine_row_device(
                    &ha[off..off + hidden],
                    &dense[off..off + hidden],
                    moe,
                    layer,
                    outer_w,
                    &mut combined,
                    &mut expert_out,
                    &mut act,
                )
                .unwrap_or_else(|| {
                    moe_combine_row(
                        &ha[off..off + hidden],
                        &dense[off..off + hidden],
                        moe,
                        layer,
                        outer_w,
                        &mut combined,
                    )
                });
            out[off..off + hidden].copy_from_slice(&row);
        }
        Array2::from_shape_vec((seq_len, hidden), out).ok()
    }

    /// Device-routed expert contribution for a single token: the routing +
    /// post-expert norm stay on the host, but every per-expert gate/up/down
    /// projection runs through the native CUDA `q4k_matvec` kernel
    /// (native-then-CPU fallback via [`QuantMatVec::q4k_matvec`]). Returns
    /// `None` (caller falls back to the substrate [`cpu_moe_forward`]) when
    /// there is no runtime or the experts aren't Q4_K × f32 (the device
    /// kernel's math — see [`moe_expert_contribution_q4k`]). On a no-CUDA
    /// host this is always `None`, so the MoE block keeps its existing
    /// `cpu_moe_forward` (Q8_K-direct SDOT on Apple Silicon) behaviour
    /// unchanged.
    ///
    /// On a CUDA host this first tries the device-resident per-expert chain
    /// ([`Self::moe_expert_contribution_device_chain`] — single upload of the
    /// expert input shared by every expert's gate/up, weights served from the
    /// Session 19 weight cache, one readback per expert) and falls back to the
    /// per-call matvec path when the chain bails (padding, non-gated
    /// activation, etc.). Both paths perform the same Q4_K × f32 math.
    pub(crate) fn moe_expert_contribution_device(
        &self,
        h: &[f32],
        moe: &larql_compute::MoeLayerWeights<'_>,
        norm_offset: f32,
        eps: f32,
        expert_out: &mut [f32],
        act: &mut [f32],
    ) -> Option<Vec<f32>> {
        let runtime = self.runtime()?;
        // Try the device-resident per-expert chain first (single input upload
        // + one readback/expert, mirroring the decode/prefill FFN device
        // chains). Falls through to the per-call matvec path when the chain
        // bails (down-padding, non-gated activation, etc.) — both paths share
        // the same Q4_K × f32 math so the two can't numerically diverge.
        if let Some(out) =
            self.moe_expert_contribution_device_chain(runtime, h, moe, norm_offset, eps, expert_out)
        {
            return Some(out);
        }
        moe_expert_contribution_q4k(h, moe, norm_offset, eps, expert_out, act, |w, x, r, k| {
            self.q4k_matvec(w, x, r, k)
        })
    }

    /// Device-resident per-expert FFN chain — the Session 24 device expert
    /// path collapsed to a single input upload + one readback per expert,
    /// mirroring the decode/prefill FFN device chains (Sessions 20-23). All
    /// top-k experts share one upload of `expert_input`; each expert's
    /// gate/up/activation/down runs as a device-resident chain (weights served
    /// from the Session 19 weight cache; the intermediate gate/up/activation
    /// outputs stay on the device between launches, dropping after the
    /// per-expert readback). Routing + the post-expert norm stay on the host,
    /// exactly as in [`moe_expert_contribution_q4k`].
    ///
    /// Returns `None` (caller falls back to the per-call matvec path) when:
    /// the experts aren't Q4_K; `hidden` isn't a 256-multiple; the down
    /// contraction needs zero-padding (`inter_padded != inter`, where the
    /// chain would feed the activation output directly into the down matvec);
    /// the activation isn't one of the native gated kernels; or any chained
    /// launch returns `Err` (the host per-call path is the documented
    /// fallback). `runtime` must already be present (the caller gates on it).
    fn moe_expert_contribution_device_chain(
        &self,
        runtime: &crate::backend::CudaRuntime,
        h: &[f32],
        moe: &larql_compute::MoeLayerWeights<'_>,
        norm_offset: f32,
        eps: f32,
        expert_out: &mut [f32],
    ) -> Option<Vec<f32>> {
        use larql_compute::cpu::ops::moe::{
            moe_expert_input, moe_post_expert_output, moe_route_from_router_input, moe_router_input,
        };
        let hidden = h.len();
        let (half, inter) = moe_expert_chain_eligible(moe, hidden)?;
        if expert_out.len() != hidden {
            return None;
        }

        let expert_input = moe_expert_input(h, moe, norm_offset, eps);
        let router_in = moe_router_input(h, &expert_input, moe, norm_offset, eps);
        let (indices, weights) = moe_route_from_router_input(&router_in, moe);

        expert_out.fill(0.0);
        // Upload the expert input once; every expert's gate/up share this
        // resident buffer. (Session 24's per-call path re-uploaded the input
        // on every gate/up/down matvec — 3 × top_k uploads per token.)
        let x_dev = runtime.upload_f32(&expert_input).ok()?;

        for (&ei, &w) in indices.iter().zip(weights.iter()) {
            if w == 0.0 {
                continue;
            }
            let Some(&gate_up_bytes) = moe.experts_gate_up.get(ei) else {
                continue;
            };
            let Some(&down_bytes) = moe.experts_down.get(ei) else {
                continue;
            };
            if gate_up_bytes.len() < 2 * half {
                continue;
            }
            let gate_bytes = &gate_up_bytes[..half];
            let up_bytes = &gate_up_bytes[half..2 * half];

            // Per-expert device chain: gate/up share `x_dev`, activation reads
            // the gate/up outputs in place, down contracts the activation. The
            // intermediates stay resident; only `down` is read back. Any
            // launch error maps to `None` (the host per-call path is the
            // documented fallback for every native dispatch in this backend).
            let down_vec: Vec<f32> = {
                let gate_dev =
                    match runtime.launch_q4k_matvec_dev(gate_bytes, &x_dev, inter, hidden) {
                        Ok(d) => d,
                        Err(_) => return None,
                    };
                let up_dev = match runtime.launch_q4k_matvec_dev(up_bytes, &x_dev, inter, hidden) {
                    Ok(d) => d,
                    Err(_) => return None,
                };
                let act_dev = match moe.activation {
                    Activation::Silu => {
                        match runtime.launch_geglu_silu_dev(&gate_dev, &up_dev, inter) {
                            Ok(d) => d,
                            Err(_) => return None,
                        }
                    }
                    Activation::GeluTanh => {
                        match runtime.launch_geglu_gelu_tanh_dev(&gate_dev, &up_dev, inter) {
                            Ok(d) => d,
                            Err(_) => return None,
                        }
                    }
                    // `moe_expert_chain_eligible` already rejected the rest.
                    _ => return None,
                };
                // down: num_rows = hidden, contraction = inter (== inter_padded
                // here — the eligibility gate rejects padding).
                let down_dev =
                    match runtime.launch_q4k_matvec_dev(down_bytes, &act_dev, hidden, inter) {
                        Ok(d) => d,
                        Err(_) => return None,
                    };
                match runtime.sync_dtoh_f32(&down_dev) {
                    Ok(v) => v,
                    Err(_) => return None,
                }
            };
            if down_vec.len() != hidden {
                return None;
            }
            for (acc, &v) in expert_out.iter_mut().zip(down_vec.iter()) {
                *acc += w * v;
            }
        }
        Some(moe_post_expert_output(expert_out, moe, norm_offset, eps))
    }

    /// Device-routed single-token MoE combine: the device expert
    /// contribution ([`Self::moe_expert_contribution_device`]) substitutes
    /// for the substrate `cpu_moe_forward` call inside [`moe_combine_row`];
    /// the dense-delta subtraction + outer post-norm + residual run through
    /// the shared [`apply_outer_combine`] so the device and host paths
    /// can't drift on the combine wiring. Returns `None` when the device
    /// expert path bails — the caller falls back to [`moe_combine_row`]
    /// (the `cpu_moe_forward` reference). `expert_out`/`act` are
    /// caller-owned scratch (see [`moe_expert_contribution_q4k`]).
    #[allow(clippy::too_many_arguments)]
    fn moe_combine_row_device(
        &self,
        ha_row: &[f32],
        dense_row: &[f32],
        moe: &larql_compute::MoeLayerWeights<'_>,
        layer: &FullPipelineLayer<'_>,
        outer_w: Option<&[f32]>,
        combined: &mut [f32],
        expert_out: &mut [f32],
        act: &mut [f32],
    ) -> Option<Vec<f32>> {
        let h2 = self.moe_expert_contribution_device(
            ha_row,
            moe,
            layer.norm_offset,
            layer.eps,
            expert_out,
            act,
        )?;
        Some(apply_outer_combine(
            ha_row, dense_row, &h2, outer_w, layer, combined,
        ))
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
    ///
    /// Env-tunable via `LARQL_CUDA_NORM_NATIVE_MIN_ELEMS` (default `8192`,
    /// resolved once — see [`crate::options::native_thresholds`]).
    ///
    /// True when a norm of `elems` elements is large enough that the native
    /// CUDA kernel is likely to beat the host reference after the per-call
    /// device round-trip.
    fn native_norm_worthwhile(elems: usize) -> bool {
        elems >= native_thresholds().norm_native_min_elems
    }

    /// Minimum element count for a native activation dispatch to be worth the
    /// host→device upload + kernel launch + sync + device→host readback
    /// round-trip. Below this, the host `apply_activation_*` reference is
    /// faster (the activation output is read straight back to host and
    /// re-uploaded by the down projection, so there's no fusion benefit —
    /// only transfer+sync overhead). Mirrors the norm gate.
    ///
    /// Env-tunable via `LARQL_CUDA_ACTIVATION_NATIVE_MIN_ELEMS` (default
    /// `8192`).
    ///
    /// True when an activation of `elems` elements is large enough that the
    /// native CUDA kernel is likely to beat the host reference after the
    /// per-call device round-trip.
    fn native_activation_worthwhile(elems: usize) -> bool {
        elems >= native_thresholds().activation_native_min_elems
    }

    /// Minimum element count for a native residual add to be worth the
    /// host→device upload + kernel launch + sync + device→host readback
    /// round-trip. Below this, the host `h + b_scale * x` reference is faster
    /// (the residual output is read straight back to host and re-uploaded by
    /// the next op, so there's no fusion benefit — only transfer+sync
    /// overhead). Mirrors the norm / activation gates. Tuned conservatively
    /// for correctness-first (the host-orchestrated path is the parity
    /// oracle); the fully-fused single-command-buffer pipeline lifts this
    /// gate. The decode residual (`[1, hidden]`, ~3-9k elements) typically
    /// stays below this gate and keeps the host path; the prefill residual
    /// (`[seq, hidden]`) clears it once `seq * hidden >= 8192`.
    ///
    /// Env-tunable via `LARQL_CUDA_RESIDUAL_NATIVE_MIN_ELEMS` (default
    /// `8192`).
    ///
    /// True when a residual add of `elems` elements is large enough that the
    /// native CUDA kernel is likely to beat the host reference after the
    /// per-call device round-trip.
    fn native_residual_worthwhile(elems: usize) -> bool {
        elems >= native_thresholds().residual_native_min_elems
    }

    /// Minimum element count for a native RoPE dispatch to be worth the
    /// host→device upload + kernel launch + sync + device→host readback
    /// round-trip. Below this, the host `apply_rope_partial_at_full`
    /// reference is faster (the RoPE output is read straight back to host and
    /// re-uploaded by the attention dispatch, so there's no fusion benefit —
    /// only transfer+sync overhead). Mirrors the other native-path gates.
    /// Tuned conservatively for correctness-first (the host-orchestrated path
    /// is the parity oracle); the fully-fused single-command-buffer pipeline
    /// lifts this gate. The decode Q/K tensor (`[1, q_dim]`, typically a few
    /// thousand elements) often stays below this gate and keeps the host path;
    /// the prefill Q/K tensor (`[seq, q_dim]`) clears it once
    /// `seq * q_dim >= 8192`.
    ///
    /// Env-tunable via `LARQL_CUDA_ROPE_NATIVE_MIN_ELEMS` (default `8192`).
    ///
    /// True when a RoPE over `elems` elements is large enough that the native
    /// CUDA kernel is likely to beat the host reference after the per-call
    /// device round-trip.
    fn native_rope_worthwhile(elems: usize) -> bool {
        elems >= native_thresholds().rope_native_min_elems
    }

    /// Minimum attention work (num_q × total_len × head_dim) for a native
    /// decode-attention dispatch to be worth the host→device upload + kernel
    /// launch + sync + device→host readback round-trip. Below this the host
    /// `gqa_attention_decode_step` reference (rayon/spin-pool parallel over
    /// heads) is faster — there's no fusion benefit, only transfer+sync
    /// overhead, when the context is short. Mirrors the other native-path
    /// gates; tuned conservatively for correctness-first (the host-orchestrated
    /// path is the parity oracle); the fully-fused single-command-buffer
    /// pipeline lifts this gate.
    ///
    /// Env-tunable via `LARQL_CUDA_DECODE_ATTN_NATIVE_MIN_WORK` (default
    /// `8192`).
    ///
    /// True when a decode-attention over `work = num_q × total_len × head_dim`
    /// is large enough that the native CUDA kernel is likely to beat the host
    /// reference after the per-call device round-trip.
    fn native_decode_attention_worthwhile(work: usize) -> bool {
        work >= native_thresholds().decode_attn_native_min_work
    }

    /// RoPE with split-half pairing — the device twin of
    /// `larql_compute::attention::rope::apply_rope_partial_at_full`. Routes
    /// through the native CUDA `rope` kernel when a runtime is present AND
    /// the tensor is large enough to amortise the device round-trip (see
    /// `LARQL_CUDA_ROPE_NATIVE_MIN_ELEMS`); falls back to the host reference on
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
    /// device round-trip (see `LARQL_CUDA_DECODE_ATTN_NATIVE_MIN_WORK`); falls back to the
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
    ///
    /// Env-tunable via `LARQL_CUDA_PREFILL_ATTN_NATIVE_MIN_WORK` (default
    /// `8192`).
    ///
    /// True when a prefill attention over
    /// `work = seq_len × num_q × seq_len × head_dim` (causal, so ~half the
    /// QKᵀ + all the weighted-V) is large enough that the native CUDA kernel
    /// is likely to beat the host reference after the per-call device
    /// round-trip.
    fn native_prefill_attention_worthwhile(work: usize) -> bool {
        work >= native_thresholds().prefill_attn_native_min_work
    }

    /// Fused prefill (seq×seq) causal GQA attention — the device twin of
    /// `gqa_attention_with_weights` (the symmetric `gqa_attention_capture`
    /// path). Routes through the native CUDA `prefill_attention` kernel when a
    /// runtime is present AND the attention work is large enough to amortise
    /// the device round-trip (see `LARQL_CUDA_PREFILL_ATTN_NATIVE_MIN_WORK`); falls back
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
    /// amortise the device round-trip (see `LARQL_CUDA_NORM_NATIVE_MIN_ELEMS`); falls
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
    apply_outer_combine(ha_row, dense_row, &h2, outer_w, layer, combined)
}

/// Gemma-4 outer combine shared by the host ([`moe_combine_row`]) and device
/// ([`CudaBackend::moe_combine_row_device`]) MoE paths so the dense-delta
/// subtraction + outer post-norm + residual live in exactly one place — the
/// only difference between the two paths is how the expert contribution `h2`
/// is produced. `combined` is caller-owned `[hidden]` scratch, fully
/// overwritten here.
pub(crate) fn apply_outer_combine(
    ha_row: &[f32],
    dense_row: &[f32],
    h2: &[f32],
    outer_w: Option<&[f32]>,
    layer: &FullPipelineLayer<'_>,
    combined: &mut [f32],
) -> Vec<f32> {
    for (i, c) in combined.iter_mut().enumerate() {
        *c = (dense_row[i] - ha_row[i]) + h2[i];
    }
    outer_post_norm_residual(ha_row, combined, outer_w, layer.norm_offset, layer.eps)
}

/// Eligibility gate for the device-resident per-expert FFN chain
/// ([`CudaBackend::moe_expert_contribution_device_chain`]). Returns
/// `(half_byte_span, inter)` when the chain can run on this MoE layer, else
/// `None`. Pure (no device touch) so the eligibility logic — the same Q4_K +
/// 256-alignment gates as [`moe_expert_contribution_q4k`], plus the two
/// chain-specific gates (no down zero-padding, since the chain feeds the
/// activation output directly into the down matvec; and a gated activation,
/// since only `Silu`/`GeluTanh` have device-resident launchers) — is testable
/// on every host.
pub(crate) fn moe_expert_chain_eligible(
    moe: &larql_compute::MoeLayerWeights<'_>,
    hidden: usize,
) -> Option<(usize, usize)> {
    let inter = moe.intermediate_size;
    if inter == 0 || hidden == 0 {
        return None;
    }
    if !matches!(moe.expert_data_format, QuantFormat::Q4_K) {
        return None;
    }
    if !hidden.is_multiple_of(Q4_K_BLOCK_ELEMS) {
        return None;
    }
    // The chain feeds the `[inter]` activation output straight into the down
    // matvec; a padded contraction (`inter_padded > inter`) would need a
    // zero-pad step the chain doesn't perform, so bail to the per-call path
    // (which pads the host `act` scratch).
    if moe.inter_padded() != inter {
        return None;
    }
    if !matches!(moe.activation, Activation::Silu | Activation::GeluTanh) {
        return None;
    }
    let half = larql_compute::cpu::ops::moe::q4k_gate_up_half(inter, hidden)?;
    Some((half, inter))
}

/// Per-token expert-block contribution computed with **Q4_K × f32** matvecs
/// (the device path's math). Mirrors the structure of the substrate
/// [`cpu_moe_forward`] — routing → per-expert gated FFN → weighted sum →
/// post-expert norm — but runs every gate/up/down projection through the
/// supplied `matvec` closure, which performs `out[rows] = W[rows,k] @ x[k]`
/// on Q4_K weights against an f32 input (dequantize-then-dot).
///
/// **Why not reuse `cpu_moe_forward` directly:** its default hot path is
/// Q4_K-**direct** (Q8_K quantization of the input + integer SDOT), an
/// Apple-Silicon-only optimization (NEON `SDOT`). CUDA has no SDOT, so the
/// device path dequantizes Q4_K to f32 and dots with the f32 input — the
/// same math `QuantMatVec::q4k_matvec` performs. The closure is
/// `self.q4k_matvec` (native-then-CPU) on the device path and
/// `CpuBackend::q4k_matvec` on the host-only parity path; both feed the
/// same Q4_K × f32 kernel (the native one is parity-tested against the CPU
/// twin), so the device and host-only outputs match within the kernel
/// tolerance.
///
/// Returns `None` (caller falls back to `cpu_moe_forward`) when the experts
/// aren't Q4_K, the hidden dim isn't a 256-multiple (the gate/up byte split
/// assumes whole Q4_K super-blocks), or a matvec returns the wrong length.
pub(crate) fn moe_expert_contribution_q4k<M>(
    h: &[f32],
    moe: &larql_compute::MoeLayerWeights<'_>,
    norm_offset: f32,
    eps: f32,
    expert_out: &mut [f32],
    act: &mut [f32],
    mut matvec: M,
) -> Option<Vec<f32>>
where
    M: FnMut(&[u8], &[f32], usize, usize) -> Option<Vec<f32>>,
{
    let hidden = h.len();
    let inter = moe.intermediate_size;
    if inter == 0 || hidden == 0 {
        return None;
    }
    // Only Q4_K experts: the native + CPU `q4k_matvec` decode 144-byte Q4_K
    // super-blocks. Other formats (BF16 monolith, etc.) keep the
    // `cpu_moe_forward` path.
    if !matches!(moe.expert_data_format, QuantFormat::Q4_K) {
        return None;
    }
    // The gate/up byte split (and the Q4_K super-block decode) need a whole
    // number of super-blocks per row.
    if !hidden.is_multiple_of(Q4_K_BLOCK_ELEMS) {
        return None;
    }
    let inter_padded = moe.inter_padded();
    // Caller-owned scratch must match the layer's geometry. `expert_out` is
    // `[hidden]` (zeroed at the start of each call before accumulation);
    // `act` is `[inter_padded]` with the padding columns `[inter..]` zero on
    // entry (they are never written, so the down matvec reads them as zero —
    // matches the substrate `ExpertScratch::act` discipline). Hoisting these
    // out of the per-position prefill loop avoids `2 * seq_len` allocations.
    if expert_out.len() != hidden || act.len() != inter_padded {
        return None;
    }
    // gate_up layout: [2*inter, hidden] (gate rows first, then up rows). `half`
    // is one projection's byte span, sourced from the shared substrate
    // `q4k_gate_up_half` so the Q4_K row-stride lives in exactly one place.
    let half = larql_compute::cpu::ops::moe::q4k_gate_up_half(inter, hidden)?;

    let expert_input = moe_expert_input(h, moe, norm_offset, eps);
    let router_in = moe_router_input(h, &expert_input, moe, norm_offset, eps);
    let (indices, weights) = moe_route_from_router_input(&router_in, moe);

    let activation = moe.activation;
    expert_out.fill(0.0);
    for (&ei, &w) in indices.iter().zip(weights.iter()) {
        if w == 0.0 {
            continue;
        }
        let Some(&gate_up_bytes) = moe.experts_gate_up.get(ei) else {
            continue;
        };
        let Some(&down_bytes) = moe.experts_down.get(ei) else {
            continue;
        };
        if gate_up_bytes.len() < 2 * half {
            continue;
        }
        let gate_bytes = &gate_up_bytes[..half];
        let up_bytes = &gate_up_bytes[half..2 * half];
        let gate_out = matvec(gate_bytes, &expert_input, inter, hidden)?;
        let up_out = matvec(up_bytes, &expert_input, inter, hidden)?;
        if gate_out.len() != inter || up_out.len() != inter {
            return None;
        }
        // act(gate) * up into act[..inter]; padding stays zero.
        apply_activation_gated(activation, &gate_out, &up_out, &mut act[..inter]);
        let down_out = matvec(down_bytes, act, hidden, inter_padded)?;
        if down_out.len() != hidden {
            return None;
        }
        for (acc, &v) in expert_out.iter_mut().zip(down_out.iter()) {
            *acc += w * v;
        }
    }
    Some(moe_post_expert_output(expert_out, moe, norm_offset, eps))
}

/// Host-only Q4_K × f32 expert contribution — the device path's parity
/// oracle. Same structure as [`moe_expert_contribution_q4k`] but every
/// matvec runs the CPU `CpuBackend::q4k_matvec` reference (Q4_K × f32). Used
/// by the runtime-gated native parity test to lock the device-vs-host match
/// against a fresh composition. Test-only (the device path's host fallback
/// is the substrate `cpu_moe_forward`, not this oracle).
#[cfg(test)]
pub(crate) fn moe_expert_contribution_hostonly(
    h: &[f32],
    moe: &larql_compute::MoeLayerWeights<'_>,
    norm_offset: f32,
    eps: f32,
    expert_out: &mut [f32],
    act: &mut [f32],
) -> Option<Vec<f32>> {
    use larql_compute::CpuBackend;
    const CPU: CpuBackend = CpuBackend;
    moe_expert_contribution_q4k(h, moe, norm_offset, eps, expert_out, act, |w, x, r, k| {
        CPU.q4k_matvec(w, x, r, k)
    })
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

/// The down projection's stored column count (the contraction width) derived
/// from its byte length, without allocating the padded activation. The
/// device-resident FFN chain uses this to decide whether the `[seq, inter]`
/// activation feeds the down matmul directly (`stored_cols == inter`) or needs
/// a host-side zero-pad step (`stored_cols > inter`, where the chain bails to
/// the host path). Mirrors the `stored_cols` derivation in
/// [`down_padded_activation`].
fn down_stored_cols(layer: &FullPipelineLayer<'_>, hidden: usize, inter: usize) -> Option<usize> {
    let down = &layer.down;
    let bytes_per_sb = super_block_bytes(down.format)?;
    if hidden == 0 || down.data.is_empty() {
        return None;
    }
    let down_bytes_per_row = down.data.len() / hidden;
    if down_bytes_per_row == 0 || !down_bytes_per_row.is_multiple_of(bytes_per_sb) {
        return None;
    }
    let stored_cols = down_bytes_per_row / bytes_per_sb * 256;
    (stored_cols >= inter).then_some(stored_cols)
}

/// Device-resident amortised matmul dispatch by quant format: Q4_K →
/// `launch_q4k_matmul_dev`, Q6_K → `launch_q6k_matmul_dev`. Used by the
/// device-resident FFN chain so the gate/up/down projections share one upload
/// of the input and keep their outputs on the device. `None` for other
/// formats (the chain's gate condition rejects them before reaching here, but
/// the exhaustive match keeps a future format loud rather than silently
/// skipping a projection).
fn matmul_dev_by_fmt(
    runtime: &crate::backend::CudaRuntime,
    format: QuantFormat,
    weights: &[u8],
    x_dev: &cudarc::driver::CudaSlice<f32>,
    num_rows: usize,
    hidden: usize,
    seq_len: usize,
) -> Result<cudarc::driver::CudaSlice<f32>, crate::backend::RuntimeError> {
    match format {
        QuantFormat::Q4_K => {
            runtime.launch_q4k_matmul_dev(weights, x_dev, num_rows, hidden, seq_len)
        }
        QuantFormat::Q6_K => {
            runtime.launch_q6k_matmul_dev(weights, x_dev, num_rows, hidden, seq_len)
        }
        _ => Err(crate::backend::RuntimeError::usage(format!(
            "matmul_dev_by_fmt: unsupported device-chain format {format:?}"
        ))),
    }
}

// `HostKv` is held under a `Mutex` on `CudaBackend`; re-export the type alias
// for the backend module.
pub(crate) type HostKvType = HostKv;

/// Device-resident quant matvec dispatch by quant format: Q4_K →
/// `launch_q4k_matvec_dev`, Q6_K → `launch_q6k_matvec_dev`. The decode twin
/// of [`matmul_dev_by_fmt`]: used by the device-resident decode FFN chain so
/// the gate/up/down projections share one upload of the input and keep their
/// outputs on the device. `None` (as `Err`) for other formats — the chain's
/// gate condition rejects them before reaching here, but the exhaustive match
/// keeps a future format loud rather than silently skipping a projection.
fn matvec_dev_by_fmt(
    runtime: &crate::backend::CudaRuntime,
    format: QuantFormat,
    weights: &[u8],
    x_dev: &cudarc::driver::CudaSlice<f32>,
    num_rows: usize,
    hidden: usize,
) -> Result<cudarc::driver::CudaSlice<f32>, crate::backend::RuntimeError> {
    match format {
        QuantFormat::Q4_K => runtime.launch_q4k_matvec_dev(weights, x_dev, num_rows, hidden),
        QuantFormat::Q6_K => runtime.launch_q6k_matvec_dev(weights, x_dev, num_rows, hidden),
        _ => Err(crate::backend::RuntimeError::usage(format!(
            "matvec_dev_by_fmt: unsupported device-chain format {format:?}"
        ))),
    }
}

/// Supported resident-hidden FFN (gate, up, down) k-quant triples. Each
/// projection is dispatched independently through [`matvec_dev_by_fmt`] and
/// produces f32 device activations, so mixed weight formats are technically
/// possible — the activation kernel consumes f32 device outputs and does not
/// care which quant format produced them, and the down projection likewise
/// consumes the f32 device activation regardless of its own weight format.
///
/// This helper intentionally accepts only model layouts that LARQL produces
/// and validates:
/// - `Q4_K / Q4_K / Q4_K` (uniform-Q4_K, the `--down-q4k` build),
/// - `Q6_K / Q6_K / Q6_K` (uniform-Q6_K),
/// - `Q4_K / Q4_K / Q6_K` (the default Q4_K_M FFN mix: gate/up Q4_K, down
///   Q6_K).
///
/// Other permutations remain rejected until backed by a real produced format,
/// a concrete need, and dedicated parity coverage. In particular
/// `Q6_K / Q6_K / Q4_K` and mixed gate/up formats (e.g. `Q4_K / Q6_K / _`)
/// are not produced by any LARQL extraction path and stay unsupported.
fn supported_resident_ffn_triple(gate: QuantFormat, up: QuantFormat, down: QuantFormat) -> bool {
    matches!(
        (gate, up, down),
        (QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q4_K)
            | (QuantFormat::Q6_K, QuantFormat::Q6_K, QuantFormat::Q6_K)
            | (QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q6_K)
    )
}

/// Number of physical kernel nodes in a captured resident-FFN graph (B3A-5).
/// The chain is: pre-norm + gate + up + activation + down + residual = 6, plus
/// a post-ffn norm (+1) when `has_post_norms`. Used by the capture-aware
/// profiling counters (`captured_kernel_nodes` at build,
/// `logical_graph_kernel_executions` at replay).
fn resident_ffn_node_count(has_post_norms: bool) -> u32 {
    if has_post_norms {
        7
    } else {
        6
    }
}

/// RAII guard that balances [`crate::backend::CudaRuntime::enter_capture`] with
/// [`crate::backend::CudaRuntime::exit_capture`] on every return path, so the
/// capture-depth suppression of `note_launch`/`note_htod`/`note_dtoh`/
/// `note_sync` (B3A review point 8) is always balanced even if `begin_capture`
/// or the launch closure fails. Construct with [`CaptureExitGuard::enter`].
struct CaptureExitGuard<'a> {
    runtime: &'a crate::backend::CudaRuntime,
}

impl CaptureExitGuard<'_> {
    /// Increment the capture depth and return a guard that decrements it on
    /// drop. While alive, the four `note_*` runtime recorders are suppressed.
    fn enter<'a>(runtime: &'a crate::backend::CudaRuntime) -> CaptureExitGuard<'a> {
        runtime.enter_capture();
        CaptureExitGuard { runtime }
    }
}

impl Drop for CaptureExitGuard<'_> {
    fn drop(&mut self) {
        self.runtime.exit_capture();
    }
}

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

    // ── D6-B/E: supported_resident_ffn_triple + eligibility host tests ──
    //
    // Pure (no device) so they run on every host, including CI without a GPU.
    // `supported_resident_ffn_triple` is the single source of truth shared by
    // `resident_hidden_layer_eligible` and `host_ffn_block_device_resident`.

    use larql_compute::QuantFormat as QF;

    #[test]
    fn supported_resident_ffn_triple_accepts_uniform_q4k() {
        assert!(supported_resident_ffn_triple(QF::Q4_K, QF::Q4_K, QF::Q4_K));
    }

    #[test]
    fn supported_resident_ffn_triple_accepts_uniform_q6k() {
        assert!(supported_resident_ffn_triple(QF::Q6_K, QF::Q6_K, QF::Q6_K));
    }

    #[test]
    fn supported_resident_ffn_triple_accepts_default_q4km_mix() {
        // The production default Q4_K_M FFN layout: gate/up Q4_K, down Q6_K.
        assert!(supported_resident_ffn_triple(QF::Q4_K, QF::Q4_K, QF::Q6_K));
    }

    #[test]
    fn supported_resident_ffn_triple_rejects_q6k_q6k_q4k() {
        // Not a produced LARQL format; stays unsupported.
        assert!(!supported_resident_ffn_triple(QF::Q6_K, QF::Q6_K, QF::Q4_K));
    }

    #[test]
    fn supported_resident_ffn_triple_rejects_mixed_gate_up() {
        // Mixed gate/up formats are not produced and not validated.
        assert!(!supported_resident_ffn_triple(QF::Q4_K, QF::Q6_K, QF::Q4_K));
        assert!(!supported_resident_ffn_triple(QF::Q4_K, QF::Q6_K, QF::Q6_K));
        assert!(!supported_resident_ffn_triple(QF::Q6_K, QF::Q4_K, QF::Q4_K));
    }

    #[test]
    fn supported_resident_ffn_triple_rejects_unsupported_quant_formats() {
        assert!(!supported_resident_ffn_triple(QF::Q4_0, QF::Q4_0, QF::Q4_0));
        assert!(!supported_resident_ffn_triple(QF::Q4_K, QF::Q4_K, QF::Q4_0));
        assert!(!supported_resident_ffn_triple(QF::BF16, QF::BF16, QF::BF16));
        assert!(!supported_resident_ffn_triple(QF::Q4_K, QF::Q4_K, QF::Q8_0));
    }

    /// Padded-down contraction (`stored_cols > inter`) stays rejected even when
    /// the format triple is supported — the resident chain assumes a contiguous
    /// `[seq, inter]` activation feeds the down matmul directly, with no
    /// device-side zero-pad step. Builds a `FullPipelineLayer` whose `down.data`
    /// length implies one extra Q4_K super-block row beyond `inter`, then
    /// asserts `down_stored_cols` returns the padded width (and thus the
    /// eligibility gate's `stored_cols == inter` check rejects it).
    #[test]
    fn down_stored_cols_rejects_padded_down_contraction() {
        // hidden=256, inter=256: a valid Q4_K down matrix is
        // 256 rows × 144 bytes/super-block × (256/256 super-blocks) = 256*144 bytes.
        // Pad to 512 stored cols (2 super-blocks/row): 256 rows × 2 × 144 bytes.
        let hidden = 256usize;
        let inter = 256usize;
        let stored_cols_padded = 512usize;
        let bytes_per_sb = super_block_bytes(QuantFormat::Q4_K).unwrap();
        let mut layer = make_minimal_layer();
        layer.down.data = down_bytes_static(hidden, stored_cols_padded, bytes_per_sb);

        let stored = down_stored_cols(&layer, hidden, inter).expect("down_stored_cols Some");
        assert_eq!(
            stored, stored_cols_padded,
            "padded down should report its padded stored width"
        );
        assert_ne!(
            stored, inter,
            "padded down must fail the `stored_cols == inter` eligibility check"
        );
    }

    /// Allocate a static byte buffer of the right length for the padded-down
    /// fixture (contents don't matter — only the length drives
    /// `down_stored_cols`).
    fn down_bytes_static(hidden: usize, stored_cols: usize, bytes_per_sb: usize) -> &'static [u8] {
        let len = hidden * (stored_cols / 256) * bytes_per_sb;
        let v = vec![0u8; len];
        Box::leak(v.into_boxed_slice())
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

//! CUDA MoE expert dispatch.
//!
//! Mirrors [`cpu::run_experts_cpu_batch`]'s per-expert decomposition but routes
//! each projection's matvec through `CudaBackend`'s native Q4_K/Q6_K kernels.
//! Unlike the Metal route (`metal.rs`), which fuses all K experts into a single
//! `run_experts_preselected_metal` dispatch and inherits the inter=704 accuracy
//! bug, this path runs **one expert at a time** through the individually
//! parity-verified `q4k_dual_matvec` / `q4k_matvec` primitives — the same
//! primitives the dense FFN pipeline already uses at cos ≥ 0.9999.
//!
//! # Why per-expert, not fused
//!
//! The Metal inter=704 bug (see `larql-compute/ROADMAP.md → "Open: Metal MoE
//! expert kernel — accuracy bug at inter=704"`) is specific to the fused
//! multi-expert dispatch pattern where `n_rows = K × inter` is fed to a single
//! `q4k_ffn_gate_up` launch. cos ≈ 0.7 vs CPU for Gemma 4 26B-A4B-it
//! (inter=704, hidden=2816, top_k=8). The per-projection CUDA kernels are
//! validated against CPU at element-equality on those same shapes, so running
//! the standard gate→up→activation→down chain per expert avoids the fused-path
//! bug class entirely. The cost is K kernel launches instead of 1; the win is
//! correctness and the persistent weight cache (experts stay resident on the
//! GPU across requests — see requirement 3).
//!
//! # Opt-in
//!
//! Default-off behind `LARQL_USE_CUDA_EXPERTS=1` while on-device throughput is
//! validated. `LARQL_DISABLE_CUDA_EXPERTS=1` hard-disables even on `cuda-experts`
//! builds. Returns `Ok(None)` when unavailable so the caller falls back to CPU.

#![cfg(feature = "cuda-experts")]

use std::time::Instant;

use larql_compute::QuantMatVec;
use larql_compute_cuda::CudaBackend;

use crate::env_flags;
use crate::error::ServerError;
use crate::state::AppState;


/// Run a layer's pre-selected experts on the CUDA GPU and return the weighted
/// sum of their outputs. Returns `Ok(None)` when CUDA is unavailable, the model
/// is not hybrid-MoE, or per-layer Q4_K weights are missing — caller should
/// fall back to the CPU path ([`run_experts_cpu_batch`]).
///
/// `h_post_attn` is the residual the streaming RPC carries (pre-norm not yet
/// applied). `expert_ids` and `expert_weights` are already client-routed (no
/// router run on the server). Returns the weighted sum WITHOUT post-experts
/// norm; the client applies post-norm once after summing across shards.
///
/// This mirrors `run_experts_cpu_batch`'s shape and pre-norm contract so the
/// route selection in `layer_batch.rs` can substitute one for the other.
pub fn run_experts_cuda_batch(
    state: &AppState,
    layer: usize,
    h_post_attn: &[f32],
    expert_ids: &[usize],
    expert_weights: &[f32],
) -> Result<Option<Vec<f32>>, ServerError> {
    // Opt-in gate — mirrors Metal's `LARQL_USE_METAL_EXPERTS` contract.
    // CUDA expert dispatch is newer than the CPU path and on-device throughput
    // is still being validated on the runner, so it stays off by default.
    // `LARQL_DISABLE_CUDA_EXPERTS=1` hard-disables even when the env opt-in is
    // set (e.g. to A/B against the CPU path on a CUDA host).
    if !env_flags::use_cuda_experts() || env_flags::disable_cuda_experts() {
        return Ok(None);
    }
    let timing_enabled = env_flags::moe_timing_enabled();
    let t_start = Instant::now();

    let model = state.model_or_err(None)?;
    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let arch = &*weights.arch;
    let t_state = t_start.elapsed();

    // Hybrid-MoE + per-layer Q4_K FFN is the only weight layout this route
    // supports. Dense FFN layers don't reach the expert endpoint, and the
    // legacy packed-BF16 layout (`!has_per_layer_ffn`) has no Q4_K bytes to
    // hand to the CUDA kernels. Bail to CPU in both cases.
    if !arch.is_hybrid_moe() || !weights.has_per_layer_ffn() {
        return Ok(None);
    }

    let hidden = h_post_attn.len();
    if hidden == 0 || expert_ids.is_empty() {
        return Ok(Some(vec![0.0f32; hidden]));
    }
    let inter = arch.moe_intermediate_size();
    let activation = larql_inference::activation_from_arch(arch);
    let inter_padded = {
        let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
        inter.div_ceil(block) * block
    };

    // Lazily initialise the backend. `OnceLock` keeps one `CudaBackend` alive
    // for the model's lifetime so the device-resident weight cache persists
    // across requests (requirement 3). `CudaBackend::new` falls back to a
    // no-runtime scaffold when no GPU is present (`allow_cpu_delegate` default
    // true) — `native_runtime_available()` distinguishes the two.
    let backend_slot = model
        .cuda_backend
        .get_or_init(|| CudaBackend::new().ok());
    let Some(backend) = backend_slot.as_ref() else {
        return Ok(None);
    };
    if !backend.native_runtime_available() {
        return Ok(None);
    }

    // Hoist pre_experts_norm onto the CPU — same optimisation as
    // `run_experts_cpu_batch`. RMS-norm is invariant of expert id and is
    // memory-bound, so the host does it once and uploads one normed vector
    // instead of K.
    let t_norm_start = Instant::now();
    let pre_norm_slice: &[f32] = arch
        .moe_pre_experts_norm_key(layer)
        .and_then(|key| weights.vectors.get(&key))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let h_norm = larql_compute::cpu::ops::moe::pre_experts_norm(
        h_post_attn,
        pre_norm_slice,
        arch.norm_weight_offset(),
        arch.norm_eps(),
    );
    let t_norm = t_norm_start.elapsed();

    // Per-expert Q4_K gate/up half-stride — same single source of truth the CPU
    // path uses (`q4k_gate_up_half`). gate_up bytes are [gate(inter,hidden) |
    // up(inter,hidden)] concatenated; `half` is one projection's byte span.
    let half = match larql_compute::cpu::ops::moe::q4k_gate_up_half(inter, hidden) {
        Some(h) => h,
        None => return Ok(None),
    };

    // K accumulators (one rayon worker each, like the CPU path), then reduce.
    // Folding in parallel keeps multi-expert latency bounded by the slowest
    // single-expert GPU dispatch rather than K×serial. Each expert is still
    // an independent dispatch (3 kernels: dual gate+up, then down) — there is
    // no fused multi-expert launch, which is the bug class that broke Metal.
    use rayon::prelude::*;
    let t_gpu_start = Instant::now();
    // Extract the ModelWeights reference for the rayon closure. `weights` is a
    // RwLockReadGuard; deref once so the closure borrows `&ModelWeights`
    // (Sync), matching the CPU path's `resolve_bytes` pattern.
    let weights_ref: &larql_models::ModelWeights = &weights;
    let out = expert_ids
        .par_iter()
        .zip(expert_weights.par_iter())
        .filter(|(_, &w)| w != 0.0)
        .fold(
            || vec![0.0f32; hidden],
            |mut acc, (&eid, &w)| {
                if let Some(h2) = run_single_expert_cuda(
                    backend,
                    &h_norm,
                    layer,
                    eid,
                    weights_ref,
                    inter,
                    inter_padded,
                    hidden,
                    half,
                    activation,
                ) {
                    for (a, &v) in acc.iter_mut().zip(h2.iter()) {
                        *a += w * v;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f32; hidden],
            |mut a, b| {
                for (x, &y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );
    let t_gpu = t_gpu_start.elapsed();

    if timing_enabled {
        eprintln!(
            "[expert_cuda_batch] layer={layer} K={} state={:.2}ms norm={:.2}ms \
             gpu={:.2}ms total={:.2}ms",
            expert_ids.len(),
            t_state.as_secs_f32() * 1000.0,
            t_norm.as_secs_f32() * 1000.0,
            t_gpu.as_secs_f32() * 1000.0,
            t_start.elapsed().as_secs_f32() * 1000.0,
        );
    }

    Ok(Some(out))
}

/// Run one expert's gated FFN on the CUDA device.
///
/// Decomposition (mirrors the CPU `run_single_expert_q4k_q8k_into` chain but
/// via `QuantMatVec` so the device kernel runs):
///
/// 1. `q4k_dual_matvec(gate, up, h_norm)` → `(gate_out[inter], up_out[inter])`
///    in one launch (shared input upload + shared weight cache lookups).
/// 2. host-side activation `act[j] = gelu_tanh(gate_out[j]) * up_out[j]`,
///    zero-padded to `inter_padded` for the down matvec.
/// 3. `q4k_matvec(down, act)` → `out[hidden]`.
///
/// Returns `None` if the weight bytes for this `(layer, expert)` aren't present
/// (the fold closure skips it — same as the CPU path's `resolve_bytes` miss).
#[allow(clippy::too_many_arguments)]
fn run_single_expert_cuda(
    backend: &CudaBackend,
    h_norm: &[f32],
    layer: usize,
    eid: usize,
    weights: &larql_models::ModelWeights,
    inter: usize,
    inter_padded: usize,
    hidden: usize,
    half: usize,
    activation: larql_compute::Activation,
) -> Option<Vec<f32>> {
    let (gate_up_bytes, down_bytes) = weights.get_layer_entry_bytes(layer, eid)?;
    let gate_bytes = &gate_up_bytes[..half];
    let up_bytes = &gate_up_bytes[half..2 * half];

    // (1) Dual gate+up matvec. One launch, one input upload — the CUDA runtime
    // shares the device buffer for x across both projections.
    let (gate_out, up_out) = backend
        .q4k_dual_matvec(gate_bytes, up_bytes, h_norm, inter, hidden)
        .or_else(|| {
            // Fallback to two sequential matvecs if the dual kernel is
            // unavailable on this runtime (older device, disabled feature).
            let g = backend.q4k_matvec(gate_bytes, h_norm, inter, hidden)?;
            let u = backend.q4k_matvec(up_bytes, h_norm, inter, hidden)?;
            Some((g, u))
        })?;

    // (2) Activation on the host. The elementwise op is cheap and bandwidth-
    // bound; doing it on-device would add a launch without saving a transfer
    // (we read gate_out/up_out back either way for the down input). Build at
    // inter_padded so the down matvec sees zeros in the Q4_K padding tail.
    let mut act = vec![0.0f32; inter_padded];
    for j in 0..inter {
        let g = gate_out[j];
        let u = up_out[j];
        act[j] = match activation {
            larql_compute::Activation::GeluTanh => gelu_tanh(g) * u,
            _ => silu(g) * u,
        };
    }

    // (3) Down projection. down is Q4_K [hidden, inter_padded].
    backend.q4k_matvec(down_bytes, &act, hidden, inter_padded)
}

/// SiLU activation: x * sigmoid(x). Mirrors `larql_compute::cpu::ops::moe::math::silu`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU with tanh approximation (Gemma expert FFN activation). Mirrors
/// `larql_compute::cpu::ops::moe::math::gelu_tanh`.
#[inline]
fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

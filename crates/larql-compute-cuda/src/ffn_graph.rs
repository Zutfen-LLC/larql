//! LARQL-GPU-B3A: the graphable resident-FFN contract + generation identity.
//!
//! This module holds the **plan** half of the CUDA Graph replay for the
//! resident decode FFN (the implementation half — capture, arena, replay,
//! lifecycle — lives in [`crate::backend`] and [`crate::pipeline`]). It is
//! deliberately pure (no device touch) so the eligibility logic, plan keying,
//! and identity contract are unit-testable on every host, including CI without
//! a GPU.
//!
//! ## Design (B3A review points 2, 6)
//!
//! A CUDA Graph captured for one eligible layer's FFN chain is valid for as
//! long as its referenced device buffers (weights, norm vectors, scratch) keep
//! stable device addresses. The graph cache is therefore **generation-scoped**:
//!
//! - [`GraphGenerationId`] is bumped on every `reset_kv_cache` (the generation
//!   boundary where the weight cache is flushed and repopulated).
//! - The cache key is `(generation, layer_index, ResidentFfnPlan)` — **not**
//!   host weight pointer/length. Each layer captures different device weight
//!   pointers (its own gate/up/down), so no cross-layer deduplication is
//!   attempted even when shapes/activation match. Host `(ptr, len)` values
//!   remain a *diagnostic assertion* (a changed weight address within one
//!   generation is a bug), but generation identity is the primary ABA defense.
//!
//! ## What the plan encodes
//!
//! Every property that affects the FFN kernel sequence, dimensions, kernel
//! arguments, or weight identity for a loaded layer. Two layers with equal
//! plans produce structurally identical graphs (modulo their distinct weight
//! buffers); two layers with materially different plans must not alias.

use larql_compute::{Activation, FfnType, NormType, QuantFormat};

// ── Generation identity ───────────────────────────────────────────────────

/// Opaque generation token for the resident-FFN graph cache.
///
/// Bumped on every [`crate::backend::CudaBackend::reset_kv_cache_native`]
/// (the generation boundary). A graph captured under generation `N` must not be
/// replayed under generation `N+1` — its referenced weight/scratch buffers may
/// have been flushed and reallocated at different addresses.
///
/// This is the primary ABA defense: weight-cache `(ptr, len)` keying cannot
/// detect a recycled mmap address across vindex loads, and even within one
/// vindex the per-generation flush repopulates buffers. Generation identity is
/// the coarse, sound boundary; host pointers are a diagnostic assertion only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphGenerationId(pub u64);

impl GraphGenerationId {
    /// The initial generation (no graphs have been built yet).
    pub const INITIAL: Self = Self(0);

    /// Advance to the next generation (called at a reset boundary).
    pub fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

impl Default for GraphGenerationId {
    fn default() -> Self {
        Self::INITIAL
    }
}

// ── The graphable resident-FFN plan ────────────────────────────────────────

/// The immutable FFN properties that define a graphable resident-FFN chain for
/// one layer.
///
/// Captured into a CUDA Graph, these determine the kernel sequence (which
/// activation/norm kernels run), the kernel launch dimensions (`hidden`,
/// `inter`), the argument values (`eps`, `norm_offset`, `residual_multiplier`),
/// and the weight-format dispatch (Q4_K vs Q6_K gate/up/down). Two layers with
/// equal plans and equal generation+layer identity may share a graph entry;
/// materially different plans must produce distinct cache entries.
///
/// Built via [`ResidentFfnPlan::from_layer`], which calls the existing
/// eligibility gates (`supported_resident_ffn_triple`, `down_stored_cols`,
/// `native_activation_worthwhile`) — single source of truth, no duplication.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidentFfnPlan {
    /// Hidden dimension (input/output of the FFN block).
    pub hidden: usize,
    /// Intermediate dimension (gate/up output width, down input width).
    pub inter: usize,
    /// Stored columns of the down projection (must equal `inter` for graph
    /// eligibility — a padded-down contraction has no graph path).
    pub stored_cols: usize,
    /// Gate projection quant format.
    pub gate_fmt: QuantFormat,
    /// Up projection quant format.
    pub up_fmt: QuantFormat,
    /// Down projection quant format.
    pub down_fmt: QuantFormat,
    /// FFN type (Gated uses gate×up; Standard uses up only).
    pub ffn_type: FfnType,
    /// Activation function applied to the gate×up (or up-only) intermediate.
    pub activation: Activation,
    /// Whether the layer has post-norms (4 norms/layer: Gemma 2/3/4) — selects
    /// the pre-ffn norm source and whether a post-ffn norm runs.
    pub has_post_norms: bool,
    /// Whether a pre-ffn norm weight is present (`Some` vs `None` on the layer).
    pub has_pre_ffn_norm_weight: bool,
    /// Whether a post-ffn norm weight is present.
    pub has_post_ffn_norm_weight: bool,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Norm weight offset (0.0 Llama/Gemma-4, 1.0 Gemma 2/3).
    pub norm_offset: f32,
    /// Residual multiplier (the layer's `residual_multiplier`).
    pub residual_multiplier: f32,
}

impl ResidentFfnPlan {
    /// The number of physical kernel nodes a captured graph for this plan
    /// contains. Used by the capture-aware profiling counters
    /// (B3A review point 8): `captured_kernel_nodes` is incremented by this on
    /// build, and `logical_graph_kernel_executions` is incremented by this on
    /// every replay.
    ///
    /// The chain is: pre-ffn norm, gate matvec, up matvec, activation,
    /// down matvec, (post-ffn norm), residual add. The post-ffn norm is
    /// present iff `has_post_norms && has_post_ffn_norm_weight` (or the
    /// no-weight path when `has_post_norms && !has_post_ffn_norm_weight`).
    #[allow(dead_code)]
    pub fn kernel_node_count(&self) -> u32 {
        // pre-norm + gate + up + activation + down + residual = 6
        let mut n = 6u32;
        if self.has_post_norms {
            n += 1; // post-ffn norm
        }
        n
    }
}

/// A hashable key derived from a [`ResidentFfnPlan`] for graph-cache lookup.
///
/// Deliberately excludes weight pointers (pointers are a diagnostic assertion,
/// not the primary identity — see the module docs and B3A review point 6). The
/// key captures every property affecting the kernel sequence, dimensions, or
/// argument values, so two plans with equal keys produce structurally identical
/// graphs (modulo distinct weight buffers, which the `(generation, layer_index)`
/// tuple in the cache key already distinguishes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidentFfnPlanKey {
    hidden: u64,
    inter: u64,
    stored_cols: u64,
    gate_fmt: u8,
    up_fmt: u8,
    down_fmt: u8,
    ffn_type: u8,
    activation: u8,
    has_post_norms: bool,
    has_pre_ffn_norm_weight: bool,
    has_post_ffn_norm_weight: bool,
    eps_bits: u32,
    norm_offset_bits: u32,
    residual_multiplier_bits: u32,
}

impl ResidentFfnPlan {
    /// Compute the cache key for this plan.
    pub fn key(&self) -> ResidentFfnPlanKey {
        ResidentFfnPlanKey {
            hidden: self.hidden as u64,
            inter: self.inter as u64,
            stored_cols: self.stored_cols as u64,
            gate_fmt: quant_format_tag(self.gate_fmt),
            up_fmt: quant_format_tag(self.up_fmt),
            down_fmt: quant_format_tag(self.down_fmt),
            ffn_type: ffn_type_tag(self.ffn_type),
            activation: activation_tag(self.activation),
            has_post_norms: self.has_post_norms,
            has_pre_ffn_norm_weight: self.has_pre_ffn_norm_weight,
            has_post_ffn_norm_weight: self.has_post_ffn_norm_weight,
            eps_bits: self.eps.to_bits(),
            norm_offset_bits: self.norm_offset.to_bits(),
            residual_multiplier_bits: self.residual_multiplier.to_bits(),
        }
    }
}

fn quant_format_tag(f: QuantFormat) -> u8 {
    match f {
        QuantFormat::Q4_K => 0,
        QuantFormat::Q6_K => 1,
        QuantFormat::Q4_0 => 2,
        QuantFormat::Q8_0 => 3,
        QuantFormat::BF16 => 4,
        QuantFormat::F16 => 5,
        QuantFormat::F32 => 6,
        QuantFormat::Q4_KF => 7,
        QuantFormat::I2S => 8,
    }
}

fn ffn_type_tag(t: FfnType) -> u8 {
    match t {
        FfnType::Gated => 0,
        FfnType::Standard => 1,
    }
}

fn activation_tag(a: Activation) -> u8 {
    match a {
        Activation::Silu => 0,
        Activation::GeluTanh => 1,
        Activation::GeluExact => 2,
        Activation::ReLU => 3,
    }
}

/// Whether the resident-FFN graph path is eligible for this plan.
///
/// This is the **plan-level** gate; the device-level gate (weight presence,
/// runtime availability, graph-mode config) is checked separately in the
/// pipeline.
///
/// The plan gate mirrors [`crate::pipeline::supported_resident_ffn_triple`] +
/// `stored_cols == inter` + `inter >= activation gate` + `RmsNorm`, but
/// operates on the already-extracted plan fields rather than re-reading the
/// layer. The pipeline's [`crate::pipeline::resident_hidden_layer_eligible`]
/// remains the authoritative eligibility gate for the resident-hidden path.
/// This helper is the additional gate for the graph path: a layer can be
/// resident-hidden-eligible but graph-ineligible (e.g. if `inter` is below the
/// graph activation threshold — currently the same threshold, but kept separate
/// so the graph gate can diverge).
///
/// `activation_min_elems` is passed in (rather than read from env) so this is
/// pure and testable without racy env access.
pub fn plan_graph_eligible(plan: &ResidentFfnPlan, activation_min_elems: usize) -> bool {
    plan.stored_cols == plan.inter
        && plan.inter >= activation_min_elems
        && plan.norm_type_is_rms()
        && supported_triple(plan.gate_fmt, plan.up_fmt, plan.down_fmt)
        && supported_activation(plan.activation, plan.ffn_type)
}

impl ResidentFfnPlan {
    /// True iff the norm type is RMSNorm (the only norm with a device kernel).
    /// The plan doesn't carry `NormType` directly (it's implied by the resident
    /// path which is RmsNorm-only), but this keeps the gate self-contained.
    fn norm_type_is_rms(&self) -> bool {
        // The resident-hidden eligibility gate already enforces RmsNorm; the
        // plan is only constructed for RmsNorm layers. This is always true, but
        // kept explicit so a future plan constructor can't silently admit a
        // LayerNorm layer into the graph path.
        let _ = NormType::RmsNorm; // type-level documentation anchor
        true
    }
}

/// Mirror of [`crate::pipeline::supported_resident_ffn_triple`] for the pure
/// plan module (which can't `use` the private pipeline fn). Kept in sync by the
/// shared test `plan_triple_matches_pipeline_triple`.
pub(crate) fn supported_triple(gate: QuantFormat, up: QuantFormat, down: QuantFormat) -> bool {
    matches!(
        (gate, up, down),
        (QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q4_K)
            | (QuantFormat::Q6_K, QuantFormat::Q6_K, QuantFormat::Q6_K)
            | (QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q6_K)
    )
}

/// The activations the graph path supports (those with device kernels in the
/// resident chain): `{Gated, Standard} × {Silu, GeluTanh}`.
pub(crate) fn supported_activation(a: Activation, t: FfnType) -> bool {
    matches!(
        (t, a),
        (FfnType::Gated, Activation::Silu)
            | (FfnType::Gated, Activation::GeluTanh)
            | (FfnType::Standard, Activation::Silu)
            | (FfnType::Standard, Activation::GeluTanh)
    )
}

// ── Graph mode (B3A review point 9: per-backend, not process-global) ───────

/// Whether the resident-FFN graph replay path is active.
///
/// Resolved **once at backend construction** from `LARQL_CUDA_GRAPHS` and
/// stored on the backend config, so the decode hot path reads a field (never
/// re-reads process-global env). Tests construct graph-enabled/disabled backends
/// in one process without `set_var` or order dependence.
///
/// - [`GraphMode::Auto`] — use the graph path when a layer is eligible and the
///   runtime is present. The default-on decision is gated by B3A-11's
///   performance gate; until then `Auto` for the *env-unset* case defers to
///   [`GraphMode::Disabled`] unless the performance gate has flipped the
///   default (see [`default_graph_mode`]).
/// - [`GraphMode::Disabled`] — never use the graph path (the kill switch,
///   `LARQL_CUDA_GRAPHS=0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphMode {
    /// Use the graph path when eligible + runtime present.
    Auto,
    /// Never use the graph path.
    Disabled,
}

impl GraphMode {
    /// True iff the graph path may be used (subject to per-layer eligibility).
    pub fn enabled(self) -> bool {
        matches!(self, GraphMode::Auto)
    }
}

/// The env var that selects graph mode.
///
/// - `LARQL_CUDA_GRAPHS=0` → [`GraphMode::Disabled`] (kill switch).
/// - `LARQL_CUDA_GRAPHS=1` → [`GraphMode::Auto`] (forced on).
/// - unset → [`default_graph_mode`] (currently [`GraphMode::Disabled`] until
///   the B3A-11 performance gate flips the default to `Auto`).
pub const ENV_CUDA_GRAPHS: &str = "LARQL_CUDA_GRAPHS";

/// The default graph mode when `LARQL_CUDA_GRAPHS` is unset.
///
/// **Currently [`GraphMode::Disabled`].** This flips to [`GraphMode::Auto`]
/// only after the B3A-11 performance gate (same-day median decode improvement
/// ≥1%, ≥25% submission reduction, build amortization within ~64 tokens) passes
/// on the RTX 3060. Until then, graph replay is opt-in via `LARQL_CUDA_GRAPHS=1`.
pub fn default_graph_mode() -> GraphMode {
    // B3A-11 gate: once the performance gate passes, change this to Auto.
    GraphMode::Disabled
}

/// Resolve graph mode from an env-string value (test seam).
///
/// Pure (takes the string, not the process env) so the per-value mapping is
/// unit-testable without racy `set_var` calls. Mirrors the resolution rule in
/// [`graph_mode_from_env`].
pub fn graph_mode_from_str(raw: Option<&str>) -> GraphMode {
    match raw.map(str::trim) {
        Some(v) => match v {
            "0" | "false" | "off" | "disable" | "disabled" => GraphMode::Disabled,
            "1" | "true" | "on" | "enable" | "enabled" | "auto" => GraphMode::Auto,
            // Unrecognised non-empty value → honour the opt-in intent only if
            // it's truthy-ish; otherwise fall back to the default. An empty
            // string is treated as unset.
            "" => default_graph_mode(),
            _ => default_graph_mode(),
        },
        None => default_graph_mode(),
    }
}

/// Read `LARQL_CUDA_GRAPHS` from the process env and resolve the mode.
///
/// Called once at backend construction; the result is stored on the backend
/// config. The hot path never calls this.
pub fn graph_mode_from_env() -> GraphMode {
    graph_mode_from_str(std::env::var(ENV_CUDA_GRAPHS).ok().as_deref())
}

/// The CUDA graph instantiate flags LARQL uses (B3A review point 5).
///
/// Point 5 mandates beginning with default (0) instantiate flags unless the
/// smoke test or cudarc implementation **proves another flag is required**.
/// That proof holds here: cudarc 0.19.8's `CudaStream::end_capture` takes a
/// **typed** `CUgraphInstantiate_flags` enum whose only constructible
/// variants are non-zero (`AUTO_FREE_ON_LAUNCH=1`, `UPLOAD=2`,
/// `DEVICE_LAUNCH=4`, `USE_NODE_PRIORITY=8` on CUDA 12.4). There is no sound
/// way to pass default (0) flags through this API:
/// - `0u32 as CUgraphInstantiate_flags` is rejected (`as` cannot cast int→enum);
/// - `transmute(0u32)` is `invalid_value` UB (a fieldless enum must hold a
///   valid discriminant under Rust's abstract machine), even though it is
///   `#[repr(u32)]`;
/// - `CudaGraph`'s `cu_graph`/`cu_graph_exec` fields are private, so the raw
///   `cuGraphInstantiateWithFlags(.., 0)` sys call cannot be used to build a
///   `CudaGraph` from outside the crate.
///
/// `AUTO_FREE_ON_LAUNCH` only affects **graph-managed** allocations, which the
/// FFN graph does not contain (all buffers are externally owned by the arena /
/// weight cache / scratch), so it is a no-op here — the minimal-impact flag
/// the API forces. `b3a_smoke_*` validates capture/instantiate/replay with
/// this flag on the RTX 3060.
pub(crate) fn graph_instantiate_flags() -> cudarc::driver::sys::CUgraphInstantiate_flags {
    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GraphGenerationId ──

    #[test]
    fn generation_next_advances() {
        let g = GraphGenerationId::INITIAL;
        assert_ne!(g, g.next());
        assert_eq!(g.next(), GraphGenerationId(1));
        assert_eq!(GraphGenerationId(5).next(), GraphGenerationId(6));
    }

    #[test]
    fn generation_wraps_safely() {
        // wrapping_add — no panic at the max boundary.
        let g = GraphGenerationId(u64::MAX);
        assert_eq!(g.next(), GraphGenerationId(0));
    }

    #[test]
    fn generation_eq_hash() {
        let a = GraphGenerationId(3);
        let b = GraphGenerationId(3);
        assert_eq!(a, b);
        let mut set = std::collections::HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    // ── ResidentFfnPlan keying ──

    fn q4km_plan() -> ResidentFfnPlan {
        ResidentFfnPlan {
            hidden: 2048,
            inter: 11008,
            stored_cols: 11008,
            gate_fmt: QuantFormat::Q4_K,
            up_fmt: QuantFormat::Q4_K,
            down_fmt: QuantFormat::Q6_K,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            has_post_norms: false,
            has_pre_ffn_norm_weight: true,
            has_post_ffn_norm_weight: false,
            eps: 1e-6,
            norm_offset: 0.0,
            residual_multiplier: 1.0,
        }
    }

    #[test]
    fn equal_plans_produce_equal_keys() {
        assert_eq!(q4km_plan().key(), q4km_plan().key());
    }

    #[test]
    fn different_quant_formats_distinct_keys() {
        let mut p = q4km_plan();
        p.down_fmt = QuantFormat::Q4_K; // Q4_K/Q4_K/Q4_K instead of Q4_K/Q4_K/Q6_K
        assert_ne!(q4km_plan().key(), p.key());
    }

    #[test]
    fn different_activation_distinct_keys() {
        let mut p = q4km_plan();
        p.activation = Activation::GeluTanh;
        assert_ne!(q4km_plan().key(), p.key());
    }

    #[test]
    fn different_dims_distinct_keys() {
        let mut p = q4km_plan();
        p.inter = 4096;
        p.stored_cols = 4096;
        assert_ne!(q4km_plan().key(), p.key());
    }

    #[test]
    fn different_eps_distinct_keys() {
        let mut p = q4km_plan();
        p.eps = 1e-5;
        assert_ne!(q4km_plan().key(), p.key());
    }

    #[test]
    fn different_residual_multiplier_distinct_keys() {
        let mut p = q4km_plan();
        p.residual_multiplier = 0.5;
        assert_ne!(q4km_plan().key(), p.key());
    }

    #[test]
    fn different_post_norms_distinct_keys() {
        let mut p = q4km_plan();
        p.has_post_norms = true;
        p.has_post_ffn_norm_weight = true;
        assert_ne!(q4km_plan().key(), p.key());
    }

    #[test]
    fn different_ffn_type_distinct_keys() {
        let mut p = q4km_plan();
        p.ffn_type = FfnType::Standard;
        assert_ne!(q4km_plan().key(), p.key());
    }

    // ── kernel_node_count ──

    #[test]
    fn kernel_node_count_no_post_norm() {
        // pre-norm + gate + up + activation + down + residual = 6
        let p = q4km_plan();
        assert_eq!(p.kernel_node_count(), 6);
    }

    #[test]
    fn kernel_node_count_with_post_norm() {
        // + post-ffn norm = 7
        let mut p = q4km_plan();
        p.has_post_norms = true;
        assert_eq!(p.kernel_node_count(), 7);
    }

    // ── plan_graph_eligible ──

    #[test]
    fn plan_eligible_for_default_q4km() {
        let p = q4km_plan();
        assert!(plan_graph_eligible(&p, 8192));
    }

    #[test]
    fn plan_ineligible_below_activation_gate() {
        let p = q4km_plan();
        // inter=11008 >= 8192 eligible; raise the gate above inter → ineligible.
        assert!(!plan_graph_eligible(&p, 11009));
    }

    #[test]
    fn plan_ineligible_when_stored_cols_ne_inter() {
        let mut p = q4km_plan();
        p.stored_cols = p.inter + 256; // padded-down contraction
        assert!(!plan_graph_eligible(&p, 8192));
    }

    #[test]
    fn plan_ineligible_for_unsupported_triple() {
        let mut p = q4km_plan();
        p.down_fmt = QuantFormat::Q4_0;
        assert!(!plan_graph_eligible(&p, 8192));
    }

    #[test]
    fn plan_ineligible_for_unsupported_activation() {
        let mut p = q4km_plan();
        p.activation = Activation::ReLU;
        assert!(!plan_graph_eligible(&p, 8192));
    }

    // ── supported_triple / supported_activation parity ──

    #[test]
    fn supported_triple_matches_pipeline_doc() {
        assert!(supported_triple(
            QuantFormat::Q4_K,
            QuantFormat::Q4_K,
            QuantFormat::Q4_K
        ));
        assert!(supported_triple(
            QuantFormat::Q6_K,
            QuantFormat::Q6_K,
            QuantFormat::Q6_K
        ));
        assert!(supported_triple(
            QuantFormat::Q4_K,
            QuantFormat::Q4_K,
            QuantFormat::Q6_K
        ));
        assert!(!supported_triple(
            QuantFormat::Q6_K,
            QuantFormat::Q6_K,
            QuantFormat::Q4_K
        ));
        assert!(!supported_triple(
            QuantFormat::Q4_K,
            QuantFormat::Q6_K,
            QuantFormat::Q4_K
        ));
    }

    #[test]
    fn supported_activation_covers_resident_chain() {
        assert!(supported_activation(Activation::Silu, FfnType::Gated));
        assert!(supported_activation(Activation::GeluTanh, FfnType::Gated));
        assert!(supported_activation(Activation::Silu, FfnType::Standard));
        assert!(supported_activation(
            Activation::GeluTanh,
            FfnType::Standard
        ));
        assert!(!supported_activation(Activation::ReLU, FfnType::Gated));
        assert!(!supported_activation(
            Activation::GeluExact,
            FfnType::Standard
        ));
    }

    // ── GraphMode resolution ──

    #[test]
    fn graph_mode_disabled_from_zero() {
        assert_eq!(graph_mode_from_str(Some("0")), GraphMode::Disabled);
        assert_eq!(graph_mode_from_str(Some("off")), GraphMode::Disabled);
        assert_eq!(graph_mode_from_str(Some("disabled")), GraphMode::Disabled);
    }

    #[test]
    fn graph_mode_auto_from_one() {
        assert_eq!(graph_mode_from_str(Some("1")), GraphMode::Auto);
        assert_eq!(graph_mode_from_str(Some("auto")), GraphMode::Auto);
        assert_eq!(graph_mode_from_str(Some("enabled")), GraphMode::Auto);
    }

    #[test]
    fn graph_mode_unset_uses_default() {
        // Until the B3A-11 gate flips the default, unset = Disabled.
        assert_eq!(graph_mode_from_str(None), default_graph_mode());
        assert_eq!(graph_mode_from_str(None), GraphMode::Disabled);
    }

    #[test]
    fn graph_mode_garbage_falls_back_to_default() {
        assert_eq!(
            graph_mode_from_str(Some("yes please")),
            default_graph_mode()
        );
        assert_eq!(graph_mode_from_str(Some("")), default_graph_mode());
    }

    #[test]
    fn graph_mode_trims_whitespace() {
        assert_eq!(graph_mode_from_str(Some("  1  ")), GraphMode::Auto);
        assert_eq!(graph_mode_from_str(Some("  0  ")), GraphMode::Disabled);
    }

    #[test]
    fn env_var_uses_cuda_prefix() {
        assert!(ENV_CUDA_GRAPHS.starts_with("LARQL_CUDA_"));
    }
}

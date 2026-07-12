//! larql-inference — full transformer forward pass + mechanistic-interp surface.
//!
//! Two roles:
//!
//! - **Inference**: prefill, decode, sampling, KV engines, Metal GPU path,
//!   chat templates. `predict`, `generate`, `predict_with_temperature`.
//! - **Mechanistic interp**: programmatic hooks at every layer boundary,
//!   logit lens, embedding-neighbor lookups, activation patching, KV-cache
//!   surgery. The primitives lazarus-style MCP servers build on.
//!
//! ## Mechanistic interp surface
//!
//! Five callbacks fire inside [`forward::trace_forward_full_hooked`]; two of
//! them take `&mut Array2<f32>` so a hook can mutate the residual in place:
//!
//! ```text
//! pre_layer  →  attention  →  on_post_attention(&mut h)  →  FFN  →  on_post_layer(&mut h)
//!                                  ^                              ^
//!                                  └─ patching, pre-FFN steer ────┘
//! ```
//!
//! Built-in hooks live in [`forward::hooks`]:
//! [`RecordHook`](forward::RecordHook) (capture),
//! [`ZeroAblateHook`](forward::ZeroAblateHook) (zero-out),
//! [`SteerHook`](forward::SteerHook) (`x + α·v`),
//! [`CompositeHook`](forward::CompositeHook) (compose multiple). Implement
//! [`forward::LayerHook`] for custom transforms.
//!
//! Sibling primitives:
//!
//! - [`forward::lens`] — full logit lens, `track_token`, `track_race`.
//! - [`forward::vocab_proj`] — `W_E` / `W_U` access, `embedding_neighbors`,
//!   raw `project_through_unembed` (DLA without final norm).
//! - [`forward::patching`] — donor/recipient activation patching built on
//!   the hook surface.
//! - `larql_kv::KvCache` — `get_layer` / `set_layer` /
//!   `clone_layer_position_range` for KV-cache surgery (e.g. lazarus's
//!   `prefill_inject` and `kv_inject_test`). Lives in `larql-kv` since
//!   2026-05-16 — it is engine state, not substrate.
//!
//! See `examples/mech_interp_demo.rs` for an end-to-end walkthrough on
//! synthetic weights (no vindex required).

#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "macos",
    target_os = "windows"
))]
extern crate blas_src;

pub mod async_compute_backend;
pub mod attention;
pub mod capture;
pub mod chat;
pub mod decode_stages;
pub mod error;
pub mod experts;
pub mod ffn;
pub mod ffn_policy;
pub mod forward;
pub mod forward_overrides;
pub mod kv_dispatch;
pub mod kv_engine;
pub mod layer_executor;
pub mod layer_graph;
pub mod model;
pub mod prompt;
pub mod prompt_render;
pub mod residual;
pub mod residual_diff;
pub mod ternary;
pub mod test_utils;
pub mod tokenizer;
pub mod trace;
pub mod vindex;

// Re-export dependencies for downstream crates.
pub use larql_models;
pub use larql_vindex;
pub use ndarray;
pub use safetensors;
pub use tokenizers;

// Backend re-exports — only the names with external consumers via
// `larql_inference::*`. Callers wanting other compute types should
// `use larql_compute::...` directly.
pub use larql_compute::cpu::ops::moe::{run_single_expert, run_single_expert_with_norm};
pub use larql_compute::QuantFormat;
pub use larql_compute::{cpu_backend, default_backend, ComputeBackend};

/// Process-wide override for the `Auto` backend factories. When set to an
/// explicit backend name (`cpu|metal|cuda|vulkan`), the `Auto` arms of
/// [`try_compute_backend`] / [`try_engine_backend`] /
/// [`try_async_engine_backend`] resolve to exactly that backend and fail
/// loudly if it is unavailable — instead of walking the platform Auto
/// order. Unset or `auto` preserves the platform default order. Same
/// vocabulary as the CLI `--backend` flag ([`ComputeBackendKind::parse`]).
///
/// Read through `larql_compute`'s env helper so the same thread-local
/// test override applies (no `std::env::set_var` races in tests).
pub const ENV_LARQL_BACKEND: &str = "LARQL_BACKEND";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeBackendKind {
    Auto,
    Cpu,
    Metal,
    Cuda,
    Vulkan,
}

impl ComputeBackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
        }
    }

    /// Parse a backend name from a raw string (case-insensitive, whitespace
    /// trimmed). The single source of truth for the backend vocabulary —
    /// shared by the CLI `--backend` parser and the [`ENV_LARQL_BACKEND`]
    /// env override so the two can't drift.
    ///
    /// Accepts: `auto`, `cpu`, `metal`, `cuda`, `vulkan`.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            "vulkan" => Ok(Self::Vulkan),
            other => Err(format!(
                "unknown backend `{other}` (expected one of: auto, cpu, metal, cuda, vulkan)"
            )),
        }
    }
}

impl std::fmt::Display for ComputeBackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub enum BackendSelectionError {
    /// The requested backend exists in the vocabulary but isn't available
    /// in this build / on this host (e.g. `--backend cuda` without the
    /// `cuda` feature, or Metal off-macOS).
    Unavailable {
        kind: ComputeBackendKind,
        reason: String,
    },
    /// A backend name couldn't be parsed — currently only raised by the
    /// [`ENV_LARQL_BACKEND`] override when it holds a value outside the
    /// [`ComputeBackendKind::parse`] vocabulary.
    InvalidEnv { raw: String, reason: String },
}

impl std::fmt::Display for BackendSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable { kind, reason } => {
                write!(f, "backend `{kind}` unavailable: {reason}")
            }
            Self::InvalidEnv { raw, reason } => {
                write!(f, "invalid {ENV_LARQL_BACKEND}=`{raw}`: {reason}")
            }
        }
    }
}

impl std::error::Error for BackendSelectionError {}

/// CPU backend boxed as `EngineBackend`. Use when you need to construct
/// a backend that satisfies the `EngineBackend` umbrella (so engines
/// can dispatch through both `ComputeBackend` and `KvDispatch`).
/// `cpu_backend()` returns `Box<dyn ComputeBackend>` and can't be
/// upcast to `Box<dyn EngineBackend>` at the trait-object level — use
/// this factory instead.
pub fn cpu_engine_backend() -> Box<dyn EngineBackend> {
    Box::new(larql_compute::CpuBackend)
}

/// CPU backend boxed as `AsyncComputeBackend`. Use when constructing
/// engines via the `with_async_backend` opt-in (e.g.
/// `StandardEngine::with_async_backend`). Output is bit-identical to
/// the synchronous `cpu_engine_backend()` path — CPU is a degenerate
/// `Ready*`-wrapped impl (A2 parity reference).
pub fn cpu_async_engine_backend() -> Box<dyn AsyncComputeBackend> {
    Box::new(larql_compute::CpuBackend)
}

fn auto_backend_order() -> &'static [ComputeBackendKind] {
    #[cfg(target_os = "macos")]
    {
        &[
            ComputeBackendKind::Metal,
            ComputeBackendKind::Vulkan,
            ComputeBackendKind::Cpu,
        ]
    }
    #[cfg(not(target_os = "macos"))]
    {
        &[
            ComputeBackendKind::Cuda,
            ComputeBackendKind::Vulkan,
            ComputeBackendKind::Cpu,
        ]
    }
}

/// What `Auto` should resolve to, honouring the [`ENV_LARQL_BACKEND`]
/// override.
///
/// Returns:
/// - `Ok(None)` — no override (unset or `auto`); the caller walks the
///   platform Auto order with its normal CPU fallback.
/// - `Ok(Some(kind))` — explicit override; the caller MUST try exactly
///   that backend and NOT fall back, so an unavailable request fails
///   loudly (matching the CLI `--backend` contract).
/// - `Err(InvalidEnv)` — override present but unparseable.
///
/// Read through `larql_compute`'s env helper so the thread-local test
/// override applies (no `std::env::set_var` races under parallel tests).
fn auto_override_kind() -> Result<Option<ComputeBackendKind>, BackendSelectionError> {
    match larql_compute::options::env_value(ENV_LARQL_BACKEND) {
        None => Ok(None),
        Some(raw) => match ComputeBackendKind::parse(&raw) {
            Ok(ComputeBackendKind::Auto) => Ok(None),
            Ok(kind) => Ok(Some(kind)),
            Err(reason) => Err(BackendSelectionError::InvalidEnv { raw, reason }),
        },
    }
}

fn try_compute_backend(
    kind: ComputeBackendKind,
) -> Result<Box<dyn larql_compute::ComputeBackend>, BackendSelectionError> {
    match kind {
        ComputeBackendKind::Auto => {
            // Explicit LARQL_BACKEND override wins and must NOT fall back,
            // so an unavailable request fails loudly (matches `--backend`).
            if let Some(kind) = auto_override_kind()? {
                return try_compute_backend(kind);
            }
            for candidate in auto_backend_order() {
                if let Ok(backend) = try_compute_backend(*candidate) {
                    return Ok(backend);
                }
            }
            Ok(larql_compute::cpu_backend())
        }
        ComputeBackendKind::Cpu => Ok(larql_compute::cpu_backend()),
        ComputeBackendKind::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                if let Some(metal) = larql_compute_metal::metal_backend() {
                    return Ok(Box::new(metal));
                }
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "no Metal device or pipeline initialization failed".to_string(),
                })
            }
            #[cfg(all(feature = "metal", not(target_os = "macos")))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "Metal backend is only available on macOS".to_string(),
                })
            }
            #[cfg(not(feature = "metal"))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "crate built without the `metal` feature".to_string(),
                })
            }
        }
        ComputeBackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                larql_compute_cuda::cuda_backend()
                    .map(|backend| Box::new(backend) as Box<dyn larql_compute::ComputeBackend>)
                    .map_err(|err| BackendSelectionError::Unavailable {
                        kind,
                        reason: err.to_string(),
                    })
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "crate built without the `cuda` feature".to_string(),
                })
            }
        }
        ComputeBackendKind::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                larql_compute_vulkan::vulkan_backend()
                    .map(|backend| Box::new(backend) as Box<dyn larql_compute::ComputeBackend>)
                    .map_err(|err| BackendSelectionError::Unavailable {
                        kind,
                        reason: err.to_string(),
                    })
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "crate built without the `vulkan` feature".to_string(),
                })
            }
        }
    }
}

fn try_engine_backend(
    kind: ComputeBackendKind,
) -> Result<Box<dyn EngineBackend>, BackendSelectionError> {
    match kind {
        ComputeBackendKind::Auto => {
            if let Some(kind) = auto_override_kind()? {
                return try_engine_backend(kind);
            }
            for candidate in auto_backend_order() {
                if let Ok(backend) = try_engine_backend(*candidate) {
                    return Ok(backend);
                }
            }
            Ok(cpu_engine_backend())
        }
        ComputeBackendKind::Cpu => Ok(cpu_engine_backend()),
        ComputeBackendKind::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                if let Some(metal) = larql_compute_metal::metal_backend() {
                    return Ok(Box::new(metal));
                }
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "no Metal device or pipeline initialization failed".to_string(),
                })
            }
            #[cfg(all(feature = "metal", not(target_os = "macos")))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "Metal backend is only available on macOS".to_string(),
                })
            }
            #[cfg(not(feature = "metal"))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "crate built without the `metal` feature".to_string(),
                })
            }
        }
        ComputeBackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                larql_compute_cuda::cuda_backend()
                    .map(|backend| Box::new(backend) as Box<dyn EngineBackend>)
                    .map_err(|err| BackendSelectionError::Unavailable {
                        kind,
                        reason: err.to_string(),
                    })
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "crate built without the `cuda` feature".to_string(),
                })
            }
        }
        ComputeBackendKind::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                larql_compute_vulkan::vulkan_backend()
                    .map(|backend| Box::new(backend) as Box<dyn EngineBackend>)
                    .map_err(|err| BackendSelectionError::Unavailable {
                        kind,
                        reason: err.to_string(),
                    })
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "crate built without the `vulkan` feature".to_string(),
                })
            }
        }
    }
}

fn try_async_engine_backend(
    kind: ComputeBackendKind,
) -> Result<Box<dyn AsyncComputeBackend>, BackendSelectionError> {
    match kind {
        ComputeBackendKind::Auto => {
            if let Some(kind) = auto_override_kind()? {
                return try_async_engine_backend(kind);
            }
            for candidate in auto_backend_order() {
                if let Ok(backend) = try_async_engine_backend(*candidate) {
                    return Ok(backend);
                }
            }
            Ok(cpu_async_engine_backend())
        }
        ComputeBackendKind::Cpu => Ok(cpu_async_engine_backend()),
        ComputeBackendKind::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                if let Some(metal) = larql_compute_metal::metal_backend() {
                    return Ok(Box::new(metal));
                }
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "no Metal device or pipeline initialization failed".to_string(),
                })
            }
            #[cfg(all(feature = "metal", not(target_os = "macos")))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "Metal backend is only available on macOS".to_string(),
                })
            }
            #[cfg(not(feature = "metal"))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "crate built without the `metal` feature".to_string(),
                })
            }
        }
        ComputeBackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                larql_compute_cuda::cuda_backend()
                    .map(|backend| Box::new(backend) as Box<dyn AsyncComputeBackend>)
                    .map_err(|err| BackendSelectionError::Unavailable {
                        kind,
                        reason: err.to_string(),
                    })
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "crate built without the `cuda` feature".to_string(),
                })
            }
        }
        ComputeBackendKind::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                larql_compute_vulkan::vulkan_backend()
                    .map(|backend| Box::new(backend) as Box<dyn AsyncComputeBackend>)
                    .map_err(|err| BackendSelectionError::Unavailable {
                        kind,
                        reason: err.to_string(),
                    })
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err(BackendSelectionError::Unavailable {
                    kind,
                    reason: "crate built without the `vulkan` feature".to_string(),
                })
            }
        }
    }
}

pub fn compute_backend(
    kind: ComputeBackendKind,
) -> Result<Box<dyn larql_compute::ComputeBackend>, BackendSelectionError> {
    try_compute_backend(kind)
}

pub fn engine_backend(
    kind: ComputeBackendKind,
) -> Result<Box<dyn EngineBackend>, BackendSelectionError> {
    try_engine_backend(kind)
}

pub fn async_engine_backend(
    kind: ComputeBackendKind,
) -> Result<Box<dyn AsyncComputeBackend>, BackendSelectionError> {
    try_async_engine_backend(kind)
}

/// Default backend as `Box<dyn EngineBackend>`, following
/// [`ComputeBackendKind::Auto`].
///
/// Honours the [`ENV_LARQL_BACKEND`] override. Normal Auto resolution
/// (override unset/`auto`) stays non-panicking and falls back to CPU when
/// no GPU backend is available. An explicit-but-invalid or unavailable
/// override fails loudly here — silently degrading an explicit request to
/// CPU would violate the CLI contract. Callers that prefer a `Result`
/// should use [`engine_backend`] directly.
pub fn default_engine_backend() -> Box<dyn EngineBackend> {
    match engine_backend(ComputeBackendKind::Auto) {
        Ok(backend) => backend,
        Err(err) => panic!(
            "{ENV_LARQL_BACKEND} override rejected (use `auto` or unset to fall back): {err}"
        ),
    }
}

/// Default async backend as `Box<dyn AsyncComputeBackend>`, following
/// [`ComputeBackendKind::Auto`].
///
/// See [`default_engine_backend`] for the override/failure contract.
pub fn default_async_engine_backend() -> Box<dyn AsyncComputeBackend> {
    match async_engine_backend(ComputeBackendKind::Auto) {
        Ok(backend) => backend,
        Err(err) => panic!(
            "{ENV_LARQL_BACKEND} override rejected (use `auto` or unset to fall back): {err}"
        ),
    }
}

/// Default compute backend as `Box<dyn ComputeBackend>`, following
/// [`ComputeBackendKind::Auto`].
///
/// See [`default_engine_backend`] for the override/failure contract.
pub fn default_compute_backend() -> Box<dyn larql_compute::ComputeBackend> {
    match compute_backend(ComputeBackendKind::Auto) {
        Ok(backend) => backend,
        Err(err) => panic!(
            "{ENV_LARQL_BACKEND} override rejected (use `auto` or unset to fall back): {err}"
        ),
    }
}

/// Map a model's activation function to the compute-layer `Activation` enum.
pub fn activation_from_arch(
    arch: &dyn larql_models::ModelArchitecture,
) -> larql_compute::Activation {
    match arch.activation() {
        larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
        _ => larql_compute::Activation::Silu,
    }
}

// Re-export essentials at crate root.
pub use async_compute_backend::{
    AsyncComputeBackend, AsyncDispatchError, AttentionHandle, AttentionHandleInner, ReadyAttention,
    ReadyResidualUpload, ResidualUploadHandle, ResidualUploadHandleInner,
};
pub use attention::AttentionWeights;
pub use capture::{
    CaptureCallbacks, CaptureConfig, InferenceModel, TopKEntry, VectorFileHeader, VectorRecord,
    DEFAULT_ACTIVATION_TOP_K, DEFAULT_RESIDUAL_TOP_K,
};
pub use chat::{wrap_chat_prompt, wrap_prompt_raw, wrap_with_vindex_template, ChatWrap};
pub use error::InferenceError;
pub use ffn::graph_backend::{GateIndex, IndexBuildCallbacks, SilentIndexCallbacks};
pub use ffn::{
    BackendFfn, FfnBackend, LayerFfnRouter, LayerShardedBackend, MoeRouterWeights, RemoteFfnConfig,
    RemoteFfnError, RemoteLatencyStats, RemoteMoeBackend, RemoteMoeError, RemoteWalkBackend,
    ShardConfig, SparseFfn, WeightFfn, WirePreference,
};
pub use kv_dispatch::{
    CompressionCodec, EngineBackend, KvDispatch, KvHandle, KvHandleInner, PerLayerDecodeState,
    ResidualHandle, ResidualHandleInner,
};
pub use kv_engine::{DecodeStageSummary, EngineInfo, KvEngine};
// Crate-root forward re-exports — kept for any name with external use OR
// in-crate examples/tests/benches that already import from the root. The
// curated `research` module (below) re-sources these from subpaths so it
// keeps working when individual root re-exports are dropped.
//
// Truly-unused root re-exports (no external + no inference example/test
// usage) were dropped 2026-05-09: `capture_ffn_activation_matrix`,
// `estimate_ffn_covariance`, `forward_raw_logits`, `infer_patched_q4k`,
// `predict_from_hidden_with_ffn`, `predict_with_ffn_trace`,
// `trace_forward_with_ffn`, `InferPatchedResult`, `LayerMode`,
// `MemitFactResult`, `PredictResultWithAttention`,
// `PredictResultWithResiduals`, `RawForward`, `SpecCapture`,
// `TargetDelta`, `TraceResult`, `KNN_COSINE_THRESHOLD`. They remain
// accessible via `larql_inference::forward::*` and `research::*`.
pub use forward::{
    apply_knn_override, apply_knn_override_two_tier, apply_knn_override_verified,
    calibrate_scalar_gains, capture_decoy_residuals, capture_residuals, capture_spec_residuals,
    forward_from_layer, forward_to_layer, hidden_to_raw_logits, infer_patched, logit_lens_top1,
    predict, predict_from_hidden, predict_with_ffn, predict_with_ffn_attention,
    predict_with_router, predict_with_strategy, run_memit, run_memit_with_target_opt,
    trace_forward, trace_forward_full, walk_trace_from_residuals, InferenceWeights, KnnOverride,
    KnnRouteMode, LayerAttentionCapture, MemitFact, MemitResult, PredictResult, TargetDeltaOpts,
};
// Crate-root layer_graph re-exports — kept for any name with external use
// OR in-crate examples/tests/benches that import via the root. Truly-unused
// names (no external + no inference example/test usage) dropped 2026-05-09:
// `GridGenerateResult`, `ChatMLRenderer`, `GemmaRenderer`, `LayerOutput`,
// `Llama3Renderer`, `PerLayerGraph`, `TurnRenderer`. They remain reachable
// via `larql_inference::layer_graph::*`.
pub use layer_graph::{
    build_adaptive_graph,
    detect_template,
    generate,
    generate_streaming,
    generate_with_sampling,
    // Expert grid generation
    grid::{
        generate_with_remote_ffn, generate_with_remote_ffn_batch, generate_with_remote_moe,
        generate_with_remote_moe_batch,
    },
    hybrid::predict_hybrid,
    predict_honest,
    predict_pipeline,
    predict_split_cached,
    predict_split_pass,
    predict_with_graph,
    predict_with_graph_vindex_logits,
    trace_with_graph,
    try_generate,
    try_generate_streaming,
    try_generate_with_sampling,
    AttentionCache,
    CachedLayerGraph,
    // Multi-turn chat session
    ChatSession,
    DenseLayerGraph,
    // Generation building blocks (EOS, detok, sampling)
    Detokenizer,
    EosConfig,
    GenerateError,
    GenerateResult,
    GuidedWalkLayerGraph,
    // Production
    LayerGraph,
    PipelinedLayerGraph,
    Sampler,
    SamplingConfig,
    // Analysis/validation
    TemplatePattern,
    TemplateUniverse,
    WalkLayerGraph,
};
pub use model::{load_model_dir, resolve_model_path, DequantScratch, ModelWeights, WeightsView};
pub use prompt_render::{
    ChatMessage, ChatRole, PromptAssets, PromptEncoding, PromptInput, TemplateRevision,
    ThinkingMode, TokenizerPolicy,
};
pub use tokenizer::{decode_token, decode_token_raw, encode_prompt, load_tokenizer};
pub use trace::{
    trace as trace_decomposed, trace_residuals, AnswerWaypoint, BoundaryStore, BoundaryWriter,
    ContextStore, ContextTier, ContextWriter, LayerSummary, ResidualTrace, TraceNode,
    TracePositions, TraceStore, TraceWriter,
};
pub use vindex::{
    generate_kquant_cpu_remote, open_inference_vindex, predict_kquant, FfnL1Cache, WalkFfn,
    WalkFfnConfig,
};

/// Stable, application-facing inference imports.
///
/// New downstream code should prefer this module over broad crate-root
/// glob imports. The crate root remains source-compatible while the public
/// surface is gradually narrowed.
pub mod prelude {
    pub use crate::{
        default_backend, generate, generate_streaming, generate_with_sampling, load_model_dir,
        load_tokenizer, open_inference_vindex, predict, predict_kquant, resolve_model_path,
        try_generate, try_generate_streaming, try_generate_with_sampling, wrap_chat_prompt,
        wrap_prompt_raw, wrap_with_vindex_template, ChatWrap, ComputeBackend, Detokenizer,
        EosConfig, GenerateError, GenerateResult, InferenceError, ModelWeights, Sampler,
        SamplingConfig, WalkFfn, WalkFfnConfig,
    };
    pub use larql_compute::CpuBackend;
}

/// Mechanistic-interpretability and research-facing imports.
///
/// These APIs are intentionally more experimental than [`prelude`]. Grouping
/// them here makes that boundary visible without breaking existing crate-root
/// users in one large move.
///
/// `KvEngine`, `EngineInfo`, and `DecodeStageSummary` are defined in
/// this crate's [`kv_engine`](crate::kv_engine) module and re-exported
/// at the crate root. Concrete engine implementations
/// (`MarkovResidualEngine`, `UnlimitedContextEngine`, `StandardEngine`,
/// `NoCacheEngine`, `TurboQuantEngine`, `ApolloEngine`) plus
/// `EngineKind` and accuracy helpers (`compare_hidden`,
/// `cosine_similarity`, `kl_divergence`, …) live in the `larql-kv`
/// crate — depend on it directly when you need concrete engines.
pub mod research {
    // Source directly from subpaths so this curated surface keeps working
    // even when individual root re-exports are dropped. Kept as a single
    // import block per source module so the surface is easy to scan.
    pub use crate::forward::{
        apply_knn_override, calibrate_scalar_gains, capture_decoy_residuals,
        capture_ffn_activation_matrix, capture_residuals, capture_spec_residuals,
        estimate_ffn_covariance, forward_from_layer, forward_raw_logits, forward_to_layer,
        hidden_to_raw_logits, infer_patched, infer_patched_q4k, logit_lens_top1,
        predict_from_hidden, predict_from_hidden_with_ffn, predict_with_ffn,
        predict_with_ffn_attention, predict_with_ffn_trace, predict_with_router,
        predict_with_strategy, run_memit, run_memit_with_target_opt, trace_forward,
        trace_forward_full, trace_forward_with_ffn, walk_trace_from_residuals, InferPatchedResult,
        InferenceWeights, KnnOverride, LayerAttentionCapture, LayerMode, MemitFact,
        MemitFactResult, MemitResult, PredictResult, PredictResultWithAttention,
        PredictResultWithResiduals, RawForward, SpecCapture, TargetDelta, TargetDeltaOpts,
        TraceResult, KNN_COSINE_THRESHOLD,
    };
    pub use crate::layer_graph::{
        predict_honest, predict_pipeline, predict_with_graph, predict_with_graph_vindex_logits,
        trace_with_graph, AttentionCache, TemplatePattern, TemplateUniverse,
    };
    pub use crate::trace::{
        trace as trace_decomposed, trace_residuals, AnswerWaypoint, BoundaryStore, BoundaryWriter,
        ContextStore, ContextTier, ContextWriter, LayerSummary, ResidualTrace, TraceNode,
        TracePositions, TraceStore, TraceWriter,
    };
}

#[cfg(test)]
mod factory_tests {
    //! Coverage for the engine/backend factory functions at the crate root.
    //!
    //! Each factory exists so engines can construct themselves without the
    //! caller branching on `#[cfg(feature = "gpu")]`. The tests verify
    //! that the returned trait object is the CPU backend on the default
    //! build (no `metal` feature on the test runner CI matrix) and that
    //! each factory's pipeline-back name plumbs through.
    //!
    //! On Apple Silicon with the `metal` feature these tests still pass —
    //! the factories fall back to CPU when `MetalBackend::new()` returns
    //! `None`, and on metal-capable hardware the returned backend just
    //! reports a different name. The assertions are deliberately scoped
    //! to "factory returned a usable backend" rather than "the backend is
    //! CPU" so both build configurations pass.
    use super::*;
    use larql_models::Activation as ArchActivation;

    #[test]
    fn cpu_engine_backend_returns_named_backend() {
        let backend = cpu_engine_backend();
        // CpuBackend's name starts with "cpu" — exact suffix varies with
        // BLAS / kernel selection (e.g. "cpu (BLAS + C Q4 kernel)").
        assert!(backend.as_compute().name().starts_with("cpu"));
    }

    #[test]
    fn cpu_async_engine_backend_returns_named_backend() {
        let backend = cpu_async_engine_backend();
        let name = <dyn AsyncComputeBackend as larql_compute::ComputeBackend>::name(&*backend);
        assert!(name.starts_with("cpu"));
    }

    #[test]
    fn default_engine_backend_constructs() {
        // Returns Metal when the feature + hardware align, CPU otherwise.
        // The factory is exercised either way — we only check it returns
        // a backend with a non-empty name.
        let backend = default_engine_backend();
        assert!(!backend.as_compute().name().is_empty());
    }

    #[test]
    fn default_async_engine_backend_constructs() {
        let backend = default_async_engine_backend();
        assert!(
            !<dyn AsyncComputeBackend as larql_compute::ComputeBackend>::name(&*backend).is_empty()
        );
    }

    #[test]
    fn default_compute_backend_constructs() {
        let backend = default_compute_backend();
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn explicit_cpu_backend_kind_constructs() {
        let backend = compute_backend(ComputeBackendKind::Cpu).expect("cpu backend");
        assert!(backend.name().starts_with("cpu"));
    }

    #[test]
    fn auto_backend_kind_constructs() {
        let backend = compute_backend(ComputeBackendKind::Auto).expect("auto backend");
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn unavailable_explicit_backend_errors_loudly() {
        #[cfg(not(feature = "cuda"))]
        {
            match compute_backend(ComputeBackendKind::Cuda) {
                Err(err) => assert!(err.to_string().contains("cuda")),
                Ok(_) => panic!("cuda backend should be gated when the `cuda` feature is off"),
            }
        }
    }

    #[test]
    fn activation_from_arch_maps_gelu_tanh() {
        // Use a tiny ModelWeights so we have a real ModelArchitecture.
        let weights = test_utils::make_test_weights();
        // make_test_weights uses TinyModelArch which reports Silu — verify
        // the non-tanh branch maps to Silu.
        let act = activation_from_arch(&*weights.arch);
        assert!(matches!(act, larql_compute::Activation::Silu));

        // For the GeluTanh branch we need an arch that reports it. Build
        // a Gemma 3 arch via the existing fixture (gemma3 uses GeluTanh).
        let gemma = test_utils::make_gemma3_test_weights();
        assert!(matches!(gemma.arch.activation(), ArchActivation::GeluTanh));
        assert!(matches!(
            activation_from_arch(&*gemma.arch),
            larql_compute::Activation::GeluTanh
        ));
    }
}

#[cfg(test)]
mod backend_env_tests {
    //! Coverage for [`ENV_LARQL_BACKEND`] parsing and override behavior.
    //!
    //! All tests mutate the thread-local env override exposed by
    //! `larql_compute::options` (NOT `std::env::set_var`, which races the
    //! concurrent `getenv` that other parallel tests do on the decode path
    //! and can SIGSEGV libc). The `OverrideGuard` clears the override on
    //! drop so no cross-test leakage.

    use super::*;

    /// RAII guard that applies a `LARQL_BACKEND` override on the current
    /// thread and clears it (and any other overrides) on drop.
    struct OverrideGuard;

    impl OverrideGuard {
        fn set(value: Option<&str>) -> Self {
            larql_compute::options::set_env_override(ENV_LARQL_BACKEND, value);
            OverrideGuard
        }
    }

    impl Drop for OverrideGuard {
        fn drop(&mut self) {
            larql_compute::options::clear_fast_path_overrides();
        }
    }

    // ── ComputeBackendKind::parse ──────────────────────────────────────

    #[test]
    fn parse_accepts_known_vocab_case_and_whitespace_insensitive() {
        for (raw, expected) in [
            ("auto", ComputeBackendKind::Auto),
            ("cpu", ComputeBackendKind::Cpu),
            ("metal", ComputeBackendKind::Metal),
            ("cuda", ComputeBackendKind::Cuda),
            ("vulkan", ComputeBackendKind::Vulkan),
        ] {
            assert_eq!(ComputeBackendKind::parse(raw).unwrap(), expected);
        }
        // Mixed case + surrounding whitespace are normalized.
        assert_eq!(
            ComputeBackendKind::parse("  CUDA ").unwrap(),
            ComputeBackendKind::Cuda
        );
        assert_eq!(
            ComputeBackendKind::parse("Metal").unwrap(),
            ComputeBackendKind::Metal
        );
        assert_eq!(
            ComputeBackendKind::parse("\tVuLkAn\n").unwrap(),
            ComputeBackendKind::Vulkan
        );
    }

    #[test]
    fn parse_rejects_unknown_vocab_with_clear_message() {
        let err = ComputeBackendKind::parse("tpu").unwrap_err();
        assert!(err.contains("unknown backend `tpu`"), "got: {err}");
        // Message lists the accepted vocabulary so the fix is obvious.
        for expected in ["auto", "cpu", "metal", "cuda", "vulkan"] {
            assert!(
                err.contains(expected),
                "message should list `{expected}`: {err}"
            );
        }
    }

    #[test]
    fn parse_rejects_empty_string() {
        let err = ComputeBackendKind::parse("").unwrap_err();
        assert!(err.contains("unknown backend"), "got: {err}");
    }

    // ── auto_override_kind ─────────────────────────────────────────────

    #[test]
    fn auto_override_unset_returns_none() {
        let _g = OverrideGuard::set(None);
        assert_eq!(auto_override_kind().unwrap(), None);
    }

    #[test]
    fn auto_override_auto_returns_none() {
        // Both literal and mixed-case `auto` mean "no override".
        for raw in ["auto", "AUTO", "  Auto  "] {
            let _g = OverrideGuard::set(Some(raw));
            assert_eq!(auto_override_kind().unwrap(), None, "raw=`{raw}`");
        }
    }

    #[test]
    fn auto_override_cpu_returns_some_cpu() {
        let _g = OverrideGuard::set(Some("cpu"));
        assert_eq!(auto_override_kind().unwrap(), Some(ComputeBackendKind::Cpu));
    }

    #[test]
    fn auto_override_metal_returns_some_metal() {
        let _g = OverrideGuard::set(Some("metal"));
        assert_eq!(
            auto_override_kind().unwrap(),
            Some(ComputeBackendKind::Metal)
        );
    }

    #[test]
    fn auto_override_invalid_returns_invalid_env_error() {
        let _g = OverrideGuard::set(Some("quantum"));
        match auto_override_kind() {
            Err(BackendSelectionError::InvalidEnv { raw, reason }) => {
                assert_eq!(raw, "quantum");
                assert!(reason.contains("unknown backend"), "got: {reason}");
            }
            other => panic!("expected InvalidEnv, got {other:?}"),
        }
    }

    #[test]
    fn invalid_env_error_message_names_the_var() {
        let _g = OverrideGuard::set(Some("nope"));
        let err = auto_override_kind().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("LARQL_BACKEND"), "got: {msg}");
        assert!(msg.contains("nope"), "got: {msg}");
    }

    // ── try_compute_backend(Auto) honors the override ──────────────────

    #[test]
    fn auto_with_cpu_override_selects_cpu() {
        let _g = OverrideGuard::set(Some("cpu"));
        let backend = try_compute_backend(ComputeBackendKind::Auto).expect("cpu via override");
        assert!(backend.name().starts_with("cpu"));
    }

    #[test]
    fn auto_with_invalid_override_errors_instead_of_falling_back() {
        let _g = OverrideGuard::set(Some("bogus"));
        match try_compute_backend(ComputeBackendKind::Auto) {
            Err(BackendSelectionError::InvalidEnv { .. }) => {}
            Ok(backend) => panic!(
                "invalid override silently produced backend `{}`",
                backend.name()
            ),
            Err(other) => panic!("expected InvalidEnv, got {other:?}"),
        }
    }

    #[test]
    fn auto_with_unavailable_override_does_not_silently_pick_cpu() {
        // CUDA is gated behind the `cuda` feature. An explicit override to
        // it must NOT silently degrade to CPU — it must surface Unavailable.
        #[cfg(not(feature = "cuda"))]
        {
            let _g = OverrideGuard::set(Some("cuda"));
            match try_compute_backend(ComputeBackendKind::Auto) {
                Err(BackendSelectionError::Unavailable { kind, .. }) => {
                    assert_eq!(kind, ComputeBackendKind::Cuda);
                }
                Ok(backend) => panic!(
                    "explicit `cuda` override silently fell back to `{}`",
                    backend.name()
                ),
                Err(other) => panic!("expected Unavailable, got {other:?}"),
            }
        }
    }

    #[test]
    fn auto_with_no_override_falls_back_to_cpu_safely() {
        // No override: the platform Auto order must remain non-panicking and
        // always yield a usable backend (CPU on a GPU-less / no-feature host).
        let _g = OverrideGuard::set(None);
        let backend = try_compute_backend(ComputeBackendKind::Auto).expect("auto fallback");
        assert!(!backend.name().is_empty());
    }

    // ── default_* panic on bad override, stay quiet otherwise ──────────

    #[test]
    fn default_compute_backend_quiet_without_override() {
        let _g = OverrideGuard::set(None);
        let backend = default_compute_backend();
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn default_compute_backend_panics_on_invalid_override() {
        let _g = OverrideGuard::set(Some("garbage"));
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(default_compute_backend));
        assert!(
            result.is_err(),
            "invalid LARQL_BACKEND override should panic, not silently degrade"
        );
    }
}

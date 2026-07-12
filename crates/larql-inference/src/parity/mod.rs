//! ST5 first-token semantic parity — trace capture, interchange format, and
//! the committed numerical comparator with first-divergence drill-down.
//!
//! This module is the LARQL side of the LARQL-INFERENCE-TRUST-001A-ST5 slice.
//! It proves that LARQL's canonical F32 CPU forward path produces the same
//! first-next-token logits (and the same per-layer semantic residuals) as a
//! pinned Transformers CPU float32 eager oracle for the official Gemma 4 E2B
//! model.
//!
//! Pipeline:
//!
//! 1. [`capture::write_larql_trace`] runs the production F32 forward via
//!    `larql_compute::forward::forward_raw_logits_traced` and writes a trace
//!    directory.
//! 2. A separate Python process (`scripts/gemma4_first_token_oracle.py`)
//!    runs the Transformers CPU float32 eager forward and writes a trace
//!    directory in the same format.
//! 3. [`compare::compare_traces`] diffs the two traces against the committed
//!    [`compare::Policy::st5_default`], reporting the earliest failing
//!    boundary, layer, and stage.
//!
//! Scope: F32 CPU first-token logits only. No generation, sampling, KV-cached
//! decode, Q4_K, or GPU.

pub mod capture;
pub mod compare;
pub mod format;

pub use capture::{capture_prompt, write_larql_trace};
pub use compare::{
    compare_detailed_stage, compare_tensor, compare_traces, Decision, FirstDivergence, LogitTopK,
    ParityResult, Policy, PolicyView, PromptComparison, StageMetrics,
};
pub use format::{
    coarse_stage_order, entry_at, entry_from, read_tensor, required_coarse_stages, StageId,
    TraceError, TraceManifest, TracePrompt, TraceTensor, STAGE_EMBEDDING, STAGE_FINAL_LOGITS,
    STAGE_FINAL_NORM, STAGE_LAYER_INPUT, STAGE_LM_HEAD_RAW, STAGE_POST_ATTENTION, STAGE_POST_FFN,
    STAGE_POST_LAYER, STAGE_POST_PLE, STAGE_PRE_FINAL_NORM,
};

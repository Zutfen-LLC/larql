//! Logits/forward-pass orchestration. `raw` (forward_from_layer) and
//! `types` (PredictResult + capture types) live here. `dense` and
//! `ffn` remain in `larql-inference` — they're orchestration around
//! engine state.

pub mod raw;
pub mod types;

pub use raw::{
    forward_from_layer, forward_raw_logits, forward_raw_logits_traced,
    forward_raw_logits_with_prefix, RawForward, TracedTail,
};
pub use types::{
    LayerAttentionCapture, LayerMode, PredictResult, PredictResultWithAttention,
    PredictResultWithResiduals, TraceResult,
};

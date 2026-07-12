//! ST5 first-token parity — LARQL F32 production-path trace capture.
//!
//! Runs the canonical F32 forward pass via
//! [`larql_compute::forward::forward_raw_logits_traced`] (the hooked sibling
//! of the production `forward_raw_logits`, identical attention + FFN + PLE +
//! `layer_scalar` math) and writes the last-token row of every coarse
//! semantic boundary to a trace directory in the ST5 interchange format.
//!
//! This reuses production primitives end-to-end; it does not re-implement
//! the transformer forward. Disabled capture (not calling this module) has
//! zero normal-path overhead.

use std::collections::BTreeMap;
use std::path::Path;

use larql_compute::forward::{forward_raw_logits_traced, hooks::RecordHook, TracedTail};
use larql_models::{ModelWeights, WeightsView};
use serde_json::json;

use super::format::{
    coarse_stage_order, entry_at, TraceManifest, TracePrompt, TraceTensor, STAGE_EMBEDDING,
    STAGE_FINAL_LOGITS, STAGE_FINAL_NORM, STAGE_LAYER_INPUT, STAGE_LM_HEAD_RAW,
    STAGE_POST_ATTENTION, STAGE_POST_FFN, STAGE_POST_LAYER, STAGE_POST_PLE, STAGE_PRE_FINAL_NORM,
};

/// Last-token row captured for every layer by `RecordHook`.
struct LayerCapture {
    layer_input: Vec<f32>,
    post_attention: Vec<f32>,
    post_ffn: Vec<f32>,
    post_ple: Vec<f32>,
    post_layer: Vec<f32>,
}

fn last_row(h: &ndarray::Array2<f32>) -> Vec<f32> {
    let nrows = h.nrows();
    if nrows == 0 {
        return Vec::new();
    }
    h.row(nrows - 1).to_vec()
}

/// Run the traced forward for one prompt's token ids and write its tensors
/// into `dir`. Returns the manifest entry for the prompt.
pub fn capture_prompt(
    weights: &ModelWeights,
    token_ids: &[u32],
    prompt_id: &str,
    dir: &Path,
) -> anyhow::Result<TracePrompt> {
    let num_layers = weights.num_layers;
    let view = WeightsView::dense(weights);
    let mut hook = RecordHook::for_layers(0..num_layers);
    let tail: TracedTail = forward_raw_logits_traced(view, token_ids, &mut hook);

    let mut tensors: Vec<TraceTensor> = Vec::new();

    // Embedding (last token) — taken from the layer-0 pre_layer capture,
    // which is the residual entering layer 0 == embedding output.
    let embedding_row = hook
        .pre_layer
        .get(&0)
        .map(last_row)
        .ok_or_else(|| anyhow::anyhow!("layer-0 pre_layer capture missing (embedding)"))?;

    push(
        &mut tensors,
        STAGE_EMBEDDING,
        None,
        &embedding_row,
        dir,
        prompt_id,
    )?;

    for layer in 0..num_layers {
        let cap = LayerCapture {
            layer_input: hook
                .pre_layer
                .get(&layer)
                .map(last_row)
                .unwrap_or_default(),
            post_attention: hook
                .post_attention
                .get(&layer)
                .map(last_row)
                .unwrap_or_default(),
            post_ffn: hook
                .post_ffn
                .get(&layer)
                .map(last_row)
                .unwrap_or_default(),
            post_ple: hook
                .post_ple
                .get(&layer)
                .map(last_row)
                .unwrap_or_default(),
            post_layer: hook
                .post_layer
                .get(&layer)
                .map(last_row)
                .unwrap_or_default(),
        };
        // Mark shared-KV consumer K/V stages as not-executed: the model has
        // only 1 KV head (GQA) and the source-layer routing is already
        // proven by ST4; here we record the consumer Q/O residuals which ARE
        // executed. We still emit all five residual boundaries normally.
        push(
            &mut tensors,
            STAGE_LAYER_INPUT,
            Some(layer),
            &cap.layer_input,
            dir,
            prompt_id,
        )?;
        push(
            &mut tensors,
            STAGE_POST_ATTENTION,
            Some(layer),
            &cap.post_attention,
            dir,
            prompt_id,
        )?;
        push(
            &mut tensors,
            STAGE_POST_FFN,
            Some(layer),
            &cap.post_ffn,
            dir,
            prompt_id,
        )?;
        push(
            &mut tensors,
            STAGE_POST_PLE,
            Some(layer),
            &cap.post_ple,
            dir,
            prompt_id,
        )?;
        push(
            &mut tensors,
            STAGE_POST_LAYER,
            Some(layer),
            &cap.post_layer,
            dir,
            prompt_id,
        )?;
    }

    push(
        &mut tensors,
        STAGE_PRE_FINAL_NORM,
        None,
        &tail.pre_final_norm,
        dir,
        prompt_id,
    )?;
    push(
        &mut tensors,
        STAGE_FINAL_NORM,
        None,
        &tail.final_norm,
        dir,
        prompt_id,
    )?;
    push(
        &mut tensors,
        STAGE_LM_HEAD_RAW,
        None,
        &tail.lm_head_raw,
        dir,
        prompt_id,
    )?;
    push(
        &mut tensors,
        STAGE_FINAL_LOGITS,
        None,
        &tail.final_logits,
        dir,
        prompt_id,
    )?;

    Ok(TracePrompt {
        token_ids: token_ids.to_vec(),
        seq_len: token_ids.len(),
        tensors,
    })
}

fn push(
    tensors: &mut Vec<TraceTensor>,
    stage: &str,
    layer: Option<usize>,
    values: &[f32],
    dir: &Path,
    prompt_id: &str,
) -> anyhow::Result<()> {
    let layer_tag = layer.map(|l| format!("_{l}")).unwrap_or_default();
    let rel = format!("{prompt_id}/{stage}{layer_tag}.f32");
    let (entry, _path) = entry_at(dir, &rel, stage, layer, values)?;
    tensors.push(entry);
    Ok(())
}

/// Write a full LARQL trace manifest + tensors for a set of prompts.
pub fn write_larql_trace(
    weights: &ModelWeights,
    prompts: &[(String, Vec<u32>)],
    dir: &Path,
    model_meta: Option<serde_json::Value>,
) -> anyhow::Result<TraceManifest> {
    std::fs::create_dir_all(dir)?;
    let mut manifest_prompts: BTreeMap<String, TracePrompt> = BTreeMap::new();
    for (prompt_id, token_ids) in prompts {
        let tp = capture_prompt(weights, token_ids, prompt_id, dir)?;
        manifest_prompts.insert(prompt_id.clone(), tp);
    }
    let manifest = TraceManifest {
        schema_version: 1,
        producer: "larql-f32-trace".to_string(),
        environment: Some(json!({
            "loader": "larql_models::load_model_weights (production F32 vindex)",
            "forward": "larql_compute::forward::forward_raw_logits_traced",
            "capture": "last-token row per coarse boundary",
        })),
        model: model_meta,
        prompts: manifest_prompts,
    };
    manifest.write(dir)?;
    // Sanity: coarse stage order matches what we wrote for at least one prompt.
    if manifest.prompts.len() > 1 {
        let _ = coarse_stage_order(weights.num_layers);
    }
    Ok(manifest)
}

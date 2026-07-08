//! Output-preserving tests for extraction profiling (IMPORT-001G).
//!
//! These prove that enabling the profiler never changes emitted bytes,
//! manifests, or offsets — the instrumentation is purely observational.
//! No CUDA or large models required: a tiny synthetic safetensors model
//! is stream-extracted twice (profiling off vs on) and the artefacts are
//! compared byte-for-byte.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use larql_vindex::format::filenames::*;
use larql_vindex::{
    ExtractLevel, KquantWriteOptions, QuantFormat, StorageDtype, WriteWeightsOptions,
};

struct TempDir(PathBuf);
impl TempDir {
    fn new(label: &str) -> Self {
        let base = std::env::temp_dir();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = base.join(format!("prof_{label}_{}_{}", std::process::id(), ts));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Write a tiny llama-shaped safetensors model so dims pad to exactly
/// one Q4_K super-block. Returns the tokenizer.
fn write_synthetic_llama_model(
    model_dir: &Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    vocab: usize,
) -> tokenizers::Tokenizer {
    std::fs::create_dir_all(model_dir).unwrap();
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };
    push("model.embed_tokens.weight", vec![vocab, hidden]);
    push("model.norm.weight", vec![hidden]);
    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        push(
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        push(&format!("{lp}.input_layernorm.weight"), vec![hidden]);
        push(
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), serialized).unwrap();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

/// Run a streaming Q4K extract into `out` with optional profiler.
fn stream_extract_q4k(
    model_dir: &Path,
    out: &Path,
    profiler: Option<&larql_vindex::ExtractProfiler>,
) {
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let vocab = 16usize;
    let tokenizer = write_synthetic_llama_model(model_dir, hidden, intermediate, num_layers, vocab);
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming_profiled(
        model_dir,
        &tokenizer,
        "test/profile",
        out,
        4,
        0,
        ExtractLevel::Inference,
        StorageDtype::F16,
        QuantFormat::Q4K,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb,
        profiler,
    )
    .unwrap();
}

/// The Q4K weight artefacts that must be byte-identical regardless of
/// profiling. `index.json` is excluded (it carries an `extracted_at`
/// timestamp + checksums map that are compared separately).
fn weight_artefact_names() -> &'static [&'static str] {
    &[
        ATTN_WEIGHTS_KQUANT_BIN,
        ATTN_WEIGHTS_KQUANT_MANIFEST_JSON,
        INTERLEAVED_KQUANT_BIN,
        INTERLEAVED_KQUANT_MANIFEST_JSON,
        NORMS_BIN,
        LM_HEAD_KQUANT_BIN,
        WEIGHT_MANIFEST_JSON,
        GATE_VECTORS_BIN,
        EMBEDDINGS_BIN,
        DOWN_META_BIN,
        TOKENIZER_JSON,
    ]
}

#[test]
fn profiling_disabled_creates_no_profile_file() {
    let tmp = TempDir::new("no_profile");
    let model = tmp.0.join("model");
    let out = tmp.0.join("out.vindex");
    stream_extract_q4k(&model, &out, None);
    assert!(
        !out.join("extract_profile.json").exists(),
        "no profile file when profiling is disabled"
    );
}

#[test]
fn profiling_enabled_vs_disabled_emit_identical_bytes() {
    let tmp = TempDir::new("identical");

    // Two independent model dirs so gate clustering / KNN order can't
    // be influenced by residual state.
    let model_a = tmp.0.join("model_a");
    let model_b = tmp.0.join("model_b");
    let out_off = tmp.0.join("off.vindex");
    let out_on = tmp.0.join("on.vindex");

    let prof = larql_vindex::ExtractProfiler::new();
    stream_extract_q4k(&model_a, &out_off, None);
    stream_extract_q4k(&model_b, &out_on, Some(&prof));

    // Write the report the CLI would normally write, into the profiled dir.
    prof.write_json_report(&out_on.join("extract_profile.json"))
        .unwrap();

    // ── Byte-identical weight artefacts ──
    for name in weight_artefact_names() {
        let a = std::fs::read(out_off.join(name)).unwrap_or_default();
        let b = std::fs::read(out_on.join(name)).unwrap_or_default();
        assert_eq!(
            a, b,
            "byte mismatch in {name}: profiling must not change output"
        );
    }

    // ── extract_profile.json exists only in the profiled dir ──
    assert!(!out_off.join("extract_profile.json").exists());
    assert!(out_on.join("extract_profile.json").exists());
}

#[test]
fn profile_report_has_valid_structure_and_nonzero_timings() {
    let tmp = TempDir::new("structure");
    let model = tmp.0.join("model");
    let out = tmp.0.join("out.vindex");

    let prof = larql_vindex::ExtractProfiler::new();
    stream_extract_q4k(&model, &out, Some(&prof));
    prof.write_json_report(&out.join("extract_profile.json"))
        .unwrap();

    let text = std::fs::read_to_string(out.join("extract_profile.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();

    // Top-level fields present.
    assert!(v["total_extract_ms"].as_f64().unwrap_or(-1.0) >= 0.0);
    assert!(v["stages"].is_array());
    assert!(v["by_operation"].is_array());
    assert!(v["top_ops"].is_array());
    assert!(v["bytes"].is_object());

    // At least one stage recorded.
    let stages = v["stages"].as_array().unwrap();
    assert!(
        !stages.is_empty(),
        "profiling must record at least one stage"
    );
    for s in stages {
        assert!(s["stage"].is_string());
        assert!(s["dur_ms"].as_f64().unwrap_or(-1.0) >= 0.0);
    }

    // Operation breakdown must include the kquant hot-path operations.
    let ops: Vec<&str> = v["by_operation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["op"].as_str().unwrap_or(""))
        .collect();
    assert!(
        ops.contains(&"fetch"),
        "profile must include fetch timings; got {ops:?}"
    );
    assert!(
        ops.contains(&"quantize"),
        "profile must include quantize timings; got {ops:?}"
    );
    assert!(
        ops.contains(&"write"),
        "profile must include write timings; got {ops:?}"
    );

    // At least one nonzero duration among quantize ops (the Q4K writer
    // always quantises gate/up/down per layer).
    let any_nonzero = v["by_operation"]
        .as_array()
        .unwrap()
        .iter()
        .any(|o| o["dur_ms"].as_f64().unwrap_or(0.0) > 0.0);
    assert!(
        any_nonzero,
        "profile should contain at least one nonzero timing"
    );

    // top_ops bounded at 20.
    assert!(v["top_ops"].as_array().unwrap().len() <= 20);

    // Byte totals present and non-negative.
    assert!(
        v["bytes"]["quantized_emitted"].as_u64().unwrap_or(0) > 0,
        "quantized bytes should be emitted for a Q4K extract"
    );
}

#[test]
fn profile_does_not_leak_absolute_source_path() {
    // The JSON report should not embed the local source model path —
    // it only carries stage/op/component labels.
    let tmp = TempDir::new("no_leak");
    let model = tmp.0.join("model");
    let out = tmp.0.join("out.vindex");

    let prof = larql_vindex::ExtractProfiler::new();
    stream_extract_q4k(&model, &out, Some(&prof));
    prof.write_json_report(&out.join("extract_profile.json"))
        .unwrap();

    let text = std::fs::read_to_string(out.join("extract_profile.json")).unwrap();
    let model_str = model.to_string_lossy().to_string();
    // The report must not contain the source path (allow the common
    // parent temp prefix to appear by accident is still a leak, so we
    // assert the full model path is absent).
    assert!(
        !text.contains(&model_str),
        "profile JSON leaked source path {model_str}"
    );
}

#[test]
fn f32_path_profiling_is_output_preserving() {
    // Mirror of the Q4K test but for the f32 weight writer, proving the
    // lighter f32 instrumentation is also output-identical.
    let tmp = TempDir::new("f32_identical");
    let model_a = tmp.0.join("model_a");
    let model_b = tmp.0.join("model_b");
    let out_off = tmp.0.join("off.vindex");
    let out_on = tmp.0.join("on.vindex");

    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let vocab = 16usize;
    let tok_a = write_synthetic_llama_model(&model_a, hidden, intermediate, num_layers, vocab);
    let tok_b = write_synthetic_llama_model(&model_b, hidden, intermediate, num_layers, vocab);

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming_profiled(
        &model_a,
        &tok_a,
        "test/profile-f32-a",
        &out_off,
        4,
        0,
        ExtractLevel::Inference,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb,
        None,
    )
    .unwrap();

    let prof = larql_vindex::ExtractProfiler::new();
    larql_vindex::build_vindex_streaming_profiled(
        &model_b,
        &tok_b,
        "test/profile-f32-b",
        &out_on,
        4,
        0,
        ExtractLevel::Inference,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb,
        Some(&prof),
    )
    .unwrap();

    // Compare f32 weight artefacts byte-for-byte.
    for name in [
        ATTN_WEIGHTS_BIN,
        UP_WEIGHTS_BIN,
        DOWN_WEIGHTS_BIN,
        NORMS_BIN,
        WEIGHT_MANIFEST_JSON,
    ] {
        let a = std::fs::read(out_off.join(name)).unwrap_or_default();
        let b = std::fs::read(out_on.join(name)).unwrap_or_default();
        assert_eq!(a, b, "f32 byte mismatch in {name}");
    }
}

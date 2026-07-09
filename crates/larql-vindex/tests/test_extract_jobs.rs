//! Output-preserving tests for bounded-parallel kquant extraction
//! (IMPORT-002F). Proves `--jobs 1` and `--jobs N>1` emit byte-identical
//! attention + dense-FFN artefacts and manifests — parallelising the
//! per-layer transform must never change what's written, only how many
//! threads do the transforming.
//!
//! No CUDA or large models required: a tiny synthetic multi-layer
//! safetensors model is stream-extracted twice (jobs=1 vs jobs=N) and the
//! artefacts are compared byte-for-byte. `extract_profile.json` is
//! excluded from the comparison (it legitimately differs: different
//! `jobs`/`parallel` fields and non-deterministic timings).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use larql_vindex::format::filenames::*;
use larql_vindex::{
    DownProjFormat, ExtractLevel, KquantWriteOptions, QuantFormat, StorageDtype,
    WriteWeightsOptions,
};

struct TempDir(PathBuf);
impl TempDir {
    fn new(label: &str) -> Self {
        let base = std::env::temp_dir();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = base.join(format!("jobs_{label}_{}_{}", std::process::id(), ts));
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
/// one Q4_K super-block. `num_layers` is deliberately > any `jobs` value
/// under test so chunked ordering bugs (wrong layer written out of
/// order across chunk boundaries) would be caught. Returns the tokenizer.
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
        // Layer-dependent values (not just index-dependent) so a
        // layer-swap bug in the parallel writer would perturb bytes.
        let salt = name
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as u32).wrapping_add(salt) % 997) as f32 * 0.001)
            .collect();
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

/// Run a streaming Q4K extract into `out` with the given `jobs` count.
fn stream_extract_q4k(model_dir: &Path, out: &Path, num_layers: usize, opts: KquantWriteOptions) {
    let hidden = 8usize;
    let intermediate = 4usize;
    let vocab = 16usize;
    let tokenizer = write_synthetic_llama_model(model_dir, hidden, intermediate, num_layers, vocab);
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming_profiled(
        model_dir,
        &tokenizer,
        "test/jobs",
        out,
        4,
        0,
        ExtractLevel::Inference,
        StorageDtype::F16,
        QuantFormat::Q4K,
        WriteWeightsOptions::default(),
        opts,
        false,
        &mut cb,
        None,
    )
    .unwrap();
}

/// The Q4K weight artefacts that must be byte-identical regardless of
/// `jobs`. `index.json` is excluded (timestamp/checksums compared
/// separately elsewhere); `extract_profile.json` is excluded entirely
/// per IMPORT-002F (it legitimately differs by design).
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

fn assert_artefacts_identical(out_a: &Path, out_b: &Path, extra: &[&str], ctx: &str) {
    for name in weight_artefact_names().iter().chain(extra.iter()) {
        let a = std::fs::read(out_a.join(name)).unwrap_or_default();
        let b = std::fs::read(out_b.join(name)).unwrap_or_default();
        assert_eq!(a, b, "[{ctx}] byte mismatch in {name}");
    }
}

#[test]
fn jobs_1_vs_jobs_2_emit_identical_bytes_multi_layer() {
    let tmp = TempDir::new("j1_vs_j2");
    // Two independent model dirs so gate clustering / KNN order can't be
    // influenced by residual state; both built with the same synthetic
    // generator so their tensor bytes are identical, only the writer's
    // `jobs` setting differs.
    let model_a = tmp.0.join("model_a");
    let model_b = tmp.0.join("model_b");
    let out_1 = tmp.0.join("jobs1.vindex");
    let out_2 = tmp.0.join("jobs2.vindex");

    // 5 layers: not evenly divisible by jobs=2, exercising a ragged final
    // chunk so off-by-one chunk-boundary bugs would surface.
    let num_layers = 5;
    stream_extract_q4k(
        &model_a,
        &out_1,
        num_layers,
        KquantWriteOptions {
            jobs: 1,
            ..Default::default()
        },
    );
    stream_extract_q4k(
        &model_b,
        &out_2,
        num_layers,
        KquantWriteOptions {
            jobs: 2,
            ..Default::default()
        },
    );

    assert_artefacts_identical(&out_1, &out_2, &[], "jobs=1 vs jobs=2");
}

#[test]
fn jobs_1_vs_jobs_4_emit_identical_bytes_multi_layer() {
    let tmp = TempDir::new("j1_vs_j4");
    let model_a = tmp.0.join("model_a");
    let model_b = tmp.0.join("model_b");
    let out_1 = tmp.0.join("jobs1.vindex");
    let out_4 = tmp.0.join("jobs4.vindex");

    // 6 layers over jobs=4: two chunks (4 + 2), exercising both a full
    // and a partial chunk.
    let num_layers = 6;
    stream_extract_q4k(
        &model_a,
        &out_1,
        num_layers,
        KquantWriteOptions {
            jobs: 1,
            ..Default::default()
        },
    );
    stream_extract_q4k(
        &model_b,
        &out_4,
        num_layers,
        KquantWriteOptions {
            jobs: 4,
            ..Default::default()
        },
    );

    assert_artefacts_identical(&out_1, &out_4, &[], "jobs=1 vs jobs=4");
}

#[test]
fn jobs_1_vs_jobs_4_down_q4k_identical() {
    // Covers the `down_proj = Q4K` branch (uniform Q4_K across
    // gate/up/down) under parallel transforms, not just the Q6_K default.
    let tmp = TempDir::new("j1_vs_j4_downq4k");
    let model_a = tmp.0.join("model_a");
    let model_b = tmp.0.join("model_b");
    let out_1 = tmp.0.join("jobs1.vindex");
    let out_4 = tmp.0.join("jobs4.vindex");

    let num_layers = 6;
    stream_extract_q4k(
        &model_a,
        &out_1,
        num_layers,
        KquantWriteOptions {
            jobs: 1,
            down_proj: DownProjFormat::Q4K,
            ..Default::default()
        },
    );
    stream_extract_q4k(
        &model_b,
        &out_4,
        num_layers,
        KquantWriteOptions {
            jobs: 4,
            down_proj: DownProjFormat::Q4K,
            ..Default::default()
        },
    );

    assert_artefacts_identical(&out_1, &out_4, &[], "jobs=1 vs jobs=4, down_proj=Q4K");
}

#[test]
fn jobs_1_vs_jobs_4_feature_major_down_identical() {
    // Covers the feature-major-down sidecar (kept serial by design even
    // under parallel primary transforms) alongside `down_features_q4k.bin`
    // + its manifest.
    let tmp = TempDir::new("j1_vs_j4_fmdown");
    let model_a = tmp.0.join("model_a");
    let model_b = tmp.0.join("model_b");
    let out_1 = tmp.0.join("jobs1.vindex");
    let out_4 = tmp.0.join("jobs4.vindex");

    let num_layers = 6;
    stream_extract_q4k(
        &model_a,
        &out_1,
        num_layers,
        KquantWriteOptions {
            jobs: 1,
            feature_major_down: true,
            ..Default::default()
        },
    );
    stream_extract_q4k(
        &model_b,
        &out_4,
        num_layers,
        KquantWriteOptions {
            jobs: 4,
            feature_major_down: true,
            ..Default::default()
        },
    );

    assert_artefacts_identical(
        &out_1,
        &out_4,
        &[DOWN_FEATURES_KQUANT_BIN, DOWN_FEATURES_KQUANT_MANIFEST_JSON],
        "jobs=1 vs jobs=4, feature_major_down",
    );
}

#[test]
fn jobs_0_and_jobs_1_are_equivalent_to_default_serial_path() {
    // `KquantWriteOptions::default()` (used by ~30 other call sites in
    // this workspace) resolves `jobs: 0`, which must behave exactly like
    // an explicit `jobs: 1` — both take the serial transform_then_write
    // branch with no thread pool constructed.
    let tmp = TempDir::new("jobs0_vs_jobs1");
    let model_a = tmp.0.join("model_a");
    let model_b = tmp.0.join("model_b");
    let out_default = tmp.0.join("default.vindex");
    let out_1 = tmp.0.join("jobs1.vindex");

    let num_layers = 3;
    stream_extract_q4k(
        &model_a,
        &out_default,
        num_layers,
        KquantWriteOptions::default(),
    );
    stream_extract_q4k(
        &model_b,
        &out_1,
        num_layers,
        KquantWriteOptions {
            jobs: 1,
            ..Default::default()
        },
    );

    assert_artefacts_identical(&out_default, &out_1, &[], "jobs=0 (default) vs jobs=1");
}

#[test]
fn profile_json_reports_jobs_and_parallel_flag() {
    let tmp = TempDir::new("profile_jobs");
    let model = tmp.0.join("model");
    let out = tmp.0.join("out.vindex");

    let prof = larql_vindex::ExtractProfiler::new();
    let num_layers = 6;
    let hidden = 8usize;
    let intermediate = 4usize;
    let vocab = 16usize;
    let tokenizer = write_synthetic_llama_model(&model, hidden, intermediate, num_layers, vocab);
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming_profiled(
        &model,
        &tokenizer,
        "test/jobs-profile",
        &out,
        4,
        0,
        ExtractLevel::Inference,
        StorageDtype::F16,
        QuantFormat::Q4K,
        WriteWeightsOptions::default(),
        KquantWriteOptions {
            jobs: 4,
            ..Default::default()
        },
        false,
        &mut cb,
        Some(&prof),
    )
    .unwrap();
    prof.write_json_report(&out.join("extract_profile.json"))
        .unwrap();

    let text = std::fs::read_to_string(out.join("extract_profile.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["jobs"].as_u64(), Some(4), "profile must report jobs=4");
    assert_eq!(
        v["parallel"].as_bool(),
        Some(true),
        "profile must report parallel=true when jobs>1"
    );

    // fetch/pad/quantize/write categories must still be present (worker
    // timings aggregate into the same op categories as the serial path).
    let ops: Vec<&str> = v["by_operation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["op"].as_str().unwrap_or(""))
        .collect();
    for expected in ["fetch", "pad", "quantize", "write"] {
        assert!(
            ops.contains(&expected),
            "parallel profile must still record {expected}; got {ops:?}"
        );
    }
}

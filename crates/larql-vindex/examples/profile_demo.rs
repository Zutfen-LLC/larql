//! Extraction profiling demo (IMPORT-001).
//!
//! Builds a tiny synthetic safetensors model, runs the streaming Q4_K
//! extract WITH profiling enabled, and prints the stderr hotspot summary
//! + writes `extract_profile.json`. This is what `larql extract
//! --profile-extract` produces, exercised through the library API.
//!
//! Run: cargo run --release -p larql-vindex --example profile_demo

use std::collections::HashMap;
use std::path::Path;

use larql_vindex::{
    build_vindex_streaming_profiled, ExtractLevel, ExtractProfiler, QuantFormat,
    SilentBuildCallbacks, StorageDtype,
};

const MINIMAL_TOKENIZER: &[u8] =
    br#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;

fn main() {
    println!("=== extraction profiling demo ===\n");

    let tmp = std::env::temp_dir().join("larql_profile_demo");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let model_dir = tmp.join("synth_model");
    let out = tmp.join("out_q4k.vindex");
    std::fs::create_dir_all(&model_dir).unwrap();

    let hidden = 64usize;
    let intermediate = 128usize;
    let num_layers = 4usize;
    let vocab = 32usize;
    make_synthetic_model(&model_dir, hidden, intermediate, num_layers, vocab);

    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(MINIMAL_TOKENIZER).unwrap();
    let prof = ExtractProfiler::new();
    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming_profiled(
        &model_dir,
        &tokenizer,
        "demo/profile",
        &out,
        5,
        0,
        ExtractLevel::All,
        StorageDtype::F32,
        QuantFormat::Q4K,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
        Some(&prof),
    )
    .expect("q4k extract");

    prof.write_json_report(&out.join("extract_profile.json"))
        .unwrap();
    prof.print_summary();

    let report = std::fs::read_to_string(out.join("extract_profile.json")).unwrap();
    println!("\n--- extract_profile.json (first 600 chars) ---");
    println!("{}", &report[..report.len().min(600)]);

    let _ = std::fs::remove_dir_all(&tmp);
    println!("\n=== done ===");
}

fn make_synthetic_model(
    dir: &Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    vocab: usize,
) {
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
        dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), MINIMAL_TOKENIZER).unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();
    let push = |tensors: &mut HashMap<String, Vec<f32>>,
                metadata: &mut Vec<(String, Vec<usize>)>,
                name: &str,
                shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };
    push(
        &mut tensors,
        &mut metadata,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.norm.weight",
        vec![hidden],
    );
    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.input_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
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
    std::fs::write(dir.join("model.safetensors"), serialized).unwrap();
}

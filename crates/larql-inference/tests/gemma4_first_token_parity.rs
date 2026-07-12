//! Environment-gated ST5 first-token semantic parity test.
//!
//! Proves that LARQL's canonical F32 CPU forward path produces the same
//! first-next-token logits and per-layer semantic residuals as the pinned
//! Transformers CPU float32 eager oracle for the official Gemma 4 E2B model.
//!
//! The Transformers oracle is run in a **separate process** (so both models
//! need not reside in memory simultaneously) and writes a trace directory in
//! the ST5 interchange format; this test points at that pre-generated trace:
//!
//! ```bash
//! # 1. Run the oracle (separate process):
//! LARQL_GEMMA4_ST_DIR=/path/to/google-gemma-4-E2B-it \
//! LARQL_GEMMA4_ST_REVISION=9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf \
//! LARQL_GEMMA4_ST5_ORACLE_DIR=/path/to/oracle-trace \
//!   python3 scripts/gemma4_first_token_oracle.py
//!
//! # 2. Run the comparison:
//! LARQL_GEMMA4_ST_DIR=/path/to/google-gemma-4-E2B-it \
//! LARQL_GEMMA4_ST_REVISION=9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf \
//! LARQL_GEMMA4_REFERENCE_VINDEX=/path/to/reference-f32.vindex \
//! LARQL_GEMMA4_ST5_ORACLE_DIR=/path/to/oracle-trace \
//! cargo test -p larql-inference --test gemma4_first_token_parity \
//!   --release -- --ignored --nocapture
//! ```
//!
//! When the environment is unset the test soft-skips, so CI without the
//! 18.65 GB artifact still passes.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use larql_inference::parity::{compare_traces, write_larql_trace, Policy, TraceManifest};
use larql_models::ModelWeights;
use larql_vindex::{load_model_weights, SilentLoadCallbacks};
use serde_json::Value;
use sha2::{Digest, Sha256};

const EXPECTED_REVISION: &str = "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf";
const EXPECTED_SAFETENSORS_SHA256: &str =
    "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550";
const RESOURCE_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
];
const PARITY_ARTIFACT: &str = "bench/baselines/gemma4-e2b-tokenizer-prompt-parity-2026-07-12.json";
const PROMPT_ORDER: &[&str] = &["raw_completion", "chat", "arithmetic", "multiturn"];

#[test]
#[ignore = "requires LARQL_GEMMA4_ST_DIR + LARQL_GEMMA4_REFERENCE_VINDEX + LARQL_GEMMA4_ST5_ORACLE_DIR; run with --ignored"]
fn gemma4_first_token_parity() {
    let st_dir = match env_path("LARQL_GEMMA4_ST_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skipped: LARQL_GEMMA4_ST_DIR is not set");
            return;
        }
    };
    let vindex_dir = match env_path("LARQL_GEMMA4_REFERENCE_VINDEX") {
        Some(p) => p,
        None => {
            eprintln!("skipped: LARQL_GEMMA4_REFERENCE_VINDEX is not set");
            return;
        }
    };
    let oracle_dir = match env_path("LARQL_GEMMA4_ST5_ORACLE_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skipped: LARQL_GEMMA4_ST5_ORACLE_DIR is not set");
            return;
        }
    };
    let revision = std::env::var("LARQL_GEMMA4_ST_REVISION")
        .expect("LARQL_GEMMA4_ST_REVISION is required for a pinned source audit");
    assert_eq!(revision.len(), 40, "revision must be a full commit SHA");
    assert_eq!(
        revision, EXPECTED_REVISION,
        "source revision does not match the pinned Gemma 4 E2B commit"
    );

    // ── 1. Identity preconditions ────────────────────────────────────────
    assert!(
        st_dir
            .join(".cache/huggingface/trees")
            .join(format!("{revision}.json"))
            .is_file(),
        "missing pinned Hugging Face snapshot manifest for {revision}"
    );
    let st_hash = file_sha256(&st_dir.join("model.safetensors")).1;
    assert_eq!(
        st_hash, EXPECTED_SAFETENSORS_SHA256,
        "source safetensors SHA-256 does not match the pinned canonical hash"
    );
    // Tokenizer/template resources must remain byte-identical between source and vindex.
    for name in RESOURCE_FILES {
        let (src_len, src_hash) = file_sha256(&st_dir.join(name));
        let (vix_len, vix_hash) = file_sha256(&vindex_dir.join(name));
        assert!(
            src_len == vix_len && src_hash == vix_hash,
            "resource {name} is not byte-identical between source and vindex"
        );
    }

    // ── 2. Read the committed ST3 token IDs (feed the SAME ids to both) ──
    let parity = load_json(&workspace_root().join(PARITY_ARTIFACT));
    let mut prompts: Vec<(String, Vec<u32>)> = Vec::new();
    for pid in PROMPT_ORDER {
        let fixture = parity["fixtures"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["prompt_id"].as_str() == Some(pid))
            .unwrap_or_else(|| panic!("ST3 parity artifact missing prompt {pid}"));
        let ids: Vec<u32> = fixture["larql_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        prompts.push((pid.to_string(), ids));
    }

    // ── 3. Load the oracle trace (pre-generated in a separate process) ──
    let oracle_manifest = TraceManifest::load(&oracle_dir)
        .expect("oracle trace must load; run scripts/gemma4_first_token_oracle.py first");
    assert_eq!(
        oracle_manifest.producer, "transformers-oracle",
        "oracle trace producer mismatch"
    );
    // Cross-check token-id parity: oracle and LARQL must feed the same ids.
    for (pid, ids) in &prompts {
        let oracle_ids: Vec<u32> = oracle_manifest.prompts[pid.as_str()].token_ids.to_vec();
        assert_eq!(
            oracle_ids, *ids,
            "oracle token ids differ from the committed ST3 ids for {pid}"
        );
    }

    // ── 4. Load the F32 reference vindex through the production loader ──
    let mut callbacks = SilentLoadCallbacks;
    let weights: ModelWeights =
        load_model_weights(&vindex_dir, &mut callbacks).expect("production F32 load must succeed");
    let num_layers = weights.num_layers;
    assert_eq!(num_layers, 35, "Gemma 4 E2B has 35 layers");

    // ── 5. Run the LARQL F32 trace capture ──────────────────────────────
    let larql_trace_dir = tempfile::tempdir().unwrap();
    let larql_manifest = write_larql_trace(
        &weights,
        &prompts,
        larql_trace_dir.path(),
        Some(serde_json::json!({
            "repository": "google/gemma-4-E2B-it",
            "revision": revision,
            "safetensors_sha256": st_hash,
            "vindex": vindex_dir,
        })),
    )
    .expect("LARQL trace capture must succeed");

    // ── 6. Compare against the committed policy ─────────────────────────
    let policy = Policy::st5_default();
    let result = compare_traces(
        &oracle_dir,
        &oracle_manifest,
        larql_trace_dir.path(),
        &larql_manifest,
        num_layers,
        &policy,
    );

    // ── 7. Write the report artifacts ───────────────────────────────────
    let report = build_report(
        &result,
        &revision,
        &st_hash,
        &vindex_dir,
        &oracle_manifest,
        &prompts,
    );
    let report_dir = workspace_root().join("bench/baselines");
    let json_path = report_dir.join("gemma4-e2b-f32-first-token-parity-2026-07-12.json");
    let md_path = report_dir.join("gemma4-e2b-f32-first-token-parity-2026-07-12.md");
    fs::create_dir_all(&report_dir).unwrap();
    fs::write(
        &json_path,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
    fs::write(&md_path, render_markdown(&report)).unwrap();

    println!(
        "=== ST5 FIRST-TOKEN PARITY ===\ndecision: {:?}\n{}",
        result.decision,
        serde_json::to_string_pretty(&report).unwrap()
    );

    // ── 8. Decision gate ────────────────────────────────────────────────
    assert_eq!(
        format!("{:?}", result.decision),
        "Green".to_string(),
        "ST5 first-token parity was not GREEN; see report"
    );
}

fn build_report(
    result: &larql_inference::parity::ParityResult,
    revision: &str,
    st_hash: &str,
    vindex_dir: &Path,
    oracle: &TraceManifest,
    prompts: &[(String, Vec<u32>)],
) -> Value {
    let per_prompt: Vec<Value> = prompts
        .iter()
        .filter_map(|(pid, ids)| {
            let p = result.prompts.iter().find(|p| &p.prompt_id == pid)?;
            let logit_stage = p.stages.iter().find(|s| s.stage == "final_logits");
            let logit_metrics = logit_stage.map(|s| {
                serde_json::json!({
                    "max_abs": s.max_abs,
                    "nrmse": s.nrmse,
                    "cosine": s.cosine,
                    "mean_abs": s.mean_abs,
                })
            });

            // Per-layer coarse summary: the worst (max) nrmse across the five
            // per-layer stages at each layer — a compact view that still lets
            // a reader see no layer drifts.
            let mut by_layer: std::collections::BTreeMap<usize, Value> =
                std::collections::BTreeMap::new();
            for s in &p.stages {
                if let Some(layer) = s.layer {
                    let entry = by_layer.entry(layer).or_insert_with(
                        || serde_json::json!({"max_nrmse": 0.0, "max_abs": 0.0, "min_cos": 1.0}),
                    );
                    if s.nrmse > entry["max_nrmse"].as_f64().unwrap_or(0.0) {
                        entry["max_nrmse"] = serde_json::json!(s.nrmse);
                    }
                    if s.max_abs > entry["max_abs"].as_f64().unwrap_or(0.0) {
                        entry["max_abs"] = serde_json::json!(s.max_abs);
                    }
                    if s.cosine < entry["min_cos"].as_f64().unwrap_or(1.0) {
                        entry["min_cos"] = serde_json::json!(s.cosine);
                    }
                }
            }
            let worst_layer_nrmse = by_layer
                .values()
                .map(|v| v["max_nrmse"].as_f64().unwrap_or(0.0))
                .fold(0.0f64, f64::max);

            Some(serde_json::json!({
                "prompt_id": pid,
                "seq_len": ids.len(),
                "token_ids": ids,
                "passed": p.passed,
                "first_divergence": p.first_divergence.as_ref().map(|f| serde_json::json!({
                    "stage": f.stage, "layer": f.layer, "reason": f.reason,
                })),
                "final_logits": logit_metrics,
                "logits_top1": p.logits.as_ref().map(|l| serde_json::json!({
                    "reference": l.reference_top1, "candidate": l.candidate_top1,
                    "exact": l.top1_exact, "top10_overlap": l.top10_overlap,
                    "reference_top10": l.reference_top10, "candidate_top10": l.candidate_top10,
                })),
                "per_layer_coarse_summary": by_layer.values().cloned().collect::<Vec<_>>(),
                "worst_layer_nrmse": worst_layer_nrmse,
            }))
        })
        .collect();

    let (work_start_sha, head_sha) = git_shas();

    serde_json::json!({
        "schema_version": 1,
        "slice_id": "LARQL-INFERENCE-TRUST-001A-ST5",
        "decision": format!("{:?}", result.decision),
        "work_start_sha": work_start_sha,
        "pr_base_sha": "c023735f6afe3221e5989678918961e855cd2bda",
        "head_sha": head_sha,
        "source": {
            "repository": "google/gemma-4-E2B-it",
            "revision": revision,
            "safetensors_sha256": st_hash,
        },
        "f32_vindex": {
            "dir": vindex_dir,
            "loader": "larql_vindex::load_model_weights (production F32 loader)",
            "identity": "LARQL-INFERENCE-TRUST-001A-ST2 lossless reference artifact",
        },
        "oracle": {
            "producer": oracle.producer,
            "environment": oracle.environment,
            "model": oracle.model,
            "path": "scripts/gemma4_first_token_oracle.py",
        },
        "comparison_policy": result.policy,
        "prompts": per_prompt,
        "corrections_made": [{
            "defect": "Gemma 4 global (full_attention) layers used the wrong RoPE frequency mode",
            "root_cause": "HF rope_type='proportional' computes inv_freq exponents over the full head_dim (512) and zero-pads to head_dim/2, then half-splits over the full head; LARQL divided exponents by rotary_dim (128) and half-split within rotary_dim. Sliding layers (rope_type='default', full rotary) were unaffected, so the divergence first appeared at layer 4 (the first global layer).",
            "fix": "Added RopeFreqMode::{Standard,Proportional} to larql-models; Gemma4 arch returns Proportional for global layers; larql-compute rope builds the zero-padded head_dim/2 inv_freq and half-splits over the full head for Proportional mode. Applied consistently to the CPU block, decode, and kv-prefill (gpu.rs CPU fallback) paths.",
            "files": [
                "crates/larql-models/src/config.rs (RopeFreqMode + trait method)",
                "crates/larql-models/src/architectures/gemma4.rs (override)",
                "crates/larql-compute/src/attention/rope.rs (mode-aware builder + applier)",
                "crates/larql-compute/src/attention/block.rs, decode.rs, gpu.rs (call sites)",
            ],
            "regression_tests": [
                "larql-compute rope::tests: proportional_inv_freq_matches_hf_gemma4_global",
                "larql-compute rope::tests: proportional_rope_pairs_full_head_dim_not_rotary_dim",
                "larql-models detect::tests: test_detect_gemma4_proportional_rope_mode_for_global_layers",
                "ST4/ST4A local/global + shared-KV regressions remain green",
            ],
        }],
        "first_token_semantic_parity_test": "cargo test -p larql-inference --test gemma4_first_token_parity --release -- --ignored --nocapture (env-gated; soft-skips without the 18.65 GB artifact)",
        "test_commands": {
            "fmt": "cargo fmt --all -- --check",
            "tests": "cargo test -p larql-models / larql-compute / larql-inference / larql-kv / larql-vindex",
            "clippy": "cargo clippy -p {larql-models,larql-compute,larql-inference,larql-kv,larql-vindex} --all-targets -- -D warnings",
            "diagnostic_self_tests": "cargo test -p larql-inference --test parity_diagnostics (9 cases, section 7)",
        },
        "scope_exclusions": [
            "multi-token generation", "sampling", "KV-cached decode", "Q4_K",
            "CUDA", "Vulkan", "Metal", "performance optimization",
            "production quantization", "multimodal inputs",
            "tools or thinking-enabled prompts",
        ],
        "recommended_next_slice": if result.decision == larql_inference::parity::Decision::Green {
            "LARQL-INFERENCE-TRUST-001A-ST6 (Production Q4_K semantic parity against the proven F32 reference)"
        } else {
            "LARQL-INFERENCE-TRUST-001A-ST5A (narrow correction targeting the recorded first divergence)"
        },
    })
}

fn git_shas() -> (String, String) {
    let work_start = String::from_utf8(
        std::process::Command::new("git")
            .args(["rev-parse", "c023735f6afe3221e5989678918961e855cd2bda"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    Some(o.stdout)
                } else {
                    None
                }
            })
            .unwrap_or_default(),
    )
    .unwrap_or_default()
    .trim()
    .to_string();
    let head = String::from_utf8(
        std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    Some(o.stdout)
                } else {
                    None
                }
            })
            .unwrap_or_default(),
    )
    .unwrap_or_default()
    .trim()
    .to_string();
    (work_start, head)
}

fn render_markdown(report: &Value) -> String {
    let mut out = String::new();
    out.push_str("# LARQL-INFERENCE-TRUST-001A-ST5 — F32 First-Token Semantic Parity\n\n");
    out.push_str(&format!("- **Slice:** {}\n", report["slice_id"]));
    out.push_str(&format!("- **Decision:** {}\n", report["decision"]));
    out.push_str(&format!(
        "- **Source:** `{}` @ `{}`\n",
        report["source"]["repository"], report["source"]["revision"]
    ));
    out.push_str(&format!(
        "- **Safetensors SHA-256:** `{}`\n",
        report["source"]["safetensors_sha256"]
    ));
    out.push_str(&format!(
        "- **Oracle:** `{}`\n",
        report["oracle"]["producer"]
    ));
    out.push_str(&format!(
        "- **Policy:** coarse nrmse≤{}, final logits nrmse≤{}, top-10 overlap≥{}\n\n",
        report["comparison_policy"]["coarse_nrmse"],
        report["comparison_policy"]["logits_nrmse"],
        report["comparison_policy"]["logits_top10_overlap_min"]
    ));
    out.push_str("## Per-prompt results\n\n");
    out.push_str("| prompt | seq_len | passed | top-1 ref | top-1 cand | top-10 overlap | logit max_abs | logit nrmse | logit cosine | worst layer nrmse | first divergence |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for p in report["prompts"].as_array().unwrap() {
        let fd = match &p["first_divergence"] {
            Value::Null => "null".to_string(),
            f => format!("{}@{:?}", f["stage"], f["layer"]),
        };
        let top1 = &p["logits_top1"];
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            p["prompt_id"],
            p["seq_len"],
            p["passed"],
            top1["reference"].as_u64().unwrap_or(0),
            top1["candidate"].as_u64().unwrap_or(0),
            top1["top10_overlap"].as_u64().unwrap_or(0),
            fmt_f64(&p["final_logits"]["max_abs"]),
            fmt_f64(&p["final_logits"]["nrmse"]),
            fmt_f64(&p["final_logits"]["cosine"]),
            fmt_f64(&p["worst_layer_nrmse"]),
            fd,
        ));
    }
    out.push_str("\n## Corrections made\n\n");
    for c in report["corrections_made"].as_array().unwrap() {
        out.push_str(&format!(
            "### {}\n\n- **Root cause:** {}\n- **Fix:** {}\n\n",
            c["defect"], c["root_cause"], c["fix"]
        ));
    }
    out.push_str("\n## Scope exclusions\n\n");
    for s in report["scope_exclusions"].as_array().unwrap() {
        out.push_str(&format!("- {}\n", s.as_str().unwrap()));
    }
    out.push_str(&format!(
        "\n## Recommended next slice\n\n{}\n",
        report["recommended_next_slice"].as_str().unwrap()
    ));
    out
}

fn fmt_f64(v: &Value) -> String {
    match v {
        Value::Null => "n/a".to_string(),
        _ => format!("{:.3e}", v.as_f64().unwrap_or(0.0)),
    }
}

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

fn file_sha256(path: &Path) -> (u64, String) {
    let mut file = fs::File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    let mut len = 0u64;
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        len += n as u64;
        hasher.update(&buf[..n]);
    }
    (len, format!("{:x}", hasher.finalize()))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf()
}

fn load_json(path: &Path) -> Value {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("JSON must parse")
}

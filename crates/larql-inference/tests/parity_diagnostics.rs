//! ST5 §7 — diagnostic self-tests for the first-divergence machinery.
//!
//! These prove the comparator independently of the official-model run. They
//! build synthetic trace directories in the ST5 interchange format, inject
//! controlled differences, and assert the comparator returns GREEN, localises
//! the exact stage, and respects execution-order precedence. They run in
//! ordinary CI (no 18 GB artifact required).
//!
//! Run: `cargo test -p larql-inference --test parity_diagnostics`

use std::collections::BTreeMap;
use std::path::Path;

use larql_inference::parity::{
    coarse_stage_order, compare_traces, entry_at, Policy, TraceManifest, TracePrompt, TraceTensor,
    STAGE_EMBEDDING, STAGE_FINAL_LOGITS, STAGE_FINAL_NORM, STAGE_LAYER_INPUT, STAGE_LM_HEAD_RAW,
    STAGE_POST_ATTENTION, STAGE_POST_FFN, STAGE_POST_LAYER, STAGE_POST_PLE, STAGE_PRE_FINAL_NORM,
};

const NUM_LAYERS: usize = 4;
const HIDDEN: usize = 16;
const VOCAB: usize = 24;

/// A synthetic "green" stage value generator: deterministic, non-trivial,
/// finite. Different per stage so each stage is distinguishable.
fn base_value(stage_idx: usize, i: usize) -> f32 {
    let s = stage_idx as f32;
    (0.1 * s + 0.01 * i as f32 + 1.0) * if i.is_multiple_of(2) { 1.0 } else { -0.5 }
}

/// Names of every coarse stage in execution order (with layer).
fn all_stages() -> Vec<(&'static str, Option<usize>)> {
    coarse_stage_order(NUM_LAYERS)
        .into_iter()
        .map(|s| {
            let st: &'static str = match s.stage.as_str() {
                STAGE_EMBEDDING => STAGE_EMBEDDING,
                STAGE_LAYER_INPUT => STAGE_LAYER_INPUT,
                STAGE_POST_ATTENTION => STAGE_POST_ATTENTION,
                STAGE_POST_FFN => STAGE_POST_FFN,
                STAGE_POST_PLE => STAGE_POST_PLE,
                STAGE_POST_LAYER => STAGE_POST_LAYER,
                STAGE_PRE_FINAL_NORM => STAGE_PRE_FINAL_NORM,
                STAGE_FINAL_NORM => STAGE_FINAL_NORM,
                STAGE_LM_HEAD_RAW => STAGE_LM_HEAD_RAW,
                STAGE_FINAL_LOGITS => STAGE_FINAL_LOGITS,
                _ => STAGE_EMBEDDING,
            };
            (st, s.layer)
        })
        .collect()
}

fn stage_len(stage: &str) -> usize {
    if stage == STAGE_FINAL_LOGITS || stage == STAGE_LM_HEAD_RAW {
        VOCAB
    } else {
        HIDDEN
    }
}

/// Perturbation: map from (stage, layer) -> closure that mutates the value vec.
type Perturb = Box<dyn Fn(&mut Vec<f32>)>;

/// Build a green baseline prompt at `dir/prompt_id`.
fn write_prompt_with_tensors(
    dir: &Path,
    prompt_id: &str,
    perturbs: &[((&'static str, Option<usize>), Perturb)],
) -> TracePrompt {
    let mut tensors: Vec<TraceTensor> = Vec::new();
    for (idx, (stage, layer)) in all_stages().into_iter().enumerate() {
        let len = stage_len(stage);
        let mut values: Vec<f32> = (0..len).map(|i| base_value(idx, i)).collect();
        for ((s, l), f) in perturbs {
            if (*s == stage) && (*l == layer) {
                f(&mut values);
            }
        }
        let layer_tag = layer.map(|l| format!("_{l}")).unwrap_or_default();
        let rel = format!("{prompt_id}/{stage}{layer_tag}.f32");
        let (entry, _path) = entry_at(dir, &rel, stage, layer, &values).unwrap();
        tensors.push(entry);
    }
    TracePrompt {
        token_ids: vec![2, 818, 5279],
        seq_len: 3,
        tensors,
    }
}

fn write_manifest(dir: &Path, producer: &str, prompts: BTreeMap<String, TracePrompt>) {
    let manifest = TraceManifest {
        schema_version: 1,
        producer: producer.to_string(),
        environment: None,
        model: None,
        prompts,
    };
    manifest.write(dir).unwrap();
}

fn build_pair(
    ref_perturbs: &[((&'static str, Option<usize>), Perturb)],
    cand_perturbs: &[((&'static str, Option<usize>), Perturb)],
) -> (tempfile::TempDir, tempfile::TempDir) {
    let ref_dir = tempfile::tempdir().unwrap();
    let cand_dir = tempfile::tempdir().unwrap();
    let mut ref_prompts = BTreeMap::new();
    ref_prompts.insert(
        "p0".to_string(),
        write_prompt_with_tensors(ref_dir.path(), "p0", ref_perturbs),
    );
    write_manifest(ref_dir.path(), "synthetic-reference", ref_prompts);
    let mut cand_prompts = BTreeMap::new();
    cand_prompts.insert(
        "p0".to_string(),
        write_prompt_with_tensors(cand_dir.path(), "p0", cand_perturbs),
    );
    write_manifest(cand_dir.path(), "synthetic-candidate", cand_prompts);
    (ref_dir, cand_dir)
}

fn load_manifest(dir: &Path) -> TraceManifest {
    TraceManifest::load(dir).unwrap()
}

fn run(ref_dir: &Path, cand_dir: &Path) -> larql_inference::parity::ParityResult {
    let reference = load_manifest(ref_dir);
    let candidate = load_manifest(cand_dir);
    let result = compare_traces(
        ref_dir,
        &reference,
        cand_dir,
        &candidate,
        NUM_LAYERS,
        &Policy::st5_default(),
    );
    if std::env::var_os("ST5_DEBUG").is_some() {
        eprintln!(
            "DEBUG prompts={} decision={:?}",
            result.prompts.len(),
            result.decision
        );
        for p in &result.prompts {
            eprintln!(
                "[{}] passed={} integrity={:?} first={:?}",
                p.prompt_id,
                p.passed,
                p.integrity_errors,
                p.first_divergence
                    .as_ref()
                    .map(|f| (&f.stage, f.layer, &f.reason))
            );
            for s in p.stages.iter().filter(|s| !s.passed).take(3) {
                eprintln!(
                    "    FAIL stage={} layer={:?} reason={}",
                    s.stage,
                    s.layer,
                    s.failure_reason.as_deref().unwrap_or("?")
                );
            }
        }
    }
    result
}

// Case 1: identical traces → GREEN.
#[test]
fn case1_identical_traces_are_green() {
    let (r, c) = build_pair(&[], &[]);
    let result = run(r.path(), c.path());
    assert_eq!(result.decision, larql_inference::parity::Decision::Green);
    let p = &result.prompts[0];
    assert!(p.first_divergence.is_none());
    assert!(p.stages.iter().all(|s| s.passed));
}

// Case 2: inject a difference at a known layer's post-attention stage and
// identify it exactly.
#[test]
fn case2_localises_post_attention_spike() {
    // Poison post_attention@2 only; everything else green.
    let spike: Perturb = Box::new(|v| v[3] += 100.0);
    let (r, c) = build_pair(&[], &[((STAGE_POST_ATTENTION, Some(2)), spike)]);
    let result = run(r.path(), c.path());
    assert_eq!(result.decision, larql_inference::parity::Decision::Red);
    let p = &result.prompts[0];
    let fd = p.first_divergence.as_ref().expect("first divergence");
    assert_eq!(fd.stage, STAGE_POST_ATTENTION);
    assert_eq!(fd.layer, Some(2));
}

// Case 3: inject a later FFN difference while an earlier stage remains equal.
// Earliest failure must win (covered more in case 8); here we confirm a lone
// later FFN diff is reported at that FFN stage with all earlier stages green.
#[test]
fn case3_lone_ffn_difference_reported() {
    let spike: Perturb = Box::new(|v| v[0] += 100.0);
    let (r, c) = build_pair(&[], &[((STAGE_POST_FFN, Some(3)), spike)]);
    let result = run(r.path(), c.path());
    assert_eq!(result.decision, larql_inference::parity::Decision::Red);
    let p = &result.prompts[0];
    let fd = p.first_divergence.as_ref().unwrap();
    assert_eq!(fd.stage, STAGE_POST_FFN);
    assert_eq!(fd.layer, Some(3));
    // Every earlier stage passed.
    let earlier_green = p
        .stages
        .iter()
        .filter(|s| {
            // stages before post_ffn@3 in execution order
            stage_precedes(&s.stage, s.layer, STAGE_POST_FFN, Some(3))
        })
        .all(|s| s.passed);
    assert!(earlier_green);
}

// Case 4: detect shape and missing-stage failures.
#[test]
fn case4_missing_stage_detected() {
    let (r, c) = build_pair(&[], &[]);
    // Delete one stage file + entry from the candidate manifest to simulate
    // a missing required stage.
    let mut candidate = load_manifest(c.path());
    let prompt = candidate.prompts.get_mut("p0").unwrap();
    prompt
        .tensors
        .retain(|t| !(t.stage == STAGE_POST_PLE && t.layer == Some(1)));
    write_manifest(
        c.path(),
        "synthetic-candidate",
        std::mem::take(&mut candidate.prompts),
    );
    let result = run(r.path(), c.path());
    assert_eq!(result.decision, larql_inference::parity::Decision::Red);
    assert!(result.prompts[0]
        .integrity_errors
        .iter()
        .any(|e| e.contains("missing required stage") && e.contains("post_ple")));
}

#[test]
fn case4_shape_mismatch_detected() {
    // Make candidate embedding a different length than reference.
    let wrong_shape: Perturb = Box::new(|v| v.truncate(HIDDEN / 2));
    // Also pad-back so len differs cleanly. truncate already changes len.
    let (r, c) = build_pair(&[], &[((STAGE_EMBEDDING, None), wrong_shape)]);
    let result = run(r.path(), c.path());
    assert_eq!(result.decision, larql_inference::parity::Decision::Red);
    let emb = result.prompts[0]
        .stages
        .iter()
        .find(|s| s.stage == STAGE_EMBEDDING)
        .unwrap();
    assert!(!emb.passed);
    assert!(emb
        .failure_reason
        .as_deref()
        .unwrap()
        .contains("shape mismatch"));
}

// Case 5: detect non-finite values.
#[test]
fn case5_non_finite_detected() {
    let nan: Perturb = Box::new(|v| v[1] = f32::NAN);
    let (r, c) = build_pair(&[((STAGE_FINAL_NORM, None), nan)], &[]);
    let result = run(r.path(), c.path());
    assert_eq!(result.decision, larql_inference::parity::Decision::Red);
    assert!(result.prompts[0]
        .integrity_errors
        .iter()
        .any(|e| e.contains("non-finite") && e.contains("final_norm")));
}

// Case 6: shared-KV consumer with K/V stages marked not-executed.
// The coarse residuals (Q/O derived) ARE executed; a synthetic not_executed
// marker on a consumer K/V stage must compare equal (both empty).
#[test]
fn case6_shared_kv_not_executed_equal() {
    // Build a custom pair where we inject a not_executed consumer KV stage
    // into BOTH manifests at the same identity. The comparator must treat
    // them as equal (no divergence).
    let ref_dir = tempfile::tempdir().unwrap();
    let cand_dir = tempfile::tempdir().unwrap();
    for dir in [ref_dir.path(), cand_dir.path()] {
        let prompt = write_prompt_with_tensors(dir, "p0", &[]);
        // Append a not_executed consumer KV stage to both.
        let mut prompt = prompt;
        prompt.tensors.push(TraceTensor {
            stage: "consumer_kv".to_string(),
            layer: Some(3),
            shape: vec![0],
            dtype: "f32".to_string(),
            element_count: 0,
            filename: "unused".to_string(),
            sha256: "0".repeat(64),
            not_executed: true,
        });
        let mut map = BTreeMap::new();
        map.insert("p0".to_string(), prompt);
        write_manifest(dir, "synthetic", map);
    }
    let result = run(ref_dir.path(), cand_dir.path());
    // The required-coarse check passes; the not_executed stage is skipped.
    assert_eq!(result.decision, larql_inference::parity::Decision::Green);
}

// Case 7: identify final-norm and lm-head divergences separately.
#[test]
fn case7_final_norm_and_lm_head_diverge_separately() {
    // Poison final_norm only → first divergence is final_norm.
    let spike: Perturb = Box::new(|v| v[5] += 50.0);
    let (r, c) = build_pair(&[], &[((STAGE_FINAL_NORM, None), spike)]);
    let result = run(r.path(), c.path());
    let fd = result.prompts[0].first_divergence.as_ref().unwrap();
    assert_eq!(fd.stage, STAGE_FINAL_NORM);
    assert_ne!(fd.stage, STAGE_LM_HEAD_RAW);

    // Poison lm_head_raw only → first divergence is lm_head_raw (final_norm green).
    let spike2: Perturb = Box::new(|v| v[2] += 50.0);
    let (r2, c2) = build_pair(&[], &[((STAGE_LM_HEAD_RAW, None), spike2)]);
    let result2 = run(r2.path(), c2.path());
    let fd2 = result2.prompts[0].first_divergence.as_ref().unwrap();
    assert_eq!(fd2.stage, STAGE_LM_HEAD_RAW);
}

// Case 8: earliest failure wins when several stages are poisoned.
#[test]
fn case8_earliest_failure_wins() {
    // Poison embedding, post_layer@1, and final_logits. Earliest = embedding.
    let spike1: Perturb = Box::new(|v| v[0] += 100.0);
    let spike2: Perturb = Box::new(|v| v[0] += 100.0);
    let spike3: Perturb = Box::new(|v| v[0] += 100.0);
    let (r, c) = build_pair(
        &[],
        &[
            ((STAGE_EMBEDDING, None), spike1),
            ((STAGE_POST_LAYER, Some(1)), spike2),
            ((STAGE_FINAL_LOGITS, None), spike3),
        ],
    );
    let result = run(r.path(), c.path());
    let fd = result.prompts[0].first_divergence.as_ref().unwrap();
    assert_eq!(fd.stage, STAGE_EMBEDDING, "earliest failure must win");
}

/// Execution-order precedence helper.
fn stage_precedes(
    a_stage: &str,
    a_layer: Option<usize>,
    b_stage: &str,
    b_layer: Option<usize>,
) -> bool {
    let order = coarse_stage_order(NUM_LAYERS);
    let pos = |stage: &str, layer: Option<usize>| {
        order
            .iter()
            .position(|s| s.stage == stage && s.layer == layer)
    };
    match (pos(a_stage, a_layer), pos(b_stage, b_layer)) {
        (Some(pa), Some(pb)) => pa < pb,
        _ => false,
    }
}

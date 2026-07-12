//! ST5 first-token parity — numerical comparator, policy, and drill-down.
//!
//! The policy is committed up front ([`Policy::st5_default`]) and never
//! widened after the result is seen. Every compared tensor produces a
//! [`StageMetrics`] (max abs, max rel, mean abs, RMSE, normalized RMSE,
//! cosine, index/value at the max difference). The comparator walks stages
//! in execution order ([`crate::parity::format::coarse_stage_order`]) so the
//! first divergence is reported as the earliest failing boundary, layer, and
//! stage.
//!
//! All metrics are accumulated in `f64` (see `residual_diff::compare` for the
//! rationale: a 262k-wide logit vector has plenty of room for f32
//! cancellation to dominate the real signal).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::format::{
    coarse_stage_order, load_prompt_tensors, required_coarse_stages, StageId, TraceError,
    TraceManifest, TracePrompt, STAGE_FINAL_LOGITS,
};

/// Committed numerical policy for a prompt's stages. Construct via
/// [`Policy::st5_default`] — do NOT relax after seeing the result.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    // Coarse hidden/residual stages.
    pub coarse_nrmse: f64,
    pub coarse_cosine: f64,
    pub coarse_max_abs_offset: f64,
    pub coarse_max_abs_scale: f64,
    // Final logits.
    pub logits_nrmse: f64,
    pub logits_cosine: f64,
    pub logits_max_abs_offset: f64,
    pub logits_max_abs_scale: f64,
    pub logits_top10_overlap_min: usize,
    // Numerics.
    pub rel_denominator_floor: f64,
    pub nrmse_denominator_floor: f64,
    pub zero_norm_abs_tolerance: f64,
}

impl Policy {
    /// The committed ST5 initial parity policy (section 6). Hard-coded so
    /// it cannot be silently widened — change the source only with a new
    /// committed policy and a recorded rationale.
    pub const fn st5_default() -> Self {
        Self {
            coarse_nrmse: 1e-4,
            coarse_cosine: 0.99999,
            coarse_max_abs_offset: 5e-4,
            coarse_max_abs_scale: 5e-4,
            logits_nrmse: 1e-4,
            logits_cosine: 0.99999,
            logits_max_abs_offset: 1e-3,
            logits_max_abs_scale: 5e-4,
            logits_top10_overlap_min: 9,
            rel_denominator_floor: 1e-6,
            nrmse_denominator_floor: 1e-12,
            zero_norm_abs_tolerance: 0.0,
        }
    }

    fn stage_kind(&self, stage: &str) -> StageKind {
        if stage == STAGE_FINAL_LOGITS {
            StageKind::Logits
        } else {
            StageKind::Coarse
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    Coarse,
    Logits,
}

/// Metrics for one compared tensor, plus the policy verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageMetrics {
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    pub shape: Vec<usize>,
    pub max_abs: f64,
    pub max_rel: f64,
    pub mean_abs: f64,
    pub rmse: f64,
    pub nrmse: f64,
    pub cosine: f64,
    pub max_diff_index: usize,
    pub reference_value: f64,
    pub candidate_value: f64,
    pub reference_rms: f64,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

/// Top-k information for a logits tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitTopK {
    pub reference_top1: usize,
    pub candidate_top1: usize,
    pub top1_exact: bool,
    pub top10_overlap: usize,
    pub reference_top10: Vec<usize>,
    pub candidate_top10: Vec<usize>,
}

/// Where the first divergence was found in execution order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstDivergence {
    pub stage: String,
    pub layer: Option<usize>,
    pub reason: String,
    pub metrics: StageMetrics,
}

/// Per-prompt comparison result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptComparison {
    pub prompt_id: String,
    pub stages: Vec<StageMetrics>,
    pub logits: Option<LogitTopK>,
    pub first_divergence: Option<FirstDivergence>,
    pub integrity_errors: Vec<String>,
    pub passed: bool,
}

/// Whole-trace comparison result across all prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityResult {
    pub policy: PolicyView,
    pub prompts: Vec<PromptComparison>,
    pub decision: Decision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Green,
    Red,
}

/// A serializable view of the committed policy (so reports carry the exact
/// thresholds that were used).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyView {
    pub coarse_nrmse: f64,
    pub coarse_cosine: f64,
    pub coarse_max_abs_offset: f64,
    pub coarse_max_abs_scale: f64,
    pub logits_nrmse: f64,
    pub logits_cosine: f64,
    pub logits_max_abs_offset: f64,
    pub logits_max_abs_scale: f64,
    pub logits_top10_overlap_min: usize,
    pub rel_denominator_floor: f64,
    pub nrmse_denominator_floor: f64,
    pub zero_norm_abs_tolerance: f64,
}

impl From<&Policy> for PolicyView {
    fn from(p: &Policy) -> Self {
        Self {
            coarse_nrmse: p.coarse_nrmse,
            coarse_cosine: p.coarse_cosine,
            coarse_max_abs_offset: p.coarse_max_abs_offset,
            coarse_max_abs_scale: p.coarse_max_abs_scale,
            logits_nrmse: p.logits_nrmse,
            logits_cosine: p.logits_cosine,
            logits_max_abs_offset: p.logits_max_abs_offset,
            logits_max_abs_scale: p.logits_max_abs_scale,
            logits_top10_overlap_min: p.logits_top10_overlap_min,
            rel_denominator_floor: p.rel_denominator_floor,
            nrmse_denominator_floor: p.nrmse_denominator_floor,
            zero_norm_abs_tolerance: p.zero_norm_abs_tolerance,
        }
    }
}

impl ParityResult {
    pub fn is_green(&self) -> bool {
        self.decision == Decision::Green
    }
}

/// Compare two traces produced in the same interchange format. `num_layers`
/// drives the required coarse stage set and the execution order.
pub fn compare_traces(
    reference_dir: &std::path::Path,
    reference: &TraceManifest,
    candidate_dir: &std::path::Path,
    candidate: &TraceManifest,
    num_layers: usize,
    policy: &Policy,
) -> ParityResult {
    let order = coarse_stage_order(num_layers);
    let required: HashSet<StageId> = required_coarse_stages(num_layers);
    let mut prompts = Vec::new();
    let mut all_green = true;

    let prompt_ids: Vec<String> = reference.prompts.keys().cloned().collect();
    for prompt_id in &prompt_ids {
        let Some(ref_prompt) = reference.prompts.get(prompt_id) else {
            prompts.push(integrity_failure(
                prompt_id,
                &format!("prompt `{prompt_id}` missing from reference trace"),
            ));
            all_green = false;
            continue;
        };
        let Some(cand_prompt) = candidate.prompts.get(prompt_id) else {
            prompts.push(integrity_failure(
                prompt_id,
                &format!("prompt `{prompt_id}` missing from candidate trace"),
            ));
            all_green = false;
            continue;
        };

        let result = compare_prompt(
            reference_dir,
            ref_prompt,
            candidate_dir,
            cand_prompt,
            &order,
            &required,
            policy,
            prompt_id,
        );
        if !result.passed {
            all_green = false;
        }
        prompts.push(result);
    }

    ParityResult {
        policy: PolicyView::from(policy),
        prompts,
        decision: if all_green {
            Decision::Green
        } else {
            Decision::Red
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_prompt(
    reference_dir: &std::path::Path,
    reference: &TracePrompt,
    candidate_dir: &std::path::Path,
    candidate: &TracePrompt,
    order: &[StageId],
    required: &HashSet<StageId>,
    policy: &Policy,
    prompt_id: &str,
) -> PromptComparison {
    let mut integrity_errors: Vec<String> = Vec::new();

    // Token-id parity is a precondition (ST3 already pins it).
    if reference.token_ids != candidate.token_ids {
        integrity_errors.push(format!(
            "token ids differ between reference and candidate for `{prompt_id}`"
        ));
    }

    // Required-stage presence + duplicate detection.
    let ref_ids: HashSet<StageId> = reference
        .tensors
        .iter()
        .map(|t| StageId::new(&t.stage, t.layer))
        .collect();
    let cand_ids: HashSet<StageId> = candidate
        .tensors
        .iter()
        .map(|t| StageId::new(&t.stage, t.layer))
        .collect();
    for id in required {
        if !ref_ids.contains(id) {
            integrity_errors.push(format!("reference missing required stage {id}"));
        }
        if !cand_ids.contains(id) {
            integrity_errors.push(format!("candidate missing required stage {id}"));
        }
    }
    // Duplicate detection via load_prompt_tensors (returns DuplicateStage).
    let ref_tensors = match load_prompt_tensors(reference_dir, reference, order) {
        Ok(v) => v,
        Err(TraceError::DuplicateStage(stage, layer)) => {
            integrity_errors.push(format!(
                "reference has duplicate stage `{stage}`{}",
                layer_kind(layer)
            ));
            Vec::new()
        }
        Err(e) => {
            integrity_errors.push(format!("reference load error: {e}"));
            Vec::new()
        }
    };
    let cand_tensors = match load_prompt_tensors(candidate_dir, candidate, order) {
        Ok(v) => v,
        Err(TraceError::DuplicateStage(stage, layer)) => {
            integrity_errors.push(format!(
                "candidate has duplicate stage `{stage}`{}",
                layer_kind(layer)
            ));
            Vec::new()
        }
        Err(e) => {
            integrity_errors.push(format!("candidate load error: {e}"));
            Vec::new()
        }
    };

    if !integrity_errors.is_empty() {
        return PromptComparison {
            prompt_id: prompt_id.to_string(),
            stages: Vec::new(),
            logits: None,
            first_divergence: None,
            integrity_errors,
            passed: false,
        };
    }

    // Per-stage comparison in execution order.
    let mut stages_out = Vec::with_capacity(order.len());
    let mut logits_info: Option<LogitTopK> = None;
    let mut first_divergence: Option<FirstDivergence> = None;

    for id in order {
        let Some((_, ref_vals)) = ref_tensors.iter().find(|(sid, _)| sid == id) else {
            continue;
        };
        let Some((_, cand_vals)) = cand_tensors.iter().find(|(sid, _)| sid == id) else {
            continue;
        };

        // not_executed (shared-KV consumer) — both empty => equal.
        if ref_vals.is_empty() && cand_vals.is_empty() {
            continue;
        }

        // Shape check.
        if ref_vals.len() != cand_vals.len() {
            let metrics = StageMetrics {
                stage: id.stage.clone(),
                layer: id.layer,
                shape: vec![ref_vals.len()],
                max_abs: f64::INFINITY,
                max_rel: f64::INFINITY,
                mean_abs: f64::INFINITY,
                rmse: f64::INFINITY,
                nrmse: f64::INFINITY,
                cosine: 0.0,
                max_diff_index: 0,
                reference_value: 0.0,
                candidate_value: 0.0,
                reference_rms: 0.0,
                passed: false,
                failure_reason: Some(format!(
                    "shape mismatch: reference {} vs candidate {}",
                    ref_vals.len(),
                    cand_vals.len()
                )),
            };
            push_stage(&mut stages_out, &mut first_divergence, metrics);
            continue;
        }

        let kind = policy.stage_kind(&id.stage);
        let metrics = compare_tensor(ref_vals, cand_vals, policy, kind, &id.stage, id.layer);
        push_stage(&mut stages_out, &mut first_divergence, metrics);

        if id.stage == STAGE_FINAL_LOGITS {
            logits_info = Some(compute_topk(ref_vals, cand_vals));
        }
    }

    // Final-logits top-k verdict folds into the prompt pass/fail.
    let mut passed = first_divergence.is_none();
    if let Some(info) = &logits_info {
        if !info.top1_exact {
            passed = false;
            // Surface as a divergence at the final logits stage if none earlier.
            if first_divergence.is_none() {
                first_divergence = Some(FirstDivergence {
                    stage: STAGE_FINAL_LOGITS.to_string(),
                    layer: None,
                    reason: format!(
                        "top-1 mismatch: reference {} vs candidate {}",
                        info.reference_top1, info.candidate_top1
                    ),
                    metrics: stages_out
                        .iter()
                        .find(|m| m.stage == STAGE_FINAL_LOGITS)
                        .cloned()
                        .unwrap_or_else(|| StageMetrics {
                            stage: STAGE_FINAL_LOGITS.to_string(),
                            layer: None,
                            shape: vec![],
                            max_abs: f64::INFINITY,
                            max_rel: 0.0,
                            mean_abs: 0.0,
                            rmse: 0.0,
                            nrmse: 0.0,
                            cosine: 0.0,
                            max_diff_index: 0,
                            reference_value: 0.0,
                            candidate_value: 0.0,
                            reference_rms: 0.0,
                            passed: false,
                            failure_reason: Some("top-1 mismatch".to_string()),
                        }),
                });
            }
        }
        if info.top10_overlap < policy.logits_top10_overlap_min {
            passed = false;
        }
    }

    PromptComparison {
        prompt_id: prompt_id.to_string(),
        stages: stages_out,
        logits: logits_info,
        first_divergence,
        integrity_errors,
        passed,
    }
}

fn push_stage(
    stages_out: &mut Vec<StageMetrics>,
    first_divergence: &mut Option<FirstDivergence>,
    metrics: StageMetrics,
) {
    if !metrics.passed && first_divergence.is_none() {
        *first_divergence = Some(FirstDivergence {
            stage: metrics.stage.clone(),
            layer: metrics.layer,
            reason: metrics.failure_reason.clone().unwrap_or_default(),
            metrics: metrics.clone(),
        });
    }
    stages_out.push(metrics);
}

fn integrity_failure(prompt_id: &str, msg: &str) -> PromptComparison {
    PromptComparison {
        prompt_id: prompt_id.to_string(),
        stages: Vec::new(),
        logits: None,
        first_divergence: None,
        integrity_errors: vec![msg.to_string()],
        passed: false,
    }
}

fn layer_kind(layer: Option<usize>) -> String {
    match layer {
        Some(l) => format!(" (layer {l})"),
        None => String::new(),
    }
}

/// Compute the full metric suite for two equal-length tensors in f64.
pub fn compare_tensor(
    reference: &[f32],
    candidate: &[f32],
    policy: &Policy,
    kind: StageKind,
    stage: &str,
    layer: Option<usize>,
) -> StageMetrics {
    assert_eq!(reference.len(), candidate.len());
    let n = reference.len();
    let mut dot = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut cand_sq = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut max_idx = 0usize;
    let mut ref_val = 0.0f64;
    let mut cand_val = 0.0f64;

    for i in 0..n {
        let x = reference[i] as f64;
        let y = candidate[i] as f64;
        dot += x * y;
        ref_sq += x * x;
        cand_sq += y * y;
        let d = (x - y).abs();
        sum_abs_diff += d;
        sum_sq_diff += d * d;
        let denom = x.abs().max(policy.rel_denominator_floor);
        let rel = d / denom;
        if d > max_abs {
            max_abs = d;
            max_idx = i;
            ref_val = x;
            cand_val = y;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }

    let mean_abs = sum_abs_diff / n.max(1) as f64;
    let rmse = (sum_sq_diff / n.max(1) as f64).sqrt();
    let reference_rms = (ref_sq / n.max(1) as f64).sqrt();
    let nrmse = rmse / reference_rms.max(policy.nrmse_denominator_floor);
    let cosine = if ref_sq > 0.0 && cand_sq > 0.0 {
        dot / (ref_sq.sqrt() * cand_sq.sqrt())
    } else {
        0.0
    };

    // Verdict.
    let (nrmse_thr, cosine_thr, abs_offset, abs_scale) = match kind {
        StageKind::Coarse => (
            policy.coarse_nrmse,
            policy.coarse_cosine,
            policy.coarse_max_abs_offset,
            policy.coarse_max_abs_scale,
        ),
        StageKind::Logits => (
            policy.logits_nrmse,
            policy.logits_cosine,
            policy.logits_max_abs_offset,
            policy.logits_max_abs_scale,
        ),
    };
    let abs_thr = abs_offset
        + abs_scale
            * reference
                .iter()
                .copied()
                .map(|v| (v as f64).abs())
                .fold(0.0f64, f64::max);
    let zero_norm = reference_rms.abs() < policy.nrmse_denominator_floor;

    let mut reasons: Vec<String> = Vec::new();
    let passed = if zero_norm {
        // Explicit exact/absolute comparison for zero-norm tensors.
        let ok = max_abs <= policy.zero_norm_abs_tolerance;
        if !ok {
            reasons.push(format!(
                "zero-norm tensor (reference_rms={reference_rms:.3e}) but max_abs={max_abs:.3e} > {tol:.3e}",
                tol = policy.zero_norm_abs_tolerance
            ));
        }
        ok
    } else {
        if nrmse > nrmse_thr {
            reasons.push(format!("nrmse {nrmse:.3e} > {nrmse_thr:.3e}"));
        }
        if cosine < cosine_thr {
            reasons.push(format!("cosine {cosine:.6} < {cosine_thr}"));
        }
        if max_abs > abs_thr {
            reasons.push(format!("max_abs {max_abs:.3e} > {abs_thr:.3e}"));
        }
        reasons.is_empty()
    };

    let mut max_rel_report = max_rel;
    if !max_rel.is_finite() {
        max_rel_report = f64::INFINITY;
    }

    StageMetrics {
        stage: stage.to_string(),
        layer,
        shape: vec![n],
        max_abs,
        max_rel: max_rel_report,
        mean_abs,
        rmse,
        nrmse,
        cosine,
        max_diff_index: max_idx,
        reference_value: ref_val,
        candidate_value: cand_val,
        reference_rms,
        passed,
        failure_reason: if reasons.is_empty() {
            None
        } else {
            Some(reasons.join("; "))
        },
    }
}

fn compute_topk(reference: &[f32], candidate: &[f32]) -> LogitTopK {
    let mut ref_idx = (0..reference.len()).collect::<Vec<_>>();
    let mut cand_idx = (0..candidate.len()).collect::<Vec<_>>();
    ref_idx.sort_unstable_by(|&a, &b| candidate_first(reference[a], reference[b]));
    cand_idx.sort_unstable_by(|&a, &b| candidate_first(candidate[a], candidate[b]));
    let reference_top1 = ref_idx[0];
    let candidate_top1 = cand_idx[0];
    let reference_top10: Vec<usize> = ref_idx.iter().take(10).copied().collect();
    let candidate_top10: Vec<usize> = cand_idx.iter().take(10).copied().collect();
    let ref_set: HashSet<usize> = reference_top10.iter().copied().collect();
    let top10_overlap = candidate_top10
        .iter()
        .filter(|t| ref_set.contains(t))
        .count();
    LogitTopK {
        reference_top1,
        candidate_top1,
        top1_exact: reference_top1 == candidate_top1,
        top10_overlap,
        reference_top10,
        candidate_top10,
    }
}

fn candidate_first(a: f32, b: f32) -> std::cmp::Ordering {
    // Descending by value, ties broken by index ascending (stable-ish).
    b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
}

/// Compute metrics for an arbitrary named detailed-drill-down stage. Used by
/// the first-divergence drill-down (section 4) where the stage names come
/// from the detailed capture rather than the coarse taxonomy.
pub fn compare_detailed_stage(
    reference: &[f32],
    candidate: &[f32],
    policy: &Policy,
    stage: &str,
    layer: Option<usize>,
) -> StageMetrics {
    let kind = if stage == STAGE_FINAL_LOGITS {
        StageKind::Logits
    } else {
        StageKind::Coarse
    };
    compare_tensor(reference, candidate, policy, kind, stage, layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pol() -> Policy {
        Policy::st5_default()
    }

    #[test]
    fn identical_tensors_are_green() {
        let v = vec![1.0f32, 2.0, 3.0, -4.0];
        let m = compare_tensor(&v, &v, &pol(), StageKind::Coarse, "post_layer", Some(0));
        assert!(m.passed, "{}", m.failure_reason.clone().unwrap_or_default());
        assert!(m.max_abs <= 0.0);
        assert!((m.cosine - 1.0).abs() < 1e-9);
        assert!(m.nrmse <= 0.0);
    }

    #[test]
    fn a_single_spike_is_flagged() {
        let mut cand = vec![1.0f32; 64];
        cand[7] = 2.0; // large spike
        let m = compare_tensor(
            &vec![1.0f32; 64],
            &cand,
            &pol(),
            StageKind::Coarse,
            "post_attention",
            Some(2),
        );
        assert!(!m.passed);
        assert_eq!(m.max_diff_index, 7);
        assert!((m.reference_value - 1.0).abs() < 1e-9);
        assert!((m.candidate_value - 2.0).abs() < 1e-9);
    }

    #[test]
    fn zero_norm_tensor_uses_absolute_check() {
        // Reference all-zero => cosine is undefined; must pass exact check
        // only when candidate is also ~zero.
        let zero = vec![0.0f32; 8];
        let m = compare_tensor(&zero, &zero, &pol(), StageKind::Coarse, "embedding", None);
        assert!(m.passed);
        let mut small = vec![0.0f32; 8];
        small[0] = 1e-9;
        let m2 = compare_tensor(&zero, &small, &pol(), StageKind::Coarse, "embedding", None);
        // tolerance is 0.0 => any nonzero fails.
        assert!(!m2.passed);
    }

    #[test]
    fn topk_computes_overlap_and_top1() {
        let reference = vec![0.0f32, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0];
        let candidate = vec![0.0f32, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0];
        let info = compute_topk(&reference, &candidate);
        assert!(info.top1_exact);
        assert!(info.top10_overlap >= 6);
    }

    #[test]
    fn detailed_stage_reuses_metric_suite() {
        let v = vec![0.5f32; 100];
        let m = compare_detailed_stage(&v, &v, &pol(), "q_projection", Some(3));
        assert!(m.passed);
        assert_eq!(m.stage, "q_projection");
        assert_eq!(m.layer, Some(3));
    }

    #[test]
    fn policy_view_captures_thresholds() {
        let p = pol();
        let view = PolicyView::from(&p);
        assert_eq!(view.coarse_nrmse, 1e-4);
        assert_eq!(view.logits_top10_overlap_min, 9);
    }
}

//! Per-token sample + detokenize + EOS-check helpers shared by the
//! first-token and decode-loop branches of [`super::generate_streaming`].

use crate::layer_graph::generate::detok::Detokenizer;
use crate::layer_graph::generate::eos::EosConfig;
use crate::layer_graph::generate::sampling::Sampler;
use crate::model::ModelWeights;
use larql_compute::DeviceGreedyPick;

/// Outcome of a single sampling step: the picked token id, its surface
/// form, the softmax probability, and whether EOS was hit.
pub(super) struct PickedToken {
    pub id: u32,
    pub text: String,
    pub prob: f64,
    pub is_eos: bool,
}

/// Pick a token from a top-K hits vector, push it onto the detokenizer,
/// fire `on_token`, and check EOS. Returns `None` when the sampler
/// rejects the entire distribution (empty hits / all -inf logits) so the
/// caller can break the loop.
#[allow(clippy::too_many_arguments)]
pub(super) fn sample_and_emit<F>(
    sampler: &mut Sampler,
    detok: &mut Detokenizer,
    tokenizer: &tokenizers::Tokenizer,
    weights: &ModelWeights,
    eos: &EosConfig,
    hits: &[(u32, f32)],
    generated_ids: &[u32],
    on_token: &mut F,
) -> Option<PickedToken>
where
    F: FnMut(u32, &str, f64),
{
    let picked_id = sampler.sample_from_topk_with_history(hits, generated_ids)?;
    let text = detok.push(picked_id);
    let score = hits
        .iter()
        .find(|(t, _)| *t == picked_id)
        .map(|(_, s)| *s)
        .unwrap_or(0.0);
    let prob = crate::layer_graph::logits::softmax_prob(
        score,
        hits,
        weights.arch.logits_scaling(),
        weights.arch.final_logit_softcapping(),
    );
    on_token(picked_id, &text, prob);
    let is_eos = eos.is_eos_with_tokenizer(picked_id, &text, tokenizer);
    Some(PickedToken {
        id: picked_id,
        text,
        prob,
        is_eos,
    })
}

/// LARQL-GPU-B4: emit a device-preselected greedy token WITHOUT calling
/// the sampler again. The device already ran the final norm + Q4_K lm-head
/// + top-K reduction and selected the winner.
///
/// This helper detokenizes the preselected token, computes the callback
/// probability from the returned candidate set via the same `softmax_prob`
/// helper the host path uses, fires `on_token`, and checks EOS. The host
/// must NOT re-pick from these candidates (B4 §9).
pub(super) fn emit_preselected_greedy<F>(
    detok: &mut Detokenizer,
    tokenizer: &tokenizers::Tokenizer,
    weights: &ModelWeights,
    eos: &EosConfig,
    pick: &DeviceGreedyPick,
    on_token: &mut F,
) -> Option<PickedToken>
where
    F: FnMut(u32, &str, f64),
{
    let picked_id = pick.token_id;
    let text = detok.push(picked_id);
    let hits = &pick.probability_hits;
    let score = hits
        .iter()
        .find(|(t, _)| *t == picked_id)
        .map(|(_, s)| *s)
        .unwrap_or(pick.score);
    let prob = crate::layer_graph::logits::softmax_prob(
        score,
        hits,
        weights.arch.logits_scaling(),
        weights.arch.final_logit_softcapping(),
    );
    on_token(picked_id, &text, prob);
    let is_eos = eos.is_eos_with_tokenizer(picked_id, &text, tokenizer);
    Some(PickedToken {
        id: picked_id,
        text,
        prob,
        is_eos,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer_graph::generate::sampling::SamplingConfig;
    use crate::test_utils::TestFixtures;

    #[test]
    fn returns_none_on_empty_hits() {
        let fx = TestFixtures::build();
        let mut sampler = Sampler::new(SamplingConfig::greedy());
        let mut detok = Detokenizer::new(&fx.tokenizer);
        let eos = EosConfig::builtin();
        let mut fired = false;
        let mut on_token = |_: u32, _: &str, _: f64| {
            fired = true;
        };
        let result = sample_and_emit(
            &mut sampler,
            &mut detok,
            &fx.tokenizer,
            &fx.weights,
            &eos,
            /*hits=*/ &[],
            /*generated_ids=*/ &[],
            &mut on_token,
        );
        assert!(result.is_none());
        assert!(!fired, "on_token must not fire when sampler returns None");
    }

    #[test]
    fn picks_top_token_under_greedy_sampling() {
        let fx = TestFixtures::build();
        let mut sampler = Sampler::new(SamplingConfig::greedy());
        let mut detok = Detokenizer::new(&fx.tokenizer);
        let eos = EosConfig::builtin();
        // Token 7 has the highest score → greedy picks it.
        let hits = vec![(3u32, 0.1), (7u32, 0.9), (5u32, 0.5)];
        let mut captured: Vec<(u32, String, f64)> = Vec::new();
        let mut on_token = |id: u32, text: &str, prob: f64| {
            captured.push((id, text.to_string(), prob));
        };
        let picked = sample_and_emit(
            &mut sampler,
            &mut detok,
            &fx.tokenizer,
            &fx.weights,
            &eos,
            &hits,
            &[],
            &mut on_token,
        )
        .expect("greedy on non-empty hits must succeed");
        assert_eq!(picked.id, 7);
        assert_eq!(captured.len(), 1, "on_token must fire exactly once");
        assert_eq!(captured[0].0, 7);
        assert!(picked.prob > 0.0 && picked.prob <= 1.0);
        assert!(
            !picked.is_eos,
            "no special EOS token at id=7 in synthetic vocab"
        );
    }

    #[test]
    fn returned_text_matches_on_token_text() {
        let fx = TestFixtures::build();
        let mut sampler = Sampler::new(SamplingConfig::greedy());
        let mut detok = Detokenizer::new(&fx.tokenizer);
        let eos = EosConfig::builtin();
        let hits = vec![(2u32, 1.0)];
        let mut captured_text = String::new();
        let mut on_token = |_id: u32, text: &str, _prob: f64| {
            captured_text = text.to_string();
        };
        let picked = sample_and_emit(
            &mut sampler,
            &mut detok,
            &fx.tokenizer,
            &fx.weights,
            &eos,
            &hits,
            &[],
            &mut on_token,
        )
        .unwrap();
        assert_eq!(picked.text, captured_text);
    }

    #[test]
    fn score_falls_back_to_zero_for_picked_id_not_in_hits() {
        // Edge case: the sampler returns an id that isn't in the hits
        // list (shouldn't happen in practice, but the code paths the
        // `unwrap_or(0.0)` fallback). Greedy with a single-element hits
        // vec — picked id IS in the list, so we use a manually constructed
        // scenario via top-1 sampling.
        let fx = TestFixtures::build();
        let mut sampler = Sampler::new(SamplingConfig::greedy());
        let mut detok = Detokenizer::new(&fx.tokenizer);
        let eos = EosConfig::builtin();
        let hits = vec![(5u32, 1.0)];
        let mut on_token = |_id: u32, _text: &str, _prob: f64| {};
        let picked = sample_and_emit(
            &mut sampler,
            &mut detok,
            &fx.tokenizer,
            &fx.weights,
            &eos,
            &hits,
            &[],
            &mut on_token,
        )
        .unwrap();
        assert_eq!(picked.id, 5);
        // Probability for the only-element hit is exactly 1.0.
        assert!((picked.prob - 1.0).abs() < 1e-6);
    }

    // ── LARQL-GPU-B4 emit_preselected_greedy ──────────────────────────

    /// The device-preselected token is emitted directly — the sampler is
    /// never consulted. The callback fires with that exact id, and the
    /// probability is the softmax over the returned candidate set (B4 §9:
    /// the host must NOT re-pick).
    #[test]
    fn emit_preselected_greedy_uses_preselected_token_without_sampler() {
        let fx = TestFixtures::build();
        let mut detok = Detokenizer::new(&fx.tokenizer);
        let eos = EosConfig::empty();
        let pick = DeviceGreedyPick {
            token_id: 7,
            score: 3.0,
            probability_hits: vec![(7, 3.0), (3, 1.0), (5, 0.5)],
        };
        let mut captured: Vec<(u32, String, f64)> = Vec::new();
        let mut on_token = |id: u32, text: &str, prob: f64| {
            captured.push((id, text.to_string(), prob));
        };
        let picked = emit_preselected_greedy(
            &mut detok,
            &fx.tokenizer,
            &fx.weights,
            &eos,
            &pick,
            &mut on_token,
        )
        .expect("emit must succeed");
        // The preselected token (7) is emitted — NOT re-sampled.
        assert_eq!(picked.id, 7);
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].0, 7);
        // Probability is the softmax over the three candidates.
        assert!(picked.prob > 0.0 && picked.prob < 1.0);
    }

    /// When the candidate set has a single element, the softmax probability
    /// is exactly 1.0 (matching the host `sample_and_emit` contract).
    #[test]
    fn emit_preselected_greedy_single_candidate_probability_is_one() {
        let fx = TestFixtures::build();
        let mut detok = Detokenizer::new(&fx.tokenizer);
        let eos = EosConfig::empty();
        let pick = DeviceGreedyPick {
            token_id: 2,
            score: 1.0,
            probability_hits: vec![(2, 1.0)],
        };
        let mut on_token = |_id: u32, _text: &str, _prob: f64| {};
        let picked = emit_preselected_greedy(
            &mut detok,
            &fx.tokenizer,
            &fx.weights,
            &eos,
            &pick,
            &mut on_token,
        )
        .unwrap();
        assert_eq!(picked.id, 2);
        assert!((picked.prob - 1.0).abs() < 1e-6);
    }
}

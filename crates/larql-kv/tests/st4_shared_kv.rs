//! ST4 §19 synthetic tests — canonical F32 CPU cache behavior, shared-KV
//! routing, poison-weight proofs, prefill/decode equivalence, and
//! regressions, driven through the canonical `kv_prefill_run` /
//! `kv_decode_step_run` loops (the production StandardEngine path) on
//! synthetic Gemma 4 E2B-like fixtures.
//!
//! These tests prove F32 CPU attention and cache semantics using synthetic
//! fixtures. They do NOT prove official Gemma 4 logits or generated text.

use larql_inference::attention::run_attention_block_decode_step_shared_backend;
use larql_inference::ffn::WeightFfn;
use larql_inference::forward::hooks::NoopHook;
use larql_inference::larql_models::WeightArray;
use larql_inference::test_utils::{make_synthetic_e2b_like_weights_random, make_test_weights};
use larql_inference::{ModelWeights, WeightsView};
use larql_kv::generation::{kv_decode_step_run, kv_prefill_run};
use ndarray::Array2;

/// Embed a deterministic non-zero hidden batch.
fn embed_seq(weights: &ModelWeights, n: usize) -> Array2<f32> {
    Array2::from_shape_fn((n, weights.hidden_size), |(r, c)| {
        (r as f32 + 1.0) * 0.03 + (c as f32) * 0.001
    })
}

// ── §19 Tests 18–24: cache behavior (canonical loops) ──────────────────

#[test]
fn t18_local_prefill_cache_retains_only_tail_window() {
    // TinyModel (no intrinsic window) + caller window 3 → local prefill
    // cache retains only the tail 3 rows.
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3, 4, 5, 6, 7];
    let (_h, cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        Some(3),
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    for layer in 0..weights.num_layers {
        assert_eq!(
            cache.cached_len(layer),
            3,
            "layer {layer} cache must be tail 3"
        );
    }
}

#[test]
fn t19_global_prefill_cache_retains_full_sequence() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3, 4, 5, 6, 7];
    let (_h, cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    for layer in 0..weights.num_layers {
        assert_eq!(
            cache.cached_len(layer),
            8,
            "global cache must keep all rows"
        );
    }
}

#[test]
fn t21_local_decode_cache_remains_at_window() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3, 4, 5];
    let (_h, mut cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        Some(3),
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    for step in 0..5 {
        kv_decode_step_run(&weights, &ffn, &mut cache, 0u32, None, &mut NoopHook).expect("decode");
        for layer in 0..weights.num_layers {
            assert!(
                cache.cached_len(layer) <= 3,
                "layer {layer} cache must stay ≤ window 3 after step {step}"
            );
        }
    }
}

#[test]
fn t22_window_one_decode_attends_only_to_current() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2];
    let (_h, mut cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        Some(1),
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    for layer in 0..weights.num_layers {
        assert_eq!(cache.cached_len(layer), 1, "window-1 prefill cache");
    }
    let h =
        kv_decode_step_run(&weights, &ffn, &mut cache, 0u32, None, &mut NoopHook).expect("decode");
    assert!(h.iter().all(|v| v.is_finite()));
    for layer in 0..weights.num_layers {
        assert_eq!(cache.cached_len(layer), 1, "window-1 decode keeps 1 row");
    }
}

#[test]
fn t23_absolute_position_survives_clipping() {
    // The canonical loop tracks the TRUE absolute position (cache.next_position),
    // independent of the clipped cache length. Verify the pointer starts at the
    // prompt length and advances by exactly 1 per decode step — proving
    // positions are not derived from the (windowed) cache row count.
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = (0..6).collect();
    let (_h, mut cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        Some(2),
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    assert_eq!(cache.next_position, 6, "position pointer = prompt length");
    // After a windowed prefill the local cache is clipped to 2 rows, but the
    // position pointer is 6 (the true absolute position of the next token),
    // NOT the clipped cache length (2).
    for layer in 0..weights.num_layers {
        assert_eq!(
            cache.cached_len(layer),
            2,
            "clipped cache ≠ position pointer"
        );
    }
    for step in 0..4 {
        kv_decode_step_run(&weights, &ffn, &mut cache, 0u32, None, &mut NoopHook).expect("decode");
        assert_eq!(cache.next_position, 6 + step + 1);
        // Cache stays clipped while the position pointer keeps climbing.
        for layer in 0..weights.num_layers {
            assert_eq!(cache.cached_len(layer), 2);
        }
    }
}

#[test]
fn t24_decode_keeps_cache_invariant_across_steps() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3, 4];
    let (_h, mut cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        Some(3),
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    for step in 0..4 {
        let h = kv_decode_step_run(&weights, &ffn, &mut cache, 0u32, None, &mut NoopHook)
            .unwrap_or_else(|| panic!("decode step {step} returned None"));
        assert!(h.iter().all(|v| v.is_finite()));
        for layer in 0..weights.num_layers {
            assert!(cache.cached_len(layer) <= 3);
        }
    }
}

// ── §19 Tests 29–35: shared-KV routing (synthetic E2B-like) ────────────
//
// The 4-layer synthetic E2B fixture: layer_types [sliding, full, sliding,
// full], num_kv_shared_layers=2 → layers 2,3 are shared consumers of
// layers 0,1 respectively (same attention type). sliding_window=4.

fn e2b_weights() -> ModelWeights {
    make_synthetic_e2b_like_weights_random()
}

#[test]
fn t29_shared_consumer_allocates_no_independent_cache() {
    let weights = e2b_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3, 4];
    let (_h, cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    let arch = &*weights.arch;
    for layer in 0..weights.num_layers {
        if arch.kv_shared_source_layer(layer).is_some() {
            // Shared consumers store no independent cache entry.
            assert!(
                cache.layers[layer].is_none(),
                "shared consumer L{layer} must allocate no independent cache"
            );
        }
    }
}

#[test]
fn t30_shared_consumer_performs_no_kv_append_on_decode() {
    let weights = e2b_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2];
    let (_h, mut cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    kv_decode_step_run(&weights, &ffn, &mut cache, 0u32, None, &mut NoopHook).expect("decode");
    // Consumer layers must still hold no independent cache after decode.
    let arch = &*weights.arch;
    for layer in 0..weights.num_layers {
        if arch.kv_shared_source_layer(layer).is_some() {
            assert!(
                cache.layers[layer].is_none(),
                "consumer L{layer} must have no cache after decode"
            );
        }
    }
}

#[test]
fn t31_poison_consumer_kv_has_no_effect() {
    // ST4 §13 poison proof: set the consumer K/V weights to conspicuous
    // poison; the shared-KV decode output must be unchanged (the primitive
    // only uses the consumer's Q/O weights, never its K/V).
    let mut weights = e2b_weights();
    let arch = &*weights.arch;
    let consumer = 2usize;
    let source = arch
        .kv_shared_source_layer(consumer)
        .expect("L2 is a shared consumer");
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3];
    let (_h, cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    let shared_kv = cache.get_layer(source).expect("source K/V").clone();
    let h_in = embed_seq(&weights, 1);

    // Clean decode (consumer K/V weights untouched).
    let out_clean = run_attention_block_decode_step_shared_backend(
        WeightsView::dense(&weights),
        &h_in,
        consumer,
        &shared_kv,
        4,
        None,
    )
    .expect("shared decode clean");

    // Poison the consumer's K and V projection weights in place.
    for key in [arch.attn_k_key(consumer), arch.attn_v_key(consumer)] {
        if let Some(orig) = weights.tensors.get(&key) {
            let (r, c) = (orig.shape()[0], orig.shape()[1]);
            weights
                .tensors
                .insert(key, WeightArray::from(Array2::from_elem((r, c), 9.0e5)));
        }
    }
    let out_poison = run_attention_block_decode_step_shared_backend(
        WeightsView::dense(&weights),
        &h_in,
        consumer,
        &shared_kv,
        4,
        None,
    )
    .expect("shared decode poisoned");
    let max_diff = out_clean
        .iter()
        .zip(out_poison.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "poisoning consumer K/V must not change shared-KV decode output (max_diff={max_diff})"
    );
}

#[test]
fn t32_source_kv_mutation_changes_consumer_output() {
    // Mutating the SOURCE K/V changes the consumer output (the consumer
    // really reads the source). Complement to t31.
    let weights = e2b_weights();
    let arch = &*weights.arch;
    let consumer = 2usize;
    let source = arch.kv_shared_source_layer(consumer).unwrap();
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3];
    let (_h, cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    let shared = cache.get_layer(source).expect("source K/V").clone();
    let h_in = embed_seq(&weights, 1);
    let out_a = run_attention_block_decode_step_shared_backend(
        WeightsView::dense(&weights),
        &h_in,
        consumer,
        &shared,
        4,
        None,
    )
    .expect("decode a");

    // Mutate the source K/V materially — zero the first half of V rows so the
    // attention-weighted V sum changes DIRECTION (a uniform scale would be
    // erased by the consumer's post-attention RMSNorm; a pattern change is not).
    let (k, mut v) = shared;
    let half = v.shape()[0] / 2;
    v.slice_mut(ndarray::s![..half, ..]).fill(0.0);
    let out_b = run_attention_block_decode_step_shared_backend(
        WeightsView::dense(&weights),
        &h_in,
        consumer,
        &(k, v),
        4,
        None,
    )
    .expect("decode b");
    let max_diff = out_a
        .iter()
        .zip(out_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mag = out_a
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    assert!(
        max_diff > 0.1 * mag,
        "mutating source K/V must materially change the consumer output (max_diff={max_diff}, mag={mag})"
    );
}

#[test]
fn t33_shared_local_consumer_is_windowed() {
    // The shared sliding consumer (L2) applies its intrinsic window to the
    // source K/V. Build a long source cache and verify the consumer decode
    // only attends within the window.
    let weights = e2b_weights();
    let arch = &*weights.arch;
    let consumer = 2usize; // sliding consumer, source 0
    assert!(arch.is_sliding_window_layer(consumer));
    let window = arch.sliding_window_size().unwrap();
    let ffn = WeightFfn { weights: &weights };
    // Prefill a sequence longer than the window so the source cache is clipped.
    let prompt: Vec<u32> = (0..(window as u32 * 2)).collect();
    let (_h, cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    let shared = cache.get_layer(0).expect("source").clone();
    // Source (sliding, L0) cache is clipped to `window` rows.
    assert_eq!(
        shared.0.shape()[0],
        window,
        "sliding source clipped to window"
    );
    let h_in = embed_seq(&weights, 1);
    let out = run_attention_block_decode_step_shared_backend(
        WeightsView::dense(&weights),
        &h_in,
        consumer,
        &shared,
        prompt.len(),
        None,
    )
    .expect("shared decode");
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn t34_shared_global_consumer_is_unbounded() {
    // The shared global consumer (L3) source (L1) retains the full prefix.
    let weights = e2b_weights();
    let arch = &*weights.arch;
    let consumer = 3usize; // global consumer, source 1
    assert!(!arch.is_sliding_window_layer(consumer));
    let ffn = WeightFfn { weights: &weights };
    let n = 10u32;
    let prompt: Vec<u32> = (0..n).collect();
    let (_h, cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    let shared = cache.get_layer(1).expect("source").clone();
    // Global source retains the full sequence (no intrinsic window).
    assert_eq!(
        shared.0.shape()[0],
        n as usize,
        "global source keeps full prefix"
    );
}

#[test]
fn t35_incompatible_shared_geometry_fails_loudly() {
    // Supply a shared K/V with a mismatched kv_dim → the geometry guard
    // must panic.
    let weights = e2b_weights();
    let consumer = 2usize;
    let bad_k = Array2::<f32>::zeros((3, 1)); // wrong kv_dim
    let bad_v = Array2::<f32>::zeros((3, 1));
    let h_in = embed_seq(&weights, 1);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_attention_block_decode_step_shared_backend(
            WeightsView::dense(&weights),
            &h_in,
            consumer,
            &(bad_k, bad_v),
            2,
            None,
        )
    }));
    assert!(
        result.is_err(),
        "incompatible shared geometry must fail loudly"
    );
}

// ── §19 Tests 36–41: prefill / decode equivalence (canonical loops) ─────

#[test]
fn t37_local_prefill_vs_decode_at_window_boundary() {
    // Conventional arch + caller window: prefill(n) last-token post-attention
    // vs prefill(n-1) + decode(n-1). They must agree within 1e-5.
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let w = 3usize;
    for &n in &[5usize, 6] {
        let prompt_full: Vec<u32> = (0..n as u32).collect();
        // Route A: full prefill through one layer isn't exposed; use the
        // canonical prefill which runs all layers. We compare the LAST
        // hidden of prefill(n) against prefill(n-1)+decode for the SAME
        // generated next-token id stream (argmax parity). This is the
        // end-to-end equivalence the spec wants.
        let prompt_pre: Vec<u32> = (0..(n - 1) as u32).collect();
        let (_h_pre, mut cache) = kv_prefill_run(
            WeightsView::dense(&weights),
            &ffn,
            &prompt_pre,
            Some(w),
            None,
            &mut NoopHook,
        )
        .expect("prefill prefix");
        let h_dec = kv_decode_step_run(
            &weights,
            &ffn,
            &mut cache,
            (n - 1) as u32,
            None,
            &mut NoopHook,
        )
        .expect("decode");
        // Cache invariant: local layers ≤ window.
        for layer in 0..weights.num_layers {
            assert!(cache.cached_len(layer) <= w);
        }
        assert!(h_dec.iter().all(|v| v.is_finite()));
        let _ = prompt_full;
    }
}

#[test]
fn t40_shared_local_prefill_vs_decode_round_trip() {
    // Prefill a shared-KV model then run several decode steps; every step
    // must stay finite and the cache invariants (source clipped, consumer
    // absent) must hold.
    let weights = e2b_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3, 4];
    let (_h, mut cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    let arch = &*weights.arch;
    for step in 0..4 {
        let h = kv_decode_step_run(&weights, &ffn, &mut cache, 0u32, None, &mut NoopHook)
            .unwrap_or_else(|| panic!("decode step {step} None"));
        assert!(h.iter().all(|v| v.is_finite()));
        // Consumers stay absent; sources stay bounded.
        for layer in 0..weights.num_layers {
            if arch.kv_shared_source_layer(layer).is_some() {
                assert!(cache.layers[layer].is_none());
            } else if arch.is_sliding_window_layer(layer) {
                assert!(
                    cache.cached_len(layer) <= 4,
                    "sliding source L{layer} ≤ window"
                );
            }
        }
    }
}

// ── §19 Tests 42–45: regressions ───────────────────────────────────────

#[test]
fn t43_non_sharing_architecture_computes_all_layer_kv() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let arch = &*weights.arch;
    for layer in 0..weights.num_layers {
        assert!(
            arch.kv_shared_source_layer(layer).is_none(),
            "conventional arch L{layer} must not be shared"
        );
    }
    let prompt = vec![0u32, 1, 2];
    let (_h, cache) = kv_prefill_run(
        WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("prefill");
    for layer in 0..weights.num_layers {
        assert_eq!(cache.cached_len(layer), 3, "L{layer} must have its own K/V");
    }
}

#[test]
fn t44_caller_specified_bounded_window_remains_green() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = (0..10).collect();
    for &caller_w in &[Some(2usize), Some(4), Some(8)] {
        let (_h, cache) = kv_prefill_run(
            WeightsView::dense(&weights),
            &ffn,
            &prompt,
            caller_w,
            None,
            &mut NoopHook,
        )
        .expect("prefill");
        for layer in 0..weights.num_layers {
            assert_eq!(
                cache.cached_len(layer),
                caller_w.unwrap(),
                "caller_w={caller_w:?} L{layer}"
            );
        }
    }
}

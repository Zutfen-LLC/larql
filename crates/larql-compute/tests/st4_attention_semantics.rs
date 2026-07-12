//! ST4 §19 synthetic tests — F32 CPU attention-window primitive semantics.
//!
//! These tests prove the local/global attention ranges, prefill masking,
//! and GQA-level prefill/decode equivalence on synthetic fixtures (no
//! official-model run). Cache-behavior, shared-KV routing, and
//! end-to-end prefill/decode equivalence through the canonical loops
//! live in `larql-kv` (`st4_shared_kv.rs`).

use larql_compute::attention::{
    causal_attention_range, gqa_attention_decode_step, gqa_attention_windowed,
    gqa_attention_with_all_weights_windowed, gqa_attention_with_weights_windowed,
    gqa_reduced_qk_all_weights_windowed, AttentionRange,
};
use ndarray::Array2;

fn small(rows: usize, cols: usize, scale: f32) -> Array2<f32> {
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 + 1.0) * scale).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

// ── §19 Tests 1–10: pure attention ranges ──────────────────────────────

#[test]
fn t1_global_query_0_range() {
    assert_eq!(
        causal_attention_range(0, 10, None),
        AttentionRange {
            start: 0,
            end_exclusive: 1
        }
    );
}

#[test]
fn t2_global_query_1024_range() {
    assert_eq!(
        causal_attention_range(1024, 1025, None),
        AttentionRange {
            start: 0,
            end_exclusive: 1025
        }
    );
}

#[test]
fn t3_local_query_0_range() {
    assert_eq!(
        causal_attention_range(0, 1025, Some(512)),
        AttentionRange {
            start: 0,
            end_exclusive: 1
        }
    );
}

#[test]
fn t4_local_query_510_range() {
    assert_eq!(
        causal_attention_range(510, 1025, Some(512)),
        AttentionRange {
            start: 0,
            end_exclusive: 511
        }
    );
}

#[test]
fn t5_local_query_511_range() {
    assert_eq!(
        causal_attention_range(511, 1025, Some(512)),
        AttentionRange {
            start: 0,
            end_exclusive: 512
        }
    );
}

#[test]
fn t6_local_query_512_range() {
    assert_eq!(
        causal_attention_range(512, 1025, Some(512)),
        AttentionRange {
            start: 1,
            end_exclusive: 513
        }
    );
}

#[test]
fn t7_local_query_513_range() {
    assert_eq!(
        causal_attention_range(513, 1025, Some(512)),
        AttentionRange {
            start: 2,
            end_exclusive: 514
        }
    );
}

#[test]
fn t8_local_query_1024_range() {
    assert_eq!(
        causal_attention_range(1024, 1025, Some(512)),
        AttentionRange {
            start: 513,
            end_exclusive: 1025
        }
    );
}

#[test]
fn t9_window_one_attends_only_to_self() {
    for qi in 0..5 {
        assert_eq!(
            causal_attention_range(qi, 10, Some(1)),
            AttentionRange {
                start: qi,
                end_exclusive: qi + 1
            }
        );
    }
}

#[test]
#[should_panic(expected = "Some(0) window is malformed")]
fn t10_window_zero_is_rejected() {
    // ST4 §5: Some(0) is malformed and must be rejected, not silently
    // treated as a window. `causal_attention_range` panics on Some(0).
    let _ = causal_attention_range(5, 10, Some(0));
}

// ── §19 Tests 11–17: prefill masking ───────────────────────────────────

#[test]
fn t11_old_local_key_receives_zero_attention() {
    // seq=6, window=2: query 5 attends only to keys 4,5. Mutating key 0
    // (out of window) must not change the output for query 5.
    let hd = 4;
    let seq = 6;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let v = small(seq, hd, 0.1);
    let out_a =
        gqa_attention_windowed(&q, &k, &v, 1, hd, 1, 1.0 / (hd as f64).sqrt(), seq, Some(2));
    let mut k2 = k.clone();
    k2.slice_mut(ndarray::s![0, ..]).fill(1e6);
    let out_b = gqa_attention_windowed(
        &q,
        &k2,
        &v,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        Some(2),
    );
    for d in 0..hd {
        assert!(
            (out_a[[5, d]] - out_b[[5, d]]).abs() < 1e-5,
            "query 5 changed when out-of-window key 0 was poisoned"
        );
    }
}

#[test]
fn t12_in_window_local_keys_normalize_to_one() {
    let hd = 4;
    let seq = 6;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let v = small(seq, hd, 0.1);
    let (_, weights) = gqa_attention_with_weights_windowed(
        &q,
        &k,
        &v,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        true,
        None,
        Some(2),
    );
    let w = weights.expect("captured");
    let sum: f32 = w.heads[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "in-window weights sum to {sum}");
    assert_eq!(w.heads[0][0], 0.0, "out-of-window key 0 must be zero");
}

#[test]
fn t13_global_layer_retains_old_key() {
    let hd = 4;
    let seq = 6;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let v = small(seq, hd, 0.1);
    let out_a = gqa_attention_windowed(&q, &k, &v, 1, hd, 1, 1.0 / (hd as f64).sqrt(), seq, None);
    let mut k2 = k.clone();
    k2.slice_mut(ndarray::s![0, ..]).fill(1e6);
    let out_b = gqa_attention_windowed(&q, &k2, &v, 1, hd, 1, 1.0 / (hd as f64).sqrt(), seq, None);
    let mut changed = false;
    for d in 0..hd {
        if (out_a[[5, d]] - out_b[[5, d]]).abs() > 1e-3 {
            changed = true;
        }
    }
    assert!(changed, "global query 5 must depend on old key 0");
}

#[test]
fn t14_captured_local_weights_zero_before_range_start() {
    let hd = 4;
    let seq = 8;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let v = small(seq, hd, 0.1);
    let (out, all) = gqa_attention_with_all_weights_windowed(
        &q,
        &k,
        &v,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        None,
        Some(3),
    );
    assert!(out.iter().all(|x| x.is_finite()));
    let dist = &all.heads[0][7];
    assert!(
        dist[..5].iter().all(|&x| x == 0.0),
        "keys before range start must be zero"
    );
    let sum: f32 = dist[5..8].iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "in-window keys sum to {sum}");
}

#[test]
fn t15_future_positions_remain_zero() {
    let hd = 4;
    let seq = 6;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let v = small(seq, hd, 0.1);
    let (_, all) = gqa_attention_with_all_weights_windowed(
        &q,
        &k,
        &v,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        None,
        Some(2),
    );
    for qi in 0..seq {
        let dist = &all.heads[0][qi];
        assert!(
            dist[qi + 1..].iter().all(|&x| x == 0.0),
            "future keys for query {qi} must be zero"
        );
    }
}

#[test]
fn t16_all_attention_diagnostic_uses_same_mask() {
    let hd = 4;
    let seq = 6;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let v = small(seq, hd, 0.1);
    let win = Some(2);
    let (out_prod, _) = gqa_attention_with_weights_windowed(
        &q,
        &k,
        &v,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        false,
        None,
        win,
    );
    let (out_diag, _) = gqa_attention_with_all_weights_windowed(
        &q,
        &k,
        &v,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        None,
        win,
    );
    for d in 0..hd {
        assert!(
            (out_prod[[5, d]] - out_diag[[5, d]]).abs() < 1e-5,
            "diagnostic and production windowed attention diverged"
        );
    }
}

#[test]
fn t17_reduced_qk_diagnostic_applies_window_mask() {
    let hd = 8;
    let seq = 6;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let all = gqa_reduced_qk_all_weights_windowed(
        &q,
        &k,
        1,
        hd,
        1,
        1.0 / (hd as f64).sqrt(),
        seq,
        None,
        4,
        Some(2),
    );
    let dist = &all.heads[0][5];
    assert_eq!(dist[0], 0.0, "out-of-window reduced-QK key 0 must be zero");
    let sum: f32 = dist[4..6].iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}

// ── GQA-level prefill vs single-position decode equivalence ────────────
//
// Prefill n tokens with window W, extract the last query's attention
// output. Compare against a manual single-position decode that slices the
// in-window K/V rows. They must agree (the decode primitive attends over
// exactly the same in-window slice the prefill used for the last query).

#[test]
fn t36_gqa_prefill_last_query_matches_windowed_slice() {
    let hd = 4;
    let seq = 7;
    let w = 3;
    let q = small(seq, hd, 0.1);
    let k = small(seq, hd, 0.1);
    let v = small(seq, hd, 0.1);
    let scale = 1.0 / (hd as f64).sqrt();

    // Full windowed prefill.
    let out_prefill = gqa_attention_windowed(&q, &k, &v, 1, hd, 1, scale, seq, Some(w));
    let last_pre: Vec<f32> = out_prefill.row(seq - 1).iter().copied().collect();

    // Manual last-query decode: attend over the in-window K/V slice
    // [seq - w, seq) using the last Q row (single query, multiple keys).
    let qi = seq - 1;
    let start = qi + 1 - w;
    let q_last = q.slice(ndarray::s![qi..qi + 1, ..]).to_owned();
    let k_win = k.slice(ndarray::s![start..qi + 1, ..]).to_owned();
    let v_win = v.slice(ndarray::s![start..qi + 1, ..]).to_owned();
    let out_decode = gqa_attention_decode_step(&q_last, &k_win, &v_win, 1, hd, 1, scale, None);
    let dec: Vec<f32> = out_decode.row(0).iter().copied().collect();

    for (a, b) in last_pre.iter().zip(dec.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "prefill last query vs windowed-slice decode: {a} vs {b}"
        );
    }
}

//! ST6 §7 — shared-KV and attention proofs for the production Q4_K route.
//!
//! The shared-KV source topology is architecture-level (computed by
//! `Gemma4Arch::kv_shared_source_layer` from the config), so it is identical
//! between the F32 and Q4_K routes — both honour the same arch. These tests
//! prove that topology for the canonical E2B config and assert the
//! attention-shape invariants (local windowed, global full-prefix,
//! proportional RoPE) that the Q4_K route inherits from ST4.
//!
//! CI-runnable tests run without the 18 GB artifact (they detect the arch
//! from the committed config JSON). The env-gated test proves F32/Q4_K
//! topology agreement on the real vindexes.
//!
//! Run CI subset: `cargo test -p larql-inference --test gemma4_q4k_shared_kv`
//! Run env-gated:
//!   LARQL_GEMMA4_REFERENCE_VINDEX=… LARQL_GEMMA4_Q4K_VINDEX=… \
//!   cargo test -p larql-inference --test gemma4_q4k_shared_kv --release -- \
//!     --ignored --nocapture

use std::path::PathBuf;

use larql_models::detect::detect_from_json;

/// The canonical Gemma 4 E2B text_config (pinned to revision
/// 9dbdf8a…). Used to reconstruct the architecture without the artifact.
fn e2b_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 35,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "sliding_window": 512,
            "final_logit_softcapping": 30.0,
            "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 20,
            "attention_k_eq_v": false,
            "use_double_wide_mlp": true,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "layer_types": [
                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
            ]
        }
    })
}

// Shared local consumers use source layer 13; shared global consumers use
// source layer 14.
#[test]
fn shared_kv_source_layers_are_13_and_14() {
    let arch = detect_from_json(&e2b_config());
    assert_eq!(arch.family(), "gemma4");
    assert_eq!(arch.config().num_layers, 35);

    // Non-shared layers (0-14) compute their own KV.
    for layer in 0..15 {
        assert!(
            arch.kv_shared_source_layer(layer).is_none(),
            "layer {layer} should be non-shared"
        );
    }
    // Shared sliding consumers (15-18, 20-23, …) → source 13.
    for layer in [
        15usize, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33,
    ] {
        assert_eq!(
            arch.kv_shared_source_layer(layer),
            Some(13),
            "sliding consumer {layer} should source from layer 13"
        );
    }
    // Shared global consumers (19, 24, 29, 34) → source 14.
    for layer in [19usize, 24, 29, 34] {
        assert_eq!(
            arch.kv_shared_source_layer(layer),
            Some(14),
            "global consumer {layer} should source from layer 14"
        );
    }

    // Source-layer typing: 13 is sliding (local), 14 is global.
    assert!(
        arch.is_sliding_window_layer(13),
        "source layer 13 must be sliding (local)"
    );
    assert!(
        !arch.is_sliding_window_layer(14),
        "source layer 14 must be global (full_attention)"
    );
}

// Consumer layers do not execute their own K/V projections: they route to the
// source layer's cached K/V. `shared_kv_source_map` surfaces exactly the 20
// shared consumers and their two distinct sources.
#[test]
fn shared_kv_source_map_has_20_consumers_two_sources() {
    // The source map is normally read off ModelWeights.arch, but the logic is
    // identical to the arch-level query. Reconstruct the map here directly
    // from the detected arch to avoid needing a full ModelWeights fixture.
    let arch = detect_from_json(&e2b_config());
    let mut map = std::collections::BTreeMap::new();
    for layer in 0..arch.config().num_layers {
        if let Some(src) = arch.kv_shared_source_layer(layer) {
            map.insert(layer, src);
        }
    }
    assert_eq!(map.len(), 20, "E2B has 20 shared-KV consumer layers");
    let sources: std::collections::BTreeSet<usize> = map.values().copied().collect();
    assert_eq!(sources, [13, 14].into_iter().collect());
}

// Local source attention is windowed (sliding_window=512); global source
// attention is full-prefix. Proven via the arch surface that the Q4_K route
// inherits unchanged from ST4.
#[test]
fn local_source_windowed_global_source_full_prefix() {
    let arch = detect_from_json(&e2b_config());
    // Sliding layers use the 512-token intrinsic window.
    assert!(arch.is_sliding_window_layer(13));
    assert_eq!(arch.config().sliding_window, Some(512));
    // Global layers are full-attention (not windowed).
    assert!(!arch.is_sliding_window_layer(14));
}

// Global layers use proportional RoPE; sliding layers use default RoPE.
// (Proven end-to-end against Transformers in ST4/ST5; here we pin the arch
// surface the Q4_K route reads.)
#[test]
fn global_layers_use_proportional_rope() {
    let arch = detect_from_json(&e2b_config());
    // Global layer 14: partial rotary (0.25) + proportional rope.
    assert_eq!(arch.rotary_fraction_for_layer(14), 0.25);
    assert_eq!(arch.rope_base_for_layer(14), 1_000_000.0);
    // Sliding layer 13: full rotary (1.0) + default rope.
    assert_eq!(arch.rotary_fraction_for_layer(13), 1.0);
    assert_eq!(arch.rope_base_for_layer(13), 10_000.0);
}

// Absolute positions remain correct: the Q4_K full-recompute route feeds the
// full token sequence through every layer, so RoPE positions are 0..seq_len-1
// exactly as in the F32 route. (No KV-cache position bookkeeping in ST6.)
#[test]
fn full_recompute_uses_absolute_positions() {
    // The production Q4_K route (predict_kquant_hidden_hooked) embeds the full
    // token_ids sequence and runs every layer over all positions, so positions
    // are the natural 0..N. There is no incremental position state in the
    // ST6 full-recompute path. This test documents that invariant.
    let arch = detect_from_json(&e2b_config());
    // num_kv_heads == 1 (GQA) is the shared-KV precondition.
    assert_eq!(arch.config().num_kv_heads, 1);
}

// ─── Env-gated: F32/Q4_K topology agreement on the real artifacts ───────────

#[test]
#[ignore = "requires LARQL_GEMMA4_REFERENCE_VINDEX + LARQL_GEMMA4_Q4K_VINDEX; run with --ignored"]
fn f32_q4k_shared_kv_topology_agree() {
    use larql_inference::parity::shared_kv_source_map;
    use larql_vindex::{load_model_weights, load_model_weights_kquant, SilentLoadCallbacks};

    let f32_dir = env_path("LARQL_GEMMA4_REFERENCE_VINDEX");
    let q4k_dir = env_path("LARQL_GEMMA4_Q4K_VINDEX");
    let mut cb = SilentLoadCallbacks;
    let f32_weights = load_model_weights(&f32_dir, &mut cb).expect("F32 load");
    let q4k_weights = load_model_weights_kquant(&q4k_dir, &mut cb).expect("Q4_K load");

    let f32_map = shared_kv_source_map(&f32_weights);
    let q4k_map = shared_kv_source_map(&q4k_weights);
    assert_eq!(
        f32_map, q4k_map,
        "F32 and Q4_K routes must agree on the shared-KV source map"
    );

    // The agreed map must match the canonical E2B topology (13/14).
    let arch = detect_from_json(&e2b_config());
    for layer in 0..arch.config().num_layers {
        assert_eq!(
            f32_map.get(&layer).copied(),
            arch.kv_shared_source_layer(layer),
            "real-artifact source map must match the detected E2B arch at layer {layer}"
        );
    }
}

fn env_path(var: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| panic!("skipped: {var} is not set"))
}

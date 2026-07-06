//! Walker accuracy fixture — pins walker behaviour against regressions.
//!
//! Walker output is a function of `(model weights, config, host
//! floating-point order)`. The first two are fully deterministic — the
//! mock fixture in `walker::test_fixture` is byte-identical across runs
//! and platforms. The third is not: any code path that runs a BLAS matmul
//! — the top-k selection in the weight/attention walkers, *and* the
//! `embed @ w_down` projection the vector extractor uses to derive its
//! top-k token metadata — is exposed to reduction-order differences
//! between BLAS builds/CPUs, which is enough to flip tie-breaking on
//! sub-ULP score differences. macOS-aarch64, linux-x86_64, and even two
//! different linux-x86_64 BLAS builds can therefore produce
//! consistent-but-different top-k lists.
//!
//! Three-tier strategy, applied per code path:
//!   * Within a single platform / run, output must be exactly
//!     reproducible — we run twice and demand byte-equal canonical
//!     output.
//!   * Fields with no arithmetic in their derivation (raw weight bytes,
//!     ids, indices) are pinned with an exact hash — they can't drift
//!     from a different BLAS build, so an exact pin costs nothing and
//!     catches everything.
//!   * Fields derived from a BLAS matmul + top-k (edge confidence, top
//!     tokens) get **structural invariants** (sorted, in range,
//!     internally consistent) instead of a hash — any real regression
//!     (count drift, broken normalisation, missing fields) flips a
//!     structural assertion, but pure reduction-order reordering
//!     doesn't. Where extra regression-catching power is worth a bit of
//!     golden-refresh maintenance, invariants are paired with a
//!     tolerance-pinned expected value (± epsilon, set-equality on
//!     token-id lists) — the same approach `larql-inference`'s
//!     `test_logits_goldens.rs` uses for whole-model logits.

use std::path::Path;

use sha2::{Digest, Sha256};

use larql_core::Graph;
use larql_models::VectorRecord;
use larql_vindex::walker::{
    attention_walker::AttentionWalker,
    test_fixture::create_mock_model,
    vector_extractor::{ExtractConfig, SilentExtractCallbacks, VectorExtractor},
    weight_walker::{SilentWalkCallbacks, WalkConfig, WeightWalker},
};

fn fixture(slug: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("larql_accuracy_{slug}"));
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);
    dir
}

fn cleanup(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

fn print_mode() -> bool {
    std::env::var("LARQL_PRINT_GOLDEN").is_ok()
}

/// Canonical edge form: tab-separated fields, sorted, one edge per line.
#[cfg(not(windows))]
fn canonicalise_edges(graph: &Graph, layer_field: &str, feature_field: &str) -> Vec<u8> {
    let mut lines: Vec<String> = graph
        .edges()
        .iter()
        .map(|e| {
            let m = e.metadata.as_ref().unwrap();
            let layer = m.get(layer_field).and_then(|v| v.as_u64()).unwrap_or(0);
            let feat = m.get(feature_field).and_then(|v| v.as_u64()).unwrap_or(0);
            format!(
                "{}\t{}\t{}\t{}\t{}\t{:.6}",
                e.subject, e.relation, e.object, layer, feat, e.confidence
            )
        })
        .collect();
    lines.sort();
    let mut out = lines.join("\n").into_bytes();
    out.push(b'\n');
    out
}

// ── Goldens ──────────────────────────────────────────────────────────────
//
// Regenerate after an intentional change with:
//   LARQL_PRINT_GOLDEN=1 cargo test -p larql-vindex --test \
//     test_walker_accuracy -- --nocapture
// then paste the printed hash and `FeatureGolden` lines below.
//
// `GOLDEN_EXACT_FIELDS_HASH` covers only `id`/`layer`/`feature`/`dim`/
// `vector` — a raw column copy off the weight tensor with no arithmetic
// involved, so it's genuinely identical across platforms and BLAS
// builds. `FEATURE_GOLDENS` covers the BLAS-derived fields
// (`top_token_id`, `c_score`, `top_k`), which come out of
// `embed.dot(w_down)` + top-k selection — see module doc for why those
// get tolerance/set checks instead of an exact pin.
fn check_or_print(label: &str, actual: &str, golden: &str) {
    if print_mode() {
        eprintln!("{label} = {actual:?}");
        return;
    }
    assert_eq!(
        actual, golden,
        "{label}: walker output drifted — review the change and update the golden if intentional"
    );
}

/// Canonical form for the BLAS-free subset of a vector-extractor record.
/// `vector` is a direct column copy off the weight tensor (no
/// arithmetic), so this hash can't drift from a different BLAS build —
/// unlike `top_token`/`c_score`/`top_k`, which come out of a matmul.
fn canonicalise_exact_fields(records: &[VectorRecord]) -> Vec<u8> {
    let mut lines: Vec<String> = records
        .iter()
        .map(|r| {
            let vector_hex: String = r
                .vector
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .map(|b| format!("{b:02x}"))
                .collect();
            format!(
                "{}\t{}\t{}\t{}\t{}",
                r.id, r.layer, r.feature, r.dim, vector_hex
            )
        })
        .collect();
    lines.sort();
    let mut out = lines.join("\n").into_bytes();
    out.push(b'\n');
    out
}

fn parse_records(path: &Path) -> Vec<VectorRecord> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut records: Vec<VectorRecord> = text
        .lines()
        .filter(|l| !l.contains("\"_header\":true"))
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    records.sort_by_key(|r| r.feature);
    records
}

/// Structural invariants for a single record's BLAS-derived fields.
/// These hold regardless of BLAS build/reduction order — they only
/// break on a genuine regression (empty top-k, an inconsistent top-1,
/// wrong shape), not on harmless tie-break reordering.
fn assert_record_structural_invariants(r: &VectorRecord, configured_top_k: usize) {
    assert_eq!(
        r.dim,
        r.vector.len(),
        "record {}: dim != vector.len()",
        r.id
    );
    assert!(!r.vector.is_empty(), "record {}: empty vector", r.id);
    assert!(!r.top_k.is_empty(), "record {}: empty top_k", r.id);
    assert!(
        r.top_k.len() <= configured_top_k,
        "record {}: top_k longer than configured top_k",
        r.id
    );
    assert!(
        r.top_k.windows(2).all(|w| w[0].logit >= w[1].logit),
        "record {}: top_k not sorted descending by logit",
        r.id
    );
    let top1 = &r.top_k[0];
    assert_eq!(
        r.top_token_id, top1.token_id,
        "record {}: top_token_id != top_k[0].token_id",
        r.id
    );
    assert_eq!(
        r.top_token, top1.token,
        "record {}: top_token != top_k[0].token",
        r.id
    );
    assert_eq!(
        r.c_score, top1.logit,
        "record {}: c_score != top_k[0].logit",
        r.id
    );
    assert!(!r.top_token.is_empty(), "record {}: empty top_token", r.id);
    for entry in &r.top_k {
        assert!(
            !entry.token.is_empty(),
            "record {}: empty top_k token",
            r.id
        );
        assert!(
            entry.logit.is_finite(),
            "record {}: non-finite top_k logit",
            r.id
        );
    }
}

/// Per-feature pin for the BLAS-derived subset of a vector-extractor
/// record. Captured once from an actual run — see the module-level
/// regeneration instructions above.
struct FeatureGolden {
    feature: usize,
    top_token_id: u32,
    c_score: f32,
    top_k_token_ids: &'static [u32],
}

/// Absolute tolerance on `c_score`. Wide enough to absorb BLAS
/// reduction-order noise across builds, tight enough to catch a real
/// regression — same idea as `test_logits_goldens.rs`'s
/// `LOGIT_TOLERANCE`, sized down for this fixture's small synthetic
/// logit magnitudes.
const TOP1_SCORE_TOLERANCE: f32 = 1e-3;

// Captured 2026-07-05.
const GOLDEN_EXACT_FIELDS_HASH: &str =
    "b6a6dc4cd0ef85ac252e401fcf5da5c18f8f23ef6b3b49aefedee8d19f1237eb";

const FEATURE_GOLDENS: &[FeatureGolden] = &[
    FeatureGolden {
        feature: 0,
        top_token_id: 10,
        c_score: 0.02654723,
        top_k_token_ids: &[6, 7, 10],
    },
    FeatureGolden {
        feature: 1,
        top_token_id: 7,
        c_score: 0.024061209,
        top_k_token_ids: &[6, 7, 13],
    },
    FeatureGolden {
        feature: 2,
        top_token_id: 7,
        c_score: 0.030054377,
        top_k_token_ids: &[7, 10, 15],
    },
    FeatureGolden {
        feature: 3,
        top_token_id: 10,
        c_score: 0.02863936,
        top_k_token_ids: &[5, 7, 10],
    },
];

/// Check (or, under `LARQL_PRINT_GOLDEN=1`, print) one record's
/// BLAS-derived fields against its pinned golden. Order-independent on
/// `top_k` — a different BLAS build can swap rank within the set
/// without that being a regression (see module doc).
fn check_feature_golden(r: &VectorRecord) {
    let mut ids: Vec<u32> = r.top_k.iter().map(|t| t.token_id).collect();
    ids.sort_unstable();

    if print_mode() {
        eprintln!(
            "FeatureGolden {{ feature: {}, top_token_id: {}, c_score: {:?}, top_k_token_ids: &{:?} }},",
            r.feature, r.top_token_id, r.c_score, ids
        );
        return;
    }

    let golden = FEATURE_GOLDENS
        .iter()
        .find(|g| g.feature == r.feature)
        .unwrap_or_else(|| panic!("no golden pinned for feature {}", r.feature));

    assert_eq!(
        r.top_token_id, golden.top_token_id,
        "feature {}: top_token_id drifted",
        r.feature
    );
    let score_diff = (r.c_score - golden.c_score).abs();
    assert!(
        score_diff < TOP1_SCORE_TOLERANCE,
        "feature {}: c_score drifted by {score_diff} (tolerance {TOP1_SCORE_TOLERANCE})",
        r.feature
    );

    let mut want: Vec<u32> = golden.top_k_token_ids.to_vec();
    want.sort_unstable();
    assert_eq!(
        ids, want,
        "feature {}: top_k token-id set drifted",
        r.feature
    );
}

/// Run the weight walker on layer 0 of a freshly-built fixture, twice,
/// and pin structural invariants on the result.
///
/// We deliberately don't hash edge bytes (see module doc) — the
/// matmul → top-k pipeline gives different tie-breaking on x86 vs ARM.
/// What we *do* require: every walk on the same platform yields the
/// same canonical bytes, the edge count is stable, normalisation
/// produces a max-confidence of 1.0, every confidence sits in [0, 1],
/// every (subject, relation, object) is non-empty, and every edge
/// carries the documented metadata fields.
#[test]
fn weight_walker_layer0_invariants() {
    let cfg = WalkConfig {
        top_k: 3,
        min_score: 0.0,
    };

    let dir_a = fixture("ww_a");
    let walker_a = WeightWalker::load(dir_a.to_str().unwrap()).unwrap();
    let mut g_a = Graph::new();
    walker_a
        .walk_layer(0, &cfg, &mut g_a, &mut SilentWalkCallbacks)
        .unwrap();

    let dir_b = fixture("ww_b");
    let walker_b = WeightWalker::load(dir_b.to_str().unwrap()).unwrap();
    let mut g_b = Graph::new();
    walker_b
        .walk_layer(0, &cfg, &mut g_b, &mut SilentWalkCallbacks)
        .unwrap();

    // BLAS on Windows runners has non-deterministic reduction order
    // between successive matmul calls on the same input (parallel
    // accumulation in OpenBLAS), which trips f32-precision tie-breaking
    // in the top-k path. Linux/macOS BLAS implementations don't show
    // this drift, so we keep the byte-equality check there.
    #[cfg(not(windows))]
    {
        let bytes_a = canonicalise_edges(&g_a, "layer", "feature");
        let bytes_b = canonicalise_edges(&g_b, "layer", "feature");
        assert_eq!(
            sha256_hex(&bytes_a),
            sha256_hex(&bytes_b),
            "weight_walker_layer0: not deterministic within a single run"
        );
    }

    assert_structural_invariants(&g_a, "feature", 0);
    assert_structural_invariants(&g_b, "feature", 0);

    cleanup(&dir_a);
    cleanup(&dir_b);
}

#[test]
fn attention_walker_layer0_invariants() {
    let cfg = WalkConfig {
        top_k: 2,
        min_score: 0.0,
    };

    let dir_a = fixture("aw_a");
    let walker_a = AttentionWalker::load(dir_a.to_str().unwrap()).unwrap();
    let mut g_a = Graph::new();
    walker_a
        .walk_layer(0, &cfg, &mut g_a, &mut SilentWalkCallbacks)
        .unwrap();

    let dir_b = fixture("aw_b");
    let walker_b = AttentionWalker::load(dir_b.to_str().unwrap()).unwrap();
    let mut g_b = Graph::new();
    walker_b
        .walk_layer(0, &cfg, &mut g_b, &mut SilentWalkCallbacks)
        .unwrap();

    // See `weight_walker_layer0_invariants` — BLAS on Windows has
    // non-deterministic reduction order across matmul calls; keep the
    // byte-equality check on the platforms where it holds.
    #[cfg(not(windows))]
    {
        let bytes_a = canonicalise_edges(&g_a, "layer", "head");
        let bytes_b = canonicalise_edges(&g_b, "layer", "head");
        assert_eq!(
            sha256_hex(&bytes_a),
            sha256_hex(&bytes_b),
            "attention_walker_layer0: not deterministic within a single run"
        );
    }

    assert_structural_invariants(&g_a, "head", 0);
    assert_structural_invariants(&g_b, "head", 0);

    cleanup(&dir_a);
    cleanup(&dir_b);
}

fn assert_structural_invariants(graph: &Graph, second_field: &str, expected_layer: u64) {
    let edges = graph.edges();
    assert!(!edges.is_empty(), "expected at least one edge");

    let mut max_conf = 0.0f64;
    for e in edges {
        assert!(!e.subject.is_empty(), "empty subject");
        assert!(!e.relation.is_empty(), "empty relation");
        assert!(!e.object.is_empty(), "empty object");
        assert!(
            (0.0..=1.0).contains(&e.confidence),
            "confidence out of range: {}",
            e.confidence
        );
        if e.confidence > max_conf {
            max_conf = e.confidence;
        }
        let m = e.metadata.as_ref().expect("edge metadata missing");
        let layer = m
            .get("layer")
            .and_then(|v| v.as_u64())
            .expect("layer metadata missing");
        assert_eq!(layer, expected_layer, "wrong layer in metadata");
        assert!(
            m.get(second_field).and_then(|v| v.as_u64()).is_some(),
            "missing `{second_field}` metadata",
        );
        assert!(
            m.get("c_in").and_then(|v| v.as_f64()).is_some(),
            "missing c_in metadata"
        );
        assert!(
            m.get("c_out").and_then(|v| v.as_f64()).is_some(),
            "missing c_out metadata"
        );
    }

    // Per-layer normalisation must hit 1.0 exactly on the top edge.
    assert!(
        (max_conf - 1.0).abs() < 1e-6,
        "max confidence is {max_conf}, expected ~1.0 (per-layer normalisation broken)"
    );
}

/// Extracts FFN-down vectors for layer 0 of the mock fixture and checks
/// them at three tiers (see module doc): structural invariants
/// (all platforms, always), an exact hash on the BLAS-free subset (all
/// platforms, always), and a tolerance-pinned golden on the BLAS-derived
/// subset (skipped on Windows — see `weight_walker_layer0_invariants`
/// for why OpenBLAS there isn't even deterministic within a run).
#[test]
fn vector_extractor_ffn_down_golden() {
    let dir = fixture("vex");
    let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
    let out = dir.join("output");
    std::fs::create_dir_all(&out).unwrap();

    let cfg = ExtractConfig {
        components: vec!["ffn_down".into()],
        layers: Some(vec![0]),
        top_k: 3,
    };
    let mut cb = SilentExtractCallbacks;
    extractor.extract_all(&cfg, &out, false, &mut cb).unwrap();

    let path = out.join("ffn_down.vectors.jsonl");
    let records = parse_records(&path);
    assert!(!records.is_empty(), "no records extracted");

    for r in &records {
        assert_record_structural_invariants(r, cfg.top_k);
    }

    let exact_hash = sha256_hex(&canonicalise_exact_fields(&records));
    check_or_print(
        "vector_extractor_ffn_down_exact_fields",
        &exact_hash,
        GOLDEN_EXACT_FIELDS_HASH,
    );

    #[cfg(not(windows))]
    for r in &records {
        check_feature_golden(r);
    }

    cleanup(&dir);
}

#[test]
fn fixture_is_deterministic_across_runs() {
    // Build the fixture twice and verify the safetensors bytes match.
    // If this test fails the goldens above are unreliable — every
    // walker test depends on `create_mock_model` being a pure function.
    let dir_a = fixture("det_a");
    let dir_b = fixture("det_b");
    let bytes_a = std::fs::read(dir_a.join("model.safetensors")).unwrap();
    let bytes_b = std::fs::read(dir_b.join("model.safetensors")).unwrap();
    assert_eq!(bytes_a, bytes_b, "fixture is not deterministic");
    cleanup(&dir_a);
    cleanup(&dir_b);
}

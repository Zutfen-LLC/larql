# Gemma 4 E2B F32 CPU Attention Semantics (ST4 + ST4A closeout)

Slice: `LARQL-INFERENCE-TRUST-001A-ST4` (closed by `…-ST4A`)

## Decision

**GREEN — F32 CPU attention and cache semantics proven.**

This slice proves F32 CPU attention and cache semantics using synthetic
fixtures. It does **not** prove official Gemma 4 logits or generated text.
A GREEN result means only that the F32 CPU path now implements the
required local/global attention ranges, absolute positions, and shared-KV
routing needed for the next external semantic comparison.

ST4A closed the remaining evidence gaps without reopening or expanding the
ST4 implementation scope: it added the missing canonical-loop prefill/decode
equivalence at the 512-window boundaries (511/512/513/1024), an explicit
Qwen2 full-attention regression, and a source append-count proof, and
corrected this report's ST4 metadata.

## Revisions (corrected by ST4A)

- ST4 work-start SHA: `15864226bd3ffb70a2cdc35ca3e52dafe76692c7`
- ST4 head SHA: `2889157f29ea85d690db5d08290bad41154a346f`
- ST4 merge commit (squash into main): `b2584c0acf7691d9963dff6a557ec7bac58755d9`
- ST4 CI: all 11 triggered workflows completed successfully.
- ST4A work-start SHA: `eb40807ef0c392cd34456406b936692c13741716`
- ST4A tested implementation commit SHA: `97ff07bb28e36420e07a015af0e4b32ccc736ef0`
  (the ST4A tests commit immediately before this evidence commit; checks out
  to the ST4 implementation plus the ST4A tests that verify it).

## What changed (F32-only)

- `larql-compute/attention/window.rs`: `intrinsic_attention_window`,
  `causal_attention_range`, `effective_window` (single source of truth).
- GQA prefill kernel: per-query windowed range (computes scores/softmax/V
  only over the in-window slice — never full-prefix-then-mask). Windowed
  wrappers for `gqa_attention`, `gqa_attention_with_weights`,
  `gqa_attention_with_all_weights`, `gqa_reduced_qk_all_weights`.
- `run_attention_block_core` + `run_attention_with_kv_backend`: thread the
  intrinsic window so diagnostics and production share one implementation.
- `CpuBackend::attention_step_windowed`: clip BEFORE attention so the
  current query never attends to an out-of-window key.
- `run_attention_block_decode_step_shared_backend`: shared-KV decode
  primitive (consumer Q only; source K/V; no consumer K/V/append).
- `kv_prefill_run` / `kv_decode_step_run` + dispatch helpers (`larql-inference`
  `kv_dispatch/helpers.rs`): route shared-KV, clip local sources to the
  window tail (prefill final pass), clip-before-attend decode.

## Explicitly excluded Q4_K paths (ST4 §18)

- `cached_decode_step_q4k` / `CpuQ4kCacheHandle`
- `predict_kquant_prefill` / `predict_kquant_decode_step`
- Q4K-direct attention (`decode_step_*_q4k_direct`)
- coarse `coarse_prefill` / `coarse_decode_step`

These pass `effective_window = None` (full causal) and do not route
shared-KV. The intrinsic Gemma 4 local/global and shared-KV parity must be
verified separately before they become semantic-oracle candidates. **Q4_K
remains excluded from the ST4/ST4A GREEN claim**; it may run as a
regression only.

## E2B local/global pattern + shared-KV map

- intrinsic window = 512
- layer pattern = SSSSG repeated seven times; final layer (34) = global
- shared local source = layer 13 (last non-shared sliding)
- shared global source = layer 14 (last non-shared global)
- layers 0–14 compute own K/V; layers 15–34 are shared consumers
- every shared layer points to a source of the same attention type
- complete shared-KV map result: PASS (5 mapping tests)

## Synthetic test results (ST4 §19, refreshed by ST4A)

| Group | Count | Result |
|-------|-------|--------|
| Pure attention ranges (t1–t10) | 10 | PASS |
| Prefill masking (t11–t17) | 7 | PASS |
| Cache behavior (t18–t24) | 7 | PASS |
| Shared KV (t29–t35) | 7 | PASS |
| Equivalence (t36, t37, t39, t40) | 4 | PASS |
| Regressions (t43, t44) | 2 | PASS |
| E2B 35-layer mapping | 5 | PASS |

Totals (current): `st4_attention_semantics` 26, `st4_shared_kv` 28
(includes 11 ST4A closeout tests), `test_architectures` st4_* 5.

- range mismatch count: 0
- cache-length mismatch count: 0
- absolute-position mismatch count: 0
- prefill/decode comparison: real canonical-loop numerical comparison at 4
  boundary positions (see ST4A closeout below for measured values).
- shared consumers tested: 2 (sliding L2, global L3 on the 4-layer fixture)
- independent shared-cache allocation count: 0
- shared consumer K/V append count: 0
- poison-weight test result: PASS (consumer K/V poison has no effect;
  source K/V mutation changes output)
- incompatible-geometry result: fails loudly (panic)
- Qwen2 regression result: **PASS** (full-attention regression; see ST4A §2)
- Q4_K regression result: PASS (regression-only, NOT part of the GREEN claim)
- semantic official-model run = false

## Verification commands and totals

```
cargo fmt --all -- --check                              # clean
cargo test -p larql-compute --test st4_attention_semantics   # 26 passed
cargo test -p larql-kv --test st4_shared_kv                  # 28 passed
cargo test -p larql-models --test test_architectures         # 90 passed (st4_* = 5)
cargo test -p larql-models                                   # green
cargo test -p larql-compute                                  # green
cargo test -p larql-inference                                # green
cargo test -p larql-kv                                       # green
cargo clippy -p larql-{models,compute,inference,kv} --all-targets -- -D warnings  # clean
```

CI status (ST4 PR #60): all 11 triggered workflows completed successfully.

---

## ST4A closeout

- work-start SHA: `eb40807ef0c392cd34456406b936692c13741716`
- tested implementation commit SHA: `97ff07bb28e36420e07a015af0e4b32ccc736ef0`
  (the ST4A tests commit immediately before this evidence commit; ST4A is
  tests + evidence only — no attention implementation change).
- new fixtures: `make_synthetic_e2b_like_weights_random_window512`
  (4-layer E2B-like Gemma 4 with a 512-token intrinsic window) and
  `make_qwen2_test_weights` (Qwen2 full-attention with Q/K/V biases).

### Boundary positions tested

511, 512, 513, 1024 — the 512-window boundary and the first clipped
position, plus a far-clipped position.

### Per-position canonical prefill/decode equivalence

Route A = one-shot `kv_prefill_run` of tokens `0..=target`; Route B =
`kv_prefill_run` of `0..target` then one `kv_decode_step_run` at the true
absolute position. Absolute tolerance `1e-5`.

| target | hidden max abs | hidden max rel | local K/V tail max abs | local len (A=B) | global len (A=B) | abs position (B) |
|--------|---------------:|---------------:|-----------------------:|----------------:|-----------------:|-----------------:|
| 511    | 1.073e-6       | 1.023e-6       | 0                      | 512             | 512              | 512              |
| 512    | 0              | 0              | 0                      | 512             | 513              | 513              |
| 513    | 0              | 0              | 0                      | 512             | 514              | 514              |
| 1024   | 0              | 0              | 0                      | 512             | 1025             | 1025             |

The largest hidden-state difference across all four positions is `1.073e-6`
(target 511); positions 512/513/1024 are bit-exact between routes. The
local source K/V tail is bit-exact (`0`) at every position, confirming
both routes hold the same absolute RoPE positions after clipping.

### Local cache evidence

Every target yields a 512-row local source cache in both routes, matching
the intrinsic window:

```
target 511  → 512 rows, absolute range 0..=511
target 512  → 512 rows, absolute range 1..=512
target 513  → 512 rows, absolute range 2..=513
target 1024 → 512 rows, absolute range 513..=1024
```

### Global cache evidence

The global (full-attention) source retains the entire prefix — `target+1`
rows in both routes (512 / 513 / 514 / 1025).

### Absolute-position evidence

After the decode, `cache.next_position == target+1` (512 / 513 / 514 /
1025), not the clipped 512 cache length. The bit-exact local K/V tail
between Route A (natural positions) and Route B (`next_position`-derived
RoPE) is the direct proof that the absolute RoPE position is the true
target position.

### Source append deltas (per decode token, 4-layer E2B-style fixture)

```
local source   → +1 row (clipped to its intrinsic window)
global source  → +1 row (unbounded)
local consumer → 0 rows
global consumer → 0 rows
```

Invariant holds on every one of 6 consecutive decode steps.

### Shared consumer append count

0 — shared consumer layers hold no independent cache before, during, or
after any decode step (verified by `cache.layers[c].is_none()`).

### Qwen2 regression

PASS. On a real Qwen2 architecture (`family == "qwen2"`, Q/K/V biases):

- `intrinsic_attention_window == None` on every layer;
- prefill of 600 tokens retains the full 600-row causal prefix (no
  clipping) on every layer;
- decode grows the cache by exactly one row per token (5 steps);
- no layer activates Gemma 4 shared-KV routing (`kv_shared_source_layer == None`);
- old-key sensitivity: zeroing the K/V of position 0 changes the
  position-600 decode output by `abs = 3.679e-4`, `rel = 1.334e-2`. Position
  0 lies outside a 512-token local window for position 600 (window =
  89..600), so this change proves Qwen2 attends to the early key under full
  attention.

### Q4_K scope statement

Excluded from the ST4/ST4A GREEN claim. Q4_K paths run `effective_window
= None` and do not route shared-KV; they pass as regressions only and
require separate semantic verification before becoming oracle candidates.

### Decision

**GREEN.** All 15 ST4A decision-gate criteria pass; the canonical
prefill/decode outputs are numerically equivalent (≤ `1.073e-6` abs) at
511/512/513/1024, local/global cache lengths and absolute positions are
correct, Qwen2 full attention is regressed, source append counts are
exact, and Q4_K remains outside the claim.

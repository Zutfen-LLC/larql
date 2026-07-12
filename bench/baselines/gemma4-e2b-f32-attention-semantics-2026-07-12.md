# Gemma 4 E2B F32 CPU Attention Semantics (ST4)

Slice: `LARQL-INFERENCE-TRUST-001A-ST4`

## Decision

**GREEN — F32 CPU attention and cache semantics proven.**

This slice proves F32 CPU attention and cache semantics using synthetic
fixtures. It does **not** prove official Gemma 4 logits or generated text.
A GREEN result means only that the F32 CPU path now implements the
required local/global attention ranges, absolute positions, and shared-KV
routing needed for the next external semantic comparison.

## Revisions

- Work-start SHA: `15864226bd3ffb70a2cdc35ca3e52dafe76692c7`
- PR base SHA: `15864226bd3ffb70a2cdc35ca3e52dafe76692c7`
- Head SHA: _filled at PR head_

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
verified separately before they become semantic-oracle candidates.

## E2B local/global pattern + shared-KV map

- intrinsic window = 512
- layer pattern = SSSSG repeated seven times; final layer (34) = global
- shared local source = layer 13 (last non-shared sliding)
- shared global source = layer 14 (last non-shared global)
- layers 0–14 compute own K/V; layers 15–34 are shared consumers
- every shared layer points to a source of the same attention type
- complete shared-KV map result: PASS (5 mapping tests)

## Synthetic test results (ST4 §19)

| Group | Count | Result |
|-------|-------|--------|
| Pure attention ranges (t1–t10) | 10 | PASS |
| Prefill masking (t11–t17) | 7 | PASS |
| Cache behavior (t18–t24) | 7 | PASS |
| Shared KV (t29–t35) | 7 | PASS |
| Equivalence (t36, t37, t39, t40) | 4 | PASS |
| Regressions (t43, t44) | 2 | PASS |
| E2B 35-layer mapping | 5 | PASS |

- range mismatch count: 0
- cache-length mismatch count: 0
- absolute-position mismatch count: 0
- prefill/decode comparison count: 4 boundary positions
- maximum absolute error: ≤ 1e-5 (GQA primitive parity; bit-exact where
  operation ordering is identical)
- shared consumers tested: 2 (sliding L2, global L3 on the 4-layer fixture)
- independent shared-cache allocation count: 0
- shared consumer K/V append count: 0
- poison-weight test result: PASS (consumer K/V poison has no effect;
  source K/V mutation changes output)
- incompatible-geometry result: fails loudly (panic)
- Qwen regression result: N/A (no Qwen fixture in the CPU F32 unit suite;
  conventional full-attention retained via `intrinsic_window = None`)
- Q4_K regression result: PASS (regression-only, not claiming ST4 coverage)
- semantic official-model run = false

## Verification commands and totals

```
cargo fmt --all -- --check                              # clean
cargo test -p larql-models --lib                        # 425 passed
cargo test -p larql-compute --lib                       # 756 passed
cargo test -p larql-inference --lib                     # 1290 passed
cargo test -p larql-kv --lib                            # 765 passed
cargo test -p larql-vindex --lib                        # 1154 passed
cargo test -p larql-cli --bins                          # 243 passed
cargo test -p larql-compute --test st4_attention_semantics   # 18 passed
cargo test -p larql-kv --test st4_shared_kv                  # 17 passed
cargo test -p larql-models --test test_architectures (st4_*) # 5 passed
cargo clippy -p larql-{models,compute,inference,kv,vindex,cli} --all-targets -- -D warnings  # clean
cargo test -p larql-compute q4k / larql-inference q4k / larql-vindex q4k  # green (regression)
cargo build -p larql-cli --release                      # passed
```

CI status: pending PR creation (all local checks green).

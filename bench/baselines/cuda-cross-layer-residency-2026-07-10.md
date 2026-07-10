# CUDA cross-layer hidden-state residency (GPU-007 / B2) — 2026-07-10

> **Status: COMPLETE — hardware-validated on RTX 3090 (sm_86, NVRTC 12.4).**
>
> Validation of `feat/cuda-cross-layer-hidden-state-residency` (commit
> `f9f0fde0`) on `3090rig` (RTX 3090, 24 GB VRAM, driver 550.163.01):
> **161/161 tests green** (160 lib + 1 hardware_probe) with default settings
> (no env overrides). All 7 resident_hidden tests pass natively on real
> hardware, including the mixed-eligibility transition test
> (Host→Device→Host→Device→final Host within one decode step).
> Three-repeat stability confirmed. Release-mode focused tests pass.
> CPU/CUDA e2e parity confirmed on the synthetic Q4_K vindex.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| Compute capability | sm_86 (NVRTC target compute_86) |
| Driver | 550.163.01 |
| CUDA/NVRTC | 12.4 |
| Rust | 1.96.1 (31fca3adb 2026-06-26) |
| Cargo | 1.96.1 (356927216 2026-06-26) |
| Branch | `feat/cuda-cross-layer-hidden-state-residency` |
| Commit | `f9f0fde0185cb2cdccf413cf7d3b8f5d11f4003a` |
| VRAM | 24,576 MiB (full, no other processes) |
| LARQL_* env vars | None (clean defaults) |

## Static validation

| Check | Result |
|---|---|
| `cargo fmt --all --check` | PASS |
| `cargo check -p larql-compute-cuda` | PASS |
| `cargo check -p larql-inference --features cuda` | PASS |
| `cargo clippy -p larql-compute-cuda --tests -- -D warnings` | PASS |
| `cargo clippy -p larql-models --tests -- -D warnings` | PASS |

## Hardware probe

`hardware_probe_native_runtime_is_active`: **PASS**
- `supports(QuantMatVec) = true`
- `supports(DecodeToken) = true`
- RTX 3090 / sm_86 detected, NVRTC target compute_86
- 21 native kernels loaded: q4k_matvec, q6k_matvec, q4k_matmul, q6k_matmul,
  q4k_dual_matvec, f32_gemv, f16_gemv, q4_matvec, q4_vecmat, kv_append,
  rms_norm, rms_norm_heads, geglu_silu, geglu_gelu_tanh, activation_silu,
  activation_gelu_tanh, residual_add, rope, decode_attention, prefill_attention

## GPU-007 resident_hidden test results (7 tests, all PASS)

| Test | Result |
|---|---|
| `resident_hidden_decode_matches_cpu_reference_after_prefill` | PASS |
| `resident_hidden_multi_token_decode_matches_cpu_reference` | PASS |
| `resident_hidden_diag_surfaces_in_device_info` | PASS |
| `resident_hidden_fallback_when_ineligible` | PASS |
| `resident_hidden_decode_keeps_kv_lockstep` | PASS |
| `resident_hidden_runs_across_consecutive_layers` | PASS |
| `resident_hidden_mixed_eligibility_transitions_and_matches_cpu` | PASS |

### Key assertions confirmed

- **Eligible path runs**: resident-hidden `uses > 0` in eligible tests (not
  merely fallbacks). The `ensure_device` upload step correctly transitions
  the hidden state from Host→Device at the first eligible layer.
- **Multi-token parity**: every decode step below 1e-3, no progressive drift.
- **Consecutive-layer residency**: 4-layer test reports `uses >= num_layers`,
  output parity below 1e-3.
- **Forced fallback**: below-gate path matches CPU below 1e-3, `fallbacks > 0`.
- **Mixed transition**: Host→Device→Host→Device→final Host within one decode.
  Both `uses > 0` and `fallbacks > 0`. Output below 1e-3. KV lengths in lockstep.
- **KV lockstep**: after resident-hidden decode, `host_kv_len ==
  kv_cache_len_native == kv_cache_len`, expected = prompt_len + decoded tokens.
- **Tolerance**: unchanged 1e-3 ASTAB/GPU-006 parity tolerance throughout.

## Three-repeat stability

| Run | Result |
|---|---|
| Run 1 | 7/7 PASS (17.20s) |
| Run 2 | 7/7 PASS (17.11s) |
| Run 3 | 7/7 PASS (17.51s) |

No stale device buffers, CudaSlice ownership/drop defects, or cache state
leakage detected across runs.

## ASTAB-001 regression

| Test | Result |
|---|---|
| `decode_token_matches_cpu_reference_when_runtime_available` | PASS |
| `multi_token_decode_matches_cpu_reference` | PASS |

Tolerance preserved at 1e-3, no CPU fallback.

## GPU-006 resident_kv regression (8 tests, all PASS)

| Test | Result |
|---|---|
| `resident_kv_decode_lockstep_after_prefill_then_one_decode` | PASS |
| `resident_kv_decode_lockstep_across_multi_token_decode` | PASS |
| `resident_kv_decode_matches_cpu_reference_after_prefill` | PASS |
| `resident_kv_decode_reads_only_valid_rows` | PASS |
| `resident_kv_fallback_to_upload_when_no_cache` | PASS |
| `resident_kv_multi_token_decode_matches_cpu_reference` | PASS |
| `resident_kv_reset_clears_both_caches` | PASS |
| `resident_kv_truncate_keeps_lockstep` | PASS |

No regression from GPU-007.

## State-dump and fallback regression

| Test | Result |
|---|---|
| `decode_token_with_state_dump_respects_mask` | PASS |
| `device_info_reports_native_or_fallback_status` | PASS |
| `resident_kv_fallback_to_upload_when_no_cache` | PASS |
| `resident_hidden_fallback_when_ineligible` | PASS |

## Full diagnostic suite (LARQL_GPU_DIAG=1)

- **160 lib tests + 1 hardware_probe = 161/161 PASS**, 0 failed, 19.80s
- No CUDA launch errors, no panics, no invalid memory access
- No scaffold-only execution

## Full default suite (no diagnostics)

- **160 lib + 1 probe = 161/161 PASS**, 0 failed, 19.91s
- No diagnostic noise, no threshold overrides, no env-specific workaround

## Release-mode focused validation

| Group | Result |
|---|---|
| `resident_hidden` (7 tests) | 7/7 PASS (3.46s) |
| `resident_kv` (8 tests) | 8/8 PASS (0.23s) |

No release-only ownership, overflow, indexing, or synchronization defects.

## End-to-end validation

- Vindex: `~/models/qwen2.5-3b-q4k.vindex` (synthetic Q4_K test fixture)
- CUDA and CPU produce identical output for the same deterministic prompt
- The vindex is a synthetic test fixture (not a trained model), so output is
  not meaningful text, but CPU/CUDA parity is confirmed
- The `larql` CLI required `--features cuda` to enable the CUDA backend

## Defects found and fixes applied

Two defects were identified during validation and fixed in commits 97cecfb1
and f9f0fde0:

1. **Missing device upload (pipeline.rs)**: `host_decode_token_resident`
   initialized the hidden state as `DecodeHiddenState::Host`, but
   `host_attention_block_device_resident` only accepts `Device` input. The
   resident path never fired because there was no Host→Device upload step
   when entering an eligible layer. Fixed by adding `ensure_device()` to
   `DecodeHiddenState` and calling it before the device-resident attention
   block.

2. **Test fixture below activation gate (test_fixtures.rs + lib.rs)**: The
   default Q4_K test fixture has `inter = 256`, but
   `DEFAULT_ACTIVATION_NATIVE_MIN_ELEMS = 8192`. The resident-hidden
   eligibility check includes `native_activation_worthwhile(inter)` which
   fails at `256 < 8192`, making every layer ineligible. Fixed by adding
   `make_test_q4k_weights_inter(inter)` with `inter = 8192` (Q4_K-safe:
   32 × 256) for the resident-hidden tests.

3. **Mixed-eligibility test parity bug (lib.rs)**: The test assumed zeroing
   norm weights makes RmsNorm and LayerNorm produce the same result. With
   Gemma 3's `norm_offset = 1.0`, RmsNorm with zero weights produces `x/rms`
   (nonzero), while LayerNorm with zero weights produces `0.0`. Fixed by
   using a different ineligibility mechanism that preserves CPU reference
   parity.

All fixes are test-only or infrastructure-only — no production thresholds
were altered, no tolerance was loosened.

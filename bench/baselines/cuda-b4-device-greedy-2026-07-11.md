# LARQL-GPU-B4 — Resident final norm, Q4_K lm-head, device-side greedy selection

**Date:** 2026-07-11  **Host:** NVIDIA RTX 3060 12 GB (sm_86), CUDA driver 610.43.03, NVRTC 12.4.127
**Slice:** `LARQL-GPU-B4`  **Vindex:** production-default Qwen2.5-3B Q4_K_M
**Decision:** **opt-in** (`LARQL_CUDA_DEVICE_GREEDY=1`); kill switch `LARQL_LM_HEAD_SKIP_Q4K=1`.

---

## 1. Verified real-model lm-head representation

The production vindex carries a verified **Q4_K** lm-head (not f16/f32):

| field | value |
|---|---|
| `lm_head_kquant.bin` | 175 030 272 bytes |
| `weight_manifest.json` kind | `tensor_q4k` |
| shape | `[151936, 2048]` |
| `lm_head_representation()` (runtime) | `Q4K { bytes, hidden=2048, physical=151936, logical=151643 }` |
| final norm | `RmsNorm`, key `norm.weight`, length 2048, eps 1e-6, offset 0.0 |
| candidate width | 5 (`LMHEAD_TOPK_GREEDY`) |

---

## 2. Implemented architecture (additive, no semantic change to `decode_token`)

* **Substrate types** (`larql-compute/src/backend/greedy.rs`): `GreedyDecodeOutput`
  (`DevicePick` | `HostHidden`), `DeviceGreedyPick`, `GreedyQ4kHeadSpec` (with
  `validate()` + `logical_rows()`). Object-safe through `&dyn ComputeBackend`.
* **Capability** `DeviceGreedyLmHead`; **additive method**
  `DecodeBackend::decode_token_greedy_q4k` (default → `HostHidden`).
* **Vindex accessor** `VectorIndex::lm_head_representation()` → `LmHeadRepresentation`
  (`Q4K{..}` | `F16` | `F32` | `Absent`) — the single place the "active lm-head is
  Q4_K" assumption is verified.
* **CUDA device path** (`pipeline::host_decode_token_greedy_q4k`): the resident
  layer loop was extracted into `host_decode_layers_resident` so both the
  host-readback finalizer and the B4 device finalizer consume the final hidden in
  its native residency (owned `CudaSlice` / graph-arena slot / host fallback).
  The device chain: final RMSNorm (`launch_rms_norm_into_dev`) → Q4_K lm-head GEMV
  over `[0, logical)` rows (`launch_q4k_matvec_into`) → two-stage top-K reduction
  → fixed-size result readback.
* **Reduction kernels** (`greedy_topk_partial` + `greedy_topk_final`): bounded
  two-stage, k sequential argmax passes (k ≤ 8), strict `>` tie-break (lowest
  index wins, matching host), non-finite scores treated as `-inf` (NVRTC-safe
  `__int_as_float(0xff800000)` — NVRTC does not define `INFINITY`), **no
  floating-point atomics**.
* **Generation-scoped workspace** (`greedy_workspace.rs`): normalised-hidden,
  scores, partial, result buffers reused across tokens; rebuilt on shape change,
  dropped at `reset_kv_cache`.
* **KV-append invariant**: on any device-chain failure after the layers ran, the
  already-computed hidden is materialised once and returned as `HostHidden`
  (never `None` unless even that readback fails) — the token step is never re-run.
* **Inference routing** (`decode_loop`): eligibility resolved once
  (`is_greedy && !has_repetition_penalty && !skip_q4k && device_greedy_enabled &&
  supports(DeviceGreedyLmHead) && !has_per_layer_ffn && spec.is_some()`). `DevicePick`
  emits via `emit_preselected_greedy` (no re-sampling); `HostHidden` runs the exact
  existing host norm → lm-head → sample path.

## 3. Old vs new data flow

**Old (host boundary):** resident decode → final hidden DtoH → host RMSNorm →
normalised-query HtoD → Q4_K GEMV (device) → **full score DtoH (151 936 × 4 B)**
→ host top-K → host sample.

**New (B4, eligible plain greedy):** resident decode → final hidden **stays
resident** → device RMSNorm → Q4_K GEMV over logical rows (device) → device
two-stage top-K reduction → **fixed-size result DtoH (5 × 8 B = 40 B)** → emit.

---

## 4. Pre-implementation timing decomposition (instrumented baseline, B4 off)

| stage | ms/tok | share |
|---|---|---|
| GPU fwd | 95.5 | 88.1% |
| final_norm (host) | 0.006 | 0.0% |
| lm_head (host: norm+HtoD+GEMV+**full DtoH**+topK) | 12.86 | 11.9% |
| hidden readback (the DtoH B4 removes) | 2.84 | — |

> Per B4 §2.2: the Q4_K GEMV is already device-resident — the ~12.7 ms `lm_head`
> stage is **not** fully removable. The directly-removable subsegment is the
> final-hidden readback (2.84 ms/tok) + the full score DtoH + host top-K.

---

## 5. Correctness gate — PASS

| check | result |
|---|---|
| Token-ID sequence parity (3 reps × 40 tokens, baseline vs B4-on) | **IDENTICAL** |
| No token id ≥ logical_vocab (151643) | unit-tested + structural |
| Five-hit callback probability preserved (`softmax_prob` over returned candidates) | unit-tested |
| Both graph modes pass the full CUDA suite | **215/215** each |
| Non-greedy / diagnostic paths retain the old implementation | unit-tested |

## 6. Structural gate — PASS (per engaged steady-state decode token)

| counter | baseline | B4 on |
|---|---|---|
| host final_norm ms/tok | 0.006 | **0.000** |
| host lm_head ms/tok | 12.66 | **0.000** |
| final-hidden readback ms/tok | 2.839 | **0.000** |
| dtoh MiB/tok | 13.193 | 12.410 (−0.78) |
| dtoh copies/tok | 146.9 | 145.6 |
| device-greedy engagement | — | **100%** of eligible steps |
| fallbacks / failures | — | 0 / 0 |
| per-token lm-head weight upload | 0 (cached) | 0 (cached) |

---

## 7. Performance gate — **NOT MET** (≤ 1% in both graph modes)

5 reps × 79 measured decode steps, same prompt/model/GPU, greedy, uninstrumented
release (source of truth), MAD ≤ 0.07 ms.

| mode | graphs | B4 | p50 median (ms) | MAD |
|---|---|---|---|---|
| Baseline A | 0 | 0 | 108.31 | 0.01 |
| B4 A | 0 | 1 | 108.72 | 0.05 |
| Baseline B | 1 | 0 | 108.43 | 0.07 |
| B4 B | 1 | 1 | 107.97 | 0.01 |

| comparison | Δ p50 (ms) | Δ p50 (%) |
|---|---|---|
| graph-off (A) | +0.41 | +0.38% (B4 slightly slower) |
| graph-on (B) | −0.46 | −0.42% (B4 slightly faster) |

The run-to-run noise (MAD ≈ 0.01–0.07 ms) is tiny, so the ~0.4 ms deltas are
*statistically* real, but they are **~0.4%** — well under the 1% gate in either
graph mode. **Why neutral:** the Q4_K lm-head GEMV was already device-resident
(B4 §2.2), so eliminating the host boundary (final-hidden DtoH 2.84 ms/tok +
full-score DtoH + host top-K) only removes ~0.4 ms of wall time — the device GEMV
+ new reduction kernel absorb the rest.

## 8. Default-on vs opt-in decision

**Opt-in.** Per B4 §14.3: the 1% gate is not met, so B4 is **not** default-on.
It is engaged only when `LARQL_CUDA_DEVICE_GREEDY=1` is set (and all dynamic
eligibility conditions hold). `LARQL_CUDA_GRAPHS` default is unchanged (still
opt-in).

## 9. Test totals

| suite | result |
|---|---|
| `cargo test -p larql-compute --lib` | 750 pass |
| `cargo test -p larql-vindex --lib` | 1131 pass |
| `cargo test -p larql-inference --lib` (generate) | 279 pass |
| `LARQL_CUDA_GRAPHS=0 cargo test -p larql-compute-cuda --lib` | **215 pass** |
| `LARQL_CUDA_GRAPHS=1 cargo test -p larql-compute-cuda --lib` | **215 pass** |
| `cargo build -p larql-cli --release --features cuda` | OK |
| fmt + clippy (all 4 modified crates + cli) | clean |

## 10. Remaining host boundaries

* Non-greedy sampling, repetition penalties, constrained decoding → host path
  (by design).
* f16/f32 lm-head, LayerNorm final norm, MoE per-layer FFN → host path.
* First-token prefill output → host path (B4 is decode-only).
* The Q4_K GEMV itself stays a device op (not a "host boundary" but the dominant
  residual lm-head cost).

## 11. Recommended follow-up (clearly separated from landed scope)

A **fused Q4_K GEMV + top-K reduction kernel** (B4 §8 explicit follow-up) could
remove the logical-vocab score materialisation entirely — but the measured ~0.4%
net shows the GEMV-to-reduction handoff is already cheap, so a fused kernel is
**not** justified by this evidence alone. The higher-value follow-up is the
resident-hidden path for the **mixed Q4_K_M attention triple** (gate/up/down are
Q4_K_M but the attention V/QK-norm path still reads back), which the D6 report
ranked above B4.

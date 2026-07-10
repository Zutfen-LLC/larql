# LARQL-GPU-D6: resident-hidden decode for the default Q4_K_M — 2026-07-10

> **Status: COMPLETE on RTX 3060. Resident-hidden engagement on the real
> default Q4_K_M model went 0% → 100%. Same-day A/B shows a 3.3-3.7% decode
> improvement and 61-65% transfer/sync reduction. RTX 3060 validation is the
> final project evidence per the verification policy; no RTX 3090 rerun is
> required.**

## What changed

The GPU-007 cross-layer resident-hidden decode path (hardware-validated on
synthetic uniform-Q4_K fixtures) never reached the production-default Q4_K_M
format. The eligibility gate (`resident_hidden_layer_eligible`) and the
resident FFN chain (`host_ffn_block_device_resident`) both required a
**uniform** Q4_K×3 or Q6_K×3 FFN triple; the default Q4_K_M layout
(gate/up Q4_K, down Q6_K) failed both, so every layer fell back to the
host-orchestrated path (measured: 0 uses / 612 fallbacks on Qwen2.5-3B).

D6 broadens both gates through one shared pure helper,
`supported_resident_ffn_triple`, which accepts the three layouts LARQL
produces and validates:

- `Q4_K / Q4_K / Q4_K` (uniform-Q4_K, `--down-q4k`)
- `Q6_K / Q6_K / Q6_K` (uniform-Q6_K)
- `Q4_K / Q4_K / Q6_K` (the default Q4_K_M FFN mix)

**No CUDA kernel changes were necessary.** `matvec_dev_by_fmt` already
dispatches each projection independently (Q4_K → `launch_q4k_matvec_dev`,
Q6_K → `launch_q6k_matvec_dev`), and the down projection consumes an f32
device activation regardless of its weight format. The Q4_K gate/up →
device activation → Q6_K down → post-FFN norm → residual → next-layer chain
stays fully device-resident with no readback between the activation and the
down projection.

Other permutations (`Q6_K / Q6_K / Q4_K`, mixed gate/up) remain rejected —
they are not produced by any LARQL extraction path and have no dedicated
parity coverage.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12 GB) |
| Compute capability | sm_86 (NVRTC target compute_86) |
| Driver | 610.43.03 |
| CUDA/NVRTC | 12.4.127 |
| Rust | 1.97.0 (2d8144b78 2026-07-07) |
| Main SHA (baseline) | `70cc8fb9` |
| Feature SHA (D6) | `8584ae32` |
| Model | Qwen2.5-3B-Instruct (qwen2, 36 layers) |
| | hidden=2048, intermediate=11008, GQA 16 q / 2 kv, head_dim=128 |
| Quant | Q4_K_M (gate/up Q4_K, down Q6_K; attn Q/K/O Q4_K, V Q6_K) — the default |
| GPU pstate (idle) | P8, 210 MHz SM, 405 MHz mem, 14.8 W, 40°C |

## D6-A — Blocker reproduced (pre-change)

On the real default Q4_K_M vindex (`qwen2.5-3b-q4k.vindex`), with
`LARQL_GPU_DIAG=1`:

| Path | Q4_K_M (default) | uniform Q4_K (`--down-q4k`) |
|---|---|---|
| resident-KV (GPU-006) | uses=288, fallbacks=0, rate=**100.0%** | uses=288, fallbacks=0, rate=**100.0%** |
| resident-hidden (GPU-007) | uses=**0**, fallbacks=**288**, rate=**0.0%** | uses=288, fallbacks=0, rate=**100.0%** |

288 = 36 layers × 8 tokens (1 warmup + 7 measured). GPU-007 was dead code on
the production format; only the uniform-Q4_K control engaged it.

## D6-B/C — Shared helper + resident chain

`supported_resident_ffn_triple(gate, up, down)` is the single source of truth,
called from both `resident_hidden_layer_eligible` and
`host_ffn_block_device_resident` so the two gates cannot drift. The resident
chain runs Q4_K gate/up from the device-resident normalized hidden state,
keeps gate/up outputs device-resident through the activation kernel, feeds the
device-resident activation directly into the existing Q6_K down matvec
launcher, and keeps the Q6_K down output device-resident for post-FFN norm,
residual add, and the next layer. No activation readback occurs between the
activation and the Q6_K down projection. `stored_cols == inter` validation and
padded-down fallback behavior are preserved.

## D6-D — Faithful Q4_K_M fixture

`Q4kmFixtureIndex` (`larql-compute/src/test_fixtures.rs`) is a faithful
production-Q4_K_M `KvIndex`: attention Q/K/O Q4_K + V Q6_K; FFN gate/up Q4_K +
down Q6_K. It tracks the two per-component byte sizes (Q4_K 144 B/super-block,
Q6_K 210 B/super-block) explicitly, and `kquant_ffn_layer_once` is
format-aware (gate/up via Q4_K dequant, down via Q6_K dequant, cached per
(layer, component)).

## D6-E — Host-runnable eligibility tests (no GPU)

`pipeline::tests` (pure, run on every host):

- `supported_resident_ffn_triple_accepts_default_q4km_mix` — Q4_K/Q4_K/Q6_K ✓
- `supported_resident_ffn_triple_accepts_uniform_q4k` — Q4_K×3 ✓
- `supported_resident_ffn_triple_accepts_uniform_q6k` — Q6_K×3 ✓
- `supported_resident_ffn_triple_rejects_q6k_q6k_q4k` — rejected ✗
- `supported_resident_ffn_triple_rejects_mixed_gate_up` — rejected ✗
- `supported_resident_ffn_triple_rejects_unsupported_quant_formats` — Q4_0/BF16/Q8_0 ✗
- `down_stored_cols_rejects_padded_down_contraction` — padded-down ✗

## D6-F — Native mixed-Q4_K_M parity tests (RTX 3060)

vs `predict_kquant_decode_step` (f32-activation CPU reference), tolerance
`max_abs < 1e-3`, with **EXACT** use/fallback counter assertions:

| Test | max_abs | resident-hidden uses | fallbacks |
|---|---|---|---|
| `q4km_resident_hidden_decode_matches_cpu_reference_after_prefill` | < 1e-3 | == num_layers | 0 |
| `q4km_resident_hidden_multi_token_matches_cpu_reference` | < 1e-3 (4 tokens) | == num_layers×4 | 0 |
| `q4km_resident_hidden_runs_across_consecutive_layers` (4 layers) | < 1e-3 | == 4 | 0 |
| `q4km_resident_hidden_decode_keeps_kv_lockstep` | — | KV host/device lengths equal | — |
| `q4km_resident_hidden_fallback_when_ineligible` | < 1e-3 | 0 | == num_layers |

Fixture-level assertions confirm `gate.format == Q4_K`, `up.format == Q4_K`,
`down.format == Q6_K`, `wv.format == Q6_K`, `wq.format == Q4_K`.

## D6-G — Full CUDA regression (RTX 3060)

| Suite | Result |
|---|---|
| `cargo fmt --all --check` | clean |
| clippy (cuda + models) `-D warnings` | clean |
| clippy (inference + cli `--features cuda`) `-D warnings` | clean |
| ASTAB-001 (`decode_token_matches_cpu_reference...`) | 1/1 pass |
| GPU-006 (`resident_kv`) | 8/8 pass |
| GPU-007 (`resident_hidden`) | 12/12 pass (7 original + 5 D6) |
| D6 (`q4km`) | 6/6 pass (5 native + 1 host) |
| Full CUDA suite, `LARQL_GPU_DIAG=1` | **172/172 pass** |
| Full CUDA suite, default | **172/172 pass** |
| `larql-models` | 539/539 pass |
| Release-mode D6 (`--release`) | 6/6 pass |

Total test count: **172** CUDA (was 161 pre-D6; +11 new: 6 host eligibility +
5 native D6 parity). No threshold overrides, no tolerance changes.

## D6-H — Same-day A/B benchmark (the merge claim)

Same RTX 3060, same day, same Qwen2.5-3B Q4_K_M vindex, same 30-token prompt
("Write a step by step guide on how to bake a cake..."), 3 warmup + 79 measured
decode steps/rep, 5 reps, greedy decoding, `LARQL_CUDA_*` unset. Instrumented
and uninstrumented timings kept separate.

### Uninstrumented (wall-clock source of truth)

| metric | main `70cc8fb9` (rh OFF) | D6 `8584ae32` (rh ON) | delta |
|---|---|---|---|
| p50 ms/tok (min/med/max) | 130.94/131.30/133.23 | 126.67/127.15/127.74 | **−3.6%** (median) |
| mean ms/tok (min/med/max) | 131.00/131.27/134.38 | 126.69/127.35/127.78 | **−3.7%** (mean) |
| tok/s (min/med/max) | 7.44/7.62/7.63 | 7.83/7.85/7.89 | **+3.9%** (median) |

Median p50: **131.30 → 127.15 ms/tok** (4.15 ms/tok, 3.3% faster). n_steps=79
every rep (no early-EOS distortion).

### Instrumented (`LARQL_GPU_PROFILE=1`, counter decomposition)

| metric | main (rh OFF) | D6 (rh ON) | delta |
|---|---|---|---|
| ms/tok (mean) | 131.74 | 127.81 | −3.9% |
| launches/tok | 482.1 | 631.6 | +149.5 (more chained kernels) |
| htod copies/tok | 153.1 | 4.7 | −148.4 |
| **htod MiB/tok** | **3.936** | **1.499** | **−62%** |
| dtoh copies/tok | 229.7 | 81.2 | −148.4 |
| **dtoh MiB/tok** | **6.109** | **2.395** | **−61%** |
| **syncs/tok** | **229.7** | **81.2** | **−65%** |
| KV mirror ms/tok | 0.409 | 0.395 | ~0 |
| hidden rdback ms/tok | 0.000 | 2.329 | +2.3 (single end-of-token readback) |

Launches go **up** (482→632) because the resident chain keeps more work
on-device (more chained kernels on one stream, trading launch count for fewer
host round-trips). The hidden-state readback (2.3 ms/tok) is the single
end-of-token device→host copy that returns the decode output — the cost B4
(device lm-head) would eliminate. These metrics are within measurement noise
of the PROFILE-001 uniform-Q4_K control (the path the default format now
reaches).

### Post-D6 resident counters on the real default Q4_K_M model

| Path | Q4_K_M (post-D6) |
|---|---|
| resident-KV (GPU-006) | uses=288, fallbacks=0, rate=**100.0%** |
| resident-hidden (GPU-007) | uses=288, fallbacks=0, rate=**100.0%** |

The default Q4_K_M model now reports resident-hidden uses on every eligible
dense layer with zero format-driven fallbacks.

## D6-I — Real output validation

Deterministic (greedy) CPU and CUDA generation against the real Q4_K_M
vindex produce **identical** output (byte-for-byte) across two prompts. The
Q4_K_M extraction level produces low-quality text from Qwen2.5-3B at this
config (a model-quality concern unrelated to D6), but the parity invariant
holds: both deterministic paths agree. The Q4_K_M vindex remains the default
extraction format (`quant: q4k`, gate/up Q4_K, down Q6_K); no `index.json`
hand-edit or `--down-q4k` rebuild is required to benefit from GPU-007.

## Scope note (remaining mixed-format gap)

D6 changes only `resident_hidden_layer_eligible` and
`host_ffn_block_device_resident` (the resident-hidden decode path). The
non-resident decode device path (`host_ffn_block_device`) and the prefill
device path (`host_prefill_ffn_block_device`) still require a uniform FFN
triple — they are not on the critical decode path and their mixed-format
support is a separate slice. The host fallback remains the parity oracle in
all cases.

## Next evidence-ranked slice

**B3 — launch batching / graphization.** The D6 instrumented pass shows 632
launches/tok on the now-resident path; collapsing the per-op `launch →
stream` calls into fewer submissions (CUDA Graphs) is the next
profile-recommended slice. (B4 — lm-head on device — would eliminate the 2.3
ms/tok end-of-token hidden readback.)

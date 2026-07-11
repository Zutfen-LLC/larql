# LARQL-GPU-B4-CORRECTION — Result accounting, tie parity, norm-weight residency, benchmark reproducibility

**Date:** 2026-07-11  **Slice:** `LARQL-GPU-B4-CORRECTION`
**Base:** `main` @ `7e5928b635fdd1b9068e949fb7ebe002b0fea675` (PR #52 / GPU-B4)
**Correction commit:** `5fa7c821757421d71d99851bf50da25402fc539e`
**Host:** NVIDIA RTX 3060 12 GB (sm_86), driver 610.43.03, NVRTC 12.4.127
**Vindex:** production-default Qwen2.5-3B Q4_K_M  **Decision:** **B4 stays OPT-IN** (no change)

This report corrects the four B4 evidence-integrity / parity defects without
redesigning the B4 architecture or expanding it into a new optimization
project. The prior B4 report (`cuda-b4-device-greedy-2026-07-11.md`) is
**superseded for the affected metrics** (result-readback accounting, the
overbroad lm-head counter, final-norm-weight HtoD, and the performance
methodology). It remains valid for the architecture description and the
correctness gate.

---

## 1. Defects corrected

| # | Defect | Correction |
|---|---|---|
| A | Result-readback accounting understated DtoH ops (1) and bytes (40 B) while the code performed 2 `clone_dtoh` calls on `GREEDY_MAX_K`-wide buffers (64 B for k=5). | Result buffers resized to `k`; the two `clone_dtoh` calls (scores + ids) are recorded as **2 DtoH ops × k·4 B = 40 B** for k=5. |
| B | Exact-score tie behavior was not guaranteed: the final reduction compared only `partial_scores` and broke ties on partial-buffer position / strided-scan order, which can pick the wrong (higher) token id when the strided scan wraps (>256 partials). | Both kernels carry the **global token id** through every comparison and use one explicit lexicographic comparator `(score desc, then id asc)`, shared with the host reference. |
| C | The final RMSNorm weight was uploaded every eligible decode token via `clone_htod`. | The weight is resolved **once** through the f32 weight cache and reused (cold upload 1×/generation, steady-state cache hits); per-token HtoD = 0. |
| §9 | The `lm_head_full_score_dtoh` counter labelled every Q4_K matvec result as lm-head traffic, but `q4k_matvec` is a general op also reached by Q4_K MoE/FFN matvecs. | Counter renamed `q4k_matvec_dtoh`; describes all Q4_K matvec readbacks. lm-head equivalence is a property of the dense route, not the counter. |
| D | The committed `scripts/bench_b4.sh` set `LARQL_GPU_PROFILE=1` for the performance runs, so it did not reproduce the stated uninstrumented methodology. | Script split into an **uninstrumented performance phase** (source of truth) and a separate **instrumented structural phase**; `set -euo pipefail`; raw per-rep JSONL preserved; medians/MAD computed from raw records. |

---

## 2. Result-readback accounting (Correction A)

**Production `candidate_width = 5`.** The greedy-head workspace result buffers
are now sized to `k` (`greedy_workspace.rs`), and the production path records
the two `clone_dtoh` calls honestly:

| metric | pre-correction (claimed) | pre-correction (actual) | corrected |
|---|---|---|---|
| DtoH operations / engaged token | 1 | **2** | **2** |
| DtoH payload / engaged token | 40 B | 64 B (`GREEDY_MAX_K`-wide buffers) | **40 B** (`k` f32 + `k` u32) |

Structural counter (B4A, instrumented): `result_dtoh 2.1 ops/tok, 42.6 B/tok`
(2 ops × k·4 B = 40 B per engaged token; the small overshoot is warmup-step
inflation — counters accumulate over warmup+measured while the rate divides by
measured steps only). Validated by `b4c_result_buffers_match_candidate_width`
(buffers == k for k∈{1,3,5,8}, clamp at `GREEDY_MAX_K`) and
`b4c_result_dtoh_accounting_for_known_k` (2 copies, 8k bytes).

---

## 3. Tie-breaking contract (Correction B)

One explicit comparator shared by the host reference and both device kernels:

```
Candidate A ranks ahead of B when:
  A.score > B.score;  OR
  A.score == B.score AND A.token_id < B.token_id.
```

Non-finite scores remain excluded. The partial kernel carries the global token
id (`base + lane`) through its tree reduction; the final kernel carries the
global token id (read from `partial_ids`) through both the per-thread strided
scan and the shared reduction, so the winner of an exact tie is the lower token
id regardless of block, thread, scan position, partial-buffer position, or
number of partials.

### Adversarial tie tests (`b4c_tie_*`, all PASS on RTX 3060)

The decisive fixture is **all-equal scores over 15 360 rows** (60 blocks → 300
partials > 256, so the final strided scan wraps). The correct top-5 is exactly
`[0,1,2,3,4]`; the pre-correction score-only final comparator returns the wrong
set (it lets a high-id candidate at partial position 256 — visited by thread 0
in its second iteration — outrank a low-id candidate at position 1 visited by
thread 1). The corrected kernels return `[0,1,2,3,4]`.

| case | what it pins |
|---|---|
| within one 256-row block | lower id within a block |
| across two blocks | lower id across blocks |
| first & last logical row | boundary tie |
| across final-reduction threads + strided scan (>256 partials) | the decisive adversarial case |
| second-through-fifth ordering | full returned set, not just the winner |
| equal negative scores | negative-score ties |
| adjacent to NaN / ±∞ | finite tie beside non-finite |
| padded rows under ties | padding excluded; logical tie order preserved |
| 8× repeat | byte-identical candidate order (determinism) |

Plus a pure host-comparator unit test (`b4c_greedy_comparator_ordering`).

---

## 4. Final-RMSNorm weight residency (Correction C)

The immutable final-norm weight is resolved once through the existing f32
weight cache (`resolve_f32_weight`) and held on the `GreedyHeadWorkspace`; the
production path passes the device handle to `launch_rms_norm_into` (reused from
the B3A graph path) instead of re-uploading every token.

| metric | pre-correction | corrected |
|---|---|---|
| final-norm-weight HtoD / token (steady state) | **1** (`clone_htod` every eligible token) | **0** |
| cold HtoD / generation | 1 (unreported) | 1 (now counted in a dedicated counter) |

Structural counter (B4A, instrumented): `final-norm weight htod 0.0/tok (cold),
105.0 B/tok, cache_hits 1.1/tok` — i.e. one cold upload of 2048×4 = 8192 B
amortized over ~78 measured steps (≈105 B/tok) plus steady-state cache hits,
**zero per-token HtoD in steady state**. Validated by
`b4c_final_norm_weight_uploaded_once_and_reused` (1 miss + 2 hits across 3
resolutions) and `b4c_parameter_free_norm_remains_supported` (parameter-free
placeholder cached). The workspace re-resolves when the source pointer changes
(head-spec swap without reset); the cache flushes at `reset_kv_cache`.

---

## 5. Q4_K matvec counter scope (§9)

`QuantMatVec::q4k_matvec` is a general op reached by the host lm-head path AND
by Q4_K MoE/FFN gate/up/down matvecs (see `pipeline.rs`
`moe_expert_contribution_q4k`). The counter is renamed `q4k_matvec_dtoh` so it
accurately describes all Q4_K matvec result readbacks. For the canonical dense
Qwen2.5-3B benchmark the only decode-time caller IS the host lm-head path (no
MoE, no other Q4_K matvec in the measured route), so the value is
lm-head-equivalent there — a property of the route, proven by call-site audit,
not of the counter. Validated by `b4c_q4k_matvec_counter_is_not_lm_head_exclusive`
(the counter increments for a non-lm-head Q4_K matvec shape).

---

## 6. Performance re-validation (Correction D) — UNINSTRUMENTED, source of truth

5 reps × (5 warmup + 79 requested) decode steps, same prompt/model/GPU, plain
greedy, release binary, one long-lived process per rep. **All reps early-stopped
at step 78/79 (EOS), consistently across modes.** Medians + MAD computed from
the raw per-rep records by `scripts/b4_aggregate.py` (no hand-entered values).

Raw per-rep p50 (ms):

| mode | rep1 | rep2 | rep3 | rep4 | rep5 |
|---|---|---|---|---|---|
| BaselineA (g0 b0) | 108.56 | 109.09 | 109.11 | 108.24 | 108.24 |
| B4A (g0 b1) | 108.53 | 109.12 | 109.13 | 108.27 | 108.25 |
| BaselineB (g1 b0) | 107.88 | 108.50 | 107.65 | 107.69 | 108.20 |
| B4B (g1 b1) | 108.28 | 108.55 | 107.68 | 107.73 | 108.22 |

Aggregated:

| mode | graphs | B4 | p50 median | p50 MAD | mean median | mean MAD | min | max | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| Baseline A | 0 | 0 | 108.56 | 0.32 | 108.57 | 0.34 | 108.24 | 109.11 | 9.2 |
| B4 A | 0 | 1 | 108.53 | 0.28 | 108.58 | 0.31 | 108.25 | 109.13 | 9.2 |
| Baseline B | 1 | 0 | 107.88 | 0.23 | 107.86 | 0.19 | 107.65 | 108.50 | 9.3 |
| B4 B | 1 | 1 | 108.22 | 0.33 | 108.21 | 0.39 | 107.68 | 108.55 | 9.2 |

| comparison | Δ p50 (ms) | Δ p50 (%) | within noise? |
|---|---|---|---|
| graph-off (B4A − BaselineA) | −0.03 | −0.03 % | yes (MAD 0.28–0.32) |
| graph-on (B4B − BaselineB) | +0.34 | +0.32 % | yes (MAD 0.23–0.33) |

**Gate: NOT MET.** B4 is **neutral within run-to-run noise** in both graph modes
(well under the 1% gate). The graph-on sign flipped relative to the prior report
(+0.32 % now vs −0.42 % prior), which confirms the prior per-rep values were
inside noise and not reproducible — exactly the integrity gap Correction D was
meant to expose. The corrected run-to-run MAD (0.19–0.39 ms) is also materially
larger than the prior report's 0.01–0.07 ms; the prior MAD was computed from
instrumented runs.

**Why neutral (unchanged conclusion):** the Q4_K lm-head GEMV was already
device-resident. B4 moves the host lm-head work into the device stream — the
structural run shows the `lm_head` stage (12.72 ms/tok baseline) drop to 0.000
while `GPU fwd` rises by the same amount, leaving wall-clock unchanged.

---

## 7. Structural phase (INSTRUMENTED, counters only — not wall-clock truth)

Per engaged decode token, B4A vs BaselineA (graph-off); B4B/BaselineB identical
in shape (graph submissions added):

| counter | BaselineA (b4=0) | B4A (b4=1) |
|---|---|---|
| host `final_norm` ms/tok | 0.004 | **0.000** |
| host `lm_head` ms/tok | 12.721 | **0.000** |
| final-hidden readback ms/tok | 2.267 | **0.000** |
| final-hidden readback B/tok | 8717.1 | **0.0** |
| dtoh MiB/tok | 2.121 | 1.496 (−0.625) |
| device-greedy attempts/engaged/fallback/failure per tok | 0/0/0/0 | **1.1/1.1/0/0** (100 % engaged) |
| `q4k_matvec` dtoh (renamed) | 1.1/tok, 662 285 B/tok | 0.0/tok, 15 583 B/tok |
| result dtoh (Correction A) | 0 | **2.1 ops/tok, 42.6 B/tok** (2×k·4 B) |
| final-norm weight HtoD (Correction C) | n/a | 0.0/tok cold (1/gen), 105 B/tok, **cache_hits 1.1/tok** |

The residual ~15 KB/tok `q4k_matvec` dtoh under B4 is the host-path Q4_K matvec
from the first/prefill step (B4 is decode-only); decode-step lm-head traffic is
0 on the B4 route.

> **Prior-report discrepancy noted honestly:** the prior report's baseline
> `dtoh 13.19 MiB/tok` is not reproduced here (corrected: 2.121 MiB/tok). The
> prior value likely came from a different measurement configuration; the
> corrected number is from the same `larql bench --backends cuda` command used
> for every mode, so the B4-vs-baseline comparison is internally consistent.
> This does not affect the gate (wall-clock is neutral either way).

---

## 8. Existing B4 gates — all preserved

| gate | result |
|---|---|
| token-sequence parity (baseline vs B4-on) | preserved — B4 selects the argmax token; the corrected tie-break only changes behaviour on exact-score ties, which now match the host |
| logical-vocabulary protection | intact (winner always < logical_vocab; `b4c_tie_padding_rows_remain_excluded_under_ties`) |
| eligible device-greedy engagement | **100 %**, 0 fallbacks, 0 failures |
| resident-KV / resident-hidden engagement | no regression |
| graph-off + graph-on | both pass full CUDA suite (230/230 each) |
| default-on / opt-in | **unchanged: OPT-IN** (`LARQL_CUDA_DEVICE_GREEDY=1`); `LARQL_CUDA_GRAPHS` default unchanged |

---

## 9. Test totals (corrected branch `5fa7c821`)

| suite | result |
|---|---|
| `cargo test -p larql-compute --lib` | **750 pass** |
| `cargo test -p larql-vindex --lib` | **1131 pass** |
| `cargo test -p larql-inference --lib` | **1262 pass** |
| `LARQL_CUDA_GRAPHS=0 cargo test -p larql-compute-cuda --lib -- --test-threads=1` | **230 pass** (+15 `b4c_*`) |
| `LARQL_CUDA_GRAPHS=1 cargo test -p larql-compute-cuda --lib -- --test-threads=1` | **230 pass** |
| `scripts/b4_aggregate.py --selftest` | OK (median/MAD/failed-missing/instrumented-label) |
| fmt + clippy (compute, vindex, inference, cuda, cli `--features cuda`) | clean |
| `cargo build -p larql-cli --release --features cuda` | OK |

---

## 10. Documentation / artifacts committed

* `bench/baselines/cuda-b4-correction-2026-07-11.md` (this file)
* `bench/baselines/cuda-b4-correction-2026-07-11.json`
* `bench/baselines/b4-correction-raw/` — raw per-rep JSONL, aggregated JSON,
  per-mode structural stdout, provenance header
* `scripts/bench_b4.sh` — corrected two-phase script
* `scripts/b4_aggregate.py` — pure aggregation + self-test
* prior report `cuda-b4-device-greedy-2026-07-11.md` retained, superseded for
  affected metrics

## 11. Scope deviations

None beyond the explicit `rep` label in the bench JSONL being constant (cosmetic
— aggregation groups by mode, so all 5 reps per mode are aggregated correctly;
the per-rep values are distinguishable in the raw stdout logs). The benchmark
early-stopped at step 78/79 (EOS) for every rep and mode; this is consistent
across the comparison and recorded in each record's `early_stop`/`n_steps`.

## 12. Recommended next slice

Unchanged from the prior report and NOT part of this correction: the
resident-hidden path for the **mixed Q4_K_M attention triple** (gate/up/down are
Q4_K_M but the attention V/QK-norm path still reads back), which the D6 report
ranked above B4. A fused Q4_K GEMV + top-K reduction kernel remains
**unjustified** — the corrected measurement confirms B4 is wall-clock-neutral,
so fusion would not clear the 1 % gate either.

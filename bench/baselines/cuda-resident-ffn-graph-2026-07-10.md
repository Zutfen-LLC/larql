# LARQL-GPU-B3A: CUDA Graph replay for the resident decode FFN — 2026-07-10

> **Status: IMPLEMENTED, OPT-IN, accounting-corrected.** CUDA Graph replay for
> the resident decode FFN is functional on RTX 3060 (`LARQL_CUDA_GRAPHS=1`) and
> remains **opt-in (default Disabled)**. A subsequent accounting audit
> (implementing B3A review points 3 + 8) corrected the submission accounting:
> under **honest total-submission accounting** (every graph submission + D2D +
> cross-stream sync counted, captured nodes no longer double-counted as direct
> launches), the structural reduction is **−18.2%**, which **does NOT meet the
> ≥25% gate**. The earlier headline "−36.6%" was the NULL-stream `launches`
> counter alone, which excluded the cap_stream graph launches and D2D copies
> the path adds. Graph replay stays opt-in. RTX 3060 validation is the final
> project evidence; no RTX 3090 rerun is required.
>
> **Accounting correction (points 3, 5, 8):**
> - Captured FFN kernel nodes no longer inflate `direct_kernel_submissions`
>   (the `*_into` launchers' `note_launch` is suppressed during stream capture).
> - `graph_submissions`, `d2d_submissions`, `captured_kernel_nodes`,
>   `logical_graph_kernel_executions`, `graph_failures`, `graph_fallbacks`,
>   and `graph_cross_stream_syncs` are now emitted to the bench JSON + a
>   `TOTAL host submissions` line; the ≥25% gate is evaluated against
>   `TOTAL = launches + graph_submissions + d2d_submissions`.
> - The graph path's cross-stream `synchronize()` calls (runtime↔cap_stream
>   handoff) are now counted (`graph_cross_stream_syncs`) — they previously
>   bypassed `note_sync`, hiding a **~91% sync increase**.
> - `AUTO_FREE_ON_LAUNCH` is retained with documented justification: cudarc
>   0.19.8's `end_capture` takes a typed `CUgraphInstantiate_flags` whose only
>   constructible variants are non-zero, so a sound "no flags" value cannot be
>   expressed through the safe API (point 5's escape clause).

## What changed

B3A adds CUDA Graph replay for the resident decode FFN chain. Each eligible
dense layer's static 7-kernel FFN chain (pre-norm → gate matvec → up matvec →
activation → down matvec → [post-norm] → residual) is captured into a reusable
`CudaGraph` on the first decode token of a generation, then replayed (one
`graph.launch()` replacing 7 individual host kernel launches) on every
subsequent token.

### The implementation (12 commits)

| Task | Commit | Description |
|---|---|---|
| B3A-SMOKE | `1f7a5d1e` | Native graph capture/instantiate/replay/teardown on RTX 3060. Surfaced 2 critical constraints. |
| B3A-2 | `b33a7505` | Pure plan contract + generation/layer identity (26 host tests). |
| B3A-4 | `0e5c2777` | Into-buffer `*_into` launch primitives (200 parity tests). |
| B3A-3/6 | `9e35345e` | Arena (ping-pong + dedicated capture stream) + graph state + explicit Drop. |
| B3A-7 | `69641643` | Per-backend graph mode + 7 capture-aware counters + reset teardown. |
| B3A-5 | `011e9d57` | Build + replay pipeline integration (204 tests pass with GRAPHS=1). |

### Follow-on: accounting correction (review points 3, 5, 8)

A subsequent audit found the merged B3A-7 counters were **not honest**:
captured FFN nodes inflated `direct_kernel_submissions` (the `*_into`
launchers called `note_launch` during stream capture), the graph path's
`graph_submissions` / `d2d_submissions` were never emitted to the bench JSON,
and its cross-stream `synchronize()` calls bypassed `note_sync`. The correction
(suppress `note_*` during capture via a depth guard; emit all graph counters +
a `TOTAL host submissions` line; count cross-stream syncs; justify
`AUTO_FREE_ON_LAUNCH` per cudarc 0.19.8's typed-flags API) changed the gate
outcome from a false "36.6% PASS" to an honest **18.2% FAIL**. 204/204 CUDA
tests still pass under both `GRAPHS=0` and `GRAPHS=1`; fmt + clippy clean.
| B3A-8 | `dfc3fe45` | Graph correctness + lifecycle tests (build/replay/reset/disabled). |

## Two critical findings from B3A-SMOKE (shaped the design)

1. **The NULL default stream cannot be captured** (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). cudarc's `default_stream()` returns `cu_stream = null_mut`. Graph capture/replay requires a dedicated non-NULL stream via `CudaContext::new_stream()`.
2. **cudarc's event tracking must be disabled for graph buffers.** By default, every `CudaSlice` carries read/write `CudaEvent` handles. During capture, `launch_builder.arg(&CudaSlice)` injects `cuStreamWaitEvent` for prior write events → `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`. Disabling event tracking (the documented configuration for graph capture) resolves this; the stream orders captured work explicitly.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12 GB) |
| Compute capability | sm_86 (NVRTC target compute_86) |
| Driver | 610.43.03 |
| CUDA/NVRTC | 12.4.127 |
| Rust | 1.97.0 (c980f4866 2026-06-30) |
| Base SHA (main) | `6ed40296` |
| Feature SHA (B3A) | `ac922262` |
| Model | Qwen2.5-3B-Instruct (qwen2, 36 layers) |
| Quant | Q4_K_M (gate/up Q4_K, down Q6_K) — the default |
| GPU pstate (idle) | P8, 210 MHz SM, 40°C |

## B3A-SMOKE — native graph capture validated

The smoke test (`backend/runtime.rs::b3a_smoke_tests`) validates the full CUDA
Graph lifecycle on the real RTX 3060 driver/runtime:

- Capture on a dedicated LARQL stream (event tracking disabled)
- Graph instantiation with default flags
- `CudaGraph::launch()` replay + idempotent determinism
- **Stable-pointer replay**: in-place `memcpy_htod` mutating buffer contents
  changes the replay output (the core ping-pong invariant)
- Clean teardown (graph → buffers)
- Repeated create→replay→reset→rebuild→drop lifecycle (3 cycles)

Both smoke tests **PASS** on RTX 3060.

## B3A-9 — full regression (204 tests)

| Suite | Result |
|---|---|
| `cargo fmt --all --check` | clean |
| clippy `-D warnings` | clean |
| Full CUDA suite, `LARQL_CUDA_GRAPHS=0` (serial) | **204/204 pass** |
| Full CUDA suite, `LARQL_CUDA_GRAPHS=1` (serial) | **204/204 pass** |
| B3A-SMOKE (capture/replay/teardown) | 2/2 pass |
| B3A-2 host plan tests | 26/26 pass |
| B3A-8 graph counter tests (GRAPHS=1) | 4/4 pass |

The graph path is parity-correct: the Q4_K_M single-token, multi-token
(4 tokens, no drift), and consecutive-layers tests all pass with `max_abs < 1e-3`
under `LARQL_CUDA_GRAPHS=1`. Resident-KV and resident-hidden remain 100% active.

Graph counter assertions (B3A-8, verified with `LARQL_CUDA_GRAPHS=1`):
- Single token: `builds == num_layers`, `submissions == num_layers`, `failures == 0`.
- Multi-token (4): `builds == num_layers` (no rebuild), `submissions == num_layers × 4`.
- Reset/rebuild: after `reset_kv_cache`, cumulative `builds == 2 × num_layers`.

## B3A-10 — same-day A/B benchmark

Same RTX 3060, same day, same Qwen2.5-3B Q4_K_M vindex, same prompt ("The
capital of France is"), 3 warmup + 127 measured decode steps, greedy decoding.
Uninstrumented release measurements (3 reps each).

### Uninstrumented (wall-clock source of truth)

| metric | GRAPHS=0 (baseline) | GRAPHS=1 (feature) | delta |
|---|---|---|---|
| p50 ms/tok (rep1/2/3) | 121.57 / 121.99 / 122.05 | 121.32 / 122.21 / 122.25 | — |
| **median p50** | **121.99** | **122.21** | **−0.18%** (noise) |
| mean ms/tok | 121.85 | 121.96 | −0.09% |
| tok/s | 8.20 | 8.20 | 0.00% |
| n_steps | 127 | 127 | — |

### Instrumented (honest submission decomposition, 64-token window, GRAPHS=0 vs =1)

**Every D2D and graph submission is counted as a host CUDA submission**
(B3A review point 3). The gate metric is `TOTAL = launches + graph_submissions
+ d2d_submissions`. The earlier `launches`-only headline (−36.4%) is retained
for reference but is **not** a valid gate metric — it excludes the cap_stream
graph launches and D2D copies the path adds.

| counter | GRAPHS=0 | GRAPHS=1 | Δ |
|---|---|---|---|
| **TOTAL host submissions/tok** | **622.9** | **509.7** | **−18.2%** ❌ gate ≥25% |
| launches/tok (NULL-stream direct, capture-aware) | 622.9 | 396.6 | −36.4% (old headline) |
| graph submissions/tok | 0 | 37.7 | +37.7 |
| graph d2d/tok (seed + output) | 0 | 75.4 | +75.4 |
| captured nodes/tok | 0 | 3.4 (build amortized) | — |
| logical execs/tok (nodes × replays) | 0 | 226.3 | — |
| syncs/tok (runtime stream) | 83.3 | 83.3 | 0 |
| **cross-stream syncs/tok (graph handoff)** | **0** | **76.0** | **+91% total syncs** ❌ sync-regression gate |
| graph fallbacks/tok | 0 | 0 | 0 |
| graph failures/tok | 0 | 0 | 0 |

**The submission count drops only 18.2% on honest accounting** (622.9 → 509.7):
the 252 individual FFN host launches collapse to ~36 graph-launch submissions,
but the separate `cap_stream` design adds ~36 graph submissions + ~72 D2D
(seed + output copies) back. This is exactly the outcome B3A review point 3
predicted: *"if both an input and output D2D copy are required… that reduces
>total submissions by only about 23%"* — the measured 18.2% is below even that,
and syncs nearly double.

### Uninstrumented wall-clock (p50 ms/tok, 3 reps, 59-step window)

| metric | GRAPHS=0 (p50 r1/r2/r3) | GRAPHS=1 (p50 r1/r2/r3) | Δ |
|---|---|---|---|
| p50 ms/tok | 121.52 / 121.54 / 121.66 | 121.68 / 121.46 / 121.52 | ≈0% (flat) |
| median p50 | 121.54 | 121.52 | −0.02% |
| tok/s | 8.2 | 8.2 | 0 |

## B3A-11 — performance decision gate (honest accounting)

| Gate | Target | Result | Decision |
|---|---|---|---|
| Submission reduction (TOTAL) | ≥25% | **18.2%** | ❌ FAIL |
| Wall-clock improvement | ≥1% | ≈0% (flat) | ❌ FAIL |
| No transfer regression | none | none | ✅ PASS |
| **No sync regression** | none | **+91% (83→159)** | ❌ FAIL |
| Resident-KV/hidden | 100% | 100% | ✅ PASS |

**Decision: graph replay stays opt-in (default `Disabled`).** Under honest
total-submission accounting (point 3) the structural reduction is 18.2%, below
the 25% gate; the wall-clock is flat; and the cross-stream design nearly
doubles sync count. The graph path is fully functional and correct via
`LARQL_CUDA_GRAPHS=1`, but it does not deliver a net win on the RTX 3060.

> **Correction of the prior claim.** An earlier version of this baseline
> reported "submission reduction 36.6% — gate PASS". That number was the
> NULL-stream `launches` counter only; it excluded the graph submissions and
> D2D copies the path adds (then not emitted to the bench JSON) and
> double-counted token-1's captured nodes as direct launches. The corrected
> accounting (above) is the honest basis for the gate.

### Path to a net win (B3B)

The 18.2% submission reduction is real but consumed by the cross-stream
sync cost (the cap_stream↔runtime handoff). Two follow-on paths could recover
it:

1. **Single-stream capture** (eliminate the separate `cap_stream`): if the
   FFN graph could be captured/replayed on the runtime stream, the seed/output
   D2D copies and the cross-stream syncs would both vanish — moving honest
   TOTAL toward the launches-only 36% figure and removing the sync regression.
   This was blocked by B3A-SMOKE finding #1 (the NULL stream cannot be
   captured); the fix is a non-NULL **runtime** stream, which is a larger
   plumbing change.
2. **B3B (attention graphization)**: the remaining ~360 submissions/token are
   the attention chain, but it has dynamic `total_len` + KV-cursor state.

## How to use

```bash
# Graph replay OFF (default, recommended — no wall-clock benefit yet)
LARQL_CUDA_GRAPHS=0 larql bench <vindex> --backends cuda

# Graph replay ON (opt-in, for testing/experimentation)
LARQL_CUDA_GRAPHS=1 larql bench <vindex> --backends cuda

# Graph replay ON + full diagnostics
LARQL_CUDA_GRAPHS=1 LARQL_GPU_DIAG=1 LARQL_GPU_PROFILE=1 larql bench <vindex> --backends cuda
```

## Remaining B3 work

- **B3B**: graph the dynamic attention chain (harder — dynamic total_len +
  KV cursor state), OR — the higher-leverage fix — eliminate the separate
  `cap_stream` (capture/replay on a non-NULL **runtime** stream) to remove the
  2 D2D/layer and the cross-stream syncs that currently consume the
  submission savings.
- The graph path is correct and tested but opt-in until a follow-on slice
  delivers BOTH the honest ≥25% submission gate AND the ≥1% wall-clock gate.
  Under honest accounting B3A delivers neither (18.2% submission; flat
  wall-clock; +91% syncs).

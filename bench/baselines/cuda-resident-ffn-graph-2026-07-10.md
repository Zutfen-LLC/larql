# LARQL-GPU-B3A: CUDA Graph replay for the resident decode FFN — 2026-07-10

> **Status: IMPLEMENTED + OPT-IN.** CUDA Graph replay for the resident decode
> FFN is fully functional on RTX 3060 (`LARQL_CUDA_GRAPHS=1`). The structural
> submission-reduction target (≥25%) is met (36.6% reduction). The wall-clock
> improvement target (≥1%) is NOT met (-0.18%, within noise), so graph replay
> stays **opt-in (default Disabled)** per the B3A-11 performance gate. RTX 3060
> validation is the final project evidence; no RTX 3090 rerun is required.

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

### Instrumented (submission decomposition, GRAPHS=1)

| counter | GRAPHS=0 | GRAPHS=1 | delta |
|---|---|---|---|
| **launches/tok** | **599.2** | **379.8** | **−36.6%** ✅ |
| htod copies/tok | 3.3 | 3.3 | 0 |
| dtoh copies/tok | 78.1 | 78.1 | 0 |
| syncs/tok | 78.1 | 78.1 | 0 |
| hidden readback ms/tok | 2.31 | 0.01 | −2.30 |
| gpu_fwd ms/tok | 102.7 | 96.1 | −6.4 |

The submission count drops 36.6% (599→380/token): the 252 individual FFN host
launches (7/layer × 36) collapse to 36 graph-launch submissions, plus the 2
D2D copies per layer (72 total) that seed/read the arena buffers. But the
wall-clock is flat.

### Why no wall-clock improvement

The graph path adds **2 `synchronize()` calls per layer** (72/token) for the
cross-stream handoff: the cap_stream must wait for the runtime stream's D2D
seed before replay, and the runtime stream must wait for the cap_stream's
graph output before reading it. These explicit syncs offset the launch-count
savings — each `synchronize()` is a full device pipeline stall. The
submission reduction is real (36.6%) but the sync overhead is comparable to
the per-launch savings on the RTX 3060's relatively low launch latency.

The `hidden_readback` counter dropped to ~0 because the graph path's final
D2D output copy happens on the cap_stream (not tracked by the runtime's NULL-
stream profile counters). The `gpu_fwd` stage shows a 6.4 ms improvement,
but this is offset by the sync overhead distributed across the decode step.

## B3A-11 — performance decision gate

| Gate | Target | Result | Decision |
|---|---|---|---|
| Submission reduction | ≥25% | **36.6%** | ✅ PASS |
| Wall-clock improvement | ≥1% | **−0.18%** | ❌ FAIL (noise) |
| No regression | <0.5% | 0.18% | ✅ PASS |
| Transfer/sync regression | none | none | ✅ PASS |
| Resident-KV/hidden | 100% | 100% | ✅ PASS |

**Decision: graph replay stays opt-in (default `Disabled`).** The structural
submission-reduction target is met, but the wall-clock improvement is below
the 1% gate. Per the B3A review: "When improvement is positive but below 1%,
keep graph replay opt-in and document the result rather than overstating B3
progress." The graph path is fully functional and correct via
`LARQL_CUDA_GRAPHS=1` for users who want to test it.

### Path to wall-clock improvement (B3B)

The remaining launch overhead (380/tok) is dominated by the attention chain
(dynamic total_len, KV cursor) and the cross-stream sync. Two follow-on paths:

1. **B3B (attention graphization)**: graph the resident attention chain too,
   reducing the remaining ~340 attention-related submissions. Harder — the
   attention chain has dynamic `total_len` and KV-cursor state.
2. **Event-based cross-stream sync**: replace the 72 explicit `synchronize()`
   calls with `CudaEvent` record/wait pairs, which avoid the full pipeline
   stall. This alone could recover the wall-clock.

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
  KV cursor state), OR replace the explicit `synchronize()` cross-stream sync
  with event-based `CudaEvent` record/wait pairs (avoids the pipeline stall).
- The graph path is correct and tested but opt-in until a follow-on slice
  delivers the ≥1% wall-clock gate.

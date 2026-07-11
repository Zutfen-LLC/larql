# LARQL-GPU-B3B: Single non-NULL decode stream for CUDA-Graph FFN replay — 2026-07-10

> **Status: IMPLEMENTED, OPT-IN.** B3B replaces B3A's separate `cap_stream` with
> **one dedicated non-NULL runtime stream** that carries the entire resident
> decode critical path — attention, KV append, residual, the captured FFN graph,
> and the layer-to-layer hidden handoff. Graph capture AND replay now run on the
> same stream as every other decode kernel, so layer ordering is by stream
> submission alone.
>
> **Measured result (RTX 3060, Qwen2.5-3B Q4_K_M, 79 measured decode steps,
> 5 reps × 2 modes, same prompt/warmup):**
>
> | metric | GRAPHS=0 (baseline) | GRAPHS=1 (B3B) | delta |
> |---|---|---|---|
> | **TOTAL host submissions/tok** | **627.9** | **436.6** | **−30.5% (✅ ≥25%)** |
> | launches/tok | 627.9 | 398.3 | −36.6% |
> | graph submissions/tok | 0 | 38.3 | +38.3 |
> | graph d2d/tok | 0 | **0** | — (B3A two-stream was +75) |
> | cross-stream syncs/tok | 0 | **0** | — (B3A two-stream was +76) |
> | blocking syncs/tok | 83.1 | 83.1 | 0 (no regression) |
> | median p50 ms/tok | 121.50 | 120.87 | −0.52% (within noise) |
> | mean ms/tok | 121.72 | 120.97 | −0.62% |
> | tok/s | 8.2 | 8.25 | — |
> | time to first decode (prefill) | 409.5ms | 399.7ms | −2.4% |
> | resident-KV / resident-hidden | 100% | 100% | — |
>
> **Gate outcome (B3B section 8):** the structural gate **passes**
> (−30.5% TOTAL host submissions ≥ 25%; **zero** per-layer D2D; **zero**
> cross-stream syncs; syncs unchanged at 83.1 — no regression; resident-KV and
> resident-hidden 100%). The **wall-clock gate does NOT pass**: the median
> decode improvement is **−0.52%**, which is **within run-to-run noise**
> (GRAPHS=0 spread 0.81ms vs the 0.63ms improvement). Graph mode therefore
> **remains opt-in (default Disabled)**. The host-submission reduction does not
> translate to measurable wall-clock on the RTX 3060 because the eliminated
> ~191 launches/tok are a small fraction of the 122ms/token, which is dominated
> by the device lm-head (25.8ms, 21%) + GPU forward (96ms, 79%). RTX 3060 is the
> canonical validation host; no RTX 3090 rerun required.

## What changed (B3B single-stream architecture)

| Area | B3A (two-stream) | B3B (single non-NULL stream) |
|---|---|---|
| runtime stream | NULL/default (`default_stream()` = `null_mut`) | one dedicated non-NULL (`context.new_stream()`) |
| graph capture/replay stream | separate `cap_stream` (`ctx.new_stream()`) | the runtime stream itself |
| per-layer handoff | runtime↔cap_stream: 2 D2D + 4 cross-stream syncs | **none** — stream submission order |
| attention → FFN input | D2D seed copy into the arena input | in-place residual add into the arena input slot |
| FFN output → next layer | D2D clone out of the arena output | the arena output IS the next layer's input (flip) |
| event tracking | disabled when the arena is created | disabled once at runtime init |
| capture mode | `GLOBAL` | `RELAXED` (production-equivalent — no syncs occur during the capture window) |

### The five core edits

1. **`CudaRuntime::initialize_impl`** (`backend/runtime.rs`): `context.default_stream()` →
   `context.new_stream()` + a one-time context-wide `disable_event_tracking()`
   (sound: the runtime stream is the only stream, so stream order is the only
   ordering). Every consumer takes the stream as a `&Arc<CudaStream>` parameter
   (weight cache, KV cache, every launch), so all CUDA work transparently moves
   onto this stream — weight-cache uploads, KV allocation/append, resident-KV
   attention, resident-hidden decode, prefill, state-dump readbacks,
   truncate/reset, final hidden readback, and teardown all run on it with no
   independent stream creation.
2. **`ResidentDecodeArena`** (`ffn_graph_state.rs`): the `cap_stream` field is
   removed. The arena keeps `hidden_a` / `hidden_b` / `generation` and allocates
   its two `[hidden]` buffers on the runtime stream.
3. **Attention → arena input** (`pipeline.rs`): a new `attention_into_arena`
   runs the shared attention chain (`resident_attention_chain_to_o_dev`, extracted
   from `host_attention_block_device_resident` so the two cannot drift) then
   writes the post-attn residual **in place** into the arena input slot via the
   new `launch_residual_add_inplace_into` primitive (binds one buffer as both
   the read base and the write output — element-wise independent, sound on a
   single stream). The FFN graph reads the post-attn residual from that exact
   stable address.
4. **Graph build/replay** (`pipeline.rs`): `build_ffn_graph_single_stream` /
   `replay_ffn_graph_single_stream` capture and replay on `runtime.stream()`
   (`CudaGraph` stores the stream it captured on, so `launch()` lands there).
   The seed `memcpy_dtod`, the output `clone_dtod`, and all four cross-stream
   `synchronize()` calls are deleted.
5. **Decode loop** (`host_decode_token_resident`): a new `host_graph_decode_layer`
   drives one graph-eligible layer end-to-end (place hidden into the arena input
   slot → attention in place → K/V mirror append → graph build/replay) and
   reports a `GraphLayerOutcome` (`ArenaOut { flip }` / `DeviceFallback` /
   `NotAttempted`). The loop carries `arena_out_flip` so consecutive graph layers
   pass the hidden by flip alone (layer 0 re-uploads the embedding each token —
   an HtoD, not a D2D; re-entry after a non-graph/scalar layer is one boundary
   D2D, never per-layer steady state).

## Arena slot flip / ownership rules (documented, B3B section 3)

The arena owns two stable `[hidden]` buffers, `hidden_a` and `hidden_b`, for the
whole generation (graph capture binds their device addresses). Per layer
`flip = li % 2 == 1`:

- **`input(flip)`** = the layer's hidden input **and** the FFN graph's input
  slot. Attention reads it for the input norm, then writes its post-attn
  residual **in place** (`buf[i] += res_mult · x[i]`, element-wise safe).
- **`output(flip)`** = the FFN graph's output slot **=** `input(¬flip)` (the
  next layer's input). The graph writes the post-FFN state there; the next
  layer's attention reads it directly.

Steady-state invariants (all on the one runtime stream, submission-ordered):
- **zero** per-layer D2D seed copies (attention writes the arena input in place);
- **zero** per-layer D2D output copies (the next layer reads the arena output by flip);
- **zero** per-layer cross-stream syncs (single stream);
- **no** intermediate host readback (the only readback is the final token output).

A graph never overwrites an input an earlier queued op still needs: layer 0
re-uploads the embedding each token, and each layer's graph write refreshes the
next slot, so every buffer is re-derived each token before it is read.

## Corrected submission accounting (B3B section 7)

Every category is measured independently (`LARQL_GPU_PROFILE=1`):

| category | GRAPHS=0 | B3B (GRAPHS=1) |
|---|---|---|
| direct kernel launches | 627.9/tok | 398.3/tok |
| graph submissions | 0 | 38.3/tok |
| D2D submissions | 0 | **0** |
| cross-stream syncs | 0 | **0** |
| blocking syncs | 83.1/tok | 83.1/tok |
| captured kernel nodes (one-time at build) | 0 | 36 layers × 6–7 nodes |
| logical graph kernel execs | 0 | 229.7/tok |

Capture-time node construction is **not** counted as executed direct kernels
(the `CaptureExitGuard` suppresses `note_launch` while `capture_depth > 0`;
nodes are counted once at build via `note_graph_captured_nodes`).

## Graph construction semantics (B3B section 6)

Verified by `graph_q4km_*` (Q4_K_M fixture, `LARQL_CUDA_GRAPHS=1`):

- Token 1: **construct** the graph, **instantiate** it, **launch** it to produce
  the token's FFN output.
- Later tokens: **reuse** the existing executable graph; one submission per
  eligible layer; no rebuild.

Counters for `num_layers` layers and `num_tokens` tokens:

```
graph_builds      == num_layers
graph_submissions == num_layers * num_tokens
warm_replays      == num_layers * (num_tokens - 1)
```

(measured 2-layer × 4-token fixture: builds=2, submissions=8, replays=6 ✓.)

## Capture mode: `RELAXED` (not `GLOBAL`)

B3A used `GLOBAL` (the strictest mode). B3B uses `RELAXED`
(`ffn_graph::graph_capture_mode`). The only behavioral difference is that
`GLOBAL` forbids `cuStreamSynchronize` on any of the context's streams while a
capture is active. B3B's single-stream decode issues **no** host sync during the
capture window (the window is exactly the 7 FFN kernel captures; every sync —
`kv_append`, K/V row readbacks, the final hidden readback — is before capture or
after `end_capture`), so the two modes are **production-equivalent**. `RELAXED`
is required for the parallel test harness (all CUDA tests share the device-0
primary context; `RELAXED` permits the concurrent `cuStreamSynchronize` calls the
parallel tests issue). See the `graph_capture_mode` doc comment for full detail.

## Test reliability: `--test-threads=1` (matches the Metal backend convention)

CUDA stream capture on the shared device-0 primary context is single-threaded:
the CUDA driver does not support two concurrent captures on one context, so the
parallel test harness races (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`). This is a
test-harness concern, not a production limitation (production decode is
single-threaded — concurrent capture never arises). The `larql-compute-cuda`
test suite therefore runs **single-threaded**, matching the existing
`larql-compute-metal` convention (`cargo test -p larql-compute-cuda -- --test-threads=1`).
A `CUDA_CAPTURE_TEST_LOCK` additionally serializes the capture-performing tests
against each other. **Result: 205/205 lib tests pass deterministically under
both `LARQL_CUDA_GRAPHS=0` and `LARQL_CUDA_GRAPHS=1`** (was 204/204 pre-B3B; +3
single-stream lifecycle/in-place tests, −2 two-stream smoke tests).

The standalone capture-lifecycle test (`b3b_single_stream_repeated_capture_
reset_rebuild_lifecycle`, the honest single-stream replacement for the old flaky
two-stream `b3a_smoke_repeated_capture_teardown_lifecycle`) exercises
create-runtime → capture → instantiate → token-1 launch → token-2+ replay →
reset → destroy → rebuild → replay → drop, for **5 repeated cycles** on the same
runtime stream.

## Scope (B3B section 9 — narrow)

No: attention graphization; CUDA event orchestration (unnecessary — single
stream); lm-head changes; MoE graph support; prefill graphization; kernel
fusion; numerical threshold/tolerance changes. ASTAB-001 numerics unchanged
(same kernels, same f32-activation reference, tolerance still 1e-3).

## How to use

```bash
# Graph replay OFF (default — no net wall-clock benefit; structural win only)
LARQL_CUDA_GRAPHS=0 larql bench <vindex> --backends cuda

# Graph replay ON (opt-in)
LARQL_CUDA_GRAPHS=1 larql bench <vindex> --backends cuda

# Graph replay ON + full submission breakdown
LARQL_CUDA_GRAPHS=1 LARQL_GPU_PROFILE=1 larql bench <vindex> --backends cuda
```

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12 GB) |
| Compute capability | sm_86 (NVRTC target compute_86) |
| Model | Qwen2.5-3B (qwen2, 36 layers) |
| Quant | Q4_K_M |
| Prompt | "The capital of France is" (5 warmup, 79 measured decode steps) |

## Why default-on is NOT flipped

The default-on gate (B3B section 8) requires the median decode improvement to
be **≥1% AND exceed run-to-run noise**. The measured −0.52% median is within
the GRAPHS=0 spread (0.81ms). The structural win (−30.5% host submissions,
zero D2D, zero cross-stream syncs) is real and honest, but it does not move
wall-clock on this host/GPU because the eliminated host launches are a small
fraction of token time. Graph mode stays opt-in until a host where launch
overhead is a larger fraction shows a measurable decode improvement.

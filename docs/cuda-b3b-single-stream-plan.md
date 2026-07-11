# LARQL-GPU-B3B: Single non-NULL decode stream (revised plan + feasibility)

> **Status: IMPLEMENTED, OPT-IN (2026-07-10).** See
> `bench/baselines/cuda-b3b-single-stream-2026-07-10.md` for the measured
> result. Structural gate **PASSES**: −0.5% TOTAL host submissions (§25%), **zero**
> per-layer D2D, **zero** cross-stream syncs, no sync regression. Wall-clock gate
> **does NOT pass** (−0.52% median, within noise) → graph mode stays opt-in.
>
> The sections below are the original approved design; the implementation
> follows them with two documented deviations: (1) capture mode is `RELAXED`,
> not `GLOBAL` — production-equivalent for B3B (no syncs occur during the
> capture window) and required for the parallel test harness; (2) the CUDA test
> suite runs `--test-threads=1` (the existing Metal-backend convention) because
> stream capture on the shared primary context is single-threaded.

## Goal

Eliminate the separate `cap_stream` and the per-layer runtime↔cap_stream
handoff that the merged B3A introduced. Route the **entire** resident decode
critical path — attention, KV append, residual/norm, the captured FFN graph,
and the layer-to-layer hidden handoff — through **one dedicated non-NULL
stream**, so that:

- the per-layer **seed + output D2D copies** (72/tok) are removed by writing
  attention's residual-add directly into the arena input slot;
- the per-layer **cross-stream `synchronize()` calls** (76/tok) are removed by
  relying on the single stream's in-order execution;
- the FFN graph is captured **and** replayed on that same non-NULL stream.

## Why B3A's two-stream design exists (and what it costs)

`CudaRuntime::initialize_impl` creates the runtime stream via
`context.default_stream()`, which returns `cu_stream = null_mut` (the NULL
stream). B3A-SMOKE proved the NULL stream cannot be captured
(`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`), so B3A was forced onto a *separate*
non-NULL `cap_stream` for graph capture/replay. Crossing between the runtime
(NULL) stream and `cap_stream` then required, per layer per token:

1. `runtime.stream().synchronize()` — wait for attention's output,
2. `cap_stream.memcpy_dtod(h_post_attn_dev → arena input)` — **D2D seed**,
3. `cap_stream` graph replay,
4. `cap_stream.synchronize()` — wait for the graph output,
5. `runtime.stream().clone_dtod(arena output)` — **D2D output read**.

Under honest accounting (PR #50) this yields only **18.2%** total submission
reduction (622.9 → 509.7/tok) and **+91% syncs** — failing the ≥25% gate and
the no-sync-regression gate. The single-stream redesign removes items 1, 2, 4,
5 entirely.

---

## The six questions, answered with code evidence + measurement

### Q1. Can the runtime stream change from `default_stream()` to `new_stream()` without breaking KV cache, weight cache, state-dump, prefill, or no-CUDA behavior?

**Yes — measured.** The runtime stream is created in exactly one place
(`runtime.rs:175`). Every other consumer takes the stream as a
`&Arc<CudaStream>` parameter and is therefore stream-agnostic:

| consumer | coupling | non-NULL-safe? |
|---|---|---|
| `WeightCache::get_or_upload_bytes/f32` (`weight_cache.rs:168,190`) | `stream.clone_htod(...)` | ✅ stream-scoped |
| `ResidentKvCache` (`kv_cache.rs:44,95,118`) | `stream: &Arc<CudaStream>` param | ✅ stream-scoped |
| every `launch_*_into` / `launch_*_dev` (`runtime.rs`) | `&CudaStream` / `self.stream` | ✅ stream-scoped |
| `sync_dtoh_f32` (state-dump + K/V readback) | `stream.clone_dtoh` + `stream.synchronize()` | ✅ works on any stream |
| prefill (`host_prefill_kquant` → `host_prefill_attention_block`) | routes through `self`/`runtime.stream()` | ✅ same path |
| no-CUDA scaffold (`runtime: None`) | no stream at all | ✅ unaffected |

`new_stream()` (cudarc 0.19.8 `core.rs:674`) creates a real
`CU_STREAM_NON_BLOCKING` stream (non-NULL). The only behavior change versus
the NULL/legacy stream is losing the legacy implicit global sync — irrelevant
here because LARQL is single-stream.

**Measured experiment (reverted; not in PR #50):** change
`runtime.rs:175` to `context.new_stream()?` and run the full suite:

| mode | result |
|---|---|
| `GRAPHS=0` (non-NULL runtime stream) | **204/204 pass** |
| `GRAPHS=1` (non-NULL runtime stream + old `cap_stream`) | **203/204** — all resident-hidden parity tests pass; the 1 failure is `b3a_smoke_repeated_capture_teardown_lifecycle` (`STREAM_CAPTURE_INVALIDATED`), the known-fragile two-stream capture lifecycle that **B3B removes** |

**Conclusion Q1:** moving every resident kernel onto one non-NULL stream is
measured-feasible and breaks nothing on the critical path.

### Q2. Can every resident decode operation on the critical path use the same non-NULL stream?

**Yes.** The decode critical path (`host_decode_token_resident`) is already
single-streamed through `runtime.stream()` / `self.stream`:

```
embed (host) → upload → decode_attention → kv_append → residual/norm/RoPE
  → [FFN graph] → next layer → … → final norm → hidden readback → lm-head
```

Every device kernel in this chain is launched on `runtime.stream()` today.
B3B additionally moves the FFN graph capture + replay onto that same stream
(see Q3). No operation on the critical path requires a second stream.

### Q3. Can graph capture occur on that stream while preserving stable buffers and avoiding cudarc event-tracking capture failures?

**Yes.** Two sub-questions:

1. **Capture on a non-NULL stream** — already proven: B3A's `cap_stream` is
   itself a `ctx.new_stream()`, and capture/instantiate/replay work on it
   (`b3a_smoke_*`, 204/204). B3B simply captures on the runtime stream instead
   of a second stream.

2. **Event tracking** — B3A-SMOKE finding #2: `launch_builder.arg(&CudaSlice)`
   injects `cuStreamWaitEvent` when event tracking is enabled, which inside a
   capture yields `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`. B3A already solves
   this by calling `cap_stream.context().disable_event_tracking()`
   *context-wide* (`ffn_graph_state.rs:268`), and the **204/204 GRAPHS=1 suite
   proves** the resident decode path is correct with event tracking disabled.
   Rationale: on a **single stream**, CUDA executes work in submission order,
   so explicit event ordering is redundant for correctness — disabling it is
   safe, and is what makes same-stream capture possible.

**Stable buffers:** the arena still owns `hidden_a`/`hidden_b` at fixed
addresses for the generation; each layer's `ResidentFfnGraph` still owns its
scratch + retained weights. The graph binds those addresses at capture and
reads whatever the attention block wrote there on replay (stable-pointer
replay, validated by B3A-SMOKE step 4).

### Q4. Can the implementation remove BOTH per-layer D2D handoff submissions, rather than merely replacing synchronization with events?

**Yes — this is the core of B3B and the reason it can meet the gate.** The two
copies exist only because attention writes into a *fresh* buffer
(`h_post_attn_dev`) while the graph reads a *different* stable buffer (the
arena input). On a single stream the addresses can be unified:

- **Attention residual-add → arena input slot.** Today
  `resident_kv_decode_attention` (`pipeline.rs:1973-1992`) ends with
  `launch_residual_add_dev(...)` which **allocates** a fresh `h_post_attn_dev`.
  B3B gives it an `*_into(&mut arena.input(layer.flip))` form using the
  **already-existing** `launch_residual_add_into` primitive (B3A-4). Attention
  then writes its residual **directly into the graph's stable input buffer**.
- **Graph output → next layer's attention input.** The graph writes into
  `arena.output(layer.flip)` (= `arena.input(next layer flip)`); the next
  layer's attention reads that same address. No copy.

Stream in-order execution guarantees: attention-write → graph-read →
graph-write → next-attention-read, with **zero** D2D and **zero** explicit
sync per layer. This removes both `memcpy_dtod` (seed) and `clone_dtod`
(output) — 72/tok — **and** the cross-stream syncs that bracket them.

Boundary conditions (the token-boundary upload for layer 0's input, and the
final-layer output readback) remain one HtoD and one DtoH per token — these
already exist today and are not graph-path overhead.

### Q5. What operations still require separate streams or host synchronization after the redesign?

- **Separate streams: none** in steady-state decode. Everything is on the one
  non-NULL runtime stream. The `cap_stream` field and its creation are deleted.
- **Host synchronization** survives only at true host-readback boundaries:
  - **Final hidden-state readback** (hidden → host) before lm-head / output —
    one `synchronize()` + `clone_dtoh` per token (exists today).
  - **K/V host-mirror readbacks** (`k_new_row`, `v_new_row` via `sync_dtoh_f32`)
    that maintain the resident-KV host mirror for fallback/state-dump — these
    are inherent to the resident-KV design, not graph-path overhead, and remain.
  - **State-dump readback** (diagnostic only).
- **No per-layer syncs.** The 76/tok cross-stream syncs are deleted; layer-to-
  layer ordering is by stream submission order.

### Q6. Corrected expected total-host-submission reduction (every graph, D2D, event, sync counted)

Baseline (GRAPHS=0, measured in PR #50): **622.9/tok** = 622.9 launches
(incl. 252 FFN direct launches = 7/layer × 36) + 83.3 syncs.

B3B single-stream (projected, GRAPHS=1):

| component | /tok | basis |
|---|---|---|
| non-FFN direct launches | ~370.9 | 622.9 − 252 FFN direct launches |
| FFN graph submissions | ~36 | one `graph.launch()` per layer |
| D2D handoff | **0** | both copies removed (Q4) |
| cross-stream syncs / events | **0** | single stream, no host blocking per layer |
| **TOTAL host submissions** | **~407** | |
| syncs (readback only) | ~83.3 | unchanged from baseline — no regression |

**Projected reduction: (622.9 − 407) / 622.9 ≈ 34.6% — clears the ≥25% gate**,
with **no sync regression** (syncs stay at ~83.3, the +76 cross-stream gone).

For comparison: B3A two-stream measured **18.2%** with +91% syncs. The
difference (~16 pp + the sync regression) is exactly the seed/output D2D
(72/tok) and cross-stream syncs (76/tok) that single-stream removes.

> **Wall-clock is an open measurement**, not a projection. B3A showed wall-clock
> flat because the cross-stream syncs (pipeline stalls) offset the launch-count
> savings. Removing ~76 syncs/tok should recover wall-clock, but the ≥1% gate
> must be measured on the RTX 3060 before default-on (per the B3A-11 gate).

---

## Design (implementation outline — not yet coded)

1. **`CudaRuntime::initialize_impl`** (`runtime.rs:175`): `context.default_stream()`
   → `context.new_stream()?`. Add a one-time
   `context.disable_event_tracking()` (safe for single-stream; proven by B3A's
   context-wide disable + 204/204).
2. **Delete `cap_stream`** from `ResidentDecodeArena` (`ffn_graph_state.rs`).
   The arena keeps `hidden_a`/`hidden_b`/`flip` but captures and replays on
   `runtime.stream()`.
3. **Attention residual-add into the arena** (`pipeline.rs:1973-1992`): add an
   `*_into(&mut arena.input(flip))` form to `resident_kv_decode_attention` so
   the FFN input is the arena slot directly (Q4).
4. **`build_and_launch_ffn_graph` / `replay_ffn_graph`** (`pipeline.rs`): drop
   the seed `memcpy_dtod`, the output `clone_dtod`, and all four cross-stream
   `synchronize()` calls; capture/replay on `runtime.stream()`; the graph's
   output is the arena slot the next layer reads.
5. **Counters** (already correct from PR #50): with D2D=0 and cross-stream
   syncs=0 on the happy path, `TOTAL host submissions` becomes the gate metric
   directly.

## Risks / unknowns (honest)

- **Attention→arena plumbing (Q4) is the bulk of the real work.** It touches
  `resident_kv_decode_attention`'s return shape (currently returns an owned
  `CudaSlice`) and the decode loop's buffer handoff. Contained, but it is the
  non-trivial part.
- **Non-blocking stream semantics:** `CU_STREAM_NON_BLOCKING` cannot be
  implicitly synced by the NULL stream. Irrelevant for single-stream decode;
  flagged for any future host code that might assume legacy NULL-stream sync.
- **Event-tracking-off context-wide:** proven safe by B3A (204/204 GRAPHS=1),
  but B3B makes it unconditional (not just when the arena exists). Re-validate
  the full suite both modes.
- **Wall-clock gate unmeasured** — the ≥1% improvement is plausible (fewer
  syncs) but not guaranteed; must be measured same-day A/B before default-on.
- **Capture interleaved with normal kernels on one stream** is standard CUDA,
  but the B3A smoke lifecycle showed a rare `STREAM_CAPTURE_INVALIDATED` under
  repeated capture — B3B must add a create→replay→reset→rebuild stress test on
  the single stream.

## Merge gates (B3B)

- 204/204 `larql-compute-cuda` under both `GRAPHS=0` and `GRAPHS=1`.
- `fmt` + `clippy -D warnings` clean.
- Honest `TOTAL host submissions` reduction **≥25%** (projected ~34.6%) on the
  real Q4_K_M benchmark, RTX 3060.
- **No sync regression** (syncs ≈ baseline; the +76 cross-stream gone).
- Same-day median decode **≥1%** faster before default-on (measured, not
  projected).
- Graph parity `max_abs < 1e-3`; resident-KV/hidden 100% active.
- Reset/rebuild/fallback/teardown + repeated single-stream capture lifecycle
  tests pass.
- Events are adopted **only** if single-stream is proven infeasible.

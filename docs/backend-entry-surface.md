# Backend entry surface: should `KvDispatch` become a primary CUDA surface?

**Status:** Decision (SPIKE — no code changes).
**Task:** GPU-6001.
**Scope:** `crates/larql-compute-cuda`, `crates/larql-kv/src/engines/*`.
**Base branch:** `feat/cuda-native-activation-kernels`.

## TL;DR

1. **Recommend option (c): a narrow bridge** — wire the highest-value engines
   (`markov_residual`, `markov_residual_codec`, `unlimited_context`,
   `boundary_per_layer`) into the **existing** `DecodeBackend` host-orchestrated
   decode/prefill surface by giving `KvDispatch` a device-resident
   `coarse_*_with_state` fast path that *delegates to the `DecodeBackend`
   pipeline internally* rather than re-implementing a second CUDA attention
   path. Do **not** implement `KvDispatch` natively on CUDA as a parallel
   kernel surface (option a), and do **not** force a broad engine rewrite onto
   `DecodeBackend` (option b).
2. **GPU-2001 (local CUDA attention + remote FFN via `DecodeBackend`) proceeds
   regardless of the `KvDispatch` decision**, via the `DecodeBackend` surface.
   `DecodeBackend` is already the primary useful CUDA path today and GPU-2001
   depends only on it.

## 1. The gap

`impl KvDispatch for CudaBackend`
(`crates/larql-compute-cuda/src/kv_dispatch_impl.rs:10-167`) is a **pure CPU
delegate**: every method forwards to `const CPU: CpuBackend = CpuBackend;`.
Meanwhile the same backend ships a *native* `DecodeBackend` impl
(`trait_impl.rs:238-…`) that runs a host-orchestrated decode/prefill pipeline
with q4k/q6k matvec/matmul + device-resident KV cache
(`pipeline::host_decode_token`, `pipeline::host_prefill_kquant`).

The consequence: most `larql-kv` engines drive compute through `KvDispatch`
(attention_step / coarse_prefill / coarse_decode_step_with_state / …), so on
CUDA they get **zero** GPU benefit even though the kernels they conceptually
need already exist on the `DecodeBackend` side. This is the largest long-term
gap between "the kernels exist" and "CUDA is useful for LARQL inference".

## 2. Engine inventory — entry surface per engine

Each engine in `crates/larql-kv/src/engines/*` was read and classified by the
trait surface its compute path actually calls into on the backend:

| Engine | Primary entry surface on backend | Gets CUDA benefit today? | Via `DecodeBackend`? |
|---|---|---|---|
| `standard` | `KvDispatch::attention_prefill` / `attention_step` (via `kv_prefill_via_dispatch` / `kv_decode_step_via_dispatch`) | **No** (CUDA delegates to CPU) | No |
| `no_cache` | `KvDispatch` (full re-forward each step) | **No** | No |
| `boundary_kv` | wraps `standard` (`StandardEngine`) | **No** | No |
| `markov_residual` | `KvDispatch::coarse_prefill_with_state` / `coarse_decode_step_with_state_masked` (`dispatch.rs`) | **No** | No |
| `markov_residual_codec` | same coarse surface as markov_residual | **No** | No |
| `boundary_per_layer` | same coarse surface (`dispatch.rs`) | **No** | No |
| `unlimited_context` | same coarse surface (`dispatch.rs`) | **No** | No |
| `turbo_quant` | `KvDispatch::coarse_prefill_with_state` / `coarse_decode_step_with_state` + `compressed_kv_append` (codec encode on host) | **No** | No |
| `apollo` | `forward_from_layer` / `forward_raw_logits` (residual injection, crystal_layer fast path) — **not** KvDispatch coarse | **No** | No |

**Key methods assessed** (the `KvDispatch` surface the engines hit):
`attention_step`, `coarse_prefill`, `coarse_decode_step_with_state`,
`compressed_kv_append`, `recompute_kv_from_residuals`,
`attention_prefill`. On CUDA every one of these is a CPU forward today.

**DecodeBackend surface** (the one that IS native on CUDA): `decode_token`,
`decode_token_with_state_dump_masked`, `prefill_kquant`, plus
`has_kv_cache` / `populate_kv_layer` / `kv_cache_len` / `truncate_kv_cache` /
`reset_kv_cache`. The native `decode_attention` / `prefill_attention` kernels
operate on a device-resident KV cache directly — **not** through `KvDispatch`.

**Net:** every engine listed above is CPU-only on CUDA today. None of them
route through `DecodeBackend`.

## 3. The three options compared

### (a) Implement `KvDispatch` natively on CUDA

Implement each `KvDispatch` method as its own CUDA kernel surface:
`attention_step`, `attention_prefill`, `coarse_prefill_with_state`,
`coarse_decode_step_with_state`, `compressed_kv_append`,
`recompute_kv_from_residuals`, …

- **Pro:** every engine lights up with no engine-side change; the trait
  contract (opaque `KvHandle`) stays intact.
- **Con (decisive):** this builds a **second** CUDA attention + KV-cache
  surface parallel to the `DecodeBackend` host-orchestrated pipeline that
  already has `decode_attention` / `prefill_attention` + device KV. Two device
  KV layouts, two readback contracts, two state-dump shapes, two things to keep
  bit-identical to the CPU reference. The `KvDispatch::KvHandle` is an opaque
  host-owned type today (`alloc_kv_buffer` returns a host `Array2`-backed
  handle); making it device-resident is a cross-cutting type change that
  touches every engines `dispatch.rs`.
- **Cost:** large. Every method needs a native path + parity gate + a
  device-resident `KvHandle` variant.
- **Verdict: rejected.** Duplicates the kernel surface that already exists on
  `DecodeBackend`.

### (b) Migrate engines toward `DecodeBackend`

Rewrite each engines compute path to call `DecodeBackend::decode_token` /
`prefill_kquant` directly instead of the `KvDispatch` coarse surface.

- **Pro:** single CUDA surface (`DecodeBackend`); no `KvDispatch` native work
  needed.
- **Con (decisive):** `DecodeBackend` is a **full-pipeline** surface (embed →
  all layers → logits). The engines under `larql-kv` exist precisely because
  they manage KV state *differently* from a plain full forward — residual
  replacement, per-window checkpoints, codec compression, boundary injection.
  Their `dispatch.rs` paths consume a **per-layer state dump**
  (`PerLayerDecodeState`: `h_in_per_layer`, `k_new_per_layer`,
  `v_new_per_layer`) and repack it into their own store. `DecodeBackend` today
  exposes a full-token state dump (`decode_token_with_state_dump_masked`) but
  not the per-engine KV repackaging seam those engines need. Migrating them
  means re-architecting the engine/backend contract for 6+ engines.
- **Cost:** very large; high blast radius across the most-used engines.
- **Verdict: rejected as the primary direction.** Too invasive for the
  near term; the per-layer state-dump seam the engines want is the real
  ask, and it can be delivered without a full rewrite (see (c)).

### (c) Narrow bridge — make `KvDispatch` coarse delegate to `DecodeBackend` internally ✅ RECOMMENDED

Keep `KvDispatch` as the engine-facing trait (no engine rewrites), but replace
the CPU forward inside the **coarse** methods with an internal dispatch to the
native `DecodeBackend` host-orchestrated pipeline when a runtime is present.
Concretely, on `CudaBackend`:

- `coarse_prefill_with_state` / `coarse_decode_step_with_state[_masked]`:
  when `native_runtime_available()`, route through
  `host_prefill_kquant` / `host_decode_token` (the same code
  `DecodeBackend` already uses), capture `PerLayerDecodeState` from the
  pipelines state-dump path, and return it through the `KvDispatch` signature.
  CPU fallback unchanged.
- `attention_step` / `attention_prefill` (used by `standard` / `no_cache` /
  `boundary_kv`): defer — these engines are lower priority (see §4) and can
  stay CPU until a follow-up. `standard` already has an async-backend slot
  that could carry a later DecodeBackend bridge.
- `compressed_kv_append` / `recompute_kv_from_residuals`: leave on CPU — these
  are codec/recompute utilities where host work dominates and the codec is
  host-side by design.

- **Pro:** lights up the four high-value engines with **one** bridge, reusing
  the existing native kernels. No second CUDA surface. No engine-side change.
  The `KvDispatch` trait contract (opaque `KvHandle`) is preserved — the
  handle stays host-owned; the device KV cache lives inside the pipeline as it
  does today.
- **Con:** `KvHandle` remains a host-owned type; there is no device-resident
  `KvHandle` variant. This is acceptable because the device KV cache is already
  managed internally by the `DecodeBackend` pipeline (`populate_kv_layer`,
  `kv_cache_len_native`); exposing it as a `KvHandle` would be option (a)
  again.
- **Cost:** medium. One internal delegation layer + per-layer state-dump
  wiring + parity gates against the existing CPU `KvDispatch` reference.
- **Verdict: recommended.** Maximum engine coverage for minimum new kernel
  surface, and it composes cleanly with GPU-2001.

### Is a device-resident `KvHandle` needed?

**No, not for the recommended path.** The device-resident KV cache already
exists — it lives inside the `DecodeBackend`/pipeline lifecycle
(`preallocate_kv_cache_per_layer`, `populate_kv_layer`, `reset_kv_cache`).
Making `KvHandle` itself device-resident would require every engines
`dispatch.rs` to handle a device pointer, which is exactly the cross-cutting
change option (a) demands. Keep `KvHandle` host-owned; let the bridge own the
device cache internally. `DecodeBackend` remains the primary useful CUDA path.

## 4. Impact estimate & which engines stay CPU

With option (c) the coarse-bridge, coverage on CUDA becomes:

| Engine | After bridge | Notes |
|---|---|---|
| `markov_residual` | **CUDA** via coarse→DecodeBackend bridge | highest-value residual engine |
| `markov_residual_codec` | **CUDA** via coarse→DecodeBackend bridge | mirrors markov_residual |
| `unlimited_context` | **CUDA** via coarse→DecodeBackend bridge | per-window checkpoints |
| `boundary_per_layer` | **CUDA** via coarse→DecodeBackend bridge | per-layer codec policy |
| `turbo_quant` | **partial** — coarse path bridges, but `compressed_kv_append` + codec encode stay host | K/V capture goes fast; compression stays CPU |
| `standard` | **CPU only** until follow-up | uses `attention_step`/`attention_prefill`, not coarse; lower priority (full-KV reference) |
| `no_cache` | **CPU only** until follow-up | debug/correctness fallback; O(N²), not worth accelerating |
| `boundary_kv` | **CPU only** until follow-up | wraps `standard`; rides whatever `standard` gets |
| `apollo` | **CPU only** until follow-up | uses `forward_from_layer` residual injection, a different seam; separate task |

**Engines that remain CPU-only until follow-up lands:** `standard`, `no_cache`,
`boundary_kv`, `apollo`. (`turbo_quant` is partial.)

This is acceptable: `standard`/`boundary_kv` are the full-KV reference path
(lowest marginal win once the residual/checkpoint engines are fast), `no_cache`
is a debug fallback, and `apollo` has its own crystal-layer fast path that
needs a separate design.

## 5. GPU-2001 — proceeds regardless ✅

**GPU-2001 (local CUDA attention + remote FFN via `DecodeBackend`) proceeds
via the `DecodeBackend` surface regardless of the `KvDispatch` decision.**

Rationale: GPU-2001s mechanism is the `DecodeBackend::decode_token` /
`prefill_kquant` host-orchestrated pipeline, which is already native on CUDA
and is *independent* of `KvDispatch`. The remote-FFN split plugs into the
pipelines FFN step, not into any `KvDispatch` method. The recommendation in
§3(c) *reuses* the same pipeline for the coarse bridge, so GPU-2001 and the
bridge are complementary, not sequenced. Do not block GPU-2001 on GPU-6001.

## 6. Follow-up implementation items (file as separate tasks)

1. **GPU-6002 — coarse→DecodeBackend bridge for `CudaBackend`.** Implement
   `coarse_prefill_with_state` / `coarse_decode_step_with_state[_masked]` on
   `CudaBackend` by delegating to `host_prefill_kquant` /
   `host_decode_token` when `native_runtime_available()`, capturing
   `PerLayerDecodeState` from the state-dump path. CPU fallback unchanged.
   Parity gate vs CPU `KvDispatch` reference. Unblocks: markov_residual,
   markov_residual_codec, unlimited_context, boundary_per_layer on CUDA.
2. **GPU-6003 — `turbo_quant` partial bridge.** Verify the coarse bridge gives
   the K/V capture fast path; characterise the residual host cost of
   `compressed_kv_append` + codec encode; decide whether a host-side codec
   kernel is worth it.
3. **GPU-6004 — `standard` / `boundary_kv` bridge via `attention_step` /
   `attention_prefill`.** Either bridge those `KvDispatch` methods to
   `DecodeBackend`, or migrate `StandardEngine` to the async-backend slot with
   a DecodeBackend-backed async impl. Lower priority (full-KV reference).
4. **GPU-6005 — `apollo` device path.** Separate design: `forward_from_layer`
   residual injection on CUDA, distinct from the coarse surface.
5. **`docs/gpu.md` update.** Add a "KvDispatch coarse bridge" subsection to the
   "What is NOT accelerated" list once GPU-6002 lands, mirroring the existing
   `KvDispatch` (CUDA) note.

## 7. Evidence

- `crates/larql-compute-cuda/src/kv_dispatch_impl.rs:10-167` — entire
  `impl KvDispatch for CudaBackend` delegates to `const CPU: CpuBackend`.
- `crates/larql-compute-cuda/src/trait_impl.rs:238` —
  `impl DecodeBackend for CudaBackend` (native decode/prefill pipeline).
- `crates/larql-compute-cuda/src/trait_impl.rs:465` — `supports()` advertises
  only `QuantMatVec | DecodeToken | PrefillQ4` when native.
- Engine dispatch paths (`crates/larql-kv/src/engines/*/dispatch.rs`) all call
  `coarse_prefill_with_state` / `coarse_decode_step_with_state[_masked]`.
- `crates/larql-inference/src/kv_dispatch/helpers.rs` —
  `kv_prefill_via_dispatch` / `kv_decode_step_via_dispatch` call
  `backend.attention_prefill` / `backend.attention_step` (used by `standard`,
  `no_cache`, `boundary_kv`).
- `docs/gpu.md` — "What is NOT accelerated" already documents the
  `KvDispatch` (CUDA) CPU-delegate status.

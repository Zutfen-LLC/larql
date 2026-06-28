# CUDA + Vulkan Implementation Plan

## Purpose

This document is the full implementation plan for adding CUDA and Vulkan backends to LARQL.

It is intentionally broader than `HANDOFF.md`:

- `HANDOFF.md` = current state, immediate resume notes, likely repair points
- `CUDA_VULKAN_IMPLEMENTATION_PLAN.md` = complete end-to-end implementation target

## Summary

Add `larql-compute-cuda` and `larql-compute-vulkan` as first-class sibling crates to `larql-compute-metal`, wire explicit backend selection through inference and CLI, and land a first usable accelerator milestone around the production Q4K/Q6K decode bench path.

Implementation defaults:

1. build shared backend-selection scaffolding first
2. bring up CUDA first
3. bring up Vulkan on the same contracts
4. define MVP as:
   - Q4K/Q6K decode
   - prefill
   - lm-head top-k
   - enough integration for `larql bench`
5. keep kernel code backend-specific
6. share tests, fixtures, capability contracts, and dispatch metadata only

## Architectural Constraints

### Crate structure

New GPU backends must follow the same sibling-crate model as Metal:

- `larql-compute-metal`
- `larql-compute-cuda`
- `larql-compute-vulkan`

They are peers, not layers on top of each other.

Each backend crate owns:

- backend type
- kernel/module compilation
- dispatch geometry
- backend-specific options
- trait impls
- tests local to that backend

### Dependency flow

Keep the existing dependency chain intact:

- substrate traits and shared abstractions stay in `larql-compute`
- engine/backend selection stays in `larql-inference`
- CLI/backend flags stay in `larql-cli`
- no `larql-vindex` imports inside backend crates except through approved trait surfaces already in `larql-compute`

### Capability honesty

Backends must advertise only what they actually implement.

If a method is not implemented end-to-end, it must:

- return `None`, or
- report `supports=false`

No silent CPU fallback inside methods that are supposed to represent native backend capabilities once the backend moves beyond scaffold phase.

## Implementation Phases

### Phase 1: Workspace and feature model

#### Goals

- add new workspace members
- add explicit feature flags
- preserve `gpu` as a Metal compatibility alias

#### Deliverables

Workspace members:

- `crates/larql-compute-cuda`
- `crates/larql-compute-vulkan`

Consumer features:

- `metal`
- `cuda`
- `vulkan`
- `gpu-all`
- `gpu` -> alias of `metal`

#### Success criteria

- any consumer crate can compile with any subset of:
  - `metal`
  - `cuda`
  - `vulkan`
- repo no longer assumes `gpu == Metal only`

## Phase 2: Shared backend selection API

### Goals

Replace implicit Metal-first helpers with explicit backend selection.

### Deliverables

In `larql-inference`:

- `pub enum ComputeBackendKind { Auto, Cpu, Metal, Cuda, Vulkan }`
- `compute_backend(kind) -> Box<dyn larql_compute::ComputeBackend>`
- `engine_backend(kind) -> Box<dyn EngineBackend>`
- `async_engine_backend(kind) -> Box<dyn AsyncComputeBackend>`

Compatibility wrappers:

- `default_compute_backend() -> compute_backend(Auto)`
- `default_engine_backend() -> engine_backend(Auto)`
- `default_async_engine_backend() -> async_engine_backend(Auto)`

### Auto policy

- macOS: `metal -> vulkan -> cpu`
- Linux/Windows: `cuda -> vulkan -> cpu`

### Error behavior

- under `Auto`, unavailable backends are non-fatal
- explicit unavailable backend selection must error loudly with actionable text

## Phase 3: CLI and user-facing selection

### Goals

Move runtime selection from Metal-specific toggles to generic backend names.

### Deliverables

For single-runtime commands:

- `--backend <auto|cpu|metal|cuda|vulkan>`

For multi-backend commands:

- `--backends <LIST>`

Compatibility:

- preserve `--metal` as alias to `--backend metal`

### Commands to cover

At minimum:

- `larql run`
- `larql dev walk`
- `larql bench`
- `larql shannon`

### Docs behavior

Help text must describe supported values generically and note that feature/platform availability varies.

## Phase 4: CUDA crate MVP

### Backend technology

Use:

- `cudarc`
- dynamic loading
- NVRTC runtime compilation

Avoid forcing build-time CUDA toolchain linkage.

### Required crate shape

Expected modules:

- `src/lib.rs`
- `backend/`
- `buffers`
- `kernels`
- `ops`
- `decode`
- `trait_impl`
- `kv_dispatch_impl`
- `async_compute_backend_impl`
- `options`

### MVP API surface

Implement:

- `f32_gemv`
- `f32_gemv_topk1`
- `f16_gemv`
- `f16_gemv_topk1`
- `f16_gemv_topk`
- `q4k_matvec`
- `q4k_dual_matvec`
- `q4k_matmul`
- `q6k_matvec`
- `prefill_kquant`
- `decode_token`
- `decode_token_with_state_dump_masked`
- KV cache lifecycle:
  - `has_kv_cache`
  - `reset_kv_cache`
  - `kv_cache_len`
  - `truncate_kv_cache`
  - `preallocate_kv_cache_per_layer`

### Coarse dispatch

Implement coarse `KvDispatch` by routing through existing fused helpers, matching the current Metal coarse-path pattern.

### Async status

For first milestone:

- implement trait conformance
- do not promise throughput improvement yet
- report deferred/batching-specific capabilities as unsupported until real batching lands

## Phase 5: Vulkan crate MVP

### Backend technology

Use:

- `ash`
- runtime-loaded Vulkan
- `shaderc` for GLSL -> SPIR-V compilation under Vulkan feature

### Backend behavior

Match CUDA MVP behavior and surface exactly:

- same capability set
- same unsupported behavior
- same coarse `KvDispatch` bridge pattern

### Structure

Keep organization parallel to CUDA and Metal so future parity work is mechanical.

## Phase 6: Shared GPU conventions

### Goals

Keep backend-local kernels but shared backend conventions.

### Deliverables

Each backend should have a local kernel handle abstraction carrying:

- compiled kernel/module handle
- dispatch geometry metadata
- stable kernel identifier for diagnostics

Dispatch sites should use geometry attached to the kernel/module, not duplicated constants scattered at call sites.

### Options policy

Only move settings into shared env parsing when they are truly cross-backend.

Keep backend-specific toggles local.

## Phase 7: Capability and fallback contract

### MVP-supported only

Backends should advertise only what is really implemented.

### Explicitly unsupported in MVP unless implemented later

Leave unsupported unless there is a real end-to-end implementation for:

- `HybridAttention`
- `PerLayerEmbeddings`
- local MoE GPU dispatch
- AHORD head replacement paths
- split profiling/timing
- true deferred `AsyncComputeBackend` batching
- specialized `KvDispatch` capability flags such as:
  - `FusedAttentionStep`
  - `WindowedAttentionStep`
  - `NativeKvCodec`
  - `PipelinedBoundaryUpload`
  - `FusedResidualNorm`
  - `KvHandleNative`

### Fallback rule

Unsupported methods must fail conservatively via `None` or `supports=false`.

## Public API Targets

### larql-inference

Add:

- `ComputeBackendKind`
- explicit backend factory functions

### backend crates

Add constructors:

- `larql_compute_cuda::cuda_backend()`
- `larql_compute_vulkan::vulkan_backend()`

Prefer `Result` over `Option` when runtime failure details matter.

### CLI

Accept:

- `cpu`
- `metal`
- `cuda`
- `vulkan`
- `auto`

Keep existing names working where compatibility matters.

## Testing Plan

### Build coverage

Run:

```bash
cargo check --workspace
cargo check -p larql-compute-cuda --features cuda
cargo check -p larql-compute-vulkan --features vulkan
cargo check -p larql-inference --features metal
```

Also verify consumer crates build with subsets of:

- `metal`
- `cuda`
- `vulkan`

### Unit and parity tests

Add CUDA parity tests against CPU for:

- `q4k_matvec`
- `q4k_matmul`
- `q6k_matvec`
- `f32_gemv_topk1`
- `f16_gemv_topk1`

Add same Vulkan parity tests.

Reuse:

- `larql-compute::test_fixtures`

Do not loosen tolerances per backend without evidence.

### Decode integration tests

Add CUDA and Vulkan integration tests that mirror Metal decode integration style.

Assert:

- prefill returns correctly shaped hidden states
- decode advances KV length correctly
- masked state dump honors:
  - `Full`
  - `HOnly`
  - `None`
- heterogeneous per-layer KV shape preallocation works

Also add coarse `KvDispatch` tests for prefill/decode with Q4K fixtures.

### CLI tests

Cover:

- parser accepts `cpu`, `metal`, `cuda`, `vulkan`, `auto`
- explicit unavailable backend errors cleanly
- `auto` falls back to next available backend
- `--metal` alias still works

### Bench validation

Functional validation targets:

- `larql bench ... --backends cuda,cpu`
- `larql bench ... --backends vulkan,cpu`

MVP success criterion:

- functional completion
- parity with CPU/reference path
- no promise to match Metal tok/s yet

## Rollout Order

1. workspace crates and feature plumbing
2. `ComputeBackendKind` and shared backend selection factories
3. CLI parsing/docs for generic backend names
4. CUDA dense/quant primitives and decode MVP
5. CUDA coarse `KvDispatch` and integration tests
6. Vulkan dense/quant primitives and decode MVP
7. Vulkan coarse `KvDispatch` and integration tests
8. hardware-specific CI jobs for CUDA and Vulkan
9. follow-on planning for:
   - async batching
   - MoE
   - hybrid attention
   - PLE
   - profiling

## Current Session Status

As of the latest handoff (Session 4):

- **Phase 1 (workspace + features): DONE** and verified — all feature subsets compile.
- **Phase 2 (shared backend selection API): DONE** and verified — `ComputeBackendKind`, explicit factories, `Auto` policy, `BackendSelectionError`. The Metal arm was fixed to gate on `target_os = "macos"` (was breaking Linux + `--features metal`).
- **Phase 3 (CLI + user-facing selection): DONE** for `run`/`walk`/`bench`/`shannon` — all accept `--backend <auto|cpu|metal|cuda|vulkan>`, `--metal` preserved as alias, routed through `crates/larql-cli/src/commands/backend.rs`. Remaining polish only: stale Metal-only help text/comments, and `run` still has Metal-specific construction in remote FFN/MoE and `--experts` branches.
- **Phase 4 (CUDA crate MVP): STARTED, still early** — `larql-compute-cuda` now has:
  - `cudarc` wired with dynamic loading + NVRTC
  - a new optional runtime/bootstrap layer (`src/backend/runtime.rs`)
  - panic-safe fallback when probing CUDA on non-CUDA hosts (missing `libcuda` no longer aborts tests)
  - a first native `q4k_matvec` launch path
  Everything else in the CUDA crate still delegates dense/quant/KV/decode to CPU, and capability reporting intentionally remains conservative.
- **Phase 5 (Vulkan crate MVP): NOT STARTED** — `larql-compute-vulkan` is a parallel scaffold. No `ash`/`shaderc`, no real kernels.
- **Phase 6 (shared GPU conventions): PARTIAL** — kernel handle + dispatch geometry structs exist in both scaffolds but are not yet exercised by real kernels.
- **Phase 7 (capability + fallback contract): RECONCILED FOR SCAFFOLDS** — delegated CPU/reference methods remain callable for parity tests, but CUDA/Vulkan scaffolds now report `supports(...) == false` for accelerator capabilities and `supports_quant(...) == false` until native kernels land. `walk`'s Q4 path probes `PrefillQ4 + DecodeToken + Q4_K`: `auto` falls back to CPU when only scaffolds are present, while explicit CUDA/Vulkan fail loudly.
- **Phase 8 (CI jobs): NOT STARTED.**

Verification snapshot (CachyOS / x86_64-linux, rustc 1.96.0):

- `cargo check --workspace --exclude larql-python` — green
- `cargo check` on `metal`/`cuda`/`vulkan`/`cuda,vulkan`/`gpu-all` subsets — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo test -p larql-inference --lib` → 1243 passed
- `cargo test -p larql-cli --bins` → 243 passed
- `cargo test -p larql-compute-cuda` → 7 passed
- `cargo test -p larql-compute-vulkan` → 6 passed

Session 3 delta:

- `cargo test -p larql-compute-cuda` → 7 passed
- `cargo test -p larql-compute-vulkan` → 6 passed
- `cargo test -p larql-cli --bins` → 243 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green

Session 4 delta:

- `cargo check -p larql-compute-cuda` — green
- `cargo test -p larql-compute-cuda --offline` → 9 passed

Session 4 scope landed:

- `cudarc 0.19.8` added to `larql-compute-cuda`
- embedded NVRTC source for `q4k_matvec`
- optional CUDA runtime bootstrap with dynamic probing
- panic-safe degrade-to-scaffold behavior on hosts without `libcuda`
- native `q4k_matvec` route wired behind the existing CPU fallback

Immediate next slice:

- keep Phase 7 honesty as-is (do **not** advertise CUDA Q4/decode support yet)
- continue Phase 4 with `q4k_matmul`, `q6k_matvec`, then prefill/decode

Pre-existing environment issues (not caused by this work): `larql-python` fails on PyO3 0.24 vs Python 3.14; `larql-compute-metal`'s macOS-gated *test binary* needs `blas_src` off-Apple (lib compiles fine); OpenBLAS must be installed system-wide for any test linking `larql-compute`.

For the immediate restart state, see:

- `HANDOFF.md`

## Definition Of Done

This effort is complete when:

1. CUDA and Vulkan are real selectable backends
2. explicit selection works across inference + CLI
3. `Auto` behaves correctly per platform
4. bench path runs through the new backends
5. parity/integration tests are green
6. unsupported capabilities are reported honestly
7. docs/help text no longer imply Metal is the only GPU backend

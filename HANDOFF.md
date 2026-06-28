# HANDOFF

## Goal

Resume the CUDA + Vulkan backend implementation for LARQL.

The work completed in this session focused on:

1. adding workspace/feature plumbing
2. adding explicit backend selection APIs
3. adding CUDA/Vulkan sibling crates as compileable scaffolds
4. starting CLI migration from Metal-only flags to generic backend names

This is not done yet. The new CUDA/Vulkan crates currently delegate most compute/KV behavior to CPU/reference paths and do not contain real accelerator kernels.

## What Landed

### Workspace + features

Added workspace members:

- `crates/larql-compute-cuda`
- `crates/larql-compute-vulkan`

Updated feature plumbing in:

- `Cargo.toml`
- `crates/larql-cli/Cargo.toml`
- `crates/larql-inference/Cargo.toml`
- `crates/larql-kv/Cargo.toml`
- `crates/larql-vindex/Cargo.toml`

Current feature model added:

- `metal`
- `cuda`
- `vulkan`
- `gpu-all`
- `gpu` remains an alias to `metal`

### larql-inference backend selection

In `crates/larql-inference/src/lib.rs`:

- added `ComputeBackendKind { Auto, Cpu, Metal, Cuda, Vulkan }`
- added:
  - `compute_backend(kind)`
  - `engine_backend(kind)`
  - `async_engine_backend(kind)`
- preserved wrappers:
  - `default_compute_backend()`
  - `default_engine_backend()`
  - `default_async_engine_backend()`

Current auto policy implemented:

- macOS: `metal -> vulkan -> cpu`
- non-macOS: `cuda -> vulkan -> cpu`

Explicit unavailable backends return `BackendSelectionError`.

### New backend crates

Added scaffold crates:

- `crates/larql-compute-cuda`
- `crates/larql-compute-vulkan`

Each crate currently contains:

- constructor API
- backend options
- kernel handle + dispatch geometry structs
- `ComputeBackend` impl
- `KvDispatch` impl
- `AsyncComputeBackend` impl
- parity-style tests

Important: these are scaffold backends, not real CUDA/Vulkan implementations yet.

Current behavior:

- dense/quant ops mostly delegate to `CpuBackend`
- KV dispatch delegates to CPU
- async dispatch delegates to CPU
- capability reporting is conservative-ish but still scaffolded

### CLI migration

Added shared parser/helpers in:

- `crates/larql-cli/src/commands/backend.rs`

Started migration of CLI/backend wiring:

- `bench` now parses generic backend names
- `run` has new `--backend`
- `walk` has new `--backend`
- `--metal` is still present as a compatibility alias

Files touched:

- `crates/larql-cli/src/commands/primary/bench/args.rs`
- `crates/larql-cli/src/commands/primary/bench/local.rs`
- `crates/larql-cli/src/commands/primary/bench/local_runtime.rs`
- `crates/larql-cli/src/commands/primary/bench/engine_runtime.rs`
- `crates/larql-cli/src/commands/primary/bench/remote_ffn_runtime.rs`
- `crates/larql-cli/src/commands/primary/bench/run.rs`
- `crates/larql-cli/src/commands/primary/run_cmd.rs`
- `crates/larql-cli/src/commands/extraction/walk_cmd.rs`
- `crates/larql-cli/src/main.rs`

## Important Caveat

This environment did not have `cargo` or `rustc` installed.

I could not run:

- `cargo check`
- `cargo test`
- any compile verification

So this handoff should assume there are likely compile errors or signature mismatches still present.

## Likely Problem Areas To Fix First

### 1. Compile the workspace immediately

First command to run in the next session:

```bash
cargo check -p larql-compute-cuda -p larql-compute-vulkan -p larql-inference -p larql-cli
```

Then likely:

```bash
cargo check --workspace
```

### 2. Expect CLI struct drift

Most likely compile failures are around:

- new `backend: String` fields in `RunArgs` / `WalkArgs`
- any constructor sites that still build those structs without the new field
- any lingering `args.backends.contains("metal")` or `args.metal` assumptions

Search for:

```bash
rg -n "args\\.metal|contains\\(\"metal\"\\)|--metal|RunArgs \\{|WalkArgs \\{" crates/larql-cli
```

### 3. Expect trait signature drift in new backend crates

The new CUDA/Vulkan `KvDispatch` impls were updated against the current trait shape by inspection, but they were not compiled.

Check:

- `crates/larql-compute-cuda/src/kv_dispatch_impl.rs`
- `crates/larql-compute-vulkan/src/kv_dispatch_impl.rs`
- `crates/larql-compute-cuda/src/async_compute_backend_impl.rs`
- `crates/larql-compute-vulkan/src/async_compute_backend_impl.rs`

### 4. bench path still needs a cleanup pass

`bench` is partially migrated, but it likely still needs:

- better handling of `auto` row labeling
- clearer behavior when multiple accelerators are requested
- a sweep for stale comments/help text mentioning Metal-only behavior

### 5. run/walk/shannon are not fully generalized yet

Current state:

- `run` and `walk` accept `--backend`
- much of their actual runtime logic still treats the accelerator path as effectively Metal-shaped
- `shannon` was not migrated yet and still has Metal-specific assumptions

That follow-up work should touch:

- `crates/larql-cli/src/commands/primary/shannon_cmd.rs`
- remaining Metal-only helper functions in `run_cmd.rs` / `walk_cmd.rs`

## Honest Status Of CUDA/Vulkan MVP

Not implemented yet:

- `cudarc`
- `ash`
- `shaderc`
- NVRTC / SPIR-V kernel compilation
- real `prefill_kquant`
- real `decode_token`
- real `decode_token_with_state_dump_masked`
- real KV cache lifecycle on device
- real `f32_gemv` / `f16_gemv` / `q4k_*` / `q6k_*` device kernels

What exists now is the repo-wide control plane needed to start that work without inventing it later.

## Suggested Next Session Order

1. run `cargo check` and fix compile errors
2. finish CLI migration to generic backend naming
3. make `shannon` follow the same backend-selection helper
4. tighten capability reporting in CUDA/Vulkan scaffolds
5. replace delegated CUDA hot paths with real kernels:
   - `f32_gemv_topk1`
   - `f16_gemv_topk1`
   - `q4k_matvec`
   - `q4k_matmul`
   - `q6k_matvec`
   - `prefill_kquant`
   - `decode_token`
6. then do the same for Vulkan

## Key Files Added

- `crates/larql-compute-cuda/Cargo.toml`
- `crates/larql-compute-cuda/src/lib.rs`
- `crates/larql-compute-cuda/src/backend/mod.rs`
- `crates/larql-compute-cuda/src/options.rs`
- `crates/larql-compute-cuda/src/kernels.rs`
- `crates/larql-compute-cuda/src/buffers.rs`
- `crates/larql-compute-cuda/src/ops.rs`
- `crates/larql-compute-cuda/src/decode.rs`
- `crates/larql-compute-cuda/src/trait_impl.rs`
- `crates/larql-compute-cuda/src/kv_dispatch_impl.rs`
- `crates/larql-compute-cuda/src/async_compute_backend_impl.rs`
- `crates/larql-compute-vulkan/Cargo.toml`
- `crates/larql-compute-vulkan/src/lib.rs`
- `crates/larql-compute-vulkan/src/backend/mod.rs`
- `crates/larql-compute-vulkan/src/options.rs`
- `crates/larql-compute-vulkan/src/kernels.rs`
- `crates/larql-compute-vulkan/src/buffers.rs`
- `crates/larql-compute-vulkan/src/ops.rs`
- `crates/larql-compute-vulkan/src/decode.rs`
- `crates/larql-compute-vulkan/src/trait_impl.rs`
- `crates/larql-compute-vulkan/src/kv_dispatch_impl.rs`
- `crates/larql-compute-vulkan/src/async_compute_backend_impl.rs`
- `crates/larql-cli/src/commands/backend.rs`

## Short Summary

Good progress on architecture and repo wiring.

Not production-ready yet.

The next session should begin with compilation and repair, then move from scaffold backends to real CUDA/Vulkan kernels.

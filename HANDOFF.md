# HANDOFF

## Goal

Resume the CUDA + Vulkan backend implementation for LARQL.

## Current Status (verified)

Three sessions of work have landed:

- **Session 1** (original scaffolding): workspace/feature plumbing, explicit backend selection APIs, CUDA/Vulkan sibling crates as compileable scaffolds, partial CLI migration. Could not compile-verify (no Rust toolchain on PATH).
- **Session 2** (verification + repair + finish CLI): brought up `cargo`/`rustc` (rustup, off-PATH), ran `cargo check`, fixed the one real compile breakage, finished the `shannon` CLI migration, and fixed three test/lint issues so the affected crates are green under `cargo test` and `cargo clippy -- -D warnings`.
- **Session 3** (capability honesty + walk backend dispatch): reconciled CUDA/Vulkan scaffold capability reporting so delegated CPU/reference methods remain callable for parity tests but no longer advertise native `QuantMatVec`/`F32Gemv`/`F16Gemv`/`Q4_K` support. `walk`'s Q4 predict/generate path now constructs the requested backend generically and gates the fused fast path on `PrefillQ4 + DecodeToken + Q4_K`; `--backend auto` falls back to CPU when only scaffolds are present, while explicit CUDA/Vulkan fail loudly until native kernels land.

**The new CUDA/Vulkan crates are still scaffold backends.** They delegate most compute/KV behavior to CPU/reference paths and contain no real accelerator kernels. What exists is the repo-wide control plane needed to start that work.

### Verification snapshot (Session 2, CachyOS / x86_64-linux, rustc 1.96.0)

Green:

- `cargo check --workspace --exclude larql-python`
- `cargo check` on every relevant feature subset: `metal`, `cuda`, `vulkan`, `cuda,vulkan`, `gpu-all` (for `larql-inference` and `larql-cli`)
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings`
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings`
- `cargo test -p larql-inference --lib` → 1243 passed
- `cargo test -p larql-cli --bins` → 243 passed
- `cargo test -p larql-compute-cuda` → 7 passed
- `cargo test -p larql-compute-vulkan` → 6 passed

Session 3 delta verified:

- `cargo test -p larql-compute-cuda` → 7 passed
- `cargo test -p larql-compute-vulkan` → 6 passed
- `cargo test -p larql-cli --bins` → 243 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green

Pre-existing environment issues (NOT caused by this work, NOT fixed):

- `larql-python` — PyO3 0.24.2 does not support Python 3.14 (needs `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` or a PyO3 upgrade). Excluded from checks.
- `larql-compute-metal` *test binary* — `tests/test_pipeline_and_moe.rs` does `extern crate blas_src;` and the macOS-gated test path needs `blas_src` off-Apple. The crate's *lib* compiles fine (empty off-macOS). Excluded from the `-D warnings` clippy pass; not a regression.
- OpenBLAS must be installed system-wide for any test that links `larql-compute` (`openblas-src` is configured with `features = ["system"]`). On this host: `sudo pacman -S openblas`.

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

Current feature model:

- `metal`
- `cuda`
- `vulkan`
- `gpu-all`
- `gpu` remains an alias to `metal`

Note: the backend crates themselves (`larql-compute-cuda`, `larql-compute-vulkan`) do **not** define their own `cuda`/`vulkan` features — the features live on the consumer crates. So `cargo check -p larql-compute-cuda --features cuda` errors with "does not contain this feature"; use `cargo check -p larql-inference --features cuda` instead.

### larql-inference backend selection

In `crates/larql-inference/src/lib.rs`:

- `pub enum ComputeBackendKind { Auto, Cpu, Metal, Cuda, Vulkan }`
- `pub enum BackendSelectionError` (with `Display` + `std::error::Error`)
- `compute_backend(kind) -> Result<Box<dyn ComputeBackend>, BackendSelectionError>`
- `engine_backend(kind) -> Result<Box<dyn EngineBackend>, BackendSelectionError>`
- `async_engine_backend(kind) -> Result<Box<dyn AsyncComputeBackend>, BackendSelectionError>`
- preserved wrappers: `default_compute_backend()`, `default_engine_backend()`, `default_async_engine_backend()` (all follow `Auto`)

Auto policy:

- macOS: `metal -> vulkan -> cpu`
- non-macOS: `cuda -> vulkan -> cpu`

Explicit unavailable backends return `BackendSelectionError::Unavailable`.

**Session 2 fix**: the `ComputeBackendKind::Metal` arms originally called `larql_compute_metal::metal_backend()` under only `#[cfg(feature = "metal")]`, but that function is additionally gated `#[cfg(target_os = "macos")]` in the metal crate (it compiles to an empty crate off-Apple). The arms now add a `target_os = "macos"` predicate and return `BackendSelectionError::Unavailable { reason: "Metal backend is only available on macOS" }` on non-macOS hosts with the `metal` feature on. This is what unblocked `cargo check --features metal` on Linux.

### New backend crates (scaffolds)

`crates/larql-compute-cuda` and `crates/larql-compute-vulkan`, each containing:

- constructor API (`cuda_backend()` / `vulkan_backend()` returning `Result`)
- backend options
- kernel handle + dispatch geometry structs
- `ComputeBackend` impl
- `KvDispatch` impl
- `AsyncComputeBackend` impl
- parity-style tests

**These are scaffold backends, not real CUDA/Vulkan implementations.** Current behavior:

- dense/quant ops mostly delegate to `CpuBackend`
- KV dispatch delegates to CPU
- async dispatch delegates to CPU
- capability reporting is conservative-ish but still scaffolded (see Phase 7 caveat below)

**Session 2 fixes in the scaffolds**:

- `crates/larql-compute-cuda/src/trait_impl.rs` and `crates/larql-compute-vulkan/src/trait_impl.rs` — rewrote `f16_gemv` inner loops to satisfy `clippy::needless_range_loop` (was failing `make ci`'s `-D warnings`).
- `crates/larql-compute-cuda/src/lib.rs` — fixed `q4_input_format_routes_like_cpu` test: it used 64 weight elements but `quantize_q4_k` requires a multiple of 256; bumped to `cols=128, rows=2` (256 elements).
- `crates/larql-inference/src/lib.rs` — fixed `unavailable_explicit_backend_errors_loudly` test: it used `expect_err` on `Result<Box<dyn ComputeBackend>, _>`, but `dyn ComputeBackend` isn't `Debug`, so the test didn't compile. Switched to a `match` on `Err`.

### CLI migration

Shared parser/helpers in `crates/larql-cli/src/commands/backend.rs`:

- `parse_backend_kind`, `parse_backend_list`
- `backend_kind_from_args` (honors `--metal` alias)
- `backend_kinds_from_args` (honors `--cpu`/`--metal` aliases)
- `primary_backend_kind`, `backend_label`
- `compute_backend_or_err`, `engine_backend_or_err`

Commands migrated to generic `--backend`/`--backends`:

- `bench` — parses generic backend names
- `run` — `--backend <auto|cpu|metal|cuda|vulkan>`, `--metal` alias preserved
- `walk` — `--backend`, `--metal` alias preserved
- `shannon encode` / `shannon decode` — **migrated in Session 2**: `--backend` added, `--metal` alias preserved, routed through the shared helper. Replaced the bespoke `metal_backend_box()` with `shannon_backend_box()` → `compute_backend_or_err(kind)`. The hard `--metal` requirement is now a backend-agnostic "selected backend must advertise fused Q4K decode" check.

Files touched (Session 1 + Session 2):

- `crates/larql-cli/src/commands/primary/bench/args.rs`
- `crates/larql-cli/src/commands/primary/bench/local.rs`
- `crates/larql-cli/src/commands/primary/bench/local_runtime.rs`
- `crates/larql-cli/src/commands/primary/bench/engine_runtime.rs`
- `crates/larql-cli/src/commands/primary/bench/remote_ffn_runtime.rs`
- `crates/larql-cli/src/commands/primary/bench/run.rs`
- `crates/larql-cli/src/commands/primary/run_cmd.rs`
- `crates/larql-cli/src/commands/extraction/walk_cmd.rs`
- `crates/larql-cli/src/commands/primary/shannon_cmd.rs` (Session 2)
- `crates/larql-cli/src/main.rs`

## Honest Status Of CUDA/Vulkan MVP

Not implemented yet (this is the Phase 4/5 work):

- `cudarc` / `ash` / `shaderc` dependencies
- NVRTC / SPIR-V kernel compilation
- real `prefill_kquant`
- real `decode_token`
- real `decode_token_with_state_dump_masked`
- real KV cache lifecycle on device (`has_kv_cache`, `reset_kv_cache`, `kv_cache_len`, `truncate_kv_cache`, `preallocate_kv_cache_per_layer`)
- real `f32_gemv` / `f16_gemv` / `q4k_*` / `q6k_*` device kernels
- real coarse `KvDispatch` (currently delegates to CPU)
- real `AsyncComputeBackend` batching (currently delegates to CPU)
- hardware-specific CI jobs for CUDA and Vulkan

What exists now is the repo-wide control plane needed to start that work without inventing it later.

## Remaining Work (in suggested order)

1. ~~compile + repair~~ DONE
2. ~~CLI migration: `run`/`walk`/`bench`/`shannon` accept `--backend`~~ DONE. Remaining polish:
   - sweep for stale Metal-only help text/comments across the migrated commands
   - `walk` Q4 predict/generate now constructs backends generically; `run` still has Metal-specific construction in remote FFN/MoE and `--experts` branches.
3. ~~tighten capability reporting in CUDA/Vulkan scaffolds per the plan's Phase 7~~ DONE in Session 3. Delegated scaffold methods still exist for parity tests, but `supports(...)` and `supports_quant(...)` report false until native kernels land.
4. replace delegated CUDA hot paths with real kernels (Phase 4):
   - `f32_gemv_topk1`
   - `f16_gemv_topk1`
   - `q4k_matvec`
   - `q4k_matmul`
   - `q6k_matvec`
   - `prefill_kquant`
   - `decode_token`
   - `decode_token_with_state_dump_masked`
   - KV cache lifecycle
   - coarse `KvDispatch` bridge
5. then do the same for Vulkan (Phase 5), keeping structure parallel to CUDA and Metal
6. integration tests mirroring Metal decode integration style (prefill shape, KV length, masked state dump `Full`/`HOnly`/`None`, heterogeneous per-layer KV preallocation, coarse `KvDispatch` prefill/decode with Q4K fixtures)
7. hardware-specific CI jobs for CUDA and Vulkan

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

Phases 1-3 are done and verified green (workspace compiles, clippy clean, ~1499 tests pass across the affected crates).

Phases 4-5 and 8 (real CUDA/Vulkan kernels and hardware CI) are not started — the scaffolds still delegate to CPU. Phase 7's first honesty pass is done for the scaffolds.

The next session should start at Phase 4: wire `cudarc` into `larql-compute-cuda` and replace the delegated hot paths with real device kernels, beginning with the Q4K decode bench path.

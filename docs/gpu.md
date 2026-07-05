# GPU acceleration (CUDA / Vulkan)

This document is the operator-facing reference for GPU acceleration in
LARQL. It covers how to build a GPU-enabled binary, what runtime
dependencies it expects, how CPU fallback works, how to confirm a native
runtime is actually active, and exactly which operations a backend
advertises today.

Everything here reflects the **current** code surface — feature flags,
capability values, and which ops are native vs CPU-delegated. When the
kernel surface changes, this doc must change with it. Cross-checks:

- `Capability` enum: `crates/larql-compute/src/backend/capability.rs`
- CUDA `supports()` + `device_info()`: `crates/larql-compute-cuda/src/trait_impl.rs`
- Vulkan `supports()` + `device_info()`: `crates/larql-compute-vulkan/src/trait_impl.rs`
- CUDA `KvDispatch` (CPU-delegated): `crates/larql-compute-cuda/src/kv_dispatch_impl.rs`

## Feature flags

GPU backends are opt-in Cargo features defined on `crates/larql-cli/Cargo.toml`:

| Flag | Crates enabled | Meaning |
|------|----------------|---------|
| `metal` | `larql-compute-metal` (+ inference/kv/vindex `metal`) | Apple Metal backend (macOS). |
| `cuda` | `larql-compute-cuda` (+ inference/kv/vindex `cuda`) | NVIDIA CUDA backend. |
| `vulkan` | `larql-compute-vulkan` (+ inference/kv/vindex `vulkan`) | Vulkan backend (scaffold). |
| `gpu` | `metal` | Default CLI feature on macOS. Alias for `metal`. |
| `gpu-all` | `metal` + `cuda` + `vulkan` | Build every backend in one binary. |

Build commands:

```bash
# CUDA only
cargo build --release --features cuda -p larql-cli

# CUDA + Vulkan + Metal in one binary
cargo build --release --features gpu-all -p larql-cli

# Default (Metal on macOS)
cargo build --release
```

The CI compile gate (GPU-0002) runs `cargo check --features cuda` and
`cargo check --features vulkan` on every PR, so a feature-flag break is
caught before merge.

## Runtime dependencies (CUDA)

The CUDA backend does not statically link the CUDA toolkit. It loads the
driver + NVRTC at runtime via `cudarc`'s `fallback-dynamic-loading`. The
crate pins `cudarc = "0.19.8"` with features `["std", "driver", "nvrtc",
"fallback-dynamic-loading", "cuda-11040"]`
(`crates/larql-compute-cuda/Cargo.toml`).

- **`cuda-11040` target** — the kernel PTX is compiled against the
  CUDA 11.4 toolkit headers. The build does not require a system CUDA
  11.4 install (NVRTC ships in the cudarc build), but the installed
  driver must support that target's PTX.
- **NVIDIA driver** — `libcuda.so` (Linux) / `libcuda.dylib` (macOS) /
  `nvcuda.dll` (Windows) must be present and loadable. The driver must
  support the device's compute capability.
- **NVRTC** — `libnvrtc.so` / `libnvrtc.dylib` / `nvrtc64_*.dll` must be
  loadable. Kernels are JIT-compiled to PTX on first backend
  initialisation via `compile_ptx_with_opts`.
- **sm_ (compute capability)** — there is no hardcoded `sm_` floor. At
  init the backend queries `context.compute_capability()` and the device
  reports its `(cc_major, cc_minor)`, surfaced as `sm_{cc}{cc}` in
  `device_info()`. Every kernel must JIT-compile for the device's sm_
  tier; if the NVRTC compile fails for a given sm_, backend init fails
  and the backend falls back to CPU (see below).

There is **no system CUDA toolkit (nvcc / cudart) install requirement**
— only the driver and NVRTC shared libraries, loaded dynamically.

## Runtime dependencies (Vulkan)

Vulkan has **no** runtime dependencies today. The crate depends only on
`larql-compute` + `larql-models` + `ndarray` + `half`
(`crates/larql-compute-vulkan/Cargo.toml`) — no Vulkan SDK, no
`vulkan-loader`, no device. Every trait method delegates to the CPU
reference. The crate is a pure scaffold: it exists so the backend
selection wiring compiles, but nothing is accelerated.

## How CPU fallback works

Both backends are constructed through `*::with_options(BackendOptions)`
which defaults to `allow_cpu_delegate = true`,
`device_ordinal = 0` (`crates/larql-compute-cuda/src/options.rs`). The
construction sequence (CUDA):

1. `CudaRuntime::initialize(ordinal)` probes the device, queries
   compute capability, JIT-compiles the NVRTC module, and loads every
   kernel function.
2. If init **succeeds**, the backend holds `runtime: Some(...)` and is
   "native".
3. If init **fails** (no device, driver missing, NVRTC compile failure,
   panic during probe) and `allow_cpu_delegate` is true, the backend
   returns anyway with `runtime: None` and stores the failure reason in
   `runtime_status`. Every native method then short-circuits to the CPU
   reference path, and `supports()` returns `false` for everything.

So a binary built with `--features cuda` runs correctly on a machine
with no NVIDIA GPU — it just runs on CPU. The backend `name()` reports
`"cuda (cpu-delegate scaffold)"` when there is no runtime, and
`"cuda (native k-quant + gemv + host-orchestrated decode/prefill + cpu
fallback)"` when there is.

Vulkan is always in the scaffold state: its constructor never attempts a
device and `supports()` is unconditionally `false`.

## How to verify a native runtime is active

Three probe surfaces, from quickest to most thorough:

**1. `device_info()`** — `ComputeBackend::device_info()` returns
`runtime_summary().to_string()`. On a native CUDA runtime this is the
device summary string, e.g.:

```
CUDA device NVIDIA GeForce RTX 4090 (ordinal 0, sm_89); native
q4k_matvec/q6k_matvec/q4k_matmul/q6k_matmul/q4k_dual_matvec/f32_gemv/
f16_gemv/q4_matvec/q4_vecmat/kv_append/rms_norm/rms_norm_heads/
geglu_silu/geglu_gelu_tanh/activation_silu/activation_gelu_tanh/
residual_add/rope/decode_attention/prefill_attention loaded, remaining
ops use CPU fallback
```

If there is no runtime, `device_info()` returns the stored failure
reason (e.g. `"CUDA runtime unavailable; using CPU delegate scaffold"`).
The presence of `sm_` and `loaded` is the definitive native-runtime
signal.

**2. `name()`** — `"cuda (native ...)"` vs `"cuda (cpu-delegate
scaffold)"`.

**3. `supports(Capability)`** — see next section. Any `true` answer
implies a native CUDA runtime is present (the CUDA `supports()` gates
every `true` on `native_runtime_available()`).

From Rust:

```rust
use larql_compute::backend::{Capability, ComputeBackend};
use larql_compute_cuda::cuda_backend;

let backend = cuda_backend().expect("cuda backend (CPU-delegated ok)");
println!("{}", backend.device_info());          // sm_ + loaded kernels, or failure reason
println!("{}", backend.name());                  // native vs scaffold
println!("quant_matvec accelerated: {}", backend.supports(Capability::QuantMatVec));
```

## The `supports()` capability contract (today)

`Capability` is the "ask before you call" enum
(`crates/larql-compute/src/backend/capability.rs`, defined at the
`pub enum Capability {` line). A backend advertises what it can
accelerate via `ComputeBackend::supports(cap)`. The default trait impl
returns `false` for everything; backends override to enable.

### CUDA — `CudaBackend::supports()`

Gated on `native_runtime_available()`. With **no** native runtime it
returns `false` for every capability. With a native runtime, the
matching set is exactly:

| `Capability` | CUDA advertises? |
|---|---|
| `QuantMatVec` | **yes** |
| `DecodeToken` | **yes** |
| `PrefillQ4` | **yes** |
| every other variant | no |

Source — `crates/larql-compute-cuda/src/trait_impl.rs`, the `supports()`
override:

```rust
if !self.native_runtime_available() { return false; }
matches!(cap, Capability::QuantMatVec | Capability::DecodeToken | Capability::PrefillQ4)
```

The three `true` answers reflect what is actually native:

- **`QuantMatVec`** — Q4_K and Q6_K matvec/matmul/dual-matvec run on the
  device. The `supports_quant(QuantFormat)` probe additionally confirms
  `Q4_K | Q6_K` (and **not** `Q4_KF` — see the in-source comment on
  `supports_quant` for why).
- **`DecodeToken`** — KV-cached single-token decode runs through the
  host-orchestrated device pipeline.
- **`PrefillQ4`** — multi-position prefill with KV-cache population
  runs through the host-orchestrated device pipeline.

### Vulkan — `VulkanBackend::supports()`

**Always `false`**, for every capability. Source —
`crates/larql-compute-vulkan/src/trait_impl.rs`:

```rust
fn supports(&self, cap: Capability) -> bool { let _ = cap; false }
```

Vulkan is a pure CPU-delegate scaffold. `supports_quant` is also
unconditionally `false`. Do not treat `--features vulkan` as an
acceleration path.

## What is NOT accelerated

These operations route through the CPU reference even when a native CUDA
runtime is active. They are deliberate, not bugs:

- **`matmul` and `matmul_transb`** —
  `CudaBackend::matmul()` / `matmul_transb()` delegate to
  `CpuBackend` unconditionally
  (`crates/larql-compute-cuda/src/trait_impl.rs`). Dense f32 GEMM is not
  on the device; callers that need the device path go through the
  quantised matvec/matmul kernels (`QuantMatVec`) instead.

- **Dense GEMV below ~500M flops** — `CudaBackend::f32_gemv()` returns
  `None` when `2 * n * k < GEMV_FLOP_THRESHOLD` (`GEMV_FLOP_THRESHOLD =
  500_000_000`), so the htod + kernel + dtoh round-trip is skipped in
  favour of the CPU loop. The same `None`-then-fallback shape applies
  to non-contiguous views and any native-launch failure. The ~500M flop
  threshold is the break-even point against CPU.

- **`KvDispatch` (CUDA)** — the entire `impl KvDispatch for CudaBackend`
  delegates to `CpuBackend`. Every method — `alloc_kv_buffer`,
  `append_kv`, `attention_step`, `attention_prefill`,
  `coarse_prefill`, `coarse_decode_step`, `compressed_kv_append`, and
  all the rest — forwards to the `const CPU: CpuBackend = CpuBackend;`
  delegate. See `crates/larql-compute-cuda/src/kv_dispatch_impl.rs` for
  the full surface. The KV cache itself *can* be device-resident (the
  native decode path uses `kv_cache`), but the `KvDispatch` trait
  surface is CPU-backed. The native `decode_attention` /
  `prefill_attention` kernels operate on the device-resident cache
  directly, not through `KvDispatch`.

- **`matmul` / `matmul_transb` on Vulkan** — CPU-delegated (and
  Vulkan's `f32_gemv` / `f16_gemv` are hand-rolled CPU loops, not
  device kernels).

In short: a native CUDA runtime accelerates **quantised matvec/matmul,
single-token decode, and multi-position prefill**. Everything else —
dense GEMM, small GEMV, the `KvDispatch` trait surface — is CPU.

## Selecting a backend at runtime

Backend selection lives in `larql-inference`
(`ComputeBackendKind::{Auto, Cpu, Metal, Cuda, Vulkan}`). When `cuda` is
compiled in, `cuda_backend()` is constructed with default options
(`allow_cpu_delegate = true`, `device_ordinal = 0`), so a CUDA-enabled
binary on a GPU box initialises native and on a non-GPU box silently
CPU-delegates. There is no separate "is there a GPU?" flag to pass —
construction handles it. `--features vulkan` is the same shape but, as
noted, always CPU-delegates today.

## Summary checklist for an operator

1. Build: `cargo build --release --features cuda -p larql-cli`.
2. Ensure the NVIDIA driver + NVRTC libs are loadable on the target host.
3. Run any inference path that prints `device_info()` — look for
   `sm_` and `loaded` to confirm native.
4. Probe `supports(Capability::QuantMatVec)` — `true` means native CUDA
   is active and the quant matvec/matmul kernels are on the device.
5. Vulkan is CPU-only today; `--features vulkan` is for wiring/compile
   coverage, not throughput.

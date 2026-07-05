# Vulkan MVP scope & SPIR-V strategy

**Status:** Decision (SPIKE — docs/design only, no kernel code).
**Task:** GPU-4001.
**Scope:** `crates/larql-compute-vulkan`, `docs/`.
**Base branch:** `feat/cuda-native-activation-kernels`.

## TL;DR

1. **MVP op subset = device-resident Q4_K matmul chain + RMSNorm + RoPE +
   residual add only.** Attention, MoE, and GEMV stay on CPU. This is
   strictly *smaller* than CUDA's current native surface
   (`QuantMatVec + DecodeToken + DecodeMoe + PrefillQ4`).
2. **SPIR-V strategy = build-time compile via `shaderc` with `.spv` checked
   into the repo.** No runtime compilation, no Vulkan SDK requirement on
   the target machine.
3. **Entry surface = `DecodeBackend` host-orchestrated** (the same surface
   CUDA's working path uses), **NOT** `KvDispatch`. Vulkan will not
   ship a native `KvDispatch` impl in the MVP. See §8.
4. **Capability contract = advertise only `QuantMatVec`** (Q4_K only) via
   `supports()` / `supports_quant()`. Nothing else is `true`.
5. **Vulkan dep = `ash`** (explicit Vulkan loader binding), chosen for parity
   with CUDA's low-level `cudarc` control. `vulkano` / `wgpu` rejected.
6. **Validation layers on in dev/test; off in release builds.**
7. **Attention and MoE are explicitly deferred** from the MVP (see §7).

## 1. Current state of `larql-compute-vulkan`

The crate is a pure CPU-delegate stub today:

- `trait_impl.rs` — every `MatMul` / `QuantMatVec` / `DecodeBackend` /
  `ComputeBackend` method forwards to `const CPU: CpuBackend = CpuBackend;`.
- `kv_dispatch_impl.rs` — the entire `impl KvDispatch for VulkanBackend`
  forwards to the same `const CPU` delegate (this mirrors the CUDA gap
  documented in `docs/backend-entry-surface.md`).
- `supports()` returns `false` for every `Capability`; `supports_quant()`
  returns `false` for every `QuantFormat`.
- `Cargo.toml` has **no** Vulkan dependency — only `larql-compute`,
  `larql-models`, `ndarray`, `half`.

So `VulkanBackend` compiles and produces correct results identical to
`CpuBackend`, but accelerates nothing. The MVP must move *some* of that to a
device without breaking the CPU-delegate fallback that lets the crate run on
a GPU-less host.

## 2. MVP op subset

The MVP accelerates exactly this set on the device:

| Op | Why it's in the MVP | Source |
|---|---|---|
| **`q4k_matmul`** (device-resident, seq>1) | Dominant cost of the Q4_K projection chain; reuses the 144-byte super-block layout CUDA already kernels | `QuantMatVec::q4k_matmul` |
| **`q4k_matvec`** (single-token decode path) | The seq=1 entry of the same Q4_K chain | `QuantMatVec::q4k_matvec` |
| **`rms_norm`** (post-embedding + per-layer) | Cheap, bounded, sequence-wide; the first non-matmul on the decode path | `DecodeBackend`-internal |
| **`rope`** (rotary embedding per layer) | Bounded, sequence-wide; required to produce a usable Q/K | `DecodeBackend`-internal |
| **`residual_add`** (after attention + after FFN) | Trivial elementwise; completes the chain | `DecodeBackend`-internal |

Everything else stays on CPU in the MVP:

| Op | Why it's deferred |
|---|---|
| **Attention (`decode_attention` / `prefill_attention`)** | see §7 — explicitly deferred |
| **MoE dispatch (`DecodeMoe`, expert routing)** | see §7 — explicitly deferred |
| **GEMV (`f32_gemv` / `f16_gemv` / topk variants)** | lm_head logits; CPU mmap zero-copy is competitive at decode batch=1 |
| **`q6k_matvec` / `q6k_matmul` / `q6k_dual_matvec`** | out of the Q4_K-only MVP |
| **`q4_matvec` / `q4_vecmat` (legacy Q4_0)** | out of the Q4_K-only MVP |
| **Dense `matmul` / `matmul_transb`** | never accelerated on CUDA either; stays CPU |

### 2.1 Why this subset is the right MVP

It is the **device-resident Q4_K matmul chain + the elementwise glue
(RMSNorm / RoPE / residual) needed to feed it**. That chain is:

```
embed → [rms_norm → q4k_matmul(gate) → q4k_matmul(up)] →
        ... attention (CPU) ...
        → [rms_norm → q4k_matmul(down)] → residual_add → ...
```

The attention block in the middle stays CPU. The Q4_K projections and their
surrounding elementwise ops move to the device. This is the smallest
device-resident unit that exercises: device memory allocation, quantised
kernel dispatch, host orchestration of a multi-op sequence, and a real
readback contract — without taking on attention's KV-cache lifecycle or
MoE's routing/composition.

### 2.2 Why it is strictly smaller than CUDA's surface

CUDA today (native) advertises `QuantMatVec | DecodeToken | DecodeMoe |
PrefillQ4` with `supports_quant` covering both `Q4_K` *and* `Q6_K`, and runs
native `decode_attention` / `prefill_attention` over a device-resident KV
cache. The Vulkan MVP advertises only `QuantMatVec` (Q4_K only) and runs
**no** native attention. So the MVP is a strict subset on every axis:

| Axis | CUDA (today) | Vulkan MVP |
|---|---|---|
| Capabilities advertised | `QuantMatVec, DecodeToken, DecodeMoe, PrefillQ4` | `QuantMatVec` only |
| Quant formats accelerated | `Q4_K, Q6_K` | `Q4_K` only |
| Native attention | yes (`decode_attention`, `prefill_attention`) | **no** |
| Device KV cache | yes | **no** |
| Native full decode/prefill pipeline | yes (`DecodeToken`, `PrefillQ4`) | **no** (CPU fallback) |

The MVP does not claim `DecodeToken` or `PrefillQ4` because those require
native attention, which is deferred. Until attention lands, the
host-orchestrated `decode_token` / `prefill_kquant` pipeline cannot be made
native on Vulkan — so it stays on the CPU-delegate path and the capabilities
stay `false`.

## 3. SPIR-V strategy — build-time compile via `shaderc`

**Decision: SPIR-V is compiled at *build* time, and the resulting `.spv`
byte arrays are checked into the repo (as `crates/larql-compute-vulkan/spv/*.spv`
and embedded via `include_bytes!`).**

| Option | Verdict |
|---|---|
| **Build-time compile with `shaderc`, `.spv` checked in** | **CHOSEN** |
| Runtime compile via `shaderc` at backend init | rejected — adds a build-tool runtime dependency |
| `vulkan-sdk` shader compiler at runtime | rejected — same |

### 3.1 Why build-time + checked-in `.spv`

- **No Vulkan SDK requirement on the target / build host for the *runtime*.**
  The `.spv` files are compiled once during development and committed. A CI
  runner or deploy box needs only the Vulkan loader (`libvulkan.so.1`) and a
  device driver — not `glslangValidator` / `glslc` / the SDK.
- **Deterministic bytecode.** Checked-in `.spv` is byte-stable; there is no
  "which shader compiler version produced this kernel" question at runtime.
- **Cheap `cargo check`.** Because the SPIR-V is embedded data, not a build
  step, `cargo check --features vulkan` does not invoke a shader compiler.
  This matches how the CUDA compile gate works (NVRTC at runtime, not build
  time) and keeps the existing CI compile gate cheap.

### 3.2 The `shaderc` build dependency

A `build.rs` (added in the *implementation* tasks, not here) will:

1. Glob `crates/larql-compute-vulkan/shaders/*.comp` (GLSL).
2. Compile each to SPIR-V via `shaderc`'s Rust crate
   (`shaderc = "0.8"`, `Compiler` + `CompileOptions`).
3. Write the `.spv` to `crates/larql-compute-vulkan/spv/*.spv`.
4. The compiled output is **committed**; the `build.rs` is a *rebuild*
   convenience for shader authors, not a required CI step. CI compiles from
   the checked-in `.spv`.

`shaderc-rs` vendors `glslang` so the build does not need a system
`glslangValidator`. The `shaderc` crate is a **build-dependency only**; it is
not linked into the runtime binary.

### 3.3 Rebuild contract

- Shader authors run `cargo build -p larql-compute-vulkan` to regenerate
  `.spv` from edited `.comp` sources, then commit the updated `.spv`.
- CI does **not** run the shader compile; it embeds the committed `.spv`.
  A `scripts/check-spv-fresh.sh` (future task) can diff regenerated `.spv`
  against committed `.spv` to catch uncommitted shader changes, but that is
  out of scope for this spike.

## 4. Entry surface — `DecodeBackend`, host-orchestrated

**Decision: the MVP accelerates ops only through the `DecodeBackend`
host-orchestrated pipeline. Vulkan does NOT ship a native `KvDispatch` impl in
the MVP.**

This matches CUDA's working path: CUDA's only native surface today is
`DecodeBackend` (`decode_token` / `prefill_kquant` driving the device-resident
KV cache), and the entire `impl KvDispatch for CudaBackend` is a CPU delegate
(see `docs/backend-entry-surface.md` §1).

### 4.1 What "host-orchestrated" means here

The CPU owns the decode loop. For each token:

1. CPU uploads the residual / hidden state to a device buffer.
2. CPU dispatches `rms_norm` → `q4k_matmul(gate)` → `q4k_matmul(up)` as
   separate Vulkan command buffers, recording barriers between them.
3. CPU reads back the gate/up results (or keeps them device-resident for the
   attention step, which is CPU in the MVP — so a readback happens at the
   attention boundary).
4. CPU runs attention on host (MVP fallback).
5. CPU re-uploads, dispatches `rms_norm` → `q4k_matmul(down)` → `residual_add`.
6. Read back the next-layer residual.

This is exactly the shape of CUDA's `host_decode_token` /
`host_prefill_kquant` (the functions CUDA's `DecodeBackend` impl delegates
to). The MVP ports that host orchestration to Vulkan command buffers; it does
not invent a new orchestration model.

### 4.2 Why NOT `KvDispatch`

`KvDispatch` is the *engine-facing* trait (`larql-kv` engines call
`coarse_prefill_with_state` / `coarse_decode_step_with_state`, etc.). On CUDA
it is a pure CPU delegate while the *useful* native path is `DecodeBackend`.
Making `KvDispatch` native would mean building a **second** device KV-cache
+ attention surface parallel to `DecodeBackend` — the exact duplication
`docs/backend-entry-surface.md` rejected for CUDA (option (a), "decisive
con" = duplicate device KV layout + duplicate readback contract).

Vulkan avoids creating that second surface from day one. See §8 for the
decision record.

## 5. Capability contract — advertise only what is implemented

`VulkanBackend::supports()` and `supports_quant()` must remain **honestly
gated**. The MVP's contract:

```rust
// crates/larql-compute-vulkan/src/trait_impl.rs (MVP target shape)
impl QuantMatVec for VulkanBackend {
    fn supports_quant(&self, format: QuantFormat) -> bool {
        self.native_runtime_available()
            && matches!(format, QuantFormat::Q4_K)
    }
    // q4k_matmul / q4k_matvec: device path when native + format gate passes,
    // else CPU delegate (unchanged from today).
    // q6k_* / q4_* : unchanged CPU delegate, supports_quant already says no.
}

impl ComputeBackend for VulkanBackend {
    fn supports(&self, cap: Capability) -> bool {
        if !self.native_runtime_available() { return false; }
        matches!(cap, Capability::QuantMatVec)
    }
}
```

Rules:

1. **No capability is `true` without a native runtime** (mirrors CUDA's
   `native_runtime_available()` gate). The CPU-delegate scaffold path keeps
   `supports()` / `supports_quant()` returning `false`, exactly as today.
2. **Only `QuantMatVec` is advertised**, and `supports_quant` narrows it to
   `Q4_K` only. `Q6_K`, `Q4_KF`, `Q4_0` all report `false`.
3. **`DecodeToken`, `PrefillQ4`, `DecodeMoe` stay `false`** because they
   require native attention, which is deferred (§7). The
   host-orchestrated `decode_token` / `prefill_kquant` pipeline stays on the
   CPU-delegate path until attention lands.
4. **Every native method keeps its CPU fallback**: if a native launch fails
   (no device, allocation failure, etc.), the method falls through to the
   existing `CPU` delegate. `supports()` going `true` is a promise that the
   *fast* path exists; the *correct* path always exists.

### 5.1 `device_info()` / `name()` shape

- `name()`: `"vulkan (cpu-delegate scaffold)"` when no runtime (unchanged),
  `"vulkan (native q4_k matmul chain; attention/moe CPU fallback)"` when native.
- `device_info()`: when native, report the physical device name + the loaded
  shader set (e.g. `"Vulkan device <name> (ordinal 0); native
  q4k_matmul/q4k_matvec/rms_norm/rope/residual_add loaded, attention/moe use
  CPU fallback"`). When no runtime, the failure reason (unchanged).

## 6. Vulkan dependency decision — `ash`

**Decision: depend on [`ash`](https://crates.io/crates/ash) (explicit Vulkan
loader binding), not `vulkano` or `wgpu`.**

| Crate | Layer | Verdict | Reason |
|---|---|---|---|
| **`ash`** | thin unsafe bindings over the Vulkan C API | **CHOSEN** | parity with CUDA's `cudarc` (low-level driver API, explicit command-buffer recording, explicit barriers). The MVP needs precise control over command-buffer recording, pipeline barriers between `rms_norm` → `q4k_matmul`, and host-coherent device memory — `ash` exposes exactly the Vulkan calls, nothing more. |
| `vulkano` | safe wrapper, owns a lot of state | rejected | hides the barrier / command-buffer seams the MVP must control to match CUDA's host orchestration; its abstractions impose a model that would have to be fought. |
| `wgpu` | portable abstraction over Vulkan/Metal/DX12/WebGPU | rejected | wrong layer — we already have a per-backend crate (`larql-compute-metal`, `-cuda`, `-vulkan`). `wgpu` would re-introduce the portability layer the per-backend split is designed to avoid, and its shader model (WGSL) conflicts with the checked-in-SPIR-V strategy in §3. |

### 6.1 Loader + device assumptions

- The Vulkan loader (`libvulkan.so.1` on Linux, `libvulkan.1.dylib` on macOS,
  `vulkan-1.dll` on Windows) must be present and loadable. `ash`'s
  `Entry::load()` resolves it via the standard platform mechanism.
- A **portable device** is assumed: any Vulkan 1.1+ physical device that
  exposes the `VK_KHR_storage_buffer_storage_class` (core in 1.1) and a
  compute queue. No vendor-specific extensions in the MVP.
- `ash` is loaded via `Entry::linked_default()` (statically linked loader
  entry points) OR `Entry::load()` (dynamically resolved). The MVP uses
  `Entry::load()` so a missing loader becomes a clean
  `BackendInitError::VulkanLoaderUnavailable` rather than a link failure —
  this preserves the "compiles on a GPU-less host, runs on CPU" property the
  crate has today.

### 6.2 Why not dynamic-loading like cudarc

CUDA uses `cudarc`'s `fallback-dynamic-loading` so the crate links
nothing at build time and resolves the driver at runtime. Vulkan's
equivalent is `ash`'s `Entry::load()` against the system loader. The
important property — **no hard link dependency, clean runtime fallback** — is
preserved. `ash`'s `linked` feature is *not* used; we depend only on the
runtime-resolved loader.

## 7. Explicitly deferred: attention and MoE

**Attention (`decode_attention`, `prefill_attention`) and MoE
(`DecodeMoe`, expert routing) are out of scope for the MVP.**

### 7.1 Attention

Native attention requires a device-resident KV cache with a lifecycle managed
across decode steps (`populate_kv_layer`, `kv_cache_len`, `truncate_kv_cache`,
`reset_kv_cache`). That is the largest single piece of device state in the
CUDA backend, and it is the piece that most entangles the backend with the
engine contract. Building it on Vulkan before the simpler Q4_K chain is
proven would repeat the CUDA ordering risk (attention landed before the
matmul chain was fully characterised).

In the MVP, the host-orchestrated decode/prefill pipeline calls attention on
**CPU** (the existing `CpuBackend` attention). The device handles only the
Q4_K projections and the surrounding elementwise ops. This means a readback
happens at the attention boundary each layer — acceptable for an MVP whose
goal is "prove the Vulkan dispatch path end-to-end", not "beat CPU latency".

**Consequence:** `DecodeToken` and `PrefillQ4` stay `false` (§5 rule 3) until
attention lands in a follow-up task.

### 7.2 MoE

MoE (expert routing + per-expert matmul + composition) is deferred for the
same reason it is partially deferred on CUDA: it is a *composition* problem
(routing tensor on device, expert tensors, reduction) that builds on top of a
working Q4_K matmul + attention surface. There is no point MoE-accelerating
before the underlying matmul and attention paths are native. `DecodeMoe` stays
`false`.

### 7.3 Follow-up task shape

The natural follow-ups (filed separately, not in this spike):

- **GPU-4002** — device-resident KV cache for Vulkan (`ash` `VkBuffer` per
  layer, the `populate/truncate/reset` surface), unblocking native attention.
- **GPU-4003** — native `decode_attention` / `prefill_attention` SPIR-V,
  flipping `DecodeToken` / `PrefillQ4` to `true`.
- **GPU-4004** — Q6_K matmul/matvec on Vulkan (widens `supports_quant`).
- **GPU-4005** — MoE dispatch on Vulkan (`DecodeMoe`).

## 8. Decision record — why Vulkan does NOT copy CUDA's KvDispatch-delegate gap

### 8.1 The gap

On CUDA, `impl KvDispatch for CudaBackend`
(`crates/larql-compute-cuda/src/kv_dispatch_impl.rs`) is a **pure CPU
delegate**: every method forwards to `const CPU: CpuBackend = CpuBackend;`.
Meanwhile the *useful* native CUDA surface is `DecodeBackend`
(`decode_token` / `prefill_kquant` driving a device-resident KV cache). The
consequence, documented in `docs/backend-entry-surface.md` §1, is that every
`larql-kv` engine that drives compute through `KvDispatch` gets **zero** GPU
benefit on CUDA, even though the kernels it conceptually needs exist on the
`DecodeBackend` side. Closing that gap is a medium-cost bridge task
(GPU-6002).

`larql-compute-vulkan` today has the **same** shape: `kv_dispatch_impl.rs`
delegates everything to `const CPU`, while `trait_impl.rs` (the
`DecodeBackend`/`ComputeBackend` surface) is also all-CPU but is the
intended native surface.

### 8.2 The decision

**Vulkan will not grow a native `KvDispatch` impl.** The native surface is
`DecodeBackend` only. `impl KvDispatch for VulkanBackend` remains a CPU
delegate permanently — or, if a bridge becomes valuable later, it follows the
CUDA GPU-6002 pattern (delegate the coarse methods *internally* to the
`DecodeBackend` pipeline), never a parallel device surface.

### 8.3 Why

1. **Avoid the second-surface trap.** A native `KvDispatch` requires a
   device-resident `KvHandle`, which means a *second* device KV-cache layout
   + readback contract alongside whatever `DecodeBackend` owns. CUDA fell
   into this (two surfaces, one native, one CPU-delegate) and the recovery is
   a bridge task. Vulkan can skip the trap entirely by never starting the
   second surface.
2. **`DecodeBackend` is the proven useful path.** On CUDA, every documented
   GPU win (`QuantMatVec`, `DecodeToken`, `PrefillQ4`) flows through
   `DecodeBackend`, not `KvDispatch`. Starting Vulkan on `DecodeBackend`
   means starting on the path that has already paid off.
3. **`KvDispatch`'s `KvHandle` is a host-owned type today.** Making it
   device-resident is a cross-cutting type change that touches every engine's
   `dispatch.rs` (`docs/backend-entry-surface.md` §3, "Is a device-resident
   `KvHandle` needed? — No"). Vulkan gains nothing by re-litigating that.
4. **Engines that need the bridge can get it the same way CUDA will.** If a
   residual/checkpoint engine later wants GPU benefit on Vulkan, the fix is
   the GPU-6002-style internal delegation (coarse → `DecodeBackend`), not a
   new kernel surface. That keeps the engine-facing trait intact and the
   device KV cache owned by one place.

### 8.4 Scope invariant

This decision is what keeps the Vulkan MVP **strictly smaller** than CUDA's
current surface (§2.2). Because Vulkan does not claim `DecodeToken` /
`PrefillQ4` / `DecodeMoe` (those need native attention, deferred) and does
not native-implement `KvDispatch`, its advertised surface is just
`QuantMatVec` (Q4_K). CUDA advertises four capabilities and two quant
formats. The MVP is a proper subset.

## 9. Validation layers & portable-device assumptions

### 9.1 Validation layers

- **Dev / test builds:** enable `VK_LAYER_KHRONOS_validation` via
  `BackendOptions` (the `validation: bool` field to be added in the
  implementation task). Backend init logs any validation messages. Tests in
  `crates/larql-compute-vulkan` run with validation on by default when a
  device is present.
- **Release / production builds:** validation **off**. The layer costs
  non-trivial overhead per command. `BackendOptions::default()` sets
  `validation: false`.
- **No validation dependency at link time.** The layers are loaded by the
  Vulkan loader at runtime; if `VK_LAYER_KHRONOS_validation` is not
  installed, backend init with `validation: true` logs a warning and
  continues without validation rather than failing.

### 9.2 Portable-device assumptions (MVP)

- Vulkan **1.1+** (for `VK_KHR_storage_buffer_storage_class` core + subgroup
  ops). Query `api_version` at init; reject below 1.1 with a clear
  `BackendInitError`.
- A **compute queue** (`VK_QUEUE_COMPUTE_BIT`). Most discrete GPUs and all
  recent integrated GPUs expose one.
- **No vendor-specific extensions** in the MVP. No `VK_KHR_cooperative_matrix`,
  no `VK_EXT_subgroup_size_control`, no `VK_NV_*` / `VK_AMD_*`. The Q4_K
  super-block decode is written in portable GLSL (subgroup shuffle within a
  workgroup). Vendor-tuned variants are a follow-up.
- **Host-visible device memory for staging** is assumed available (it is on
  every conformant Vulkan 1.1 implementation). The MVP stages weights/inputs
  through a host-visible buffer and device-local buffers for the hot path;
  this matches how the CUDA backend stages q4k weights.

### 9.3 What the MVP does NOT assume

- No unified memory / no `VK_KHR_external_memory_fd`. The MVP does explicit
  staging; it does not rely on shared host/device memory.
- No swapchain, no graphics, no presentation. Compute-only.
- No multi-device. `device_ordinal` is accepted (for API parity with CUDA)
  but the MVP uses ordinal 0 only.

## 10. Out of scope for this spike

This document is **docs/design only** (GPU-4001). It produces no kernel code,
no `build.rs`, no `.comp` shaders, no `.spv`, no `ash` dependency in
`Cargo.toml`. Those land in the implementation tasks (GPU-4002+) that this
scope gates.

Concrete artefacts this spike does NOT touch:

- `crates/larql-compute-vulkan/Cargo.toml` (no `ash` / `shaderc` added here).
- `crates/larql-compute-vulkan/src/trait_impl.rs` (no `supports()` change).
- Any `shaders/*.comp` or `spv/*.spv`.
- Any `build.rs`.

The only artefact this spike produces is this file, `docs/vulkan-mvp.md`.

## 11. Evidence

- `crates/larql-compute-vulkan/src/trait_impl.rs` — pure CPU delegate today;
  `supports()` unconditionally `false`.
- `crates/larql-compute-vulkan/src/kv_dispatch_impl.rs` — pure CPU delegate
  (`const CPU: CpuBackend = CpuBackend;`), same gap shape as CUDA.
- `crates/larql-compute-vulkan/Cargo.toml` — no Vulkan dependency.
- `crates/larql-compute-cuda/src/trait_impl.rs:520` — CUDA `supports()`:
  `QuantMatVec | DecodeToken | DecodeMoe | PrefillQ4` (native-gated).
- `crates/larql-compute-cuda/src/trait_impl.rs:216` — CUDA
  `supports_quant()`: `Q4_K | Q6_K` (native-gated).
- `crates/larql-compute/src/backend/capability.rs` — `Capability` enum
  definition (the contract `supports()` must honour).
- `docs/backend-entry-surface.md` — the CUDA `KvDispatch` vs `DecodeBackend`
  gap analysis this spike's §8 is based on.
- `docs/gpu.md` — "What is NOT accelerated" / capability contract sections
  this MVP's narrower contract must stay consistent with.

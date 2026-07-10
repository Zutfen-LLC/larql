# CUDA + Vulkan Completion Plan

Status date: 2026-07-09 (updated after GPU-004 hardware validation).
Companion documents:

- `CUDA_VULKAN_IMPLEMENTATION_PLAN.md` — the original end-to-end target (Phases 1-8)
- `HANDOFF.md` — per-session log of what landed (Sessions 1-25)
- `CROSS_PLATFORM_QUICK_WINS.md` — low-effort improvements across CPU/Metal/CUDA/Vulkan (sibling to this doc)

This document supersedes the "Remaining Work" sections of both older docs as the
forward-looking plan. It is based on a fresh code vetting (below), not just the
session logs.

---

## Part 1 — Vetting: where the code actually is

The claims in `HANDOFF.md` were checked against the code on this branch. They are
**accurate**. Summary of the verified state:

### Done and real

| Area | State |
|---|---|
| Phase 1 (workspace + features) | DONE. `larql-compute-cuda` / `larql-compute-vulkan` are workspace members; `metal`/`cuda`/`vulkan`/`gpu-all` features compile in all subsets; `gpu` aliases Metal. |
| Phase 2 (backend selection) | DONE. `ComputeBackendKind`, `compute_backend()`/`engine_backend()`/`async_engine_backend()` factories, per-platform `Auto` order (macOS: metal→vulkan→cpu; elsewhere: cuda→vulkan→cpu), loud errors on explicit-unavailable (`larql-inference/src/lib.rs:92-410`). |
| Phase 3 (CLI) | DONE for `run`/`walk`/`bench`/`shannon` via `larql-cli/src/commands/backend.rs` (`--backend`, `--backends`, `--metal` alias). Residual polish: stale Metal-only help text; `run`'s remote-FFN/MoE and `--experts` branches still construct Metal specifically. |
| Phase 4 (CUDA MVP) | Far beyond MVP. Twenty native NVRTC kernels (5 k-quant, 2 dense GEMV, 2 legacy Q4_0, `kv_append`, 2 RMSNorm, 4 activations, residual add, RoPE, fused decode attention, fused prefill attention). Fused `prefill_kquant`/`decode_token`/`decode_token_with_state_dump_masked` pipelines live (`pipeline.rs`, ~3000 lines), including hybrid-MoE (dense slab + device expert matvecs + device-chained per-expert FFN). Persistent weight cache (upload-once, `(ptr,len)`-keyed, flushed at `reset_kv_cache` + explicit `flush_weight_cache()`). Device-resident activation chains for all four blocks (prefill/decode × FFN/attention) with single end-of-chain readback. Device KV cache + full `DecodeBackend` lifecycle. Capability advertisement is honest: `supports(...)`/`supports_quant(...)` true only with a live runtime, Q4_K/Q6_K only, `DecodeMoe` not advertised (hybrid-MoE `auto` routes to CPU at the engine layer). |
| Phase 7 (capability honesty) | Held throughout. Scaffold (no device) reports everything `false`; delegated CPU methods stay callable for parity tests. |
| Test discipline | ~140 tests in the CUDA crate (141 including the hardware probe integration test). Every kernel has a host-runnable CPU-fallback contract test plus runtime-gated native parity, overflow-rejection, and shape-rejection tests. **Validated on real hardware (GPU-004):** 139/141 pass on RTX 3090; the 2 failures are a real decode attention parity bug (max_abs=0.13, deferred). A hardware probe test (`tests/hardware_probe.rs`) asserts the native runtime is active, preventing silent scaffold regressions. |

### Not done

| Area | State |
|---|---|
| **Hardware validation** | **DONE (GPU-004, 2026-07-08).** First real-hardware run on RTX 3090 (sm_86, CUDA 12.4). Three NVRTC compilation blockers fixed (missing `cuda_fp16.h` include path, duplicate symbol definitions from concatenated kernel sources, undefined `INFINITY` macro). After fixes: 139/141 tests green with native CUDA kernels executing. Two failures are a real decode attention parity bug (max_abs=0.13, deterministic — deferred to a stabilization slice). CLI backend selection, PTX cache cold/hot, and end-to-end inference pipeline all verified on hardware. Full report: `bench/baselines/cuda-hardware-validation-2026-07-08.md`. |
| Phase 5 (Vulkan) | Not started. `larql-compute-vulkan` is a 681-line pure scaffold: no `ash`, no `shaderc`, no SPIR-V; every trait method delegates to `CpuBackend`; all capabilities report `false`. |
| Coarse `KvDispatch` (CUDA) | 100% CPU delegation (`kv_dispatch_impl.rs`) — the walk/browse/engine coarse paths never touch the GPU. |
| `AsyncComputeBackend` (CUDA) | 100% CPU delegation (`async_compute_backend_impl.rs`). Honest (deferred capabilities report unsupported) but unimplemented. |
| Decode KV round-trip | The decode attention chain still reads back the new K/V row and re-uploads the **full** `[len, kv_dim]` KV from the host mirror every token — O(context) PCIe traffic per token. The device `CudaKVCache` exists but the decode chain doesn't attend over it yet. This will dominate decode latency on real hardware. |
| Cross-layer residency | The hidden state `h` returns to host between blocks/layers; each block pays one upload + one readback. |
| PLE / remote-FFN in the fused pipeline | Both bail to CPU (PLE needs data not on the trait surface; remote-FFN needs the dispatch callback only `decode_token_with_moe` carries). |
| Hardware CI | CUDA/Vulkan workflows are compile-only (correct and useful, but no runtime coverage). |
| `larql-python` | Pre-existing: fails on PyO3 0.24 vs Python 3.14 (not caused by this work). |

### Vetting concerns to carry into the plan

1. **~~Untested-on-hardware surface is very deep.~~** — **Validated (GPU-004).**
   Twenty kernels, a 3k-line pipeline, a weight cache, and four device-resident
   chains were written against cudarc's API without a single real launch. The
   real-hardware run (RTX 3090, CUDA 12.4) confirmed the prediction: three NVRTC
   compilation blockers (missing include path, duplicate symbols, undefined
   `INFINITY`) prevented any kernel from ever launching — every "passing" test
   was silently on the CPU scaffold. After fixes, 139/141 tests pass with native
   kernels executing. Two failures are a real decode attention parity bug
   (max_abs=0.13, deterministic). **Lesson:** the fallback contract (degrade to
   CPU silently) is good for correctness but bad for noticing — the hardware
   probe test (`tests/hardware_probe.rs`) now guards against this.
2. **The host KV mirror is load-bearing.** `kv_cache_len`, RoPE position, and
   host attention all read the mirror; the device cache is lockstep-secondary.
   Fine as a bridge, but the resident-KV slice (Phase B1) inverts that
   relationship and must be done carefully (it touches position accounting).
3. **Numerics policy is consistent and documented** (fmad off, f64 softmax
   accumulation, libm-divergence tolerances ≤1e-4/1e-5) — keep it; do not
   trade it away for speed until a parity-vs-fast-math toggle exists.
4. **Vulkan estimates should assume CUDA-fallout learnings first.** Porting 20
   kernels to GLSL against a backend model (buffers, no NVRTC-style JIT of C)
   is a bigger unit of work than any single CUDA session was.

---

## Part 2 — The plan

Phases are ordered by risk retirement, not by the original doc's numbering.
"Session" = one focused working session of the size that produced Sessions 4-25.

### Phase A — CUDA hardware validation gate — ✅ COMPLETE (GPU-004)

**Completed 2026-07-08.** See `bench/baselines/cuda-hardware-validation-2026-07-08.md`
for the full report.

- A1. ✅ Provisioned RTX 3090 (sm_86, CUDA 12.4, driver 550.163.01).
- A2. ✅ Ran the full runtime-gated suite. Three NVRTC compilation blockers
  fixed (include path, duplicate symbols, `INFINITY`). 139/141 tests green
  with native kernels. Two failures: decode attention parity (max_abs=0.13,
  deterministic — deferred to a stabilization slice, not a regression).
- A3. ✅ CLI integration surface verified: `--backend cuda`, `LARQL_BACKEND=cuda`,
  `--backend cpu`, invalid-backend-loud-fail. PTX cache cold/hot verified.
  End-to-end `larql run --backend cuda` executes but produced garbage output
  due to the decode parity bug + a vindex vocab_size padding issue. The
  vocab padding issue is now **fixed (VINDEX-001)**; rebuild the vindex
  without hand-editing `index.json` and only the decode-parity issue remains.
- A4. Partial — CUDA is faster than CPU (~36s vs >300s for 8 tokens on debug
  build), but absolute numbers are unreliable until the decode parity bug is
  fixed. llama.cpp comparison skipped (not CUDA-built).
- **Deferred:** decode attention parity bug (max_abs=0.13), extraction
  pipeline serial bottlenecks (down_meta, clustering).
  (vindex vocab_size padding — **resolved by VINDEX-001**.)

### Phase A-stabilization — CUDA decode parity fix — ⏳ FIX APPLIED, PENDING HARDWARE VALIDATION (ASTAB-001)

**Fix applied 2026-07-09 (ASTAB-001).** See
`bench/baselines/cuda-decode-parity-fix-2026-07-09.md` for the full report.

- **Root cause:** NOT a kernel bug. The two failing decode parity tests
  compared CUDA's f32-activation decode pipeline (`host_decode_token` →
  `q4k_matvec` dequant-then-f32-dot, the same numerics CUDA's prefill
  pipeline and `predict_kquant_prefill` use) against the production CPU
  decode reference `predict_kquant_decode_step_direct`, which uses int8
  Q8_K SDOT matvec (`q4k_q8k_matvec_into`). CUDA has no SDOT instruction, so
  its decode intentionally uses f32-activation numerics (documented in
  `pipeline.rs` `moe_expert_contribution_q4k`). The int8-vs-f32 mismatch is
  ~2% scale-relative by design (pinned by
  `q8k_direct_proj_matches_f32_activation_within_quant_tolerance`), far
  above the 1e-3 parity tolerance — producing the deterministic 0.1314532
  divergence. The prefill parity test passed because both sides used
  f32-activation; the decode parity tests failed because the CPU reference
  used int8.
- **Fix:** The two decode parity tests now compare against
  `predict_kquant_decode_step` (f32-activation decode reference, the decode
  twin of `predict_kquant_prefill`) instead of
  `predict_kquant_decode_step_direct` (int8 production path). No tolerance
  was loosened; no CUDA path was routed to CPU. The decode-attention kernel
  itself was confirmed correct via 5 new focused native parity tests
  (ASTAB-001C: single-head, multi-head, GQA asymmetric, softcap
  multi-position, fixture-shape shrink).
- **Hardware validation:** PENDING. The fix was developed on a no-CUDA host
  (scaffold tests green: 145/145 lib tests pass, clippy clean, fmt clean).
  The PR must be run on real CUDA hardware (`cargo test -p larql-compute-cuda
  --features cuda`) before merge — per slice blocked_policy, do not claim
  complete without hardware validation.

### Phase B — CUDA perf completion (4-7 sessions, after A)

Ordered by expected wall-clock impact; re-rank against the A4 profile.

- B1. **Resident-KV decode attention** (1-2 sessions). Make the decode chain
  append the new K/V row to the device `CudaKVCache` (the `kv_append` kernel
  already exists) and attend over the device-resident KV, eliminating the
  per-token full-KV upload + row readback. The host mirror stays as the
  parity oracle and the source for `truncate`/state-dump paths. This is the
  single biggest remaining CUDA perf item.
  <!-- GPU-006 (2026-07-09): COMPLETE — decode now appends the current token's
       K/V to the device cache and attends over the resident K/V via
       launch_decode_attention_resident_dev, with an explicit full-KV-upload
       fallback when ineligible. 8 runtime-gated tests added (lockstep, parity
       vs CPU f32 reference, valid-rows-only, fallback). HARDWARE-VALIDATED on
       RTX 3090 (sm_86, NVRTC 12.4): 154/154 tests green with default settings
       (hardware_probe + ASTAB-001 decode parity + all 8 resident_kv), 21
       native kernels loaded. See bench/baselines/cuda-resident-kv-2026-07-09.md. -->
  <!-- GPU-007 (2026-07-10): HARDWARE-VALIDATED on RTX 3090 (sm_86, NVRTC 12.4) —
       decode now threads the hidden state device-resident through eligible
       dense layers via host_decode_token_resident (+ host_attention_block_
       device_resident + host_ffn_block_device_resident), collapsing the
       per-block hidden-state readback/upload boundaries the host-orchestrated
       loop pays. The input norm, post-attn norm+residual, and post-ffn
       norm+residual now run on device (launch_rms_norm_dev + the new
       launch_residual_add_dev — a thin wrapper over the existing
       launch_elementwise_binary_dev; no new kernel). 7 runtime-gated tests
       added (single-token parity, multi-token parity, diag-surface, forced
       fallback, KV-lockstep-unchanged, consecutive-layers, mixed-eligibility
       transitions). Fallback is per-layer (MoE/PLE/remote/LayerNorm/sub-gate/
       non-kquant/padded-down → host path) and counted (resident_hidden_uses/
       fallbacks under LARQL_GPU_DIAG=1). 160/160 lib tests green + 1
       hardware_probe on RTX 3090 with default settings (no env overrides).
       All 7 resident_hidden tests pass natively, including the
       mixed-eligibility transition test (Host→Device→Host→Device→final Host).
       Three-repeat stability confirmed. Release-mode focused tests pass.
       See bench/baselines/cuda-cross-layer-residency-2026-07-10.md.
       lm-head-on-device, launch batching, and MoE/router polish
       remain later slices. -->
- B2. **Cross-layer residency** ✅ **HARDWARE-VALIDATED (2026-07-10)**.
  Keep `h` resident across blocks and layers within a decode step / prefill
  pass; read back once per token (logits input) instead of once per block.
  The residual-add kernel already exists; this is plumbing, not kernels.
  Validated on RTX 3090 (sm_86, NVRTC 12.4): 161/161 tests green with default
  settings, all 7 resident_hidden tests pass natively, three-repeat stability
  confirmed, release-mode focused tests pass. See
  bench/baselines/cuda-cross-layer-residency-2026-07-10.md.
  <!-- LARQL-GPU-PROFILE-001 (2026-07-10, RTX 3060): the resident-hidden
       eligibility gate (pipeline.rs resident_hidden_layer_eligible) required a
       uniform Q4_K or Q6_K FFN (gate,up,down) triple. The default Q4_K_M mix
       (gate/up Q4_K, down Q6_K — the `convert quantize q4k` default and the
       Ollama-compatible format) failed this gate, so GPU-007 engaged 0% on a
       real Q4_K_M model (measured: 0 uses / 612 fallbacks). It engaged 100%
       only with `--down-q4k` (uniform Q4_K). The synthetic fixtures used for
       the GPU-007 validation were uniform Q4_K, so the code was validated but
       the production default format never reached it. See
       bench/baselines/cuda-post-residency-profile-2026-07-10.md.
       ✅ RESOLVED by D6 (2026-07-10, RTX 3060) — see below. -->
- ✅ **D6. resident-hidden Q4_K_M eligibility — DONE (2026-07-10, RTX 3060).**
  Broadened the GPU-007 resident-hidden eligibility gate and the resident FFN
  chain through one shared pure helper (`supported_resident_ffn_triple`) so
  they accept the production-default Q4_K_M FFN triple (gate/up Q4_K, down
  Q6_K) in addition to uniform Q4_K×3 and Q6_K×3. No CUDA kernel changes
  needed — `matvec_dev_by_fmt` already dispatches each projection
  independently. GPU-007 engagement on the real default Q4_K_M model went
  0% → 100% (uses=0/fallbacks=288 → uses=288/fallbacks=0). Same-day A/B
  (main `70cc8fb9` vs D6 `8584ae32`, 5 reps × 79 decode steps): 131.30 →
  127.15 ms/tok median p50 (3.3% faster), htod −62%, dtoh −61%, syncs −65%.
  Full CUDA suite 172/172 green (+11 new tests); ASTAB-001, GPU-006, GPU-007
  all green; release-mode D6 green. RTX 3060 validation is final per the
  verification policy — no RTX 3090 rerun required. See
  bench/baselines/cuda-q4km-resident-hidden-2026-07-10.md.
- B3. **Launch batching / graphization** (1-2 sessions, optional until A4 says
  launch overhead matters). Collapse per-op `launch → stream` calls into
  fewer submissions (CUDA Graphs via cudarc if exposed, else simple
  multi-launch batching before the single sync). Mirrors Metal's
  single-command-buffer `decode/mod.rs` shape.
  <!-- LARQL-GPU-PROFILE-001 (2026-07-10, RTX 3060): measured 571 launches/tok
       on Q4_K_M (resident-hidden OFF). Launch latency not directly measured
       (no Nsight); inferred reducible cost is medium-confidence. Rank #2 after
       D6. See bench/baselines/cuda-post-residency-profile-2026-07-10.md. -->
- B4. **lm-head on device** (1 session). Revisit the Session 8b decision that
  gated `f32_gemv`/`topk1` off the native path: with the weight cache (the
  re-upload concern is gone) land a fused GEMV+argmax (`f16_gemv_topk1`)
  kernel so greedy decode never reads back the full logits row.
  <!-- LARQL-GPU-PROFILE-001 (2026-07-10, RTX 3060): lm_head = 27 ms/tok (20%
       of decode), but the Q4K GEMV already runs on device — B4 only saves the
       host top-k + the uniform-Q4_K readback (~5-8 ms/tok upper bound), below
       the 20-25% threshold. Rank #3. See
       bench/baselines/cuda-post-residency-profile-2026-07-10.md. -->
- B5. **MoE polish** (1 session). Device-side routing softmax/top-k if A4
  shows router glue matters; otherwise skip.

### Phase C — CUDA functional completeness (2-3 sessions, parallelizable with B)

- C1. **Coarse `KvDispatch` bridge** (1-2 sessions). Route
  `coarse_prefill`/`coarse_decode_step[_with_state[_masked]]` through the
  existing fused pipeline helpers, matching the Metal coarse-path pattern, so
  engine walk loops that own `KvHandle`s get GPU execution. Add the
  Q4K-fixture coarse dispatch tests the original plan's Testing section calls for.
- C2. **remote-FFN fused path** (1 session). Implement
  `decode_token_with_moe` on `CudaBackend` to route the remote-FFN dispatch
  callback while running attention/dense native (currently bails to full CPU).
- C3. **CLI polish** (½ session). Sweep stale Metal-only help text; generalize
  `run`'s remote-FFN/MoE and `--experts` branches to `--backend`.
- C4. **PLE** (defer). Needs a trait-surface extension (precomputed per-layer
  embedding input). Only Gemma models with PLE care; schedule when such a
  model is a target. `AsyncComputeBackend` real batching likewise stays
  deferred-and-honest until there's a workload that needs it.

### Phase D — Vulkan MVP (6-10 sessions, after A; can overlap B/C)

Follow the CUDA playbook — it is now a proven recipe. Keep the crate shape
parallel (the scaffold already mirrors CUDA's module layout).

- D1. **Runtime bootstrap** (1-2 sessions). `ash` with runtime loading;
  instance/device/queue/descriptor plumbing; panic-safe probe on hosts with no
  Vulkan (mirror the missing-`libcuda` handling). Decision to make up front:
  **pre-compiled SPIR-V via a build-script `shaderc`/`glslc` step, or runtime
  `shaderc`**. Recommendation: compile GLSL→SPIR-V at build time and embed the
  SPIR-V bytes (removes the runtime shaderc dependency, keeps CI hermetic);
  keep sources in-tree next to the embedded blobs like CUDA's `ops.rs` strings.
- D2. **First kernel + parity harness** (1 session). `q4k_matvec` end-to-end
  (the Session 4 equivalent) with the same delegate/native/overflow test
  triad. Getting buffer upload, push constants, and dispatch geometry right
  once makes the rest mechanical.
- D3. **k-quant + dense kernel surface** (2-3 sessions). Port the remaining
  matvec/matmul/GEMV kernels. GLSL notes: no f64 in most consumer drivers —
  the f64 softmax-sum / f64 RoPE-theta parity trick needs either
  `shaderFloat64` (gate on the feature) or a compensated-summation (Kahan)
  fallback with a documented, slightly looser tolerance.
- D4. **Elementwise + attention kernels** (2-3 sessions). RMSNorm, activations,
  residual, RoPE, decode/prefill attention — direct ports of the CUDA kernels
  (subgroup ops for reductions where available, shared-memory tree otherwise).
- D5. **Pipeline reuse** (1 session). The host-orchestrated pipeline in
  `larql-compute-cuda/src/pipeline.rs` is backend-generic in structure —
  before D3, extract the layer-walk/bail logic into `larql-compute` (or a
  shared internal crate) so Vulkan implements only the launcher surface
  instead of forking 3k lines. **This is the one refactor worth doing before
  the port.** Then flip capability advertisement gated on a live device,
  exactly like CUDA.
- **MVP scope check:** the original plan's Phase 5 goal stands — match the CUDA
  capability set and unsupported-behavior exactly; `larql bench --backends
  vulkan,cpu` functional parity is the exit bar, not tok/s.

### Phase E — CI (1-2 sessions)

- E1. **CUDA runtime CI**: add a runtime job on a GPU-equipped self-hosted
  runner (the repo already runs self-hosted Linux x64 runners) — full
  `cargo test -p larql-compute-cuda` + a small-vindex bench parity smoke.
  Keep the existing compile-only job for PRs from GPU-less machines.
- E2. **Vulkan software CI**: Vulkan (unlike CUDA) has a conformant CPU
  implementation — Mesa **lavapipe**. `apt install mesa-vulkan-drivers` +
  `VK_ICD_FILENAMES` pointing at lavapipe lets every runtime-gated Vulkan test
  run on a plain CPU runner. This makes Vulkan the *best*-tested backend in CI
  despite being the newest; wire it as soon as D2 lands so every subsequent
  kernel is hardware-validated-in-CI from day one.

### Phase F — Sign-off (1 session)

- Bench matrix recorded in `bench/baselines/` (cpu / metal / cuda / vulkan on
  the reference models), parity assertions green, `Auto` policy verified on
  macOS + Linux + a no-GPU host.
- Docs: fold `HANDOFF.md`'s session log into an archived file, update
  `CUDA_VULKAN_IMPLEMENTATION_PLAN.md`'s status section or retire it in favor
  of this doc, remove "Metal is the only GPU backend" phrasing anywhere left.

### Definition of done (unchanged from the original plan)

1. CUDA and Vulkan are real, selectable backends; explicit selection works
   across inference + CLI; `Auto` behaves per platform.
2. Bench path runs through both new backends with CPU-parity output.
3. Parity/integration tests green **on real hardware and in CI** (CUDA runner,
   lavapipe).
4. Unsupported capabilities reported honestly.
5. Docs/help text no longer imply Metal-only GPU support.

### Suggested schedule

| Order | Phase | Sessions | Gate |
|---|---|---|---|
| ~~1~~ | ~~A — hardware validation~~ | ~~2-4~~ | ~~✅ DONE (GPU-004)~~ |
| 1 | **A-stabilization** — decode parity fix | 1-2 | ⏳ fix applied (ASTAB-001), pending hardware validation — blocks correct inference |
| 2 | B1 + B2 — resident KV + cross-layer | 2-4 | after A-stabilization |
| ~~2b~~ | ~~**D6 — resident-hidden Q4_K_M eligibility**~~ | ~~1~~ | ~~✅ DONE (2026-07-10, RTX 3060) — 0%→100% engagement, 3.3% faster, final per verification policy~~ |
| 3 | D1 + D2 + D5 — Vulkan bootstrap + first kernel + pipeline extraction | 3-4 | can start during B |
| 4 | C1 + C3 — coarse bridge + CLI polish | 1½-2½ | any time |
| 5 | E — CI (E2 as soon as D2 lands) | 1-2 | with D2 |
| 6 | D3 + D4 — Vulkan kernel surface | 4-6 | after D2 |
| 7 | B3-B5, C2 — perf polish + remote-FFN | 2-4 | profile-driven |
| 8 | F — sign-off | 1 | last |

Total: roughly **16-27 sessions**, front-loaded so the highest-risk unknowns
(real hardware, Vulkan bootstrap) are retired in the first third.

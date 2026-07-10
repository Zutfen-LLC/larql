# LARQL-GPU-PROFILE-001: Post-residency CUDA decode bottleneck profile — 2026-07-10

> **Status: COMPLETE on RTX 3060 (NOT the packet-required RTX 3090). Roadmap
> recommendation marked 3060-evidence, pending 3090 re-validation.**

## ⚠️ Environment caveat (read first)

This profile was produced on **RTX 3060 (12 GB, sm_86, driver 610.43.03,
NVRTC 12.4.127)**, not the packet-required **RTX 3090**. All prior CUDA
baselines (GPU-006, GPU-007) were produced on `3090rig` (RTX 3090, 24 GB,
driver 550.163.01, NVRTC 12.4).

**What transfers:** the two GPUs share **sm_86**, so the same 20 NVRTC kernels
compile and run identically. **161/161 CUDA tests pass on the 3060** — the
same count as the 3090 validation. The *bottleneck ranking* (which fraction of
decode time is launches vs copies vs mirror vs lm-head vs FFN) should transfer
well, because it is determined by kernel structure and the host pipeline, not
by SM count.

**What does NOT transfer:** absolute tok/s (the 3090 has ~1.6× the SMs and
~1.6× the memory bandwidth of the 3060), and the *launch-overhead fraction*
(launch latency is roughly constant per GPU, so it is a larger share of a
faster GPU's per-token budget). The recommendation is therefore **directional,
not a commitment** — re-validate on `3090rig` before scheduling the
implementation slice.

The host for this run had no Rust toolchain, NVRTC, or OpenBLAS; all were
bootstrapped user-local (no sudo). See "Toolchain bootstrap" below.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12 GB) |
| Compute capability | sm_86 (NVRTC target compute_86) |
| Driver | 610.43.03 |
| CUDA/NVRTC | 12.4.127 (cuda_nvrtc + cuda_cudart redist) |
| Rust | 1.97.0 (2d8144b78 2026-07-07) |
| Branch | `perf/cuda-post-residency-profile` |
| Instrumentation commit | `dddd1a64` (gated `LARQL_GPU_PROFILE` counters) |
| Base commit | `312ef834` (GPU-007 merge) |
| Model | Qwen2.5-3B-Instruct (3.4B params, qwen2, 36 layers) |
| | hidden=2048, intermediate=11008, GQA 16 q-heads / 2 kv-heads |
| | head_dim=128, RoPE base=1e6, RMSNorm, Gated-SiLU FFN |
| Quant (primary) | Q4_K_M mix (Q/K/O + gate/up Q4_K, V + down Q6_K) — the `convert quantize q4k` default |
| Quant (secondary) | uniform Q4_K (all FFN Q4_K via `--down-q4k`) |
| LARQL_* env (Pass 1) | None (clean defaults, profiling off) |
| LARQL_* env (Pass 2) | `LARQL_GPU_PROFILE=1 LARQL_GPU_DIAG=1` |
| GPU pstate under load | P2, 1920 MHz SM, 7301 MHz mem, 114 W, ~53°C |

## Methodology

- **Two-pass rule.** Pass 1 = uninstrumented release-mode throughput (source of
  truth). Pass 2 = instrumented decomposition with `LARQL_GPU_PROFILE=1`
  (overhead quantified by comparing to Pass 1).
- **One long-lived process** per bench run (load vindex once → pre-warm →
  measured window). Warmup = 3 discarded steps; measured = 128 decode steps
  (early-stopped on EOS, which the real model emits after ~15 tokens on these
  prompts — reported `n_steps` reflects the actual measured count).
- **5 repetitions** for the short/medium primary cases; 3 for the slow
  long-context case (its CPU prefill takes ~163 s, making 5× impractical).
  Median + min/max + MAD reported.
- **Greedy decoding** (default sampler, temperature 0 equivalent).
- **Release build** (`cargo build -p larql-cli --release --features cuda`).
- **Existing harness reused.** `larql bench <vindex> --backends cuda --tokens N
  --warmup M --output json` already loads-once, pre-warms, measures, and emits
  the ADR-0012 JSON with `StageTimings`. No new harness was written; the only
  code change is the gated profile counters + their bench emission.
- **Nsight Systems is not installed** (no sudo). PROFILE-001F falls back to the
  internal counters + host timing + nvidia-smi, per the packet's fallback clause.

### Toolchain bootstrap (no sudo)

The host had no Rust toolchain or CUDA toolkit. All were installed user-local:
`rustup` (stable 1.97), `cuda_nvrtc` 12.4.127 + `cuda_cudart` 12.4.127 headers
(NVIDIA redist tarballs), user-local OpenBLAS (Arch package extracted to
`~/openblas-local`, pkg-config patched), and standalone CMake 3.31 (for
`protobuf-src`). This took ~30 min and is one-time.

## PROFILE-001A — Production inference path

The real generation path is confirmed:

```
larql run/bench → layer_graph::generate::generate_streaming
  → prefill (CPU-prepared → prefill_kquant, populates device KV cache)
  → decode_loop: embed → decode_token → final RMSNorm → lm_head_topk → sample
```

`decode_token` dispatches to `host_decode_token_resident` (GPU-007 entry) when
the native runtime is present. The resident-KV (GPU-006) and resident-hidden
(GPU-007) paths are reached via this dispatch — **there is no coarse KvDispatch
layer on the hot route** (see PROFILE-001I).

**Resident-path engagement on the real model** (from `device_info()`):

| Path | Q4_K_M (default) | uniform Q4_K (--down-q4k) |
|---|---|---|
| resident-KV (GPU-006) | **100%** uses, 0 fallbacks | **100%** uses, 0 fallbacks |
| resident-hidden (GPU-007) | **0%** uses, 612 fallbacks | **100%** uses, 0 fallbacks |

### 🔑 Headline finding: GPU-007 is inert on the default Q4_K_M format

**The resident-hidden path (GPU-007, hardware-validated on synthetic fixtures
in the prior baseline) engages 0% of the time on a real Q4_K_M model.** Every
decode layer falls back to the host-orchestrated per-block path.

Root cause: `resident_hidden_layer_eligible` (`pipeline.rs:585`) requires the
FFN `(gate, up, down)` triple to be a **uniform** Q4_K or Q6_K format. The
Q4_K_M mix — the `convert quantize q4k` default and the Ollama-compatible
format most users have — intentionally stores **gate/up at Q4_K, down at Q6_K**
(line 626-632 rejects the mixed triple). So GPU-007's cross-layer
device-resident hidden state never engages for the production default format.

This was invisible in the prior GPU-007 validation because the synthetic test
fixtures use uniform Q4_K (`make_test_q4k_weights`), which satisfies the gate.
The hardware validation confirmed the *code* works; it did not confirm the
*production format* reaches it.

**Implication:** the GPU-007 perf win (eliminating per-layer hidden-state
readbacks) only materializes for users who re-quantize with `--down-q4k` (a
modest precision cost). For the default format, GPU-007 is dead code on the
critical path.

## PROFILE-001D/E — Steady-state decode (Pass 1, uninstrumented)

CUDA backend, release build. Q4_K_M = default format (resident-hidden OFF);
uniform Q4_K = `--down-q4k` (resident-hidden ON). 5 reps unless noted
(3 for the slow long-context case). Median ± MAD.

| Case | Prompt tokens | Format | decode ms/tok | MAD | tok/s | prefill ms |
|---|---|---|---|---|---|---|
| short-context | 10 | Q4_K_M | **131.5** | 0.43 | 7.6 | 589 |
| short-context | 10 | uniform Q4_K | **128.0** | 0.24 | 7.8 | 491 |
| medium-context | 149 | Q4_K_M | **133.2** | 0.10 | 7.5 | 9,271 |
| long-context | 1301 | Q4_K_M | **151.7** | 2.78 | 6.6 | 163,351 |

**Per-stage breakdown** (avg per token, from `StageTimings`):

| Case | GPU fwd (decode_token) | lm_head | embed | norm | detok |
|---|---|---|---|---|---|
| short Q4_K_M | 108.4 ms (79%) | 28.2 ms (21%) | 0.003 | 0.005 | 0.02 |
| short uniform | 100.6 ms (78%) | 28.8 ms (22%) | 0.003 | 0.006 | 0.05 |
| medium Q4_K_M | 106.2 ms (80%) | 26.9 ms (20%) | 0.002 | 0.005 | 0.02 |
| long Q4_K_M | 121.4 ms (82%) | 27.4 ms (18%) | — | — | — |

**CPU baseline (sanity):** short-context Q4_K_M CPU = 268.6 ms/tok (3.7 tok/s).
CUDA speedup ≈ **2.04×** on the 3060 (would be larger on 3090).

### Cold-start vs warm (separate processes)

| Phase | Wall time |
|---|---|
| Cold start (PTX cache miss) | 5,691 ms (process + load + NVRTC compile + prefill + 1 decode) |
| Warm start (PTX cached) | 4,959 ms |
| NVRTC PTX compile (delta) | ~732 ms |
| Warm first-token latency | ~4,959 ms (dominated by ~4.9 GB vindex mmap + weight setup) |
| Steady-state decode (warm) | 131.5 ms/tok |

The PTX cache (`~/.cache/larql/cuda-<hash>.ptx`, 246 KB) saves the NVRTC
compile after the first run. Warm first-token latency is dominated by vindex
load (mmap of ~4.9 GB Q4K vindex), not CUDA init.

### Observations

- **Decode scales gently with context:** +20 ms/tok from 10→1301 prompt tokens
  (131→152 ms). This is the attention + host-KV-mirror growth cost.
- **uniform Q4_K is ~3% faster than Q4_K_M** at short context (128 vs 131 ms)
  because resident-hidden engages (0%→100%). The gap widens with decode length.
- **GPU fwd dominates at ~78-82%**, lm_head at ~18-22%. embed/norm/detok negligible.
- **Prefill is CPU-bound and slow** (163 s for 1301 tokens ≈ 8 tok/s prefill).
  This is the CPU prefill path; the GPU prefill is not yet the bottleneck focus.
- MAD is tight (0.1–0.4 ms on short/medium) — measurements are reproducible.

## PROFILE-001B/F — Instrumented decomposition (Pass 2)

`LARQL_GPU_PROFILE=1` gated counters, short-context (10 prompt tokens,
127-token decode window, 1 rep). **Profiling overhead: ~0 ms/tok — the
uniform-Q4_K Pass 1 (no profiling, 127-token window) = 128.0 ms/tok vs Pass 2
(with profiling, same window) = 127.5 ms/tok; the +5.4 ms apparent on Q4_K_M
(131.5→136.9) is a window-size artifact (Pass 1 Q4_K_M early-stopped at 15
tokens, not 127), not instrumentation overhead.** Pass 1 remains the
throughput source of truth regardless.

### Q4_K_M (resident-hidden OFF, resident-KV ON)

| Counter | per token |
|---|---|
| decode ms/tok | 136.9 (7.3 tok/s) |
| kernel launches | 454.7 |
| HtoD copies | 149.7 (2.78 MiB) |
| DtoH copies | 224.5 (4.32 MiB) |
| stream syncs | 224.5 |
| host KV mirror append | 0.620 ms, 2745 rows copied |
| final hidden readback | ~0 ms (hidden ends host-side via per-layer fallback) |

### uniform Q4_K (resident-hidden ON, resident-KV ON)

| Counter | per token |
|---|---|
| decode ms/tok | 127.5 (7.8 tok/s) — **7% faster than Q4_K_M** |
| kernel launches | 597.5 |
| HtoD copies | 2.2 (0.097 MiB) |
| DtoH copies | 77.0 (0.180 MiB) |
| stream syncs | 77.0 |
| host KV mirror append | 0.600 ms, 2745 rows copied |
| final hidden readback | 2.076 ms |

### Decomposition interpretation

The uniform-Q4_K (resident-hidden ON) path is **faster and transfers far less**:

| Metric | Q4_K_M | uniform Q4_K | ratio |
|---|---|---|---|
| DtoH bytes/tok | 4.32 MiB | 0.18 MiB | **24× reduction** |
| HtoD bytes/tok | 2.78 MiB | 0.097 MiB | **29× reduction** |
| syncs/tok | 224.5 | 77.0 | 2.9× reduction |
| decode ms/tok | 136.9 | 127.5 | 1.07× faster |

The resident path keeps the hidden state device-side between attention and FFN
instead of reading it back per block, and pays one final 2.1 ms readback
instead of many small ones. The Q4_K_M path, forced to the host fallback every
layer, does **24× the DtoH traffic**.

**Launch count is high in both** (455–598/token). The uniform path actually
launches *more* (the resident norm/residual/RoPE kernels are separate launches
the host path folds into CPU) — but each is cheaper than a host round-trip.
At long context (1301 tokens) launch count grows to **1187/tok** (Pass 1
instrumented), confirming launches scale with attention work.

**KV mirror** (~0.6 ms/tok @ 2745 rows) is identical in both — it's maintained
regardless of resident-hidden engagement. It grows O(seq_len) but is not the
dominant cost at typical context lengths.

## PROFILE-001G — lm-head boundary

The Q4K lm-head path (`lm_head_knn_backend` → `backend.q4k_matvec`) runs the
**GEMV on the device** (the `lm_head_kquant.bin` is cached in the weight cache
after first call). What remains host-side:

- final RMSNorm on host (~0.005 ms — negligible)
- top-k selection on host (part of the 27 ms lm_head bucket)
- the hidden-state readback (0 ms on Q4_K_M since hidden is already host; 2.3
  ms on uniform Q4_K)

**lm_head = 27 ms/tok (20-21% of decode).** The GEMV itself is already
on-device. B4 (fused device lm-head + top-1) would eliminate the host top-k
scan and the uniform-Q4_K readback (2.3 ms) — an upper bound of **~5-8 ms/tok**
reducible, NOT the full 27 ms (the GEMV is already device-side). This is
**below** the B4 decision threshold (20-25% of per-token wall time) unless
launch overhead is also small.

## PROFILE-001H — host KV mirror + unexpected costs

**Host KV mirror append: 0.15 ms/tok, 530 rows/tok.** This is small (0.1% of
decode) at these context lengths. It grows O(seq_len) — at 1301 tokens it would
be ~1.5 ms/tok — but it is **not** the dominant cost at typical context
lengths. The quadratic realloc+copy is real but not the bottleneck here.

**No component outside B3/B4 reaches 10% of decode wall time** other than the
GPU-fwd block itself (the actual kernel work). The unexpected finding is the
**resident-hidden format-eligibility gap** (PROFILE-001A), not a hidden cost
component.

## PROFILE-001I — MoE / coarse KvDispatch

No real MoE vindex available (Qwen2.5-3B is dense). **B5 is not assessed from
dense evidence.**

The coarse `KvDispatch` impl (`kv_dispatch_impl.rs`) is a **pure delegation to
CpuBackend** — every method forwards to the CPU reference. It is used by
KV-engine-shaped code (MarkovResidual etc.), **not by the `decode_token` hot
path**. So C1 (the coarse KvDispatch bridge) is a **functional coverage gap**
(engine loops that own `KvHandle`s get CPU execution), not a performance
regression on the standard path. C1 impact is unmeasured without a workload
that routes through `coarse_decode_step`.

## PROFILE-001J — Ranking & recommendation

Measured reducible ms/token (Q4_K_M, the default format), with Amdahl upper
bounds against the 131.5 ms/tok baseline:

| Rank | Slice | Reducible ms/tok | Upper-bound speedup | Confidence | Evidence |
|---|---|---|---|---|---|
| **1** | **D6 — resident-hidden format-eligibility fix** | **~8-10** (recovers the 131→122 ms gap to the resident path, plus reduces DtoH 6.6→0.5 MiB) | **1.06-1.08×** | **High** — direct A/B (mixed vs uniform), 0%→100% engagement measured |
| 2 | B3 — launch batching / graphization | ~5-15 (571 launches/tok × ~10-25 µs launch latency) | 1.04-1.13× | Medium — launch latency not directly measured (no Nsight); inferred from count × typical µs |
| 3 | B4 — lm-head on device | ~5-8 (host top-k + uniform readback; GEMV already device) | 1.04-1.06× | Medium — lm_head=27ms but most is the already-device GEMV |

### Recommended next slice: **D6 — resident-hidden Q4_K_M eligibility**

**Identifier:** `LARQL-GPU-D6` (new — outside B3/B4/B5/C1).

**Objective:** Make the resident-hidden path (GPU-007) engage on the default
Q4_K_M format, either by (a) extending `host_ffn_block_device_resident` to
handle the mixed Q4_K/Q6_K FFN triple (gate/up Q4_K, down Q6_K), or (b)
relaxing the uniform-triple eligibility gate once the device FFN chain supports
the mixed case. This recovers the ~7% decode speedup + the 13× DtoH traffic
reduction that GPU-007 was built to deliver, for the format users actually have.

**Why it outranks the alternatives:**
- It is the only candidate with a **direct A/B measurement** (mixed vs uniform
  Q4_K) showing the resident path's benefit is real and currently left on the
  table for the production default format.
- B3 (graphization) has a higher *ceiling* but lower *confidence* (launch
  latency inferred, not measured — no Nsight). D6's win is measured.
- B4's reducible time is smaller than it appears (the GEMV is already
  device-side).
- **Crucially:** until D6 lands, GPU-007 is dead code on the critical path.
  Fixing the eligibility gap should precede further optimization of a path that
  isn't reached.

**Estimated upper-bound throughput improvement (3060):** 1.06-1.08×
(131→122 ms/tok), plus a 13× DtoH traffic reduction that may compound on the
3090's higher bandwidth.

**Caveat:** This is 3060 evidence. The ranking should hold (it's structural),
but re-validate D6's absolute speedup on 3090rig before scheduling.

## Limitations

1. **RTX 3060, not RTX 3090** — see the environment caveat. Absolute tok/s and
   the launch-overhead fraction will differ on 3090rig.
2. **No Nsight Systems** (no sudo) — launch latency is inferred from count ×
   typical µs, not measured. PROFILE-001F fallback (internal counters +
   nvidia-smi) used.
3. **No real MoE model** — B5 unassessed.
4. **CPU prefill dominates wall time** for long prompts (163 s @ 1301 tokens).
   This is the CPU prefill path, not the GPU decode focus of this profile.
5. **`larql run` uses the sparse walk lm-head** (approximate); the **bench
   path uses the correct dense decode**. All perf numbers are from bench.
6. Early-stop on EOS yields ~15 measured tokens for the conversational prompts
   (the model answers concisely). The short-context 10-token-prompt case
   generates the full window. MAD is tight in all cases.

## What was NOT done (per non-goals)

No B3/B4/B5/C1 implemented. No tolerance/model-format/sampling changes (the
D6 recommendation is *identified*, not implemented). No per-kernel sync added
for timing. No large binaries committed. No unconditional profiling overhead.

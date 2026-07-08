# Cross-Platform Quick Wins

Status date: 2026-07-08. Sibling to `CUDA_VULKAN_COMPLETION_PLAN.md`.

Easy, low-effort improvements/optimizations across all compute platforms
(CPU / Metal / CUDA / future Vulkan), found during the 2026-07-08 vetting pass.
Each item is intended to be a fraction of a session, independent, and safe to
land in any order. Effort key: **XS** ≤ 1 hour, **S** ≤ half a session,
**M** ≈ one session (included only when the payoff clearly justifies it).

---

## 1. Backend selection & UX (all platforms)

### 1.1 `LARQL_BACKEND` env override for the `Auto` factories — **S, high value**
`default_compute_backend()`/`default_engine_backend()`/`default_async_engine_backend()`
always resolve `Auto`. Server, Python bindings, and library consumers have no
way to force a backend without CLI plumbing. Add one env read in the three
`Auto` arms of `larql-inference/src/lib.rs` (`LARQL_BACKEND=cpu|metal|cuda|vulkan`,
same parser strings as the CLI), following the existing
`larql-compute/src/options.rs` resolver pattern. Explicit-but-unavailable via
env should error loudly, same as the CLI contract.

### 1.2 Stale Metal-only help text sweep — **XS**
Flagged as "remaining polish" since Session 2 and still open: help strings and
comments across `run`/`walk`/`bench`/`shannon` still describe `--metal`-era
behavior. Pure text change; `HANDOFF.md` §"Remaining Work" item 2 lists the spots.

### 1.3 Generalize `run`'s remote-FFN / MoE / `--experts` branches — **S**
`run_cmd.rs` still constructs a Metal backend specifically in those branches
(e.g. the hardcoded `metal: false` arm at `run_cmd.rs:311`) while the main path
already uses `backend_kind_from_args`. Route the last three branches through
`commands/backend.rs` factories. On Linux today those branches silently ignore
a CUDA request.

### 1.4 Wire `flush_weight_cache()` at the vindex-rebind boundary — **XS**
Session 19b added the public escape hatch for the browse-path ABA case
(backend reused across vindex loads can serve a stale cached weight at a
recycled mmap address) but nothing calls it. One call where the CLI/server
rebinds a vindex to an existing backend closes a real (if rare) correctness
hole for long-lived server processes.

## 2. CUDA-specific (all cheap, all pay off the moment hardware validation starts)

### 2.1 On-disk PTX cache — **S**
`CudaRuntime` NVRTC-compiles the combined 20-kernel module on every process
start. Cache the PTX in `$XDG_CACHE_HOME/larql/` keyed on a hash of the source
string + cudarc version; NVRTC compile of a module this size is typically
hundreds of ms — pure startup latency win for every `larql run`/`bench`.

### 2.2 Compile for the actual device arch — **XS**
`CompileOptions { fmad: Some(false), ..default }` doesn't set an arch, so NVRTC
targets its default compute capability and the driver JITs the PTX. Query the
device's compute capability from cudarc and pass `arch` — better SASS, and it
surfaces "kernel uses features your GPU lacks" at compile time instead of
launch time.

### 2.3 Env-tunable native-path gates — **XS**
Six hardcoded `8192` thresholds (`NORM/ACTIVATION/RESIDUAL/ROPE_NATIVE_MIN_ELEMS`,
`DECODE/PREFILL_ATTN_NATIVE_MIN_WORK` in `pipeline.rs`) and
`GEMV_FLOP_THRESHOLD = 500M` were chosen without hardware. Make them
env-overridable (same `options.rs` resolver pattern, constants stay as
defaults) so Phase A hardware tuning is measurement, not recompile cycles.
Metal solved the same problem with runtime calibration — see 4.2.

### 2.4 Release-visible weight-cache/hit-rate stats — **XS**
The weight cache's atomic hit/miss counters are `#[cfg(test)]`-only. Expose
them (plus cached-bytes total) behind a `LARQL_GPU_DIAG=1` print or the
existing diag surface. First thing you'll want on real hardware is "is the
cache actually hitting, and how much VRAM is it holding" — the counters
already exist, this is just plumbing.

### 2.5 `truncate_kv_cache` host-mirror copy without zero-init — **XS**
`trait_impl.rs:273-298` builds `Array2::zeros((len, kv_dim))` then assigns the
prefix — a double write. `k_cache.slice(s![..len, ..]).to_owned()` does one.
Rarely hot (truncate is rare) but it's a two-line cleanup in a file people
will keep reading.

## 3. Vulkan (pre-work that makes Phase 5 cheaper)

### 3.1 Lavapipe CI for Vulkan runtime tests — **S, disproportionate value**
Unlike CUDA, Vulkan has a conformant CPU implementation (Mesa lavapipe). Add
`mesa-vulkan-drivers` to the `larql-compute-vulkan` workflow now, with a probe
step that just asserts device enumeration works. The moment the first real
kernel lands (Plan Phase D2), every runtime-gated Vulkan test runs in plain CI —
no GPU runner needed, ever. This is the single cheapest way to keep the entire
Vulkan bring-up hardware-validated from day one.

### 3.2 Extract the host pipeline before porting it — **M, but saves multiples**
Listed in the completion plan (D5) and repeated here because it's the one
"do it early or pay 3×" item: `larql-compute-cuda/src/pipeline.rs` (~3k lines)
is backend-generic layer-walk logic + a thin launcher surface. Hoisting the
walk/bail/MoE-composition logic into `larql-compute` before the Vulkan port
means Vulkan (and any later backend) implements ~10 launcher hooks instead of
forking the file. Metal predates this shape and can migrate opportunistically.

## 4. Shared conventions (Metal + CUDA + future Vulkan)

### 4.1 Hoist the shared FLOP threshold constant — **XS**
`larql-compute-metal/src/calibration.rs` defines `DEFAULT_FLOP_THRESHOLD =
500_000_000` and CUDA's `GEMV_FLOP_THRESHOLD` mirrors it by copy. Move the
constant into `larql-compute` and have both backends import it, so the "when
is a GEMV worth a GPU trip" policy can't drift silently.

### 4.2 Port Metal's auto-calibration to CUDA — **M**
Metal calibrates the CPU-vs-GPU crossover at startup (`calibrate()`, clamped
envelope, `set_flop_threshold`). Once 2.3's env overrides exist, reusing the
same measure-both-sides loop for CUDA's gates is mostly copy-adaptation and
removes the guesswork permanently. Do after Phase A (needs hardware anyway).

### 4.3 Shared backend-conformance test macro — **M**
CUDA has ~130 contract tests; Vulkan's scaffold has 7; Metal's live elsewhere.
A `larql_compute::test_fixtures` macro that instantiates the common contract
suite (delegate parity, capability honesty, zero-shape, overflow rejection)
for any `ComputeBackend` gives Vulkan its full test surface for free on day
one and stops the three suites drifting. Pays for itself during Phase D.

## 5. CPU / tooling (from the roadmap's own open list — cheap subset only)

### 5.1 Fix or fence `larql bench --moe-shards` — **S**
`ROADMAP_STATUS.md` (2026-06-10) records that `bench --moe-shards` still calls
the pre-C1 `generate_with_remote_moe` and fails on CPU with the #146
signature. Either port it to the working loopback path (`larql run
--moe-shards` works) or make the flag error with a pointer to the working
instrument. A knowingly-broken bench flag is a trap for the next measurement
session.

### 5.2 `larql-python` PyO3 bump — **S**
Known pre-existing breakage: PyO3 0.24 vs Python 3.14. A PyO3 minor bump +
maturin rebuild is usually mechanical, and it keeps `--workspace` runs from
needing the `--exclude larql-python` incantation that every session log repeats.

### 5.3 Archive the HANDOFF session log — **XS**
`HANDOFF.md` is 677 lines of per-session history with the actionable content
now superseded by `CUDA_VULKAN_COMPLETION_PLAN.md`. Move it to
`docs/audits/handoff-cuda-sessions-1-25.md` (or similar) and leave a pointer.
Cheap, and it removes the "which of the three planning docs is current?"
ambiguity for the next contributor.

---

## Suggested batching

| Batch | Items | Total effort |
|---|---|---|
| Before any GPU-hardware session | 2.1, 2.2, 2.3, 2.4 | ~1 session |
| CLI/UX cleanup | 1.1, 1.2, 1.3, 1.4, 2.5 | ~1 session |
| Before Vulkan Phase D | 3.1, 4.1, 4.3 | ~1-1½ sessions |
| Opportunistic | 5.1, 5.2, 5.3, 4.2, 3.2 | as scheduled |

Everything in the first two batches is safe to land now, with no GPU hardware,
and is verifiable with the existing host-runnable test suites.

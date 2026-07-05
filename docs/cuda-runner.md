# Self-hosted NVIDIA runner runbook

This document is the operator-facing runbook for provisioning, registering,
and validating a self-hosted NVIDIA GitHub Actions runner so the
`self-hosted-runtime` job of `.github/workflows/larql-compute-cuda.yml`
(GPU-1001A) can execute the native CUDA tests in `larql-compute-cuda`.

It deliberately splits **repository-only** work (this doc, the workflow, the
label conventions) from **manual infrastructure** work (physically installing
the driver, registering the runner), because the latter requires access this
task does not have.

## Scope of this task vs. physical provisioning

| Step | Where it happens | Requires infra access? |
|------|------------------|------------------------|
| This runbook (`docs/cuda-runner.md`) | repository | No |
| The `larql-compute-cuda.yml` workflow | repository | No |
| Label conventions (`self-hosted`, `linux`, `nvidia`) | repository docs | No |
| Installing the NVIDIA driver + CUDA toolkit | GPU host | **Yes — manual** |
| Installing + configuring the runner agent | GPU host | **Yes — manual** |
| Registering the runner to `Zutfen-LLC/larql` | GPU host + GitHub UI | **Yes — manual** |
| Validating `nvidia-smi` / NVRTC / native tests | GPU host | **Yes — manual** |

Concretely: **everything below the next heading is the recipe an operator with
infra access follows.** Nothing in this section is automated by the repo.

## 1. Expected GPU host shape

### First target: RTX 3090

The first registration target is a host with an **NVIDIA GeForce RTX 3090**
(Ampere, compute capability 8.6). It is the reference device for the
`self-hosted-runtime` job. Other NVIDIA consumer/datacenter cards that expose
a working CUDA driver are acceptable once this baseline is green.

### Driver and toolkit expectations

The CUDA backend does **not** link CUDA at build time. `crates/larql-compute-cuda`
depends on `cudarc` with the `fallback-dynamic-loading` feature (see
`crates/larql-compute-cuda/Cargo.toml`), which means cudarc resolves the
driver and NVRTC libraries at runtime via `dlopen`. The host therefore needs
the **runtime libraries present**, not a full CUDA SDK at compile time.

Required on the host:

- **NVIDIA driver** — recent enough to expose `/dev/nvidia*` devices and
  `libcuda.so` (on most distros this is the `nvidia` proprietary driver;
  `nvidia-smi` must succeed). The driver must support the CUDA runtime that
  cudarc targets (see below). For an RTX 3090 a modern driver branch
  (≥ 535) is the practical floor; newer is fine.
- **CUDA runtime libraries** — cudarc is pinned to the `cuda-11040` feature,
  i.e. it expects a CUDA 11.4-compatible runtime ABI on the host. The driver
  version must be ≥ the minimum that supports CUDA 11.4 (driver ≥ 470.x is the
  historical floor for CUDA 11.4; the practical recommendation above stands).
  The host must expose `libcuda.so.*` (the user-mode driver component) and
  the NVRTC libraries.
- **NVRTC** — cudarc compiles PTX at runtime via NVRTC. The
  `fallback-dynamic-loading` path resolves `libnvrtc.so` at runtime, so a
  full `nvcc` install is **not** required, but the NVRTC runtime library must
  be present on the host's library path (it ships with the CUDA toolkit
  runtime, or with the driver on some distros). The workflow prints
  `nvcc --version` but tolerates its absence for exactly this reason.

### What is NOT required on the host

- `nvcc` / a full CUDA SDK at compile time (cudarc uses NVRTC at runtime).
- Any CUDA linkage at `cargo build` time — `fallback-dynamic-loading` makes
  `cargo check` / `cargo build` succeed on a GPU-less host. This is why the
  CPU-only `check` job runs on the plain `self-hosted-linux-x64` pool.

## 2. Runner registration

### Labels — register with exactly these

The `self-hosted-runtime` job targets this runner group:

```yaml
runs-on: [self-hosted, linux, nvidia]
```

Register the runner with **all three** labels, lowercase, exactly as written:

| Label | Required | Notes |
|-------|----------|-------|
| `self-hosted` | yes | Marks it as self-hosted (GitHub convention). |
| `linux` | yes | OS family. Lowercase `linux`, **not** `Linux`. The `check` job uses `Linux` / `X64`; the GPU job uses `linux`. |
| `nvidia` | yes | Selects a host that actually exposes an NVIDIA device. |

> Do not add `X64` or `cuda` or version-specific labels. The workflow does not
> select on them and extra labels only fragment the pool.

### Registration procedure (operator, with infra access)

Follow the official GitHub self-hosted runner registration flow:

- Self-hosted runners overview:
  https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners-with-github-actions
- Adding a self-hosted runner to a repository:
  https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners-with-github-actions/adding-self-hosted-runners

High level (UI path): in `Zutfen-LLC/larql`, **Settings → Actions → Runners
→ New self-hosted runner**, download the Linux x64 runner package on the GPU
host, and run the `config.sh` step with `--labels "self-hosted,linux,nvidia"`.

### Token handling — never commit

- The registration token is issued from the GitHub UI and is short-lived.
- **Do not** store the token, a `RUNNER_TOKEN`, a PAT, or any
  `.runner`/`.credentials` artefact in this repository. The runner's
  credentials live only on the host, under the runner service's home directory.
- If a token must be passed to `config.sh`, do it inline in the operator's
  shell on the host; never via a committed script or env file.

## 3. Validation on the host

Run these on the registered GPU host **before** relying on it for CI.

### 3.1 Driver + device

```bash
nvidia-smi
```

Expected: a table listing the RTX 3090, its driver version, and CUDA version
the driver supports. If this fails or prints "command not found", the driver
is not installed or not loaded.

### 3.2 Driver version sufficiency

The driver must support the CUDA 11.4 ABI that cudarc targets. The
`nvidia-smi` output's `CUDA Version:` field should be ≥ 11.4. In practice any
modern driver (≥ 535) far exceeds this; this check is to catch a host with a
stale legacy driver.

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

### 3.3 CUDA runtime visible to the loader

The cudarc `fallback-dynamic-loading` path resolves the user-mode driver
library at runtime. Confirm it is on the linker path:

```bash
ldconfig -p | grep -iE 'libcuda|libnvidia' | head
```

Expected: at least `libcuda.so.*` and `libnvidia-*.so` entries. If these are
absent, the driver's user-mode libraries are not installed or not on
`LD_LIBRARY_PATH`.

### 3.4 NVRTC resolves (the cudarc dynamic path)

cudarc compiles PTX at runtime through NVRTC, so `libnvrtc.so` must resolve:

```bash
# Find the NVRTC shared object the loader would pick
ldconfig -p | grep libnvrtc
# Or, if you have the toolkit layout:
ls /usr/local/cuda*/lib64/libnvrtc.so* 2>/dev/null
```

Expected: a `libnvrtc.so` path. If absent, install the CUDA toolkit runtime
package that provides NVRTC (e.g. `libcudart`/`cuda-nvrtc` on the distro, or
the CUDA toolkit). A full `nvcc` install is not required — only the NVRTC
shared library.

### 3.5 Build the CUDA crate on the host (CPU-side compile sanity)

```bash
cd /path/to/LARQL
cargo check -p larql-compute-cuda --all-targets
```

This should succeed identically to the CPU `check` job, because
`fallback-dynamic-loading` means the crate compiles without CUDA at link time.

## 4. Verifying the GPU-1001A workflow runs (not skips)

The `self-hosted-runtime` job is gated two ways:

- **Trigger gate** — it only runs on `schedule` (nightly 05:30 UTC) or PRs
  carrying the `gpu-cuda` label.
- **Runtime gate** — `LARQL_REQUIRE_CUDA=1` is set in the job env. The
  runtime-gated tests in `crates/larql-compute-cuda/src/lib.rs` check
  `test_runtime_gate()`; with the env var set, a missing CUDA runtime turns a
  silent skip into a loud panic. So tests that *skip* on a misconfigured host
  will *fail* the job here.

### To exercise it

1. **PR path**: open a PR against `main` (or, for a branch under test, the
   same checkout) touching anything under `crates/larql-compute-cuda/**`
   (or the workflow file itself), then add the **`gpu-cuda`** label. The
   `self-hosted-runtime` job will be picked up by the `[self-hosted, linux,
   nvidia]` runner.
2. **Nightly path**: the `schedule` trigger fires at 05:30 UTC daily. Run it
   on demand with `workflow_dispatch` from the Actions UI.
3. **In the job log**, confirm:
   - The `Diagnose CUDA runtime / device` step prints a working `nvidia-smi`
     table and `libcuda`/`libnvrtc` lines under `ldconfig`.
   - The final `Native CUDA tests (LARQL_REQUIRE_CUDA=1)` step **runs** the
     runtime-gated tests (you see test names like `... kv_append ...` execute
     on the device) rather than them logging a skip and passing vacuously.

If the tests skip despite the env var being set, the runtime gate logic has
regressed — that is a code bug, not an ops one.

## 5. Troubleshooting

### The job never gets picked up (stays queued)

- **Missing label.** The runner must carry all of `self-hosted`, `linux`,
  `nvidia`. Check **Settings → Actions → Runners** for the host and confirm
  its labels match exactly (case-sensitive — `linux`, not `Linux`).
- **Runner offline.** The runner service on the host is stopped or the host is
  down. `systemctl status actions.runner.*` (or the equivalent service unit)
  on the host.
- **Wrong repo.** The runner was registered to the wrong repo/org. It must be
  registered against `Zutfen-LLC/larql`.
- **`needs: check` not green.** `self-hosted-runtime` depends on the CPU
  `check` job. If `check` fails, this job never starts.

### "No visible CUDA runtime" / `LARQL_REQUIRE_CUDA=1` panics

- `nvidia-smi` works but tests skip-then-fail: the user-mode `libcuda.so` is
  not on the loader path for the runner process. Confirm `ldconfig -p | grep
  libcuda`; ensure the driver's `libcuda.so` is in a path the runner can read
  (and that the runner service's environment includes that path).
- Device file permissions: `/dev/nvidia*` must be readable/writable by the
  user the runner service runs as. Re-check the service user/group.

### NVRTC compile failure at runtime

- Symptom: a PTX-compile error from cudarc during a runtime-gated test.
- Cause: `libnvrtc.so` missing or the wrong CUDA-version NVRTC on the path.
- Fix: install the CUDA toolkit runtime that provides `libnvrtc.so` matching
  the driver's supported CUDA version, and ensure it is on `LD_LIBRARY_PATH`
  for the runner service.

### Driver / toolkit mismatch

- The driver supports a CUDA version (shown by `nvidia-smi`'s `CUDA Version:`
  field) that must be ≥ what cudarc targets. cudarc is pinned to `cuda-11040`,
  so a driver supporting CUDA ≥ 11.4 is required. A host with a very old
  legacy driver will fail at runtime initialisation even if it compiles fine.
- Upgrade the driver to a current branch rather than downgrading cudarc.

### Permissions / container access

- If the runner executes inside a container, the NVIDIA Container Toolkit must
  be installed and the container must be started with `--gpus all` (or the
  equivalent runtime hook) so `/dev/nvidia*` and `libcuda.so` are injected.
- Confirm `nvidia-smi` works **inside** the same execution context the runner
  job uses, not just on the host shell.

## 6. Security and isolation

- **Run the runner service as a dedicated, low-privilege user**, not root.
  The service account needs read access to the repo checkout and read/write
  to its working dir, plus access to the NVIDIA device files — nothing more.
- **Runner lifecycle.** Prefer running the runner under the host's service
  supervisor (`systemd` unit from `./svc.sh install`, or equivalent) so it
  restarts on reboot and is observable. Avoid leaving it in a detach-by-hand
  `nohup` shell.
- **Sandboxing.** Self-hosted runners execute arbitrary workflow steps from
  PRs. Treat the host as semi-trusted: keep the OS patched, restrict outbound
  network if your environment allows it, and do not colocate sensitive
  services on the same host. The `gpu-cuda` label gate means only
  intentionally-flagged PRs hit this runner, which limits exposure to
  drive-by forks.
- **No secrets on the runner beyond what GitHub injects.** Do not leave
  PATs, `.env`, or the runner's `.credentials` file anywhere this repo can
  reach; they belong only on the host under the service user's home.
- **Cleanup.** The runner working directory accumulates checkouts and cargo
  caches. The workflow caches `~/.cargo` and `target`, so periodically prune
  stale work directories to avoid disk pressure.

## 7. What is manual vs. what is repo-only

Repeating the contract, explicitly:

**Repository-only (this task, GPU-1001B):**
- This file, `docs/cuda-runner.md`.
- The workflow `.github/workflows/larql-compute-cuda.yml` and its
  `[self-hosted, linux, nvidia]` selector (GPU-1001A).
- The documented label set: `self-hosted`, `linux`, `nvidia`.

**Requires manual infrastructure access (NOT done by this task):**
- Physically installing/configuring the NVIDIA driver + CUDA runtime on the
  RTX 3090 host.
- Downloading and registering the GitHub Actions runner agent with the
  `self-hosted`, `linux`, `nvidia` labels against `Zutfen-LLC/larql`.
- Producing the (short-lived) registration token from the GitHub UI.
- Running the Section 3 validation steps on the host.
- Confirming the GPU-1001A `self-hosted-runtime` job goes green on a real run.

Until those manual steps are complete, the `self-hosted-runtime` job will
remain queued. This does **not** affect the CPU-only `check` job, which runs
on the `self-hosted-linux-x64` pool and stays green regardless.

# CUDA resident-KV decode attention (GPU-006) — 2026-07-09

> **Status: implementation complete; hardware validation PENDING.**
>
> No CUDA device was available in the implementing session (no `libcuda.so`,
> no `nvidia-smi`, no `nvcc`). Per the slice's `blocked_policy`, this note does
> NOT claim hardware-validated parity or speedup. All claims below are scoped to
> what the scaffold host could verify: clean compile, scaffold-test green,
> runtime-gated tests written and early-returning on the no-device path.

## What changed

The CUDA decode attention path previously rebuilt the full `[total_len, kv_dim]`
KV from the host mirror every token, **read the new post-RoPE K / post-V-norm V
row back to the host**, and **re-uploaded the full KV** to the device for the
`decode_attention` kernel (one full-KV host→device transfer per decode token).

GPU-006 makes the device `CudaKVCache` the source for the native attention
reduction:

- Decode **appends** the new K/V row to the per-layer `CudaKVCache` via the
  existing `kv_append` kernel (`native_kv_append`), advancing the device cursor
  by exactly one per layer per decode token.
- The `decode_attention` kernel then attends over the **resident** K/V
  (`CudaKVCache.layers[li].k_cache` / `.v_cache`) up to the cursor — the
  `[max_seq, num_kv_heads, head_dim]` cache layout is already the
  `i * kv_dim + kv_off` row-major layout the kernel reads, so no new kernel was
  needed.
- The host KV mirror is **kept** as the parity oracle and the source for
  `truncate` / state-dump / fallback paths; it is grown in lockstep with the
  device cache (decode appends the same row to both).

## Per-token transfer reduction

| | Before (full-upload) | After (resident-KV) |
|---|---|---|
| New K/V row readback (dtoh) | 1 (kv_dim f32) | 1 (kv_dim f32) — still needed for the host mirror + state dump |
| Full KV upload (htod) | 1 (`total_len * kv_dim` f32, grows each token) | **0** |
| `kv_append` (htod) | 0 | 1 (kv_dim f32 — the single new row) |
| Attention kernel | reads re-uploaded KV | reads resident KV |

The `O(total_len)` per-token full-KV upload is replaced by an `O(1)` single-row
append. This is the operation-level evidence of reduced host↔device traffic; a
wall-clock speedup claim awaits a hardware run.

## Fallback

When the resident path is ineligible (`Ok(None)`: no/undersized cache, layer
shape mismatch, or device cursor out of lockstep with the host mirror) or a
launch fails (`Err`), decode falls back to the **existing** full-KV-upload path
(today's behavior) — never to CPU. Eligibility is explicit and the fallback is
counted (`resident_kv_decode_fallbacks`) for the `LARQL_GPU_DIAG` surface.

## Diagnostics

`LARQL_GPU_DIAG=1` now surfaces, via `device_info()`:

```
cuda resident-KV decode: uses=<N>, fallbacks=<M>, resident_rate=<%>
```

## Tests added (runtime-gated; scaffold early-return)

- `resident_kv_decode_lockstep_after_prefill_then_one_decode`
- `resident_kv_decode_lockstep_across_multi_token_decode`
- `resident_kv_reset_clears_both_caches`
- `resident_kv_truncate_keeps_lockstep`
- `resident_kv_decode_matches_cpu_reference_after_prefill`
- `resident_kv_multi_token_decode_matches_cpu_reference`
- `resident_kv_decode_reads_only_valid_rows`
- `resident_kv_fallback_to_upload_when_no_cache`

## What a hardware run must still confirm

Before marking B1 (in `docs/cuda-vulkan-completion-plan.md`) complete:

```
CUDA_VISIBLE_DEVICES=0 cargo test -p larql-compute-cuda --features cuda hardware_probe -- --nocapture
CUDA_VISIBLE_DEVICES=0 cargo test -p larql-compute-cuda --features cuda decode_token_matches_cpu_reference_when_runtime_available -- --nocapture
CUDA_VISIBLE_DEVICES=0 cargo test -p larql-compute-cuda --features cuda multi_token_decode_matches_cpu_reference -- --nocapture
CUDA_VISIBLE_DEVICES=0 cargo test -p larql-compute-cuda --features cuda resident_kv -- --nocapture
CUDA_VISIBLE_DEVICES=0 LARQL_GPU_DIAG=1 cargo test -p larql-compute-cuda --features cuda -- --nocapture
```

(`--features cuda` is the parent-crate flag; `larql-compute-cuda` itself has no
`cuda` feature — native code is always compiled, cudarc dynamic-loads.)

Record the ASTAB-001 `max_abs` after the resident-KV change (tolerance is
unchanged at 1e-3) and confirm diagnostics report `uses>0` for eligible layers.

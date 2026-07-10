# CUDA resident-KV decode attention (GPU-006) — 2026-07-09

> **Status: COMPLETE — hardware-validated on RTX 3090 (sm_86, NVRTC 12.4).**
>
> Re-validation of `feat/cuda-resident-kv-decode-attention` (commit `ec95287d`)
> on `3090rig` (RTX 3090, 24 GB VRAM, driver 550.163.01): **154/154 tests green
> with default settings** (no env overrides) — `hardware_probe` (21 native
> kernels loaded including `decode_attention`), ASTAB-001 decode parity
> (`decode_token_matches_cpu_reference`, `multi_token_decode_matches_cpu_reference`),
> all 8 `resident_kv` tests, and the full suite under `LARQL_GPU_DIAG=1`
> (153 lib + 1 probe, 0 failed, 2.72s). The implementation was developed and
> scaffold-validated on a no-CUDA host in the implementing session; this note
> was flipped from PENDING to COMPLETE after the hardware run confirmed parity.

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

## Hardware confirmation (RTX 3090, sm_86, NVRTC 12.4)

Re-validation of `feat/cuda-resident-kv-decode-attention` at commit `ec95287d`,
default settings (no env overrides):

```
cargo test -p larql-compute-cuda hardware_probe -- --nocapture               # PASS
cargo test -p larql-compute-cuda decode_token_matches_cpu_reference_when_runtime_available  # PASS
cargo test -p larql-compute-cuda multi_token_decode_matches_cpu_reference                    # PASS
cargo test -p larql-compute-cuda resident_kv -- --nocapture                  # 8/8 PASS
LARQL_GPU_DIAG=1 cargo test -p larql-compute-cuda -- --nocapture             # 153 lib + 1 probe
```

(Note: `larql-compute-cuda` has no `cuda` Cargo feature — CUDA is compiled
unconditionally via cudarc, which dynamic-loads. `--features cuda` is the
parent-crate flag and errors on this crate; omit it.)

**Result: 154/154 green, 0 failed, 0 ignored (2.72s).** ASTAB-001 decode parity
within the unchanged 1e-3 tolerance; the 8 `resident_kv` tests run the resident
path (a 40-token prompt clears the `DECODE_ATTN_NATIVE_MIN_WORK = 8192` work
gate: first decode `work = 4·41·64 = 10496`). B1 is marked complete.

### Test bug found and fixed during validation

The first hardware pass found 5 of 8 `resident_kv` tests failing under default
settings: the original 3-token prompt gave `work = 4·4·64 = 1024 < 8192`, so
`host_attention_block_device` bailed and the resident path never ran — the
device-cursor / diag-counter assertions failed. This was a **test bug**, not an
implementation bug (the implementation was confirmed correct: lowering the
threshold made all 8 pass). Fixed by switching to a 40-token prompt that clears
the gate; no production code or threshold changed. Commit `ec95287d`.

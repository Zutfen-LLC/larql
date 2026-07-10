# HANDOFF

## Goal

Resume the CUDA + Vulkan backend implementation for LARQL.

## Current Status (verified)

Four sessions of work have landed:

- **Session 1** (original scaffolding): workspace/feature plumbing, explicit backend selection APIs, CUDA/Vulkan sibling crates as compileable scaffolds, partial CLI migration. Could not compile-verify (no Rust toolchain on PATH).
- **Session 2** (verification + repair + finish CLI): brought up `cargo`/`rustc` (rustup, off-PATH), ran `cargo check`, fixed the one real compile breakage, finished the `shannon` CLI migration, and fixed three test/lint issues so the affected crates are green under `cargo test` and `cargo clippy -- -D warnings`.
- **Session 3** (capability honesty + walk backend dispatch): reconciled CUDA/Vulkan scaffold capability reporting so delegated CPU/reference methods remain callable for parity tests but no longer advertise native `QuantMatVec`/`F32Gemv`/`F16Gemv`/`Q4_K` support. `walk`'s Q4 predict/generate path now constructs the requested backend generically and gates the fused fast path on `PrefillQ4 + DecodeToken + Q4_K`; `--backend auto` falls back to CPU when only scaffolds are present, while explicit CUDA/Vulkan fail loudly until native kernels land.
- **Session 4** (CUDA runtime bring-up + first native kernel): wired `cudarc` into `larql-compute-cuda`, added an optional dynamic CUDA/NVRTC bootstrap, caught missing-`libcuda` probe panics so non-CUDA hosts still degrade cleanly to the scaffold path, and landed a first native `q4k_matvec` kernel launch path behind the existing CPU fallback. Capability reporting was intentionally left conservative: CUDA still does **not** advertise native Q4/decode support until more of Phase 4 is real.
- **Session 5** (CUDA: two more native kernels): added native `q6k_matvec` and `q4k_matmul` CUDA kernels alongside the existing `q4k_matvec`, wired them through `CudaRuntime` (one combined NVRTC module holding all three entry points) → `CudaBackend` → `trait_impl`, each behind its CPU fallback. Added runtime-gated parity tests for both. Capability reporting stays conservative (still `supports_quant(Q4_K) == false` etc.). Also applied a `cargo fmt --all` pass that folded in the prior session's pending reformatting.
- **Session 6** (CUDA: remaining k-quant kernels): added native `q6k_matmul` and `q4k_dual_matvec` CUDA kernels. `q4k_dual_matvec` is wired through the existing `QuantMatVec::q4k_dual_matvec` trait method (native-then-CPU fallback) with a delegate parity test plus a runtime-gated native parity test. `q6k_matmul` is loaded into the NVRTC module and exposed via `CudaBackend::native_q6k_matmul`, but is **not yet routed through a trait method** — there is no `q6k_matmul` on `QuantMatVec` (the amortised Q6_K matmul is currently a CPU-only free function `q6k_matmul_into` called from `ffn/weight.rs::quant_matmul`); it's staged with `#[allow(dead_code)]` for the prefill-kquant backend-routing slice, parity-verified by a runtime-gated test. Combined NVRTC unit now holds five entry points. Capability reporting unchanged (still conservative: `supports`/`supports_quant` all false).
- **Session 7** (route staged `q6k_matmul` through a trait method + backend-aware `quant_matmul`): added a `QuantMatVec::q6k_matmul` trait method (default `None`) — the Q6_K twin of `q4k_matmul`. Implemented it on `CpuBackend` (wraps `q6k_matmul_into`), routed CUDA's `q6k_matmul` through the trait (native-then-CPU fallback) and dropped the `#[allow(dead_code)]` on `native_q6k_matmul`/`launch_q6k_matmul` so the staged kernel is now live, and added a Vulkan delegate. Made `ffn/weight.rs::quant_matmul` + `quant_proj` backend-aware: both take `Option<&dyn ComputeBackend>`; the Q6_K arm first tries the backend's `q6k_matmul` and falls back to the CPU free function on `None`. Threaded the backend through the attention `gpu.rs` Q/K/V/O `quant_proj` calls (so the Q6_K V projection routes through the backend when present); `Q4kMatmulFfn` (no backend field) keeps passing `None`, preserving its current behaviour. Added three parity tests: a `larql-compute` trait test (CPU matches free function + default returns `None`), a CUDA always-runs delegate test, and a CUDA runtime-gated native-via-trait test. Capability reporting unchanged (still conservative).
- **Session 8** (CUDA: dense f32/f16 GEMV kernels): added native `f32_gemv` and `f16_gemv` CUDA kernels (one row per thread; the f16 kernel reuses the larql_decode_f16 helper union shared across the module). Wired both into `CudaRuntime` (the combined NVRTC module now holds seven entry points) → `CudaBackend::native_f32_gemv`/`native_f16_gemv` → `MatMul::f32_gemv`/`f16_gemv` trait methods, each behind its CPU fallback. The `f32_gemv` trait routing only takes the native path when the `ArrayView2` is row-major contiguous (`as_slice()`); non-contiguous views fall through to the CPU reference. `f32_gemv_topk1`/`f16_gemv_topk1`/`f16_gemv_topk` inherit the routing automatically (they call through `f32_gemv`/`f16_gemv`). Added four parity tests: two always-runs delegate tests and two runtime-gated native parity tests. Capability reporting stays conservative (still `supports(F32Gemv)`/`supports(F16Gemv)` == false; advertisement waits until the prefill/decode path is native, not just the standalone GEMVs).
- **Session 8b** (code-review fixes to Session 8): addressed four review findings. (1) f32 + f16 device kernels now index in 64-bit (`unsigned long long`) so large vocab heads (`n*k > 2^32` for f32, `2*n*k > 2^32` for f16) no longer wrap a 32-bit row/element offset and read OOB global memory. (2) Both launchers now reject dims exceeding `u32::MAX` (the kernel-arg width) before upload, falling back to CPU instead of truncating the dispatch. (3) The `f32_gemv`/`f16_gemv` trait routing is flop-threshold-gated (`GEMV_FLOP_THRESHOLD = 500M` flops, mirroring Metal's `calibration::DEFAULT_FLOP_THRESHOLD`): below it the native kernel is skipped and the caller keeps the zero-copy mmap CPU `matmul_transb` path — preventing a per-token GB-scale weight re-upload regression on the lm_head decode hot path (CUDA is the first `auto` candidate on Linux). (4) `f32_gemv_topk1`/`f16_gemv_topk1` now return `None` (the trait default) so greedy-decode top-1 keeps the CPU fast path instead of the un-fused full-upload + full-readback + CPU argmax; `f16_gemv_topk` inherits the gate via `f16_gemv`. Tests updated: the old topk1/delegate tests converted to assert the gated `None` contract; added runtime-gated dim-overflow rejection tests for both kernels.
- **Session 9** (CUDA: native Q4_0 matvec + vecmat kernels): added native `q4_matvec` (Q4_0 × Q8, one row per thread, decodes the 18-byte Q4_0 block: 2-byte f16 scale + 16 packed nibble bytes, value = nibble - 8) and `q4_vecmat` (Q4_0 vector-matrix, one output column per thread gather across all `intermediate` rows, with the CPU's `|act| < 1e-10` zero-skip) CUDA kernels. Wired both into `CudaRuntime` (the combined NVRTC module now holds nine entry points) → `CudaBackend::native_q4_matvec`/`native_q4_vecmat` → `QuantMatVec::q4_matvec`/`q4_vecmat` trait methods, each behind its CPU fallback. Both device kernels index row offsets in 64-bit and the host launchers reject dims exceeding `u32::MAX` before upload (mirroring the Session 8b f32/f16 GEMV overflow guards). Added six parity tests: two always-runs delegate tests (pin the CPU fallback contract on every host), two runtime-gated native-via-trait tests, and two runtime-gated dim-overflow rejection tests. Capability reporting stays conservative (still `supports`/`supports_quant` all false).
- **Session 10** (CUDA: KV cache lifecycle + native `kv_append` kernel): landed the foundation of the Phase 4 prefill/decode path — a device-side KV cache. New `crates/larql-compute-cuda/src/kv_cache.rs` mirrors Metal's `LayerKVCache`/`KVCache` using `cudarc::driver::CudaSlice<f32>` owned device buffers (held via `Arc<CudaStream>`), supporting uniform and heterogeneous per-layer shapes (Gemma 4 sliding/global) with `new_per_layer`/`grow_to_shapes`/`has_shape_mismatch`/`clear`/`current_len`. The cache is wired into `CudaBackend` via `Mutex<Option<CudaKVCache>>` (interior mutability, mirrors Metal's `kv_cache: Mutex<Option<KVCache>>`; this required dropping the `Clone` derive on `CudaBackend` — Metal's `MetalBackend` isn't `Clone` either, and no caller clones it). Added a native `kv_append` CUDA kernel (one thread per K/V element across all `seq_len` rows; 64-bit slot-offset guard; `row_elems` passed as a precomputed 32-bit value so the device-side multiplication cannot wrap) in `ops.rs`, registered in the combined NVRTC module (now ten entry points) and exposed via `CudaRuntime::launch_kv_append` → `CudaBackend::native_kv_append` (with `pos > u32::MAX`, `row_elems > u32::MAX`, `seq_len * row_elems` overflow, and out-of-bounds-slot rejection — all in `checked` arithmetic). `populate_kv_layer` uploads the full contiguous K/V block in a single host→device transfer and launches one kernel over all rows + one sync (no per-row stalls). Implemented the `DecodeBackend` KV lifecycle methods on `CudaBackend` — `has_kv_cache` (true only with a runtime *and* an allocated cache), `preallocate_kv_cache_per_layer` (lazy device alloc; reallocates from scratch when an existing layer's `max_seq` is undersized or geometry mismatches; no-op + `has_kv_cache=false` on the scaffold path), `reset_kv_cache`/`kv_cache_len`/`truncate_kv_cache` (cursor-only; `truncate` sets `current_len` unconditionally per layer to restore lockstep, matching Metal; device buffers not zeroed), and `populate_kv_layer` (single batched `native_kv_append`; `checked_mul` geometry guard returns early on overflow/short data; no-op when no runtime/cache). Mutex poisoning is recovered via a `lock_kv_cache` helper (`unwrap_or_else(into_inner)`) instead of `.expect`, preserving the documented CPU-fallback contract. Capability reporting stays conservative (`supports(DecodeToken)` still false — attention-over-cache and the fused decode/prefill pipelines are the next slices). Added 14 tests: 5 always-runs scaffold-fallback contract tests + 2 runtime-gated native lifecycle tests (preallocate → `has_kv_cache` flips; `populate_kv_layer` advances the cursor and `truncate`/`reset` roll it back) + 1 runtime-gated realloc-on-larger-`max_seq` test + 4 runtime-gated overflow/capacity rejection tests (pos > u32, row_elems product, seq_len*row_elems block overflow, out-of-bounds slot) + 2 unit tests for `KvCacheError` Display and shape-mismatch detection. A local code review of this slice found and fixed: three wrapping-`usize` bounds-overflow paths that could cause device OOB writes, the per-row upload+sync prefill hot-path regression, the `max_seq`-ignored reallocation, the `truncate` lockstep divergence from Metal, the `.expect`-on-poisoned-mutex abort, and three dead-code items (unused `CudaKVCache::new`, unused `row_elems()`, the then-unused `max_seq` field — now read by the realloc check).
- **Session 11** (CUDA: host-orchestrated fused decode/prefill pipeline + capability advertisement): landed the Phase 4 fused `decode_token`/`decode_token_with_state_dump_masked`/`prefill_kquant` on `CudaBackend` via a new `crates/larql-compute-cuda/src/pipeline.rs` module. This is a **host-orchestrated pipeline** (not a single-command-buffer fused kernel): every Q/K/V/O + gate/up/down projection dispatches through the native CUDA q4k/q6k matvec (decode) / matmul (prefill) kernels via the `QuantMatVec` trait (native-then-CPU fallback, the Session 4-9 path); elementwise ops (RMSNorm / QK-norm / RoPE / GQA softmax / V-norm / GEGLU/SiLU/GELU activation / residual adds / per-layer scalar) run on host using the shared `larql_compute` primitives (same f64 accumulation as the CPU reference → numerically identical). A **host-side KV mirror** (`Mutex<Vec<(Array2<f32>, Array2<f32>)>>` per layer, `pub(crate) host_kv` on `CudaBackend`) holds the `[len, kv_dim]` K/V the host attention reads; the device `CudaKVCache` from Session 10 is populated in lockstep via `populate_kv_layer` so the `DecodeBackend` lifecycle contract stays consistent. `decode_token` reads `kv_cache_len_native()` for the RoPE position; `prefill_kquant` uses the amortised `q4k_matmul`/`q6k_matmul` kernels across all `seq_len` positions and causal `gqa_attention_with_weights` (with the caller's `softcap`). The pipeline bails to `None` (caller's CPU fallback) for layer features it doesn't handle yet: MoE / PLE / remote-FFN / non-k-quant formats. **Capability advertisement is now live when a runtime is present**: `supports_quant(Q4_K | Q4_KF | Q6_K) == true`, `supports(QuantMatVec | DecodeToken | PrefillQ4) == true`; the scaffold path (no device) keeps everything `false` so `fused_prefill`/`fused_decode_step` bail to the CPU reference. This unblocks the `walk`/`bench` fast paths and the `auto` policy on Linux (CUDA is the first `auto` candidate on non-macOS). Added 8 tests: scaffold-fallback fused-pipelines-return-None, native-capability advertisement, prefill parity vs `predict_kquant_prefill`, decode parity vs `predict_kquant_decode_step_direct`, and `decode_token_with_state_dump_masked` `Full`/`HOnly` mask conformance — all runtime-gated except the scaffold-fallback + capability tests which run on every host. `reset_kv_cache` now also clears the host mirror. No new CUDA kernels (reuses the ten from Sessions 4-10). **A local code review of this slice found and fixed five issues:** (1) **multi-token decode RoPE position freeze** — `decode_token` derived `pos` from the *device* cursor (`kv_cache_len_native`), which `decode` never advances (only `prefill_kquant` populates the device cache), so the second+ decode token reused the post-prefill position and garbled RoPE; fixed by deriving `pos` from the host KV mirror length (`host_kv_len`, the source of truth for the host attention) and making `kv_cache_len`/`truncate_kv_cache` operate on the host mirror. Added a multi-token decode parity test (4 steps, asserts each matches CPU + `kv_cache_len` advances). (2) **MoE deploy regression** — advertising `PrefillQ4`/`DecodeToken` made `backend_supports_fused_q4_pipeline(cuda)` return true, so on Linux `auto` hybrid-MoE models (Gemma 4 26B A4B) entered the CUDA path where `prefill_kquant` bails to `None` → `prefill_failed` hard error instead of CPU fallback; fixed in `larql-inference/layer_graph/generate/gpu/mod.rs` by routing `is_hybrid_moe() && !backend.supports(DecodeMoe)` to `generate_via_cpu_q4k` (Metal advertises `DecodeMoe` and stays on GPU). (3) **scaled-RoPE divergence** — the pipeline hardcoded RoPE `position_divisor=1.0`/`llama3=None`, diverging from the CPU reference for Gemma 3 global layers (divisor 8) and llama3-rope models; fixed by adding `rope_position_divisor`/`rope_llama3_scaling` fields to `FullPipelineLayer` (populated in `build_arch_params` from the effective overrides, mirroring `rope_base`) and threading them into both decode + prefill `apply_rope_partial_at_full` calls; added a rope-scaled prefill parity test (`make_test_q4k_weights_rope_scaled`). (4) **panic-safety gaps** — prefill/decode activation indexed projection outputs without length validation (a short/overflowed matvec/matmul return would panic instead of returning `None`), and `down_padded_activation` divided by `hidden` unguarded (`hidden==0` panics); added length checks (return `None`) + a `hidden==0` guard. (5) **dead code + Q4_KF misadvertisement** — the `GeluExact`/`ReLU` activation arms + a hand-rolled `erf` were unreachable (builders only emit Silu/GeluTanh); replaced with `unreachable!()` (mirrors Metal) and dropped `erf`+its test. `supports_quant`/`super_block_bytes`/`quant_matmul` no longer advertise/route Q4_KF (the native q4k kernels decode 144-byte Q4_K super-blocks, not Q4_KF's 160-byte layout, and `build_pipeline_layers` never produces Q4_KF). Net test delta: +2 (rope-scaled prefill parity, multi-token decode parity), both runtime-gated.
- **Session 12** (CUDA: hybrid-MoE host pipeline): extended the host-orchestrated fused decode/prefill pipeline so hybrid-MoE layers (Gemma 4 26B-A4B) no longer bail to `None`→CPU. New `host_ffn_block_moe_decode` / `host_prefill_ffn_block_moe` in `pipeline.rs` mirror the structure of `larql-inference::moe_ffn_block_cpu_with_index`: the dense slab (the existing `host_ffn_block`/`host_prefill_ffn_block`, native quant matvec/matmul projections) supplies `h1` (the dense delta = slab − residual); the substrate reference `larql_compute::cpu::ops::moe::cpu_moe_forward` supplies `h2` (the expert contribution); the two are combined via the substrate `outer_post_norm_residual` (with the dedicated `moe_outer_post_norm` selected by a new `moe_outer_norm` helper when `moe_combined_output_norm` is set). The per-layer scalar is now applied uniformly in the decode/prefill loops for dense **and** MoE (PLE is a no-op on 26B-A4B, so the scalar is the final step in both). The bail condition narrowed: `layer.moe.is_some()` no longer bails, while `ple_input_gate.is_some()` (needs the precomputed per-layer embedding input, not on the trait surface) and `ffn_is_remote` (needs a dispatch callback — only `decode_token_with_moe` carries one) still bail. Expert projections run on CPU via `cpu_moe_forward` (correctness-first; routing expert matvecs through the native CUDA kernels is the device-fusion follow-on). Added 5 tests: `moe_outer_norm_selection_matches_reference`, `moe_decode_block_matches_independent_composition` (host-runnable — locks the dense-delta + expert + outer-combine wiring against an independently-composed substrate reference), `moe_prefill_block_matches_independent_composition` (host-runnable, multi-position), `ple_and_remote_ffn_layers_bail_to_none` (host-runnable, via the pub(crate) host path so it runs on every host), and `moe_prefill_and_decode_run_through_trait_surface` (runtime-gated e2e smoke).
- **Session 13** (CUDA: first native elementwise kernels — RMSNorm): began the device-kernel-fusion follow-on to Session 11's host-orchestrated pipeline by moving the highest-frequency elementwise op off the host. Added two native CUDA kernels to `ops.rs`: `rms_norm` (body norm over a `[rows, cols]` matrix — the device twin of `larql_compute::residual::rms_norm_eps`; one thread-block per row with a f64 warp+block reduction in shared memory, then `out = x/rms * (offset + w[j])` with `w = 1.0` when `has_weight = 0` mirroring the `None`-weight arm; `eps` is a `double` arg so the `+ eps` happens at f64 precision exactly as on host) and `rms_norm_heads` (per-head norm over `[seq, num_heads*head_dim]` — the device twin of `rms_norm_heads`/`rms_norm_heads_no_weight`; one thread-block per (position, head); `has_weight = 0` selects the parameter-free path so a single entry point serves both CPU references). Both kernels index in 64-bit and the launchers reject dims exceeding `u32::MAX` before upload (mirroring the Session 8b/9 overflow guards); the block size is capped at 1024 threads (32 warps) since the reduction uses a fixed 32-slot shared array. Registered both in `CudaRuntime` (the combined NVRTC module now holds twelve entry points) → `CudaBackend::native_rms_norm`/`native_rms_norm_heads` (each `Ok(true)` on a native launch, `Ok(false)` on the scaffold path). Routed the pipeline's host norm path through them: `norm_2d`/`norm_1d`/`norm_2d_no_weight`/`rms_norm_heads_array` are now `pub(crate)` methods on `CudaBackend` that try the native kernel first and fall back to the substrate reference on `Ok(false)`/`Err`/non-contiguous views (the per-head path uses `DEFAULT_EPS = 1e-6`, matching the CPU `rms_norm_heads`/`rms_norm_heads_no_weight` hard-coding). Capability reporting unchanged (still conservative — elementwise kernels alone don't flip the `supports` advertisement). Added 8 tests: 4 host-runnable fallback-contract parity (norm_2d RmsNorm/LayerNorm arms, norm_2d_no_weight, rms_norm_heads_array weighted+no-weight — pin the CPU fallback on every host) + 4 runtime-gated native parity (body norm weighted/no-weight, per-head weighted/no-weight — no-op on this no-CUDA host). This is the first native elementwise step toward the fully-fused single-command-buffer pipeline (Metal's `decode/mod.rs` shape); the remaining host elementwise ops (RoPE / GQA softmax / V-norm / GEGLU/SiLU activation / residual adds) and the collapse of the per-projection htod/launch/dtoh round-trips are the follow-on.
- **Session 14** (CUDA: native activation kernels — GEGLU + standard): continued the device-kernel-fusion follow-on by moving the second-highest-frequency elementwise op (the FFN activation, which runs on every gate/up projection) off the host. Added four native CUDA kernels to `ops.rs`: `geglu_silu` (`out[i] = silu(gate[i]) * up[i]`, one thread per element, the device twin of `cpu::ops::geglu::geglu_silu` + the host `apply_activation_gated(Silu, …)` path), `geglu_gelu_tanh` (the GeluTanh twin; clamps the tanh argument to ±15 before `tanhf` to avoid NVIDIA `tanhf`'s `(expf(2y)-1)/(expf(2y)+1)` overflow at |y|≳44 — `tanhf(15)` differs from 1.0f by < 1e-13, so the clamp is a numerical-safe parity-preserving fix since the host Rust `tanh` saturates without overflow), `activation_silu` (standard non-gated `out[i] = silu(x[i])`, the `apply_activation_std(Silu, …)` path, used when `ffn_type == Standard`), and `activation_gelu_tanh` (the standard GeluTanh twin, same ±15 clamp). All four index with a 32-bit `unsigned int n` arg and the launchers reject `n > u32::MAX` before upload (mirroring the other elementwise kernels' overflow guards). Wired all four into `CudaRuntime` via two shared launch helpers — `launch_elementwise_binary` (gate, up → out, used by the two GEGLU kernels) and `launch_elementwise_unary` (input → out, used by the two standard activations) — so the upload/launch/sync/readback boilerplate lives in one place (a new `RuntimeError::context_concat` helper composes the per-kernel error strings, since the existing `context` takes a single `&'static str`). The combined NVRTC module now holds sixteen entry points. Exposed `CudaBackend::native_geglu_silu`/`native_geglu_gelu_tanh`/`native_activation_silu`/`native_activation_gelu_tanh` (each `Ok(true)` on a native launch, `Ok(false)` on the scaffold path). Routed the pipeline's FFN activation through them: new `pub(crate) apply_activation_gated_native`/`apply_activation_std_native` methods on `CudaBackend` try the native kernel first (gated on `ACTIVATION_NATIVE_MIN_ELEMS = 8192`, mirroring `NORM_NATIVE_MIN_ELEMS`) and fall back to the host `apply_activation_gated`/`apply_activation_std` reference on `Ok(false)`/`Err`/small inputs; the prefill FFN block (`host_prefill_ffn_block`, activating `seq_len * inter` elements) and the decode FFN block (`host_ffn_block`, activating `inter` per token) both now call the native-aware helpers. The `ACTIVATION_NATIVE_MIN_ELEMS` gate keeps the host reference on the small test fixtures (`inter=256`), preserving the existing `assert_eq!` composition-parity tests; real models with `inter` ≈ 9216 hit the native path on prefill (seq*inter ≫ 8192) and on decode (inter ≈ 9216 > 8192). Capability reporting unchanged (still conservative — elementwise kernels alone don't flip the `supports` advertisement). Added 9 tests: 4 host-runnable fallback-contract parity (gated SiLu/GeluTanh + standard SiLu/GeluTanh, below the 8192 gate, asserting `assert_eq!` against the host `apply_activation_*` reference), 4 runtime-gated native parity (above the gate, asserting `< 1e-5` tolerance since the device `expf`/`tanhf` may differ by ~1 ULP from the host `exp`/`tanh`; the GeluTanh tests include large-magnitude gates ±25 to exercise the device clamp boundary vs the host `tanh` saturation), and 1 runtime-gated dim-mismatch rejection test (mismatched gate/up lengths error rather than silently launching).

- **Session 15** (CUDA: native residual-add kernel): continued the device-kernel-fusion follow-on by moving the residual add (`out = h + b_scale * x`, which runs twice per layer — post-attention and post-FFN, for both dense and MoE) off the host. Added one native CUDA kernel to `ops.rs`: `residual_add` (`out[i] = h[i] + b_scale * x[i]`, one thread per element; the device twin of the host `add_residual` helper — fuses the `b_scale == 1.0` (`h + x`) and scaled (`h + b_scale * x`) arms into the single `h + b_scale * x` form, which is numerically identical, so no branch is needed on the device). Indexes with a 32-bit `unsigned int n` arg and the launcher rejects `n > u32::MAX` before upload (mirroring the other elementwise kernels' overflow guards); carries an extra `b_scale` scalar arg so it's a self-contained launcher (not routed through the binary/unary shared helpers). Wired into `CudaRuntime` (the combined NVRTC module now holds seventeen entry points) → `CudaBackend::native_residual_add` (`Ok(true)` on a native launch, `Ok(false)` on the scaffold path). Routed the pipeline's residual path through it: a new `pub(crate) add_residual_native` method on `CudaBackend` tries the native kernel first (only when both inputs are contiguous `Array2` slices AND the element count clears `RESIDUAL_NATIVE_MIN_ELEMS = 8192`) and falls back to the host `add_residual` reference otherwise; all eight residual call sites in the decode/prefill attention+FFN blocks now go through it. The host `add_residual` free fn is kept `pub(crate)` as the parity oracle. Capability reporting unchanged (still conservative — elementwise kernels alone don't flip the `supports` advertisement). Added 3 tests: a host-runnable fallback-contract parity (small input below the gate exercises the host reference for both the unit and scaled arms), a runtime-gated native parity (large input above the gate; exact equality since residual add is pure IEEE-754 add/mul with `fmad` disabled at NVRTC compile time), and a runtime-gated dim-overflow rejection.

- **Session 16** (CUDA: native RoPE kernel): continued the device-kernel-fusion follow-on by moving the fourth elementwise op — Rotary Position Embedding (RoPE), which runs on Q and K at every attention block (twice per layer) — off the host. Added one native CUDA kernel to `ops.rs`: `rope` (the device twin of `larql_compute::attention::rope::apply_rope_partial_at_full`; one thread per output element over the full `[seq_len, num_heads*head_dim]` Q/K tensor; split-half pairing — channels in `[0, half_rotary)` write `x0*cos(theta) − x1*sin(theta)`, channels in `[half_rotary, 2*half_rotary)` write `x0*sin(theta) + x1*cos(theta)`, channels `>= 2*half_rotary` pass through unchanged; `2*half_rotary` (not `rotary_dim`) bounds the rotary region so an odd `rotary_dim` leaves its trailing channel as pass-through, exactly mirroring the host loop which iterates `i in 0..half_rotary` and writes two channels per `i`). The `inv_freq[half_rotary]` frequency array is precomputed on the host with the exact formula the reference uses (`1 / base^(2i/rotary_dim)`, with HF's `llama3` wavelength-band rescaling applied via the shared substrate `apply_llama3_inv_freq`) and uploaded as `double`, so the device computes `theta`/`cos`/`sin` in f64 and only narrows cos/sin to f32 — matching the reference's `theta.cos() as f32`; with `fmad` disabled at NVRTC compile time the f32 rotation arithmetic is identical to the host path. The kernel indexes the flat element offset and the row/head/channel decomposition in 64-bit (`unsigned long long`), and the host launcher (`launch_rope`) guards `total > u32::MAX` / per-dim `> u32::MAX` / `half_rotary == 0` / `inv_freq.len() != half_rotary` / shape-length mismatches before upload. Wired into `CudaRuntime` (the combined NVRTC module now holds eighteen entry points) → `CudaBackend::native_rope` (`Ok(true)` on a native launch, `Ok(false)` on the scaffold path). Routed the pipeline's RoPE path through it: a new `pub(crate) rope_native` method on `CudaBackend` builds `inv_freq` identically to the reference (so `llama3` scaling is handled before upload), tries the native kernel first (gated on a contiguous `Array2` slice AND the element count clearing `ROPE_NATIVE_MIN_ELEMS = 8192`), and falls back to `apply_rope_partial_at_full`; all four RoPE call sites in the decode/prefill attention blocks (Q and K, each for decode + prefill) now go through it. Capability reporting unchanged (still conservative). Added 3 tests: a host-runnable fallback-contract parity (small input below the gate, partial-rotation fraction so both the rotary region and the pass-through tail are exercised, non-zero position offset), a runtime-gated native parity (large input above the gate; the pass-through tail is bit-identical, the rotary channels agree to ≤ 1e-5 since the device's double `cos`/`sin` are a different libm than the host's — both compute theta/cos/sin in f64 and narrow to f32, so only the transcendentals can differ by a few ULP), and a runtime-gated invalid-shape rejection test (mismatched `inv_freq` length + shape/x length mismatch).

- **Session 17** (CUDA: native fused decode-attention kernel): moved the decode-step attention math itself off the host — the first *non-elementwise* device kernel (a reduction-heavy fused kernel, not a one-thread-per-element op). Added one native CUDA kernel to `ops.rs`: `decode_attention` (the device twin of `larql_compute::attention::decode::gqa_attention_decode_step`; **one thread-block per query head**, block size `DECODE_ATTN_BLOCK = 256`; the block collaboratively fuses QKᵀ → scale (+ optional softcap) → softmax → weighted-V in a single launch over the full `[total_len, kv_dim]` KV cache, collapsing the host `k_block.dot(&q_row)` + softmax + `v_block.t().dot(&scores)` per-head loop into one kernel). Five phases per head: (1) each thread computes strided scores `s = dot(K[i],Q)*scale` (`+tanhf(s/softcap)*softcap` when `has_softcap!=0`), stored to a `num_q*total_len` device scratch; (2) f32 max via a 256-slot `__shared__` tree reduction (stride loop + warp-style halving); (3) `exp((s-max) as f64)` narrowed to f32 for storage + f64 sum reduction over a 256-slot `__shared__ double` array (matching the reference's f64 sum accumulation exactly); (4) normalize `scores *= inv_sum` **before** the dot (mirrors the reference's normalize-then-`v_block.t().dot` rounding order, so the f32 weighted-V matches); (5) weighted-V `out[d] = sum_i scores[i]*V[i]` in f32, `d = 0..head_dim` order. With `fmad` disabled at NVRTC compile time the f32 arithmetic matches the reference; the only divergence is the device `exp`/`tanhf` libm (parity-gated ≤ 1e-4). GQA via `kv_h = h/reps` (`kv_off = kv_h*head_dim`); K/V/scores base offsets index in 64-bit, and `launch_decode_attention` guards `q_len`/`kv_len`/`score_len`/all dims `<= u32::MAX`, `reps >= 1`, `total_len >= 1`, and shape/length mismatches before upload (allocates a `scores` scratch + output per call; the fully-fused single-command-buffer pipeline will fold the scratch into a persistent device buffer). Wired into `CudaRuntime` (the combined NVRTC module now holds **nineteen** entry points; `shared_mem_bytes = 256*4 + 256*8`) → `CudaBackend::native_decode_attention` (`Ok(true)` on a native launch, `Ok(false)` on the scaffold path). Routed the pipeline's decode attention through it: a new `pub(crate) decode_attention_native` method on `CudaBackend` tries the native kernel first (only when Q/K/V are contiguous `Array2` slices AND the attention work `num_q*total_len*head_dim >= DECODE_ATTN_NATIVE_MIN_WORK = 8192` — short contexts keep the host reference since there's no fusion benefit, only transfer+sync overhead) and falls back to `gqa_attention_decode_step` otherwise; the single decode-step attention call site now goes through it. **This completes the device-elementwise + decode-attention fusion follow-on** — the four highest-frequency elementwise ops (RMSNorm/activation/residual/RoPE, Sessions 13–16) plus the decode attention itself are now off the host. Capability reporting unchanged (still conservative — this is the *same* `DecodeToken` path Session 11 already advertises; the native attention is a perf refinement of that host-orchestrated pipeline, not a new capability). Added 3 tests: a host-runnable below-gate parity (short context, GQA `reps>1`, multi-row KV cache; bit-exact since it takes the host reference), a runtime-gated native parity (above-gate; scale-relative ≤ 1e-4 bound for the device `exp`/`tanhf` libm difference), and a runtime-gated invalid-shape rejection (mismatched q/kv lengths + `reps==0`).

- **Session 18** (CUDA: native fused prefill (seq×seq) attention kernel): landed the prefill twin of the Session 17 `decode_attention` kernel — the second *non-elementwise* fused device kernel and the last attention primitive the host-orchestrated prefill pipeline was still running on the host. Added one native CUDA kernel to `ops.rs`: `prefill_attention` (the device twin of `larql_compute::attention::gqa::gqa_attention_capture` reached via `gqa_attention_with_weights` — the symmetric causal GQA path; **one thread-block per `(query head h, query position qi)`** via `gridDim = (num_q, seq_len)`, block size `PREFILL_ATTN_BLOCK = 256`; the decode kernel attends over the *full* KV cache, prefill is causal so each block only fuses QKᵀ → scale (+ optional softcap) → softmax → weighted-V over `causal_len = qi + 1` keys). Same five-phase structure as `decode_attention` but bounded by `causal_len` and using **dynamic `extern __shared__` memory** (`3072 + seq_len*4` bytes) for a block-local `scores` scratch (`shm_sum[256]` doubles | `shm_max[256]` floats | `scores[seq_len]` floats) instead of the decode kernel's `num_q*total_len` global scratch — each `(h, qi)` block is independent so there's no `seq_len*num_q*seq_len` global allocation. f32 QKᵀ/weighted-V + f64 softmax sum, normalize-before-dot (mirroring the reference's rounding order); `q_dim = num_q*head_dim` computed in 64-bit so the Q/out row offset can't wrap. With `fmad` disabled at NVRTC compile time the f32 arithmetic matches the reference; the only divergence is the device `exp`/`tanhf` libm (parity-gated ≤ 1e-4 relative). The kernel indexes Q/K/V/out rows in 64-bit and `launch_prefill_attention` guards `q_len`/`kv_len`/all dims `<= u32::MAX`, `reps >= 1`, `seq_len >= 1`, shape/length mismatches, AND the dynamic shared-mem budget (`3072 + seq_len*4 <= 48 KB`, i.e. `seq_len ≤ ~11520`) before upload — longer prompts fall back to the host reference (the single-command-buffer follow-on will revisit larger shared via `cudaFuncSetAttribute`). Wired into `CudaRuntime` (the combined NVRTC module now holds twenty entry points) → `CudaBackend::native_prefill_attention` (`Ok(true)` on a native launch, `Ok(false)` on the scaffold path). Routed the prefill pipeline's attention through it: a new `pub(crate) prefill_attention_native` method on `CudaBackend` tries the native kernel first (only when Q/K/V are contiguous `Array2` slices AND the work `seq_len*num_q*seq_len*head_dim ≥ PREFILL_ATTN_NATIVE_MIN_WORK = 8192`) and falls back to `gqa_attention_with_weights` on `Ok(false)`/`Err`/short prompts; the single causal-attention call site in `host_prefill_attention_block` now goes through it. Capability reporting unchanged (still conservative — the attention kernel alone doesn't flip the `supports` advertisement; prefill already advertises `supports(PrefillQ4)` via the Session 11 host pipeline). Added 3 tests: a host-runnable below-gate bit-exact parity (GQA + causal masking, multi-row Q/K/V), a runtime-gated native parity (≤ 1e-4 relative, exercises a softcap so the device `tanhf` path is covered), and a runtime-gated invalid-shape rejection (mismatched lengths + `reps==0`).

- **Session 19** (CUDA: persistent device weight cache — first slice of the round-trip collapse): began the headline remaining Phase 4 perf work (collapsing the per-projection htod/launch/dtoh round-trips toward Metal's single-command-buffer shape) by eliminating the largest, most repeated transfer first — the weight re-uploads. Historically every native projection launcher re-uploaded its full weight matrix via `CudaStream::clone_htod` on **every** call; for decode that is the dominant host→device cost, since the same per-layer `wq`/`wk`/`wv`/`wo`/`gate`/`up`/`down` weight matrices (GB-scale on a 4–26B model) were re-uploaded for every generated token even though weights never change. Added a new `crates/larql-compute-cuda/src/weight_cache.rs` module: a `WeightCache` holding two typed maps (`Mutex<HashMap<WeightKey, Arc<CudaSlice<u8>>>>` for quant/f16 byte weights, `…<f32>` for dense weight matrices) keyed on the host slice's `(pointer, element-count)`. `get_or_upload_bytes`/`get_or_upload_f32` upload once and return a cheaply-cloned `Arc<CudaSlice<T>>` handle (each `CudaSlice` already owns an `Arc<CudaStream>`, mirroring the KV cache's storage discipline); subsequent calls with the same slice address are a cache hit (one locked-map lookup + `Arc::clone`) that skips the upload entirely. **Only weights are cached** — activations (`x`, projections, KV rows, the per-token Q8 quantization of the input in `q4_matvec`/`q4_vecmat`) keep the fresh `clone_htod` path exactly as before. Cache-key soundness: LARQL weights are zero-copy `mmap`'d slices with a stable address for the vindex's lifetime, so `(ptr, len)` is exact within one backend instance; the cache is flushed at each `DecodeBackend::reset_kv_cache` (the documented "start of a new prompt/generation" boundary) so a backend reused across vindex loads can't serve a stale buffer mapped at a recycled address (an ABA guard) — flushing per-generation still captures the dominant reuse win (weights uploaded during prefill's projections are reused across every decode token of that generation). Wired the `weight_cache` field onto `CudaRuntime` and routed the weight arg of all **nine** weight-bearing launchers through it — `launch_q4k_matvec`, `launch_q6k_matvec`, `launch_q4k_matmul`, `launch_q6k_matmul`, `launch_q4k_dual_matvec` (both `q4k_a` + `q4k_b`), `launch_f32_gemv`, `launch_f16_gemv`, `launch_q4_matvec`, `launch_q4_vecmat` — each swapping `clone_htod(weight)` for `weight_cache.get_or_upload_…(&self.stream, weight)` and passing `&*w_dev` to the launch builder (no signature changes; activations/output still fresh). Added `flush_weight_cache` on the runtime (called from `CudaBackend::reset_kv_cache_native`) + a `#[cfg(test)]` `weight_cache_stats`/`weight_cache_stats` diagnostic chain (hit/miss counters). Capability reporting unchanged (this is purely a transfer optimization of the already-advertised paths — no new `supports` flips). Added 8 tests: 5 host-runnable `weight_cache` unit tests (byte/f32 key distinctness + same-slice stability + length-distinction + empty-key well-formedness + `flush` clears maps without a device) + 2 runtime-gated CUDA tests (`weight_cache_reuses_bytes_across_launches` asserts a second `q4k_matvec` with the same weight slice registers a hit AND matches the CPU reference for a *different* activation — proving the cached buffer is correct, not stale; `reset_kv_cache_flushes_weight_cache` asserts a post-reset launch re-uploads) + the host-runnable flush test counted above. No kernel count change (still twenty native kernels) — this collapses transfers, not kernels.

  **Session 19 review fixes** (4 findings from a local review): (1) **ABA scope corrected** — the per-`reset_kv_cache` flush only covers decode/prefill (generation boundary); the browse path (`f16_gemv`/`f32_gemv` via `larql-vindex` DESCRIBE/WALK/SELECT gate/lm-head KNN) never resets the KV cache, so a backend reused across vindex loads for browse could read a prior model's cached weights at a recycled mmap address. Added a public `CudaBackend::flush_weight_cache()` escape hatch (call at the vindex-rebind boundary) and corrected the module + reset docstrings to honestly scope the guarantee (unconditionally correct for the normal one-backend-per-vindex case; decode/prefill flushed per generation; browse cross-vindex reuse needs an explicit `flush_weight_cache()`). (2) **stats lock removed** — replaced `Mutex<CacheStats>` (acquired on every weight launch, read only under `#[cfg(test)]`) with four lock-free `AtomicU64` (`fetch_add(Relaxed)`); `CacheStats` is now a test-only snapshot struct returned by `stats()`, so the decode hot path pays no lock cost for unobservable-in-release counters. (3) **single-lock miss path** — the miss path previously dropped the map lock across the expensive upload then re-locked to insert (a TOCTOU allowing racing duplicate uploads + transient 2× VRAM under future concurrency); it now uploads while still holding the map lock so a concurrent miss of the same key finds the entry populated. (4) **dead counter consumed** — `CacheStats::float_misses` was written by `get_or_upload_f32` but never read; added `weight_cache_reuses_floats_across_launches` (runtime-gated) which asserts a float miss then a float hit via `native_f32_gemv` (bypassing the trait's flop gate) and also exercises the new public `flush_weight_cache`, giving the f32 cache path its first coverage.

- **Session 20** (CUDA: device-resident activation chain for the prefill FFN block — second slice of the round-trip collapse): continued the headline Phase 4 perf work by keeping the per-projection **activations** resident on-device between sequential kernels, collapsing the per-op htod(input)+dtoh(output) round-trips the host-orchestrated pipeline paid between every sequential kernel. Session 19 collapsed the *weight* round-trips (uploaded once, cached); this collapses the *activation* round-trips for the dense FFN chain. Added **device-resident launch variants** on `CudaRuntime` that take an input already on the device (`&CudaSlice<f32>`) and return a device-resident output (`CudaSlice<f32>`) with NO internal htod/sync/dtoh: `launch_q4k_matmul_dev` / `launch_q6k_matmul_dev` (twins of the amortised matmul launchers — same shape validation + weight-cache upload, but the input stays resident and the output isn't read back), `launch_geglu_silu_dev` / `launch_geglu_gelu_tanh_dev` / `launch_activation_silu_dev` / `launch_activation_gelu_tanh_dev` (twins of the elementwise activation launchers, via two shared helpers `launch_elementwise_binary_dev` / `launch_elementwise_unary_dev` that mirror the host-readback launchers' arg layout so the device kernels see identical args), plus `upload_f32` (one input upload) and `sync_dtoh_f32` (the single sync+readback at the end of a chain). All chained launches run on the same CUDA stream (stream-ordered — a kernel reading a buffer written by an earlier kernel on the same stream sees the data without an inter-kernel sync), so an N-kernel chain pays exactly one sync + one dtoh instead of N. Wired a fused **device-resident prefill FFN chain** `host_prefill_ffn_block_device` on `CudaBackend`: pre-ffn norm (host) → upload normed input once → gate/up amortised matmul (device, outputs stay) → activation (device, reads gate/up outputs in place) → down amortised matmul (device) → single `sync_dtoh` readback → post-ffn norm + residual (host). The post-ffn norm + residual stay host-side (the residual needs `h_post_attn`, host-resident from the previous layer; threading it onto the device too is the next collapse slice) and were factored into a shared `apply_post_ffn_residual_prefill` helper used by both the device and host paths so they can't drift. `host_prefill_ffn_block` now dispatches device-first, falling back to the renamed `host_prefill_ffn_block_hostonly` (the prior host-orchestrated body, now the parity oracle). The chain bails to `None`→host when: no runtime; work below the `ACTIVATION_NATIVE_MIN_ELEMS` gate (`seq*inter < 8192`); gate/up/down formats aren't a uniform Q4_K or Q6_K; the down matrix's stored width is padded beyond `inter` (the chain assumes a contiguous `[seq, inter]` activation feeds the down matmul directly — a new `down_stored_cols` helper derives the contraction width without allocating the pad, mirroring `down_padded_activation`); or the activation/ffn-type isn't one of the native kernels. Added a `pub(crate) CudaBackend::runtime()` accessor so `pipeline.rs` can drive the chain. The matmul kernels are bit-exact with their CPU twins, so the device-chain output diverges from the host reference only in the activation transcendental (device `tanhf`/`expf` vs host Rust libm, ≤ 1e-5 on the raw activation, amplified by the down matmul's linear contraction). Added 3 tests: `device_ffn_chain_bails_on_scaffold` (host-runnable — pins the no-runtime bail + dispatch↔hostonly exact match), `device_ffn_chain_matches_host_orchestrated_when_runtime_available` (runtime-gated — device chain vs host reference, max_abs < 1e-3 tolerance accommodating the activation-libm divergence), `device_ffn_chain_bails_below_activation_gate` (host-runnable — small-seq bail). No new device kernels (reuses the Session 5/14 q4k/q6k matmul + activation kernels); no capability-reporting change (the chain is a perf path under the already-advertised `PrefillQ4`).

  **Session 20 review fixes** (2 duplication-with-drift-risk findings from a local review): the four host-readback launchers (`launch_q4k_matmul`, `launch_q6k_matmul`, `launch_elementwise_binary`, `launch_elementwise_unary`) duplicated the new `*_dev` twins' shape-validation + weight-cache-upload + kernel-arg layout against the **same** `CudaFunction`. Both paths feeding the same kernel from two arg-construction sites is silent-UB drift risk: if a kernel arg changes, updating only one path feeds garbage to the shared kernel. Made the host launchers **delegate** to the device-resident variants so the kernel-arg layout + `LaunchConfig` live in exactly one place: the matmul host launchers now do `upload_f32` → `launch_q{k}_matmul_dev` → `sync_dtoh_f32` (the dev variant owns all shape validation + the weight-cache upload + the launch); the elementwise host helpers keep their cheap host-slice length guard (avoids paying for an upload on a bad shape) + ctx-specific error messages, then delegate the launch to `launch_elementwise_{binary,unary}_dev` and do their own sync+dtoh+`copy_from_slice`. The remaining shared logic is only the trivial length check (low drift risk — a divergence changes which error fires, never produces wrong results). Business-logic verification confirmed the arg layouts were already bit-identical pre-refactor, so this is latent-risk removal, not a behaviour change; all 100 cuda tests + the runtime-gated parity tests still pass.

  **Session 20 review-fix #2** (1 behaviour-regression finding from a second local review): the delegation refactor dropped the `num_rows == 0 || seq_len == 0` early-return guard from `launch_q4k_matmul`/`launch_q6k_matmul`. The old host launchers returned `Ok(vec![])` with no device work for the degenerate shape; the new path hit `upload_f32` → `clone_htod(&[])` (or the dev variant's `alloc_zeros(0)`), both of which cudarc rejects (the codebase documents at `runtime.rs:968` that cudarc requires a non-empty slice and works around it elsewhere with a placeholder) — so the zero-output case flipped from `Ok(vec![])` to `Err`. Restored the guard ahead of the upload in both host launchers (`if num_rows == 0 || seq_len == 0 { return Ok(Vec::new()); }`), preserving the original short-circuit contract. The pipeline currently guards `seq_len == 0` upstream (`host_prefill_kquant`) and the device chain gates `seq*inter >= 8192`, so the path was unreachable in practice, but the launchers are reachable via the public `QuantMatVec::q4k_matmul`/`q6k_matmul` trait surface where a zero-shape call is a real edge case. Added a runtime-gated test `q4k_matmul_zero_shape_returns_empty_without_device` pinning both zero sub-cases (`seq_len==0` and `num_rows==0`) so the contract can't regress again.

- **Session 21** (CUDA: device-resident activation chain for the decode FFN block — third slice of the round-trip collapse): extended Session 20's prefill FFN device chain to the **decode** path — the per-token FFN chain (the hottest path during generation). Added two device-resident matvec launchers on `CudaRuntime` — `launch_q4k_matvec_dev` / `launch_q6k_matvec_dev` (twins of the host-readback matvec launchers: take an already-resident `&CudaSlice<f32>` input, return a device-resident `CudaSlice<f32>` output, no internal htod/sync/dtoh; same shape validation + weight-cache upload as the host path). Refactored the host `launch_q4k_matvec` / `launch_q6k_matvec` launchers to **delegate** to the `_dev` variants (Session 20's single-source-of-truth pattern: `upload_f32` → `launch_q{k}_matvec_dev` → `sync_dtoh_f32`, keeping the `num_rows == 0` short-circuit ahead of the upload so the shared `_dev` arg layout + weight-cache upload can't drift from the host-readback path). Added a `matvec_dev_by_fmt` dispatch helper (the decode twin of `matmul_dev_by_fmt`) + a fused **device-resident decode FFN chain** `host_ffn_block_device` on `CudaBackend`: pre-ffn norm (host) → upload normed input once → gate/up matvec (device, outputs stay) → activation (device, reads gate/up outputs in place via the existing `*_dev` activation launchers) → down matvec (device) → single `sync_dtoh` readback → post-ffn norm + residual (host). Split the decode FFN block into `host_ffn_block` (dispatcher: device-then-hostonly) + `host_ffn_block_hostonly` (host-orchestrated reference, the parity oracle + fallback) mirroring the prefill split, so the two paths can't drift on the residual/norm wiring. The chain bails to `None` when: no runtime; the work is below the activation gate (`inter < ACTIVATION_NATIVE_MIN_ELEMS = 8192`, so small-`inter` models keep the host path); the gate/up/down formats aren't uniform Q4_K/Q6_K; the down matrix's stored width is padded beyond `inter`; the input isn't a contiguous `[1, hidden]` row; or the activation/ffn-type combination isn't a native kernel. Added 3 tests: a host-runnable scaffold bail + dispatch↔hostonly bit-exact match (runs on every host), a runtime-gated device-vs-host parity on a synthetic large Q4_K FFN (`hidden=256, inter=8192`, ≤ 1e-3 tolerance), and a host-runnable below-gate bail. The decode FFN device chain fires on real models (Gemma 4B `inter=14336`, 26B-A4B dense slab larger) where `inter >= 8192`.
 
 - **Session 22** (CUDA: device-resident activation chain for the prefill attention block — fourth slice of the round-trip collapse): extended the round-trip collapse from the FFN block (Sessions 20/21) into the **attention** block — the Q/K/V → QK-norm → V-norm → RoPE → causal attention → O chain. Added four device-resident launch variants on `CudaRuntime` — `launch_rms_norm_dev` (body norm), `launch_rms_norm_heads_dev` (per-head QK/V-norm), `launch_rope_dev`, and `launch_prefill_attention_dev` — twins of the host-readback launchers that take an already-resident `&CudaSlice<f32>` input and return a device-resident `CudaSlice<f32>` output with no internal htod/sync/dtoh (the norm weights + RoPE `inv_freq` are uploaded inside the `_dev` variant, matching the matmul `_dev` twins uploading weights through the cache). Refactored the four host launchers (`launch_rms_norm`/`launch_rms_norm_heads`/`launch_rope`/`launch_prefill_attention`) to **delegate** to their `_dev` twins (Session 20's single-source-of-truth pattern: the kernel-arg layout + `LaunchConfig` + deep shape/u32/shared-mem guards live in exactly one place — the `_dev` variant — so the host-readback per-op path and the device-resident attention chain can't drift on arg order; the host launchers keep only the cheap host-slice length validation). Added a fused **device-resident prefill attention chain** `host_prefill_attention_block_device` on `CudaBackend`: input norm (host, matching the FFN device chain's pre-norm-on-host) → upload the normed input once → Q/K/V matmul (the three projections share one resident input) → QK-norm/V-norm/RoPE (all on device, reading resident projection outputs) → causal prefill attention (resident q/k/v) → O matmul (resident attention output) → a single `sync_dtoh` readback of O, plus the post-RoPE K / post-V-norm V read back into the host KV mirror (the first readback syncs the stream; the K/V readbacks are idle-stream copies). The post-attn norm + residual stay on the host (the residual needs `h`, host-resident from the previous layer; threading it onto the device too is the next collapse slice). Split the prefill attention block into `host_prefill_attention_block` (dispatcher: device-then-hostonly) + `host_prefill_attention_block_hostonly` (the existing per-op native-then-host path, now the parity oracle + fallback) mirroring the FFN block split. The chain bails to `None` when: no runtime; the attention work is below `PREFILL_ATTN_NATIVE_MIN_WORK = 8192`; any of the Q/K/V/O projections isn't a Q4_K/Q6_K the device matmul handles; the input isn't a contiguous `[seq, hidden]` row; the O-matmul contraction (`q_dim`) isn't a multiple of 256; or the prefill attention shape exceeds the device shared-mem/index budget (`Err` from the attention launcher maps to `None`). Added 3 tests: host-runnable scaffold bail + dispatch↔hostonly bit-exact match, runtime-gated device-chain vs host reference parity (`max_abs < 1e-3` on a synthetic large Q4_K attention layer, `hidden=256`, `num_q=8`, `num_kv=2`, `head_dim=32`, `seq=8`), and host-runnable below-attention-gate bail. The decode-path attention device chain (the Session 21 twin of this slice) is the follow-on.

  
 - **Session 23** (CUDA: device-resident activation chain for the decode attention block — fifth and final slice of the round-trip collapse): extended the round-trip collapse from the prefill attention block (Session 22) into the **decode** attention block — the per-token Q/K/V → QK-norm → V-norm → RoPE → decode-attention → O chain (the hottest path during generation). Added a device-resident `launch_decode_attention_dev` on `CudaRuntime` (twin of the host-readback `launch_decode_attention`: resident `q_dev` + uploaded full-KV `k_dev`/`v_dev` → resident `CudaSlice<f32>` output, no internal sync/dtoh) and refactored the host `launch_decode_attention` to **delegate** to it (Session 20's single-source-of-truth pattern: the kernel-arg layout + `LaunchConfig` + shape/u32-index guards live in exactly one place — the `_dev` variant — so the host-readback per-op path and the device-resident decode attention chain can't drift on arg order). The `scores_dev` scratch (`num_q * total_len`) is **caller-owned** (passed as `&mut CudaSlice<f32>`), not allocated inside the `_dev` launcher — so in the device chain it's bound in the chain scope and drops after the final readback, not mid-chain (on pool-less devices a mid-chain `CudaSlice::drop` forces a stream sync; the same discipline the Session 22 review-fix established for the attention-chain intermediates). Added a fused **device-resident decode attention chain** `host_attention_block_device` on `CudaBackend` (the decode twin of `host_prefill_attention_block_device`): input norm (host) → upload normed input once → Q/K/V matvec (the three projections share one resident input) → QK-norm/V-norm/RoPE (all on device, reading resident projection outputs; `inv_freq` uploaded once + shared by Q/K) → read back the new post-RoPE K / post-V-norm V row → build the full `[prev+1, kv_dim]` KV from the prior host mirror + the new row + upload → decode-attention (resident Q + uploaded full KV) → O matvec (resident attention output) → single `sync_dtoh` readback of O → post-attn norm + residual (host). Split the decode attention block into `host_attention_block` (dispatcher: device-then-hostonly) + `host_attention_block_hostonly` (host-orchestrated reference, the parity oracle + fallback) mirroring the prefill split, so the two paths can't drift on the residual/norm/KV wiring. **Two device syncs remain** (vs ~8 per-op round trips on the host path): one to read back the new K/V row (needed to build the full KV the decode-attention kernel attends over — the mirror grows each token, so a resident device KV buffer would need an append step; that's the single-command-buffer follow-on), and one final readback of O. The chain bails to `None` when: no runtime; the attention work is below the gate (`num_q × total_len × head_dim < DECODE_ATTN_NATIVE_MIN_WORK = 8192`, so short contexts keep the host path); the four projections aren't all Q4_K/Q6_K; the input isn't a contiguous `[1, hidden]` slice. Capability reporting unchanged. Added 3 tests: a host-runnable scaffold-bail + dispatch↔hostonly exact match (also pins the K/V-row contract), a runtime-gated device-vs-host parity (max_abs < 1e-3 on a synthetic large Q4_K attention layer with a pre-populated KV mirror clearing the gate; K/V rows < 1e-4), and a host-runnable below-gate bail. **This completes the round-trip collapse follow-on** (Sessions 19–23): every dense decode/prefill projection chain — FFN (Sessions 20/21) and attention (Sessions 22/23) — now runs device-resident with a single end-of-chain readback; weights are cached once (Session 19). The remaining host round-trips are the two decode-attention KV readback/upload (the device-KV-cache append + resident-KV decode-attention is the final single-command-buffer slice) and the coarse `KvDispatch` bridge / PLE / remote-FFN / expert-matvec fusion.

  
 - **Session 24** (CUDA: device-fused MoE expert matvecs — the Session 12 follow-on): routed the per-expert gate/up/down Q4_K matvecs through the native CUDA `q4k_matvec` kernel, so the MoE block is no longer the only block running its heavy compute wholly on the host. The MoE block's dense slab already ran through the device FFN chain (Session 21); this slice moves the **expert** projections off the host. Added a shared `moe_expert_contribution_q4k` helper in `pipeline.rs` that mirrors the substrate `cpu_moe_forward` structure (routing → per-expert gated FFN → weighted sum → post-expert norm) but runs every gate/up/down projection through a caller-supplied matvec closure that performs **Q4_K × f32** (dequantize-then-dot). **Parity-target decision:** the device path's natural oracle is the CPU `q4k_matvec_into` (Q4_K × f32), **not** `cpu_moe_forward`'s default Q8_K-direct SDOT path — that is an Apple-Silicon-only (NEON `SDOT`) optimisation; CUDA has no SDOT, so the device path dequantizes Q4_K to f32 and dots with the f32 input, exactly the math `QuantMatVec::q4k_matvec` performs. The closure is `self.q4k_matvec` (native-then-CPU) on the device path and `CpuBackend::q4k_matvec` on a `#[cfg(test)]` host-only parity oracle. The gate/up split (`half = inter * (hidden/256)*144 bytes`) and the padding-column discipline (`act[inter..inter_padded]` stays zero across experts, matching the substrate `ExpertScratch::act`) mirror `run_single_expert_q4k_q8k_into`. Added `CudaBackend::moe_expert_contribution_device` (gates on `self.runtime()` first, then `moe_expert_contribution_q4k`) and `moe_combine_row_device` (the device expert contribution substitutes for the `cpu_moe_forward` call inside `moe_combine_row`; the dense-delta subtraction + outer post-norm + residual stay identical). `host_ffn_block_moe_decode`/`host_prefill_ffn_block_moe` now try the device combine first and fall back to `moe_combine_row` (`cpu_moe_forward`) on bail. **No behaviour change on non-CUDA hosts**: the device path always returns `None` without a runtime, so the MoE block keeps its existing `cpu_moe_forward` (Q8_K-direct on Apple Silicon) path; the native Q4_K × f32 expert path only fires on CUDA hosts. The device path also bails for non-Q4_K experts (BF16 monolith → `cpu_moe_forward`) and a hidden dim that isn't a 256-multiple (the gate/up byte split needs whole Q4_K super-blocks). Added 4 tests: host-runnable scaffold bail (`moe_expert_contribution_device_bails_on_scaffold`), host-runnable structure-match against a fresh `q4k_matvec_into` composition (`moe_expert_contribution_q4k_structure_matches_reference` — bit-identical since both use the same CPU matvec; verifies the gate/up split + activation + weighted-sum + post-norm wiring), host-runnable bail on non-Q4_K / non-256-multiple hidden (`moe_expert_contribution_q4k_bails_on_non_q4k_and_non_aligned`), and a runtime-gated native-vs-host parity (`moe_expert_contribution_native_matches_host_when_runtime_available`, no-op on this no-CUDA host). The existing `moe_decode_block_matches_independent_composition`/`moe_prefill_block_matches_independent_composition` tests (Session 12) still pass unchanged because the device path bails to `cpu_moe_forward` on the scaffold. The expert matvecs are not yet device-*chained* (each is a separate htod/launch/dtoh; the expert weights ARE cached via the Session 19 weight cache, so re-uploads are skipped) — device-chaining the per-expert FFN (mirroring Sessions 20-23's single-readback chains) is the follow-on. Capability reporting unchanged.

  **Session 24 review fixes** (3 findings from a local review): (1) **gate/up byte split deduped** — hoisted `q4k_gate_up_half(inter, hidden) -> Option<usize>` into `larql_compute::cpu::ops::moe::expert` (re-exported from the `moe` module) so the Q4_K row-stride (`(hidden/256)*144` bytes, one projection's span) lives in exactly one place; the two substrate sites (`run_single_expert_into` q4k path, `run_single_expert_q4k_q8k_into`) and the CUDA helper now all call it (substrate sites use `.expect("...overflow")`, matching their prior debug-build overflow-panic semantics but switching from silent release-wrap to a loud checked panic — strictly safer; the value is bit-identical for non-overflowing model dims). (2) **outer-combine deduped** — extracted `apply_outer_combine(ha, dense, h2, outer_w, layer, combined) -> Vec<f32>` and routed both `moe_combine_row` (host, `cpu_moe_forward`) and `moe_combine_row_device` (device) through it, so the Gemma-4 dense-delta + outer-post-norm + residual can't drift between the two paths (only the `h2` source differs). (3) **prefill scratch hoisted** — `moe_expert_contribution_q4k` now takes caller-owned `expert_out: &mut [f32]` (`[hidden]`, re-zeroed per call before accumulation) + `act: &mut [f32]` (`[inter_padded]`, padding columns stay zero across positions); `host_prefill_ffn_block_moe` hoists both out of the per-position loop (mirrors the existing `combined` hoist), eliminating the `2 * seq_len` per-position allocations the prior version paid. Decode allocates the scratch once (single token). All 114 cuda tests + the substrate compute tests (738) still pass; clippy clean.

- **Session 26 / GPU-004** (CUDA: first real-hardware validation — RTX 3090, sm_86, CUDA 12.4): the first time any CUDA code executed on real GPU hardware. Discovered that all 140 crate tests were **silently passing on the CPU scaffold** — the native runtime never initialized because NVRTC could not compile the kernel source. Three compilation blockers were found and fixed: (1) `cuda_fp16.h` not discoverable — NVRTC has no default system include search path, and Debian places CUDA headers at `/usr/include` rather than `/usr/local/cuda/include`; added `cuda_include_paths()` to `runtime.rs` that discovers headers from `$CUDA_HOME`, standard paths, and validates `cuda_fp16.h` exists; (2) duplicate symbol definitions — the 20 kernel sources are concatenated into one compilation unit, each defining `larql_half_bits` and `larql_decode_f16`; added `#ifndef LARQL_HALF_BITS_DEFINED` include guards to `ops.rs`; (3) `INFINITY` undefined in NVRTC — replaced `-INFINITY` with `__int_as_float(0xff800000)` in `ops.rs`. After these fixes, the hardware probe test confirmed: RTX 3090 detected, 20 native NVRTC kernels compiled targeting `compute_86`, `supports(QuantMatVec)=true`, `supports(DecodeToken)=true`. Also fixed 6 test-infrastructure issues exposed only on real hardware: 3 scaffold-only tests needed `native_runtime_available()` gating, 5 Q6_K parity tests needed float tolerance instead of exact `assert_eq!`, and 1 KV-append error message string mismatch. Result: 139/141 tests pass with native CUDA kernels executing. Two failures are a real decode attention parity bug (max_abs=0.13, deterministic) — deferred to a stabilization slice. CLI backend selection verified: `--backend cuda`, `LARQL_BACKEND=cuda`, `--backend cpu`, invalid-backend-loud-fail. PTX cache cold/hot verified. End-to-end `larql run --backend cuda` executes but produces garbage due to the decode parity bug + a vindex vocab_size padding issue. Validation report: `bench/baselines/cuda-hardware-validation-2026-07-08.md`. PR: #35 (merged). Also found extraction pipeline issues: `write_down_meta_and_clusters` and `run_clustering` run unconditionally even for `--level inference` (gated locally as workaround); the `down_meta` matmul is 252 TFLOP serial (7 hours single-threaded); OpenBLAS barely multi-threads on multi-core hosts.

- **Session 27 / ASTAB-001** (CUDA: decode parity fix — root cause was a numerics-reference mismatch, not a kernel bug): resolved the two GPU-004 decode parity failures (`decode_token_matches_cpu_reference_when_runtime_available` and `multi_token_decode_matches_cpu_reference`, both max_abs=0.1314532 vs the 1e-3 tolerance). **Root cause:** the tests compared CUDA's f32-activation decode pipeline (`host_decode_token` → `q4k_matvec` dequant-then-f32-dot, the same numerics CUDA's prefill pipeline and `predict_kquant_prefill` use) against the production CPU decode reference `predict_kquant_decode_step_direct`, which uses int8 Q8_K SDOT matvec (`q4k_q8k_matvec_into`). CUDA has no SDOT instruction, so its decode intentionally uses f32-activation numerics (documented in `pipeline.rs` `moe_expert_contribution_q4k`). The int8-vs-f32 mismatch is ~2% scale-relative by design (pinned by `q8k_direct_proj_matches_f32_activation_within_quant_tolerance`); through 2 layers × 7 matvecs it accumulates to the observed 0.13 on the small synthetic Q4K fixture. The prefill parity test passed because both sides used f32-activation; the decode tests failed because the CPU reference used int8. **The `decode_attention` kernel was NOT the bug** — audited strides/dims/GQA mapping/scale/softmax/V-indexing and confirmed correct; 5 new focused native parity tests (ASTAB-001C: single-head, multi-head, GQA asymmetric, softcap multi-position, fixture-shape shrink) pin the kernel's correctness directly at 1e-4-relative. **Fix:** the two decode parity tests now compare against `predict_kquant_decode_step` (f32-activation decode reference, the decode twin of `predict_kquant_prefill`) instead of `predict_kquant_decode_step_direct` (int8 production path). No tolerance loosened; no CUDA path routed to CPU; no kernel change; the int8 production decode path left unchanged (correct for its Apple-Silicon SDOT target). **Hardware validation: PENDING** — developed and scaffold-validated on a no-CUDA host (145/145 lib tests pass, clippy clean, fmt clean; the `hardware_probe` integration test fails as expected with no CUDA device). The PR must run `cargo test -p larql-compute-cuda --features cuda` on real hardware before merge. Report: `bench/baselines/cuda-decode-parity-fix-2026-07-09.md`. Remaining blocker: vindex vocab_size padding mismatch (151643 vs 151936) still affects end-to-end text output on both backends — tracked separately, not a CUDA parity failure.

- **Session 28 / VINDEX-001** (vindex: fix vocab_size padding / embedding dtype detection): resolved the GPU-004 "Known data issue" (P1, item #8) — the vindex vocab_size padding mismatch that produced garbage CPU **and** CUDA output for padded-vocab GGUF models (Qwen2.5-3B-Instruct Q4_K_M). **Root cause:** the GGUF in-memory loader (`larql-models/src/loading/gguf/loader.rs:439`) collapsed two distinct numbers into one — it kept the *logical* vocab (151643, from GGUF metadata `qwen2.vocab_size`) as `ModelWeights.vocab_size` and discarded the *physical* embedding row count (151936, the actual tensor shape). The `embed`/`lm_head` tensors kept their 151936 rows, so `ModelWeights` was self-inconsistent. That flowed into `build_vindex`, which wrote the **full 151936-row** tensor to `embeddings.bin` but recorded `vocab_size: 151643` in `index.json`. At load time every byte-length/dtype check used 151643, so the f32-equality dtype sniff failed and the file was misdetected as f16 (decoded as garbage); the tied-lm_head adoption gate (`[expected_f16, expected_f16*2)`) let the wrong-size file through. This is a data-shape correctness bug, not a CUDA parity bug — CPU and CUDA produced identical garbage (GPU-004 §4). **Fix (schema + extractor + loader + sampling):**
  - **Schema:** `vocab_size` is redefined to always mean *physical* row count (matches the streaming path + the safetensors loader, which already used tensor shape). New optional `logical_vocab_size: Option<usize>` field on `VindexConfig` + `ModelWeights` + `VectorIndex` (serde `default` + `skip_serializing_if="Option::is_none"` → absent when logical==physical → full backward compat; old vindexes load unchanged).
  - **Extractor:** GGUF loader now keeps `embed.shape()[0]` as `vocab_size` and records the smaller metadata/tokenizer value in `logical_vocab_size` (same change in the streaming embeddings stage + the build path + both `index.json` writers; safetensors loader mirrors it from `config.vocab_size`).
  - **Loader:** dtype detection in `load_embeddings`/`load_vindex_embeddings` now does an **exact** f32-or-f16 size match and **fails loudly** on any incompatible length (instead of silently falling back to f16); the tied-lm_head adoption gate tightened from the ambiguous range to exact f16 size (`mmap.len() == expected_f16`).
  - **Sampling (the invariant):** every lm_head dispatch path — Q4_K, f16, f32, stride32 (`larql-vindex`) and the inference `backend_lm_head_topk` (`larql-inference`) — now excludes padding rows `[logical..physical)` so sampled token IDs stay logical/tokenizer IDs. The f16/inference paths slice the mmap/tensor to `logical` rows (keeps the in-kernel top-1/top-K fast paths correct); the Q4/f32/stride32 paths fill `[logical..]` with `-inf`.
  Removed the now-dead `DEFAULT_GGUF_VOCAB_SIZE` const. **Tests (no CUDA required):** 8 new tests — GGUF loader keeps physical + records logical (`load_gguf_padded_vocab_keeps_physical_and_records_logical`, synthetic GGUF fixture with physical=64/logical=50); embeddings loader padded f16/f32 use physical rows; incompatible length fails loudly; `logical_vocab_size` serde round-trip (+ omitted when None); lm_head masks padding rows when logical set + no-masking when unset. All existing non-padded fixtures pass unchanged. **Verification (CachyOS / rustc 1.96.1):** `cargo fmt --all --check` clean; `cargo clippy -p larql-vindex -p larql-models -p larql-inference -p larql-cli -p larql-lql -p larql-server --tests -- -D warnings` clean; full test suites green — larql-vindex lib 1124 / tests all, larql-models lib 422 / tests 32, larql-inference lib 1258, larql-lql/server/cli tests all pass. **CPU/CUDA end-to-end validation: PENDING OPERATOR** — no representative Qwen2.5-3B GGUF was locally available to the agent. Operator command: rebuild the vindex without hand-editing `index.json` (`larql extract-index <qwen-q4km.gguf> -o <vindex> --quant q4k --level inference`), then `larql run <vindex> --backend cpu --max-tokens 16 "The capital of France is"` and on a CUDA host `CUDA_VISIBLE_DEVICES=0 larql run <vindex> --backend cuda --max-tokens 16 "The capital of France is"`; confirm meaningful + matching output. Docs updated: `bench/baselines/cuda-hardware-validation-2026-07-08.md` (§4/§9/§11), `docs/cuda-vulkan-completion-plan.md` (Phase A). Remaining blocker: the ASTAB-001 decode-parity hardware validation (separate slice).

- **Session 29 / GPU-006** (CUDA: resident-KV decode attention — first Phase B perf slice): eliminated the per-token full-KV host→device re-upload on the decode hot path. **Root cause / what changed:** the Session 23 decode-attention device chain kept the host KV mirror as the source for attention — every decode token read the new post-RoPE K / post-V-norm V row back to the host, rebuilt the full `[total_len, kv_dim]` KV from the host mirror prefix + the new row, and **re-uploaded the entire KV** to the device for the `decode_attention` kernel (an `O(total_len)` transfer that grew each token). The device `CudaKVCache` (Session 10) was populated in lockstep by prefill but **never read by decode** — its cursor was not advanced during decode, so it couldn't serve as the attention source. **The fix is a targeted swap:** the `decode_attention` CUDA kernel already reads K/V at `i * kv_dim + kv_off`, and the device `CudaKVCache` stores each layer as a contiguous `[max_seq, num_kv_heads, head_dim]` = `[max_seq, kv_dim]` f32 buffer — the exact same layout — so **no new kernel was needed**. Added `CudaRuntime::launch_decode_attention_resident_dev` (resident-KV twin of `launch_decode_attention_dev`: borrows the cache's `CudaSlice<f32>` K/V, attends over `0..current_len`, never the uninitialized capacity rows beyond the cursor; the kernel-arg layout is shared with the full-upload path via an extracted `launch_decode_attention_kernel` so the two can't drift). The decode attention block (`host_attention_block_device` in `pipeline.rs`) now prefers the resident path: append the new K/V row to the device cache via the existing `kv_append` kernel (`native_kv_append` → `resident_kv_decode_attention`), then attend over the resident K/V — replacing the full-KV upload with an `O(1)` single-row append. **Explicit testable fallback:** when the resident path is ineligible (`Ok(None)`: no/undersized cache, layer shape mismatch, or the device cursor out of lockstep with the host mirror) or a launch fails (`Err`), decode falls back to the **existing** full-KV-upload path (the pre-GPU-006 behavior, extracted into `decode_attention_full_upload`) — never to CPU. **Invariants preserved:** the host KV mirror is kept as the parity oracle + the source for truncate/state-dump/fallback (decode still appends the new row to both, so host and device cursors stay in lockstep — asserted by the new lifecycle tests); `reset_kv_cache`/`truncate_kv_cache` clear/roll back both caches as before; ASTAB-001 numerics unchanged (same `decode_attention` kernel, same f32-activation reference, tolerance still 1e-3). **Diagnostics:** `LARQL_GPU_DIAG=1` now surfaces `cuda resident-KV decode: uses=<N>, fallbacks=<M>, resident_rate=<%>` via `device_info()` (two `AtomicU64` counters, thread-safe, no hot-path lock). **Tests (8 new, runtime-gated, scaffold early-return):** 4 lifecycle/lockstep (prefill→decode-1, prefill→decode-multi, reset clears both, truncate keeps lockstep — all assert `kv_cache_len_native() == host_kv_len()`) + 3 parity (resident-KV single-token + multi-token decode vs CPU f32 reference within 1e-3; resident path reads only valid `0..current_len` rows, ignoring uninitialized capacity) + 1 fallback (no-cache-allocated forces the full-upload path; output still matches reference; fallback counter advances). **Hardware validation: COMPLETE** — re-validated on RTX 3090 (sm_86, NVRTC 12.4, driver 550.163.01), commit `ec95287d`: **154/154 tests green with default settings** (`hardware_probe` + ASTAB-001 decode parity + all 8 `resident_kv` + full suite under `LARQL_GPU_DIAG=1`, 153 lib + 1 probe, 0 failed, 2.72s), 21 native kernels loaded. The first hardware pass found a **test bug** (not an implementation bug): the original 3-token prompt's decode work `4·4·64 = 1024` was below the `DECODE_ATTN_NATIVE_MIN_WORK = 8192` gate, so `host_attention_block_device` bailed and the resident path never ran — fixed in commit `ec95287d` by switching to a 40-token prompt (`work = 4·41·64 = 10496 ≥ 8192`); no production code or threshold changed. B1 is marked complete in `docs/cuda-vulkan-completion-plan.md`. Report: `bench/baselines/cuda-resident-kv-2026-07-09.md`. **Next CUDA slice (B2): cross-layer hidden-state residency** — keep `h` resident across blocks/layers within a decode step, reading back once per token (logits input) instead of once per block; the residual-add kernel already exists, so this is plumbing. After that: launch batching/graphization (B3), lm-head on device (B4).

- **Session 30 / GPU-007** (CUDA: cross-layer hidden-state residency — second Phase B perf slice, **SCAFFOLD-VALIDATED, PENDING HARDWARE VALIDATION**): collapsed the remaining per-block + inter-layer hidden-state host↔device boundaries on the decode hot path. **What changed:** the Session 21/23 decode attention + FFN device chains already kept Q/K/V/norm/RoPE/attention/O and gate/up/act/down resident *within* each block, but the **post-block residual** (post-attn `h + res_mult * O` and post-ffn `h_post_attn + res_mult * down`) still ran on host (it needs the prior host-resident `h`), the **O and down projection outputs were read back** to feed those host residuals, and the **layer output was re-uploaded** by the next layer's attention input norm. That is 4 hidden-state boundaries per dense decode layer. The fix threads a device-resident hidden state through the decode layer loop via a new `DecodeHiddenState` enum (`pipeline.rs`, local to the module) holding either a host `[1, hidden]` `Array2` or a device `CudaSlice<f32>`, with explicit `ensure_host` (single `sync_dtoh`, idempotent) / `as_device` transitions so ownership + synchronization can't be confused. New `host_decode_token_resident` mirrors `host_decode_token` but carries `DecodeHiddenState` across layers: for each eligible dense layer, `host_attention_block_device_resident` (the resident-hidden twin of `host_attention_block_device`) consumes a device-resident input, runs the input norm **on device** via `launch_rms_norm_dev` (parity-safe — same kernel as the host-readback path; gated to RmsNorm-only since there's no LayerNorm device kernel), keeps the O projection output resident, and runs the post-attn norm + residual **on device** via `launch_rms_norm_dev` + the new `CudaRuntime::launch_residual_add_dev` (a thin `pub(crate)` wrapper over the existing *private* `launch_elementwise_binary_dev` with `extra_scalar = Some(b_scale)` — exactly the pattern the host-readback `launch_residual_add` uses; **no new kernel**, the completion plan confirms this). `host_ffn_block_device_resident` (the twin of `host_ffn_block_device`) does the same for the FFN chain: pre-ffn norm on device, down output resident, post-ffn norm + residual on device. The K/V row readback + host mirror append remain (GPU-006 invariant — unchanged). The layer→layer handoff keeps `h` device-resident with **zero** hidden-state readback/upload between eligible layers; the only readback is at the final decode output (logits input). `layer_scalar` (Gemma 4; rare, mostly 1.0) is applied host-side when present (one extra boundary in the rare scalar case; the common identity path stays fully resident). **Eligibility + fallback:** a pure `resident_hidden_layer_eligible` gate (testable on every host) requires dense (not MoE — the dense slab + outer combine is host-shaped), Q4_K/Q6_K all seven projections, uniform gate/up/down k-quant, `stored_cols == inter`, RmsNorm, Gated/Standard SiLu/GeluTanh, and both the decode-attention work gate (`num_q*total_len*head_dim ≥ 8192`) + activation gate (`inter ≥ 8192`). When ineligible (or a chain launch returns `Err`), the loop converts the device hidden state back to host exactly once (`ensure_host`) and runs the existing `host_attention_block`/`host_ffn_block` for that layer — the next eligible layer can re-enter the resident path (re-uploading via its device input norm). **Partial-residency correctness:** when the resident attention succeeds (K/V already appended to the host mirror) but the resident FFN bails, the loop does NOT re-run the full host attention (which would double-append K/V and corrupt the cache) — it reads the device post-attn state back once and runs ONLY the host FFN block on it. PLE/remote-FFN still bail to `None`. State dump (`StateDumpMask != None`) forces the host-orchestrated `host_decode_token` (exact host sync at every dump point). The existing `host_decode_token` is **kept verbatim** — it remains the parity oracle, the state-dump path, and the fallback. `decode_token` (the `DecodeBackend` trait impl) now calls `host_decode_token_resident`; the state-dump variants keep `host_decode_token`. **Invariants preserved:** GPU-006 KV cursor lockstep is untouched (resident hidden state is orthogonal — the K/V readback + host mirror append happen exactly as before); RoPE/abs position unchanged; operation order unchanged; ASTAB-001 numerics unchanged (same kernels, same f32-activation reference, tolerance still 1e-3); no CUDA path silently falls back to CPU. **Diagnostics:** `LARQL_GPU_DIAG=1` now also surfaces `cuda resident-hidden decode: uses=<N>, fallbacks=<M>, resident_rate=<%>` via `device_info()` (two `AtomicU64` counters `resident_hidden_uses`/`resident_hidden_fallbacks`, mirror the GPU-006 pattern). **Tests (6 new, runtime-gated, scaffold early-return):** single-token parity vs CPU f32 reference (1e-3), multi-token (4-token) parity + stability, diag-surface assertion, forced-fallback (3-token sub-gate prompt → host path, output still matches, fallbacks counter advances), KV-lockstep-unchanged (re-asserts GPU-006 `kv_cache_len_native() == host_kv_len()` after a resident-hidden decode), and consecutive-layers (4-layer fixture via `make_test_q4k_weights_layers(4)`, asserts `uses ≥ num_layers` — proves ≥2 consecutive resident layers with no intermediate host readback). **Hardware validation: PENDING** — developed and scaffold-validated on a no-CUDA host (no `libcuda.so` / `nvidia-smi`): **159/159 lib tests green** (153 prior + 6 new; all 8 `resident_kv` + 6 `resident_hidden` tests scaffold-pass via early-return), `cargo clippy -p larql-compute-cuda --tests -- -D warnings` clean, `cargo fmt --all --check` clean. The `hardware_probe` integration test fails as expected (no CUDA device). B2 stays **PENDING** in `docs/cuda-vulkan-completion-plan.md` until an RTX-class run confirms ASTAB-001 parity + the resident-hidden diag counters on real hardware. No new kernel count (still twenty-one) — `launch_residual_add_dev` reuses the existing `residual_add` kernel. **Next CUDA slices:** launch batching/graphization (B3), lm-head on device (B4), MoE/router polish (B5) — the latter two only if hardware timing shows they matter.

- **Session 31 / LARQL-GPU-PROFILE-001** (CUDA: post-residency decode bottleneck profile — **RTX 3060, NOT the packet-required RTX 3090**): profiled the post-GPU-007 decode path on real Qwen2.5-3B-Instruct (Q4_K_M) to pick the next engineering slice by evidence. Bootstrapped a full no-sudo toolchain (rustup 1.97, NVRTC 12.4.127 + cudart headers, user-local OpenBLAS, standalone CMake) — the host had none of it. **161/161 CUDA tests pass on the 3060** (same sm_86 as 3090 → same kernels/NVRTC). Added gated `LARQL_GPU_PROFILE=1` decomposition counters (launches, HtoD/DtoH copies+bytes, syncs on `CudaRuntime`; host KV mirror realloc+copy ns + final hidden readback ns on `CudaBackend`/`pipeline.rs`) — every recorder is a no-op unless the env var is set, surfaced via `device_info()` + a new `ComputeBackend::take_profile_counters()` trait method + a bench JSON `profile` block + a human-readable "GPU profile" table. Profiling disabled: 161/161 tests still green, bench output unchanged, fmt+clippy clean. **🔑 Headline finding: GPU-007 (resident-hidden) engages 0% on the default Q4_K_M format** (measured: 0 uses / 612 fallbacks) because `resident_hidden_layer_eligible` requires a uniform Q4_K or Q6_K FFN triple, and Q4_K_M intentionally mixes gate/up Q4_K + down Q6_K. It engages 100% only with `--down-q4k` (uniform Q4_K). The synthetic GPU-007 fixtures are uniform Q4_K, so the code was validated but the production default never reaches it. resident-KV (GPU-006) is 100% active on both formats. **Steady-state (Q4_K_M, 5 reps, MAD ≤ 0.4 ms):** 131.5 ms/tok (7.6 tok/s) @ 10 prompt tokens, 133.2 @ 149 tok, 148.9 @ 1301 tok; GPU-fwd 80% / lm-head 20%. **uniform Q4_K (resident-hidden ON):** 122.2 ms/tok (8.2 tok/s) — 7% faster, 13× less DtoH traffic. **Recommendation (rank #1, new slice D6):** make resident-hidden handle the mixed Q4_K/Q6_K FFN triple so GPU-007 engages on Q4_K_M — recovers the measured 7% + the DtoH reduction, and unblocks GPU-007 from being dead code on the critical path. Ranks above B3 (launch batching, #2, launch-latency inferred not measured) and B4 (lm-head, #3, GEMV already device-side). **Caveat:** 3060 evidence, pending 3090 re-validation. Report: `bench/baselines/cuda-post-residency-profile-2026-07-10.md`. No B3/B4/B5/C1 implemented (profiling-only slice).

- **Session 32 / LARQL-GPU-D6** (CUDA: resident-hidden decode for the default Q4_K_M — **RTX 3060, hardware-validated, FINAL per verification policy**): extended GPU-007 resident-hidden decode to the production-default Q4_K_M FFN triple so it engages on the format users actually have. **Root cause (D6-A reproduced):** both `resident_hidden_layer_eligible` (pipeline.rs) and `host_ffn_block_device_resident` (pipeline.rs) gated on a `matches!` accepting **only** uniform Q4_K×3 or Q6_K×3; the default Q4_K_M layout (gate/up Q4_K, down Q6_K) failed both, so GPU-007 engaged 0% on the real Qwen2.5-3B Q4_K_M vindex (measured pre-D6: 0 uses / 288 fallbacks, resident-KV 100%). **The fix needs no kernel changes.** `matvec_dev_by_fmt` already dispatches each projection independently (Q4_K → `launch_q4k_matvec_dev`, Q6_K → `launch_q6k_matvec_dev`); the down projection consumes an f32 device activation regardless of its weight format. **D6-B:** added one shared pure helper, `supported_resident_ffn_triple(gate, up, down)`, used by both gates so they cannot drift. Each projection produces f32 device activations so mixed weight formats are technically possible; the helper intentionally accepts only produced+validated layouts — Q4_K/Q4_K/Q4_K, Q6_K/Q6_K/Q6_K, and Q4_K/Q4_K/Q6_K (the default Q4_K_M mix) — and rejects Q6_K/Q6_K/Q4_K and mixed gate/up. **D6-D:** added `Q4kmFixtureIndex` (`larql-compute/src/test_fixtures.rs`) — a faithful production-Q4_K_M `KvIndex` (attn Q/K/O Q4_K, V Q6_K; FFN gate/up Q4_K, down Q6_K) with format-aware `kquant_ffn_layer_once` (gate/up Q4_K dequant, down Q6_K dequant, cached per (layer, component)). **D6-E:** 6 host-runnable eligibility tests (no GPU) covering accepted triples, rejected permutations, and padded-down contraction. **D6-F:** 5 native mixed-Q4_K_M parity tests on RTX 3060 vs the CPU f32-activation reference (`predict_kquant_decode_step`), tolerance `max_abs < 1e-3`, with **EXACT** use/fallback counter assertions (`uses == num_layers*decoded_tokens`, `fallbacks == 0`, resident-KV same; forced-fallback asserts the exact split). Fixture-level assertions confirm `gate/up == Q4_K`, `down == Q6_K`, `wv == Q6_K`. **D6-G:** fmt clean; clippy clean (cuda + models + inference + cli --features cuda); ASTAB-001 green; GPU-006 8/8; GPU-007 12/12; D6 6/6; full CUDA suite **172/172 green** with `LARQL_GPU_DIAG=1` and default (+11 new tests: 6 host eligibility + 5 native D6); `larql-models` 539/539; release-mode D6 6/6. No threshold overrides, no tolerance changes. **D6-H (same-day A/B, the merge claim):** main `70cc8fb9` vs D6 `8584ae32`, 5 reps × 79 decode steps, same prompt/model/GPU. Median p50: **131.30 → 127.15 ms/tok (3.3% faster, 7.62 → 7.85 tok/s)**, consistent across all 5 reps. Instrumented decomposition: htod 3.9→1.5 MiB/tok (−62%), dtoh 6.1→2.4 MiB/tok (−61%), syncs 229.7→81.2/tok (−65%); launches go UP (482→632, expected — the resident chain keeps more work on-device with more chained kernels). Post-D6 resident-hidden on the real default Q4_K_M model: **uses=288, fallbacks=0, rate=100.0%** (was 0%/288/0.0%); resident-KV stays 100%. **D6-I:** deterministic CPU and CUDA generation on the real Q4_K_M vindex produce byte-identical output; the Q4_K_M vindex remains the default format — no `index.json` hand-edit or `--down-q4k` rebuild required. **Scope:** only `resident_hidden_layer_eligible` + `host_ffn_block_device_resident` changed; the non-resident decode + prefill device paths still require a uniform FFN triple (a separate slice; host fallback remains the parity oracle). **Verification policy:** RTX 3060 (sm_86) is the accepted canonical CUDA host going forward — no RTX 3090 re-validation required. Report: `bench/baselines/cuda-q4km-resident-hidden-2026-07-10.md`. **Next evidence-ranked slice: B3 (launch batching / graphization)** — the D6 path now shows 632 launches/tok.

- **Session 25** (CUDA: device-chained per-expert MoE FFN — the Session 24 follow-on): collapsed the Session 24 device expert path's per-projection htod/launch/dtoh round-trips into a device-resident per-expert chain, mirroring the decode/prefill FFN device chains (Sessions 20-23). Session 24 ran each expert's gate/up/down via `self.q4k_matvec`, which uploads the expert input and syncs+reads back on **every** matvec — 3 × top_k input uploads + 3 × top_k readbacks per token. The new `CudaBackend::moe_expert_contribution_device_chain` uploads the `expert_input` **once** (shared by every expert's gate/up, both reading the resident `x_dev`), serves weights from the Session 19 weight cache, keeps each expert's gate/up/activation/down outputs resident on the device between launches, and reads back **once per expert** (the down output only). Routing + the post-expert norm stay on the host exactly as in `moe_expert_contribution_q4k`. `moe_expert_contribution_device` now tries the chain first and falls back to the per-call matvec path when the chain bails — both paths share the same Q4_K × f32 math so the two can't numerically diverge. Extracted a pure `moe_expert_chain_eligible(moe, hidden) -> Option<(half, inter)>` gate (testable on every host) holding the chain's bail conditions: non-Q4_K experts, non-256-multiple hidden, a padded down contraction (`inter_padded != inter` — the chain feeds the `[inter]` activation output straight into the down matvec with no zero-pad step), and a non-gated activation (only `Silu`/`GeluTanh` have device-resident launchers). Added 2 tests: a host-runnable `moe_expert_chain_eligibility_gate` (pins all four bail arms + the eligible happy path against the shared `q4k_gate_up_half` stride) and a runtime-gated `moe_expert_device_falls_back_when_chain_ineligible` (padded contraction forces the chain to bail; the per-call fallback still matches the host-only Q4_K × f32 reference ≤ 1e-3); the existing `moe_expert_contribution_native_matches_host_when_runtime_available` now exercises the chain end-to-end on a CUDA host. Capability reporting unchanged.

 As of Session 7, **all five k-quant kernels are trait-routed**
 through `QuantMatVec` (`q4k_dual_matvec` since Session 6, `q6k_matmul` since Session 7); the Q6_K arm of `ffn/weight.rs::quant_matmul` now dispatches through a backend's `q6k_matmul` when one is supplied, so the staged `q6k_matmul` CUDA kernel is live. Session 8 added the dense `f32_gemv`/`f16_gemv` kernels trait-routed through `MatMul`, with `f32_gemv` only taking the native path on row-major-contiguous `ArrayView2`s. Session 9 added the legacy Q4_0 `q4_matvec`/`q4_vecmat` kernels trait-routed through `QuantMatVec`, completing the `QuantMatVec` kernel surface. Session 10 added the device-side KV cache (mirroring Metal) + the native `kv_append` kernel and wired the `DecodeBackend` KV lifecycle methods (`has_kv_cache`/`preallocate_kv_cache_per_layer`/`reset_kv_cache`/`kv_cache_len`/`truncate_kv_cache`/`populate_kv_layer`) through it. **Session 11 landed the fused `decode_token`/`decode_token_with_state_dump_masked`/`prefill_kquant` pipelines** as a host-orchestrated path (native matvec/matmul projections + host elementwise ops + host KV mirror) and flipped capability advertisement on (`supports_quant(Q4_K/Q6_K)`, `supports(DecodeToken/PrefillQ4/QuantMatVec)`) when a runtime is present — so `fused_prefill`/`fused_decode_step` now route through CUDA and the `auto` policy on Linux picks CUDA first for dense k-quant models (hybrid-MoE models route to the CPU path via a `DecodeMoe`-capability gate; see the Session 11 review fixes). Everything else in CUDA (the coarse `KvDispatch` bridge, PLE/remote-FFN fused paths, device-*chaining* the MoE expert FFN) and all of Vulkan still delegate to CPU/reference paths.

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

Session 4 delta verified:

- `cargo check -p larql-compute-cuda` — green
- `cargo test -p larql-compute-cuda --offline` → 9 passed

Session 5 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo test -p larql-compute-cuda` → 11 passed (up from 9; +2 runtime-gated parity tests)
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed
- `cargo test -p larql-compute-vulkan` → 6 passed

Session 6 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo check -p larql-compute-cuda` — green (clean, no warnings)
- `cargo test -p larql-compute-cuda` → 14 passed (up from 11; +3: q4k_dual_matvec delegate parity, native q6k_matmul, native q4k_dual_matvec)
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo fmt --all -- --check` — clean
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed
- `cargo test -p larql-compute-vulkan` → 6 passed

Session 7 delta verified:

- `cargo fmt --all -- --check` — clean
- `cargo check -p larql-compute -p larql-compute-cuda -p larql-compute-vulkan` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute --lib` → 734 passed (up from 733; +1 trait test)
- `cargo test -p larql-compute-cuda` → 16 passed (up from 14; +2: q6k_matmul trait delegate + native-via-trait)
- `cargo test -p larql-compute-vulkan` → 7 passed (up from 6; +1: q6k_matmul delegate)
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 8 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda` → 24 passed (up from 16; +8: f32/f16 gemv gating + native parity + dim-overflow rejection; old topk1/delegate tests converted to assert the gated `None` contract)
- `cargo test -p larql-compute-vulkan` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
 - `cargo test -p larql-cli --bins` → 243 passed

Session 9 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda` → 30 passed (up from 24; +6: q4_matvec/q4_vecmat delegate + native + dim-overflow rejection)
- `cargo test -p larql-compute --lib` → 734 passed, 2 ignored
- `cargo test -p larql-compute-vulkan` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
 - `cargo test -p larql-cli --bins` → 243 passed

Session 10 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 44 passed (up from 30; +14: 5 scaffold-fallback KV lifecycle + 2 runtime-gated native lifecycle + 1 runtime-gated realloc-on-larger-max_seq + 4 runtime-gated overflow/capacity rejection + 2 unit tests for KvCacheError Display / shape-mismatch)
- `cargo test -p larql-compute --lib` → 734 passed, 2 ignored
- `cargo test -p larql-compute-vulkan` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 11 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 53 passed (up from 44; +9 net: scaffold-fallback fused-pipelines-return-None + native-capability advertisement + 5 runtime-gated parity/mask tests + 2 review-fix runtime-gated parity tests [rope-scaled prefill, multi-token decode] − 1 dropped `erf_zero_is_zero` test for the removed dead `erf` helper; runtime-gated tests no-op on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 734 passed, 2 ignored
- `cargo test -p larql-compute-vulkan` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 12 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 58 passed (up from 53; +5: moe_outer_norm selection, moe decode/prefill composition parity [host-runnable], ple/remote-ffn bail [host-runnable], runtime-gated e2e MoE smoke)
- `cargo test -p larql-compute --lib` → 734 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 13 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 66 passed (up from 58; +8: 4 host-runnable fallback-contract parity [norm_2d RmsNorm/LayerNorm, norm_2d_no_weight, rms_norm_heads_array weighted+no-weight] + 4 runtime-gated native parity [body norm weighted/no-weight, per-head weighted/no-weight])
- `cargo test -p larql-compute --lib` → 734 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 13 review-fix delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 67 passed (up from 66; +1 host-runnable small-norm gate test; review fixes folded in: weighted per-head kernel now indexes `weight[d]` broadcast to match the CPU `rms_norm_heads` reference + real Gemma `[head_dim]` q/k norm weights, runtime guard relaxed to accept `head_dim`-length weights, 64-bit product guard added, and the pipeline norm helpers now gate the native dispatch on `NORM_NATIVE_MIN_ELEMS = 8192` so small/frequent norms keep the host reference instead of paying a per-call device round-trip — the body + no-weight per-head paths remain reachable + correct)
- `cargo test -p larql-compute --lib` → 734 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 14 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 76 passed (up from 67; +9: 4 host-runnable fallback-contract parity [gated SiLu/GeluTanh + standard SiLu/GeluTanh] + 4 runtime-gated native parity [native GEGLU-SiLu/GEGLU-GELU-tanh/activation-SiLu/activation-GELU-tanh vs host reference] + 1 runtime-gated dim-mismatch rejection; runtime-gated tests no-op on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 734 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 15 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 79 passed (up from 76; +3: host-runnable residual fallback-contract parity [unit + scaled arms], runtime-gated native residual parity [exact equality, both arms], runtime-gated dim-overflow rejection; runtime-gated tests no-op on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 734 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
 - `cargo test -p larql-cli --bins` → 243 passed

Session 16 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 82 passed (up from 79; +3: host-runnable rope fallback-contract parity [partial-rotation fraction + non-zero offset], runtime-gated native rope parity [pass-through exact, rotary ≤ 1e-5], runtime-gated invalid-shape rejection [inv_freq length + x/shape mismatch]; runtime-gated tests no-op on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 738 passed, 2 ignored (up from 734; +4 `build_rope_inv_freq` unit tests — the review-fix shared frequency-construction helper)
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 16 review-fix (shared frequency construction): a local code review flagged that `rope_native` re-derived the RoPE `inv_freq` table inline (base construction + `apply_llama3_inv_freq`), duplicating `apply_rope_partial_at_full`'s frequency logic and risking silent device/host numerical drift if the reference changed (the only signal would be a runtime-gated parity test that doesn't run on this no-CUDA host). Extracted a shared `build_rope_inv_freq(rope_base, head_dim, fraction, llama3_scaling) -> (rotary_dim, half_rotary, inv_freq)` helper in `larql_compute::attention::rope` (re-exported from `larql_compute::attention`) — now the single source of truth called by both the host reference and `rope_native`, so the uploaded frequencies are bit-identical to the reference's and can't drift. Added 4 unit tests pinning the helper's geometry (full/partial fraction, the `rotary_dim >= 2` floor) and the llama3 arm's composition with `apply_llama3_inv_freq`.

Session 17 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 85 passed (up from 82; +3: host-runnable below-gate decode-attention parity [bit-exact, host reference], runtime-gated native parity [≤ 1e-4 scale-relative bound for the device exp/tanhf libm difference], runtime-gated invalid-shape rejection [mismatched q/kv lengths + reps==0]; runtime-gated tests no-op on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 738 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
 - `cargo test -p larql-cli --bins` → 243 passed

Session 18 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green

Session 19 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 97 passed (up from 88; +9: 5 host-runnable `weight_cache` unit tests + 2 runtime-gated weight-cache parity/flush tests + 1 host-runnable `flush_clears_maps_without_device` + 1 runtime-gated f32-cache parity test added in the review fix)
- `cargo test -p larql-compute --lib` → 738 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 88 passed (up from 85; +3: host-runnable below-gate prefill-attention bit-exact parity, runtime-gated native parity [≤ 1e-4 relative, exercises softcap], runtime-gated invalid-shape rejection; runtime-gated tests no-op on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 738 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 20 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 101 passed (up from 97; +4: host-runnable device-chain scaffold bail + dispatch↔hostonly match, runtime-gated device-chain vs host reference parity [max_abs < 1e-3], host-runnable below-gate bail, runtime-gated zero-shape short-circuit; runtime-gated tests no-op on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 738 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 21 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 104 passed (up from 101; +3: host-runnable decode device-chain scaffold bail + dispatch↔hostonly bit-exact match, runtime-gated decode device-chain vs host reference parity [max_abs < 1e-3 on a synthetic large Q4_K FFN], host-runnable decode below-gate bail; runtime-gated tests no-op on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 738 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
 - `cargo test -p larql-cli --bins` → 243 passed

Session 22 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 107 passed (up from 104; +3: host-runnable prefill device-attention scaffold bail + dispatch↔hostonly match, runtime-gated prefill device-attention vs host reference parity [max_abs < 1e-3 on a synthetic large Q4_K attention layer], host-runnable prefill below-attention-gate bail; runtime-gated test no-ops on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 738 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 22 review-fix (3 perf findings from a local review): (1) the attention device chain now uses **distinct per-step bindings** (`q_proj`/`q_normed`/`q_rope` …) instead of rebinding `q_dev`/`k_dev`/`v_dev` mid-chain, so every intermediate `CudaSlice` drops at the block-end readback (after the single `sync_dtoh_f32`) rather than immediately — on devices without memory-pool support (`CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED == 0`) cudarc's `CudaSlice::drop` forces a stream `synchronize()`, so the rebinding form would have turned the single sync into ~6; the distinct-binding form (the same discipline the FFN chains use) makes the single-sync guarantee unconditional. (2) `inv_freq` is now **uploaded once** via a new `CudaRuntime::upload_f64` and shared by the Q + K RoPE launches through a new `launch_rope_dev_with_invfreq` (resident-`inv_freq` twin; `launch_rope_dev` delegates to it, single-source arg layout) — the prior form re-uploaded the frequency table on each RoPE call. (3) the norm `_dev` launchers (`launch_rms_norm_dev`/`launch_rms_norm_heads_dev`) no longer `w.to_vec()` the norm weight on every call — the `Some` arm uploads the caller's `&[f32]` directly (only the `None`-weight path uploads a one-element placeholder).

Session 22 review-fix delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 107 passed (unchanged count; the fixes are perf-only, no new tests)

Session 23 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 110 passed (up from 107; +3: host-runnable decode device-attention scaffold bail + dispatch↔hostonly exact match [also pins the K/V-row contract], runtime-gated device-vs-host parity [max_abs < 1e-3 on a synthetic large Q4_K attention layer with a pre-populated KV mirror; K/V rows < 1e-4], host-runnable decode below-attention-gate bail; runtime-gated test no-ops on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 738 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 24 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green (no warnings)
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 114 passed (up from 110; +4: scaffold bail, Q4_K structure-match vs independent q4k_matvec_into composition, non-Q4_K/non-aligned bail, runtime-gated native-vs-host parity; runtime-gated test no-ops on this no-CUDA host)
- `cargo test -p larql-compute --lib` → 738 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

Session 25 delta verified (CachyOS / x86_64-linux, rustc 1.96.0, no CUDA hardware on host):

- `cargo fmt --all -- --check` — clean (after applying `cargo fmt --all`)
- `cargo check -p larql-compute-cuda` — green
- `cargo check --workspace --exclude larql-python` — green (no warnings)
- `cargo check -p larql-cli --features cuda,vulkan` — green
- `cargo check -p larql-inference --features gpu-all` — green
- `cargo clippy -p larql-compute-cuda --tests -- -D warnings` — green
- `cargo clippy --workspace --exclude larql-python --exclude larql-compute-metal -- -D warnings` — green
- `cargo clippy -p larql-cli --features cuda,vulkan -- -D warnings` — green
- `cargo test -p larql-compute-cuda --lib` → 116 passed (up from 114; +2: host-runnable `moe_expert_chain_eligibility_gate` [pins all four bail arms + the eligible happy path] + runtime-gated `moe_expert_device_falls_back_when_chain_ineligible` [padded contraction forces chain bail; per-call fallback matches host ≤ 1e-3]; runtime-gated test no-ops on this no-CUDA host; the existing `moe_expert_contribution_native_matches_host_when_runtime_available` now exercises the chain end-to-end on a CUDA host)
- `cargo test -p larql-compute --lib` → 738 passed, 2 ignored
- `cargo test -p larql-compute-vulkan --lib` → 7 passed
- `cargo test -p larql-inference --lib` → 1243 passed, 4 ignored
- `cargo test -p larql-cli --bins` → 243 passed

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

**These are still mostly scaffold backends, but CUDA has started its first real runtime/kernel slice.** Current behavior:

- dense/quant ops mostly delegate to `CpuBackend`
- KV dispatch delegates to CPU
- async dispatch delegates to CPU
- capability reporting is conservative-ish but still scaffolded (see Phase 7 caveat below)

**Session 2 fixes in the scaffolds**:

- `crates/larql-compute-cuda/src/trait_impl.rs` and `crates/larql-compute-vulkan/src/trait_impl.rs` — rewrote `f16_gemv` inner loops to satisfy `clippy::needless_range_loop` (was failing `make ci`'s `-D warnings`).
- `crates/larql-compute-cuda/src/lib.rs` — fixed `q4_input_format_routes_like_cpu` test: it used 64 weight elements but `quantize_q4_k` requires a multiple of 256; bumped to `cols=128, rows=2` (256 elements).
- `crates/larql-inference/src/lib.rs` — fixed `unavailable_explicit_backend_errors_loudly` test: it used `expect_err` on `Result<Box<dyn ComputeBackend>, _>`, but `dyn ComputeBackend` isn't `Debug`, so the test didn't compile. Switched to a `match` on `Err`.

**Session 4 CUDA bring-up**:

- `crates/larql-compute-cuda/Cargo.toml` — added `cudarc` (dynamic loading + NVRTC).
- `crates/larql-compute-cuda/src/backend/runtime.rs` — new runtime/bootstrap layer that:
  - creates a CUDA context when available
  - compiles an embedded kernel with NVRTC
  - loads the module/function via `cudarc`
  - catches missing-`libcuda` probe panics so non-CUDA hosts fall back to the scaffold path instead of aborting tests
- `crates/larql-compute-cuda/src/ops.rs` — embedded a first CUDA `q4k_matvec` kernel source string.
- `crates/larql-compute-cuda/src/backend/mod.rs` / `trait_impl.rs` — wired optional native runtime state into the backend and routed `q4k_matvec` through CUDA when available, otherwise through the existing CPU fallback.
- `crates/larql-compute-cuda/src/lib.rs` — added tests for runtime-status reporting and CUDA-vs-CPU parity when the runtime is present.

**Session 5 CUDA kernel expansion**:

- `crates/larql-compute-cuda/src/ops.rs` — added `Q6K_MATVEC_CUDA_SRC` (one row per thread; decodes the 210-byte Q6_K super-block: 128-byte `ql` 4-bit, 64-byte `qh` 2-bit, 16 int8 scales, f16 `d`) and `Q4K_MATMUL_CUDA_SRC` (amortised: one thread per (row, seq) tile; decodes each 144-byte Q4_K super-block once and FMA's across all `seq` columns; output is `[seq, rows]` row-major to match the CPU `kquant_matmul_into` contract). Added `Q4K_MATMUL_KERNEL` handle; bumped `Q6K_MATVEC_KERNEL` threads to 128.
- `crates/larql-compute-cuda/src/backend/runtime.rs` — `CudaRuntime` now holds `q6k_matvec` + `q4k_matmul` function handles; `initialize_impl` concatenates all three kernel sources into one NVRTC translation unit and loads all three entry points from a single module. Added `launch_q6k_matvec` and `launch_q4k_matmul` (shape-checked, panic-safe upload/launch/sync/dtoh, mirroring `launch_q4k_matvec`).
- `crates/larql-compute-cuda/src/backend/mod.rs` — added `native_q6k_matvec` and `native_q4k_matmul` wrappers returning `Result<Option<_>, RuntimeError>`.
- `crates/larql-compute-cuda/src/trait_impl.rs` — `q6k_matvec` and `q4k_matmul` now try the native path first, falling back to `CpuBackend` on `Ok(None)` / `Err`.
- `crates/larql-compute-cuda/src/lib.rs` — added `native_q6k_matvec_matches_cpu_when_runtime_is_available` and `native_q4k_matmul_matches_cpu_when_runtime_is_available` (no-op when no CUDA runtime, as on this host).
- `cargo fmt --all` applied across the repo (folded in the prior pending reformatting of `larql-inference/src/lib.rs` and several CLI files).

**Session 6 CUDA kernel expansion**:

- `crates/larql-compute-cuda/src/ops.rs` — added `Q6K_MATMUL_CUDA_SRC` (amortised: one thread per (row, seq) tile; decodes each 210-byte Q6_K super-block once and FMA's across all `seq` columns; output `[seq, rows]` row-major) and `Q4K_DUAL_MATVEC_CUDA_SRC` (fused two-weight matvec sharing one `x`: one row per thread, decodes both `w_a` and `w_b` Q4_K super-blocks against the same `x` slice, writes `out_a[row]` and `out_b[row]`). Added `Q6K_MATMUL_KERNEL` + `Q4K_DUAL_MATVEC_KERNEL` handles (both 128 threads/group).
- `crates/larql-compute-cuda/src/backend/runtime.rs` — `CudaRuntime` now holds `q6k_matmul` + `q4k_dual_matvec` function handles; `initialize_impl` concatenates all five kernel sources into one NVRTC translation unit and loads all five entry points. Added `launch_q6k_matmul` and `launch_q4k_dual_matvec`. `q6k_matmul` is `#[allow(dead_code)]`-staged (no trait routing yet); `q4k_dual_matvec` is live.
- `crates/larql-compute-cuda/src/backend/mod.rs` — added `native_q6k_matmul` (staged, `#[allow(dead_code)]`) and `native_q4k_dual_matvec` (live, `#[allow(clippy::type_complexity)]` for the `(Vec<f32>, Vec<f32>)` return).
- `crates/larql-compute-cuda/src/trait_impl.rs` — `q4k_dual_matvec` now tries the native path first, falling back to `CpuBackend`. `q6k_matmul` has no trait method (see Honest Status), so no routing change.
- `crates/larql-compute-cuda/src/lib.rs` — added `q4k_dual_matvec_matches_cpu_delegate` (always runs; exercises the CPU fallback), `native_q6k_matmul_matches_cpu_when_runtime_is_available` and `native_q4k_dual_matvec_matches_cpu_when_runtime_is_available` (runtime-gated).

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

- `ash` / `shaderc` dependencies
- SPIR-V kernel compilation
- real `prefill_kquant` ~~— DONE in Session 11 (host-orchestrated).~~ Native q4k/q6k matmul projections across all `seq_len` positions + causal GQA attention + elementwise ops on host + host KV mirror; capability advertised (`supports(PrefillQ4)`). Not a single-command-buffer fused kernel — folding elementwise ops into device kernels is the follow-on.
- real `decode_token` ~~— DONE in Session 11 (host-orchestrated).~~ Native q4k/q6k matvec projections + GQA decode attention over the host KV mirror + elementwise ops on host; capability advertised (`supports(DecodeToken)`). Same caveat as prefill.
- real `decode_token_with_state_dump_masked` ~~— DONE in Session 11.~~ `Full`/`HOnly`/`None` mask respected; per-layer `h_in`/`k_new`/`v_new` captured on the host walk (near-zero cost since the loop is already serial).
- real KV cache lifecycle on device (`has_kv_cache`, `reset_kv_cache`, `kv_cache_len`, `truncate_kv_cache`, `preallocate_kv_cache_per_layer`) ~~— DONE in Session 10.~~ The device-side `CudaKVCache` (mirroring Metal) is allocated lazily via `preallocate_kv_cache_per_layer`; `has_kv_cache` reports true only with a runtime *and* an allocated cache; `reset`/`len`/`truncate` are cursor-only; `populate_kv_layer` appends rows via the native `kv_append` kernel. The scaffold (no-device) path keeps all of these as no-ops so engines route KV through the CPU reference store.
 - `f32_gemv` / `f16_gemv` / most remaining `q4k_*` / `q6k_*` device kernels ~~— `f32_gemv`/`f16_gemv` landed in Session 8;~~ CUDA now has `q4k_matvec`, `q6k_matvec`, `q4k_matmul`, `q6k_matmul`, `q4k_dual_matvec`, `f32_gemv`, `f16_gemv`, `q4_matvec`, `q4_vecmat`, `kv_append`, `rms_norm`, `rms_norm_heads`, `geglu_silu`, `geglu_gelu_tanh`, `activation_silu`, `activation_gelu_tanh`, `residual_add`, `rope`, `decode_attention`, and `prefill_attention`; the full `QuantMatVec` kernel surface is native, the KV-append primitive is native, all four elementwise families (body + per-head RMSNorm in Session 13, FFN activation in Session 14, residual add in Session 15, RoPE in Session 16) are native, the decode-step attention is native (Session 17), and the prefill (seq×seq) causal attention is native (Session 18) (the device-kernel-fusion follow-on)
- routing of the amortised Q6_K matmul through a backend ~~— DONE in Session 7.~~ `QuantMatVec::q6k_matmul` added (default `None`); CpuBackend wraps `q6k_matmul_into`; CUDA routes native-then-CPU; the Q6_K arm of `ffn/weight.rs::quant_matmul` dispatches through a backend's `q6k_matmul` when supplied (attention `gpu.rs` Q/K/V/O pass the backend; `Q4kMatmulFfn` passes `None`).
- real coarse `KvDispatch` (currently delegates to CPU)
- real `AsyncComputeBackend` batching (currently delegates to CPU)
- hardware-specific CI jobs for CUDA and Vulkan

What exists now is:

- the repo-wide control plane from Sessions 1-3
- plus a real CUDA runtime/bootstrap path (`cudarc` + NVRTC)
- plus nineteen native CUDA kernels — five k-quant (`q4k_matvec`, `q6k_matvec`, `q4k_matmul`, `q6k_matmul`, `q4k_dual_matvec`), two dense (`f32_gemv`, `f16_gemv`), two legacy Q4_0 (`q4_matvec`, `q4_vecmat`), `kv_append`, two elementwise RMSNorm (`rms_norm`, `rms_norm_heads`, Session 13), four elementwise activations (`geglu_silu`, `geglu_gelu_tanh`, `activation_silu`, `activation_gelu_tanh`, Session 14), the residual add (`residual_add`, Session 15), RoPE (`rope`, Session 16), and the fused decode-attention (`decode_attention`, Session 17) — behind the existing CPU fallback. All five k-quant kernels and the two Q4_0 kernels are trait-routed through `QuantMatVec`; the dense GEMVs are trait-routed through `MatMul` (Session 8); `kv_append` is the device-side KV-write primitive backing the `DecodeBackend` lifecycle methods (Session 10); the two RMSNorm kernels are routed through the pipeline's `norm_2d`/`norm_1d`/`norm_2d_no_weight`/`rms_norm_heads_array` helpers (Session 13); the four activation kernels are routed through the pipeline's `apply_activation_gated_native`/`apply_activation_std_native` helpers (Session 14 — the second native elementwise step toward the fully-fused pipeline); the residual-add kernel is routed through the pipeline's `add_residual_native` helper (Session 15 — the third native elementwise step); the RoPE kernel is routed through the pipeline's `rope_native` helper (Session 16 — the fourth native elementwise step); the decode-attention kernel is routed through the pipeline's `decode_attention_native` helper (Session 17 — the first non-elementwise fused device kernel, completing the device-elementwise + decode-attention fusion follow-on).
- plus a device-side KV cache (`CudaKVCache`) with the full `DecodeBackend` lifecycle wired through it (Session 10) — the foundation for the fused `prefill_kquant`/`decode_token` pipelines.

That is enough to continue Phase 4 on the prefill/decode path.

## Remaining Work (in suggested order)

1. ~~compile + repair~~ DONE
2. ~~CLI migration: `run`/`walk`/`bench`/`shannon` accept `--backend`~~ DONE. Remaining polish:
   - sweep for stale Metal-only help text/comments across the migrated commands
   - `walk` Q4 predict/generate now constructs backends generically; `run` still has Metal-specific construction in remote FFN/MoE and `--experts` branches.
3. ~~tighten capability reporting in CUDA/Vulkan scaffolds per the plan's Phase 7~~ DONE in Session 3. Delegated scaffold methods still exist for parity tests, but `supports(...)` and `supports_quant(...)` report false until native kernels land.
4. continue replacing delegated CUDA hot paths with real kernels (Phase 4):
    - ~~`f32_gemv` / `f32_gemv_topk1`~~ — landed Session 8 (trait-routed via `MatMul`; `topk1` inherits it); still not capability-advertised
    - ~~`f16_gemv` / `f16_gemv_topk1`~~ — landed Session 8 (trait-routed via `MatMul`; `topk1`/`topk` inherit it); still not capability-advertised
    - ~~`q4k_matvec`~~ — landed Session 4; still not capability-advertised
    - ~~`q4k_matmul`~~ — landed Session 5; still not capability-advertised
    - ~~`q6k_matvec`~~ — landed Session 5; still not capability-advertised
    - ~~`q4k_dual_matvec`~~ — landed Session 6 (trait-routed); still not capability-advertised
    - ~~`q6k_matmul`~~ — landed Session 6 (kernel) + Session 7 (trait-routed: `QuantMatVec::q6k_matmul` + `ffn/weight.rs::quant_matmul` Q6_K arm dispatch); still not capability-advertised
    - `q4_matvec` / `q4_vecmat` ~~— landed Session 9 (trait-routed via `QuantMatVec`); still not capability-advertised~~
    - `prefill_kquant` ~~— DONE in Session 11 (host-orchestrated; native matmul projections + host causal attention + host KV mirror; `supports(PrefillQ4)` advertised)~~
    - `decode_token` ~~— DONE in Session 11 (host-orchestrated; native matvec projections + host GQA attention over the KV mirror; `supports(DecodeToken)` advertised)~~
    - `decode_token_with_state_dump_masked` ~~— DONE in Session 11 (`Full`/`HOnly`/`None` mask respected; per-layer state captured on the host walk)~~
    - KV cache lifecycle ~~— DONE in Session 10~~ (`CudaKVCache` + `DecodeBackend` lifecycle methods + native `kv_append` kernel; the scaffold path falls back to no-ops)
    - coarse `KvDispatch` bridge
    - (routing) ~~add `q6k_matmul` to `QuantMatVec` + dispatch `ffn/weight.rs::quant_matmul`'s Q6_K arm through the backend~~ DONE in Session 7 — the staged CUDA kernel is live
    - (Session 11 follow-on) fold the host elementwise ops (RMSNorm / QK-norm / RoPE / softmax / V-norm / GEGLU / residual) into device kernels + collapse the per-projection htod/launch/dtoh round-trips into a single-command-buffer fused pipeline (mirrors Metal's shape). The host-orchestrated path is correct and unblocks routing today, but is not the perf target.
      - ~~RMSNorm (body + per-head)~~ — DONE in Session 13 (native `rms_norm`/`rms_norm_heads` kernels; routed through the pipeline's norm helpers; parity-tested). The first device-elementwise step.
      - ~~FFN activation (GEGLU SiLu/GeluTanh + standard SiLu/GeluTanh)~~ — DONE in Session 14 (native `geglu_silu`/`geglu_gelu_tanh`/`activation_silu`/`activation_gelu_tanh` kernels; routed through the pipeline's `apply_activation_gated_native`/`apply_activation_std_native` helpers with an `ACTIVATION_NATIVE_MIN_ELEMS = 8192` gate; parity-tested). The second device-elementwise step.
      - ~~Residual add (`out = h + b_scale * x`)~~ — DONE in Session 15 (native `residual_add` kernel; routed through the pipeline's `add_residual_native` helper with a `RESIDUAL_NATIVE_MIN_ELEMS = 8192` gate, all 8 residual call sites in the decode/prefill attention+FFN blocks; parity-tested, exact equality since `fmad` is disabled at NVRTC compile time). The third device-elementwise step.
      - ~~RoPE (Q/K rotary embedding)~~ — DONE in Session 16 (native `rope` kernel; the device twin of `apply_rope_partial_at_full`, split-half pairing + pass-through tail; `inv_freq` precomputed on host incl. `llama3` scaling, uploaded as `double` so theta/cos/sin match the f64 reference; routed through the pipeline's `rope_native` helper with a `ROPE_NATIVE_MIN_ELEMS = 8192` gate, all 4 Q/K call sites in the decode/prefill attention blocks; parity-tested — pass-through exact, rotary ≤ 1e-5 since device double cos/sin are a different libm). The fourth device-elementwise step. Remaining host elementwise ops: collapsing the per-projection round-trips into a single-command-buffer fused pipeline (mirroring Metal's shape).
      - ~~decode-step attention (QKᵀ → softmax → weighted-V)~~ — DONE in Session 17 (native `decode_attention` kernel; one thread-block per query head fusing QKᵀ → scale (+ optional softcap) → softmax → weighted-V over the full KV cache; f32 QKᵀ/weighted-V + f64 softmax sum matching the reference's rounding order; routed through the pipeline's `decode_attention_native` helper with a `DECODE_ATTN_NATIVE_MIN_WORK = 8192` work gate; parity-tested — host-runnable below-gate bit-exact, runtime-gated native ≤ 1e-4 scale-relative). **Completes the device-elementwise + decode-attention fusion follow-on.** Remaining: ~~prefill (seq×seq) device attention~~ DONE in Session 18 (native `prefill_attention` kernel; one thread-block per `(query head, query position)` fusing the causal QKᵀ → scale → softmax → weighted-V over `causal_len = qi+1` keys using dynamic shared memory for the block-local `scores` scratch; routed through the pipeline's `prefill_attention_native` helper with a `PREFILL_ATTN_NATIVE_MIN_WORK = 8192` work gate + a 48 KB shared-mem budget guard; parity-tested) + the single-command-buffer round-trip collapse.
      - round-trip collapse — **Session 19 landed the first slice**: a persistent device weight cache (`weight_cache.rs`) that uploads each immutable weight matrix once and reuses the device buffer across calls (keyed on the host slice's `(ptr, element-count)`; flushed at each `reset_kv_cache` as an ABA guard). All nine weight-bearing launchers (`q4k`/`q6k` matvec+matmul, `q4k_dual_matvec`, `f32`/`f16` gemv, `q4_matvec`, `q4_vecmat`) now route their weight arg through the cache; activations stay on the fresh `clone_htod` path. This removes the dominant per-token GB-scale weight re-upload on the decode hot path. **Session 20 landed the second slice (prefill activation chaining)**: device-resident launch variants (`*_dev` on `CudaRuntime` — input `&CudaSlice<f32>`, output `CudaSlice<f32>`, no internal htod/sync/dtoh; `upload_f32` + `sync_dtoh_f32` bracket a chain) + a fused device-resident prefill FFN chain (`host_prefill_ffn_block_device`: norm(host)→gate/up matmul→activation→down matmul, all chained on one stream with a single readback). **Session 21 landed the third slice (decode activation chaining)**: the decode-path twin — device-resident matvec launchers (`launch_q{k}_matvec_dev`) + a fused device-resident decode FFN chain (`host_ffn_block_device`: norm(host)→gate/up matvec→activation→down matvec, single readback); the host matvec launchers now delegate to the `_dev` variants (single-source arg layout). An N-kernel FFN chain now pays one sync+dtoh instead of N, for both prefill and decode. **Session 22 landed the fourth slice (prefill attention activation chaining)**: the attention-path twin of the FFN chains — four new device-resident launch variants (`launch_rms_norm_dev`/`launch_rms_norm_heads_dev`/`launch_rope_dev`/`launch_prefill_attention_dev`; the four host launchers now delegate to them, single-source arg layout) + a fused device-resident prefill attention chain (`host_prefill_attention_block_device`: input norm(host)→Q/K/V matmul→QK-norm/V-norm/RoPE→causal attention→O matmul, all chained on one stream; the Q/K/V share one resident normed input and stay resident through RoPE + attention; one sync+dtoh for O plus the K/V read back into the host KV mirror). **Session 23 landed the fifth slice (decode attention activation chaining)** — the final slice of the round-trip collapse: the decode-path twin of the prefill attention chain — a new `launch_decode_attention_dev` (resident Q + uploaded full KV → resident output; the host `launch_decode_attention` delegates to it) + a fused device-resident decode attention chain (`host_attention_block_device`: input norm(host)→Q/K/V matvec→QK-norm/V-norm/RoPE→decode-attention→O matvec; the Q/K/V share one resident normed input and stay resident through RoPE + attention; one sync+dtoh to read back the new K/V row needed to build the full KV, plus one final readback of O — 2 syncs vs ~8 per-op round trips on the host path). **The dense decode/prefill projection chains (FFN + attention) now all run device-resident with a single end-of-chain readback; weights are cached once.** Remaining collapse work: a resident device KV cache so the decode attention reads the growing KV without the per-token K/V-row readback + full-KV upload (the `CudaKVCache` from Session 10 holds it, but the decode chain rebuilds+uploads from the host mirror today — the device-KV-cache append + resident-KV decode-attention is the final single-command-buffer slice), threading `h`/`h_post_attn` resident across layers (today the per-block residual stays host-side), and batching the launches into a single command buffer (mirroring Metal's `decode/mod.rs`).
    - (Session 11 follow-on) extend the fused pipeline to MoE / PLE / remote-FFN layer features (currently bail to `None` → CPU fallback).
      - ~~MoE~~ — DONE in Session 12 (host-orchestrated dense slab delta + substrate `cpu_moe_forward` expert block + `outer_post_norm_residual`; runs through `decode_token`/`prefill_kquant`; composition parity host-runnable). ~~Expert projections still run on CPU~~ — Session 24 routed the per-expert gate/up/down Q4_K matvecs through the native CUDA `q4k_matvec` kernel (Q4_K × f32; parity oracle = CPU `q4k_matvec_into`, NOT `cpu_moe_forward`'s Q8_K-direct SDOT). The device path only fires on CUDA hosts with Q4_K experts + 256-multiple hidden; non-CUDA hosts / BF16 experts keep the `cpu_moe_forward` path unchanged. Remaining MoE follow-on: device-*chaining* the per-expert FFN (single-readback chain mirroring Sessions 20-23; expert weights are already cached via the Session 19 weight cache).
      - PLE — still bails: needs the precomputed per-layer embedding input (token-embedding-derived, not on the `FullPipelineLayer`/trait surface); would require extending the trait surface or an engine-layer hook.
      - remote-FFN — still bails: needs a dispatch callback; only `decode_token_with_moe` carries one (implementing that trait method to route remote-FFN through the callback while running the rest native is the follow-on).
5. then do the same for Vulkan (Phase 5), keeping structure parallel to CUDA and Metal
6. integration tests mirroring Metal decode integration style (prefill shape, KV length, masked state dump `Full`/`HOnly`/`None`, heterogeneous per-layer KV preallocation, coarse `KvDispatch` prefill/decode with Q4K fixtures)
7. hardware-specific CI jobs for CUDA and Vulkan

## Key Files Added

- `crates/larql-compute-cuda/Cargo.toml`
- `crates/larql-compute-cuda/src/lib.rs`
- `crates/larql-compute-cuda/src/backend/mod.rs`
- `crates/larql-compute-cuda/src/backend/runtime.rs`
- `crates/larql-compute-cuda/src/options.rs`
- `crates/larql-compute-cuda/src/kernels.rs`
- `crates/larql-compute-cuda/src/buffers.rs`
- `crates/larql-compute-cuda/src/ops.rs`
- `crates/larql-compute-cuda/src/decode.rs`
- `crates/larql-compute-cuda/src/trait_impl.rs`
- `crates/larql-compute-cuda/src/kv_cache.rs`
- `crates/larql-compute-cuda/src/pipeline.rs`
- `crates/larql-compute-cuda/src/weight_cache.rs`
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

Phases 1-3 are done and verified green (workspace compiles, clippy clean, ~1508 tests pass across the affected crates).

Phase 4 is **underway in CUDA**: `cudarc`/NVRTC are wired, runtime fallback is panic-safe on non-CUDA hosts, and twenty native kernels are live — five k-quant (`q4k_matvec`, `q6k_matvec`, `q4k_matmul`, `q6k_matmul`, `q4k_dual_matvec`) plus two dense (`f32_gemv`, `f16_gemv`) plus two legacy Q4_0 (`q4_matvec`, `q4_vecmat`) plus `kv_append` plus two elementwise RMSNorm (`rms_norm`, `rms_norm_heads`) plus four elementwise activations (`geglu_silu`, `geglu_gelu_tanh`, `activation_silu`, `activation_gelu_tanh`) plus the residual add (`residual_add`) plus RoPE (`rope`) plus the fused decode-attention (`decode_attention`) plus the fused prefill-attention (`prefill_attention`, Session 18) — each behind its CPU fallback with parity tests. As of Session 7 **all five k-quant kernels are trait-routed** through `QuantMatVec`, and the `q6k_matmul` kernel is live end-to-end: the Q6_K arm of `ffn/weight.rs::quant_matmul` dispatches through a backend's `q6k_matmul` (attention `gpu.rs` passes its backend; `Q4kMatmulFfn` keeps the CPU path). Session 8 added the dense `f32_gemv`/`f16_gemv` kernels trait-routed through `MatMul`, with `f32_gemv` only taking the native path on row-major-contiguous `ArrayView2`s (non-contiguous views fall back to the CPU reference; `f32_gemv_topk1`/`f16_gemv_topk1`/`f16_gemv_topk` inherit the routing). Session 9 added the legacy Q4_0 `q4_matvec`/`q4_vecmat` kernels trait-routed through `QuantMatVec`, completing the `QuantMatVec` kernel surface. Session 10 landed the device-side KV cache (`CudaKVCache`, mirroring Metal's `LayerKVCache`/`KVCache` via `cudarc::driver::CudaSlice<f32>` owned device buffers) + the native `kv_append` kernel and wired the full `DecodeBackend` KV lifecycle (`has_kv_cache`/`preallocate_kv_cache_per_layer`/`reset_kv_cache`/`kv_cache_len`/`truncate_kv_cache`/`populate_kv_layer`) through it — the foundation for the fused `prefill_kquant`/`decode_token` pipelines. Sessions 13-17 continued the device-kernel-fusion follow-on: native elementwise RMSNorm (`rms_norm`/`rms_norm_heads`, Session 13) + native FFN activation (`geglu_silu`/`geglu_gelu_tanh`/`activation_silu`/`activation_gelu_tanh`, Session 14) + native residual add (`residual_add`, Session 15) + native RoPE (`rope`, Session 16) + native fused decode-attention (`decode_attention`, Session 17), all routed through the pipeline's helpers with min-elems/work gates. Vulkan Phase 5 and hardware CI are still not started. Phase 7's honesty pass remains in force: capabilities stay as Session 11 set them (the native decode-attention is a perf refinement of the already-advertised `DecodeToken` path, not a new capability).

The next session should continue Phase 4 inside `larql-compute-cuda`: the fused `prefill_kquant`/`decode_token`/`decode_token_with_state_dump_masked` pipelines are **live** (Session 11, host-orchestrated) and capability advertisement is on, so `fused_prefill`/`fused_decode_step` route through CUDA and the `auto` policy on Linux picks CUDA first. **Session 12 extended the host pipeline to hybrid-MoE layers** (Gemma 4 26B-A4B): `host_ffn_block_moe_decode`/`host_prefill_ffn_block_moe` compose the dense-slab delta + substrate `cpu_moe_forward` expert block + `outer_post_norm_residual`, so MoE layers no longer bail to `None`→CPU when reached (PLE and remote-FFN still bail — they need data/callbacks not on the trait surface). **Session 24 routed the per-expert gate/up/down Q4_K matvecs through the native CUDA `q4k_matvec` kernel** (Q4_K × f32; parity oracle = CPU `q4k_matvec_into`, NOT `cpu_moe_forward`'s Q8_K-direct SDOT — that's Apple-Silicon-only; the device path only fires on CUDA hosts with Q4_K experts + 256-multiple hidden, so non-CUDA hosts keep the `cpu_moe_forward` path unchanged). Remaining MoE follow-on: device-*chaining* the per-expert FFN (single-readback chain mirroring Sessions 20-23; expert weights already cached via Session 19). **Sessions 13-17 continued the perf follow-on** by landing the four native elementwise families + the fused decode-attention: `rms_norm` (body) + `rms_norm_heads` (per-head, Session 13), `geglu_silu`/`geglu_gelu_tanh`/`activation_silu`/`activation_gelu_tanh` (FFN activation, Session 14), `residual_add` (the post-attention/post-FFN residual, Session 15), `rope` (Q/K rotary embedding, Session 16), and `decode_attention` (the fused decode-step QKᵀ→softmax→weighted-V, Session 17), all routed through the pipeline's helpers (native-then-CPU fallback with min-elems/work gates) — the four highest-frequency elementwise ops AND the decode attention itself are now off the host. **Session 18 landed the fused prefill (seq×seq) causal attention** (`prefill_attention`, one thread-block per `(query head, query position)` over `causal_len` keys, dynamic shared memory for the block-local `scores` scratch; routed through `prefill_attention_native` with a work gate + 48 KB shared-mem budget guard; parity-tested) — both attention primitives are now native. **Session 19 began the round-trip collapse** with a persistent device weight cache (`weight_cache.rs`): all nine weight-bearing launchers upload each immutable weight matrix once (keyed on `(ptr, element-count)`) and reuse the device buffer across calls, flushed per-generation at `reset_kv_cache` as an ABA guard — removing the dominant per-token GB-scale weight re-upload on the decode hot path while keeping activations on the fresh upload path. **Sessions 20-22 continued the activation-residency collapse**: the device-resident launch variants (`*_dev`) + fused device-resident chains for the prefill FFN (Session 20), decode FFN (Session 21), and prefill attention (Session 22) blocks now keep per-projection activations resident on-device across each block (one sync+dtoh per block instead of per-projection). The remaining Phase 4 perf work is the **rest of the activation-residency + single-command-buffer collapse**: the decode-path attention device chain (the Session 21 twin of Session 22 — needs a resident K/V concatenation since decode attention reads the growing host KV mirror), threading `h`/`h_post_attn` resident across layers (today the per-block residual stays host-side), and batching the launches into a single command buffer (mirroring Metal's `decode/mod.rs` shape) — the host-orchestrated path is the parity oracle for that. After that: PLE (needs trait-surface/engine-hook work) and remote-FFN (implement `decode_token_with_moe` to route the callback) layer features, the coarse `KvDispatch` bridge, then Vulkan Phase 5, then hardware CI. With the k-quant matmul/matvec + dense GEMV + legacy Q4_0 kernels all trait-routed, the KV cache lifecycle in place, the fused decode/prefill pipelines routing through CUDA, the MoE host pipeline landed, the four elementwise families (RMSNorm + activation + residual add + RoPE) native, both attention primitives native, the weight re-uploads collapsed, and the FFN + prefill-attention activation round-trips collapsed, the remaining Phase 4 surface is the decode-attention activation-residency + cross-layer residency + single-command-buffer collapse + the `KvDispatch` bridge + the PLE/remote-FFN trait-surface work.


---

## Session 27 (2026-07-10): GPU-007 / B2 hardware-validated

**GPU-007 (cross-layer hidden-state residency) is HARDWARE-VALIDATED on RTX 3090.**

Validated `feat/cuda-cross-layer-hidden-state-residency` (commit `f9f0fde0`):
- 161/161 tests green (160 lib + 1 hardware_probe) with default settings
- All 7 resident_hidden tests pass natively (including mixed-eligibility
  transitions: Host→Device→Host→Device→final Host)
- Three-repeat stability confirmed (no stale buffers, no ownership defects)
- Release-mode focused tests pass (resident_hidden + resident_kv)
- ASTAB-001 decode parity preserved (1e-3 tolerance unchanged)
- GPU-006 resident_kv: no regression (8/8 pass)
- CPU/CUDA e2e parity confirmed on synthetic Q4_K vindex

Three defects found and fixed during validation:
1. Missing `ensure_device()` — hidden state started as Host, device-resident
   attention required Device, so the resident path never fired (pipeline.rs)
2. Test fixture `inter=256` below activation gate `8192` — added
   `make_test_q4k_weights_inter(8192)` for resident-hidden tests (test_fixtures.rs)
3. Mixed-eligibility test assumed RmsNorm with zero weights == LayerNorm with
   zero weights; false due to Gemma 3's `norm_offset=1.0` (lib.rs)

B2 is marked COMPLETE in `docs/cuda-vulkan-completion-plan.md`.
Baseline report: `bench/baselines/cuda-cross-layer-residency-2026-07-10.md`.

Remaining Phase 4 surface: B3 (launch batching / graphization), PLE/remote-FFN
trait-surface work, `KvDispatch` bridge, then Vulkan Phase 5, then hardware CI.

---

## Session 28 (2026-07-10): LARQL-GPU-B3A — CUDA Graph replay for the resident decode FFN (COMPLETE, OPT-IN)

**B3A is mid-implementation on `feat/cuda-resident-ffn-graph` (base `6ed40296`).**
The infrastructure (smoke test, plan contract, into-buffer launchers, arena,
graph state, cache, counters) is committed and tested; the pipeline integration
(B3A-5: build + replay in the decode loop) is the remaining piece.

### Completed (7 commits, all RTX 3060-validated where GPU-dependent)

- **B3A-SMOKE** (`1f7a5d1e`): minimal native CUDA Graph smoke test —
  capture/instantiate/replay/teardown on a dedicated stream with the real
  `residual_add` kernel. **PASS on RTX 3060.** Surfaced two critical findings:
  (1) the NULL default stream cannot be captured (`STREAM_CAPTURE_UNSUPPORTED`)
  — graph capture requires a dedicated non-NULL stream via `new_stream()`;
  (2) cudarc's per-slice CudaEvent tracking injects `cuStreamWaitEvent` during
  capture → `STREAM_CAPTURE_ISOLATION` — event tracking must be disabled for
  graph buffers. The test validates stable-pointer replay (in-place
  `memcpy_htod` mutating buffer contents changes the replay output — the core
  ping-pong invariant) + a 3-cycle create→replay→reset→rebuild→drop lifecycle.

- **B3A-0**: post-D6 baseline recorded on the fresh binary (Q4_K_M,
  `LARQL_CUDA_GRAPHS=0`, instrumented): `launches=599.2/tok, htod=3.3/tok,
  dtoh=78.1/tok, syncs=78.1/tok, hidden_readback=2.31ms/tok, p50=130.3ms/tok,
  7.64 tok/s`. Confirms resident-KV + resident-hidden 100% active on the
  default Q4_K_M. FFN chain = 7 kernels/layer × 36 = 252 host submissions/token
  (the graph-replay target → 36 graph submissions = ~34% structural reduction).

- **B3A-1**: cudarc 0.19.8 graph API audited — full lifecycle (begin_capture →
  end_capture → CudaGraph::launch → Drop) available via the safe API. No
  upgrade, no RAII wrapper, no sys binding needed.

- **B3A-2** (`b33a7505`): `ffn_graph.rs` — the pure plan contract +
  generation identity. `GraphGenerationId`, `ResidentFfnPlan` +
  `ResidentFfnPlanKey`, `plan_graph_eligible`, `GraphMode` +
  `graph_mode_from_env`. Cache key = `(generation, layer_index, plan)` — NOT
  host pointer/len (review point 6). 26 host-runnable tests PASS.

- **B3A-4** (`0e5c2777`): into-buffer launch primitives — `*_into` variants of
  the 7 FFN kernels (q4k/q6k matvec, geglu/activation, residual_add, rms_norm)
  that write into pre-allocated `&mut CudaSlice<f32>`. The `*_dev` launchers
  delegate to `*_into` (parity-preserving — 200 tests PASS). Norm weights get
  `upload_rms_norm_weight` (stable device address for capture).

- **B3A-3/6** (`9e35345e`): `ffn_graph_state.rs` — `ResidentDecodeArena`
  (ping-pong hidden_a/hidden_b + dedicated capture stream, event-tracking
  disabled), `ResidentFfnGraph` (graph + scratch + retained weights with
  explicit Drop ordering: exec graph → scratch → weights), and the explicit
  `Send+Sync` asserts (CudaGraph isn't internally synchronized, but LARQL's
  single-threaded decode + Mutex access provides the external serialization).

- **B3A-7** (`69641643` + `88995dd8`): per-backend `graph_mode` field +
  `GraphProfileCounters` (7 capture-aware counters: graph_submissions,
  graph_builds, captured_kernel_nodes, logical_graph_kernel_executions,
  graph_failures, graph_fallbacks, d2d_submissions) + the generation-scoped
  graph cache + arena wired onto CudaBackend. `reset_kv_cache_native` extended
  with the B3A-6 teardown order (graphs destroyed → arena dropped → weight
  cache flushed → generation advanced). 200 tests PASS.

### All B3A tasks COMPLETE (commits `1f7a5d1e`..`ac922262`)

- **B3A-5** (pipeline integration): DONE. `host_ffn_block_device_resident_graph`
  slots in at pipeline.rs:430, builds the graph on token 1 (7 `*_into` launches
  captured on cap_stream), launches it (review point 1), replays on tokens 2+,
  handles cross-stream sync (D2D seed + synchronize + D2D read), and falls back
  to the existing resident device FFN chain on any failure. 204/204 tests pass.
- **B3A-8** (tests): DONE. 4 dedicated graph tests (build/replay/reset/disabled
  with EXACT counter assertions). Q4_K_M parity < 1e-3 under GRAPHS=1.
- **B3A-9** (regression): DONE. 204/204 pass both GRAPHS=0 and GRAPHS=1 (serial).
- **B3A-10** (benchmark): DONE. Same-day A/B on real Qwen2.5-3B Q4_K_M.
- **B3A-11** (gate): DONE. 36.6% submission reduction (≥25% ✅); −0.18%
  wall-clock (≥1% ❌). **Graph stays opt-in (default Disabled).**
- **B3A-12** (docs): DONE. Baseline report + completion-plan + this entry.

### B3A-10 measured result (honest)

| metric | GRAPHS=0 | GRAPHS=1 | gate |
|---|---|---|---|
| median p50 ms/tok | 121.99 | 122.21 | −0.18% (❌ ≥1%) |
| launches/tok | 599.2 | 379.8 | −36.6% (✅ ≥25%) |

The 36.6% submission reduction is real but the cross-stream sync overhead
(2 synchronize/layer = 72/token) offsets the launch savings. B3B (event-based
sync or attention graphization) is the path to the ≥1% wall-clock gate.

### Key design constraints (from B3A review + smoke-test findings)

- **Dedicated capture stream** (NULL stream can't be captured); event tracking
  disabled for graph buffers.
- **D2D copies at the layer boundary** (seed input + read output) are counted
  in `d2d_submissions` and honest against the ≥25% gate. The zero-D2D design
  (attention's residual-add writes directly into the arena) is a B3B refinement.
- **Default graph mode = Disabled** (B3A-11 gate: wall-clock improvement < 1%);
  opt-in via `LARQL_CUDA_GRAPHS=1`.

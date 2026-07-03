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

**Vulkan is still a pure scaffold backend. CUDA now has eighteen native kernels** — `q4k_matvec`, `q6k_matvec`, `q4k_matmul`, `q6k_matmul`, `q4k_dual_matvec`, `f32_gemv`, `f16_gemv`, `q4_matvec`, `q4_vecmat`, `kv_append`, `rms_norm`, `rms_norm_heads`, `geglu_silu`, `geglu_gelu_tanh`, `activation_silu`, `activation_gelu_tanh`, `residual_add`, and `rope` — each behind its CPU fallback and gated by an optional runtime. As of Session 7, **all five k-quant kernels are trait-routed** through `QuantMatVec` (`q4k_dual_matvec` since Session 6, `q6k_matmul` since Session 7); the Q6_K arm of `ffn/weight.rs::quant_matmul` now dispatches through a backend's `q6k_matmul` when one is supplied, so the staged `q6k_matmul` CUDA kernel is live. Session 8 added the dense `f32_gemv`/`f16_gemv` kernels trait-routed through `MatMul`, with `f32_gemv` only taking the native path on row-major-contiguous `ArrayView2`s. Session 9 added the legacy Q4_0 `q4_matvec`/`q4_vecmat` kernels trait-routed through `QuantMatVec`, completing the `QuantMatVec` kernel surface. Session 10 added the device-side KV cache (mirroring Metal) + the native `kv_append` kernel and wired the `DecodeBackend` KV lifecycle methods (`has_kv_cache`/`preallocate_kv_cache_per_layer`/`reset_kv_cache`/`kv_cache_len`/`truncate_kv_cache`/`populate_kv_layer`) through it. **Session 11 landed the fused `decode_token`/`decode_token_with_state_dump_masked`/`prefill_kquant` pipelines** as a host-orchestrated path (native matvec/matmul projections + host elementwise ops + host KV mirror) and flipped capability advertisement on (`supports_quant(Q4_K/Q6_K)`, `supports(DecodeToken/PrefillQ4/QuantMatVec)`) when a runtime is present — so `fused_prefill`/`fused_decode_step` now route through CUDA and the `auto` policy on Linux picks CUDA first for dense k-quant models (hybrid-MoE models route to the CPU path via a `DecodeMoe`-capability gate; see the Session 11 review fixes). Everything else in CUDA (the coarse `KvDispatch` bridge, device-side attention kernels, PLE/remote-FFN fused paths, routing the expert matvecs through native CUDA kernels) and all of Vulkan still delegate to CPU/reference paths.

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
- `f32_gemv` / `f16_gemv` / most remaining `q4k_*` / `q6k_*` device kernels ~~— `f32_gemv`/`f16_gemv` landed in Session 8;~~ CUDA now has `q4k_matvec`, `q6k_matvec`, `q4k_matmul`, `q6k_matmul`, `q4k_dual_matvec`, `f32_gemv`, `f16_gemv`, `q4_matvec`, `q4_vecmat`, `kv_append`, `rms_norm`, `rms_norm_heads`, `geglu_silu`, `geglu_gelu_tanh`, `activation_silu`, `activation_gelu_tanh`, `residual_add`; the full `QuantMatVec` kernel surface is native, the KV-append primitive is native, and the first three elementwise families (body + per-head RMSNorm in Session 13, FFN activation in Session 14, residual add in Session 15) are native (the device-kernel-fusion follow-on)
- routing of the amortised Q6_K matmul through a backend ~~— DONE in Session 7.~~ `QuantMatVec::q6k_matmul` added (default `None`); CpuBackend wraps `q6k_matmul_into`; CUDA routes native-then-CPU; the Q6_K arm of `ffn/weight.rs::quant_matmul` dispatches through a backend's `q6k_matmul` when supplied (attention `gpu.rs` Q/K/V/O pass the backend; `Q4kMatmulFfn` passes `None`).
- real coarse `KvDispatch` (currently delegates to CPU)
- real `AsyncComputeBackend` batching (currently delegates to CPU)
- hardware-specific CI jobs for CUDA and Vulkan

What exists now is:

- the repo-wide control plane from Sessions 1-3
- plus a real CUDA runtime/bootstrap path (`cudarc` + NVRTC)
- plus seventeen native CUDA kernels — five k-quant (`q4k_matvec`, `q6k_matvec`, `q4k_matmul`, `q6k_matmul`, `q4k_dual_matvec`), two dense (`f32_gemv`, `f16_gemv`), two legacy Q4_0 (`q4_matvec`, `q4_vecmat`), `kv_append`, two elementwise RMSNorm (`rms_norm`, `rms_norm_heads`, Session 13), four elementwise activations (`geglu_silu`, `geglu_gelu_tanh`, `activation_silu`, `activation_gelu_tanh`, Session 14), and the residual add (`residual_add`, Session 15) — behind the existing CPU fallback. All five k-quant kernels and the two Q4_0 kernels are trait-routed through `QuantMatVec`; the dense GEMVs are trait-routed through `MatMul` (Session 8); `kv_append` is the device-side KV-write primitive backing the `DecodeBackend` lifecycle methods (Session 10); the two RMSNorm kernels are routed through the pipeline's `norm_2d`/`norm_1d`/`norm_2d_no_weight`/`rms_norm_heads_array` helpers (Session 13); the four activation kernels are routed through the pipeline's `apply_activation_gated_native`/`apply_activation_std_native` helpers (Session 14 — the second native elementwise step toward the fully-fused pipeline); the residual-add kernel is routed through the pipeline's `add_residual_native` helper (Session 15 — the third native elementwise step).
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
      - ~~RoPE (Q/K rotary embedding)~~ — DONE in Session 16 (native `rope` kernel; the device twin of `apply_rope_partial_at_full`, split-half pairing + pass-through tail; `inv_freq` precomputed on host incl. `llama3` scaling, uploaded as `double` so theta/cos/sin match the f64 reference; routed through the pipeline's `rope_native` helper with a `ROPE_NATIVE_MIN_ELEMS = 8192` gate, all 4 Q/K call sites in the decode/prefill attention blocks; parity-tested — pass-through exact, rotary ≤ 1e-5 since device double cos/sin are a different libm). The fourth device-elementwise step. Remaining host elementwise ops: GQA softmax / V-norm, then collapsing the per-projection round-trips.
    - (Session 11 follow-on) extend the fused pipeline to MoE / PLE / remote-FFN layer features (currently bail to `None` → CPU fallback).
      - ~~MoE~~ — DONE in Session 12 (host-orchestrated dense slab delta + substrate `cpu_moe_forward` expert block + `outer_post_norm_residual`; runs through `decode_token`/`prefill_kquant`; composition parity host-runnable). Expert projections still run on CPU (device-fusion of the expert matvecs is the follow-on).
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

Phase 4 is **underway in CUDA**: `cudarc`/NVRTC are wired, runtime fallback is panic-safe on non-CUDA hosts, and eighteen native kernels are live — five k-quant (`q4k_matvec`, `q6k_matvec`, `q4k_matmul`, `q6k_matmul`, `q4k_dual_matvec`) plus two dense (`f32_gemv`, `f16_gemv`) plus two legacy Q4_0 (`q4_matvec`, `q4_vecmat`) plus `kv_append` plus two elementwise RMSNorm (`rms_norm`, `rms_norm_heads`) plus four elementwise activations (`geglu_silu`, `geglu_gelu_tanh`, `activation_silu`, `activation_gelu_tanh`) plus the residual add (`residual_add`) plus RoPE (`rope`) — each behind its CPU fallback with parity tests. As of Session 7 **all five k-quant kernels are trait-routed** through `QuantMatVec`, and the `q6k_matmul` kernel is live end-to-end: the Q6_K arm of `ffn/weight.rs::quant_matmul` dispatches through a backend's `q6k_matmul` (attention `gpu.rs` passes its backend; `Q4kMatmulFfn` keeps the CPU path). Session 8 added the dense `f32_gemv`/`f16_gemv` kernels trait-routed through `MatMul`, with `f32_gemv` only taking the native path on row-major-contiguous `ArrayView2`s (non-contiguous views fall back to the CPU reference; `f32_gemv_topk1`/`f16_gemv_topk1`/`f16_gemv_topk` inherit the routing). Session 9 added the legacy Q4_0 `q4_matvec`/`q4_vecmat` kernels trait-routed through `QuantMatVec`, completing the `QuantMatVec` kernel surface. Session 10 landed the device-side KV cache (`CudaKVCache`, mirroring Metal's `LayerKVCache`/`KVCache` via `cudarc::driver::CudaSlice<f32>` owned device buffers) + the native `kv_append` kernel and wired the full `DecodeBackend` KV lifecycle (`has_kv_cache`/`preallocate_kv_cache_per_layer`/`reset_kv_cache`/`kv_cache_len`/`truncate_kv_cache`/`populate_kv_layer`) through it — the foundation for the fused `prefill_kquant`/`decode_token` pipelines. Sessions 13-16 continued the device-kernel-fusion follow-on: native elementwise RMSNorm (`rms_norm`/`rms_norm_heads`, Session 13) + native FFN activation (`geglu_silu`/`geglu_gelu_tanh`/`activation_silu`/`activation_gelu_tanh`, Session 14) + native residual add (`residual_add`, Session 15) + native RoPE (`rope`, Session 16), all routed through the pipeline's helpers with min-elems gates. Vulkan Phase 5 and hardware CI are still not started. Phase 7's honesty pass remains in force: capabilities stay false until more of CUDA is truly native (prefill/decode fused pipelines still delegate to CPU for the remaining elementwise ops: GQA softmax / V-norm).

The next session should continue Phase 4 inside `larql-compute-cuda`: the fused `prefill_kquant`/`decode_token`/`decode_token_with_state_dump_masked` pipelines are **live** (Session 11, host-orchestrated) and capability advertisement is on, so `fused_prefill`/`fused_decode_step` route through CUDA and the `auto` policy on Linux picks CUDA first. **Session 12 extended the host pipeline to hybrid-MoE layers** (Gemma 4 26B-A4B): `host_ffn_block_moe_decode`/`host_prefill_ffn_block_moe` compose the dense-slab delta + substrate `cpu_moe_forward` expert block + `outer_post_norm_residual`, so MoE layers no longer bail to `None`→CPU when reached (PLE and remote-FFN still bail — they need data/callbacks not on the trait surface; routing the expert matvecs through native CUDA kernels is the device-fusion follow-on). **Sessions 13-16 continued the perf follow-on** by landing the first four native elementwise families: `rms_norm` (body) + `rms_norm_heads` (per-head, Session 13), `geglu_silu`/`geglu_gelu_tanh`/`activation_silu`/`activation_gelu_tanh` (FFN activation, Session 14), `residual_add` (the post-attention/post-FFN residual, Session 15), and `rope` (Q/K rotary embedding, Session 16), all routed through the pipeline's helpers (native-then-CPU fallback with min-elems gates) — the four highest-frequency elementwise ops are now off the host. The remaining Phase 4 perf work is the **rest of the device-elementwise + round-trip collapse**: GQA softmax / V-norm into device kernels, then collapsing the per-projection htod/launch/dtoh round-trips into a single-command-buffer fused pipeline (mirroring Metal's `decode/mod.rs` shape) — the host-orchestrated path is the parity oracle for that. After that: PLE (needs trait-surface/engine-hook work) and remote-FFN (implement `decode_token_with_moe` to route the callback) layer features, the coarse `KvDispatch` bridge, then Vulkan Phase 5, then hardware CI. With the k-quant matmul/matvec + dense GEMV + legacy Q4_0 kernels all trait-routed, the KV cache lifecycle in place, the fused decode/prefill pipelines routing through CUDA, the MoE host pipeline landed, and the first four elementwise families (RMSNorm + activation + residual add + RoPE) now native, the remaining Phase 4 surface is the rest of the device-kernel fusion (GQA softmax / V-norm + round-trip collapse) + the `KvDispatch` bridge + the PLE/remote-FFN trait-surface work.

# DeepSeek-V4-Flash Implementation Plan

Status: Draft 2026-07-12
Goal: First-token CPU parity with the HuggingFace reference (`inference/model.py`), then Q4_K quantization, then distributed FFN dispatch.

## Architecture summary

V4-Flash (284B total / 13B active MoE, 43 layers + 1 MTP) introduces three mechanisms not present anywhere in LARQL:

1. **Hyper-Connections (HC)** — replaces residual connections with a 4-copy hidden state (`[b,s,hc_mult,d]`) that flows through every layer. Each block has hc_pre (Sinkhorn-normalized 4→1 reduction) and hc_post (1→4 expansion). This is structural, not optional.

2. **Hybrid MLA + CSA attention** — low-rank Q (wq_a/wq_b), fused single-head KV (wkv, head_dim=512), attention sink tokens, sliding window (128) always on, plus per-layer KV compression (ratio 4 with Indexer, or ratio 128 without). Sparse attention via index-gathered top-k.

3. **Hash-based + sqrtsoftplus MoE routing** — first 3 layers use token-ID→expert lookup (tid2eid); remaining layers use sqrtsoftplus scoring + noaux_tc top-k.

Reference: `inference/model.py` + `inference/kernel.py` from the HF repo.

## What LARQL already has

| Component | State |
|---|---|
| `deepseek_v4.rs` arch module | Tensor key mappings for browse-tier (embed, attn wq_a/wq_b/wkv, ffn w1/w2/w3, shared_experts, gate). Missing: wo_a/wo_b, compressor/indexer, attn_sink, hc_*, tid2eid, MTP. |
| MLA absorption (`mla_absorb.rs`) | Absorbs DS-V3-style 4-matrix MLA into dense Q/K/V. **Cannot use**: V4-Flash has fused wkv (no kv_b), different head structure, wo_a/wo_b low-rank output, and compression layers. |
| MXFP4 dequantization | `dequantize_per_expert_mxfp4` in safetensors loader handles I8+F8_E8M0. V4-Flash expert format is close but uses E2M1 packed + E8M0 per-32 (different from GPT-OSS packed blocks). |
| MoE forward | `cpu_moe_forward` supports standard routing. Needs sqrtsoftplus + noaux_tc + hash routing + shared expert + SwiGLU limit. |
| FP8 (E4M3) dequant | Not present. Need per-128×128 block scale dequant for attention weights. |
| FP4 (E2M1) dequant | LARQL's own FP4 codec (137-byte blocks) differs from V4's E2M1 packed + E8M0 per-32. Need V4-specific dequant. |
| Sparse attention | LARQL has dense GQA + decode attention. V4-Flash needs index-gathered sparse attention with attn_sink. |
| Sliding window | `window.rs` exists in larql-compute but may not match V4's circular-buffer style. |
| Distributed FFN | Fully implemented (larql-server --ffn-only --layers, larql-router). |

## Phased plan

### Phase 0 — Architecture module + config mapping (1-2 sessions)

Update `deepseek_v4.rs` to reflect the real V4-Flash tensor inventory:

- [ ] Extend `ModelConfig` with V4 fields: `head_dim`, `index_head_dim`, `index_n_heads`, `index_topk`, `hc_mult`, `hc_sinkhorn_iters`, `hc_eps`, `o_groups`, `o_lora_rank`, `compress_ratios`, `compress_rope_theta`, `num_hash_layers`, `scoring_func`, `topk_method`, `swiglu_limit`, `expert_dtype`, `num_nextn_predict_layers`
- [ ] Add tensor key methods: `wo_a_key`, `wo_b_key`, `attn_sink_key`, `compressor_*_key`, `indexer_*_key`, `hc_*_key` (per-layer + global), `tid2eid_key`, `mtp_*_key`
- [ ] Config JSON parser: map all V4-Flash config fields
- [ ] Test: synthetic config → all tensor keys resolve correctly against the real index.json

### Phase 1 — FP8 + FP4 dequantization (1 session)

- [ ] FP8 E4M3 + UE8M0 block-scale dequant (128×128 grid): attention weights stored as `.weight` (FP8) + `.scale` (UE8M0)
- [ ] FP4 E2M1 packed + E8M0 per-32 dequant: expert weights stored as `.weight` (I8, packed 2-per-byte) + `.scale` (E8M0, per-32-elements)
- [ ] Test: dequant round-trip parity against Python reference values
- [ ] Wire into safetensors loader alongside existing MXFP4 path

### Phase 2 — HC (Hyper-Connections) forward pass (1-2 sessions)

This is the structural backbone — nothing works without it.

- [ ] `hc_pre`: linear projection → RMSNorm → Sinkhorn split into pre/post/comb
- [ ] `hc_post`: post-weighted expansion + comb-weighted residual mixing
- [ ] Sinkhorn normalization (20 iterations of row/col normalization on hc×hc matrix)
- [ ] hc_head (final layer): sigmoid-based weighted reduction 4→1
- [ ] Test: synthetic hidden state [1,seq,4,4096] → hc_pre → identity attn → hc_post → verify shape + Sinkhorn doubly-stochastic property

### Phase 3 — Attention forward pass (2-3 sessions)

The most complex piece. Three layer types:

#### 3a. Pure sliding window layers (ratio=0, layers 0/1/43)
- [ ] wq_a → q_norm → wq_b (low-rank Q, 64 heads × 512 dim)
- [ ] wkv → kv_norm (fused single KV, head_dim=512)
- [ ] RoPE on last 64 dims (standard rope_theta=10000, no YaRN)
- [ ] Circular sliding window KV cache (window_size=128)
- [ ] sparse_attn with attn_sink (index-gathered: all window positions + sink)
- [ ] wo_a → wo_b (grouped low-rank O, 8 groups)
- [ ] Test: single-layer CPU parity vs Python reference (BF16, no quant)

#### 3b. Compress-128 layers (ratio=128, odd layers 3-41)
- [ ] Compressor: learned gated pooling over 128 consecutive tokens → compressed KV
- [ ] Fixed index selection (every 128th compressed position)
- [ ] Compressed KV appends to sliding window KV
- [ ] compress_rope_theta=160000, YaRN scaling
- [ ] Test: single-layer CPU parity with long context

#### 3c. Compress-4 layers (ratio=4, even layers 2-42)
- [ ] Indexer: its own Compressor (with Hadamard rotation) + wq_b + weights_proj → top-512 sparse selection
- [ ] Compressor: overlapping windows (ratio=4), gated softmax pooling
- [ ] Test: single-layer CPU parity

### Phase 4 — MoE forward pass (1-2 sessions)

- [ ] Gate: hash routing (layers 0-2, tid2eid lookup) + sqrtsoftplus + noaux_tc (layers 3+)
- [ ] Expert FFN: SwiGLU with clamp (swiglu_limit=10.0), w1/w2/w3
- [ ] Shared expert (always active)
- [ ] Test: single-layer MoE parity vs Python reference

### Phase 5 — Full transformer forward pass (1 session)

- [ ] Embed → HC expand → 43 blocks → HC head → lm_head
- [ ] First-token parity: 4 prompts (raw, chat, arithmetic, multiturn) vs Python reference
- [ ] NRMSE ≤ 1e-4, top-10 overlap ≥ 9
- [ ] Following the Gemma 4 E2B ST1-ST5 playbook

### Phase 6 — Vindex extraction (1 session)

- [ ] Map V4-Flash tensors to vindex format
- [ ] Attention: dequant FP8 → store as f16 or Q4_K
- [ ] Experts: dequant FP4 → re-quantize as Q4_K (LARQL's format)
- [ ] HC parameters: store as f32 sidecar
- [ ] Compressor/Indexer weights: store as f32 sidecar
- [ ] Test: extract → load → first-token parity with Phase 5 CPU reference

### Phase 7 — Distributed FFN dispatch (1-2 sessions)

- [ ] Slice vindex: client (embed + attention + HC head) vs server (FFN expert weights)
- [ ] larql-server serves FFN shards with V4-Flash MoE format
- [ ] Client runs attention locally, dispatches FFN to networked nodes
- [ ] Test: single-node vs 2-node distributed, token sequence identical

### Phase 8 — Q4_K quantization (1 session)

- [ ] Quantize attention weights to Q4_K at extraction
- [ ] Quantize expert weights to Q4_K (already FP4 → re-quantize to LARQL Q4_K)
- [ ] Parity: Q4_K vs F32 reference, top-10 overlap ≥ 9
- [ ] Benchmark: tok/s on Valinor RTX 3060

### Phase 9 — CUDA backend (2-3 sessions, after CPU parity proven)

- [ ] FP8/FP4 dequant kernels (or pre-dequant at extraction to Q4_K)
- [ ] HC forward pass on CUDA
- [ ] Sparse attention kernel (index-gathered, with attn_sink)
- [ ] Compressor/Indexer on CUDA
- [ ] MoE with V4-Flash routing on CUDA

## Critical unknowns

1. **Sparse attention kernel** — V4-Flash uses index-gathered attention (not dense QK^T then mask). This is fundamentally different from LARQL's existing decode_attention/prefill_attention kernels. The `sparse_attn` reference uses topk_idxs to gather specific KV positions. LARQL has no equivalent.

2. **Compression layer statefulness** — the Compressor maintains incremental state (kv_state, score_state) across decode steps. This is a new kind of KV-cache-like state that isn't just (K, V) pairs.

3. **HC hidden state shape** — `[b, s, hc_mult, d]` instead of `[b, s, d]` changes the shape contract throughout the forward pass. The vindex format, walk kernels, and FFN dispatch all assume `[layers, hidden]` — the HC dimension needs to be handled.

4. **FP8 attention weights** — LARQL stores weights as f16 or Q4_K. V4-Flash ships FP8 attention. The extraction path needs to dequant FP8→f32, then optionally re-quantize to Q4_K.

## Estimated timeline

| Phase | Sessions | Dependency |
|---|---|---|
| 0. Config + arch module | 1-2 | none |
| 1. FP8/FP4 dequant | 1 | none |
| 2. HC forward | 1-2 | Phase 0 |
| 3. Attention forward | 2-3 | Phase 0, 1, 2 |
| 4. MoE forward | 1-2 | Phase 0, 1 |
| 5. Full forward parity | 1 | Phase 2, 3, 4 |
| 6. Vindex extraction | 1 | Phase 5 |
| 7. Distributed FFN | 1-2 | Phase 6 |
| 8. Q4_K quantization | 1 | Phase 6 |
| 9. CUDA backend | 2-3 | Phase 8 |
| **Total** | **12-18** | |

## Oracle setup

The reference inference code (`inference/model.py` + `inference/kernel.py`) requires CUDA + tilelang. For CPU-only parity testing, we need a numpy/torch-CPU port of the forward pass that doesn't require tilelang kernels. The FP8/FP4 GEMM and sparse_attn kernels need CPU fallbacks.

Alternatively: run the reference on Valinor (RTX 3060) to generate oracle outputs, then compare LARQL CPU forward against those fixed oracle values.

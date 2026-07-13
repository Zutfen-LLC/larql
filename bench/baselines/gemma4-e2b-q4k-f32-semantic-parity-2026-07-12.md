# LARQL-INFERENCE-TRUST-001A-ST6 — Production Q4_K Semantic Parity

- **Slice:** "LARQL-INFERENCE-TRUST-001A-ST6"
- **Decision:** "Red"
- **Source:** `"google/gemma-4-E2B-it"` @ `"9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"`
- **Safetensors SHA-256:** `"2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"`
- **Q4_K artifact size:** 7.87 GB
- **Quantization:** attn Q/K/O→Q4_K, V→Q6_K; FFN gate/up→Q4_K, down→Q6_K; lm-head tied to f16 embeddings; norms→F32

## First-token per-prompt results

| prompt | seq_len | passed | top-1 ref | top-1 cand | top-10 overlap | logit nrmse | logit cosine | worst layer nrmse | first divergence |
|---|---|---|---|---|---|---|---|---|---|
| "raw_completion" | 6 | false | 9079 | 9079 | 6 | 5.768e-2 | 9.984e-1 | 2.779e-1 | "post_layer"@10 |
| "chat" | 31 | false | 818 | 818 | 7 | 2.318e-2 | 9.999e-1 | 1.816e-1 | "post_layer"@11 |
| "arithmetic" | 36 | false | 236800 | 236800 | 8 | 8.978e-2 | 9.991e-1 | 2.610e-1 | "post_layer"@9 |
| "multiturn" | 40 | false | 16520 | 16520 | 8 | 2.482e-2 | 9.999e-1 | 2.214e-1 | "post_attention"@11 |

## Teacher-forced continuation

- **Total positions:** 20 (100% top-1 agreement)
- **All positions within budget:** false
- **First-token top-1 exact (all):** true

## Lm-head error decomposition (§8)

| prompt | body nrmse (B vs A) | lm-head nrmse (C vs B) | total nrmse (C vs A) |
|---|---|---|---|
| "raw_completion" | 5.768e-2 | 5.835e-6 | 5.768e-2 |
| "chat" | 2.318e-2 | 3.792e-6 | 2.318e-2 |
| "arithmetic" | 8.978e-2 | 7.593e-6 | 8.978e-2 |
| "multiturn" | 2.482e-2 | 2.529e-6 | 2.482e-2 |

## Shared-KV topology (§7)

- **Source map:** `{"15":13,"16":13,"17":13,"18":13,"19":14,"20":13,"21":13,"22":13,"23":13,"24":14,"25":13,"26":13,"27":13,"28":13,"29":14,"30":13,"31":13,"32":13,"33":13,"34":14}`
- **F32/Q4_K topology agree:** true

## Scope exclusions

- KV-cached E2B decode (cached decode does not yet support shared KV)
- direct-matvec E2B decode
- CUDA
- Vulkan
- Metal
- sampling
- performance optimization
- changing to Q5/Q8/F16 to obtain GREEN
- multimodal inputs
- tools or thinking-enabled prompts
- long-form generation quality

## Recommended next slice

LARQL-INFERENCE-TRUST-001A-ST6A (narrow correction targeting the recorded first budget breach)

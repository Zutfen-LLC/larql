# LARQL-INFERENCE-TRUST-001A-ST5 — F32 First-Token Semantic Parity

- **Slice:** "LARQL-INFERENCE-TRUST-001A-ST5"
- **Decision:** "Green"
- **Source:** `"google/gemma-4-E2B-it"` @ `"9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"`
- **Safetensors SHA-256:** `"2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"`
- **Oracle:** `"transformers-oracle"`
- **Policy:** coarse nrmse≤0.0001, final logits nrmse≤0.0001, top-10 overlap≥9

## Per-prompt results

| prompt | seq_len | passed | top-1 ref | top-1 cand | top-10 overlap | logit max_abs | logit nrmse | logit cosine | worst layer nrmse | first divergence |
|---|---|---|---|---|---|---|---|---|---|---|
| "raw_completion" | 6 | true | 9079 | 9079 | 10 | 5.174e-5 | 6.895e-7 | 1.000e0 | 1.782e-6 | null |
| "chat" | 31 | true | 818 | 818 | 10 | 3.242e-5 | 4.750e-7 | 1.000e0 | 1.110e-6 | null |
| "arithmetic" | 36 | true | 236800 | 236800 | 10 | 6.485e-5 | 1.141e-6 | 1.000e0 | 3.356e-6 | null |
| "multiturn" | 40 | true | 16520 | 16520 | 10 | 3.910e-5 | 2.432e-7 | 1.000e0 | 1.062e-6 | null |

## Corrections made

### "Gemma 4 global (full_attention) layers used the wrong RoPE frequency mode"

- **Root cause:** "HF rope_type='proportional' computes inv_freq exponents over the full head_dim (512) and zero-pads to head_dim/2, then half-splits over the full head; LARQL divided exponents by rotary_dim (128) and half-split within rotary_dim. Sliding layers (rope_type='default', full rotary) were unaffected, so the divergence first appeared at layer 4 (the first global layer)."
- **Fix:** "Added RopeFreqMode::{Standard,Proportional} to larql-models; Gemma4 arch returns Proportional for global layers; larql-compute rope builds the zero-padded head_dim/2 inv_freq and half-splits over the full head for Proportional mode. Applied consistently to the CPU block, decode, and kv-prefill (gpu.rs CPU fallback) paths."


## Scope exclusions

- multi-token generation
- sampling
- KV-cached decode
- Q4_K
- CUDA
- Vulkan
- Metal
- performance optimization
- production quantization
- multimodal inputs
- tools or thinking-enabled prompts

## Recommended next slice

LARQL-INFERENCE-TRUST-001A-ST6 (Production Q4_K semantic parity against the proven F32 reference)

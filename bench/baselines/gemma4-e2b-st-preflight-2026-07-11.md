# Gemma 4 E2B Safetensors Preflight

Slice: `LARQL-INFERENCE-TRUST-001A-ST1B`

Starting main SHA: `d969c18fe93fd3269e5bf18e648c8bbde5a3a6e7`

The environment-only RED result from ST1 is superseded for source availability and exact-source findings. This run used the official `google/gemma-4-E2B-it` repository at immutable revision `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`.

## Provenance

- Canonical URL: `https://huggingface.co/google/gemma-4-E2B-it/tree/main`
- Revision resolution: Hugging Face model API, then pinned `snapshot_download`; all source sizes and LFS SHA-256/Git blob IDs match the revision-specific snapshot manifest
- Authentication: not required; repository API reports public and ungated
- License access: available; repository declares Apache-2.0
- Expected download: 10,278,846,563 bytes
- Actual source files: 10,278,846,563 bytes
- Safetensors: 1 shard, 10,246,621,918 bytes, SHA-256 `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`
- Disk available before/after: 967,526,051,840 / 957,826,449,408 bytes
- Local paths are represented as `${LARQL_GEMMA4_ST_DIR}` in committed evidence.

## Evidence Classes

- Observed from official source: all nine source files, byte lengths, SHA-256 values, 2,011 BF16 tensors, exact names/shapes/shard associations, tokenizer/config values, and tied-head geometry.
- Observed from Transformers execution: local-only official model loading, prompt rendering, complete token IDs/pieces, 16 generated IDs/pieces, decoded outputs, and finite logits/hidden states.
- Derived by LARQL: 600 required text-decoder tensors, all local/global geometry, KV reuse sources, PLE geometry, tied-head decision, and exclusion/unknown counts.
- Synthetic regression evidence: physical K/V tensors in the shared region and double-wide FFNs beginning at layer 15.
- Not observed: none of the acceptance-gate fields.

## Exact-Source Result

- Missing required: 0
- Shape mismatches: 0
- Normalized duplicates: 0
- Unknown decoder tensors: 0
- Multimodal excluded: 1,411 (658 vision, 751 audio, 2 connector/projector)
- MTP excluded: 0
- Tied-head contract: PASS (`tie_word_embeddings=true`, no `lm_head.weight`, embedding `[262144,1536]` equals expected output-projection geometry, tokenizer vocabulary and physical rows both 262,144)
- PLE contract: PASS (`256` per layer; aggregate width `8,960`)
- Source health: HEALTHY
- Output quality: COHERENT
- Ignored real-model test: ran, 1 passed, 0 ignored

## Source-Driven Fixes

The official checkpoint demonstrated two predecessor assumptions were incorrect. Runtime KV sharing does not remove physical K/V tensors from layers 15-34, and `use_double_wide_mlp=true` doubles those layers' FFN width from 6,144 to 12,288. The oracle also now enforces the raw prompt's required BOS, classifies qualitative output, excludes local Hugging Face cache metadata, and records semantic tensor classes.

## Decision

**GREEN - ST2 unblocked.** The official checkpoint, source-health oracle, complete decoder inventory, geometry contract, collision checks, unknown-tensor checks, PLE contract, and tied-head contract pass. This does not claim Gemma 4 inference support.

Next slice: `LARQL-INFERENCE-TRUST-001A-ST2` - lossless BF16/F32 safetensors-to-vindex extraction and load validation.

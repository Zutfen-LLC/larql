# Gemma 4 E2B GGUF Feasibility Audit

Date: 2026-07-11  
Slice: `LARQL-INFERENCE-TRUST-001A`  
Starting main SHA: `d65dded1db68c69dda9a2a0ff19d446b277a5cad`  
Decision: **RED - faithful import is not currently possible**

## Evidence Boundary

`LARQL_GEMMA4_GGUF` was not set and no llama.cpp executable or checkout was available. Consequently the source SHA-256, byte length, exact metadata/tensor inventory, oracle token IDs, extraction profile, vindex integrity results, and CPU smoke values are **not observed**. Null and empty fields in the paired JSON artifacts mean unavailable evidence, not zero tensors or successful checks.

The Hugging Face API identified repository revision `2ea637031baa8dc847d64b5dbb7011fd6a445849`, architecture `gemma4`, context length 131072, and an embedded chat template. These remote observations do not substitute for hashing and auditing the requested file.

## Architecture Matrix

| Feature | Source requirement | LARQL representation | Extraction | CPU runtime | CUDA runtime | Fail/fallback | Evidence |
|---|---|---|---|---|---|---|---|
| Gemma 4 architecture | required | `Gemma4Arch` | metadata mapped | implemented | host route | validated detection | `loading/gguf/loader.rs` |
| 35 layers | required | `ModelConfig.num_layers` | preserved | implemented | represented | validation error | synthetic test |
| PLE | required | `per_layer_embed_dim` and sidecar | tensor names mapped; GGUF writer blocked | implemented | unsupported | CPU route | synthetic test; source audit |
| Local/global schedule | required | exact `layer_types` | derived from per-layer key lengths | represented | unsupported | CUDA rejects heterogeneous attention | synthetic test |
| Sliding window 512 | required | `sliding_window` | metadata mapped | **not enforced** | unsupported path | runtime semantic blocker | `kv_dispatch/cpu.rs` |
| p-RoPE | required | per-layer rotary fraction | derived from rotary/global dimensions | implemented | unsupported path | explicit gate | synthetic test |
| Q/K per-head norm | required | architecture keys | tensor names mapped | implemented | geometry kernels exist | GGUF extraction blocked | synthetic test |
| Gemma norm offset | required | architecture behavior, offset 0 | represented | implemented | represented | architecture selected | `architectures/gemma4.rs` |
| Embedding scaling | inspect | architecture sqrt(hidden) | represented | implemented | represented | architecture selected | `architectures/gemma4.rs` |
| Final softcap | inspect | config field | metadata mapped | implemented | represented | config validation | synthetic test |
| Large vocabulary | required | physical/logical split exists | exact source unobserved | implemented | represented | unproven for source | source unavailable |
| Tied/separate head | inspect | implicit missing-head tie | **not source-verified** | implemented | represented | blocker | source unavailable |
| Chat template/special tokens | required | adjacent tokenizer JSON only | embedded GGUF export absent | unavailable | unavailable | blocker | loader audit |
| MTP | excluded | no implicit loading | untouched | disabled | disabled | direct file required | scope control |
| Source tensor formats | required | GGML dequantizers exist | **inference writer absent** | unavailable | unavailable | fail loudly | `model_weights.rs` |

## Corrections

- Maps per-layer GGUF key-length and KV-head arrays to exact local/global geometry and preserves the final global layer.
- Maps sliding window, SWA RoPE base, p-RoPE fraction, PLE dimension, shared-KV layers, norm epsilon, and final softcap.
- Maps llama.cpp Q/K norm and PLE tensor names to LARQL canonical names.
- Expands the GGUF inference extraction error to name omitted required tensor classes, the rejecting code path, and the required next implementation.
- Adds a manual ignored audit gated by `LARQL_GEMMA4_GGUF`.

## Ordered Blockers

1. `TRUST-001A-B01` Source provenance and health. Obtain the exact GGUF, record SHA-256/size/version, build a pinned llama.cpp, and capture both deterministic oracle runs.
2. `TRUST-001A-B02` Inference extraction. Implement a bounded per-tensor GGUF reader that classifies every tensor and explicitly dequantizes/requantizes or preserves it. Components: GGUF loader, streaming tensor source, vindex weight writers. Tests: mixed GGML types, byte identity where applicable, unsupported required type rejection.
3. `TRUST-001A-B03` Tokenizer and head semantics. Export embedded GGUF tokenizer/chat metadata, special IDs, logical vocabulary, and explicit tied-head evidence. Tests: padded physical vocabulary, added tokens, tied and untied heads.
4. `TRUST-001A-B04` CPU local attention semantics. Enforce the configured sliding window in prefill and decode. Components: CPU `KvDispatch`, f32 attention, direct kquant cached decode. Tests: contexts longer than 512 proving old keys are excluded on local layers and retained on global layers.
5. Repeat inventory, extraction, byte/finiteness validation, CPU smoke, and Qwen regression on the canonical CUDA host before advancing to semantic parity.

## Extraction Command

Canonical intended command, not run because the source was unavailable and the code path is known to reject inference GGUF:

```bash
larql extract "${LARQL_GEMMA4_GGUF}" -o /tmp/gemma4-e2b.vindex \
  --level inference --quant q4k --profile
```

## TRUST-001B Scope

Do not start `LARQL-INFERENCE-TRUST-001B` from this state. After blockers B01-B04 are closed and 001A reaches GREEN, compare prompt rendering, token IDs, first-token top-20 logits, and per-layer checkpoints against the pinned llama.cpp oracle. CUDA remains non-oracular and optimization remains out of scope.

## Verification

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | pass |
| `cargo test -p larql-models --lib` | 424 passed, 0 failed, 4 ignored |
| `cargo test -p larql-models --test gemma4_gguf_audit -- --ignored` without the model environment variable | 1 passed, 0 failed |
| `cargo test -p larql-vindex --lib` | 1131 passed, 0 failed |
| `cargo test -p larql-compute --lib` | 750 passed, 0 failed, 2 ignored |
| `cargo test -p larql-inference --lib` | 1262 passed, 0 failed, 4 ignored |
| `cargo test -p larql-cli --bins` | 243 passed, 0 failed |
| All five requested `cargo clippy ... -D warnings` commands | pass |
| `cargo build -p larql-cli --release --features cuda` | pass |

The host has `libblas` and `libcblas` but not `libopenblas`. Link steps used `LIBRARY_PATH=/tmp/kilo` with a temporary `libopenblas.so` symlink to the installed `libcblas.so`; no repository file or production configuration was changed.

## Scope Deviations

- No real-model source or oracle run was possible in this environment.
- No extraction, vindex integrity, CPU generation, CUDA execution, or real Qwen extraction was claimed.
- MTP and multimodal components were not loaded or modified.

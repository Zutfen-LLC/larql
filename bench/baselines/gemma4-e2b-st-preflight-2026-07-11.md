# Gemma 4 E2B Safetensors Preflight

Slice: `LARQL-INFERENCE-TRUST-001A-ST1`

Starting main SHA: `4391b41279acc9db0301d69d52663e318a409f19`

Remote repository evidence identifies revision `397aea7eca1853d6f3d099a56612a68ffc0229ae` for `unsloth/gemma-4-E2B-it-qat-q4_0-unquantized`. This is not a substitute for hashing an operator-provided local checkout.

## Evidence

- Observed from exact source: not observed; `LARQL_GEMMA4_ST_DIR` was not set.
- Derived from config: not observed.
- Derived from LARQL architecture: the validator derives per-layer Q/K/V/O geometry, norms, FFN, layer scalar, PLE, K=V, KV sharing, and tied-head requirements through `ModelArchitecture`.
- Synthetic-only evidence: metadata validation, collision rejection, unknown decoder rejection, and excluded-modality classification are implemented.
- Not observed: source checksums, source-health output, tokenizer evidence, tensor inventory, tied-head evidence, and exact-source preflight.

## Decision

**RED - source or contract cannot be trusted.** The exact source was unavailable, so the required positive source-health and complete-contract claims cannot be made.

## Ordered Blockers

1. Acquire revision `397aea7eca1853d6f3d099a56612a68ffc0229ae` and run `scripts/gemma4_source_oracle.py` with `LARQL_GEMMA4_ST_DIR` and `LARQL_GEMMA4_ST_REVISION` set.
2. Run the ignored `gemma4_safetensors_preflight` integration test and preserve its complete inventory and diagnostics.
3. Resolve any missing mappings, shape mismatches, collisions, or unknown decoder tensors before beginning ST2.

ST2 is not ready while this report remains RED.

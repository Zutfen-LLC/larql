# Gemma 4 E2B F32 Reference Vindex

Slice: `LARQL-INFERENCE-TRUST-001A-ST2`

## Decision

**GREEN - reference artifact proven.** The official BF16 text decoder was serialized as F32 and reconstructed through the production float loader with zero bitwise mismatches. This result makes no semantic-inference claim; no model forward pass, logits comparison, generation, or tokenizer parity run occurred.

## Revisions

- Work-start SHA: `d5922116a1ea8967a427164365baa75b370baffc`
- PR base SHA: `d5922116a1ea8967a427164365baa75b370baffc`
- Final head SHA: `62642e3da80a0a8a0f9ebd38cf26d8af7c53b13d`

## Source

- Repository: `google/gemma-4-E2B-it`
- Revision: `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`
- Safetensors: 1 shard, 10,246,621,918 bytes
- SHA-256: `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`
- Inventory: 2,011 BF16 tensors; 600 required text-decoder tensors
- ST1B exact-source preflight: passed

## Extraction

```bash
LARQL_GEMMA4_ST_REVISION=9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf \
larql extract "${LARQL_GEMMA4_ST_DIR}" \
  -o "${LARQL_GEMMA4_REFERENCE_VINDEX}" \
  --level all --quant none --dtype f32 --reference-f32 --profile
```

- Semantics: all tiers, no quantization, F32 ordinary and PLE tensors, full attention/FFN, no compact mode, mandatory exact-source gate
- Publication: sibling `.tmp-<pid>` directory, then rename after successful extraction, structural validation, profile, and provenance
- Duration: 1,582.848 seconds
- Peak RSS: 21,267,382,272 bytes
- Available disk before/after: 949,564,268,544 / 932,202,741,760 bytes
- Artifact size: 18,651,850,857 bytes
- File sizes and SHA-256 values: recorded in the companion JSON and artifact-local `reference_provenance.json`
- No `lm_head.bin` was emitted

## Validation

```bash
LARQL_GEMMA4_ST_DIR="${LARQL_GEMMA4_ST_DIR}" \
LARQL_GEMMA4_ST_REVISION="${LARQL_GEMMA4_ST_REVISION}" \
LARQL_GEMMA4_REFERENCE_VINDEX="${LARQL_GEMMA4_REFERENCE_VINDEX}" \
cargo test -p larql-vindex --test gemma4_reference_vindex_roundtrip \
  --release -- --ignored --nocapture
```

- Production float loader: passed
- Validation duration: 121.195 seconds
- Validation peak RSS: 25,188,225,024 bytes
- Manifest: 564 entries, comprising 282 tensor and 282 vector entries
- Compared: 600 tensors, 4,647,449,891 elements
- Missing / shape / dtype / bitwise mismatches: `0 / 0 / 0 / 0`
- PLE: 72 dense F32 entries, bitwise exact
- Tied head: absent on disk; loaded head shares embedding storage
- FFN widths: layer 14 = 6,144; layer 15 = 12,288
- K/V: all 35 physical K and all 35 physical V projections compared exactly

## Verification

- `cargo fmt --all -- --check`: passed
- Required tests: passed (`larql-models` 425; `larql-vindex` lib 1,154; all vindex test binaries passed including 111 tests in `test_vindex`; `larql-cli` 243; `larql-compute` 750; `larql-inference` 1,262)
- Required clippy checks for models, vindex, CLI, compute, and inference: passed with `-D warnings`
- `cargo build -p larql-cli --release`: passed
- Exact-source roundtrip: 1 passed, 0 failed
- GitHub Actions: all triggered GitHub Actions workflows passed

`LARQL-INFERENCE-TRUST-001A-ST3` is unblocked for tokenizer, chat-template, BOS/EOS, and prompt-token parity work.

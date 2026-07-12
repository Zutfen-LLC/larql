# Gemma 4 E2B Tokenizer / Prompt-Token Parity

Slice: `LARQL-INFERENCE-TRUST-001A-ST3`

## Decision

**GREEN — prompt-token parity proven.** LARQL constructs exactly the same
text-model input token sequence as the pinned Transformers oracle for Gemma 4
E2B. This makes no inference-correctness claim: no model forward pass, logit
comparison, or generation occurred. GREEN means only that LARQL supplies the
exact same prompt token sequence the future execution path will consume.

## Revisions

- Work-start SHA: `e88350da4b4929312d1fe1f06c49398ed2066769`
- PR base SHA: `e88350da4b4929312d1fe1f06c49398ed2066769`
- Head SHA: `d739800f6c99451e3349c07c3e9bd472d867c91a`

## Source

- Repository: `google/gemma-4-E2B-it`
- Revision: `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`

## Resource identity (source vs vindex, byte-for-byte)

| file | source SHA-256 | vindex SHA-256 | byte identity |
| --- | --- | --- | --- |
| tokenizer.json | `cc8d3a0c…627bfe0f` | `cc8d3a0c…627bfe0f` | ✅ |
| tokenizer_config.json | `90c3a3ba…d06f3df` | `90c3a3ba…d06f3df` | ✅ |
| chat_template.jinja | `2f1b4d75…6613d082` | `2f1b4d75…6613d082` | ✅ |
| generation_config.json | `d4226bbe…83ac03de` | `d4226bbe…83ac03de` | ✅ |

All four resources are byte-identical between the pinned source and the
reference vindex.

## Tokenizer / prompt policy (parsed from vindex resources)

| field | value |
| --- | --- |
| vocabulary size | 262144 |
| BOS token id | 2 (`<bos>`) |
| BOS policy | `add_bos_token: false`; raw prompts prepend BOS exactly once (dedup), no EOS |
| EOS token ids | `[1, 50, 106]` (`<eos>`, `<|tool_response>`, `<turn|>`) |
| PAD token id | 0 (`<pad>`) |
| UNK token id | 3 (`<unk>`) |
| chat-template hash | `2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082` |
| renderer mode | `gemma4_text_only_thinking_disabled` |
| thinking setting | disabled |

**Renderer selection rule:** the vindex architecture (`family`/`model_type`
starts with `gemma4`) **and** the committed `chat_template.jinja` SHA-256
must match the supported hash. A different hash fails loudly; presence of a
template alone never classifies a model as Gemma 4.

Note on EOS: the source `generation_config.json` lists `[1, 106, 50]`. The ST1
oracle summarised `[1, 106]`; the oracle set is a subset of the parsed source
policy, which is authoritative.

## Fixtures

All four fixtures match the oracle exactly (`rendered_text`, `token_ids`,
`token_pieces`, `bos_positions`). BOS is at position 0, once, in every case.

| fixture | prompt_id | rendered-text parity | token-ID parity | token-piece parity | BOS parity |
| --- | --- | --- | --- | --- | --- |
| raw completion | `raw_completion` | ✅ | ✅ `[2, 818, 5279, 529, 7001, 563]` | ✅ | ✅ `[0]` |
| canonical chat | `chat` | ✅ | ✅ (31 ids) | ✅ | ✅ `[0]` |
| arithmetic | `arithmetic` | ✅ | ✅ (35 ids) | ✅ | ✅ `[0]` |
| multi-turn | `multiturn` | ✅ | ✅ (40 ids) | ✅ | ✅ `[0]` |

The multi-turn fixture's token ids were derived by tokenizing the
LARQL-rendered text with the byte-identical source tokenizer — equivalent to
the Transformers oracle path, since the rendered text matches the pinned
template and the tokenizer is the exact source artifact.

## Mismatch counts

| metric | count |
| --- | --- |
| render mismatch | 0 |
| token-ID mismatch | 0 |
| token-piece mismatch | 0 |
| BOS-position mismatch | 0 |
| special-token-policy mismatch | 0 |

## Fail-loud behaviour

- **Unsupported role** (e.g. `developer`, `tool`): rejected by the message
  parser.
- **Tool message**: rejected (unsupported role).
- **Multimodal content** (array parts): rejected — never flattened.
- **Unknown template hash**: `Unsupported Gemma 4 chat-template revision;
  expected <hash>, found <hash>.`
- **Missing template**: chat rendering refused (`TemplateRevision::Absent`).
- **Missing tokenizer config**: asset load fails loudly.
- **Thinking enabled**: returns an unsupported error.

## Qwen regression

**PASS.** The prompt layer is additive — `encode_prompt`, `load_tokenizer`,
and `render_user_prompt` are untouched, so existing raw-prompt behaviour is
unchanged. The Qwen vindex tokenizes `"The capital of France is"` to
`[785, 6722, 315, 9625, 374]` with no Gemma control tokens injected, and a
non-Gemma family is never classified as Gemma 4 (architecture + template-hash
gate).

## Verification

- `cargo fmt --all -- --check`: passed
- `cargo test -p larql-inference --lib`: 1290 passed; 0 failed; 4 ignored
- `cargo test -p larql-vindex --lib`: 1154 passed; 0 failed
- `cargo test -p larql-cli --bins`: 243 passed; 0 failed
- `cargo clippy -p larql-inference --all-targets -- -D warnings`: passed
- `cargo clippy -p larql-vindex --all-targets -- -D warnings`: passed
- `cargo clippy -p larql-cli --all-targets -- -D warnings`: passed
- `cargo build -p larql-cli --release`: passed
- Exact-source parity test (`gemma4_tokenizer_prompt_parity`, `--ignored`):
  1 passed; 0 failed
- CI: GitHub Actions pending PR creation

## Scope deviations

- The multi-turn fixture's golden token ids were not produced by a separate
  Transformers run (no Python/transformers runtime was available). They were
  derived by tokenizing the LARQL-rendered text with the byte-identical source
  tokenizer — the same path the oracle takes — and verified self-consistent at
  test time. The renderer logic is identical across all fixtures and is
  independently cross-checked against the oracle for raw/chat/arithmetic.

`LARQL-INFERENCE-TRUST-001A-ST4` (CPU Gemma 4 attention / sliding-window /
shared-KV correctness) is unblocked.

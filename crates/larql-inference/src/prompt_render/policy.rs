//! Tokenizer-asset loader and parsed prompt policy.
//!
//! Loads the four HF tokenizer resources that ST2 snapshotted into the
//! vindex — `tokenizer.json`, `tokenizer_config.json`,
//! `generation_config.json`, `chat_template.jinja` — directly from the
//! vindex directory (never from the Hugging Face source at ordinary
//! runtime), and reduces them to a compact [`TokenizerPolicy`] that the
//! prompt renderer consumes.
//!
//! The policy is the single source of truth for BOS / EOS / PAD / UNK
//! behaviour. Special-token IDs are resolved from the parsed config
//! files (integer `*_token_id` fields where present, otherwise the
//! `*_token` string looked up against the loaded tokenizer vocabulary)
//! so production code never hardcodes them. Tests may pin expected
//! values.

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::Value;

use larql_vindex::format::filenames::{GENERATION_CONFIG_JSON, TOKENIZER_CONFIG_JSON};
use larql_vindex::{load_vindex_config, load_vindex_tokenizer};

use crate::error::InferenceError;

/// Filename of the committed Gemma 4 chat template inside the vindex.
pub const CHAT_TEMPLATE_JINJA: &str = "chat_template.jinja";

/// Which chat-template revision the loaded vindex carries.
///
/// The Gemma 4 text-only renderer is only selected when the vindex
/// architecture is Gemma 4 *and* the committed `chat_template.jinja`
/// hash matches [`super::gemma4::SUPPORTED_GEMMA4_TEMPLATE_HASH`]. Any
/// other combination is [`TemplateRevision::Unknown`] (a different
/// template is present) or [`TemplateRevision::Absent`] (no template at
/// all); both refuse chat rendering so an approximate render never
/// silently reaches the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateRevision {
    /// Gemma 4 text-only, thinking-disabled template at the pinned hash.
    Gemma4Text,
    /// A chat template is present but its SHA-256 is not the supported
    /// Gemma 4 revision. The hex hash is carried for diagnostics.
    Unknown(String),
    /// No `chat_template.jinja` in the vindex directory.
    Absent,
}

/// Parsed prompt/tokenizer policy for a vindex.
///
/// Every field is derived from the vindex's own committed resources so
/// the policy travels with the artifact. The renderer consults these
/// fields instead of hardcoding Gemma 4 magic numbers.
#[derive(Debug, Clone)]
pub struct TokenizerPolicy {
    /// Configured BOS token id, resolved from `generation_config.json`
    /// (`bos_token_id`) or the `bos_token` string in
    /// `tokenizer_config.json`. `None` when neither is resolvable.
    pub bos_token_id: Option<u32>,
    /// End-of-generation token ids, parsed from `generation_config.json`
    /// (`eos_token_id`, scalar or array) and merged with any
    /// `tokenizer_config.json` value. Sorted ascending, deduplicated.
    pub eos_token_ids: Vec<u32>,
    /// Padding token id from `generation_config.json` /
    /// `tokenizer_config.json`.
    pub pad_token_id: Option<u32>,
    /// Unknown-token id resolved from the `unk_token` string.
    pub unk_token_id: Option<u32>,
    /// Whether the tokenizer engine adds BOS automatically. Carried for
    /// reporting parity with the source config; the Gemma 4 raw-prompt
    /// path prepends BOS explicitly regardless of this flag (matching
    /// the pinned oracle — see [`crate::encode_raw`]).
    pub add_bos_token: bool,
    /// Whether the tokenizer engine adds EOS automatically.
    pub add_eos_token: bool,
    /// Logical (tokenizer) vocabulary size, validated to match the
    /// tokenizer's own vocabulary length.
    pub vocabulary_size: usize,
    /// SHA-256 of `chat_template.jinja` when present.
    pub chat_template_hash: Option<String>,
    /// Resolved template revision — selects the renderer.
    pub template_revision: TemplateRevision,
    /// True when the vindex architecture is Gemma 4
    /// (`family`/`model_type` starts with `gemma4`).
    pub is_gemma4: bool,
    /// Architecture family string from `index.json` (e.g. `gemma4`).
    pub family: String,
    /// `model_type` from `index.json::model_config` (e.g. `gemma4_text`).
    pub model_type: String,
}

/// Load the tokenizer and parse the prompt policy from a vindex
/// directory.
///
/// Requires `tokenizer.json`, `tokenizer_config.json`, and
/// `generation_config.json` to be present and parseable.
/// `chat_template.jinja` is optional — its absence yields
/// [`TemplateRevision::Absent`] and disables chat rendering while
/// leaving raw-prompt encoding intact.
pub fn load_assets(dir: &Path) -> Result<(tokenizers::Tokenizer, TokenizerPolicy), InferenceError> {
    let tokenizer = load_vindex_tokenizer(dir)?;
    let tokenizer_vocab = tokenizer.get_vocab_size(true);

    let cfg = load_vindex_config(dir)?;
    let logical_vocab = cfg.logical_vocab_size.unwrap_or(cfg.vocab_size);

    let family = cfg.family.clone();
    let model_type = cfg
        .model_config
        .as_ref()
        .map(|m| m.model_type.clone())
        .unwrap_or_default();
    let is_gemma4 = family.starts_with("gemma4") || model_type.starts_with("gemma4");

    let tok_cfg = read_json(dir, TOKENIZER_CONFIG_JSON, "tokenizer_config.json")?;
    let gen_cfg = read_json(dir, GENERATION_CONFIG_JSON, "generation_config.json")?;

    // ── special-token ids ───────────────────────────────────────────
    let bos_token_id = resolve_token_id(&gen_cfg, &tok_cfg, "bos", &tokenizer);
    let pad_token_id = resolve_token_id(&gen_cfg, &tok_cfg, "pad", &tokenizer);
    let unk_token_id = resolve_token_id(&gen_cfg, &tok_cfg, "unk", &tokenizer);

    // EOS may be a scalar or an array in either config; union both.
    let mut eos: BTreeSet<u32> = BTreeSet::new();
    for ids in [
        parse_eos_token_id(&gen_cfg, "eos"),
        parse_eos_token_id(&tok_cfg, "eos"),
    ] {
        eos.extend(ids);
    }
    // If eos_token_id wasn't an integer field, fall back to the
    // eos_token string (e.g. `<eos>`) resolved through the vocabulary.
    if eos.is_empty() {
        if let Some(id) = resolve_token_string(&tok_cfg, "eos", &tokenizer) {
            eos.insert(id);
        }
    }
    let eos_token_ids = eos.into_iter().collect::<Vec<_>>();

    let add_bos_token = tok_cfg
        .get("add_bos_token")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let add_eos_token = tok_cfg
        .get("add_eos_token")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // ── chat-template hash ──────────────────────────────────────────
    let template_path = dir.join(CHAT_TEMPLATE_JINJA);
    let (chat_template_hash, template_revision) = if template_path.is_file() {
        let bytes = std::fs::read(&template_path).map_err(InferenceError::Io)?;
        let hash = sha256_hex(&bytes);
        let revision = if is_gemma4 && hash == super::gemma4::SUPPORTED_GEMMA4_TEMPLATE_HASH {
            TemplateRevision::Gemma4Text
        } else {
            TemplateRevision::Unknown(hash.clone())
        };
        (Some(hash), revision)
    } else {
        (None, TemplateRevision::Absent)
    };

    let policy = TokenizerPolicy {
        bos_token_id,
        eos_token_ids,
        pad_token_id,
        unk_token_id,
        add_bos_token,
        add_eos_token,
        vocabulary_size: logical_vocab,
        chat_template_hash,
        template_revision,
        is_gemma4,
        family,
        model_type,
    };

    validate(&policy, tokenizer_vocab)?;
    Ok((tokenizer, policy))
}

fn validate(policy: &TokenizerPolicy, tokenizer_vocab: usize) -> Result<(), InferenceError> {
    // Tokenizer vocabulary must match the logical model vocabulary.
    if tokenizer_vocab != policy.vocabulary_size {
        return Err(InferenceError::PromptRender(format!(
            "tokenizer vocabulary size ({tokenizer_vocab}) does not match the logical model \
             vocabulary size ({}); the vindex tokenizer and embedding disagree",
            policy.vocabulary_size
        )));
    }

    let within = |id: u32, label: &str| {
        if (id as usize) >= policy.vocabulary_size {
            Err(InferenceError::PromptRender(format!(
                "{label} token id {id} is outside the vocabulary (size {})",
                policy.vocabulary_size
            )))
        } else {
            Ok(())
        }
    };

    if let Some(id) = policy.bos_token_id {
        within(id, "bos")?;
    }
    if let Some(id) = policy.pad_token_id {
        within(id, "pad")?;
    }
    if let Some(id) = policy.unk_token_id {
        within(id, "unk")?;
    }
    for id in &policy.eos_token_ids {
        within(*id, "eos")?;
    }
    Ok(())
}

fn read_json(dir: &Path, name: &str, label: &str) -> Result<Value, InferenceError> {
    let path = dir.join(name);
    if !path.is_file() {
        return Err(InferenceError::PromptRender(format!(
            "{label} not found in vindex directory"
        )));
    }
    let bytes = std::fs::read(&path).map_err(InferenceError::Io)?;
    serde_json::from_slice(&bytes)
        .map_err(|e| InferenceError::PromptRender(format!("{label} parse error: {e}")))
}

/// Resolve a special-token id from the integer `*_token_id` field in
/// either config, falling back to the `*_token` string looked up in the
/// tokenizer vocabulary. `None` when neither is present/resolvable.
fn resolve_token_id(
    gen_cfg: &Value,
    tok_cfg: &Value,
    name: &str,
    tokenizer: &tokenizers::Tokenizer,
) -> Option<u32> {
    let id_key = format!("{name}_token_id");
    for cfg in [gen_cfg, tok_cfg] {
        if let Some(id) = cfg.get(&id_key).and_then(value_as_u32) {
            return Some(id);
        }
    }
    resolve_token_string(tok_cfg, name, tokenizer)
}

/// Resolve a `*_token` string field (e.g. `bos_token: "<bos>"`) against
/// the loaded tokenizer vocabulary.
fn resolve_token_string(
    tok_cfg: &Value,
    name: &str,
    tokenizer: &tokenizers::Tokenizer,
) -> Option<u32> {
    let str_key = format!("{name}_token");
    let token = tok_cfg.get(&str_key).and_then(token_string_value)?;
    tokenizer.token_to_id(token)
}

/// Extract the surface string from a `*_token` field, which may be a
/// plain string (`"<bos>"`) or a `{"content": "<bos>"}` object.
fn token_string_value(v: &Value) -> Option<&str> {
    match v {
        Value::String(s) => Some(s),
        Value::Object(_) => v.get("content").and_then(Value::as_str),
        _ => None,
    }
}

/// Parse `eos_token_id` from a config value, accepting a scalar integer
/// or an array of integers. Other shapes yield an empty vec.
pub(crate) fn parse_eos_token_id(cfg: &Value, name: &str) -> Vec<u32> {
    let key = format!("{name}_token_id");
    match cfg.get(&key) {
        Some(Value::Number(n)) => value_as_u32(&Value::Number(n.clone()))
            .map(|id| vec![id])
            .unwrap_or_default(),
        Some(Value::Array(arr)) => arr.iter().filter_map(value_as_u32).collect(),
        _ => Vec::new(),
    }
}

fn value_as_u32(v: &Value) -> Option<u32> {
    v.as_u64().and_then(|n| u32::try_from(n).ok())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::{VindexConfig, VindexModelConfig};

    /// Build a synthetic tokenizer.json (WordLevel) with the named
    /// special tokens registered. The special tokens occupy their given
    /// ids within a vocabulary of exactly `vocab` entries (generics
    /// fill the remaining slots), mirroring how a real tokenizer.json
    /// keeps specials inside the base vocabulary.
    fn write_tokenizer(dir: &Path, special: &[(&str, u32)], vocab: usize) {
        use std::collections::HashSet;
        let mut vocab_map = serde_json::Map::new();
        let mut used_ids: HashSet<u32> = special.iter().map(|(_, id)| *id).collect();
        for (content, id) in special {
            vocab_map.insert(
                (*content).to_string(),
                serde_json::Value::Number((*id as u64).into()),
            );
        }
        let mut next_id = 0u32;
        while vocab_map.len() < vocab {
            if used_ids.insert(next_id) {
                vocab_map.insert(
                    format!("t{next_id}"),
                    serde_json::Value::Number((next_id as u64).into()),
                );
            }
            next_id += 1;
        }
        let added: Vec<Value> = special
            .iter()
            .map(|(content, id)| {
                serde_json::json!({
                    "id": id, "content": content, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false, "special": true
                })
            })
            .collect();
        let tj = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": added, "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab_map, "unk_token": "t0"},
        });
        std::fs::write(dir.join(TOKENIZER_JSON), serde_json::to_vec(&tj).unwrap()).unwrap();
    }

    use larql_vindex::format::filenames::{INDEX_JSON, TOKENIZER_JSON};

    fn write_index(dir: &Path, family: &str, model_type: &str, vocab: usize) {
        let cfg = VindexConfig {
            version: 1,
            model: "synthetic".into(),
            family: family.into(),
            source: None,
            checksums: None,
            num_layers: 2,
            hidden_size: 8,
            intermediate_size: 16,
            vocab_size: vocab,
            logical_vocab_size: None,
            embed_scale: 1.0,
            extract_level: larql_vindex::ExtractLevel::Browse,
            dtype: larql_vindex::StorageDtype::F32,
            quant: larql_vindex::QuantFormat::None,
            layer_bands: None,
            layers: vec![],
            down_top_k: 0,
            has_model_weights: false,
            model_config: Some(VindexModelConfig {
                model_type: model_type.into(),
                head_dim: 8,
                num_q_heads: 2,
                num_kv_heads: 1,
                rope_base: 10000.0,
                sliding_window: None,
                moe: None,
                global_head_dim: None,
                num_global_kv_heads: None,
                partial_rotary_factor: None,
                sliding_window_pattern: None,
                layer_types: None,
                attention_k_eq_v: false,
                num_kv_shared_layers: None,
                per_layer_embed_dim: None,
                rope_local_base: None,
                query_pre_attn_scalar: None,
                final_logit_softcapping: None,
                attention_multiplier: None,
                residual_multiplier: None,
                logits_scaling: None,
                norm_eps: None,
            }),
            fp4: None,
            ffn_layout: None,
            bitnet_layout: None,
        };
        std::fs::write(dir.join(INDEX_JSON), serde_json::to_vec(&cfg).unwrap()).unwrap();
    }

    fn gemma4_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let v = 262144;
        write_tokenizer(
            dir.path(),
            &[
                ("<bos>", 2),
                ("<eos>", 1),
                ("<pad>", 0),
                ("<unk>", 3),
                ("<|turn>", 105),
                ("<turn|>", 106),
            ],
            v,
        );
        write_index(dir.path(), "gemma4", "gemma4_text", v);
        std::fs::write(
            dir.path().join(TOKENIZER_CONFIG_JSON),
            serde_json::json!({
                "bos_token": "<bos>", "eos_token": "<eos>",
                "pad_token": "<pad>", "unk_token": "<unk>",
                "add_bos_token": false, "add_eos_token": false,
                "tokenizer_class": "GemmaTokenizer",
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join(GENERATION_CONFIG_JSON),
            serde_json::json!({
                "bos_token_id": 2, "eos_token_id": [1, 106, 50], "pad_token_id": 0,
            })
            .to_string(),
        )
        .unwrap();
        // Write the real Gemma 4 template hash so Gemma4Text is selected.
        let template_bytes = include_bytes!("../../tests/fixtures/gemma4_chat_template.jinja");
        std::fs::write(dir.path().join(CHAT_TEMPLATE_JINJA), template_bytes).unwrap();
        dir
    }

    #[test]
    fn loads_gemma4_policy_from_resources() {
        let dir = gemma4_dir();
        let (tok, policy) = load_assets(dir.path()).expect("load succeeds");
        assert_eq!(policy.bos_token_id, Some(2));
        assert_eq!(policy.eos_token_ids, vec![1, 50, 106]);
        assert_eq!(policy.pad_token_id, Some(0));
        assert_eq!(policy.unk_token_id, Some(3));
        assert!(!policy.add_bos_token);
        assert!(!policy.add_eos_token);
        assert_eq!(policy.vocabulary_size, 262144);
        assert!(policy.is_gemma4);
        assert_eq!(policy.template_revision, TemplateRevision::Gemma4Text);
        assert_eq!(tok.get_vocab_size(true), 262144);
    }

    #[test]
    fn scalar_eos_config_parses() {
        let v: Value = serde_json::from_str(r#"{"eos_token_id": 7}"#).unwrap();
        assert_eq!(parse_eos_token_id(&v, "eos"), vec![7]);
    }

    #[test]
    fn array_eos_config_parses() {
        // parse_eos_token_id preserves source order; load_assets sorts
        // via BTreeSet before exposing eos_token_ids.
        let v: Value = serde_json::from_str(r#"{"eos_token_id": [1, 106, 50]}"#).unwrap();
        assert_eq!(parse_eos_token_id(&v, "eos"), vec![1, 106, 50]);
    }

    #[test]
    fn invalid_eos_id_fails() {
        let dir = gemma4_dir();
        // Overwrite generation config with an out-of-vocab EOS id.
        std::fs::write(
            dir.path().join(GENERATION_CONFIG_JSON),
            serde_json::json!({"bos_token_id": 2, "eos_token_id": [1, 9_999_999], "pad_token_id": 0})
                .to_string(),
        )
        .unwrap();
        let err = load_assets(dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("outside the vocabulary"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tokenizer_vocabulary_mismatch_fails() {
        let dir = tempfile::tempdir().unwrap();
        // Tokenizer has 8 tokens, model claims 16.
        write_tokenizer(dir.path(), &[("<bos>", 2)], 8);
        write_index(dir.path(), "gemma4", "gemma4_text", 16);
        std::fs::write(
            dir.path().join(TOKENIZER_CONFIG_JSON),
            r#"{"bos_token":"<bos>","unk_token":"t0"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join(GENERATION_CONFIG_JSON),
            r#"{"bos_token_id":2,"eos_token_id":[1],"pad_token_id":0}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join(CHAT_TEMPLATE_JINJA),
            include_bytes!("../../tests/fixtures/gemma4_chat_template.jinja"),
        )
        .unwrap();
        let err = load_assets(dir.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("does not match the logical model vocabulary"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_tokenizer_config_fails_loudly() {
        let dir = tempfile::tempdir().unwrap();
        write_tokenizer(dir.path(), &[("<bos>", 2)], 16);
        write_index(dir.path(), "gemma4", "gemma4_text", 16);
        // No tokenizer_config.json, no generation_config.json.
        let err = load_assets(dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("tokenizer_config.json not found"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_template_yields_absent_revision() {
        let dir = tempfile::tempdir().unwrap();
        write_tokenizer(dir.path(), &[("<bos>", 2)], 16);
        write_index(dir.path(), "qwen2", "qwen2", 16);
        std::fs::write(
            dir.path().join(TOKENIZER_CONFIG_JSON),
            r#"{"unk_token":"t0"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join(GENERATION_CONFIG_JSON),
            r#"{"eos_token_id":[1]}"#,
        )
        .unwrap();
        let (_tok, policy) = load_assets(dir.path()).unwrap();
        assert_eq!(policy.template_revision, TemplateRevision::Absent);
        assert!(!policy.is_gemma4);
        assert_eq!(policy.chat_template_hash, None);
    }

    #[test]
    fn unknown_template_hash_yields_unknown_revision() {
        let dir = tempfile::tempdir().unwrap();
        write_tokenizer(dir.path(), &[("<bos>", 2)], 16);
        write_index(dir.path(), "gemma4", "gemma4_text", 16);
        std::fs::write(
            dir.path().join(TOKENIZER_CONFIG_JSON),
            r#"{"unk_token":"t0"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join(GENERATION_CONFIG_JSON),
            r#"{"bos_token_id":2,"eos_token_id":[1],"pad_token_id":0}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join(CHAT_TEMPLATE_JINJA),
            b"not the gemma4 template",
        )
        .unwrap();
        let (_tok, policy) = load_assets(dir.path()).unwrap();
        match policy.template_revision {
            TemplateRevision::Unknown(ref h) => {
                assert_eq!(h, &sha256_hex(b"not the gemma4 template"));
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn non_gemma4_family_not_classified_as_gemma4_with_template() {
        // Presence of chat_template.jinja alone must NOT select the
        // Gemma 4 renderer — architecture (family) is also required.
        let dir = tempfile::tempdir().unwrap();
        write_tokenizer(dir.path(), &[("<bos>", 2)], 16);
        write_index(dir.path(), "qwen2", "qwen2", 16);
        std::fs::write(
            dir.path().join(TOKENIZER_CONFIG_JSON),
            r#"{"unk_token":"t0"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join(GENERATION_CONFIG_JSON),
            r#"{"eos_token_id":[1]}"#,
        )
        .unwrap();
        // Drop the *real* Gemma 4 template bytes here; even with the
        // exact hash, a non-Gemma family stays Unknown/Absent.
        std::fs::write(
            dir.path().join(CHAT_TEMPLATE_JINJA),
            include_bytes!("../../tests/fixtures/gemma4_chat_template.jinja"),
        )
        .unwrap();
        let (_tok, policy) = load_assets(dir.path()).unwrap();
        assert!(!policy.is_gemma4);
        // Even though the hash matches, family != gemma4 → Unknown.
        assert_ne!(policy.template_revision, TemplateRevision::Gemma4Text);
    }
}

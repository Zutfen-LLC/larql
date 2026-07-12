//! Narrow prompt-construction layer for the exact official Gemma 4
//! text-only, thinking-disabled chat template.
//!
//! ST3 (`LARQL-INFERENCE-TRUST-001A-ST3`) proves that LARQL constructs
//! exactly the same text-model input token sequence as the pinned
//! Transformers oracle. This module is the deterministic
//! prompt-construction API that feeds the future model execution path.
//!
//! The layer loads all tokenizer assets from the active vindex
//! ([`PromptAssets::load_from_vindex`]), applies the Gemma 4 prompt
//! policy, and returns a [`PromptEncoding`] carrying the rendered text,
//! exact token ids, token pieces, and BOS positions. No model weights
//! are loaded — the trust boundary ends at the token-id sequence the
//! model will consume.
//!
//! This is an **additive** API: it does not alter the existing
//! `encode_prompt` / `render_user_prompt` paths used by `larql run` and
//! the LQL executor, so Qwen and other existing models keep their
//! behaviour. Chat rendering is only authorised for the Gemma 4
//! architecture whose committed `chat_template.jinja` hashes to
//! [`gemma4::SUPPORTED_GEMMA4_TEMPLATE_HASH`]; every other template or
//! architecture is refused rather than rendered approximately.

pub mod gemma4;
pub mod policy;

use std::path::Path;

use crate::error::InferenceError;

pub use policy::{TemplateRevision, TokenizerPolicy};

/// Input to prompt construction.
#[derive(Debug, Clone)]
pub enum PromptInput {
    /// Raw text completion — tokenised verbatim with the source BOS
    /// policy applied, no chat wrapping, no EOS appended.
    Raw(String),
    /// Structured chat conversation rendered through the model's chat
    /// template (thinking-disabled).
    Chat(Vec<ChatMessage>),
}

/// A single chat message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

/// Role of a chat message. The Gemma 4 template writes `model` for the
/// responding role; [`ChatRole::Assistant`] maps to that token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// Thinking-mode request. Only [`ThinkingMode::Disabled`] is supported
/// by the Gemma 4 text-only renderer in ST3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingMode {
    #[default]
    Disabled,
    Enabled,
}

/// A rendered prompt together with its exact tokenisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptEncoding {
    /// The rendered prompt text (raw text for [`PromptInput::Raw`],
    /// template output for [`PromptInput::Chat`]).
    pub rendered_text: String,
    /// Exact token ids fed to the model.
    pub token_ids: Vec<u32>,
    /// Token pieces as returned by the tokenizer (one per id).
    pub token_pieces: Vec<String>,
    /// Positions (indices into `token_ids`) holding the configured BOS
    /// token. A correctly formed Gemma 4 prompt has exactly one BOS at
    /// position 0.
    pub bos_positions: Vec<usize>,
}

/// Loaded tokenizer assets + parsed prompt policy for a vindex.
#[derive(Debug)]
pub struct PromptAssets {
    pub tokenizer: tokenizers::Tokenizer,
    pub policy: TokenizerPolicy,
}

impl PromptAssets {
    /// Load and validate all tokenizer assets from a vindex directory.
    /// Does not load model weights.
    pub fn load_from_vindex(vindex_dir: &Path) -> Result<Self, InferenceError> {
        let (tokenizer, policy) = policy::load_assets(vindex_dir)?;
        Ok(Self { tokenizer, policy })
    }

    /// Render and encode a prompt (thinking-disabled).
    pub fn encode(&self, input: &PromptInput) -> Result<PromptEncoding, InferenceError> {
        self.encode_with_thinking(input, ThinkingMode::Disabled)
    }

    /// Render and encode a prompt with an explicit thinking mode.
    pub fn encode_with_thinking(
        &self,
        input: &PromptInput,
        thinking: ThinkingMode,
    ) -> Result<PromptEncoding, InferenceError> {
        match input {
            PromptInput::Raw(text) => encode_raw(&self.tokenizer, &self.policy, text),
            PromptInput::Chat(messages) => {
                encode_chat(&self.tokenizer, &self.policy, messages, thinking)
            }
        }
    }
}

/// Tokenise a raw prompt with the Gemma 4 BOS policy.
///
/// Gemma 4's shipped `tokenizer.json` does not add BOS automatically
/// (`add_bos_token: false`), but the model requires a single leading
/// BOS, so it is prepended explicitly (matching the pinned oracle). If
/// the encoded content already starts with the BOS id it is not
/// duplicated. EOS is never appended for raw prompts.
fn encode_raw(
    tokenizer: &tokenizers::Tokenizer,
    policy: &TokenizerPolicy,
    text: &str,
) -> Result<PromptEncoding, InferenceError> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| InferenceError::Parse(format!("tokenize error: {e}")))?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    let mut pieces: Vec<String> = encoding.get_tokens().to_vec();

    if policy.is_gemma4 {
        if let Some(bos) = policy.bos_token_id {
            if ids.first().copied() != Some(bos) {
                let bos_piece = tokenizer
                    .id_to_token(bos)
                    .unwrap_or_else(|| format!("[{bos}]"));
                ids.insert(0, bos);
                pieces.insert(0, bos_piece);
            }
        }
    }

    let bos_positions = bos_positions(&ids, policy.bos_token_id);
    Ok(PromptEncoding {
        rendered_text: text.to_string(),
        token_ids: ids,
        token_pieces: pieces,
        bos_positions,
    })
}

/// Render and tokenise a chat prompt through the supported Gemma 4
/// template. Any non-Gemma4 architecture or template hash is refused.
fn encode_chat(
    tokenizer: &tokenizers::Tokenizer,
    policy: &TokenizerPolicy,
    messages: &[ChatMessage],
    thinking: ThinkingMode,
) -> Result<PromptEncoding, InferenceError> {
    let rendered = match &policy.template_revision {
        TemplateRevision::Gemma4Text => gemma4::render_chat(messages, thinking)?,
        TemplateRevision::Unknown(found) => {
            return Err(InferenceError::PromptRender(format!(
                "Unsupported Gemma 4 chat-template revision; expected {}, found {}.",
                gemma4::SUPPORTED_GEMMA4_TEMPLATE_HASH,
                found
            )));
        }
        TemplateRevision::Absent => {
            return Err(InferenceError::PromptRender(
                "chat_template.jinja is missing from the vindex; cannot render Gemma 4 chat prompts"
                    .into(),
            ));
        }
    };

    let encoding = tokenizer
        .encode(rendered.as_str(), true)
        .map_err(|e| InferenceError::Parse(format!("tokenize error: {e}")))?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let pieces: Vec<String> = encoding.get_tokens().to_vec();
    let bos_positions = bos_positions(&ids, policy.bos_token_id);
    Ok(PromptEncoding {
        rendered_text: rendered,
        token_ids: ids,
        token_pieces: pieces,
        bos_positions,
    })
}

/// Indices of the BOS token id within `ids` (preserving any later BOS
/// occurrences in ordinary content — they are recorded, never removed).
fn bos_positions(ids: &[u32], bos_token_id: Option<u32>) -> Vec<usize> {
    let Some(bos) = bos_token_id else {
        return Vec::new();
    };
    ids.iter()
        .enumerate()
        .filter(|(_, id)| **id == bos)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build PromptAssets around a tiny synthetic WordLevel tokenizer
    /// flagged as Gemma 4, with `<bos>` registered as special token 2.
    fn gemma4_assets(vocab: usize) -> PromptAssets {
        let mut v = serde_json::Map::new();
        for i in 0..vocab as u64 {
            v.insert(format!("t{i}"), serde_json::Value::Number(i.into()));
        }
        v.insert("<bos>".into(), serde_json::Value::Number(2u64.into()));
        let tj = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [{"id": 2, "content": "<bos>", "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": v, "unk_token": "t0"},
        });
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&tj).unwrap()).unwrap();
        let policy = TokenizerPolicy {
            bos_token_id: Some(2),
            eos_token_ids: vec![1, 106],
            pad_token_id: Some(0),
            unk_token_id: Some(3),
            add_bos_token: false,
            add_eos_token: false,
            vocabulary_size: vocab,
            chat_template_hash: None,
            template_revision: TemplateRevision::Gemma4Text,
            is_gemma4: true,
            family: "gemma4".into(),
            model_type: "gemma4_text".into(),
        };
        PromptAssets { tokenizer, policy }
    }

    #[test]
    fn raw_prompt_receives_one_bos() {
        let assets = gemma4_assets(64);
        // Synthetic tokenizer encodes whitespace-split tokens; "t5 t6"
        // → [5, 6], then BOS prepended.
        let enc = assets.encode(&PromptInput::Raw("t5 t6".into())).unwrap();
        assert_eq!(enc.token_ids, vec![2, 5, 6]);
        assert_eq!(enc.bos_positions, vec![0]);
        assert_eq!(enc.token_pieces[0], "<bos>");
    }

    #[test]
    fn raw_prompt_does_not_receive_eos() {
        let assets = gemma4_assets(64);
        let enc = assets.encode(&PromptInput::Raw("t5".into())).unwrap();
        assert!(!enc.token_ids.contains(&1));
        assert!(!enc.token_ids.contains(&106));
    }

    #[test]
    fn existing_leading_bos_is_not_duplicated() {
        let assets = gemma4_assets(64);
        // `<bos>` is a registered special token → encoded to id 2 first.
        let enc = assets.encode(&PromptInput::Raw("<bos> t5".into())).unwrap();
        let bos_count = enc.token_ids.iter().filter(|&&id| id == 2).count();
        assert_eq!(bos_count, 1);
        assert_eq!(enc.bos_positions, vec![0]);
    }

    #[test]
    fn raw_rendered_text_is_verbatim_input() {
        let assets = gemma4_assets(64);
        let enc = assets.encode(&PromptInput::Raw("t5 t6".into())).unwrap();
        assert_eq!(enc.rendered_text, "t5 t6");
    }

    #[test]
    fn non_gemma4_raw_prompt_has_no_bos_prepended() {
        let mut assets = gemma4_assets(64);
        assets.policy.is_gemma4 = false;
        assets.policy.bos_token_id = None;
        let enc = assets.encode(&PromptInput::Raw("t5 t6".into())).unwrap();
        assert_eq!(enc.token_ids, vec![5, 6]);
        assert!(enc.bos_positions.is_empty());
    }
}

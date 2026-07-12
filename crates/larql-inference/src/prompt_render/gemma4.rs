//! Hand-coded renderer for the exact official Gemma 4 text-only,
//! thinking-disabled chat template.
//!
//! This is deliberately **not** a general Jinja interpreter. The pinned
//! `chat_template.jinja` from `google/gemma-4-E2B-it` is reduced to a
//! single deterministic Rust formatter bound to its SHA-256. When the
//! committed template hash matches [`SUPPORTED_GEMMA4_TEMPLATE_HASH`],
//! this renderer reproduces the byte-exact prompt text the Hugging Face
//! oracle emits for the text-only, `enable_thinking=false`,
//! `add_generation_prompt=true` subset. Any other template hash is
//! refused — there is no approximate fallback.
//!
//! Rendered shape (thinking-disabled):
//!
//! ```text
//! <bos><|turn>system
//! {system content}<turn|>
//! <|turn>user
//! {user content}<turn|>
//! …
//! <|turn>model
//! ```
//!
//! Each completed turn is `<|turn>{role}\n{content}<turn|>\n`; the
//! generation prompt closes with an open `<|turn>model\n` turn (no
//! `<turn|>`, no EOS). Role mapping: System → `system`, User → `user`,
//! Assistant → `model`.

use serde_json::Value;

use crate::error::InferenceError;
use crate::prompt_render::{ChatMessage, ChatRole, ThinkingMode};

/// SHA-256 of the committed `chat_template.jinja` from
/// `google/gemma-4-E2B-it` @ `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`.
///
/// The Gemma 4 text-only renderer is only authorised when the vindex's
/// template hashes to this value. A mismatch fails loudly.
pub const SUPPORTED_GEMMA4_TEMPLATE_HASH: &str =
    "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082";

/// Turn-open marker (special token id 105).
const TURN_OPEN: &str = "<|turn>";
/// Turn-close marker (special token id 106; also an EOS id).
const TURN_CLOSE: &str = "<turn|>";
/// Literal BOS written at the head of every rendered chat prompt.
const BOS_LITERAL: &str = "<bos>";

/// Render Gemma 4 chat messages to the exact template text.
///
/// Only the thinking-disabled, `add_generation_prompt=true` subset is
/// supported. `ThinkingMode::Enabled` returns a clear unsupported error
/// so a caller can never silently receive a thinking-preamble prompt.
pub fn render_chat(
    messages: &[ChatMessage],
    thinking: ThinkingMode,
) -> Result<String, InferenceError> {
    if matches!(thinking, ThinkingMode::Enabled) {
        return Err(InferenceError::PromptRender(
            "thinking-enabled mode is not supported by the Gemma 4 text-only renderer; \
             ST3 only proves the thinking-disabled prompt path"
                .into(),
        ));
    }

    validate_messages(messages)?;

    let mut out = String::new();
    out.push_str(BOS_LITERAL);
    for msg in messages {
        out.push_str(TURN_OPEN);
        out.push_str(role_token(msg.role));
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str(TURN_CLOSE);
        out.push('\n');
    }
    // add_generation_prompt = true → open the assistant (model) turn.
    out.push_str(TURN_OPEN);
    out.push_str("model");
    out.push('\n');
    Ok(out)
}

/// Validate the message list against the supported Gemma 4 subset.
fn validate_messages(messages: &[ChatMessage]) -> Result<(), InferenceError> {
    for (i, msg) in messages.iter().enumerate() {
        if matches!(msg.role, ChatRole::System) && i != 0 {
            return Err(InferenceError::PromptRender(
                "a system message must be the first message in a Gemma 4 conversation".into(),
            ));
        }
    }
    Ok(())
}

/// Map a [`ChatRole`] to the token the template writes after `<|turn>`.
fn role_token(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        // The Gemma 4 template calls the responding role "model".
        ChatRole::Assistant => "model",
    }
}

/// Parse Hugging Face-style chat messages (`[{role, content}, …]`) into
/// typed [`ChatMessage`]s, rejecting unsupported roles and multimodal
/// content up front.
///
/// Accepts both `assistant` and `model` as the responding role (the
/// Gemma 4 template writes `model`; the HF messages convention writes
/// `assistant`). A `content` that is anything other than a string
/// (array of multimodal parts, object, null) is rejected — never
/// silently flattened.
pub fn parse_messages(messages: &[Value]) -> Result<Vec<ChatMessage>, InferenceError> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role_str = m.get("role").and_then(Value::as_str).ok_or_else(|| {
            InferenceError::PromptRender("chat message is missing a `role` string".into())
        })?;
        let role = match role_str {
            "system" => ChatRole::System,
            "user" => ChatRole::User,
            "assistant" | "model" => ChatRole::Assistant,
            other => {
                return Err(InferenceError::PromptRender(format!(
                    "unsupported chat role `{other}` for the Gemma 4 text-only template; only \
                     system, user, and assistant are supported"
                )));
            }
        };
        let content = match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(_)) => {
                return Err(InferenceError::PromptRender(format!(
                    "multimodal (array) content is not supported by the Gemma 4 text-only \
                     renderer; role `{role_str}` must carry plain text content"
                )));
            }
            Some(other) => {
                return Err(InferenceError::PromptRender(format!(
                    "unsupported content shape ({}) for role `{role_str}`; expected a string",
                    type_label(other)
                )));
            }
            None => String::new(),
        };
        out.push(ChatMessage { role, content });
    }
    Ok(out)
}

fn type_label(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys(c: &str) -> ChatMessage {
        ChatMessage {
            role: ChatRole::System,
            content: c.into(),
        }
    }
    fn usr(c: &str) -> ChatMessage {
        ChatMessage {
            role: ChatRole::User,
            content: c.into(),
        }
    }
    fn ast(c: &str) -> ChatMessage {
        ChatMessage {
            role: ChatRole::Assistant,
            content: c.into(),
        }
    }

    #[test]
    fn canonical_chat_render_is_byte_exact() {
        let msgs = [
            sys("You are a helpful assistant."),
            usr("In one sentence, explain why the sky appears blue."),
        ];
        let rendered = render_chat(&msgs, ThinkingMode::Disabled).unwrap();
        let expected = "<bos><|turn>system\nYou are a helpful assistant.<turn|>\n\
            <|turn>user\nIn one sentence, explain why the sky appears blue.<turn|>\n\
            <|turn>model\n";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn assistant_role_maps_to_model_token() {
        let msgs = [usr("Hi"), ast("Hello!"), usr("Bye")];
        let rendered = render_chat(&msgs, ThinkingMode::Disabled).unwrap();
        assert!(
            rendered.contains("<|turn>model\nHello!<turn|>\n"),
            "rendered: {rendered}"
        );
    }

    #[test]
    fn system_message_is_optional() {
        let msgs = [usr("Hello")];
        let rendered = render_chat(&msgs, ThinkingMode::Disabled).unwrap();
        assert!(rendered.starts_with("<bos><|turn>user\nHello<turn|>\n"));
        assert!(rendered.ends_with("<|turn>model\n"));
    }

    #[test]
    fn multiturn_ordering_preserved() {
        let msgs = [
            sys("You are a concise assistant."),
            usr("Name one primary color."),
            ast("Red."),
            usr("Name another."),
        ];
        let rendered = render_chat(&msgs, ThinkingMode::Disabled).unwrap();
        let expected = "<bos><|turn>system\nYou are a concise assistant.<turn|>\n\
            <|turn>user\nName one primary color.<turn|>\n\
            <|turn>model\nRed.<turn|>\n\
            <|turn>user\nName another.<turn|>\n\
            <|turn>model\n";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn generation_prompt_opens_model_turn() {
        let msgs = [usr("Hi")];
        let rendered = render_chat(&msgs, ThinkingMode::Disabled).unwrap();
        assert!(rendered.ends_with("<|turn>model\n"));
        assert!(!rendered.ends_with("model<turn|>"));
    }

    #[test]
    fn thinking_disabled_succeeds() {
        let msgs = [usr("Hi")];
        assert!(render_chat(&msgs, ThinkingMode::Disabled).is_ok());
    }

    #[test]
    fn thinking_enabled_fails_loudly() {
        let msgs = [usr("Hi")];
        let err = render_chat(&msgs, ThinkingMode::Enabled).unwrap_err();
        assert!(err.to_string().contains("thinking-enabled"), "{err}");
    }

    #[test]
    fn unsupported_role_fails_loudly() {
        let json = serde_json::json!([{"role": "developer", "content": "x"}]);
        let err = parse_messages(json.as_array().unwrap()).unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported chat role `developer`"),
            "{err}"
        );
    }

    #[test]
    fn system_after_first_message_fails() {
        let msgs = [usr("Hi"), sys("late system")];
        let err = render_chat(&msgs, ThinkingMode::Disabled).unwrap_err();
        assert!(
            err.to_string().contains("system message must be the first"),
            "{err}"
        );
    }

    #[test]
    fn parse_messages_rejects_tool_role() {
        let json = serde_json::json!([{"role": "tool", "content": "x"}]);
        let err = parse_messages(json.as_array().unwrap()).unwrap_err();
        assert!(
            err.to_string().contains("unsupported chat role `tool`"),
            "{err}"
        );
    }

    #[test]
    fn parse_messages_rejects_multimodal_content() {
        let json = serde_json::json!([{
            "role": "user",
            "content": [{"type": "text", "text": "hi"}, {"type": "image", "url": "x"}]
        }]);
        let err = parse_messages(json.as_array().unwrap()).unwrap_err();
        assert!(err.to_string().contains("multimodal"), "{err}");
    }

    #[test]
    fn parse_messages_maps_model_role_to_assistant() {
        let json = serde_json::json!([{"role": "user", "content": "q"}, {"role": "model", "content": "a"}]);
        let parsed = parse_messages(json.as_array().unwrap()).unwrap();
        assert_eq!(parsed[1].role, ChatRole::Assistant);
    }

    #[test]
    fn empty_user_content_renders_open_close_turn() {
        let msgs = [usr("")];
        let rendered = render_chat(&msgs, ThinkingMode::Disabled).unwrap();
        assert!(
            rendered.contains("<|turn>user\n<turn|>\n"),
            "rendered: {rendered}"
        );
    }

    #[test]
    fn unicode_and_newlines_preserved_verbatim() {
        let msgs = [usr("héllo—世界\nline two\t✓")];
        let rendered = render_chat(&msgs, ThinkingMode::Disabled).unwrap();
        assert!(
            rendered.contains("<|turn>user\nhéllo—世界\nline two\t✓<turn|>\n"),
            "rendered: {rendered}"
        );
    }
}

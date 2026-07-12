//! ST5 first-token parity — trace interchange format.
//!
//! A trace is a directory containing a `manifest.json` plus one raw
//! little-endian `f32` file per captured tensor. Both the LARQL F32 forward
//! path and the Transformers CPU-float32 oracle write the same format, so
//! [`crate::parity::compare`] can diff any pair of traces without knowing
//! which side produced them.
//!
//! Each tensor in the manifest binds `(prompt_id, stage, layer)` to a shape,
//! dtype, element count, filename, and SHA-256 of the file bytes. The
//! comparator rejects missing stages, duplicate identities, shape/dtype
//! mismatches, truncated files, non-finite values, and unknown required
//! stages — see [`crate::parity::compare`].

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Canonical coarse stage names captured at the last token position.
pub const STAGE_EMBEDDING: &str = "embedding";
pub const STAGE_LAYER_INPUT: &str = "layer_input";
pub const STAGE_POST_ATTENTION: &str = "post_attention";
pub const STAGE_POST_FFN: &str = "post_ffn";
pub const STAGE_POST_PLE: &str = "post_ple";
pub const STAGE_POST_LAYER: &str = "post_layer";
pub const STAGE_PRE_FINAL_NORM: &str = "pre_final_norm";
pub const STAGE_FINAL_NORM: &str = "final_norm";
pub const STAGE_LM_HEAD_RAW: &str = "lm_head_raw";
pub const STAGE_FINAL_LOGITS: &str = "final_logits";

/// One captured tensor entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceTensor {
    pub stage: String,
    /// `None` for non-layer stages (embedding, final-norm, logits).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub element_count: usize,
    pub filename: String,
    pub sha256: String,
    /// Set to `"true"` when this consumer stage's K/V projections were not
    /// executed (shared-KV consumer layers). The comparator treats a pair
    /// of matching `not_executed` entries as equal without reading files.
    #[serde(default, skip_serializing_if = "is_false")]
    pub not_executed: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Per-prompt manifest section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracePrompt {
    pub token_ids: Vec<u32>,
    pub seq_len: usize,
    pub tensors: Vec<TraceTensor>,
}

/// Top-level trace manifest written to `<dir>/manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceManifest {
    pub schema_version: u32,
    pub producer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<serde_json::Value>,
    pub prompts: BTreeMap<String, TracePrompt>,
}

impl TraceManifest {
    pub fn load(dir: &Path) -> Result<Self, TraceError> {
        let path = dir.join("manifest.json");
        let bytes = fs::read(&path)
            .map_err(|e| TraceError::Io(path.display().to_string(), e.to_string()))?;
        let manifest: TraceManifest = serde_json::from_slice(&bytes)
            .map_err(|e| TraceError::Parse(path.display().to_string(), e.to_string()))?;
        if manifest.schema_version != 1 {
            return Err(TraceError::Schema(manifest.schema_version));
        }
        Ok(manifest)
    }

    pub fn write(&self, dir: &Path) -> Result<(), TraceError> {
        fs::create_dir_all(dir)
            .map_err(|e| TraceError::Io(dir.display().to_string(), e.to_string()))?;
        let json =
            serde_json::to_string_pretty(self).map_err(|e| TraceError::Encode(e.to_string()))?;
        let path = dir.join("manifest.json");
        fs::write(&path, json + "\n")
            .map_err(|e| TraceError::Io(path.display().to_string(), e.to_string()))?;
        Ok(())
    }
}

/// Errors raised while reading or writing a trace.
#[derive(Debug, Clone)]
pub enum TraceError {
    Io(String, String),
    Parse(String, String),
    Encode(String),
    Schema(u32),
    MissingStage(String, Option<usize>, String),
    DuplicateStage(String, Option<usize>),
    ShapeMismatch {
        prompt: String,
        stage: String,
        layer: Option<usize>,
        reference: Vec<usize>,
        candidate: Vec<usize>,
    },
    DtypeMismatch {
        prompt: String,
        stage: String,
        layer: Option<usize>,
        reference: String,
        candidate: String,
    },
    Truncated {
        prompt: String,
        stage: String,
        layer: Option<usize>,
        expected: usize,
        actual: usize,
    },
    HashMismatch {
        prompt: String,
        stage: String,
        layer: Option<usize>,
        expected: String,
        actual: String,
    },
    NonFinite {
        prompt: String,
        stage: String,
        layer: Option<usize>,
        side: &'static str,
    },
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(p, e) => write!(f, "io error on {p}: {e}"),
            Self::Parse(p, e) => write!(f, "parse error in {p}: {e}"),
            Self::Encode(e) => write!(f, "encode error: {e}"),
            Self::Schema(v) => write!(f, "unsupported schema_version {v}"),
            Self::MissingStage(p, l, s) => {
                let layer = fmt_layer(*l);
                write!(f, "missing required stage `{s}`{layer} in {p}")
            }
            Self::DuplicateStage(s, l) => {
                let layer = fmt_layer(*l);
                write!(f, "duplicate stage identity `{s}`{layer}")
            }
            Self::ShapeMismatch {
                prompt,
                stage,
                layer,
                reference,
                candidate,
            } => {
                let l = fmt_layer(*layer);
                write!(
                    f,
                    "shape mismatch for `{stage}`{l} in {prompt}: {reference:?} vs {candidate:?}"
                )
            }
            Self::DtypeMismatch {
                prompt,
                stage,
                layer,
                reference,
                candidate,
            } => {
                let l = fmt_layer(*layer);
                write!(
                    f,
                    "dtype mismatch for `{stage}`{l} in {prompt}: {reference} vs {candidate}"
                )
            }
            Self::Truncated {
                prompt,
                stage,
                layer,
                expected,
                actual,
            } => {
                let l = fmt_layer(*layer);
                write!(
                    f,
                    "truncated file for `{stage}`{l} in {prompt}: {actual}/{expected} elements"
                )
            }
            Self::HashMismatch {
                prompt,
                stage,
                layer,
                expected,
                actual,
            } => {
                let l = fmt_layer(*layer);
                write!(
                    f,
                    "hash mismatch for `{stage}`{l} in {prompt}: {expected} vs {actual}"
                )
            }
            Self::NonFinite {
                prompt,
                stage,
                layer,
                side,
            } => {
                let l = fmt_layer(*layer);
                write!(f, "non-finite value in `{stage}`{l} ({side}) of {prompt}")
            }
        }
    }
}

impl std::error::Error for TraceError {}

fn fmt_layer(l: Option<usize>) -> String {
    match l {
        Some(layer) => format!(" (layer {layer})"),
        None => String::new(),
    }
}

/// Identifier for a stage, used as a map key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StageId {
    pub stage: String,
    pub layer: Option<usize>,
}

impl StageId {
    pub fn new(stage: &str, layer: Option<usize>) -> Self {
        Self {
            stage: stage.to_string(),
            layer,
        }
    }
}

impl std::fmt::Display for StageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.layer {
            Some(l) => write!(f, "{}@{}", self.stage, l),
            None => write!(f, "{}", self.stage),
        }
    }
}

/// The ordered list of coarse stages for one prompt. The comparator walks
/// stages in this order so "first divergence" is reported in execution
/// order, not alphabetic.
pub fn coarse_stage_order(num_layers: usize) -> Vec<StageId> {
    let mut order = Vec::with_capacity(5 * num_layers + 5);
    order.push(StageId::new(STAGE_EMBEDDING, None));
    for layer in 0..num_layers {
        order.push(StageId::new(STAGE_LAYER_INPUT, Some(layer)));
        order.push(StageId::new(STAGE_POST_ATTENTION, Some(layer)));
        order.push(StageId::new(STAGE_POST_FFN, Some(layer)));
        order.push(StageId::new(STAGE_POST_PLE, Some(layer)));
        order.push(StageId::new(STAGE_POST_LAYER, Some(layer)));
    }
    order.push(StageId::new(STAGE_PRE_FINAL_NORM, None));
    order.push(StageId::new(STAGE_FINAL_NORM, None));
    order.push(StageId::new(STAGE_LM_HEAD_RAW, None));
    order.push(StageId::new(STAGE_FINAL_LOGITS, None));
    order
}

/// Required coarse stage set (for missing-stage detection).
pub fn required_coarse_stages(num_layers: usize) -> HashSet<StageId> {
    coarse_stage_order(num_layers).into_iter().collect()
}

/// Write a little-endian f32 tensor file and return its SHA-256.
pub fn write_tensor(dir: &Path, rel_filename: &str, values: &[f32]) -> Result<String, TraceError> {
    let path = dir.join(rel_filename);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| TraceError::Io(parent.display().to_string(), e.to_string()))?;
    }
    let mut hasher = Sha256::new();
    let mut file = fs::File::create(&path)
        .map_err(|e| TraceError::Io(path.display().to_string(), e.to_string()))?;
    let mut buf = [0u8; 4];
    for &v in values {
        hasher.update(v.to_le_bytes());
        buf.copy_from_slice(&v.to_le_bytes());
        file.write_all(&buf)
            .map_err(|e| TraceError::Io(path.display().to_string(), e.to_string()))?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Read a little-endian f32 tensor file, verifying the byte count matches
/// `element_count` and the SHA-256 matches the manifest.
pub fn read_tensor(
    dir: &Path,
    entry: &TraceTensor,
    verify_hash: bool,
) -> Result<Vec<f32>, TraceError> {
    let path = dir.join(&entry.filename);
    let bytes =
        fs::read(&path).map_err(|e| TraceError::Io(path.display().to_string(), e.to_string()))?;
    if verify_hash {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let actual = format!("{:x}", hasher.finalize());
        if actual != entry.sha256 {
            return Err(TraceError::HashMismatch {
                prompt: String::new(),
                stage: entry.stage.clone(),
                layer: entry.layer,
                expected: entry.sha256.clone(),
                actual,
            });
        }
    }
    let expected_bytes = entry.element_count * 4;
    if bytes.len() != expected_bytes {
        return Err(TraceError::Truncated {
            prompt: String::new(),
            stage: entry.stage.clone(),
            layer: entry.layer,
            expected: expected_bytes,
            actual: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(entry.element_count);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Load every tensor for a prompt into an ordered `Vec<(StageId, values)>`,
/// rejecting duplicate stage identities and non-finite values. `order` is
/// the canonical stage order; stages not present in the manifest are simply
/// omitted (missing-stage detection happens in the comparator).
pub fn load_prompt_tensors(
    dir: &Path,
    prompt: &TracePrompt,
    order: &[StageId],
) -> Result<Vec<(StageId, Vec<f32>)>, TraceError> {
    let mut by_id: BTreeMap<StageId, &TraceTensor> = BTreeMap::new();
    for t in &prompt.tensors {
        let id = StageId::new(&t.stage, t.layer);
        if by_id.insert(id.clone(), t).is_some() {
            return Err(TraceError::DuplicateStage(t.stage.clone(), t.layer));
        }
    }
    let mut out = Vec::with_capacity(order.len());
    for id in order {
        if let Some(t) = by_id.get(id) {
            if t.not_executed {
                out.push((id.clone(), Vec::new()));
                continue;
            }
            let values = read_tensor(dir, t, /*verify_hash=*/ true)?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(TraceError::NonFinite {
                    prompt: String::new(),
                    stage: id.stage.clone(),
                    layer: id.layer,
                    side: "unknown",
                });
            }
            out.push((id.clone(), values));
        }
    }
    Ok(out)
}

/// Write a tensor at an explicit relative path (relative to `trace_root`)
/// and return its manifest entry. Use this when the file lives under a
/// prompt subdir, e.g. `rel_filename = "raw_completion/embedding.f32"`.
pub fn entry_at(
    trace_root: &Path,
    rel_filename: &str,
    stage: &str,
    layer: Option<usize>,
    values: &[f32],
) -> Result<(TraceTensor, PathBuf), TraceError> {
    let sha = write_tensor(trace_root, rel_filename, values)?;
    let entry = TraceTensor {
        stage: stage.to_string(),
        layer,
        shape: vec![values.len()],
        dtype: "f32".to_string(),
        element_count: values.len(),
        filename: rel_filename.to_string(),
        sha256: sha,
        not_executed: false,
    };
    Ok((entry, trace_root.join(rel_filename)))
}

/// Helper to build a tensor entry from in-memory values, with a flat
/// `<stage>[_<layer>].f32` filename relative to `dir` (no prompt subdir).
pub fn entry_from(
    stage: &str,
    layer: Option<usize>,
    values: &[f32],
    dir: &Path,
) -> Result<(TraceTensor, PathBuf), TraceError> {
    let layer_tag = layer.map(|l| format!("_{l}")).unwrap_or_default();
    let rel = format!("{}{}.f32", stage, layer_tag);
    entry_at(dir, &rel, stage, layer, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_order_is_in_execution_sequence() {
        let order = coarse_stage_order(2);
        // embedding, then per-layer blocks, then tail.
        assert_eq!(order[0].stage, STAGE_EMBEDDING);
        assert_eq!(order[1], StageId::new(STAGE_LAYER_INPUT, Some(0)));
        assert_eq!(order[5], StageId::new(STAGE_POST_LAYER, Some(0)));
        assert_eq!(order[6], StageId::new(STAGE_LAYER_INPUT, Some(1)));
        assert_eq!(order.last().unwrap().stage, STAGE_FINAL_LOGITS);
        // 1 embed + 5*2 layers + 4 tail = 15
        assert_eq!(order.len(), 1 + 5 * 2 + 4);
    }

    #[test]
    fn stage_id_orders_layer_before_none_consistently() {
        let a = StageId::new(STAGE_POST_LAYER, Some(0));
        let b = StageId::new(STAGE_POST_LAYER, Some(1));
        let c = StageId::new(STAGE_FINAL_NORM, None);
        assert!(a < b);
        // Sorting is stable regardless of None placement; just check it's total.
        let _ = a.cmp(&c);
    }

    #[test]
    fn write_and_read_tensor_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let values = vec![1.0f32, -2.5, 3.25, 0.0];
        let (entry, _path) = entry_from("test", None, &values, dir.path()).unwrap();
        let back = read_tensor(dir.path(), &entry, true).unwrap();
        assert_eq!(back, values);
    }

    #[test]
    fn read_tensor_detects_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let values = vec![1.0f32, 2.0, 3.0];
        let (mut entry, _) = entry_from("test", None, &values, dir.path()).unwrap();
        entry.element_count = 10; // lie about the size
        let err = read_tensor(dir.path(), &entry, false).unwrap_err();
        assert!(matches!(err, TraceError::Truncated { .. }));
    }

    #[test]
    fn read_tensor_detects_hash_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let values = vec![1.0f32, 2.0];
        let (mut entry, _) = entry_from("test", None, &values, dir.path()).unwrap();
        entry.sha256 = "deadbeef".to_string();
        let err = read_tensor(dir.path(), &entry, true).unwrap_err();
        assert!(matches!(err, TraceError::HashMismatch { .. }));
    }
}

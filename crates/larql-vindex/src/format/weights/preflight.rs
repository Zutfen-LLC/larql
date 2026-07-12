//! Metadata-only safetensors contract validation.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VindexError;

use super::WeightSource;

/// Whether automatic extraction must enforce this slice's Gemma 4 E2B contract.
pub(crate) fn is_gemma4_e2b(arch: &dyn larql_models::ModelArchitecture) -> bool {
    arch.family() == "gemma4" && arch.config().num_layers == 35
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorMetadata {
    pub normalized_name: String,
    pub source_name: String,
    pub shape: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetensorsPreflightOptions {
    /// `Some(true)` permits an omitted lm_head and requires embedding geometry.
    /// `Some(false)` requires a separate lm_head. `None` is conservative and
    /// requires a separate head; tying is never inferred from absence alone.
    pub tied_embeddings: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShapeMismatch {
    pub name: String,
    pub expected: Vec<usize>,
    pub actual: Vec<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetensorsPreflightReport {
    pub required: Vec<String>,
    pub missing: Vec<String>,
    pub shape_mismatches: Vec<ShapeMismatch>,
    pub duplicates: BTreeMap<String, Vec<String>>,
    pub unknown: Vec<String>,
    pub tied_evidence: Vec<String>,
    pub classification_counts: BTreeMap<String, usize>,
}

impl SafetensorsPreflightReport {
    pub fn is_valid(&self) -> bool {
        self.missing.is_empty()
            && self.shape_mismatches.is_empty()
            && self.duplicates.is_empty()
            && self.unknown.is_empty()
    }

    pub fn diagnostic(&self) -> String {
        let mut out = String::from("safetensors extraction preflight failed");
        if !self.missing.is_empty() {
            let _ = write!(out, "; missing: {}", self.missing.join(", "));
        }
        if !self.shape_mismatches.is_empty() {
            let values = self
                .shape_mismatches
                .iter()
                .map(|m| format!("{} expected {:?}, got {:?}", m.name, m.expected, m.actual))
                .collect::<Vec<_>>();
            let _ = write!(out, "; shape mismatch: {}", values.join(", "));
        }
        if !self.duplicates.is_empty() {
            let values = self
                .duplicates
                .iter()
                .map(|(k, v)| format!("{k} <- {}", v.join(", ")))
                .collect::<Vec<_>>();
            let _ = write!(out, "; normalized duplicates: {}", values.join("; "));
        }
        if !self.unknown.is_empty() {
            let _ = write!(
                out,
                "; unknown decoder tensors: {}",
                self.unknown.join(", ")
            );
        }
        out
    }
}

fn excluded(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    [
        "vision",
        "visual",
        "audio",
        "projector",
        "multi_modal",
        "multimodal",
        "mtp",
        "draft",
    ]
    .iter()
    .any(|part| n.contains(part))
}

fn add(expected: &mut BTreeMap<String, Vec<usize>>, key: String, shape: Vec<usize>) {
    expected.insert(key, shape);
}

fn expected_tensors(
    arch: &dyn larql_models::ModelArchitecture,
    vocab_size: usize,
) -> BTreeMap<String, Vec<usize>> {
    let cfg = arch.config();
    let h = cfg.hidden_size;
    let mut e = BTreeMap::new();
    add(&mut e, arch.embed_key().into(), vec![vocab_size, h]);
    add(&mut e, arch.final_norm_key().into(), vec![h]);
    if let Some(key) = arch.per_layer_embed_key() {
        add(
            &mut e,
            key,
            vec![vocab_size, cfg.num_layers * arch.per_layer_embed_dim()],
        );
    }
    if let Some(key) = arch.per_layer_model_projection_key() {
        add(
            &mut e,
            key,
            vec![cfg.num_layers * arch.per_layer_embed_dim(), h],
        );
    }
    if let Some(key) = arch.per_layer_projection_norm_key() {
        add(&mut e, key, vec![arch.per_layer_embed_dim()]);
    }
    for layer in 0..cfg.num_layers {
        let hd = arch.head_dim_for_layer(layer);
        let q = hd * arch.num_q_heads_for_layer(layer);
        let kv = hd * arch.num_kv_heads_for_layer(layer);
        add(&mut e, arch.attn_q_key(layer), vec![q, h]);
        // Runtime KV reuse does not imply that the official source omits the
        // redundant projection tensors in its shared region.
        add(&mut e, arch.attn_k_key(layer), vec![kv, h]);
        if !arch.v_shares_k(layer) {
            add(&mut e, arch.attn_v_key(layer), vec![kv, h]);
        }
        add(&mut e, arch.attn_o_key(layer), vec![h, q]);
        add(&mut e, arch.input_layernorm_key(layer), vec![h]);
        if arch.has_post_norms() {
            add(&mut e, arch.post_attention_layernorm_key(layer), vec![h]);
            if let Some(key) = arch.pre_feedforward_layernorm_key(layer) {
                add(&mut e, key, vec![h]);
            }
            if let Some(key) = arch.post_feedforward_layernorm_key(layer) {
                add(&mut e, key, vec![h]);
            }
        }
        if let Some(key) = arch.attn_q_norm_key(layer) {
            add(&mut e, key, vec![hd]);
        }
        if let Some(key) = arch.attn_k_norm_key(layer) {
            add(&mut e, key, vec![hd]);
        }
        let intermediate = arch.intermediate_size_for_layer(layer);
        add(&mut e, arch.ffn_gate_key(layer), vec![intermediate, h]);
        add(&mut e, arch.ffn_up_key(layer), vec![intermediate, h]);
        add(&mut e, arch.ffn_down_key(layer), vec![h, intermediate]);
        if let Some(key) = arch.layer_scalar_key(layer) {
            add(&mut e, key, vec![1]);
        }
        if let Some(key) = arch.per_layer_input_gate_key(layer) {
            add(&mut e, key, vec![arch.per_layer_embed_dim(), h]);
        }
        if let Some(key) = arch.per_layer_projection_key(layer) {
            add(&mut e, key, vec![h, arch.per_layer_embed_dim()]);
        }
        if let Some(key) = arch.post_per_layer_input_norm_key(layer) {
            add(&mut e, key, vec![h]);
        }
    }
    e
}

pub fn validate_weight_source(
    source: &dyn WeightSource,
    options: SafetensorsPreflightOptions,
) -> SafetensorsPreflightReport {
    validate_metadata(
        source.arch(),
        source.tensor_metadata(),
        SafetensorsPreflightOptions {
            tied_embeddings: options.tied_embeddings.or(source.tied_embeddings()),
        },
    )
}

pub fn validate_metadata(
    arch: &dyn larql_models::ModelArchitecture,
    metadata: Vec<TensorMetadata>,
    options: SafetensorsPreflightOptions,
) -> SafetensorsPreflightReport {
    let mut report = SafetensorsPreflightReport::default();
    if metadata.is_empty() {
        return report;
    }
    let mut actual: HashMap<String, &TensorMetadata> = HashMap::new();
    let mut names: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for item in &metadata {
        names
            .entry(item.normalized_name.clone())
            .or_default()
            .push(item.source_name.clone());
        actual.entry(item.normalized_name.clone()).or_insert(item);
    }
    report.duplicates = names.into_iter().filter(|(_, v)| v.len() > 1).collect();

    let vocab_size = arch
        .config()
        .vocab_size
        .or_else(|| {
            metadata
                .iter()
                .find(|m| m.normalized_name == arch.embed_key())
                .and_then(|m| m.shape.first().copied())
        })
        .unwrap_or(0);
    let expected = expected_tensors(arch, vocab_size);
    let lm_names = ["lm_head.weight", "output.weight"];
    let lm = lm_names.iter().find_map(|name| actual.get(*name).copied());
    let tied = options.tied_embeddings.unwrap_or(false);
    if tied {
        report
            .tied_evidence
            .push("configuration permits embed/lm_head tying".into());
        if lm.is_none() {
            report
                .tied_evidence
                .push("lm_head omitted; embedding is the output projection".into());
        }
    } else {
        report.required.push("lm_head.weight|output.weight".into());
        if lm.is_none() {
            report.missing.push("lm_head.weight|output.weight".into());
        }
    }
    if let Some(head) = lm {
        let wanted = vec![vocab_size, arch.config().hidden_size];
        if head.shape != wanted {
            report.shape_mismatches.push(ShapeMismatch {
                name: head.normalized_name.clone(),
                expected: wanted,
                actual: head.shape.clone(),
            });
        } else if tied {
            report
                .tied_evidence
                .push("lm_head geometry matches embedding geometry".into());
        }
    }

    report.required.extend(expected.keys().cloned());
    for (name, shape) in &expected {
        match actual.get(name) {
            None => report.missing.push(name.clone()),
            Some(item) if item.shape != *shape => report.shape_mismatches.push(ShapeMismatch {
                name: name.clone(),
                expected: shape.clone(),
                actual: item.shape.clone(),
            }),
            _ => {}
        }
    }
    for item in &metadata {
        if expected.contains_key(&item.normalized_name)
            || lm_names.contains(&item.normalized_name.as_str())
        {
            *report
                .classification_counts
                .entry("REQUIRED".into())
                .or_default() += 1;
        } else if excluded(&item.source_name) {
            let class = if item.source_name.to_ascii_lowercase().contains("mtp") {
                "EXCLUDED_MTP"
            } else {
                "EXCLUDED_MULTIMODAL"
            };
            *report
                .classification_counts
                .entry(class.into())
                .or_default() += 1;
        } else {
            report.unknown.push(item.normalized_name.clone());
            *report
                .classification_counts
                .entry("UNKNOWN".into())
                .or_default() += 1;
        }
    }
    report.missing.sort();
    report.unknown.sort();
    report.unknown.dedup();
    report
}

pub fn audit_safetensors_preflight(
    model_dir: &Path,
    options: SafetensorsPreflightOptions,
) -> Result<SafetensorsPreflightReport, VindexError> {
    let arch = larql_models::detect_architecture_validated(model_dir)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)
            .map_err(|e| VindexError::Parse(format!("config.json: {e}")))?;
    let text_config = config.get("text_config").unwrap_or(&config);
    let options = SafetensorsPreflightOptions {
        tied_embeddings: options.tied_embeddings.or_else(|| {
            text_config
                .get("tie_word_embeddings")
                .and_then(serde_json::Value::as_bool)
                .or_else(|| {
                    config
                        .get("tie_word_embeddings")
                        .and_then(serde_json::Value::as_bool)
                })
        }),
    };
    let prefixes = arch.key_prefixes_to_strip();
    let mut files = std::fs::read_dir(model_dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect::<Vec<_>>();
    if files.is_empty() {
        let weights = model_dir.join("weights");
        if weights.is_dir() {
            files = std::fs::read_dir(weights)?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
        }
    }
    files.sort();
    let mut metadata = Vec::new();
    for path in files {
        let bytes = std::fs::read(&path)?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| VindexError::Parse(format!("{}: {e}", path.display())))?;
        for name in st.names() {
            let normalized_name = prefixes
                .iter()
                .find_map(|p| name.strip_prefix(p))
                .unwrap_or(name)
                .to_string();
            let view = st
                .tensor(name)
                .map_err(|e| VindexError::Parse(e.to_string()))?;
            metadata.push(TensorMetadata {
                normalized_name,
                source_name: name.into(),
                shape: view.shape().to_vec(),
            });
        }
    }
    Ok(validate_metadata(&*arch, metadata, options))
}

#[cfg(test)]
mod tests {
    use std::io::{BufWriter, Write};

    use super::*;

    const VOCAB: usize = 64;
    const HIDDEN: usize = 16;
    const INTERMEDIATE: usize = 32;
    const PLE_DIM: usize = 4;

    fn arch() -> Box<dyn larql_models::ModelArchitecture> {
        let config = serde_json::json!({
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": HIDDEN,
                "intermediate_size": INTERMEDIATE,
                "num_hidden_layers": 35,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "global_head_dim": 8,
                "vocab_size": VOCAB,
                "hidden_size_per_layer_input": PLE_DIM,
                "num_kv_shared_layers": 20,
                "use_double_wide_mlp": true,
                "layer_types": [
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "full_attention",
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "full_attention",
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "full_attention",
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "full_attention",
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "full_attention",
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "full_attention",
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "full_attention"
                ]
            }
        });
        larql_models::detect_from_json_validated(&config).unwrap()
    }

    fn complete(arch: &dyn larql_models::ModelArchitecture) -> Vec<TensorMetadata> {
        expected_tensors(arch, VOCAB)
            .into_iter()
            .map(|(name, shape)| TensorMetadata {
                normalized_name: name.clone(),
                source_name: format!("model.language_model.{name}"),
                shape,
            })
            .collect()
    }

    fn tied() -> SafetensorsPreflightOptions {
        SafetensorsPreflightOptions {
            tied_embeddings: Some(true),
        }
    }

    fn item(name: &str, shape: Vec<usize>) -> TensorMetadata {
        TensorMetadata {
            normalized_name: name.into(),
            source_name: name.into(),
            shape,
        }
    }

    #[test]
    fn scenario_01_empty_metadata_is_a_noop() {
        let report = validate_metadata(&*arch(), Vec::new(), tied());
        assert!(report.is_valid());
        assert!(report.required.is_empty());
    }

    #[test]
    fn scenario_02_complete_tied_gemma4_e2b_inventory_is_valid() {
        let arch = arch();
        let report = validate_metadata(&*arch, complete(&*arch), tied());
        assert!(report.is_valid(), "{}", report.diagnostic());
        assert_eq!(
            report.classification_counts["REQUIRED"],
            report.required.len()
        );
    }

    #[test]
    fn scenario_03_local_attention_uses_local_head_geometry() {
        let arch = arch();
        let expected = expected_tensors(&*arch, VOCAB);
        assert_eq!(expected[&arch.attn_q_key(0)], vec![8, HIDDEN]);
        assert_eq!(expected[&arch.attn_o_key(0)], vec![HIDDEN, 8]);
    }

    #[test]
    fn scenario_04_global_attention_uses_global_head_geometry() {
        let arch = arch();
        let expected = expected_tensors(&*arch, VOCAB);
        assert_eq!(expected[&arch.attn_q_key(4)], vec![16, HIDDEN]);
        assert_eq!(expected[&arch.attn_o_key(4)], vec![HIDDEN, 16]);
    }

    #[test]
    fn scenario_05_kv_shared_layers_retain_source_kv_tensors() {
        let arch = arch();
        let expected = expected_tensors(&*arch, VOCAB);
        let reused = (0..35)
            .find(|&layer| arch.kv_shared_source_layer(layer).is_some())
            .unwrap();
        assert!(expected.contains_key(&arch.attn_k_key(reused)));
        assert!(expected.contains_key(&arch.attn_v_key(reused)));
    }

    #[test]
    fn scenario_06_nonshared_kv_respects_v_shares_k_contract() {
        let arch = arch();
        let expected = expected_tensors(&*arch, VOCAB);
        let layer = (0..35)
            .find(|&layer| arch.kv_shared_source_layer(layer).is_none())
            .unwrap();
        assert!(expected.contains_key(&arch.attn_k_key(layer)));
        assert_eq!(
            expected.contains_key(&arch.attn_v_key(layer)),
            !arch.v_shares_k(layer)
        );
    }

    #[test]
    fn scenario_06b_double_wide_mlp_starts_at_shared_region() {
        let arch = arch();
        let expected = expected_tensors(&*arch, VOCAB);
        assert_eq!(expected[&arch.ffn_gate_key(14)], vec![INTERMEDIATE, HIDDEN]);
        assert_eq!(expected[&arch.ffn_down_key(14)], vec![HIDDEN, INTERMEDIATE]);
        assert_eq!(
            expected[&arch.ffn_gate_key(15)],
            vec![INTERMEDIATE * 2, HIDDEN]
        );
        assert_eq!(
            expected[&arch.ffn_down_key(34)],
            vec![HIDDEN, INTERMEDIATE * 2]
        );
    }

    #[test]
    fn scenario_07_global_ple_tensors_have_35_layer_geometry() {
        let arch = arch();
        let expected = expected_tensors(&*arch, VOCAB);
        assert_eq!(
            expected[&arch.per_layer_embed_key().unwrap()],
            vec![VOCAB, 35 * PLE_DIM]
        );
        assert_eq!(
            expected[&arch.per_layer_model_projection_key().unwrap()],
            vec![35 * PLE_DIM, HIDDEN]
        );
        assert_eq!(
            expected[&arch.per_layer_projection_norm_key().unwrap()],
            vec![PLE_DIM]
        );
    }

    #[test]
    fn scenario_08_per_layer_ple_tensors_have_gate_projection_and_norm_geometry() {
        let arch = arch();
        let expected = expected_tensors(&*arch, VOCAB);
        assert_eq!(
            expected[&arch.per_layer_input_gate_key(34).unwrap()],
            vec![PLE_DIM, HIDDEN]
        );
        assert_eq!(
            expected[&arch.per_layer_projection_key(34).unwrap()],
            vec![HIDDEN, PLE_DIM]
        );
        assert_eq!(
            expected[&arch.post_per_layer_input_norm_key(34).unwrap()],
            vec![HIDDEN]
        );
    }

    #[test]
    fn scenario_09_missing_required_tensor_is_reported() {
        let arch = arch();
        let mut items = complete(&*arch);
        let missing = items.remove(0).normalized_name;
        let report = validate_metadata(&*arch, items, tied());
        assert_eq!(report.missing, [missing]);
    }

    #[test]
    fn scenario_10_required_tensor_shape_mismatch_is_reported() {
        let arch = arch();
        let mut items = complete(&*arch);
        let name = items[0].normalized_name.clone();
        items[0].shape = vec![999];
        let report = validate_metadata(&*arch, items, tied());
        assert_eq!(report.shape_mismatches[0].name, name);
    }

    #[test]
    fn scenario_11_normalized_duplicate_retains_source_provenance() {
        let arch = arch();
        let mut items = complete(&*arch);
        let mut duplicate = items[0].clone();
        duplicate.source_name = format!("alternate.{}", duplicate.normalized_name);
        let key = duplicate.normalized_name.clone();
        items.push(duplicate);
        let report = validate_metadata(&*arch, items, tied());
        assert_eq!(report.duplicates[&key].len(), 2);
        assert!(report.diagnostic().contains("normalized duplicates"));
    }

    #[test]
    fn scenario_12_unknown_decoder_tensor_is_rejected() {
        let arch = arch();
        let mut items = complete(&*arch);
        items.push(item("layers.0.mystery.weight", vec![1]));
        let report = validate_metadata(&*arch, items, tied());
        assert_eq!(report.unknown, ["layers.0.mystery.weight"]);
        assert_eq!(report.classification_counts["UNKNOWN"], 1);
    }

    #[test]
    fn scenario_13_multimodal_tensor_is_excluded_not_unknown() {
        let arch = arch();
        let mut items = complete(&*arch);
        items.push(item("vision_tower.weight", vec![1]));
        let report = validate_metadata(&*arch, items, tied());
        assert!(report.unknown.is_empty());
        assert_eq!(report.classification_counts["EXCLUDED_MULTIMODAL"], 1);
    }

    #[test]
    fn scenario_14_mtp_tensor_is_excluded_not_unknown() {
        let arch = arch();
        let mut items = complete(&*arch);
        items.push(item("mtp.weight", vec![1]));
        let report = validate_metadata(&*arch, items, tied());
        assert!(report.unknown.is_empty());
        assert_eq!(report.classification_counts["EXCLUDED_MTP"], 1);
    }

    #[test]
    fn scenario_15_tied_embeddings_permit_omitted_lm_head() {
        let arch = arch();
        let report = validate_metadata(&*arch, complete(&*arch), tied());
        assert!(report.is_valid());
        assert!(report.tied_evidence.iter().any(|s| s.contains("omitted")));
    }

    #[test]
    fn scenario_16_untied_embeddings_require_lm_head() {
        let arch = arch();
        let report = validate_metadata(
            &*arch,
            complete(&*arch),
            SafetensorsPreflightOptions {
                tied_embeddings: Some(false),
            },
        );
        assert_eq!(report.missing, ["lm_head.weight|output.weight"]);
    }

    #[test]
    fn scenario_17_tied_lm_head_with_matching_geometry_is_accepted() {
        let arch = arch();
        let mut items = complete(&*arch);
        items.push(item("lm_head.weight", vec![VOCAB, HIDDEN]));
        let report = validate_metadata(&*arch, items, tied());
        assert!(report.is_valid(), "{}", report.diagnostic());
        assert!(report.tied_evidence.iter().any(|s| s.contains("matches")));
    }

    #[test]
    fn scenario_18_lm_head_shape_mismatch_is_reported_once() {
        let arch = arch();
        let mut items = complete(&*arch);
        items.push(item("lm_head.weight", vec![VOCAB - 1, HIDDEN]));
        let report = validate_metadata(&*arch, items, tied());
        assert_eq!(
            report
                .shape_mismatches
                .iter()
                .filter(|m| m.name == "lm_head.weight")
                .count(),
            1
        );
    }

    #[test]
    fn scenario_19_output_weight_alias_satisfies_untied_head_requirement() {
        let arch = arch();
        let mut items = complete(&*arch);
        items.push(item("output.weight", vec![VOCAB, HIDDEN]));
        let report = validate_metadata(
            &*arch,
            items,
            SafetensorsPreflightOptions {
                tied_embeddings: Some(false),
            },
        );
        assert!(report.is_valid(), "{}", report.diagnostic());
    }

    #[test]
    fn scenario_20_report_artifact_round_trips_with_scoped_writer_guard() {
        let arch = arch();
        let report = validate_metadata(&*arch, complete(&*arch), tied());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("preflight-report.json");
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, &report).unwrap();
            writer.flush().unwrap();
        }
        let decoded: SafetensorsPreflightReport =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(decoded, report);
    }
}

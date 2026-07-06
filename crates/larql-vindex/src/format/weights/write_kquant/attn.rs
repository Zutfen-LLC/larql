//! Stage 1 — `attn_weights_q4k.bin` + manifest.

use std::io::{BufWriter, Write};
use std::path::Path;

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::extract::stage_labels::*;
use crate::format::filenames::*;

use super::super::manifest::Q4kManifestEntry;
use super::super::mla_absorb::{self, MlaGeometry};
use super::super::write_f32::WeightSource;
use super::{pad_rows_to_block, resolve_v_tensor, QuantBlockFormat};

/// Write Q/K/V/O attention projections to `attn_weights_q4k.bin`,
/// emitting a sidecar manifest with per-tensor offsets and formats.
///
/// Q/K/O are Q4_K; V is Q6_K. On layers where V reuses K (Gemma 4 31B
/// global layers), the K bytes go into the V slot so the 4-per-layer
/// indexing stays valid for downstream kernels reading V.
pub(super) fn write_attn_weights_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
    mla_geom: Option<&MlaGeometry>,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    let arch = source.arch();
    let attn_path = dir.join(ATTN_WEIGHTS_KQUANT_BIN);
    let mut attn_file = BufWriter::new(std::fs::File::create(&attn_path)?);
    let mut attn_offset: u64 = 0;
    let mut attn_manifest: Vec<Q4kManifestEntry> = Vec::with_capacity(num_layers * 4);

    for layer in 0..num_layers {
        callbacks.on_layer_start(COMP_ATTN_KQUANT, layer, num_layers);

        // ── MLA absorption path ──
        // When the architecture uses MLA and geometry is known, absorb the
        // four low-rank projections into dense Q/K/V (reusing mla_absorb.rs),
        // then quantise them exactly like standard attention. The O projection
        // is a standard linear and flows through unchanged.
        #[allow(clippy::type_complexity)]
        let mla_absorbed: Option<[(String, (Vec<f32>, usize, usize)); 3]> =
            if let Some(g) = mla_geom {
                let hidden = arch.config().hidden_size;
                let qk_hd = g.qk_nope + g.qk_rope;
                let kv_a_key = arch.mla_kv_a_key(layer).unwrap_or_default();
                let kv_b_key = arch.mla_kv_b_key(layer).unwrap_or_default();
                let q_a_key = arch.mla_q_a_key(layer).unwrap_or_default();
                let q_b_key = arch.mla_q_b_key(layer).unwrap_or_default();

                let kv_a_raw = source.get_tensor(&kv_a_key);
                let kv_b_raw = source.get_tensor(&kv_b_key);
                let q_a_raw = source.get_tensor(&q_a_key);
                let q_b_raw = source.get_tensor(&q_b_key);

                let (kv_a_d, _, _) = kv_a_raw.ok_or_else(|| VindexError::MissingTensor(kv_a_key.clone()))?;
                let (kv_b_d, _, _) = kv_b_raw.ok_or_else(|| VindexError::MissingTensor(kv_b_key.clone()))?;
                let (q_a_d, _, _) = q_a_raw.ok_or_else(|| VindexError::MissingTensor(q_a_key.clone()))?;
                let (q_b_d, _, _) = q_b_raw.ok_or_else(|| VindexError::MissingTensor(q_b_key.clone()))?;

                use ndarray::Array2;
                let kv_a = Array2::from_shape_vec((g.kv_lora + g.qk_rope, hidden), kv_a_d)
                    .expect("kv_a shape mismatch");
                let kv_b = Array2::from_shape_vec(
                    (g.num_kv * (g.qk_nope + g.v_hd), g.kv_lora),
                    kv_b_d,
                )
                    .expect("kv_b shape mismatch");
                let q_a = Array2::from_shape_vec((g.q_lora, hidden), q_a_d)
                    .expect("q_a shape mismatch");
                let q_b = Array2::from_shape_vec((g.num_q * qk_hd, g.q_lora), q_b_d)
                    .expect("q_b shape mismatch");

                let (w_q, w_k, w_v) = mla_absorb::absorb(&kv_a, &kv_b, &q_a, &q_b, g);
                Some([
                    (arch.attn_q_key(layer), (w_q.into_raw_vec_and_offset().0, g.num_q * qk_hd, hidden)),
                    (arch.attn_k_key(layer), (w_k.into_raw_vec_and_offset().0, g.num_kv * qk_hd, hidden)),
                    (arch.attn_v_key(layer), (w_v.into_raw_vec_and_offset().0, g.num_kv * g.v_hd, hidden)),
                ])
            } else {
                None
            };

        // Resolve each tensor. For V, fall back to K when v_shares_k=true or
        // v_proj simply isn't present (global layers on 31B).
        let q_key = arch.attn_q_key(layer);
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);

        // Build the slot list. When MLA was absorbed, Q/K/V come from the
        // absorbed tensors; otherwise from the source tensors as usual.
        #[allow(clippy::type_complexity)]
        let slots: Vec<(String, Option<(Vec<f32>, usize, usize)>)> = if let Some(absorbed) = mla_absorbed {
            let mut v: Vec<(String, Option<(Vec<f32>, usize, usize)>)> = absorbed
                .iter()
                .map(|(k, t)| (k.clone(), Some(t.clone())))
                .collect();
            // O projection — standard linear, no absorption.
            v.push((o_key.clone(), source.get_tensor(&o_key)));
            v
        } else {
            let q = source.get_tensor(&q_key);
            let k = source.get_tensor(&k_key);
            let v = resolve_v_tensor(source.get_tensor(&v_key), &k, arch.v_shares_k(layer));
            let o = source.get_tensor(&o_key);
            // Q, K, V, O in that order — use the same key string for V even when
            // the data is K's, so loaders that look up by position still work.
            vec![
                (q_key.clone(), q),
                (k_key.clone(), k),
                (v_key.clone(), v),
                (o_key.clone(), o),
            ]
        };

        for (i, (key, tensor)) in slots.iter().enumerate() {
            // V (index 2) gets Q6_K, others get Q4_K.
            let is_v = i == 2;
            let want_format = if is_v {
                QuantBlockFormat::Q6K
            } else {
                QuantBlockFormat::Q4K
            };

            // Quant-preserving fast path: GGUF Q4_K/Q6_K attention tensors
            // whose cols are a multiple of 256 emit verbatim. Attention
            // projections are usually hidden×hidden, so this hits for most
            // GGUF models and avoids f32 reification.
            if let Some(pq) = source.get_packed_quant(key) {
                if pq.format == want_format && pq.cols == pq.cols_padded {
                    let length = pq.bytes.len() as u64;
                    attn_file.write_all(&pq.bytes)?;
                    attn_manifest.push(Q4kManifestEntry {
                        key: key.to_string(),
                        shape: vec![pq.rows, pq.cols_padded],
                        format: want_format,
                        offset: attn_offset,
                        length,
                    });
                    attn_offset += length;
                    continue;
                }
            }

            let (data, rows, cols) = match tensor {
                Some(t) => t.clone(),
                None => continue, // tensor genuinely absent — skip
            };

            // Row-pad to 256 so each row aligns to a super-block boundary.
            // Critical for models with non-256 inner dims (e.g. Gemma 4 26B A4B
            // where the dense intermediate is 2112). `padded_cols` is what the
            // matvec shader must use as `K`; callers also need to zero-pad the
            // input vector to the same width.
            let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
            let q_bytes = if is_v {
                quantize_q6_k(&padded)
            } else {
                quantize_q4_k(&padded)
            };

            attn_file.write_all(&q_bytes)?;
            let length = q_bytes.len() as u64;
            attn_manifest.push(Q4kManifestEntry {
                key: key.to_string(),
                shape: vec![rows, padded_cols],
                format: want_format,
                offset: attn_offset,
                length,
            });
            attn_offset += length;
        }

        callbacks.on_layer_done(COMP_ATTN_KQUANT, layer, 0.0);
    }
    attn_file.flush()?;
    drop(attn_file);

    let manifest_json = serde_json::to_string_pretty(&attn_manifest)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(ATTN_WEIGHTS_KQUANT_MANIFEST_JSON), manifest_json)?;
    Ok(())
}

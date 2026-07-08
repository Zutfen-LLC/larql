//! Stage 1 — `attn_weights_q4k.bin` + manifest.

use std::io::{BufWriter, Write};
use std::path::Path;

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::extract::stage_labels::*;
use crate::format::filenames::*;

use super::super::manifest::Q4kManifestEntry;
use super::super::profile::Recorder;
use super::super::write_f32::WeightSource;
use super::parallel::transform_then_write;
use super::{pad_rows_to_block, resolve_v_tensor, QuantBlockFormat};

/// One quantised tensor slot (Q, K, V, or O) ready to be appended to
/// `attn_weights_q4k.bin`. Manifest offsets are assigned when this is
/// written, not when it's produced, so parallel transforms never race
/// on the running offset.
struct QuantSlot {
    key: String,
    bytes: Vec<u8>,
    shape: [usize; 2],
    format: QuantBlockFormat,
    component: &'static str,
}

/// Transformed Q/K/V/O for one layer, in slot order. `None` means the
/// tensor was genuinely absent from the source (unchanged behavior).
struct AttnLayerBlob {
    slots: [Option<QuantSlot>; 4],
}

/// Write Q/K/V/O attention projections to `attn_weights_q4k.bin`,
/// emitting a sidecar manifest with per-tensor offsets and formats.
///
/// Q/K/O are Q4_K; V is Q6_K. On layers where V reuses K (Gemma 4 31B
/// global layers), the K bytes go into the V slot so the 4-per-layer
/// indexing stays valid for downstream kernels reading V.
///
/// `jobs <= 1` transforms and writes each layer serially, identical to
/// the pre-IMPORT-002 code. `jobs > 1` transforms up to `jobs` layers'
/// Q/K/V/O concurrently (bounded, chunked — see
/// [`super::parallel::transform_then_write`]), but always writes bytes
/// and assigns manifest offsets on this thread in ascending layer order,
/// so output is byte-identical regardless of `jobs`.
pub(super) fn write_attn_weights_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
    jobs: usize,
    callbacks: &mut dyn IndexBuildCallbacks,
    rec: &Recorder<'_>,
) -> Result<(), VindexError> {
    let arch = source.arch();
    let attn_path = dir.join(ATTN_WEIGHTS_KQUANT_BIN);
    let mut attn_file = BufWriter::new(std::fs::File::create(&attn_path)?);
    let mut attn_offset: u64 = 0;
    let mut attn_manifest: Vec<Q4kManifestEntry> = Vec::with_capacity(num_layers * 4);

    transform_then_write(
        num_layers,
        jobs,
        |layer| transform_attn_layer(source, arch, layer, rec),
        |layer, blob| {
            callbacks.on_layer_start(COMP_ATTN_KQUANT, layer, num_layers);
            for slot in blob.slots.into_iter().flatten() {
                let t = rec.now();
                attn_file.write_all(&slot.bytes)?;
                rec.write(
                    t,
                    COMP_ATTN_KQUANT,
                    slot.component,
                    Some(layer),
                    slot.bytes.len() as u64,
                );
                let length = slot.bytes.len() as u64;
                attn_manifest.push(Q4kManifestEntry {
                    key: slot.key,
                    shape: slot.shape.to_vec(),
                    format: slot.format,
                    offset: attn_offset,
                    length,
                });
                attn_offset += length;
            }
            callbacks.on_layer_done(COMP_ATTN_KQUANT, layer, 0.0);
            Ok(())
        },
    )?;

    attn_file.flush()?;
    drop(attn_file);

    let t = rec.now();
    let manifest_json = serde_json::to_string_pretty(&attn_manifest)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(ATTN_WEIGHTS_KQUANT_MANIFEST_JSON), manifest_json)?;
    rec.manifest(t, COMP_ATTN_KQUANT, "attn_manifest");
    Ok(())
}

/// Fetch, pad, and quantise Q/K/V/O for one layer. Pure read-only access
/// to `source` — safe to call concurrently across layers since
/// `WeightSource: Sync` and every method returns owned data.
fn transform_attn_layer(
    source: &dyn WeightSource,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
    rec: &Recorder<'_>,
) -> Result<AttnLayerBlob, VindexError> {
    // Resolve each tensor. For V, fall back to K when v_shares_k=true or
    // v_proj simply isn't present (global layers on 31B).
    let q_key = arch.attn_q_key(layer);
    let k_key = arch.attn_k_key(layer);
    let v_key = arch.attn_v_key(layer);
    let o_key = arch.attn_o_key(layer);

    let t = rec.now();
    let q = source.get_tensor(&q_key);
    rec.fetch(
        t,
        COMP_ATTN_KQUANT,
        "Q",
        Some(layer),
        rows_of(&q),
        cols_of(&q),
    );

    let t = rec.now();
    let k = source.get_tensor(&k_key);
    rec.fetch(
        t,
        COMP_ATTN_KQUANT,
        "K",
        Some(layer),
        rows_of(&k),
        cols_of(&k),
    );

    let t = rec.now();
    let v_raw = source.get_tensor(&v_key);
    let (v_rows, v_cols) = (rows_of(&v_raw), cols_of(&v_raw));
    rec.fetch(t, COMP_ATTN_KQUANT, "V", Some(layer), v_rows, v_cols);
    let v = resolve_v_tensor(v_raw, &k, arch.v_shares_k(layer));

    let t = rec.now();
    let o = source.get_tensor(&o_key);
    rec.fetch(
        t,
        COMP_ATTN_KQUANT,
        "O",
        Some(layer),
        rows_of(&o),
        cols_of(&o),
    );

    // Q, K, V, O in that order — use the same key string for V even when
    // the data is K's, so loaders that look up by position still work.
    #[allow(clippy::type_complexity)]
    let raw_slots: [(String, Option<(Vec<f32>, usize, usize)>); 4] =
        [(q_key, q), (k_key, k), (v_key, v), (o_key, o)];

    let mut slots: [Option<QuantSlot>; 4] = [None, None, None, None];
    for (i, (key, tensor)) in raw_slots.into_iter().enumerate() {
        let (data, rows, cols) = match tensor {
            Some(t) => t,
            None => continue, // tensor genuinely absent — skip
        };

        // V (index 2) gets Q6_K, others get Q4_K.
        let is_v = i == 2;
        let component = ["Q", "K", "V", "O"][i];
        // Row-pad to 256 so each row aligns to a super-block boundary.
        // Critical for models with non-256 inner dims (e.g. Gemma 4 26B A4B
        // where the dense intermediate is 2112). `padded_cols` is what the
        // matvec shader must use as `K`; callers also need to zero-pad the
        // input vector to the same width.
        let t = rec.now();
        let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
        rec.pad(
            t,
            COMP_ATTN_KQUANT,
            component,
            Some(layer),
            cols as u64,
            padded_cols as u64,
        );
        let in_elems = (rows * padded_cols) as u64;
        let t = rec.now();
        let q_bytes = if is_v {
            quantize_q6_k(&padded)
        } else {
            quantize_q4_k(&padded)
        };
        let format = if is_v {
            QuantBlockFormat::Q6K
        } else {
            QuantBlockFormat::Q4K
        };
        rec.quantize(
            t,
            COMP_ATTN_KQUANT,
            component,
            Some(layer),
            format.tag(),
            in_elems,
            q_bytes.len() as u64,
        );

        slots[i] = Some(QuantSlot {
            key,
            bytes: q_bytes,
            shape: [rows, padded_cols],
            format,
            component,
        });
    }

    Ok(AttnLayerBlob { slots })
}

fn rows_of(t: &Option<(Vec<f32>, usize, usize)>) -> u64 {
    t.as_ref().map(|(_, r, _)| *r as u64).unwrap_or(0)
}

fn cols_of(t: &Option<(Vec<f32>, usize, usize)>) -> u64 {
    t.as_ref().map(|(_, _, c)| *c as u64).unwrap_or(0)
}

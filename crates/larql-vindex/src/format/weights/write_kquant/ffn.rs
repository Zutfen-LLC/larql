//! Stage 2 — `interleaved_kquant.bin` + manifest, plus opt-in
//! `down_features_q4k.bin` (W2 feature-major down).

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
use super::feature_major_down::FeatureMajorDownState;
use super::parallel::transform_then_write;
use super::{pad_rows_to_block, KquantWriteOptions, QuantBlockFormat};

/// One quantised tensor slot (gate/up/down) ready to be appended to
/// `interleaved_kquant.bin`. `down_padded_f32` carries the padded (but
/// not yet quantised) down data so the feature-major-down sidecar can
/// be appended on the ordered-write thread — that transpose/append
/// stays serial by design even when the primary gate/up/down transforms
/// run in parallel (see module docs on `KquantWriteOptions::feature_major_down`).
struct QuantSlot {
    key: String,
    bytes: Vec<u8>,
    shape: [usize; 2],
    format: QuantBlockFormat,
}

struct FfnLayerBlob {
    // gate, up, down — `None` when the tensor was absent from the source.
    slots: [Option<QuantSlot>; 3],
    /// Down's padded f32 data (pre-quantisation), only populated when
    /// `feature_major_down` is enabled — `state.append_layer` needs the
    /// f32 values to transpose, not the quantised bytes.
    down_padded_f32: Option<(Vec<f32>, usize, usize)>,
}

/// Write the FFN gate/up/down legs of every layer to
/// `interleaved_kquant.bin` in `[gate Q4_K | up Q4_K | down Q6_K]`
/// layer-major order, plus a sidecar manifest. When
/// `opts.feature_major_down` is set, also emit `down_features_q4k.bin`
/// with the down weights transposed into `[intermediate, hidden]`
/// orientation so per-feature decode at load time can skip the cache.
///
/// `jobs <= 1` transforms and writes each layer serially, identical to
/// the pre-IMPORT-002 code. `jobs > 1` transforms up to `jobs` layers'
/// gate/up/down concurrently (bounded, chunked), but always writes
/// bytes, assigns manifest offsets, and appends the feature-major-down
/// sidecar on this thread in ascending layer order.
pub(super) fn write_interleaved_ffn_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
    jobs: usize,
    opts: KquantWriteOptions,
    callbacks: &mut dyn IndexBuildCallbacks,
    rec: &Recorder<'_>,
) -> Result<(), VindexError> {
    let arch = source.arch();
    let ff_path = dir.join(INTERLEAVED_KQUANT_BIN);
    let mut ff_file = BufWriter::new(std::fs::File::create(&ff_path)?);
    let mut ff_offset: u64 = 0;
    let mut ff_manifest: Vec<Q4kManifestEntry> = Vec::with_capacity(num_layers * 3);

    // ── down_features_q4k.bin (W2 feature-major down, opt-in) ──
    //
    // Captures the same down-proj data as interleaved_kquant.bin's down
    // slot, but transposed to [intermediate, hidden] orientation and
    // re-quantised at the same precision. Lets per-feature decode at
    // load time skip the cache. Allocated lazily so non-opt-in
    // extracts pay nothing.
    let mut fm_state: Option<FeatureMajorDownState> = if opts.feature_major_down {
        Some(FeatureMajorDownState::new(
            &dir.join(DOWN_FEATURES_KQUANT_BIN),
            num_layers,
        )?)
    } else {
        None
    };

    transform_then_write(
        num_layers,
        jobs,
        |layer| transform_ffn_layer(source, arch, layer, opts, rec),
        |layer, blob| {
            callbacks.on_layer_start(COMP_FFN_KQUANT, layer, num_layers);
            let component_names = ["gate", "up", "down"];
            for (i, slot) in blob.slots.into_iter().enumerate() {
                let Some(slot) = slot else { continue };
                let component = component_names[i];
                let t = rec.now();
                ff_file.write_all(&slot.bytes)?;
                rec.write(
                    t,
                    COMP_FFN_KQUANT,
                    component,
                    Some(layer),
                    slot.bytes.len() as u64,
                );
                let length = slot.bytes.len() as u64;
                ff_manifest.push(Q4kManifestEntry {
                    key: slot.key.clone(),
                    shape: slot.shape.to_vec(),
                    format: slot.format.clone(),
                    offset: ff_offset,
                    length,
                });
                ff_offset += length;

                if i == 2 {
                    // down
                    let t = rec.now();
                    if let (Some(state), Some((padded, rows, padded_cols))) =
                        (fm_state.as_mut(), blob.down_padded_f32.as_ref())
                    {
                        state.append_layer(slot.key, padded, *rows, *padded_cols, slot.format)?;
                    }
                    rec.write(t, COMP_FFN_KQUANT, "down_fm", Some(layer), 0);
                }
            }
            callbacks.on_layer_done(COMP_FFN_KQUANT, layer, 0.0);
            Ok(())
        },
    )?;

    ff_file.flush()?;
    drop(ff_file);

    let t = rec.now();
    let ff_manifest_json = serde_json::to_string_pretty(&ff_manifest)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(INTERLEAVED_KQUANT_MANIFEST_JSON), ff_manifest_json)?;
    rec.manifest(t, COMP_FFN_KQUANT, "ffn_manifest");

    if let Some(state) = fm_state.take() {
        let t = rec.now();
        state.finalize(&dir.join(DOWN_FEATURES_KQUANT_MANIFEST_JSON))?;
        rec.manifest(t, COMP_FFN_KQUANT, "down_fm_finalize");
    }
    Ok(())
}

/// Fetch, pad, and quantise gate/up/down for one layer. Pure read-only
/// access to `source` — safe to call concurrently across layers since
/// `WeightSource: Sync` and every method returns owned data.
fn transform_ffn_layer(
    source: &dyn WeightSource,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
    opts: KquantWriteOptions,
    rec: &Recorder<'_>,
) -> Result<FfnLayerBlob, VindexError> {
    let mut slots: [Option<QuantSlot>; 3] = [None, None, None];
    let mut down_padded_f32 = None;

    for (i, key) in [
        arch.ffn_gate_key(layer),
        arch.ffn_up_key(layer),
        arch.ffn_down_key(layer),
    ]
    .into_iter()
    .enumerate()
    {
        let component = ["gate", "up", "down"][i];
        let t = rec.now();
        let fetched = source.get_tensor(&key);
        let (data, rows, cols) = match fetched {
            Some(t) => t,
            None => {
                rec.fetch(t, COMP_FFN_KQUANT, component, Some(layer), 0, 0);
                continue;
            }
        };
        rec.fetch(
            t,
            COMP_FFN_KQUANT,
            component,
            Some(layer),
            rows as u64,
            cols as u64,
        );

        // Row-pad to 256 so each row aligns to a super-block boundary.
        // Without this, matrices with `cols % 256 != 0` (e.g. Gemma 4
        // 26B A4B's down_proj with inner dim 2112) store contiguous
        // quantisation that every row past row 0 reads wrong. See
        // `pad_rows_to_block` docs.
        let t = rec.now();
        let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
        rec.pad(
            t,
            COMP_FFN_KQUANT,
            component,
            Some(layer),
            cols as u64,
            padded_cols as u64,
        );
        let in_elems = (rows * padded_cols) as u64;
        // Gate (i=0) and up (i=1) always Q4_K. Down (i=2) format
        // is controlled by `opts.down_proj` (Q6_K by default for
        // llama.cpp compatibility, Q4_K when the caller opts in).
        let is_down = i == 2;
        let use_q6 = is_down && opts.down_proj == super::DownProjFormat::Q6K;
        let t = rec.now();
        let q_bytes = if use_q6 {
            quantize_q6_k(&padded)
        } else {
            quantize_q4_k(&padded)
        };
        let format = if use_q6 {
            QuantBlockFormat::Q6K
        } else {
            QuantBlockFormat::Q4K
        };
        rec.quantize(
            t,
            COMP_FFN_KQUANT,
            component,
            Some(layer),
            format.tag(),
            in_elems,
            q_bytes.len() as u64,
        );

        if is_down && opts.feature_major_down {
            down_padded_f32 = Some((padded.clone(), rows, padded_cols));
        }

        slots[i] = Some(QuantSlot {
            key,
            bytes: q_bytes,
            shape: [rows, padded_cols],
            format,
        });
    }

    Ok(FfnLayerBlob {
        slots,
        down_padded_f32,
    })
}

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
use super::super::write_f32::WeightSource;
use super::feature_major_down::FeatureMajorDownState;
use super::{pad_rows_to_block, KquantWriteOptions, QuantBlockFormat};

/// Write the FFN gate/up/down legs of every layer to
/// `interleaved_kquant.bin` in `[gate Q4_K | up Q4_K | down Q6_K]`
/// layer-major order, plus a sidecar manifest. When
/// `opts.feature_major_down` is set, also emit `down_features_q4k.bin`
/// with the down weights transposed into `[intermediate, hidden]`
/// orientation so per-feature decode at load time can skip the cache.
pub(super) fn write_interleaved_ffn_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
    opts: KquantWriteOptions,
    callbacks: &mut dyn IndexBuildCallbacks,
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

    for layer in 0..num_layers {
        callbacks.on_layer_start(COMP_FFN_KQUANT, layer, num_layers);
        for (i, key) in [
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ]
        .iter()
        .enumerate()
        {
            // Desired per-slot format: gate (i=0) + up (i=1) always Q4_K;
            // down (i=2) controlled by `opts.down_proj`.
            let is_down = i == 2;
            let use_q6 = is_down && opts.down_proj == super::DownProjFormat::Q6K;
            let want_format = if use_q6 {
                QuantBlockFormat::Q6K
            } else {
                QuantBlockFormat::Q4K
            };

            // Quant-preserving fast path: if the source can hand us raw
            // bytes already in `want_format` and already 256-padded (cols
            // == cols_padded), emit them verbatim -- no f32 reification,
            // no requantisation. This is the GGUF Q4_K -> vindex Q4_K
            // passthrough that keeps MoE imports memory-bounded.
            //
            // If the source offers packed bytes but in the *wrong* format
            // (e.g. Q4_K when we want Q6_K for down, or Q6_K when we
            // want Q4_K), we cannot reuse them and fall through to the
            // f32 path -- mixing formats would corrupt the decode.
            if let Some(pq) = source.get_packed_quant(key) {
                if pq.format == want_format && pq.cols == pq.cols_padded {
                    let length = pq.bytes.len() as u64;
                    ff_file.write_all(&pq.bytes)?;
                    ff_manifest.push(Q4kManifestEntry {
                        key: key.clone(),
                        shape: vec![pq.rows, pq.cols_padded],
                        format: want_format.clone(),
                        offset: ff_offset,
                        length,
                    });
                    ff_offset += length;

                    if is_down {
                        if let Some(state) = fm_state.as_mut() {
                            // Feature-major down still needs the f32 data
                            // (it re-quantises a transposed view). The
                            // packed bytes can't be transposed in-place,
                            // so dequantise just for the sidecar when
                            // opted in. This is a bounded cost: one
                            // layer's down tensor at a time.
                            if let Some((data, rows, cols)) = source.get_tensor(key) {
                                let (padded, padded_cols) =
                                    pad_rows_to_block(&data, rows, cols);
                                state.append_layer(
                                    key.clone(),
                                    &padded,
                                    rows,
                                    padded_cols,
                                    want_format,
                                )?;
                            }
                        }
                    }
                    continue;
                }
            }

            // Standard path: dequantise to f32, pad rows to 256, requantise.
            if let Some((data, rows, cols)) = source.get_tensor(key) {
                // Row-pad to 256 so each row aligns to a super-block boundary.
                // Without this, matrices with `cols % 256 != 0` (e.g. Gemma 4
                // 26B A4B's down_proj with inner dim 2112) store contiguous
                // quantisation that every row past row 0 reads wrong. See
                // `pad_rows_to_block` docs.
                let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
                let q_bytes = if use_q6 {
                    quantize_q6_k(&padded)
                } else {
                    quantize_q4_k(&padded)
                };
                let format = want_format;
                ff_file.write_all(&q_bytes)?;
                let length = q_bytes.len() as u64;
                ff_manifest.push(Q4kManifestEntry {
                    key: key.clone(),
                    shape: vec![rows, padded_cols],
                    format: format.clone(),
                    offset: ff_offset,
                    length,
                });
                ff_offset += length;

                if is_down {
                    if let Some(state) = fm_state.as_mut() {
                        state.append_layer(key.clone(), &padded, rows, padded_cols, format)?;
                    }
                }
            }
        }
        callbacks.on_layer_done(COMP_FFN_KQUANT, layer, 0.0);
    }
    ff_file.flush()?;
    drop(ff_file);

    let ff_manifest_json = serde_json::to_string_pretty(&ff_manifest)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(INTERLEAVED_KQUANT_MANIFEST_JSON), ff_manifest_json)?;

    if let Some(state) = fm_state.take() {
        state.finalize(&dir.join(DOWN_FEATURES_KQUANT_MANIFEST_JSON))?;
    }
    Ok(())
}

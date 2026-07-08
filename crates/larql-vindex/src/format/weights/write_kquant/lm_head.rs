//! Stage 6 — `lm_head_q4.bin`.
//!
//! Q4_K of the output projection matrix. Falls back to embed_tokens
//! when the architecture ties the embed and lm_head weights (Gemma,
//! Qwen, etc.); the source layer surfaces that via `source.lm_head()`.
//! Manifest entry is appended to the running norms manifest so
//! `weight_manifest.json` references everything in one list.

use std::path::Path;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;

use crate::error::VindexError;
use crate::format::filenames::*;

use super::super::profile::Recorder;
use super::super::write_f32::{kind, WeightEntry, WeightSource};
use super::pad_rows_to_block;

pub(super) fn write_lm_head_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    norm_entries: &mut Vec<WeightEntry>,
    rec: &Recorder<'_>,
) -> Result<(), VindexError> {
    let t_fetch = rec.now();
    let lm = source.lm_head();
    let (rows, cols) = lm
        .as_ref()
        .map(|(_, r, c)| (*r as u64, *c as u64))
        .unwrap_or((0, 0));
    rec.fetch(t_fetch, "lm_head", "lm_head", None, rows, cols);
    if let Some((data, rows, cols)) = lm {
        let t = rec.now();
        let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
        rec.pad(
            t,
            "lm_head",
            "lm_head",
            None,
            cols as u64,
            padded_cols as u64,
        );
        let in_elems = (rows * padded_cols) as u64;
        let t = rec.now();
        let q_bytes = quantize_q4_k(&padded);
        rec.quantize(
            t,
            "lm_head",
            "lm_head",
            None,
            "Q4_K",
            in_elems,
            q_bytes.len() as u64,
        );
        let t = rec.now();
        std::fs::write(dir.join(LM_HEAD_KQUANT_BIN), &q_bytes)?;
        rec.write(t, "lm_head", "lm_head", None, q_bytes.len() as u64);
        // Record in norms manifest so a single weight_manifest.json references
        // everything non-quantised-via-layout. Shape records the stored
        // `padded_cols` — callers route through the matvec dispatch which
        // uses shape[1] as `K`, so the padding stays invisible provided the
        // input activation buffer is zero-padded to match.
        norm_entries.push(WeightEntry {
            key: "lm_head.weight".into(),
            kind: kind::TENSOR_Q4K.into(),
            shape: vec![rows, padded_cols],
            offset: 0,
            length: q_bytes.len() as u64,
            file: LM_HEAD_KQUANT_BIN.into(),
        });
    }
    Ok(())
}

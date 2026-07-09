//! Shared embedding-table loader.
//!
//! Both the f32 and Q4_K loaders mmap `embeddings.bin`, auto-detect
//! f32 vs f16 storage from the byte count, decode to f32, and reshape
//! to `[vocab_size, hidden_size]`. The f32 loader additionally exposes
//! a "skip embed" path for `LoadWeightsOptions::skip_embed` — used by
//! FFN-service workers that never see token IDs and don't need the
//! embedding table in heap.

use std::path::Path;

use ndarray::Array2;

use crate::config::VindexConfig;
use crate::error::VindexError;
use crate::format::filenames::*;
use crate::index::core::IndexLoadCallbacks;

/// Mmap + decode `embeddings.bin` into a `[vocab_size, hidden_size]`
/// array of f32. Detects f32 vs f16 storage from the byte count and
/// fails loudly when the file length is incompatible with the recorded
/// `vocab_size × hidden_size` (which would otherwise silently decode
/// garbage — the original padded-vocab failure mode).
///
/// Emits `on_file_start` / `on_file_done` callbacks around the work.
pub(super) fn load_embeddings(
    dir: &Path,
    config: &VindexConfig,
    callbacks: &mut dyn IndexLoadCallbacks,
) -> Result<Array2<f32>, VindexError> {
    callbacks.on_file_start(
        "embeddings",
        &dir.join(EMBEDDINGS_BIN).display().to_string(),
    );
    let embed_file = std::fs::File::open(dir.join(EMBEDDINGS_BIN))?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file)? };
    let expected_f32_bytes = config.vocab_size * config.hidden_size * 4;
    let expected_f16_bytes = config.vocab_size * config.hidden_size * 2;
    let embed_dtype = match embed_mmap.len() {
        len if len == expected_f32_bytes => crate::config::dtype::StorageDtype::F32,
        len if len == expected_f16_bytes => crate::config::dtype::StorageDtype::F16,
        len => {
            return Err(VindexError::Parse(format!(
                "embeddings.bin length {} incompatible with vocab_size={} hidden_size={} \
                 (expected f32={} or f16={} bytes). The recorded vocab_size disagrees with \
                 the embedding tensor's physical row count — rebuild the vindex with a \
                 current extractor.",
                len, config.vocab_size, config.hidden_size, expected_f32_bytes, expected_f16_bytes,
            )))
        }
    };
    let embed_floats = crate::config::dtype::decode_floats(&embed_mmap, embed_dtype);
    let arr = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    callbacks.on_file_done("embeddings", config.vocab_size, 0.0);
    Ok(arr)
}

/// Empty-shape placeholder used by f32 loader's `skip_embed` path.
/// Pinned out as a function so the callsite is self-documenting and the
/// callback ordering matches the non-skipped path.
pub(super) fn empty_embeddings(callbacks: &mut dyn IndexLoadCallbacks) -> Array2<f32> {
    callbacks.on_file_start("embeddings (skipped)", "opts.skip_embed=true");
    let arr = Array2::<f32>::zeros((0, 0));
    callbacks.on_file_done("embeddings", 0, 0.0);
    arr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::dtype::StorageDtype;
    use crate::config::types::QuantFormat;
    use std::cell::RefCell;

    /// Records callback order for assertions.
    #[derive(Default)]
    struct Recording {
        events: RefCell<Vec<String>>,
    }
    impl IndexLoadCallbacks for &Recording {
        fn on_file_start(&mut self, component: &str, path: &str) {
            self.events
                .borrow_mut()
                .push(format!("start:{component}:{path}"));
        }
        fn on_progress(&mut self, _: usize) {}
        fn on_file_done(&mut self, component: &str, records: usize, _: f64) {
            self.events
                .borrow_mut()
                .push(format!("done:{component}:{records}"));
        }
    }

    fn config_for(vocab: usize, hidden: usize) -> VindexConfig {
        VindexConfig {
            version: 2,
            model: "test/model".into(),
            family: "test".into(),
            num_layers: 2,
            hidden_size: hidden,
            intermediate_size: hidden * 2,
            vocab_size: vocab,
            logical_vocab_size: None,
            embed_scale: 1.0,
            layers: Vec::new(),
            down_top_k: 1,
            has_model_weights: true,
            source: None,
            checksums: None,
            extract_level: crate::ExtractLevel::All,
            dtype: StorageDtype::F32,
            quant: QuantFormat::None,
            layer_bands: crate::LayerBands::for_family("test", 2),
            model_config: None,
            fp4: None,
            ffn_layout: None,
            bitnet_layout: None,
        }
    }

    fn write_embeddings_bin(dir: &std::path::Path, floats: &[f32], dtype: StorageDtype) {
        let bytes = crate::config::dtype::encode_floats(floats, dtype);
        std::fs::write(dir.join(EMBEDDINGS_BIN), &bytes).unwrap();
    }

    #[test]
    fn load_embeddings_decodes_f32_storage() {
        let tmp = tempfile::tempdir().unwrap();
        let vocab = 4;
        let hidden = 3;
        let floats: Vec<f32> = (0..vocab * hidden).map(|i| i as f32).collect();
        write_embeddings_bin(tmp.path(), &floats, StorageDtype::F32);

        let config = config_for(vocab, hidden);
        let recording = Recording::default();
        let mut cb = &recording;
        let arr = load_embeddings(tmp.path(), &config, &mut cb).unwrap();

        assert_eq!(arr.shape(), &[vocab, hidden]);
        assert_eq!(arr[[0, 0]], 0.0);
        assert_eq!(arr[[3, 2]], 11.0);
        // Both callbacks fired, in order.
        let events = recording.events.borrow();
        assert_eq!(events.len(), 2);
        assert!(events[0].starts_with("start:embeddings:"));
        assert_eq!(events[1], format!("done:embeddings:{vocab}"));
    }

    #[test]
    fn load_embeddings_decodes_f16_storage() {
        // f16 file byte count exactly matches `vocab * hidden * 2`, so the
        // dtype detector routes through StorageDtype::F16.
        let tmp = tempfile::tempdir().unwrap();
        let vocab = 2;
        let hidden = 4;
        let floats = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
        write_embeddings_bin(tmp.path(), &floats, StorageDtype::F16);

        let config = config_for(vocab, hidden);
        let recording = Recording::default();
        let mut cb = &recording;
        let arr = load_embeddings(tmp.path(), &config, &mut cb).unwrap();

        assert_eq!(arr.shape(), &[vocab, hidden]);
        // f16 loses some precision; check within tolerance.
        for (got, want) in arr.iter().zip(floats.iter()) {
            assert!((got - want).abs() < 1e-2, "f16 round-trip: {got} vs {want}");
        }
    }

    /// VINDEX-001: a padded-vocab f16 embedding file (physical rows > the
    /// logical/tokenizer vocab) is loaded at the physical row count when
    /// `config.vocab_size` records the physical count. This is the Qwen2.5
    /// shape (151936 physical rows vs 151643 logical tokens); here scaled
    /// down to physical=16, logical=8.
    #[test]
    fn load_embeddings_padded_f16_uses_physical_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let physical = 16;
        let logical = 8;
        let hidden = 4;
        let floats: Vec<f32> = (0..physical * hidden).map(|i| i as f32).collect();
        write_embeddings_bin(tmp.path(), &floats, StorageDtype::F16);

        let mut config = config_for(physical, hidden);
        config.logical_vocab_size = Some(logical);
        let recording = Recording::default();
        let mut cb = &recording;
        let arr = load_embeddings(tmp.path(), &config, &mut cb).unwrap();

        // Physical row count, not the logical vocab — the file holds
        // physical rows and must reshape to physical × hidden.
        assert_eq!(arr.shape(), &[physical, hidden]);
        assert_eq!(arr[[0, 0]], 0.0);
        assert_eq!(
            arr[[physical - 1, hidden - 1]],
            (physical * hidden - 1) as f32
        );
    }

    /// VINDEX-001: padded f32 embedding file loads at the physical row count.
    #[test]
    fn load_embeddings_padded_f32_uses_physical_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let physical = 16;
        let logical = 8;
        let hidden = 4;
        let floats: Vec<f32> = (0..physical * hidden).map(|i| i as f32).collect();
        write_embeddings_bin(tmp.path(), &floats, StorageDtype::F32);

        let mut config = config_for(physical, hidden);
        config.logical_vocab_size = Some(logical);
        let recording = Recording::default();
        let mut cb = &recording;
        let arr = load_embeddings(tmp.path(), &config, &mut cb).unwrap();

        assert_eq!(arr.shape(), &[physical, hidden]);
        assert_eq!(arr[[0, 0]], 0.0);
        assert_eq!(
            arr[[physical - 1, hidden - 1]],
            (physical * hidden - 1) as f32
        );
    }

    /// VINDEX-001: an embeddings.bin whose length is incompatible with the
    /// recorded `vocab_size × hidden_size` fails loudly instead of silently
    /// misdetecting dtype and decoding garbage. This is the regression that
    /// produced garbage CPU/CUDA output for padded-vocab models.
    #[test]
    fn load_embeddings_incompatible_length_fails_loudly() {
        let tmp = tempfile::tempdir().unwrap();
        let declared_physical = 16;
        let hidden = 4;
        // File sized for only 12 rows — neither f32 nor f16 expectation.
        let floats: Vec<f32> = (0..12 * hidden).map(|i| i as f32).collect();
        write_embeddings_bin(tmp.path(), &floats, StorageDtype::F32);

        let config = config_for(declared_physical, hidden);
        let recording = Recording::default();
        let mut cb = &recording;
        let err = load_embeddings(tmp.path(), &config, &mut cb)
            .expect_err("incompatible length must error");
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("incompatible"),
            "error should mention incompatible length: {msg}"
        );
        assert!(
            msg.contains("vocab_size") || msg.contains("f32") || msg.contains("f16"),
            "error should reference the expected shapes: {msg}"
        );
    }

    /// VINDEX-001: `logical_vocab_size` round-trips through serde and is
    /// omitted from JSON when `None` (backward compat for old vindexes).
    #[test]
    fn logical_vocab_size_serde_round_trip() {
        // Some(logical) — serialised and deserialised.
        let mut cfg = config_for(16, 4);
        cfg.logical_vocab_size = Some(8);
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(
            json.contains("\"logical_vocab_size\""),
            "field present: {json}"
        );
        let parsed: VindexConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.logical_vocab_size, Some(8));

        // None — omitted from JSON; old vindexes without the field still load.
        let cfg_none = config_for(16, 4);
        let json_none = serde_json::to_string(&cfg_none).unwrap();
        assert!(
            !json_none.contains("logical_vocab_size"),
            "None should be skipped: {json_none}"
        );
        let parsed_none: VindexConfig = serde_json::from_str(&json_none).unwrap();
        assert_eq!(parsed_none.logical_vocab_size, None);
    }

    #[test]
    fn load_embeddings_errors_on_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config = config_for(2, 2);
        let recording = Recording::default();
        let mut cb = &recording;
        let err = load_embeddings(tmp.path(), &config, &mut cb).expect_err("missing file errors");
        // I/O error wrapping; just confirm we got an error rather than
        // producing zero-filled garbage.
        assert!(
            err.to_string().to_lowercase().contains("no such")
                || err.to_string().to_lowercase().contains("not found")
                || err.to_string().to_lowercase().contains("os error")
        );
    }

    #[test]
    fn empty_embeddings_returns_zero_shape_and_fires_callbacks() {
        let recording = Recording::default();
        let mut cb = &recording;
        let arr = empty_embeddings(&mut cb);
        assert_eq!(arr.shape(), &[0, 0]);

        let events = recording.events.borrow();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "start:embeddings (skipped):opts.skip_embed=true");
        assert_eq!(events[1], "done:embeddings:0");
    }
}

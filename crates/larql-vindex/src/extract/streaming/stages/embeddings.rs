//! Stage 2 — embeddings.

use crate::error::VindexError;
use crate::extract::stage_labels::*;
use crate::extract::streaming::context::StreamingContext;
use crate::extract::streaming::tensor_io::normalize_key;
use crate::format::filenames::*;

impl<'a> StreamingContext<'a> {
    /// Stage 2 — embeddings.
    pub(in crate::extract::streaming) fn write_embeddings(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_EMBEDDINGS);
        let prefixes: Vec<&str> = self.prefixes.iter().map(|s| s.as_str()).collect();
        let embed_key = normalize_key(self.arch.embed_key(), &prefixes);
        let embed = self
            .tensor_source
            .get_tensor_f32(&embed_key)?
            .ok_or_else(|| VindexError::MissingTensor(embed_key.clone()))?;
        // Physical vocab = actual embedding row count (GGUF/kquant may pad
        // above the tokenizer vocab; e.g. Qwen2.5: 151936 rows vs 151643
        // tokens). The lm_head matmul and byte-length checks all use the
        // physical count. Preserve the smaller logical/tokenizer vocab so
        // the loader can mask padding rows during sampling.
        let physical = embed.shape()[0];
        self.vocab_size = physical;
        let cfg_vocab = self
            .arch
            .config()
            .vocab_size
            .filter(|&v| v > 0 && v < physical);
        let logical = cfg_vocab.or_else(|| {
            let tok_vocab = self.tokenizer.get_vocab_size(false);
            (tok_vocab > 0 && tok_vocab < physical).then_some(tok_vocab)
        });
        self.logical_vocab_size = logical;
        let embed_data = embed.as_slice().unwrap();
        let embed_bytes = crate::config::dtype::encode_floats(embed_data, self.dtype);
        std::fs::write(self.output_dir.join(EMBEDDINGS_BIN), &embed_bytes)?;
        self.embed = Some(embed);
        self.callbacks.on_stage_done(STAGE_EMBEDDINGS, 0.0);
        Ok(())
    }
}

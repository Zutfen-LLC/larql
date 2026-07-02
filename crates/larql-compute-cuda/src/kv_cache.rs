//! KV cache management for the CUDA decode path.
//!
//! Per-layer device buffers for cached K/V vectors. Grows with generation.
//! Mirrors `larql-compute-metal::ops::kv_cache::KVCache` in shape: each
//! layer owns a `[max_seq, num_kv_heads, head_dim]` f32 K and V buffer plus a
//! `current_len` cursor; the cache supports heterogeneous per-layer shapes
//! (Gemma 4 alternates sliding/global attention geometry).
//!
//! Unlike Metal (which holds device buffers via the `metal::MTLBuffer` ref
//! type), CUDA holds them as `cudarc::driver::CudaSlice<f32>` — an owned,
//! `Send + Sync` device allocation bound to the runtime's stream. Allocation
//! is lazy: a `CudaKVCache` is only constructible when a `CudaRuntime` is
//! present, so the scaffold (no-device) path keeps the `DecodeBackend`
//! lifecycle methods delegating to no-ops / CPU.
//!
//! This slice lands the cache lifecycle + a native `kv_append` kernel
//! (one thread per K/V element, copies the new row into the slot at
//! `current_len`). Attention-over-cache and the full fused decode/prefill
//! pipelines are later Phase 4 slices; for now `populate_kv_layer` writes
//! rows to the device cache and the host reference store stays the source of
//! truth for any readback path.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

/// KV cache for one layer — pre-allocated device buffers.
///
/// Layout: `[max_seq, num_kv_heads, head_dim]` f32, row-major over the
/// sequence dimension. The slot at index `pos` is
/// `[pos * num_kv_heads * head_dim .. (pos+1) * num_kv_heads * head_dim]`.
pub struct LayerKVCache {
    pub k_cache: CudaSlice<f32>,
    pub v_cache: CudaSlice<f32>,
    pub current_len: usize,
    pub max_seq: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKVCache {
    /// Allocate empty K/V device buffers for one layer.
    pub fn new(
        stream: &Arc<CudaStream>,
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KvCacheError> {
        let elems = max_seq
            .checked_mul(num_kv_heads)
            .and_then(|e| e.checked_mul(head_dim))
            .ok_or(KvCacheError::ShapeOverflow {
                max_seq,
                num_kv_heads,
                head_dim,
            })?;
        let k_cache = stream
            .alloc_zeros::<f32>(elems)
            .map_err(|err| KvCacheError::Alloc(err.to_string()))?;
        let v_cache = stream
            .alloc_zeros::<f32>(elems)
            .map_err(|err| KvCacheError::Alloc(err.to_string()))?;
        Ok(Self {
            k_cache,
            v_cache,
            current_len: 0,
            max_seq,
            num_kv_heads,
            head_dim,
        })
    }

    /// Reset the cursor (for a new prompt). The device buffers are not
    /// zeroed — subsequent appends overwrite slots 0.. in order, and the
    /// decode/attend kernels only read `0..current_len`.
    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

/// Full KV cache for all layers.
pub struct CudaKVCache {
    pub layers: Vec<LayerKVCache>,
}

impl CudaKVCache {
    /// Allocate with per-layer shapes. Llama / Mistral / Gemma 3 (uniform
    /// geometry) pass a repeated shape; Gemma 4 31B alternates sliding
    /// (num_kv=16, head_dim=256) with global (num_kv=4, head_dim=512) layers,
    /// so a single uniform allocation would either over-size globals or
    /// under-size slidings and produce wrong attention reads.
    ///
    /// `shapes[i]` is `(num_kv_heads_i, head_dim_i)` for layer i.
    pub fn new_per_layer(
        stream: &Arc<CudaStream>,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> Result<Self, KvCacheError> {
        let mut layers = Vec::with_capacity(shapes.len());
        for &(num_kv, hd) in shapes {
            layers.push(LayerKVCache::new(stream, max_seq, num_kv, hd)?);
        }
        Ok(Self { layers })
    }

    /// Return true if any already-allocated layer disagrees with the
    /// corresponding expected `(num_kv_heads, head_dim)` shape.
    pub fn has_shape_mismatch(&self, shapes: &[(usize, usize)]) -> bool {
        self.layers
            .iter()
            .zip(shapes.iter())
            .any(|(layer, &(num_kv, hd))| layer.num_kv_heads != num_kv || layer.head_dim != hd)
    }

    /// Grow the cache to cover `shapes`, preserving existing matching layers.
    pub fn grow_to_shapes(
        &mut self,
        stream: &Arc<CudaStream>,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> Result<(), KvCacheError> {
        while self.layers.len() < shapes.len() {
            let (num_kv_heads, head_dim) = shapes[self.layers.len()];
            self.layers
                .push(LayerKVCache::new(stream, max_seq, num_kv_heads, head_dim)?);
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn current_len(&self) -> usize {
        self.layers.first().map(|l| l.current_len).unwrap_or(0)
    }
}

/// Errors arising from device KV-cache allocation.
#[derive(Debug, Clone)]
pub enum KvCacheError {
    /// `max_seq * num_kv_heads * head_dim` overflowed `usize`.
    ShapeOverflow {
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    /// The underlying `cuMemAlloc` failed.
    Alloc(String),
}

impl std::fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeOverflow {
                max_seq,
                num_kv_heads,
                head_dim,
            } => write!(
                f,
                "KV cache shape overflow: max_seq={max_seq} num_kv_heads={num_kv_heads} head_dim={head_dim}"
            ),
            Self::Alloc(msg) => write!(f, "KV cache device allocation failed: {msg}"),
        }
    }
}

impl std::error::Error for KvCacheError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape_pairs_have_mismatch(existing: &[(usize, usize)], expected: &[(usize, usize)]) -> bool {
        existing.iter().zip(expected.iter()).any(
            |(&(actual_num_kv, actual_head_dim), &(expected_num_kv, expected_head_dim))| {
                actual_num_kv != expected_num_kv || actual_head_dim != expected_head_dim
            },
        )
    }

    #[test]
    fn shape_mismatch_detects_conflicting_existing_layer() {
        let small = (2usize, 64usize);
        let large = (4usize, 128usize);
        assert!(!shape_pairs_have_mismatch(&[small], &[small, large]));
        assert!(shape_pairs_have_mismatch(&[small], &[large]));
    }

    #[test]
    fn kv_cache_error_display_mentions_overflow_dims() {
        let err = KvCacheError::ShapeOverflow {
            max_seq: 4096,
            num_kv_heads: 8,
            head_dim: 256,
        };
        let s = err.to_string();
        assert!(s.contains("4096"));
        assert!(s.contains("overflow"));
    }
}

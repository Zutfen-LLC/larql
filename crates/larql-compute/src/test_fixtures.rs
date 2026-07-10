//! Test fixtures for compute-side tests that need a real `KvIndex`
//! implementation backed by Q4_K-quantized bytes.
//!
//! Gated behind the `test-utils` feature so production builds never
//! compile the fixture code. Enabled from a consumer's
//! `[dev-dependencies]` entry, e.g.
//! `larql-compute = { path = "../larql-compute", features = ["test-utils"] }`.
//!
//! ## Why this lives in `larql-compute`
//!
//! The `KvIndex` trait is defined here, so the fixture's `impl KvIndex`
//! is an in-crate impl (no orphan-rule issues). Downstream test code
//! in `larql-compute` itself and `larql-compute-metal` can both use it
//! without a `larql-vindex` dev-dep (vindex itself impls `KvIndex` on
//! `VectorIndex`, but it depends on `larql-compute` — pulling vindex in
//! as a dev-dep would create a back-edge that's better avoided when a
//! ~150-LOC standalone fixture works just as well).

use std::collections::HashMap;
use std::sync::Arc;

use larql_models::quant::ggml::q6_k::dequantize_q6_k;
use larql_models::ModelWeights;

use crate::cpu::ops::q4_common::{dequantize_q4_k, quantize_q4_k, quantize_q6_k};
use crate::kv_index::{KvIndex, FFN_COMPONENTS_PER_LAYER};

/// Per-(layer, component) dequantised FFN block — lazily populated on first
/// request through `kquant_ffn_layer_once`. Aliased here only to keep the
/// containing struct under clippy's `type_complexity` threshold.
type FfnDequantCache = std::sync::Mutex<HashMap<(usize, usize), Arc<Vec<f32>>>>;

/// `KvIndex` backed by Q4_K-quantized weight tensors held in
/// in-process memory. Drives `kquant_forward::fused_*` and the
/// `coarse_*` paths on `KvDispatch` impls end-to-end without
/// constructing a full `VectorIndex`.
///
/// Construct via [`make_q4k_fixture_index`].
pub struct Q4kFixtureIndex {
    /// Concatenated Q4_K bytes for FFN gate/up/down across all layers,
    /// laid out as `[layer 0: gate, up, down; layer 1: gate, up, down; ...]`.
    /// `interleaved_kquant_mmap_ref` returns this whole slice;
    /// `interleaved_kquant_layer_data` slices into it at the per-layer
    /// offset.
    ffn_mmap: Vec<u8>,
    /// Per-component byte count: `Q4_K::packed_matrix_bytes(intermediate, hidden)`.
    /// Same value for every (layer, component) at this fixture scale.
    ffn_per_matrix: usize,
    /// Concatenated Q4_K bytes for attention Q/K/V/O across all layers,
    /// laid out as `[layer 0: Q, K, V, O; layer 1: Q, K, V, O; ...]`.
    attn_mmap: Vec<u8>,
    /// Per-layer (offset, length) pairs for Q/K/V/O in `attn_mmap`. Q/K/V/O
    /// have different shapes (q_dim vs kv_dim) so the offsets aren't a
    /// fixed stride.
    attn_offsets: Vec<[(usize, usize); 4]>,
    /// Per-(layer, component) dequantised FFN cache populated lazily
    /// on first request through `kquant_ffn_layer_once`.
    ffn_cache: FfnDequantCache,
    /// Intermediate dimension — `num_features` returns this.
    intermediate: usize,
    /// Vocabulary size — `vocab_size` returns this.
    vocab_size: usize,
    /// When `true`, `kquant_ffn_layer_once` returns `None` unconditionally
    /// so callers take the dequant-from-bytes fallback path. Default
    /// `false` (lazy cache enabled). Flipped on by
    /// [`Q4kFixtureIndex::without_ffn_cache`] for tests that need to
    /// drive both branches.
    disable_ffn_cache: bool,
    /// When `true`, the trait method `interleaved_kquant_mmap_ref`
    /// returns None and `interleaved_q4_mmap_ref` returns the bytes
    /// — drives the Q4_0 fallback branch in `fused_prefill`.
    use_legacy_q4_mmap: bool,
}

impl Q4kFixtureIndex {
    /// Disable the lazy dequant cache. Subsequent
    /// `kquant_ffn_layer_once` calls always return `None`, forcing
    /// callers down the `dequantize_matrix` path. Used to test the
    /// fallback branch of `kquant_ffn_forward_layer`.
    pub fn without_ffn_cache(mut self) -> Self {
        self.disable_ffn_cache = true;
        self
    }

    /// Swap the FFN mmap accessor to return `interleaved_q4_mmap_ref`
    /// (Q4_0 legacy format) instead of `interleaved_kquant_mmap_ref`.
    /// Drives the Q4_0 fallback branch in `fused_prefill` /
    /// `fused_decode_step_inner` that picks between the two mmap
    /// accessors. The underlying bytes stay Q4_K-quantized — the
    /// branch just records `ffn_is_q4k = false` and tags the format
    /// downstream.
    pub fn as_legacy_q4_mmap(mut self) -> Self {
        self.use_legacy_q4_mmap = true;
        self
    }
}

impl KvIndex for Q4kFixtureIndex {
    fn num_features(&self, _layer: usize) -> usize {
        self.intermediate
    }

    fn attn_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        let offsets = self.attn_offsets.get(layer)?;
        let attn = &self.attn_mmap;
        Some([
            (&attn[offsets[0].0..offsets[0].0 + offsets[0].1], "Q4_K"),
            (&attn[offsets[1].0..offsets[1].0 + offsets[1].1], "Q4_K"),
            (&attn[offsets[2].0..offsets[2].0 + offsets[2].1], "Q4_K"),
            (&attn[offsets[3].0..offsets[3].0 + offsets[3].1], "Q4_K"),
        ])
    }

    fn interleaved_kquant_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(&[u8], &str); FFN_COMPONENTS_PER_LAYER]> {
        let per_matrix = self.ffn_per_matrix;
        let layer_start = layer * per_matrix * FFN_COMPONENTS_PER_LAYER;
        let mmap = &self.ffn_mmap;
        if layer_start + FFN_COMPONENTS_PER_LAYER * per_matrix > mmap.len() {
            return None;
        }
        Some([
            (&mmap[layer_start..layer_start + per_matrix], "Q4_K"),
            (
                &mmap[layer_start + per_matrix..layer_start + 2 * per_matrix],
                "Q4_K",
            ),
            (
                &mmap[layer_start + 2 * per_matrix..layer_start + 3 * per_matrix],
                "Q4_K",
            ),
        ])
    }

    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        if self.use_legacy_q4_mmap {
            return None;
        }
        Some(&self.ffn_mmap)
    }

    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        if self.use_legacy_q4_mmap {
            Some(&self.ffn_mmap)
        } else {
            None
        }
    }

    fn kquant_ffn_layer_once(&self, layer: usize, component: usize) -> Option<Arc<Vec<f32>>> {
        if component >= FFN_COMPONENTS_PER_LAYER {
            return None;
        }
        if self.disable_ffn_cache {
            return None;
        }
        let mut cache = self.ffn_cache.lock().ok()?;
        if let Some(cached) = cache.get(&(layer, component)) {
            return Some(Arc::clone(cached));
        }
        let per_matrix = self.ffn_per_matrix;
        let layer_start = layer * per_matrix * FFN_COMPONENTS_PER_LAYER;
        let comp_start = layer_start + component * per_matrix;
        let comp_end = comp_start + per_matrix;
        if comp_end > self.ffn_mmap.len() {
            return None;
        }
        let bytes = &self.ffn_mmap[comp_start..comp_end];
        // Component-major: gate/up are [intermediate × hidden]; down is
        // [hidden × intermediate]. Element count is the same (`intermediate × hidden`).
        let n_elements = per_matrix / 144 * 256; // Q4_K block: 144 bytes / 256 elements
        let arc = Arc::new(dequantize_q4_k(bytes, n_elements));
        cache.insert((layer, component), Arc::clone(&arc));
        Some(arc)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Build a [`Q4kFixtureIndex`] from `weights`, quantizing every
/// per-layer Q/K/V/O and gate/up/down tensor to Q4_K bytes. Pair with
/// [`larql_models::test_fixtures::make_test_q4k_weights`] (or its
/// SiLU sibling) to satisfy the Q4_K-shape constraint that every
/// dimension be a multiple of `K_QUANT_BLOCK_ELEMS` (256).
///
/// Panics if any expected tensor key is missing or non-contiguous —
/// both are bugs in the calling weight fixture, not user-visible.
pub fn make_q4k_fixture_index(weights: &ModelWeights) -> Q4kFixtureIndex {
    let num_layers = weights.num_layers;
    let arch = &*weights.arch;
    let intermediate = weights.intermediate_size;
    let vocab_size = weights.vocab_size;

    let q4k_for = |key: &str| -> Vec<u8> {
        let tensor = weights
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
        let slice = tensor.as_slice().expect("contiguous row-major");
        quantize_q4_k(slice)
    };

    let mut attn_mmap: Vec<u8> = Vec::new();
    let mut attn_offsets: Vec<[(usize, usize); 4]> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let mut layer_offsets: [(usize, usize); 4] = [(0, 0); 4];
        for (i, key) in [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ]
        .iter()
        .enumerate()
        {
            let bytes = q4k_for(key);
            let offset = attn_mmap.len();
            let length = bytes.len();
            attn_mmap.extend_from_slice(&bytes);
            layer_offsets[i] = (offset, length);
        }
        attn_offsets.push(layer_offsets);
    }

    let mut ffn_mmap: Vec<u8> = Vec::new();
    let mut ffn_per_matrix = 0;
    for layer in 0..num_layers {
        for key in [
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ] {
            let bytes = q4k_for(&key);
            // Pin a single per-matrix size — every component in every
            // layer must produce the same byte length for the
            // contiguous-mmap layout to make sense.
            if ffn_per_matrix == 0 {
                ffn_per_matrix = bytes.len();
            } else {
                assert_eq!(
                    bytes.len(),
                    ffn_per_matrix,
                    "Q4_K per-matrix size drifted across (layer, component)"
                );
            }
            ffn_mmap.extend_from_slice(&bytes);
        }
    }

    Q4kFixtureIndex {
        ffn_mmap,
        ffn_per_matrix,
        attn_mmap,
        attn_offsets,
        ffn_cache: std::sync::Mutex::new(HashMap::new()),
        intermediate,
        vocab_size,
        disable_ffn_cache: false,
        use_legacy_q4_mmap: false,
    }
}

// ── Q4_K_M (mixed-format) fixture ─────────────────────────────────────
//
// A faithful `KvIndex` for the production-default Q4_K_M layout:
//   attention: Q/K/O = Q4_K, V = Q6_K
//   FFN:        gate/up = Q4_K, down = Q6_K
// This mirrors the real `convert quantize q4k` output (gate/up Q4_K, down
// Q6_K is the model-level mixture policy; V Q6_K is the attention-side mix
// Qwen/Llama Q4_K_M extracts produce). It exists so the CUDA resident-hidden
// path (GPU-007/D6) can be parity-tested against the mixed triple the real
// default model ships, not just the uniform-Q4_K synthetic fixture.
//
// The structural difference from [`Q4kFixtureIndex`] is that FFN gate/up and
// down have *different* per-matrix byte sizes (Q4_K = 144 B/super-block,
// Q6_K = 210 B/super-block), so the contiguous FFN mmap is no longer a uniform
// `per_matrix` stride. We track the two sizes explicitly and slice the per-
// layer triple `[gate Q4_K | up Q4_K | down Q6_K]` at the right offsets.
// Attention V is Q6_K while Q/K/O stay Q4_K, so `attn_kquant_layer_data`
// returns a per-component tag ("Q4_K"/"Q6_K") and the per-layer offsets
// already accommodate the V component's different size.

/// `KvIndex` backed by the production-default Q4_K_M mixed quantization:
/// attention Q/K/O = Q4_K, V = Q6_K; FFN gate/up = Q4_K, down = Q6_K. The
/// faithful twin of [`Q4kFixtureIndex`] for the format users actually have.
///
/// Construct via [`make_q4km_fixture_index`].
pub struct Q4kmFixtureIndex {
    /// Concatenated FFN bytes across all layers, laid out per layer as
    /// `[gate Q4_K | up Q4_K | down Q6_K]`. `gate_up_per_matrix` and
    /// `down_per_matrix` give the per-component strides.
    ffn_mmap: Vec<u8>,
    /// Per-matrix byte size for the Q4_K gate/up components
    /// (`Q4_K::packed_matrix_bytes(intermediate, hidden)`).
    gate_up_per_matrix: usize,
    /// Per-matrix byte size for the Q6_K down component
    /// (`Q6_K::packed_matrix_bytes(hidden, intermediate)`).
    down_per_matrix: usize,
    /// Concatenated attention bytes for Q/K/V/O across all layers, laid out as
    /// `[layer 0: Q, K, V, O; layer 1: Q, K, V, O; ...]`. Q/K/O are Q4_K; V is
    /// Q6_K (the production Q4_K_M attention mix).
    attn_mmap: Vec<u8>,
    /// Per-layer (offset, length) pairs for Q/K/V/O in `attn_mmap`. V has a
    /// different length (Q6_K vs Q4_K at the same shape) so the offsets aren't
    /// a fixed stride.
    attn_offsets: Vec<[(usize, usize); 4]>,
    /// Per-(layer, component) dequantised FFN cache populated lazily on first
    /// request through `kquant_ffn_layer_once` (gate/up via Q4_K dequant, down
    /// via Q6_K dequant — format-aware, unlike the uniform-Q4_K fixture).
    ffn_cache: FfnDequantCache,
    /// Intermediate dimension — `num_features` returns this.
    intermediate: usize,
    /// Vocabulary size — `vocab_size` returns this.
    vocab_size: usize,
}

impl KvIndex for Q4kmFixtureIndex {
    fn num_features(&self, _layer: usize) -> usize {
        self.intermediate
    }

    fn attn_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        let offsets = self.attn_offsets.get(layer)?;
        let attn = &self.attn_mmap;
        // Q/K/O are Q4_K; V is Q6_K (production Q4_K_M attention mix).
        Some([
            (&attn[offsets[0].0..offsets[0].0 + offsets[0].1], "Q4_K"),
            (&attn[offsets[1].0..offsets[1].0 + offsets[1].1], "Q4_K"),
            (&attn[offsets[2].0..offsets[2].0 + offsets[2].1], "Q6_K"),
            (&attn[offsets[3].0..offsets[3].0 + offsets[3].1], "Q4_K"),
        ])
    }

    fn interleaved_kquant_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(&[u8], &str); FFN_COMPONENTS_PER_LAYER]> {
        let gu = self.gate_up_per_matrix;
        let dn = self.down_per_matrix;
        let layer_start = layer * (gu * 2 + dn);
        let mmap = &self.ffn_mmap;
        if layer_start + 2 * gu + dn > mmap.len() {
            return None;
        }
        // gate/up Q4_K, down Q6_K — the production Q4_K_M FFN mix.
        Some([
            (&mmap[layer_start..layer_start + gu], "Q4_K"),
            (&mmap[layer_start + gu..layer_start + 2 * gu], "Q4_K"),
            (
                &mmap[layer_start + 2 * gu..layer_start + 2 * gu + dn],
                "Q6_K",
            ),
        ])
    }

    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        Some(&self.ffn_mmap)
    }

    fn kquant_ffn_layer_once(&self, layer: usize, component: usize) -> Option<Arc<Vec<f32>>> {
        use larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
        if component >= FFN_COMPONENTS_PER_LAYER {
            return None;
        }
        let mut cache = self.ffn_cache.lock().ok()?;
        if let Some(cached) = cache.get(&(layer, component)) {
            return Some(Arc::clone(cached));
        }
        let gu = self.gate_up_per_matrix;
        let dn = self.down_per_matrix;
        let layer_start = layer * (gu * 2 + dn);
        // Component-major: gate/up are [intermediate × hidden] (Q4_K, 144 B/sb);
        // down is [hidden × intermediate] (Q6_K, 210 B/sb) but stored transposed
        // as feature-major (intermediate × hidden) for the consumer
        // (`walk_ffn::kquant_ffn_forward_layer` indexes `w_down_t`). Element
        // count is the same for all three (`intermediate × hidden`).
        let (start, end, is_down) = match component {
            0 => (layer_start, layer_start + gu, false),
            1 => (layer_start + gu, layer_start + 2 * gu, false),
            2 => (layer_start + 2 * gu, layer_start + 2 * gu + dn, true),
            _ => return None,
        };
        if end > self.ffn_mmap.len() {
            return None;
        }
        let bytes = &self.ffn_mmap[start..end];
        // Element count per component = intermediate × hidden. Derive from the
        // per-format block geometry (Q4_K: 144 B/256 elems; Q6_K: 210 B/256).
        let bytes_per_sb = if is_down { 210 } else { 144 };
        let n_elements = bytes.len() / bytes_per_sb * K_QUANT_BLOCK_ELEMS;
        let arc = if is_down {
            // `dequantize_q6_k` returns `Result<Vec<f32>, ModelError>`.
            Arc::new(dequantize_q6_k(bytes, n_elements).ok()?)
        } else {
            Arc::new(dequantize_q4_k(bytes, n_elements))
        };
        cache.insert((layer, component), Arc::clone(&arc));
        Some(arc)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Build a [`Q4kmFixtureIndex`] from `weights`, quantizing each per-layer
/// attention Q/K/O to Q4_K and V to Q6_K, and each FFN gate/up to Q4_K and
/// down to Q6_K — the production-default Q4_K_M layout. Pair with
/// [`larql_models::test_fixtures::make_test_q4k_weights_inter`] (or its
/// `_layers` sibling); the shape constraint (every dim a multiple of 256)
/// is the same as the uniform-Q4_K fixture.
pub fn make_q4km_fixture_index(weights: &ModelWeights) -> Q4kmFixtureIndex {
    let num_layers = weights.num_layers;
    let arch = &*weights.arch;
    let intermediate = weights.intermediate_size;
    let vocab_size = weights.vocab_size;
    let hidden = weights.hidden_size;

    let q4k_for = |key: &str| -> Vec<u8> {
        let tensor = weights
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
        let slice = tensor.as_slice().expect("contiguous row-major");
        quantize_q4_k(slice)
    };
    let q6k_for = |key: &str| -> Vec<u8> {
        let tensor = weights
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
        let slice = tensor.as_slice().expect("contiguous row-major");
        quantize_q6_k(slice)
    };

    // Attention: Q/K/O = Q4_K, V = Q6_K (production Q4_K_M mix).
    let mut attn_mmap: Vec<u8> = Vec::new();
    let mut attn_offsets: Vec<[(usize, usize); 4]> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let mut layer_offsets: [(usize, usize); 4] = [(0, 0); 4];
        // Q, K, O are Q4_K; V is Q6_K. The keys come back in Q/K/V/O order but
        // the mmap lays them out Q, K, V, O (the conventional attention order).
        let q = q4k_for(&arch.attn_q_key(layer));
        let k = q4k_for(&arch.attn_k_key(layer));
        let v = q6k_for(&arch.attn_v_key(layer));
        let o = q4k_for(&arch.attn_o_key(layer));
        for (i, bytes) in [q, k, v, o].into_iter().enumerate() {
            let offset = attn_mmap.len();
            let length = bytes.len();
            attn_mmap.extend_from_slice(&bytes);
            layer_offsets[i] = (offset, length);
        }
        attn_offsets.push(layer_offsets);
    }

    // FFN: gate/up = Q4_K, down = Q6_K (production Q4_K_M mix).
    let q4k = crate::QuantFormat::Q4_K;
    let q6k = crate::QuantFormat::Q6_K;
    let gate_up_per_matrix = q4k
        .packed_matrix_bytes(intermediate, hidden)
        .expect("Q4_K gate/up per-matrix bytes");
    let down_per_matrix = q6k
        .packed_matrix_bytes(hidden, intermediate)
        .expect("Q6_K down per-matrix bytes");
    let mut ffn_mmap: Vec<u8> = Vec::new();
    for layer in 0..num_layers {
        let gate = q4k_for(&arch.ffn_gate_key(layer));
        let up = q4k_for(&arch.ffn_up_key(layer));
        let down = q6k_for(&arch.ffn_down_key(layer));
        assert_eq!(
            gate.len(),
            gate_up_per_matrix,
            "Q4_K gate per-matrix size drifted across layers"
        );
        assert_eq!(
            up.len(),
            gate_up_per_matrix,
            "Q4_K up per-matrix size drifted across layers"
        );
        assert_eq!(
            down.len(),
            down_per_matrix,
            "Q6_K down per-matrix size drifted across layers"
        );
        ffn_mmap.extend_from_slice(&gate);
        ffn_mmap.extend_from_slice(&up);
        ffn_mmap.extend_from_slice(&down);
    }

    Q4kmFixtureIndex {
        ffn_mmap,
        gate_up_per_matrix,
        down_per_matrix,
        attn_mmap,
        attn_offsets,
        ffn_cache: std::sync::Mutex::new(HashMap::new()),
        intermediate,
        vocab_size,
    }
}

/// Minimal `ComputeBackend` that overrides the `DecodeBackend` methods
/// `kquant_forward::cached::fused_*` reaches: `supports_quant(Q4_K)`,
/// `prefill_kquant`, `decode_token{,_with_state_dump}`. Each override
/// returns a synthetic zero vector of the right shape so the wrappers
/// can run their post-call shape-and-slice logic without
/// short-circuiting. End-to-end *correctness* of those kernels lives
/// in `MetalBackend` integration tests; this mock exists only to drive
/// coverage of the `kquant_forward` glue code.
pub struct MockKquantBackend;

impl crate::MatMul for MockKquantBackend {
    fn matmul(
        &self,
        _a: ndarray::ArrayView2<f32>,
        _b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        unreachable!("mock MatMul never invoked")
    }
    fn matmul_transb(
        &self,
        _a: ndarray::ArrayView2<f32>,
        _b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        unreachable!("mock MatMul never invoked")
    }
}

impl crate::QuantMatVec for MockKquantBackend {
    fn supports_quant(&self, format: crate::QuantFormat) -> bool {
        matches!(format, crate::QuantFormat::Q4_K)
    }
}

impl crate::DecodeBackend for MockKquantBackend {
    fn prefill_kquant(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
        seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        Some(vec![0.0; seq_len * hidden])
    }

    fn decode_token(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
    ) -> Option<Vec<f32>> {
        Some(vec![0.0; hidden])
    }

    fn decode_token_with_state_dump_masked(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        state: Option<&mut crate::DecodeStateDump>,
        mask: crate::StateDumpMask,
    ) -> Option<Vec<f32>> {
        if let Some(dump) = state {
            let want_kv = matches!(mask, crate::StateDumpMask::Full);
            let want_h = !matches!(mask, crate::StateDumpMask::None);
            for layer in layers {
                if want_h {
                    dump.h_in_per_layer.push(vec![0.0; hidden]);
                }
                if want_kv {
                    let kv_dim = layer.num_kv_heads * layer.head_dim;
                    dump.k_new_per_layer.push(vec![0.0; kv_dim]);
                    dump.v_new_per_layer.push(vec![0.0; kv_dim]);
                }
            }
        }
        self.decode_token(layers, x, hidden, inter)
    }
}

impl crate::ComputeBackend for MockKquantBackend {
    fn name(&self) -> &str {
        "mock-kquant"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn supports(&self, cap: crate::Capability) -> bool {
        matches!(cap, crate::Capability::QuantMatVec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::test_fixtures::make_test_q4k_weights;

    /// Smoke test: build the fixture index from Q4K-friendly weights
    /// and verify every accessor returns sensible data.
    #[test]
    fn fixture_index_returns_per_layer_q4k_slices() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);

        // Trait accessors.
        assert_eq!(idx.num_features(0), weights.intermediate_size);
        assert_eq!(idx.vocab_size(), weights.vocab_size);

        // Per-layer Q4K attention bytes.
        let attn = idx.attn_kquant_layer_data(0).expect("layer 0 attn");
        assert_eq!(attn.len(), 4);
        for (bytes, fmt) in &attn {
            assert_eq!(*fmt, "Q4_K");
            assert!(!bytes.is_empty(), "empty Q4K bytes");
        }

        // Per-layer FFN data slices into the mmap.
        let ffn = idx.interleaved_kquant_layer_data(0).expect("layer 0 ffn");
        assert_eq!(ffn.len(), FFN_COMPONENTS_PER_LAYER);
        let mmap = idx.interleaved_kquant_mmap_ref().expect("mmap");
        assert!(!mmap.is_empty());

        // Out-of-range layer returns None.
        assert!(idx.attn_kquant_layer_data(weights.num_layers).is_none());
        assert!(idx
            .interleaved_kquant_layer_data(weights.num_layers)
            .is_none());

        // Dequantised cache populates on demand.
        let cached0 = idx.kquant_ffn_layer_once(0, 0).expect("layer 0 gate cache");
        assert!(!cached0.is_empty());
        // Second call returns the same Arc.
        let cached0_again = idx.kquant_ffn_layer_once(0, 0).expect("hit cache");
        assert!(Arc::ptr_eq(&cached0, &cached0_again));

        // Out-of-range component returns None.
        assert!(idx.kquant_ffn_layer_once(0, 99).is_none());

        // Legacy Q4_0 mmap not provided — default `None`.
        assert!(idx.interleaved_q4_mmap_ref().is_none());
    }

    #[test]
    fn fixture_drives_fused_prefill_to_some_on_mock_backend() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = MockKquantBackend;
        let result = crate::kquant_forward::fused_prefill(&weights, &idx, &[0u32, 1, 2], &backend);
        let h = result.expect("MockKquantBackend.prefill_kquant returns Some");
        // `fused_prefill` slices to the last position → shape `[1 × hidden]`.
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn fixture_drives_fused_decode_step_to_some_on_mock_backend() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = MockKquantBackend;
        let result = crate::kquant_forward::fused_decode_step(&weights, &idx, 0u32, &backend);
        let h = result.expect("MockKquantBackend.decode_token returns Some");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn fixture_drives_fused_decode_step_with_state_to_some() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = MockKquantBackend;
        let mut dump = crate::DecodeStateDump::with_capacity(weights.num_layers);
        let result = crate::kquant_forward::fused_decode_step_with_state(
            &weights, &idx, 0u32, &backend, &mut dump,
        );
        let h = result.expect("decode_step_with_state returns Some");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // The mock populates per-layer dump entries.
        assert_eq!(dump.h_in_per_layer.len(), weights.num_layers);
    }

    #[test]
    fn fixture_satisfies_fused_prefill_input_gates() {
        use crate::QuantMatVec;
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        assert!(QuantMatVec::supports_quant(
            &backend,
            crate::QuantFormat::Q4_K
        ));
        assert!(idx.interleaved_kquant_mmap_ref().is_some());
        assert!(idx.attn_kquant_layer_data(0).is_some());
        assert!(idx.num_features(0) > 0);
        let result = crate::kquant_forward::fused_prefill(&weights, &idx, &[0u32, 1, 2], &backend);
        // CpuBackend's `prefill_kquant` default returns None, so the
        // chain bottoms out there — but every gate above passes.
        assert!(result.is_none());
    }
}

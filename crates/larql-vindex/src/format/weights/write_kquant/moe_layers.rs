//! Stage 3 — per-layer FFN weights for MoE models (§5.12).
//!
//! Two packed-expert formats are supported, both transcoded to Q4_K per
//! expert and written to `layers/layer_{L:02}.weights` with
//! `num_entries=num_experts`:
//!
//! - **`PackedBF16`** (Gemma 4 26B A4B): source BF16 expert tensors are
//!   quantised to Q4_K via [`quantize_moe_entries`]. Replaces the legacy
//!   ~43 GB BF16 `experts_packed.bin` monolithic blob.
//!
//! - **`PackedMxfp4`** (GPT-OSS / DeepSeek-V4): source MXFP4 nibble +
//!   e8m0-scale expert tensors are dequantized to f32 per expert (via
//!   [`larql_models::quant::mxfp4`]) and then quantized to Q4_K via
//!   [`quantize_moe_entries_f32`]. Native MXFP4 matvec kernels do not
//!   exist on CUDA yet, so transcoding to Q4_K at extract reuses the
//!   mature Q4_K read/decode path end-to-end. Before this arm existed,
//!   `inference`/`all` extracts silently dropped every expert FFN weight
//!   for GPT-OSS because the per-expert keys return `None` (packed
//!   format) and the dense `get_tensor` lookups found nothing (GPU-5002).
//!
//! For dense models: this is a no-op — `interleaved_kquant.bin` (stage 2)
//! remains the primary FFN store. Per-layer format for dense is a
//! future migration (`--ffn-layout` flag).

use std::path::Path;

use crate::error::VindexError;

use super::super::write_f32::WeightSource;
use super::super::write_layers::{
    quantize_moe_entries, quantize_moe_entries_f32, write_layer_weights, LayerWeightFormat,
};

pub(super) fn write_per_layer_moe_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
) -> Result<(), VindexError> {
    let arch = source.arch();
    let expert_format = arch.expert_format();

    let is_packed_bf16 = arch.is_hybrid_moe() && expert_format == larql_models::ExpertFormat::PackedBF16;
    let is_packed_mxfp4 = arch.is_moe() && expert_format == larql_models::ExpertFormat::PackedMxfp4;
    if !(is_packed_bf16 || is_packed_mxfp4) {
        return Ok(());
    }

    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    let hidden = arch.config().hidden_size;

    for layer in 0..num_layers {
        if is_packed_bf16 {
            write_packed_bf16_layer(source, dir, layer, num_experts, moe_inter, hidden)?;
        } else {
            write_packed_mxfp4_layer(source, dir, layer, num_experts, moe_inter, hidden)?;
        }
    }
    Ok(())
}

/// PackedBF16 path (Gemma 4 26B A4B): read raw BF16 expert tensors and
/// quantize to Q4_K via the BF16-aware [`quantize_moe_entries`].
fn write_packed_bf16_layer(
    source: &dyn WeightSource,
    dir: &Path,
    layer: usize,
    num_experts: usize,
    moe_inter: usize,
    hidden: usize,
) -> Result<(), VindexError> {
    let arch = source.arch();
    let gu_key = arch.packed_experts_gate_up_key(layer);
    let dn_key = arch.packed_experts_down_key(layer);
    let gu_bytes = gu_key.as_ref().and_then(|k| source.get_packed_bf16(k));
    let dn_bytes = dn_key.as_ref().and_then(|k| source.get_packed_bf16(k));

    if let (Some(gu), Some(dn)) = (gu_bytes, dn_bytes) {
        // Default: Q4_K for the whole file. Format is uniform — no mixing.
        let fmt = LayerWeightFormat::Q4_K;
        let entries = quantize_moe_entries(&gu, &dn, num_experts, moe_inter, hidden, fmt)?;
        write_layer_weights(dir, layer, fmt, &entries, moe_inter, hidden)?;
    }
    Ok(())
}

/// PackedMXFP4 path (GPT-OSS / DeepSeek-V4): read raw MXFP4 nibble + e8m0
/// scale tensors, dequantize to f32 per expert, then quantize to Q4_K via
/// [`quantize_moe_entries_f32`].
///
/// The fused gate_up tensor (`[num_experts, 2*inter, groups, 16]`) is
/// dequantized as-is; the resulting per-expert f32 matrix is already in
/// the `[2*inter, hidden]` row-major layout the Q4_K writer expects
/// (gate rows then up rows). The down tensor
/// (`[num_experts, hidden, groups_down, 16]`) dequantizes to
/// `[hidden, inter]` per expert.
fn write_packed_mxfp4_layer(
    source: &dyn WeightSource,
    dir: &Path,
    layer: usize,
    num_experts: usize,
    moe_inter: usize,
    hidden: usize,
) -> Result<(), VindexError> {
    let arch = source.arch();
    let gu_blocks_key = match arch.packed_gate_up_blocks_key(layer) {
        Some(k) => k,
        None => return Ok(()),
    };
    let dn_blocks_key = match arch.packed_down_blocks_key(layer) {
        Some(k) => k,
        None => return Ok(()),
    };

    let gu_packed = match source.get_packed_mxfp4(&gu_blocks_key) {
        Some(p) => p,
        None => return Ok(()), // GGUF or missing — silently skip (matches BF16 arm)
    };
    let dn_packed = match source.get_packed_mxfp4(&dn_blocks_key) {
        Some(p) => p,
        None => return Ok(()),
    };

    // Dequantize gate_up (fused) and down per expert. Both reuse the
    // proven in-memory f32 dequant reference (loading/safetensors.rs
    // calls the same functions) — this is the validation anchor.
    let gate_up_experts = larql_models::quant::mxfp4::dequantize_all_experts(
        &gu_packed.blocks,
        &gu_packed.scales,
        gu_packed.num_experts,
        gu_packed.out_features,
        gu_packed.groups,
    )
    .map_err(|e| VindexError::Parse(format!("MXFP4 gate_up dequant (layer {layer}): {e}")))?;

    let down_experts = larql_models::quant::mxfp4::dequantize_all_experts(
        &dn_packed.blocks,
        &dn_packed.scales,
        dn_packed.num_experts,
        dn_packed.out_features,
        dn_packed.groups,
    )
    .map_err(|e| VindexError::Parse(format!("MXFP4 down dequant (layer {layer}): {e}")))?;

    // The gate_up tensor's out_features is 2*intermediate (fused gate+up);
    // the dequantized per-expert matrix is already [2*inter, hidden].
    // Sanity-check the dims before handing off — a mismatched fixture
    // would silently write garbage under a Q4_K header.
    let expected_gu_eles = 2 * moe_inter * hidden;
    let expected_dn_eles = hidden * moe_inter;
    if gate_up_experts.len() != num_experts || down_experts.len() != num_experts {
        return Err(VindexError::Parse(format!(
            "MXFP4 layer {layer}: expert count mismatch (gate_up={}, down={}, expected={num_experts})",
            gate_up_experts.len(),
            down_experts.len(),
        )));
    }
    for (e, (gu, dn)) in gate_up_experts.iter().zip(down_experts.iter()).enumerate() {
        if gu.len() != expected_gu_eles {
            return Err(VindexError::Parse(format!(
                "MXFP4 layer {layer} expert {e}: gate_up dequant len {} != expected {expected_gu_eles} (2*{moe_inter}*{hidden})",
                gu.len()
            )));
        }
        if dn.len() != expected_dn_eles {
            return Err(VindexError::Parse(format!(
                "MXFP4 layer {layer} expert {e}: down dequant len {} != expected {expected_dn_eles} ({hidden}*{moe_inter})",
                dn.len()
            )));
        }
    }

    let fmt = LayerWeightFormat::Q4_K;
    let entries = quantize_moe_entries_f32(
        &gate_up_experts,
        &down_experts,
        moe_inter,
        hidden,
        fmt,
    )?;
    write_layer_weights(dir, layer, fmt, &entries, moe_inter, hidden)?;
    Ok(())
}

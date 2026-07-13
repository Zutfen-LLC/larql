//! FP8 E4M3 + UE8M0 block-scale dequantization (DeepSeek-V4-Flash attention).
//!
//! V4-Flash attention weights (wq_a, wq_b, wkv, wo_a, wo_b) and some
//! projections (indexer.wq_b, mtp e_proj/h_proj) are stored as FP8 E4M3
//! with per-128×128 block UE8M0 scales.
//!
//! Weight tensor: `[rows, cols]` as `F8_E4M3` (1 byte per element).
//! Scale tensor:  `[ceil(rows/128), ceil(cols/128)]` as `F8_E8M0` (1 byte per block).
//!
//! Each scale byte is an E8M0 exponent: value = 2^(byte − 127).
//! Dequantized element = e4m3_decode(weight_byte) × e8m0_decode(scale_byte).
//!
//! This is distinct from:
//! - LARQL's own FP4/FP8 block codec (`fp4_block.rs` — 137B/257B blocks with E4M3 sub-scales)
//! - MXFP4 expert format (`mxfp4.rs` — E2M1 packed + E8M0 per-32)

use super::fp8::e4m3_to_f32;
use super::mxfp4::e8m0_to_f32;
use crate::detect::ModelError;

/// Block size for V4-Flash FP8 attention weights.
pub const FP8_BLOCK_SIZE: usize = 128;

/// Dequantize a 2-D FP8 E4M3 weight tensor with per-128×128 UE8M0 block scales.
///
/// `weights` must contain at least `rows × cols` bytes (F8_E4M3, row-major).
/// `scales` must contain at least `ceil(rows/128) × ceil(cols/128)` bytes (E8M0).
///
/// Returns a flat `Vec<f32>` of length `rows × cols` in row-major order.
pub fn dequantize(
    weights: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, ModelError> {
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }

    let num_scale_rows = rows.div_ceil(FP8_BLOCK_SIZE);
    let num_scale_cols = cols.div_ceil(FP8_BLOCK_SIZE);

    let need_weights = rows.checked_mul(cols).ok_or_else(|| {
        ModelError::Parse(format!("FP8 block: weights size overflow ({rows}×{cols})"))
    })?;
    let need_scales = num_scale_rows.checked_mul(num_scale_cols).ok_or_else(|| {
        ModelError::Parse(format!(
            "FP8 block: scales size overflow ({num_scale_rows}×{num_scale_cols})"
        ))
    })?;

    if weights.len() < need_weights {
        return Err(ModelError::Parse(format!(
            "FP8 block: weights too short: {} bytes < expected {need_weights} ({rows}×{cols})",
            weights.len()
        )));
    }
    if scales.len() < need_scales {
        return Err(ModelError::Parse(format!(
            "FP8 block: scales too short: {} bytes < expected {need_scales} ({num_scale_rows}×{num_scale_cols})",
            scales.len()
        )));
    }

    let mut output = vec![0.0f32; rows * cols];

    for sr in 0..num_scale_rows {
        let row_start = sr * FP8_BLOCK_SIZE;
        let row_end = ((sr + 1) * FP8_BLOCK_SIZE).min(rows);

        for sc in 0..num_scale_cols {
            let scale_byte = scales[sr * num_scale_cols + sc];
            let scale = e8m0_to_f32(scale_byte);

            let col_start = sc * FP8_BLOCK_SIZE;
            let col_end = ((sc + 1) * FP8_BLOCK_SIZE).min(cols);

            for r in row_start..row_end {
                let row_offset = r * cols;
                for c in col_start..col_end {
                    let idx = row_offset + c;
                    output[idx] = e4m3_to_f32(weights[idx]) * scale;
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic dequantization ───────────────────────────────────────────────

    #[test]
    fn dequant_single_block_all_ones() {
        // 128×128 block, all weights = E4M3 for 1.0 (0x38), scale = E8M0 for 1.0 (127)
        let weights = vec![0x38u8; 128 * 128];
        let scales = vec![127u8];
        let result = dequantize(&weights, &scales, 128, 128).unwrap();
        assert_eq!(result.len(), 128 * 128);
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn dequant_single_block_with_scale() {
        // Scale = 128 → 2^(128-127) = 2.0
        let weights = vec![0x38u8; 128 * 128]; // 1.0 each
        let scales = vec![128u8]; // scale = 2.0
        let result = dequantize(&weights, &scales, 128, 128).unwrap();
        for &v in &result {
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn dequant_negative_weights() {
        // 0xB8 = sign bit set + 1.0 → -1.0
        let weights = vec![0xB8u8; 128 * 128];
        let scales = vec![127u8]; // scale = 1.0
        let result = dequantize(&weights, &scales, 128, 128).unwrap();
        for &v in &result {
            assert!((v - (-1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn dequant_zero_scale() {
        // E8M0 byte 0 → 0.0 scale. Use 0x38 (1.0) weight, not 0xFF (NaN).
        let weights = vec![0x38u8; 128 * 128];
        let scales = vec![0u8]; // E8M0 byte 0 → 0.0
        let result = dequantize(&weights, &scales, 128, 128).unwrap();
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    // ── Multi-block layouts ─────────────────────────────────────────────────

    #[test]
    fn dequant_two_blocks_different_scales() {
        // 256×128 → 2 scale blocks (2 rows × 1 col)
        // Block 0 (rows 0-127): scale 1.0, weight 1.0
        // Block 1 (rows 128-255): scale 2.0, weight 1.0
        let weights = vec![0x38u8; 256 * 128];
        let scales = vec![127u8, 128u8]; // [1.0, 2.0]
        let result = dequantize(&weights, &scales, 256, 128).unwrap();
        // First 128 rows should be 1.0
        for r in 0..128 {
            for c in 0..128 {
                assert!(
                    (result[r * 128 + c] - 1.0).abs() < 1e-6,
                    "block 0 at ({r},{c})"
                );
            }
        }
        // Next 128 rows should be 2.0
        for r in 128..256 {
            for c in 0..128 {
                assert!(
                    (result[r * 128 + c] - 2.0).abs() < 1e-6,
                    "block 1 at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn dequant_two_column_blocks() {
        // 128×256 → 1×2 scale blocks
        let weights = vec![0x38u8; 128 * 256];
        let scales = vec![127u8, 128u8]; // [1.0, 2.0]
        let result = dequantize(&weights, &scales, 128, 256).unwrap();
        for r in 0..128 {
            for c in 0..128 {
                assert!(
                    (result[r * 256 + c] - 1.0).abs() < 1e-6,
                    "col block 0 at ({r},{c})"
                );
            }
            for c in 128..256 {
                assert!(
                    (result[r * 256 + c] - 2.0).abs() < 1e-6,
                    "col block 1 at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn dequant_partial_last_block() {
        // 130×130 → ceil(130/128)=2 scale rows × 2 scale cols = 4 scales
        // The last row/col block is only 2 elements wide/tall.
        let rows = 130;
        let cols = 130;
        let weights = vec![0x38u8; rows * cols];
        let scales = vec![127u8, 128u8, 129u8, 130u8]; // 2×2
        let result = dequantize(&weights, &scales, rows, cols).unwrap();
        assert_eq!(result.len(), rows * cols);

        // Element at (0, 0): block (0,0), scale 127 → 1.0
        assert!((result[0] - 1.0).abs() < 1e-6);
        // Element at (0, 128): block (0,1), scale 128 → 2.0
        assert!((result[128] - 2.0).abs() < 1e-6);
        // Element at (128, 0): block (1,0), scale 129 → 4.0
        assert!((result[128 * 130] - 4.0).abs() < 1e-6);
        // Element at (128, 128): block (1,1), scale 130 → 8.0
        assert!((result[128 * 130 + 128] - 8.0).abs() < 1e-6);
        // Element at (129, 129): block (1,1), scale 130 → 8.0
        assert!((result[129 * 130 + 129] - 8.0).abs() < 1e-6);
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn dequant_empty() {
        let result = dequantize(&[], &[], 0, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn dequant_single_element() {
        // 1×1 → 1 scale block
        let weights = vec![0x38u8]; // 1.0
        let scales = vec![127u8]; // scale 1.0
        let result = dequantize(&weights, &scales, 1, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dequant_rejects_short_weights() {
        match dequantize(&[0u8; 10], &[127], 128, 128) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("weights too short"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn dequant_rejects_short_scales() {
        // Need 4 scales for 256×256, give 2
        match dequantize(&[0u8; 256 * 256], &[127, 128], 256, 256) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("scales too short"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn dequant_rejects_size_overflow() {
        let big = usize::MAX / 2 + 1;
        match dequantize(&[], &[], big, big) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("overflow"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    // ── Realistic V4-Flash attention dimensions ─────────────────────────────

    #[test]
    fn dequant_v4_wq_a_dimensions() {
        // wq_a: [hidden_size, q_lora_rank] = [4096, 1024]
        // Scale: [32, 8] = 256 scale bytes
        let rows = 4096;
        let cols = 1024;
        let weights = vec![0x38u8; rows * cols]; // all 1.0
        let scales = vec![127u8; 32 * 8]; // all scale 1.0
        let result = dequantize(&weights, &scales, rows, cols).unwrap();
        assert_eq!(result.len(), rows * cols);
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn dequant_v4_wkv_dimensions() {
        // wkv: [hidden_size, head_dim] = [4096, 512]
        // Scale: [32, 4] = 128 scale bytes
        let rows = 4096;
        let cols = 512;
        let weights = vec![0x38u8; rows * cols];
        let scales = vec![127u8; 32 * 4];
        let result = dequantize(&weights, &scales, rows, cols).unwrap();
        assert_eq!(result.len(), rows * cols);
    }

    #[test]
    fn dequant_v4_wq_b_dimensions() {
        // wq_b: [q_lora_rank, num_heads * head_dim] = [1024, 32768]
        // Scale: [8, 256] = 2048 scale bytes
        let rows = 1024;
        let cols = 32768;
        let weights = vec![0x38u8; rows * cols];
        let scales = vec![127u8; 8 * 256];
        let result = dequantize(&weights, &scales, rows, cols).unwrap();
        assert_eq!(result.len(), rows * cols);
    }
}

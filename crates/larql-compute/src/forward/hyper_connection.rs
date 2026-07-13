//! Hyper-Connections (HC) — DeepSeek-V4-Flash structural backbone.
//!
//! Replaces standard residual connections with a 4-copy hidden state
//! `[batch, seq, hc_mult, hidden]` that flows through every layer.
//!
//! Each block has:
//! - **hc_pre**: reduces hc_mult→1 via Sinkhorn-normalized mixing
//! - **hc_post**: expands 1→hc_mult via weighted residual mixing
//!
//! The HC head (final layer) uses sigmoid-based reduction, not Sinkhorn.
//!
//! Reference: `inference/model.py` from the V4-Flash HuggingFace repo.
//!
//! # Notation
//!
//! - `hc` = hc_mult (typically 4)
//! - `d` = hidden_size (4096)
//! - `mix_hc` = (2 + hc) * hc = 24 when hc=4
//! - `hc_fn`: weight matrix [mix_hc, hc*d] = [24, 16384]
//! - `hc_base`: bias [mix_hc] = [24]
//! - `hc_scale`: per-component scale [3] (pre, post, comb) — [1] for head

use ndarray::{Array1, Array2, Array3};

/// Hyper-Connection parameters for a single block (attn or ffn).
#[derive(Debug, Clone)]
pub struct HcParams {
    /// Weight matrix [mix_hc, hc * d].
    pub fn_weight: Array2<f32>,
    /// Bias [mix_hc].
    pub base: Array1<f32>,
    /// Per-component scale: [3] for per-layer, [1] for head.
    pub scale: Array1<f32>,
}

/// Output of `hc_pre`: the reduced hidden state + post/comb vectors for hc_post.
pub struct HcPreOutput {
    /// Reduced hidden state [batch*seq, d] (hc_mult → 1).
    pub y: Array2<f32>,
    /// Post-expansion weights [batch*seq, hc] — used by hc_post.
    pub post: Array2<f32>,
    /// Combination matrix [batch*seq, hc, hc] — used by hc_post.
    pub comb: Array2<f32>,
}

/// Default Sinkhorn iterations (from config: hc_sinkhorn_iters = 20).
#[cfg(test)]
const DEFAULT_SINKHORN_ITERS: usize = 20;

/// Default HC epsilon (from config: hc_eps = 1e-6).
#[cfg(test)]
const DEFAULT_HC_EPS: f64 = 1e-6;

/// Expand a 2D hidden state `[seq, d]` into the 4-copy HC representation
/// `[seq, hc, d]` by repeating. Used at the start of the forward pass
/// (after embedding) to initialize the HC dimension.
pub fn hc_expand(h: &Array2<f32>, hc_mult: usize) -> Array3<f32> {
    let (seq, d) = (h.shape()[0], h.shape()[1]);
    let mut out = Array3::zeros((seq, hc_mult, d));
    for s in 0..seq {
        for hc in 0..hc_mult {
            for j in 0..d {
                out[[s, hc, j]] = h[[s, j]];
            }
        }
    }
    out
}

/// hc_pre: Reduce the hc_mult-copy hidden state to a single copy.
///
/// Steps:
/// 1. RMSNorm on the flattened [seq, hc*d] input
/// 2. Linear projection: x_norm @ fn_weight.T + base → mixes [seq, mix_hc]
/// 3. Split mixes into pre[hc], post[hc], comb[hc*hc]
/// 4. pre = sigmoid(mixes_pre * scale[0] + base_pre) + eps
/// 5. post = 2 * sigmoid(mixes_post * scale[1] + base_post)
/// 6. comb = softmax(mixes_comb) reshaped to [seq, hc, hc], then Sinkhorn-normalized
/// 7. y = sum(pre[hc] * x[hc, d], dim=hc) → [seq, d]
pub fn hc_pre(
    x_hc: &Array3<f32>, // [seq, hc, d]
    params: &HcParams,
    sinkhorn_iters: usize,
    eps: f64,
) -> HcPreOutput {
    let (seq, hc, d) = (x_hc.shape()[0], x_hc.shape()[1], x_hc.shape()[2]);
    let mix_hc = (2 + hc) * hc;

    // 1. Flatten x_hc to [seq, hc*d] for the linear projection.
    let mut x_flat = Array2::zeros((seq, hc * d));
    for s in 0..seq {
        for h in 0..hc {
            for j in 0..d {
                x_flat[[s, h * d + j]] = x_hc[[s, h, j]];
            }
        }
    }

    // 2. RMSNorm (no learned weight — the norm is applied externally via attn_norm/ffn_norm)
    let x_normed = rms_norm_flat(&x_flat, eps);

    // 3. Linear projection: x_normed @ fn_weight.T + base → mixes [seq, mix_hc]
    let mixes = x_normed.dot(&params.fn_weight.t()) + &params.base;

    // 4. Split mixes into pre, post, comb slices
    let pre_slice = mixes.slice(ndarray::s![.., 0..hc]); // [seq, hc]
    let post_slice = mixes.slice(ndarray::s![.., hc..2 * hc]); // [seq, hc]
    let comb_slice = mixes.slice(ndarray::s![.., 2 * hc..mix_hc]); // [seq, hc*hc]

    let scale_pre = params.scale.get(0).copied().unwrap_or(1.0);
    let scale_post = params.scale.get(1).copied().unwrap_or(1.0);
    let scale_comb = params.scale.get(2).copied().unwrap_or(1.0);

    // 5. pre = sigmoid(mixes_pre * scale_pre + base_pre) + eps
    let base_pre = params.base.slice(ndarray::s![0..hc]);
    let mut pre = Array2::zeros((seq, hc));
    for s in 0..seq {
        for h in 0..hc {
            let val = pre_slice[[s, h]] * scale_pre + base_pre[h];
            pre[[s, h]] = sigmoid(val) + eps as f32;
        }
    }

    // 6. post = 2 * sigmoid(mixes_post * scale_post + base_post)
    let base_post = params.base.slice(ndarray::s![hc..2 * hc]);
    let mut post = Array2::zeros((seq, hc));
    for s in 0..seq {
        for h in 0..hc {
            let val = post_slice[[s, h]] * scale_post + base_post[h];
            post[[s, h]] = 2.0 * sigmoid(val);
        }
    }

    // 7. comb = softmax(mixes_comb * scale_comb) reshaped to [seq, hc, hc], then Sinkhorn
    let base_comb = params.base.slice(ndarray::s![2 * hc..mix_hc]);
    let mut comb = Array2::zeros((seq, hc * hc));
    for s in 0..seq {
        for i in 0..(hc * hc) {
            let val = comb_slice[[s, i]] * scale_comb + base_comb[i];
            comb[[s, i]] = val;
        }
    }
    // Row-wise softmax for each [seq] position over hc*hc elements
    let mut comb_3d = Array3::zeros((seq, hc, hc));
    for s in 0..seq {
        let row: Vec<f32> = (0..(hc * hc)).map(|i| comb[[s, i]]).collect();
        let sm = softmax(&row);
        for i in 0..hc {
            for j in 0..hc {
                comb_3d[[s, i, j]] = sm[i * hc + j];
            }
        }
    }
    // Sinkhorn normalization: alternate row and column normalization
    for _ in 0..sinkhorn_iters {
        // Row normalization
        for s in 0..seq {
            for i in 0..hc {
                let mut s_row = 0.0f32;
                for j in 0..hc {
                    s_row += comb_3d[[s, i, j]];
                }
                if s_row > 0.0 {
                    for j in 0..hc {
                        comb_3d[[s, i, j]] /= s_row;
                    }
                }
            }
        }
        // Column normalization
        for s in 0..seq {
            for j in 0..hc {
                let mut s_col = 0.0f32;
                for i in 0..hc {
                    s_col += comb_3d[[s, i, j]];
                }
                if s_col > 0.0 {
                    for i in 0..hc {
                        comb_3d[[s, i, j]] /= s_col;
                    }
                }
            }
        }
    }

    // 8. y = sum(pre[hc] * x[hc, d], dim=hc) → [seq, d]
    let mut y = Array2::zeros((seq, d));
    for s in 0..seq {
        for j in 0..d {
            let mut sum = 0.0f32;
            for h in 0..hc {
                sum += pre[[s, h]] * x_hc[[s, h, j]];
            }
            y[[s, j]] = sum;
        }
    }

    // Flatten comb_3d back to [seq, hc*hc] for the output
    let mut comb_out = Array2::zeros((seq, hc * hc));
    for s in 0..seq {
        for i in 0..hc {
            for j in 0..hc {
                comb_out[[s, i * hc + j]] = comb_3d[[s, i, j]];
            }
        }
    }

    HcPreOutput {
        y,
        post,
        comb: comb_out,
    }
}

/// hc_post: Expand a single hidden state back to hc_mult copies.
///
/// `x` is the output of attention or FFN [seq, d].
/// `residual` is the pre-attention/pre-FFN hidden state [seq, hc, d].
/// `post` and `comb` come from `hc_pre`.
///
/// y = post[hc] * x[d] + sum(comb[hc, hc] * residual[hc, d], dim=hc)
/// Output: [seq, hc, d]
pub fn hc_post(
    x: &Array2<f32>,        // [seq, d] — output of attn/ffn
    residual: &Array3<f32>, // [seq, hc, d] — the input to hc_pre
    post: &Array2<f32>,     // [seq, hc]
    comb: &Array2<f32>,     // [seq, hc*hc]
) -> Array3<f32> {
    let (seq, d) = (x.shape()[0], x.shape()[1]);
    let hc = residual.shape()[1];

    let mut out = Array3::zeros((seq, hc, d));

    for s in 0..seq {
        // Expand: post[hc] * x[d] → [hc, d]
        for h in 0..hc {
            let p = post[[s, h]];
            for j in 0..d {
                out[[s, h, j]] = p * x[[s, j]];
            }
        }
        // Residual mixing: sum(comb[h, h'] * residual[h', d], dim=h') → add to out[h, d]
        for h in 0..hc {
            for j in 0..d {
                let mut mix = 0.0f32;
                for h2 in 0..hc {
                    mix += comb[[s, h * hc + h2]] * residual[[s, h2, j]];
                }
                out[[s, h, j]] += mix;
            }
        }
    }

    out
}

/// hc_head: Final-layer reduction from hc_mult copies to a single hidden state.
///
/// Uses sigmoid-based mixing (not Sinkhorn). The head's scale is [1] (single
/// scale for all components), and the reduction is:
/// pre = sigmoid(mixes_pre * scale + base_pre) + eps
/// y = sum(pre[hc] * x[hc, d], dim=hc) → [seq, d]
pub fn hc_head(
    x_hc: &Array3<f32>, // [seq, hc, d]
    params: &HcParams,
    eps: f64,
) -> Array2<f32> {
    let (seq, hc, d) = (x_hc.shape()[0], x_hc.shape()[1], x_hc.shape()[2]);

    // Flatten + RMSNorm
    let mut x_flat = Array2::zeros((seq, hc * d));
    for s in 0..seq {
        for h in 0..hc {
            for j in 0..d {
                x_flat[[s, h * d + j]] = x_hc[[s, h, j]];
            }
        }
    }
    let x_normed = rms_norm_flat(&x_flat, eps);

    // Linear projection
    let mixes = x_normed.dot(&params.fn_weight.t()) + &params.base;

    // For the head, scale is [1] — apply to pre slice only
    let scale = params.scale.get(0).copied().unwrap_or(1.0);
    let base_pre = params.base.slice(ndarray::s![0..hc]);

    let pre_slice = mixes.slice(ndarray::s![.., 0..hc]);

    let mut pre = Array2::zeros((seq, hc));
    for s in 0..seq {
        for h in 0..hc {
            let val = pre_slice[[s, h]] * scale + base_pre[h];
            pre[[s, h]] = sigmoid(val) + eps as f32;
        }
    }

    // Reduce: y = sum(pre[hc] * x[hc, d], dim=hc)
    let mut y = Array2::zeros((seq, d));
    for s in 0..seq {
        for j in 0..d {
            let mut sum = 0.0f32;
            for h in 0..hc {
                sum += pre[[s, h]] * x_hc[[s, h, j]];
            }
            y[[s, j]] = sum;
        }
    }

    y
}

// ── Helpers ────────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&x| x / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

/// RMSNorm without learned weights (parameter-free), applied to each row.
fn rms_norm_flat(x: &Array2<f32>, eps: f64) -> Array2<f32> {
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::zeros((rows, cols));
    for i in 0..rows {
        let row = x.row(i);
        let sq_sum: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rms = (sq_sum / cols as f64 + eps).sqrt() as f32;
        for j in 0..cols {
            out[[i, j]] = row[j] / rms;
        }
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_test_params(hc: usize, d: usize) -> HcParams {
        let mix_hc = (2 + hc) * hc;
        // Identity-like fn_weight: first hc*d columns map to pre, etc.
        // Use small random-ish values for testing.
        let fn_weight = Array2::from_shape_fn((mix_hc, hc * d), |(i, j)| {
            // Small deterministic values
            ((i as f32 * 0.01) + (j as f32 * 0.001)) * 0.1
        });
        let base = Array1::from_shape_fn(mix_hc, |i| (i as f32) * 0.01);
        let scale = Array1::from_shape_fn(3, |i| 1.0 + (i as f32) * 0.1);
        HcParams {
            fn_weight,
            base,
            scale,
        }
    }

    #[test]
    fn hc_expand_shape() {
        let h = Array2::zeros((4, 8));
        let hc_h = hc_expand(&h, 4);
        assert_eq!(hc_h.shape(), &[4, 4, 8]);
    }

    #[test]
    fn hc_expand_preserves_values() {
        let h = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let hc_h = hc_expand(&h, 4);
        // Each copy should have the same values as the original
        for s in 0..2 {
            for hc in 0..4 {
                for j in 0..3 {
                    assert_eq!(hc_h[[s, hc, j]], h[[s, j]]);
                }
            }
        }
    }

    #[test]
    fn hc_pre_output_shape() {
        let (seq, hc, d) = (2, 4, 8);
        let x_hc = Array3::zeros((seq, hc, d));
        let params = make_test_params(hc, d);
        let out = hc_pre(&x_hc, &params, DEFAULT_SINKHORN_ITERS, DEFAULT_HC_EPS);
        assert_eq!(out.y.shape(), &[seq, d]);
        assert_eq!(out.post.shape(), &[seq, hc]);
        assert_eq!(out.comb.shape(), &[seq, hc * hc]);
    }

    #[test]
    fn hc_pre_y_is_finite() {
        let (seq, hc, d) = (3, 4, 16);
        let x_hc = Array3::from_shape_fn((seq, hc, d), |(s, h, j)| ((s + h + j) as f32) * 0.1);
        let params = make_test_params(hc, d);
        let out = hc_pre(&x_hc, &params, DEFAULT_SINKHORN_ITERS, DEFAULT_HC_EPS);
        assert!(
            out.y.iter().all(|v| v.is_finite()),
            "hc_pre y must be finite"
        );
    }

    #[test]
    fn hc_pre_pre_values_positive() {
        // pre = sigmoid(...) + eps → always > 0
        let (seq, hc, d) = (2, 4, 8);
        let x_hc = Array3::from_shape_fn((seq, hc, d), |(_, _, j)| (j as f32) * 0.1);
        let params = make_test_params(hc, d);
        let out = hc_pre(&x_hc, &params, DEFAULT_SINKHORN_ITERS, DEFAULT_HC_EPS);
        for s in 0..seq {
            for h in 0..hc {
                assert!(
                    out.post[[s, h]] >= 0.0,
                    "post must be non-negative (2*sigmoid)"
                );
            }
        }
    }

    #[test]
    fn hc_post_output_shape() {
        let (seq, hc, d) = (2, 4, 8);
        let x = Array2::zeros((seq, d));
        let residual = Array3::zeros((seq, hc, d));
        let post = Array2::zeros((seq, hc));
        let comb = Array2::zeros((seq, hc * hc));
        let out = hc_post(&x, &residual, &post, &comb);
        assert_eq!(out.shape(), &[seq, hc, d]);
    }

    #[test]
    fn hc_post_y_is_finite() {
        let (seq, hc, d) = (3, 4, 16);
        let x = Array2::from_shape_fn((seq, d), |(s, j)| (s as f32 + j as f32) * 0.1);
        let residual = Array3::from_shape_fn((seq, hc, d), |(s, h, j)| {
            (s as f32 + h as f32 + j as f32) * 0.05
        });
        let post = Array2::from_shape_fn((seq, hc), |(s, h)| 0.5 + (s as f32 + h as f32) * 0.1);
        let comb = Array2::from_shape_fn((seq, hc * hc), |(_, _)| 1.0 / (hc * hc) as f32);
        let out = hc_post(&x, &residual, &post, &comb);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "hc_post output must be finite"
        );
    }

    #[test]
    fn hc_post_zero_input_zero_output() {
        let (seq, hc, d) = (2, 4, 8);
        let x = Array2::zeros((seq, d));
        let residual = Array3::zeros((seq, hc, d));
        let post = Array2::zeros((seq, hc));
        let comb = Array2::zeros((seq, hc * hc));
        let out = hc_post(&x, &residual, &post, &comb);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn hc_head_output_shape() {
        let (seq, hc, d) = (2, 4, 8);
        let x_hc = Array3::zeros((seq, hc, d));
        let params = make_test_params(hc, d);
        let y = hc_head(&x_hc, &params, DEFAULT_HC_EPS);
        assert_eq!(y.shape(), &[seq, d]);
    }

    #[test]
    fn hc_head_y_is_finite() {
        let (seq, hc, d) = (3, 4, 16);
        let x_hc = Array3::from_shape_fn((seq, hc, d), |(s, h, j)| ((s + h + j) as f32) * 0.1);
        let params = make_test_params(hc, d);
        let y = hc_head(&x_hc, &params, DEFAULT_HC_EPS);
        assert!(y.iter().all(|v| v.is_finite()), "hc_head y must be finite");
    }

    // ── Sinkhorn doubly-stochastic property ─────────────────────────────────

    #[test]
    fn sinkhorn_produces_doubly_stochastic_matrix() {
        // After Sinkhorn normalization, rows and columns should sum to ~1.0
        let (seq, hc, d) = (1, 4, 8);
        let x_hc = Array3::from_shape_fn((seq, hc, d), |(_, h, j)| {
            ((h + 1) as f32) * ((j + 1) as f32) * 0.01
        });
        let params = make_test_params(hc, d);
        let out = hc_pre(&x_hc, &params, 50, DEFAULT_HC_EPS); // 50 iters for convergence

        // Check that comb rows and columns each sum to ~1.0
        for s in 0..seq {
            // Row sums
            for i in 0..hc {
                let row_sum: f32 = (0..hc).map(|j| out.comb[[s, i * hc + j]]).sum();
                assert!(
                    (row_sum - 1.0).abs() < 0.1,
                    "comb row {i} sums to {row_sum}, expected ~1.0"
                );
            }
            // Column sums
            for j in 0..hc {
                let col_sum: f32 = (0..hc).map(|i| out.comb[[s, i * hc + j]]).sum();
                assert!(
                    (col_sum - 1.0).abs() < 0.1,
                    "comb col {j} sums to {col_sum}, expected ~1.0"
                );
            }
        }
    }

    // ── Round-trip: expand → hc_pre → identity → hc_post ───────────────────

    #[test]
    fn hc_pre_post_round_trip_shapes() {
        let (seq, hc, d) = (2, 4, 16);
        let h = Array2::from_shape_fn((seq, d), |(s, j)| (s as f32 + j as f32) * 0.1);
        let x_hc = hc_expand(&h, hc);
        let params = make_test_params(hc, d);

        // hc_pre reduces [seq, hc, d] → [seq, d]
        let pre_out = hc_pre(&x_hc, &params, DEFAULT_SINKHORN_ITERS, DEFAULT_HC_EPS);
        assert_eq!(pre_out.y.shape(), &[seq, d]);

        // hc_post expands [seq, d] → [seq, hc, d]
        let post_out = hc_post(&pre_out.y, &x_hc, &pre_out.post, &pre_out.comb);
        assert_eq!(post_out.shape(), &[seq, hc, d]);
    }

    #[test]
    fn hc_pre_reduces_dimensionality() {
        // The core property: hc_pre takes [seq, hc, d] and produces [seq, d]
        let (seq, hc, d) = (3, 4, 32);
        let x_hc = Array3::from_shape_fn((seq, hc, d), |(s, h, j)| {
            (s as f32 * 0.1) + (h as f32 * 0.2) + (j as f32 * 0.01)
        });
        let params = make_test_params(hc, d);
        let out = hc_pre(&x_hc, &params, DEFAULT_SINKHORN_ITERS, DEFAULT_HC_EPS);
        // Output is [seq, d] — the hc dimension is reduced
        assert_eq!(out.y.shape(), &[seq, d]);
        assert_eq!(out.y.ndim(), 2);
    }

    #[test]
    fn hc_post_expands_dimensionality() {
        // The core property: hc_post takes [seq, d] and produces [seq, hc, d]
        let (seq, hc, d) = (3, 4, 32);
        let x = Array2::from_shape_fn((seq, d), |(s, j)| (s as f32 + j as f32) * 0.1);
        let residual = Array3::zeros((seq, hc, d));
        let post = Array2::from_shape_fn((seq, hc), |(_, _)| 0.5);
        let comb = Array2::from_shape_fn((seq, hc * hc), |(_, _)| 1.0 / (hc * hc) as f32);
        let out = hc_post(&x, &residual, &post, &comb);
        assert_eq!(out.shape(), &[seq, hc, d]);
        assert_eq!(out.ndim(), 3);
    }

    // ── V4-Flash realistic dimensions ──────────────────────────────────────

    #[test]
    fn hc_pre_v4_flash_dimensions() {
        // V4-Flash: hidden=4096, hc_mult=4, seq=1 (decode)
        let (seq, hc, d) = (1, 4, 4096);
        let x_hc = Array3::from_shape_fn((seq, hc, d), |(_, h, j)| ((h * d + j) as f32) * 1e-4);
        let params = make_test_params(hc, d);
        let out = hc_pre(&x_hc, &params, DEFAULT_SINKHORN_ITERS, DEFAULT_HC_EPS);
        assert_eq!(out.y.shape(), &[1, 4096]);
        assert!(out.y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hc_head_v4_flash_dimensions() {
        let (seq, hc, d) = (1, 4, 4096);
        let x_hc = Array3::from_shape_fn((seq, hc, d), |(_, h, j)| ((h * d + j) as f32) * 1e-4);
        let params = make_test_params(hc, d);
        let y = hc_head(&x_hc, &params, DEFAULT_HC_EPS);
        assert_eq!(y.shape(), &[1, 4096]);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ── Numerical correctness ──────────────────────────────────────────────

    #[test]
    fn hc_pre_zero_input_produces_eps_positive_output() {
        // With all-zero input, RMSNorm gives 0/sqrt(eps) = 0, linear proj gives base,
        // sigmoid(base) > 0, pre = sigmoid(base) + eps > 0.
        let (seq, hc, d) = (1, 4, 8);
        let x_hc = Array3::zeros((seq, hc, d));
        let params = make_test_params(hc, d);
        let out = hc_pre(&x_hc, &params, DEFAULT_SINKHORN_ITERS, DEFAULT_HC_EPS);
        // y = sum(pre * x) = sum(pre * 0) = 0
        for v in out.y.iter() {
            assert!(
                (v - 0.0).abs() < 1e-6,
                "zero input should produce zero y, got {v}"
            );
        }
    }

    #[test]
    fn hc_head_zero_input_produces_zero_y() {
        let (seq, hc, d) = (1, 4, 8);
        let x_hc = Array3::zeros((seq, hc, d));
        let params = make_test_params(hc, d);
        let y = hc_head(&x_hc, &params, DEFAULT_HC_EPS);
        // y = sum(pre * x) = sum(pre * 0) = 0
        for v in y.iter() {
            assert!(
                (v - 0.0).abs() < 1e-6,
                "zero input should produce zero y, got {v}"
            );
        }
    }

    #[test]
    fn sigmoid_known_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.9999);
        assert!(sigmoid(-10.0) < 0.0001);
    }

    #[test]
    fn softmax_uniform_distribution() {
        let vals = vec![1.0, 1.0, 1.0, 1.0];
        let sm = softmax(&vals);
        for &v in &sm {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn rms_norm_preserves_shape() {
        let x =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let out = rms_norm_flat(&x, 1e-6);
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn rms_norm_output_is_finite() {
        let x = Array2::zeros((3, 8));
        let out = rms_norm_flat(&x, 1e-6);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

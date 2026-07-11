use crate::kernels::{DispatchGeometry, KernelHandle};

pub const Q4K_MATVEC_KERNEL: KernelHandle = KernelHandle::new(
    "q4k_matvec",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const Q6K_MATVEC_KERNEL: KernelHandle = KernelHandle::new(
    "q6k_matvec",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const Q4K_MATMUL_KERNEL: KernelHandle = KernelHandle::new(
    "q4k_matmul",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const Q6K_MATMUL_KERNEL: KernelHandle = KernelHandle::new(
    "q6k_matmul",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const Q4K_DUAL_MATVEC_KERNEL: KernelHandle = KernelHandle::new(
    "q4k_dual_matvec",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const F32_GEMV_KERNEL: KernelHandle = KernelHandle::new(
    "f32_gemv",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const F16_GEMV_KERNEL: KernelHandle = KernelHandle::new(
    "f16_gemv",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const Q4_MATVEC_KERNEL: KernelHandle = KernelHandle::new(
    "q4_matvec",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const Q4_VECMAT_KERNEL: KernelHandle = KernelHandle::new(
    "q4_vecmat",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const KV_APPEND_KERNEL: KernelHandle = KernelHandle::new(
    "kv_append",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const RMS_NORM_KERNEL: KernelHandle = KernelHandle::new(
    "rms_norm",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const RMS_NORM_HEADS_KERNEL: KernelHandle = KernelHandle::new(
    "rms_norm_heads",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const GEGGLU_SILU_KERNEL: KernelHandle = KernelHandle::new(
    "geglu_silu",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const GEGGLU_GELU_TANH_KERNEL: KernelHandle = KernelHandle::new(
    "geglu_gelu_tanh",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const ACTIVATION_SILU_KERNEL: KernelHandle = KernelHandle::new(
    "activation_silu",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const ACTIVATION_GELU_TANH_KERNEL: KernelHandle = KernelHandle::new(
    "activation_gelu_tanh",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const RESIDUAL_ADD_KERNEL: KernelHandle = KernelHandle::new(
    "residual_add",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const ROPE_KERNEL: KernelHandle = KernelHandle::new(
    "rope",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [128, 1, 1],
    },
);

pub const DECODE_ATTENTION_KERNEL: KernelHandle = KernelHandle::new(
    "decode_attention",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [256, 1, 1],
    },
);

pub const PREFILL_ATTENTION_KERNEL: KernelHandle = KernelHandle::new(
    "prefill_attention",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [256, 1, 1],
    },
);

/// LARQL-GPU-B4: block size for the greedy top-K partial-reduction kernel.
/// One block reduces a `GREEDY_BLOCK_SIZE`-element slice of the score
/// vector to its top-`k` finite candidates. 256 matches the attention
/// kernels' block size and keeps the shared-memory footprint tiny (256
/// floats = 1 KB per block).
pub const GREEDY_BLOCK_SIZE: usize = 256;
/// LARQL-GPU-B4: compile-time cap on the greedy candidate width. The
/// production candidate width is five (the existing greedy candidate
/// width); eight leaves headroom and keeps the device result struct a
/// power-of-two-friendly 64 bytes. The host reads back only
/// `candidate_width` slots.
pub const GREEDY_MAX_K: usize = 8;

pub const GREEDY_TOPK_PARTIAL_KERNEL: KernelHandle = KernelHandle::new(
    "greedy_topk_partial",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [256, 1, 1],
    },
);

pub const GREEDY_TOPK_FINAL_KERNEL: KernelHandle = KernelHandle::new(
    "greedy_topk_final",
    DispatchGeometry {
        workgroups: [1, 1, 1],
        threads_per_group: [256, 1, 1],
    },
);

pub const Q4K_MATVEC_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>

#ifndef LARQL_HALF_BITS_DEFINED
#define LARQL_HALF_BITS_DEFINED
union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}
#endif

extern "C" __global__ void q4k_matvec(
    const unsigned char* w4k,
    const float* x,
    float* out,
    unsigned int n,
    unsigned int k)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    const unsigned int superblocks = k / 256u;
    const unsigned int bytes_per_row = superblocks * 144u;
    const unsigned char* row_w = w4k + row * bytes_per_row;
    float acc = 0.0f;

    for (unsigned int sb = 0; sb < superblocks; ++sb) {
        const unsigned char* block = row_w + sb * 144u;
        const float d = larql_decode_f16(
            static_cast<unsigned short>(block[0]) |
            (static_cast<unsigned short>(block[1]) << 8u));
        const float dmin = larql_decode_f16(
            static_cast<unsigned short>(block[2]) |
            (static_cast<unsigned short>(block[3]) << 8u));
        const unsigned char* packed = block + 4u;

        for (unsigned int j = 0; j < 8u; ++j) {
            unsigned int sc;
            unsigned int mn;
            if (j < 4u) {
                sc = static_cast<unsigned int>(packed[j]) & 0x3Fu;
                mn = static_cast<unsigned int>(packed[j + 4u]) & 0x3Fu;
            } else {
                sc = (static_cast<unsigned int>(packed[j + 4u]) & 0x0Fu)
                    | ((static_cast<unsigned int>(packed[j - 4u]) >> 6u) << 4u);
                mn = (static_cast<unsigned int>(packed[j + 4u]) >> 4u)
                    | ((static_cast<unsigned int>(packed[j]) >> 6u) << 4u);
            }

            const float scale = d * static_cast<float>(sc);
            const float min_scale = dmin * static_cast<float>(mn);
            const unsigned int group = j >> 1u;
            const bool high_nibble = (j & 1u) != 0u;
            const unsigned int x_base = sb * 256u + j * 32u;
            const unsigned char* qs = block + 16u + group * 32u;

            float sum_x = 0.0f;
            float dot = 0.0f;
            #pragma unroll
            for (unsigned int l = 0; l < 32u; ++l) {
                const float xv = x[x_base + l];
                const unsigned char byte = qs[l];
                const float nib = high_nibble
                    ? static_cast<float>((byte >> 4u) & 0x0Fu)
                    : static_cast<float>(byte & 0x0Fu);
                sum_x += xv;
                dot += nib * xv;
            }
            acc += scale * dot - min_scale * sum_x;
        }
    }

    out[row] = acc;
}
"#;

/// Q6_K matvec: one row per thread. Each Q6_K super-block is 210 bytes and
/// encodes 256 weights: 128 bytes of 4-bit `ql`, 64 bytes of 2-bit `qh`, 16
/// int8 per-16 scales, and one f16 super-block scale `d`. A weight value is
/// `d * scale[j] * (((lo4) | (hi2 << 4)) - 32)`. Mirrors the CPU reference
/// `decode_q6k_superblock_into` + per-row dot.
pub const Q6K_MATVEC_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>

#ifndef LARQL_HALF_BITS_DEFINED
#define LARQL_HALF_BITS_DEFINED
union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}
#endif

extern "C" __global__ void q6k_matvec(
    const unsigned char* w6k,
    const float* x,
    float* out,
    unsigned int n,
    unsigned int k)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    const unsigned int superblocks = k / 256u;
    const unsigned int bytes_per_row = superblocks * 210u;
    const unsigned char* row_w = w6k + row * bytes_per_row;
    float acc = 0.0f;

    for (unsigned int sb = 0; sb < superblocks; ++sb) {
        const unsigned char* block = row_w + sb * 210u;
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128u;
        const unsigned char* scales = block + 192u;
        const float d = larql_decode_f16(
            static_cast<unsigned short>(block[208]) |
            (static_cast<unsigned short>(block[209]) << 8u));

        for (unsigned int j = 0u; j < 16u; ++j) {
            const float sc = d * static_cast<float>(static_cast<signed char>(scales[j]));
            const unsigned int base = j * 16u;
            const unsigned int x_base = sb * 256u + base;
            float dot = 0.0f;
            #pragma unroll
            for (unsigned int i = 0u; i < 16u; ++i) {
                const unsigned int idx = base + i;
                const unsigned int lo4 = (idx & 1u) == 0u
                    ? (ql[idx >> 1u] & 0x0Fu)
                    : ((ql[idx >> 1u] >> 4u) & 0x0Fu);
                const unsigned int hi2 = (qh[idx >> 2u] >> ((idx & 3u) * 2u)) & 0x03u;
                const float val = static_cast<float>(
                    static_cast<signed int>((lo4 | (hi2 << 4u)) - 32u));
                dot += val * x[x_base + i];
            }
            acc += sc * dot;
        }
    }

    out[row] = acc;
}
"#;

/// Amortised Q4_K × f32 matmul: `out[s, r] = sum_k W[r, k] * X[s, k]`. One
/// (row, seq) pair per thread. Each weight super-block (144 bytes, 256
/// weights) is decoded once into registers and FMA'd across all `seq`
/// columns, mirroring the CPU `kquant_matmul_into` amortised pattern (weight
/// bytes read once, not `seq` times). Output is `[seq, rows]` row-major to
/// match the CPU contract.
pub const Q4K_MATMUL_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>

#ifndef LARQL_HALF_BITS_DEFINED
#define LARQL_HALF_BITS_DEFINED
union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}
#endif

extern "C" __global__ void q4k_matmul(
    const unsigned char* w4k,
    const float* x,
    float* out,
    unsigned int n,
    unsigned int k,
    unsigned int seq)
{
    // Flatten (row, seq) into a 1D thread index.
    const unsigned int tile = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = tile / seq;
    const unsigned int s = tile % seq;
    if (row >= n || s >= seq) {
        return;
    }

    const unsigned int superblocks = k / 256u;
    const unsigned int bytes_per_row = superblocks * 144u;
    const unsigned char* row_w = w4k + row * bytes_per_row;
    const float* x_row = x + s * k;
    float acc = 0.0f;

    for (unsigned int sb = 0; sb < superblocks; ++sb) {
        const unsigned char* block = row_w + sb * 144u;
        const float d = larql_decode_f16(
            static_cast<unsigned short>(block[0]) |
            (static_cast<unsigned short>(block[1]) << 8u));
        const float dmin = larql_decode_f16(
            static_cast<unsigned short>(block[2]) |
            (static_cast<unsigned short>(block[3]) << 8u));
        const unsigned char* packed = block + 4u;

        for (unsigned int j = 0u; j < 8u; ++j) {
            unsigned int sc;
            unsigned int mn;
            if (j < 4u) {
                sc = static_cast<unsigned int>(packed[j]) & 0x3Fu;
                mn = static_cast<unsigned int>(packed[j + 4u]) & 0x3Fu;
            } else {
                sc = (static_cast<unsigned int>(packed[j + 4u]) & 0x0Fu)
                    | ((static_cast<unsigned int>(packed[j - 4u]) >> 6u) << 4u);
                mn = (static_cast<unsigned int>(packed[j + 4u]) >> 4u)
                    | ((static_cast<unsigned int>(packed[j]) >> 6u) << 4u);
            }

            const float scale = d * static_cast<float>(sc);
            const float min_scale = dmin * static_cast<float>(mn);
            const unsigned int group = j >> 1u;
            const bool high_nibble = (j & 1u) != 0u;
            const unsigned int x_base = sb * 256u + j * 32u;
            const unsigned char* qs = block + 16u + group * 32u;

            float sum_x = 0.0f;
            float dot = 0.0f;
            #pragma unroll
            for (unsigned int l = 0; l < 32u; ++l) {
                const float xv = x_row[x_base + l];
                const unsigned char byte = qs[l];
                const float nib = high_nibble
                    ? static_cast<float>((byte >> 4u) & 0x0Fu)
                    : static_cast<float>(byte & 0x0Fu);
                sum_x += xv;
                dot += nib * xv;
            }
            acc += scale * dot - min_scale * sum_x;
        }
    }

    // out is [seq, rows] row-major.
    out[s * n + row] = acc;
}
"#;

/// Amortised Q6_K × f32 matmul: `out[s, r] = sum_k W[r, k] * X[s, k]`. One
/// (row, seq) pair per thread. Each Q6_K super-block (210 bytes, 256
/// weights) is decoded once into registers and FMA'd across all `seq`
/// columns, mirroring the CPU `q6k_matmul_into` amortised pattern. Output is
/// `[seq, rows]` row-major to match the CPU contract.
pub const Q6K_MATMUL_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>

#ifndef LARQL_HALF_BITS_DEFINED
#define LARQL_HALF_BITS_DEFINED
union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}
#endif

extern "C" __global__ void q6k_matmul(
    const unsigned char* w6k,
    const float* x,
    float* out,
    unsigned int n,
    unsigned int k,
    unsigned int seq)
{
    // Flatten (row, seq) into a 1D thread index.
    const unsigned int tile = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = tile / seq;
    const unsigned int s = tile % seq;
    if (row >= n || s >= seq) {
        return;
    }

    const unsigned int superblocks = k / 256u;
    const unsigned int bytes_per_row = superblocks * 210u;
    const unsigned char* row_w = w6k + row * bytes_per_row;
    const float* x_row = x + s * k;
    float acc = 0.0f;

    for (unsigned int sb = 0; sb < superblocks; ++sb) {
        const unsigned char* block = row_w + sb * 210u;
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128u;
        const unsigned char* scales = block + 192u;
        const float d = larql_decode_f16(
            static_cast<unsigned short>(block[208]) |
            (static_cast<unsigned short>(block[209]) << 8u));

        for (unsigned int j = 0u; j < 16u; ++j) {
            const float sc = d * static_cast<float>(static_cast<signed char>(scales[j]));
            const unsigned int base = j * 16u;
            const unsigned int x_base = sb * 256u + base;
            float dot = 0.0f;
            #pragma unroll
            for (unsigned int i = 0u; i < 16u; ++i) {
                const unsigned int idx = base + i;
                const unsigned int lo4 = (idx & 1u) == 0u
                    ? (ql[idx >> 1u] & 0x0Fu)
                    : ((ql[idx >> 1u] >> 4u) & 0x0Fu);
                const unsigned int hi2 = (qh[idx >> 2u] >> ((idx & 3u) * 2u)) & 0x03u;
                const float val = static_cast<float>(
                    static_cast<signed int>((lo4 | (hi2 << 4u)) - 32u));
                dot += val * x_row[x_base + i];
            }
            acc += sc * dot;
        }
    }

    // out is [seq, rows] row-major.
    out[s * n + row] = acc;
}
"#;

/// Fused two-weight Q4_K matvec sharing one input vector. One row per
/// thread. Both weight matrices `w_a` and `w_b` have identical `(rows,
/// hidden)` shape and the same input `x`. Writes `out_a[row] = W_a · x`
/// and `out_b[row] = W_b · x`. Mirrors the CPU `q4k_dual_matvec_into`
/// contract (gate+up projections share `x`).
pub const Q4K_DUAL_MATVEC_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>

#ifndef LARQL_HALF_BITS_DEFINED
#define LARQL_HALF_BITS_DEFINED
union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}
#endif

extern "C" __global__ void q4k_dual_matvec(
    const unsigned char* w_a,
    const unsigned char* w_b,
    const float* x,
    float* out_a,
    float* out_b,
    unsigned int n,
    unsigned int k)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    const unsigned int superblocks = k / 256u;
    const unsigned int bytes_per_row = superblocks * 144u;
    const unsigned char* row_a = w_a + row * bytes_per_row;
    const unsigned char* row_b = w_b + row * bytes_per_row;
    float acc_a = 0.0f;
    float acc_b = 0.0f;

    for (unsigned int sb = 0; sb < superblocks; ++sb) {
        const unsigned char* block_a = row_a + sb * 144u;
        const unsigned char* block_b = row_b + sb * 144u;

        const float d_a = larql_decode_f16(
            static_cast<unsigned short>(block_a[0]) |
            (static_cast<unsigned short>(block_a[1]) << 8u));
        const float dmin_a = larql_decode_f16(
            static_cast<unsigned short>(block_a[2]) |
            (static_cast<unsigned short>(block_a[3]) << 8u));
        const float d_b = larql_decode_f16(
            static_cast<unsigned short>(block_b[0]) |
            (static_cast<unsigned short>(block_b[1]) << 8u));
        const float dmin_b = larql_decode_f16(
            static_cast<unsigned short>(block_b[2]) |
            (static_cast<unsigned short>(block_b[3]) << 8u));
        const unsigned char* pa = block_a + 4u;
        const unsigned char* pb = block_b + 4u;

        unsigned int scales_a[8u];
        unsigned int mins_a[8u];
        unsigned int scales_b[8u];
        unsigned int mins_b[8u];
        #pragma unroll
        for (unsigned int j = 0u; j < 4u; ++j) {
            scales_a[j] = static_cast<unsigned int>(pa[j]) & 0x3Fu;
            mins_a[j] = static_cast<unsigned int>(pa[j + 4u]) & 0x3Fu;
            scales_a[j + 4u] = (static_cast<unsigned int>(pa[j + 8u]) & 0x0Fu)
                | ((static_cast<unsigned int>(pa[j]) >> 6u) << 4u);
            mins_a[j + 4u] = (static_cast<unsigned int>(pa[j + 8u]) >> 4u)
                | ((static_cast<unsigned int>(pa[j + 4u]) >> 6u) << 4u);
            scales_b[j] = static_cast<unsigned int>(pb[j]) & 0x3Fu;
            mins_b[j] = static_cast<unsigned int>(pb[j + 4u]) & 0x3Fu;
            scales_b[j + 4u] = (static_cast<unsigned int>(pb[j + 8u]) & 0x0Fu)
                | ((static_cast<unsigned int>(pb[j]) >> 6u) << 4u);
            mins_b[j + 4u] = (static_cast<unsigned int>(pb[j + 8u]) >> 4u)
                | ((static_cast<unsigned int>(pb[j + 4u]) >> 6u) << 4u);
        }

        const unsigned char* qa = block_a + 16u;
        const unsigned char* qb = block_b + 16u;
        const unsigned int x_sb_base = sb * 256u;

        #pragma unroll
        for (unsigned int g = 0u; g < 4u; ++g) {
            const unsigned int sb_lo = 2u * g;
            const unsigned int sb_hi = 2u * g + 1u;
            const float sc_a_lo = d_a * static_cast<float>(scales_a[sb_lo]);
            const float sc_a_hi = d_a * static_cast<float>(scales_a[sb_hi]);
            const float mn_a_lo = dmin_a * static_cast<float>(mins_a[sb_lo]);
            const float mn_a_hi = dmin_a * static_cast<float>(mins_a[sb_hi]);
            const float sc_b_lo = d_b * static_cast<float>(scales_b[sb_lo]);
            const float sc_b_hi = d_b * static_cast<float>(scales_b[sb_hi]);
            const float mn_b_lo = dmin_b * static_cast<float>(mins_b[sb_lo]);
            const float mn_b_hi = dmin_b * static_cast<float>(mins_b[sb_hi]);

            const unsigned int x_lo_base = x_sb_base + sb_lo * 32u;
            const unsigned int x_hi_base = x_sb_base + sb_hi * 32u;
            const unsigned char* chunk_a = qa + g * 32u;
            const unsigned char* chunk_b = qb + g * 32u;

            float sum_x_lo = 0.0f;
            float dot_a_lo = 0.0f;
            float dot_b_lo = 0.0f;
            #pragma unroll
            for (unsigned int l = 0u; l < 32u; ++l) {
                const float xv = x[x_lo_base + l];
                const unsigned char byte_a = chunk_a[l];
                const unsigned char byte_b = chunk_b[l];
                sum_x_lo += xv;
                dot_a_lo += static_cast<float>(byte_a & 0x0Fu) * xv;
                dot_b_lo += static_cast<float>(byte_b & 0x0Fu) * xv;
            }

            float sum_x_hi = 0.0f;
            float dot_a_hi = 0.0f;
            float dot_b_hi = 0.0f;
            #pragma unroll
            for (unsigned int l = 0u; l < 32u; ++l) {
                const float xv = x[x_hi_base + l];
                const unsigned char byte_a = chunk_a[l];
                const unsigned char byte_b = chunk_b[l];
                sum_x_hi += xv;
                dot_a_hi += static_cast<float>((byte_a >> 4u) & 0x0Fu) * xv;
                dot_b_hi += static_cast<float>((byte_b >> 4u) & 0x0Fu) * xv;
            }

            acc_a += sc_a_lo * dot_a_lo - mn_a_lo * sum_x_lo;
            acc_a += sc_a_hi * dot_a_hi - mn_a_hi * sum_x_hi;
            acc_b += sc_b_lo * dot_b_lo - mn_b_lo * sum_x_lo;
            acc_b += sc_b_hi * dot_b_hi - mn_b_hi * sum_x_hi;
        }
    }

    out_a[row] = acc_a;
    out_b[row] = acc_b;
}
"#;

/// Dense f32 matrix-vector multiply: `out[row] = sum_col W[row, col] * x[col]`.
/// One row per thread. `w` is row-major `[n, k]`. Matches the CPU
/// `MatMul::f32_gemv` contract (input is `ArrayView2<f32>`, flattened to a
/// row-major slice by the launcher).
pub const F32_GEMV_CUDA_SRC: &str = r#"
extern "C" __global__ void f32_gemv(
    const float* w,
    const float* x,
    float* out,
    unsigned int n,
    unsigned int k)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    // 64-bit element offset: a large dense matrix (n*k > 2^32 f32) would
    // wrap a 32-bit `row * k` and read out-of-bounds. The host launcher
    // guards the same limit.
    const float* row_w = w + (unsigned long long)row * (unsigned long long)k;
    float acc = 0.0f;
    #pragma unroll 4
    for (unsigned int col = 0u; col < k; ++col) {
        acc += row_w[col] * x[col];
    }
    out[row] = acc;
}
"#;

/// Dense f16 matrix-vector multiply: `out[row] = sum_col W[row, col] * x[col]`.
/// One row per thread. `w_f16` is row-major `[n, k]` little-endian f16 bytes
/// (the same layout the CPU `MatMul::f16_gemv` consumes). f16 → f32 decode
/// reuses the Q6_K helper union so the two halves of the module share one
/// decoder.
pub const F16_GEMV_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>

#ifndef LARQL_HALF_BITS_DEFINED
#define LARQL_HALF_BITS_DEFINED
union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}
#endif

extern "C" __global__ void f16_gemv(
    const unsigned char* w_f16,
    const float* x,
    float* out,
    unsigned int n,
    unsigned int k)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    // Use 64-bit byte/element offsets: a large vocab head (e.g. 262144 x 8192
    // f16) exceeds 2^31 bytes, so a 32-bit `2u * row * k` would wrap and read
    // out-of-bounds global memory. The host launcher guards the same limit.
    const unsigned long long row_bytes = (unsigned long long)2u * (unsigned long long)row * (unsigned long long)k;
    float acc = 0.0f;
    #pragma unroll 4
    for (unsigned int col = 0u; col < k; ++col) {
        const unsigned long long off = row_bytes + (unsigned long long)2u * (unsigned long long)col;
        const unsigned short bits = static_cast<unsigned short>(w_f16[off])
            | (static_cast<unsigned short>(w_f16[off + 1u]) << 8u);
        acc += larql_decode_f16(bits) * x[col];
    }
    out[row] = acc;
}
"#;

/// Q4_0 × Q8 matvec: `out[row] = sum_blocks (q4_scale * q8_scale * sum_j
/// ((nib - 8) * q8[j]))`. One row per thread. Each Q4_0 block is 18 bytes:
/// 2-byte little-endian f16 scale followed by 16 packed nibble bytes (lo
/// nibble = element 2j, hi nibble = element 2j+1, value = nibble - 8). The
/// input is pre-quantised Q8 (`q8_x[hidden]` int8 + per-32-block `q8_scales`
/// f32). Mirrors the CPU scalar `q4_0_matvec_c` non-ARM fallback exactly.
pub const Q4_MATVEC_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>

#ifndef LARQL_HALF_BITS_DEFINED
#define LARQL_HALF_BITS_DEFINED
union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}
#endif

extern "C" __global__ void q4_matvec(
    const unsigned char* q4_data,
    const signed char* q8_x,
    const float* q8_scales,
    float* out,
    unsigned int n,
    unsigned int k)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    const unsigned int blocks = k / 32u;
    const unsigned int bytes_per_row = blocks * 18u;
    // 64-bit row offset: a wide FFN (n*bytes_per_row > 2^32) would wrap a
    // 32-bit `row * bytes_per_row` and read OOB. The host launcher guards
    // the same limit on the dims.
    const unsigned long long row_off = (unsigned long long)row * (unsigned long long)bytes_per_row;
    const unsigned char* row_w = q4_data + row_off;
    float acc = 0.0f;

    for (unsigned int b = 0u; b < blocks; ++b) {
        const unsigned char* block = row_w + b * 18u;
        const float q4_scale = larql_decode_f16(
            static_cast<unsigned short>(block[0]) |
            (static_cast<unsigned short>(block[1]) << 8u));
        const float combined_scale = q4_scale * q8_scales[b];
        const unsigned char* quants = block + 2u;
        const signed char* q8_ptr = q8_x + b * 32u;
        #pragma unroll
        for (unsigned int j = 0u; j < 16u; ++j) {
            const unsigned char byte = quants[j];
            const int lo_v = static_cast<int>(byte & 0x0Fu) - 8;
            const int hi_v = static_cast<int>((byte >> 4u) & 0x0Fu) - 8;
            acc += static_cast<float>(lo_v) * static_cast<float>(q8_ptr[j * 2u])     * combined_scale;
            acc += static_cast<float>(hi_v) * static_cast<float>(q8_ptr[j * 2u + 1u]) * combined_scale;
        }
    }

    out[row] = acc;
}
"#;

/// Q4_0 vector-matrix: `out[col] = sum_rows (act[row] * q4_scale * (nib -
/// 8))`. One output column per thread (gather across all `intermediate`
/// rows). Each Q4_0 block is 18 bytes (2-byte f16 scale + 16 packed nibble
/// bytes; lo nibble = element 2j, hi nibble = element 2j+1, value = nibble -
/// 8). The CPU reference scatters row-major; this transposes the loop to a
/// per-output gather, which produces identical arithmetic and maps naturally
/// to one-thread-per-output-column parallelism. Rows with `|act| < 1e-10`
/// are skipped (matching the CPU zero-skip).
pub const Q4_VECMAT_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>

#ifndef LARQL_HALF_BITS_DEFINED
#define LARQL_HALF_BITS_DEFINED
union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}
#endif

extern "C" __global__ void q4_vecmat(
    const float* activation,
    const unsigned char* q4_data,
    float* out,
    unsigned int intermediate,
    unsigned int hidden)
{
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden) {
        return;
    }

    const unsigned int blocks_per_row = hidden / 32u;
    const unsigned int bytes_per_row = blocks_per_row * 18u;
    // Locate which 32-element block and intra-block position this output
    // column maps to. Within a block, element 2j is the lo nibble of
    // quants[j], element 2j+1 is the hi nibble.
    const unsigned int block_idx = col / 32u;
    const unsigned int in_block = col - block_idx * 32u;
    const bool is_hi = (in_block & 1u) != 0u;
    const unsigned int j = in_block >> 1u;

    float acc = 0.0f;
    for (unsigned int row = 0u; row < intermediate; ++row) {
        const float act = activation[row];
        // Zero-skip mirrors the CPU reference.
        if (act > -1e-10f && act < 1e-10f) {
            continue;
        }
        // 64-bit row offset guard (matches q4_matvec).
        const unsigned long long row_off = (unsigned long long)row * (unsigned long long)bytes_per_row;
        const unsigned char* block = q4_data + row_off + block_idx * 18u;
        const float scale = larql_decode_f16(
            static_cast<unsigned short>(block[0]) |
            (static_cast<unsigned short>(block[1]) << 8u)) * act;
        const unsigned char quants_byte = block[2u + j];
        const int val = is_hi
            ? static_cast<int>((quants_byte >> 4u) & 0x0Fu) - 8
            : static_cast<int>(quants_byte & 0x0Fu) - 8;
        acc += static_cast<float>(val) * scale;
    }

    out[col] = acc;
}
"#;

/// KV append: copy a contiguous block of freshly-projected K/V rows into
/// the cache starting at slot `pos`. One thread per element across all
/// `seq_len` rows (`seq_len * num_kv_heads * head_dim` elements total). The
/// cache layout is `[max_seq, num_kv_heads, head_dim]` row-major over the
/// sequence dimension, so the slot for `(row, elem)` begins at
/// `(pos + row) * row_elems + elem`. The 64-bit slot-offset guard mirrors
/// the other kernels so a cache larger than 2^32 elements doesn't wrap.
/// `row_elems` is passed as a precomputed 32-bit value by the host launcher
/// (which already verified `num_kv_heads * head_dim <= u32::MAX`), so the
/// kernel-side multiplication cannot wrap.
pub const KV_APPEND_CUDA_SRC: &str = r#"
extern "C" __global__ void kv_append(
    const float* new_k,
    const float* new_v,
    float* k_cache,
    float* v_cache,
    unsigned int pos,
    unsigned int seq_len,
    unsigned int row_elems)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = seq_len * row_elems;
    if (tid >= total) {
        return;
    }
    const unsigned int row = tid / row_elems;
    const unsigned int elem = tid - row * row_elems;
    const unsigned long long slot = (unsigned long long)(pos + row) * (unsigned long long)row_elems
        + (unsigned long long)elem;
    k_cache[slot] = new_k[tid];
    v_cache[slot] = new_v[tid];
}
"#;

/// RMSNorm over each row of a `[rows, cols]` matrix — the device twin of
/// `larql_compute::residual::rms_norm_eps`. One thread-block per row: a
/// block-reduction of the sum-of-squares in f64, then
/// `out[i,j] = x[i,j] / rms * (offset + w[j])` (or `* 1.0` when
/// `has_weight == 0`, mirroring the `None` weight arm of the CPU reference).
///
/// The reduction matches the CPU reference's f64 accumulation: each thread
/// loads its element as `double`, contributes its squared value to a
/// warp+block reduction in `double`, broadcasts the `rms = sqrt(mean_sq +
/// eps)` to all threads, and writes its output. `eps` is a `double` arg so
/// the `+ eps` happens at f64 precision exactly as on the host.
pub const RMS_NORM_CUDA_SRC: &str = r#"
extern "C" __global__ void rms_norm(
    const float* x,
    const float* weight,
    float* out,
    unsigned int rows,
    unsigned int cols,
    double eps,
    float offset,
    int has_weight)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int block_dim = blockDim.x;
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    // 64-bit row offset: a large hidden (rows*cols > 2^32) would wrap a
    // 32-bit `row * cols` and read OOB. The host launcher guards the same
    // limit on the dims.
    const unsigned long long row_off = (unsigned long long)row * (unsigned long long)cols;
    const float* x_row = x + row_off;
    float* out_row = out + row_off;

    // f64 sum-of-squares, strided across the row.
    double sq_sum = 0.0;
    for (unsigned int j = tid; j < cols; j += block_dim) {
        const double v = (double)x_row[j];
        sq_sum += v * v;
    }

    // Block reduction in f64 (warp shuffle + shared memory for the cross-
    // warp stage). 32 warps max → 1024 threads, the launcher's cap.
    __shared__ double warp_sums[32];
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    for (unsigned int off = 16u; off > 0u; off >>= 1u) {
        sq_sum += __shfl_down_sync(0xFFFFFFFFu, sq_sum, off);
    }
    if (lane == 0u) {
        warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    const unsigned int num_warps = block_dim >> 5u;
    if (warp == 0u) {
        double w_sum = (tid < num_warps) ? warp_sums[tid] : 0.0;
        for (unsigned int off = 16u; off > 0u; off >>= 1u) {
            w_sum += __shfl_down_sync(0xFFFFFFFFu, w_sum, off);
        }
        if (lane == 0u) {
            warp_sums[0u] = w_sum;
        }
    }
    __syncthreads();
    const double total_sq = warp_sums[0u];
    const double mean_sq = total_sq / (double)cols;
    const float rms = (float)sqrt(mean_sq + eps);

    for (unsigned int j = tid; j < cols; j += block_dim) {
        const float w = (has_weight != 0) ? (offset + weight[j]) : 1.0f;
        out_row[j] = (x_row[j] / rms) * w;
    }
}
"#;

/// Per-head RMSNorm over each row's head blocks — the device twin of
/// `larql_compute::residual::rms_norm_heads`. One thread-block per
/// (sequence position, head): a block-reduction of the sum-of-squares over
/// the `head_dim` elements of that head, then
/// `out[s, off+d] = x[s, off+d] / rms * (offset + weight[off+d])` (or
/// `* 1.0` when `has_weight == 0`, the parameter-free case). f64
/// accumulation matches the CPU reference. This single entry point serves
/// both the weighted (`rms_norm_heads`) and the no-weight
/// (`rms_norm_heads_no_weight`) CPU references — the host launcher passes
/// `has_weight = 0` and a dummy weight buffer for the no-weight path.
pub const RMS_NORM_HEADS_CUDA_SRC: &str = r#"
extern "C" __global__ void rms_norm_heads(
    const float* x,
    const float* weight,
    float* out,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    double eps,
    float offset,
    int has_weight)
{
    // One block per (position, head). Flatten to a 1D grid: block index =
    // pos * num_heads + head.
    const unsigned int bid = blockIdx.x;
    const unsigned int pos = bid / num_heads;
    const unsigned int head = bid - pos * num_heads;
    if (pos >= seq_len) {
        return;
    }
    // 64-bit row stride: `num_heads * head_dim` is the per-row width. Widened
    // before the multiply so the product can't wrap a 32-bit multiply (the
    // host launcher additionally guards `num_heads * head_dim <= u32::MAX`).
    const unsigned long long row_stride = (unsigned long long)num_heads
        * (unsigned long long)head_dim;
    const unsigned long long row_off = (unsigned long long)pos * row_stride;
    const float* x_row = x + row_off;
    float* out_row = out + row_off;
    // `head_off` is the within-row offset to this head's `head_dim` block.
    // Widened to 64-bit so `head * head_dim` can't wrap on a huge head count
    // (the host launcher guards the product `num_heads * head_dim`, but a
    // single head's start index `head * head_dim` is bounded by the same
    // product and is safe in 64-bit).
    const unsigned long long head_off = (unsigned long long)head
        * (unsigned long long)head_dim;

    const unsigned int tid = threadIdx.x;
    const unsigned int block_dim = blockDim.x;

    double sq_sum = 0.0;
    for (unsigned int d = tid; d < head_dim; d += block_dim) {
        const double v = (double)x_row[head_off + d];
        sq_sum += v * v;
    }

    __shared__ double warp_sums[32];
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    for (unsigned int off_s = 16u; off_s > 0u; off_s >>= 1u) {
        sq_sum += __shfl_down_sync(0xFFFFFFFFu, sq_sum, off_s);
    }
    if (lane == 0u) {
        warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    const unsigned int num_warps = block_dim >> 5u;
    if (warp == 0u) {
        double w_sum = (tid < num_warps) ? warp_sums[tid] : 0.0;
        for (unsigned int off_s = 16u; off_s > 0u; off_s >>= 1u) {
            w_sum += __shfl_down_sync(0xFFFFFFFFu, w_sum, off_s);
        }
        if (lane == 0u) {
            warp_sums[0u] = w_sum;
        }
    }
    __syncthreads();
    const double total_sq = warp_sums[0u];
    const double mean_sq = total_sq / (double)head_dim;
    const float rms = (float)sqrt(mean_sq + eps);

    for (unsigned int d = tid; d < head_dim; d += block_dim) {
        // The CPU reference `rms_norm_heads_eps` indexes the weight as
        // `weight[d]` — the same `head_dim`-length slice is broadcast across
        // all heads (Gemma3/4 `q_norm.weight` is shape `[head_dim]`). Index
        // `weight[d]`, NOT `weight[head_off + d]`.
        const float w = (has_weight != 0) ? (offset + weight[d]) : 1.0f;
        out_row[head_off + d] = (x_row[head_off + d] / rms) * w;
    }
}
"#;

/// GEGLU SiLU: `out[i] = silu(gate[i]) * up[i]` where
/// `silu(x) = x / (1 + exp(-x))`. One thread per element. The device twin of
/// `larql_compute::cpu::ops::geglu::geglu_silu` and the host
/// `apply_activation_gated(Silu, …)` path in `pipeline.rs`. Element-wise, no
/// reduction — no shared memory or cross-thread coordination.
pub const GEGGLU_SILU_CUDA_SRC: &str = r#"
extern "C" __global__ void geglu_silu(
    const float* gate,
    const float* up,
    float* out,
    unsigned int n)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    const float g = gate[tid];
    out[tid] = (g / (1.0f + expf(-g))) * up[tid];
}
"#;

/// GEGLU GELU-tanh: `out[i] = gelu_tanh(gate[i]) * up[i]` where
/// `gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
/// One thread per element. The device twin of the host
/// `apply_activation_gated(GeluTanh, …)` path.
///
/// The tanh argument is clamped to ±15 before `tanhf`: NVIDIA's `tanhf`
/// uses `(expf(2y) - 1) / (expf(2y) + 1)`, which overflows f32 and returns NaN
/// once `|y|` ≳ 44 (ln(f32_max) / 2). For gate values around ±50 the
/// argument `y` hits ~50 and poisons the activation with NaNs. Clamping at
/// ±15 is exact: `tanhf(15)` differs from 1.0f by < 1e-13, far below f32
/// precision, so the output is unchanged at any representable gate value.
/// The host reference does not clamp (Rust's `tanh` saturates to ±1.0
/// without overflow), so the clamp is a numerical-safe parity-preserving
/// fix, not a behavioural difference.
pub const GEGGLU_GELU_TANH_CUDA_SRC: &str = r#"
extern "C" __global__ void geglu_gelu_tanh(
    const float* gate,
    const float* up,
    float* out,
    unsigned int n)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    const float g = gate[tid];
    // GELU with tanh approximation:
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c = 0.7978845608f; // sqrt(2/pi)
    float y = c * (g + 0.044715f * g * g * g);
    y = fminf(fmaxf(y, -15.0f), 15.0f);
    const float t = tanhf(y);
    out[tid] = (0.5f * g * (1.0f + t)) * up[tid];
}
"#;

/// Standard (non-gated) SiLU activation: `out[i] = silu(x[i]) = x[i] / (1 +
/// exp(-x[i]))`. One thread per element. The device twin of the host
/// `apply_activation_std(Silu, …)` path. Used when `ffn_type == Standard`
/// (up → activation → down, no gate).
pub const ACTIVATION_SILU_CUDA_SRC: &str = r#"
extern "C" __global__ void activation_silu(
    const float* input,
    float* out,
    unsigned int n)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    const float x = input[tid];
    out[tid] = x / (1.0f + expf(-x));
}
"#;

/// Standard (non-gated) GELU-tanh activation: `out[i] = gelu_tanh(x[i])`.
/// One thread per element. The device twin of the host
/// `apply_activation_std(GeluTanh, …)` path. The tanh argument is clamped
/// to ±15 (see `geglu_gelu_tanh` for the rationale).
pub const ACTIVATION_GELU_TANH_CUDA_SRC: &str = r#"
extern "C" __global__ void activation_gelu_tanh(
    const float* input,
    float* out,
    unsigned int n)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    const float x = input[tid];
    const float c = 0.7978845608f; // sqrt(2/pi)
    float y = c * (x + 0.044715f * x * x * x);
    y = fminf(fmaxf(y, -15.0f), 15.0f);
    const float t = tanhf(y);
    out[tid] = 0.5f * x * (1.0f + t);
}
"#;

/// Scaled residual add: `out[i] = h[i] + b_scale * x[i]`. One thread per
/// element. The device twin of the host `add_residual` path in
/// `pipeline.rs` (`h + x` when `b_scale == 1.0`, else `h + b_scale * x` —
/// numerically identical to the single fused `h + b_scale * x` form, so no
/// branch is needed on the device). Element-wise, no reduction — no shared
/// memory or cross-thread coordination.
pub const RESIDUAL_ADD_CUDA_SRC: &str = r#"
extern "C" __global__ void residual_add(
    const float* h,
    const float* x,
    float* out,
    const float b_scale,
    unsigned int n)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    out[tid] = h[tid] + b_scale * x[tid];
}
"#;

/// Rotary Position Embedding (RoPE) with split-half pairing — the device
/// twin of `larql_compute::attention::rope::apply_rope_partial_at_full`.
/// One thread per output element over the full `[seq_len, num_heads *
/// head_dim]` Q/K tensor. For each element's `(row, head, channel)`:
///
/// - channels in `[0, half_rotary)` → first of the rotary pair:
///   `out = x0*cos(theta) - x1*sin(theta)`
/// - channels in `[half_rotary, 2*half_rotary)` → second of the pair:
///   `out = x0*sin(theta) + x1*cos(theta)`
/// - channels `>= 2*half_rotary` → pass-through (`out = x`)
///
/// where `x0 = x[offset + i]`, `x1 = x[offset + half_rotary + i]`,
/// `i = channel` (first) or `channel - half_rotary` (second), and
/// `theta = pos * inv_freq[i]` with `pos = (row + position_offset) /
/// position_divisor`. `2*half_rotary` (not `rotary_dim`) bounds the rotary
/// region so an odd `rotary_dim` leaves its trailing channel as
/// pass-through, exactly mirroring the host loop (which iterates
/// `i in 0..half_rotary` and writes two channels per `i`).
///
/// The `inv_freq[half_rotary]` array is precomputed on the host with the
/// exact formula the reference uses (`1 / base^(2i/rotary_dim)`, optionally
/// wavelength-band-rescaled by HF's `llama3` variant) and uploaded as
/// `double`, so the per-element `theta`/`cos`/`sin` are computed in f64 and
/// only the final cos/sin are narrowed to f32 — matching the reference's
/// `theta.cos() as f32`. With `fmad` disabled at NVRTC compile time the
/// f32 rotation arithmetic is identical to the host path.
pub const ROPE_CUDA_SRC: &str = r#"
extern "C" __global__ void rope(
    const float* x,
    const double* inv_freq,
    float* out,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int half_rotary,
    double position_offset,
    double position_divisor,
    unsigned long long n)
{
    const unsigned long long tid = (unsigned long long)blockIdx.x
        * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    if (tid >= n) {
        return;
    }
    // 64-bit row/head/channel decomposition so a large vocab-head tensor
    // (`num_heads * head_dim * seq_len > 2^32`) can't wrap the flat index
    // (the host launcher additionally guards `n <= u32::MAX` for the grid).
    const unsigned long long row_stride =
        (unsigned long long)num_heads * (unsigned long long)head_dim;
    if (row_stride == 0ULL) {
        return;
    }
    const unsigned long long row = tid / row_stride;
    if (row >= (unsigned long long)seq_len) {
        return;
    }
    const unsigned long long rem = tid - row * row_stride;
    const unsigned long long head = rem / (unsigned long long)head_dim;
    const unsigned long long channel = rem - head * (unsigned long long)head_dim;
    const unsigned long long base =
        row * row_stride + head * (unsigned long long)head_dim;

    // Channels beyond the rotary region pass through unchanged.
    if (channel >= (unsigned long long)2u * (unsigned long long)half_rotary) {
        out[base + channel] = x[base + channel];
        return;
    }

    // `pos` in f64 for parity with the host reference; `position_divisor`
    // is guarded `> 0.0` on the host before launch.
    const double pos = ((double)row + position_offset) / position_divisor;

    unsigned int i;
    if (channel < (unsigned long long)half_rotary) {
        i = (unsigned int)channel;
    } else {
        i = (unsigned int)(channel - (unsigned long long)half_rotary);
    }
    const double theta = pos * inv_freq[i];
    const float cos_t = (float)cos(theta);
    const float sin_t = (float)sin(theta);

    const float x0 = x[base + i];
    const float x1 = x[base + (unsigned long long)half_rotary + i];
    if (channel < (unsigned long long)half_rotary) {
        out[base + i] = x0 * cos_t - x1 * sin_t;
    } else {
        out[base + (unsigned long long)half_rotary + i] = x0 * sin_t + x1 * cos_t;
    }
}
"#;

/// Fused decode-step GQA attention — the device twin of
/// `larql_compute::attention::decode::gqa_attention_decode_step`. One
/// thread-block per query head; the block collaboratively computes
/// QKᵀ → scale (+ optional softcap) → softmax → weighted-V, all in a single
/// launch (collapsing the host `k_block.dot(&q_row)` + softmax + `v_block.t().
/// dot(&scores)` into one kernel over the full KV cache).
///
/// Numerics mirror the reference exactly:
/// - QKᵀ dot + scale: f32 accumulation in `d = 0..head_dim` order.
/// - softcap (when `has_softcap != 0`): `tanhf(s / softcap) * softcap` (f32,
///   matching the reference's `(s / cap).tanh() * cap`).
/// - softmax: `max_val` (f32), `exp((s - max) as f64)` narrowed to f32 for
///   storage, f64 sum reduction, `inv_sum = (1.0 / sum) as f32`, then
///   `scores[i] *= inv_sum` (normalize-before-dot, matching the reference's
///   normalize-then-`v_block.t().dot` rounding order).
/// - weighted-V: f32 accumulation in `i = 0..total_len` order.
///
/// With `fmad` disabled at NVRTC compile time the f32 arithmetic matches the
/// host path; the only divergence is `exp`/`tanhf` libm differences (parity
/// gated, ≤ a small tolerance). The kernel indexes K/V/scores rows in 64-bit
/// and the host launcher guards all dims against `u32::MAX` before upload.
pub const DECODE_ATTENTION_CUDA_SRC: &str = r#"
#define DECODE_ATTN_BLOCK 256u

extern "C" __global__ void decode_attention(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* scores,
    float* out,
    const float scale,
    const float softcap,
    const unsigned int has_softcap,
    const unsigned int num_q,
    const unsigned int head_dim,
    const unsigned int kv_dim,
    const unsigned int reps,
    const unsigned int total_len)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int h = blockIdx.x;
    if (h >= num_q) {
        return;
    }
    if (total_len == 0u) {
        return;
    }

    const unsigned int q_off = h * head_dim;
    const unsigned int kv_h = h / reps;
    const unsigned int kv_off = kv_h * head_dim;
    const unsigned int score_base = h * total_len;

    __shared__ float shm_max[DECODE_ATTN_BLOCK];
    __shared__ double shm_sum[DECODE_ATTN_BLOCK];

    // Phase 1: QK^T dot * scale (+ optional softcap) -> scores[].
    for (unsigned int i = tid; i < total_len; i += DECODE_ATTN_BLOCK) {
        const unsigned long long k_base =
            (unsigned long long)i * (unsigned long long)kv_dim
            + (unsigned long long)kv_off;
        float dot = 0.0f;
        for (unsigned int d = 0u; d < head_dim; ++d) {
            dot += k_cache[k_base + (unsigned long long)d] * q[q_off + d];
        }
        float s = dot * scale;
        if (has_softcap != 0u) {
            s = tanhf(s / softcap) * softcap;
        }
        scores[score_base + i] = s;
    }
    __syncthreads();

    // Phase 2: max reduction (f32) over this head's scores.
    float local_max = __int_as_float(0xff800000); /* negative infinity */
    for (unsigned int i = tid; i < total_len; i += DECODE_ATTN_BLOCK) {
        const float s = scores[score_base + i];
        if (s > local_max) {
            local_max = s;
        }
    }
    shm_max[tid] = local_max;
    __syncthreads();
    for (unsigned int s2 = DECODE_ATTN_BLOCK / 2u; s2 > 0u; s2 >>= 1u) {
        if (tid < s2) {
            const float other = shm_max[tid + s2];
            if (other > shm_max[tid]) {
                shm_max[tid] = other;
            }
        }
        __syncthreads();
    }
    const float max_val = shm_max[0];
    __syncthreads();

    // Phase 3: exp((s - max) as f64) -> scores[], f64 sum reduction.
    double local_sum = 0.0;
    for (unsigned int i = tid; i < total_len; i += DECODE_ATTN_BLOCK) {
        const float s = scores[score_base + i];
        const double e = exp((double)(s - max_val));
        scores[score_base + i] = (float)e;
        local_sum += e;
    }
    shm_sum[tid] = local_sum;
    __syncthreads();
    for (unsigned int s2 = DECODE_ATTN_BLOCK / 2u; s2 > 0u; s2 >>= 1u) {
        if (tid < s2) {
            shm_sum[tid] += shm_sum[tid + s2];
        }
        __syncthreads();
    }
    const double sum = shm_sum[0];
    const float inv_sum = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
    __syncthreads();

    // Phase 4: normalize scores *= inv_sum (matches the reference's
    // normalize-then-dot rounding order).
    for (unsigned int i = tid; i < total_len; i += DECODE_ATTN_BLOCK) {
        scores[score_base + i] *= inv_sum;
    }
    __syncthreads();

    // Phase 5: weighted V (f32 accumulation) -> out[q_off + d].
    for (unsigned int d = tid; d < head_dim; d += DECODE_ATTN_BLOCK) {
        float acc = 0.0f;
        for (unsigned int i = 0u; i < total_len; ++i) {
            const unsigned long long v_base =
                (unsigned long long)i * (unsigned long long)kv_dim
                + (unsigned long long)kv_off;
            acc += scores[score_base + i] * v_cache[v_base + (unsigned long long)d];
        }
        out[q_off + d] = acc;
    }
}
"#;

/// Native fused prefill (seq×seq) causal GQA attention — the device twin of
/// `larql_compute::attention::gqa::gqa_attention_capture` (the symmetric path
/// reached via `gqa_attention_with_weights`). The prefill twin of
/// [`DECODE_ATTENTION_CUDA_SRC`]: one thread-block per `(query head h, query
/// position qi)` (`gridDim = (num_q, seq_len)`); each block collaboratively
/// fuses the causal QKᵀ → scale (+ optional softcap) → softmax → weighted-V
/// over the keys `0..=qi` (the decode kernel attends over the *full* KV cache;
/// prefill is causal so each query position only sees `causal_len = qi + 1`
/// keys).
///
/// Shared-memory layout (dynamic `extern __shared__`, sized `3072 + seq_len*4`
/// bytes by the host launcher): `shm_sum[256]` doubles at offset 0,
/// `shm_max[256]` floats at offset `256*8`, then a `seq_len`-entry `scores`
/// scratch (one block per `(h, qi)` so the scratch is block-local — no
/// `seq_len*num_q*seq_len` global allocation).
///
/// Five phases per `(h, qi)`, mirroring `decode_attention`'s structure but
/// bounded by `causal_len` instead of `total_len`:
/// - QKᵀ → scale (+ optional `tanhf` softcap) → `scores[i]` (f32).
/// - f32 max via a 256-slot `__shared__` tree reduction (stride loop over
///   `causal_len`).
/// - `exp((s - max) as f64)` → `scores[i]` (f32) + f64 sum reduction over a
///   256-slot `__shared__ double` array (matches the reference's f64 sum).
/// - normalize `scores[i] *= inv_sum` (normalize-before-dot, matching the
///   reference's normalize-then-`v_block.t().dot` rounding order).
/// - weighted-V `out[qi, q_off + d] = Σ_i scores[i]*V[i, kv_off + d]` in f32,
///   `d = 0..head_dim` order.
///
/// With `fmad` disabled at NVRTC compile time the f32 arithmetic matches the
/// reference; the only divergence is the device `exp`/`tanhf` libm (parity
/// gated, ≤ a small tolerance). The kernel indexes Q/K/V/out rows in 64-bit
/// and the host launcher guards all dims against `u32::MAX` + the shared-mem
/// budget (≤ 48 KB, i.e. `seq_len` small enough to fit) before upload.
pub const PREFILL_ATTENTION_CUDA_SRC: &str = r#"
#define PREFILL_ATTN_BLOCK 256u

extern "C" __global__ void prefill_attention(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const float scale,
    const float softcap,
    const unsigned int has_softcap,
    const unsigned int num_q,
    const unsigned int head_dim,
    const unsigned int kv_dim,
    const unsigned int reps,
    const unsigned int seq_len)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int h = blockIdx.x;
    const unsigned int qi = blockIdx.y;
    if (h >= num_q || qi >= seq_len) {
        return;
    }
    if (seq_len == 0u) {
        return;
    }

    const unsigned int causal_len = qi + 1u;
    const unsigned int q_off = h * head_dim;
    const unsigned int kv_h = h / reps;
    const unsigned int kv_off = kv_h * head_dim;
    // q is [seq, num_q*head_dim]; q_dim computed in 64-bit (host guards the
    // 32-bit limit; this multiplication must not wrap).
    const unsigned long long q_dim = (unsigned long long)num_q * (unsigned long long)head_dim;
    const unsigned long long q_row = (unsigned long long)qi * q_dim + (unsigned long long)q_off;
    const unsigned long long out_row = (unsigned long long)qi * q_dim + (unsigned long long)q_off;

    // Dynamic shared layout: shm_sum[256] double | shm_max[256] float | scores[seq_len] float.
    extern __shared__ unsigned char sbuf_raw[];
    double* shm_sum = (double*)sbuf_raw;
    float* shm_max = (float*)(sbuf_raw + 256u * 8u);
    float* scores = (float*)(sbuf_raw + 256u * 8u + 256u * 4u);

    // Phase 1: causal QK^T dot * scale (+ optional softcap) -> scores[i].
    for (unsigned int i = tid; i < causal_len; i += PREFILL_ATTN_BLOCK) {
        const unsigned long long k_base =
            (unsigned long long)i * (unsigned long long)kv_dim
            + (unsigned long long)kv_off;
        float dot = 0.0f;
        for (unsigned int d = 0u; d < head_dim; ++d) {
            dot += k[k_base + (unsigned long long)d] * q[q_row + (unsigned long long)d];
        }
        float s = dot * scale;
        if (has_softcap != 0u) {
            s = tanhf(s / softcap) * softcap;
        }
        scores[i] = s;
    }
    __syncthreads();

    // Phase 2: max reduction (f32) over this head's causal scores.
    float local_max = __int_as_float(0xff800000); /* negative infinity */
    for (unsigned int i = tid; i < causal_len; i += PREFILL_ATTN_BLOCK) {
        const float s = scores[i];
        if (s > local_max) {
            local_max = s;
        }
    }
    shm_max[tid] = local_max;
    __syncthreads();
    for (unsigned int s2 = PREFILL_ATTN_BLOCK / 2u; s2 > 0u; s2 >>= 1u) {
        if (tid < s2) {
            const float other = shm_max[tid + s2];
            if (other > shm_max[tid]) {
                shm_max[tid] = other;
            }
        }
        __syncthreads();
    }
    const float max_val = shm_max[0];
    __syncthreads();

    // Phase 3: exp((s - max) as f64) -> scores[], f64 sum reduction.
    double local_sum = 0.0;
    for (unsigned int i = tid; i < causal_len; i += PREFILL_ATTN_BLOCK) {
        const float s = scores[i];
        const double e = exp((double)(s - max_val));
        scores[i] = (float)e;
        local_sum += e;
    }
    shm_sum[tid] = local_sum;
    __syncthreads();
    for (unsigned int s2 = PREFILL_ATTN_BLOCK / 2u; s2 > 0u; s2 >>= 1u) {
        if (tid < s2) {
            shm_sum[tid] += shm_sum[tid + s2];
        }
        __syncthreads();
    }
    const double sum = shm_sum[0];
    const float inv_sum = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
    __syncthreads();

    // Phase 4: normalize scores *= inv_sum (matches the reference's
    // normalize-then-dot rounding order).
    for (unsigned int i = tid; i < causal_len; i += PREFILL_ATTN_BLOCK) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Phase 5: weighted V (f32 accumulation) -> out[qi, q_off + d].
    for (unsigned int d = tid; d < head_dim; d += PREFILL_ATTN_BLOCK) {
        float acc = 0.0f;
        for (unsigned int i = 0u; i < causal_len; ++i) {
            const unsigned long long v_base =
                (unsigned long long)i * (unsigned long long)kv_dim
                + (unsigned long long)kv_off;
            acc += scores[i] * v[v_base + (unsigned long long)d];
        }
        out[out_row + (unsigned long long)d] = acc;
    }
}
"#;

// ── LARQL-GPU-B4 greedy top-K reduction kernels ────────────────────────────
//
// Two-stage bounded reduction over the logical-vocabulary score vector
// produced by the resident Q4_K lm-head GEMV. Stage 1 (partial) reduces
// each `GREEDY_BLOCK_SIZE`-element slice to its top-`k` finite candidates;
// stage 2 (final) reduces the per-block partials to the final sorted top-`k`.
//
// Design constraints (see the B4 packet §8.2):
//   * No floating-point atomics as the primary reduction design. Each
//     stage does `k` sequential argmax passes (k ≤ GREEDY_MAX_K = 8), each
//     pass a standard 256-way max reduction in shared memory.
//   * Strict `>` comparison so the lowest-index row wins on ties, matching
//     the host `top_k_sorted` / `backend_lm_head_topk` contract.
//   * Non-finite scores (NaN, ±∞) are treated as `-INFINITY` candidates
//     and never win; a slice with fewer than `k` finite values fills the
//     remaining partial slots with the `-INFINITY` / `0xFFFFFFFF` sentinel.
//   * The reduction operates ONLY over `[0, logical_rows)` — padded
//     physical rows above `logical_rows` are never read, so an invalid
//     padded row can never win (VINDEX-001 / B4 §2.4).

/// Partial top-K reduction. Grid: `ceil(logical_rows / GREEDY_BLOCK_SIZE)`
/// blocks, `GREEDY_BLOCK_SIZE` threads per block. Each block writes `k`
/// `(score, id)` pairs into `partial_scores[bid*k .. bid*k+k]` and
/// `partial_ids[bid*k .. bid*k+k]`. Unfilled slots are the
/// `(-INFINITY, 0xFFFFFFFF)` sentinel.
pub const GREEDY_TOPK_PARTIAL_CUDA_SRC: &str = r#"
#define GREEDY_BLK 256u
// Device-safe negative infinity (NVRTC doesn't define the C99 INFINITY
// macro under device compilation). Matches `decode_attention`'s
// `__int_as_float(0xff800000)` sentinel.
#define NEG_INF (__int_as_float(0xff800000u))

// B4-CORRECTION B: deterministic tie-break contract, shared with the host
// reference and `greedy_topk_final`. Candidate (bv, bt) ranks ahead of
// (av, at) when its score is strictly greater, OR the scores are equal and
// its global token id is strictly smaller. Non-finite scores never reach
// this comparator (they are masked to NEG_INF and excluded by `win_val >
// NEG_INF`). The partial reduction carries the GLOBAL TOKEN ID (not the
// lane / partial-buffer position) through every comparison, so an exact
// score tie always resolves to the lower token id regardless of CUDA
// thread, block, or strided-scan order.
#define BETTER(bv, bt, av, at) \
    (((bv) > (av)) || (((bv) == (av)) && ((bt) < (at))))

extern "C" __global__ void greedy_topk_partial(
    const float* scores,
    float* partial_scores,
    unsigned int* partial_ids,
    const unsigned int logical_rows,
    const unsigned int k)
{
    const unsigned int bid = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int base = bid * GREEDY_BLK;

    // Load this block's slice into shared memory. Out-of-range slots AND
    // non-finite scores (NaN, ±∞) become NEG_INF so they can never win —
    // matching the host `top_k_sorted` / `backend_lm_head_topk` `is_finite()`
    // filter (B4 §8.2).
    __shared__ float shm[GREEDY_BLK];
    {
        const unsigned int idx = base + tid;
        const float v = (idx < logical_rows) ? scores[idx] : NEG_INF;
        shm[tid] = isfinite(v) ? v : NEG_INF;
    }
    __syncthreads();

    // k sequential reductions over the slice. Each pass: copy the (masked)
    // slice into a reduction buffer carrying the GLOBAL token id, reduce to
    // a single winner with the lexicographic (score desc, id asc)
    // comparator, then mask the winner at NEG_INF for the next pass.
    __shared__ float red_val[GREEDY_BLK];
    __shared__ unsigned int red_tok[GREEDY_BLK];

    for (unsigned int kk = 0u; kk < k; ++kk) {
        // Initialise the reduction buffer from the (masked) slice. The
        // token id carried is the GLOBAL id (base + lane), so the
        // comparator breaks ties on the real vocabulary index, not the
        // lane-local position.
        red_val[tid] = shm[tid];
        red_tok[tid] = base + tid;
        __syncthreads();

        // In-place tree reduction (256 -> 1) keeping the lexicographic best.
        for (unsigned int off = GREEDY_BLK / 2u; off > 0u; off >>= 1u) {
            if (tid < off) {
                const float av = red_val[tid];
                const float bv = red_val[tid + off];
                const unsigned int at = red_tok[tid];
                const unsigned int bt = red_tok[tid + off];
                if (BETTER(bv, bt, av, at)) {
                    red_val[tid] = bv;
                    red_tok[tid] = bt;
                }
            }
            __syncthreads();
        }

        // Thread 0 records the winner and masks it for the next pass.
        if (tid == 0u) {
            const float win_val = red_val[0u];
            const unsigned int win_tok = red_tok[0u];
            const unsigned int slot = bid * k + kk;
            if (win_val > NEG_INF) {
                partial_scores[slot] = win_val;
                partial_ids[slot] = win_tok;
                // Recover the within-slice position to mask the winner.
                // win_tok is in [base, base+GREEDY_BLK) ∩ [0, logical_rows)
                // because only finite in-range lanes can win.
                const unsigned int local_idx = win_tok - base;
                shm[local_idx] = NEG_INF;
            } else {
                // No finite candidate left in this slice; sentinel.
                partial_scores[slot] = NEG_INF;
                partial_ids[slot] = 0xFFFFFFFFu;
            }
        }
        __syncthreads();
    }
}
"#;

/// Final top-K reduction. Single block, `GREEDY_BLK` threads. Strided
/// scan over the `num_partials = num_blocks * k` partial candidates,
/// `k` sequential passes (each thread keeps a register-local best, then a
/// 256-way shared-memory reduction). Writes the final sorted (descending)
/// top-`k` into `result_scores[k]` / `result_ids[k]`. Unfilled slots (fewer
/// than `k` finite candidates overall) are the `(-INFINITY, 0xFFFFFFFF)`
/// sentinel.
///
/// B4-CORRECTION B: the comparator is the explicit lexicographic
/// (score desc, global token id asc) order shared with `greedy_topk_partial`
/// and the host reference. The strided per-thread scan and the shared
/// reduction BOTH carry the global token id (read from `partial_ids`), so
/// an exact score tie always resolves to the lower token id regardless of
/// partial-buffer position, CUDA thread, or strided-scan order. The prior
/// score-only final comparator compared only `partial_scores` and broke
/// ties on partial-buffer position, which could select the wrong token id
/// when two partial candidates from different blocks shared a score.
pub const GREEDY_TOPK_FINAL_CUDA_SRC: &str = r#"
#define GREEDY_BLK 256u
#define NEG_INF (__int_as_float(0xff800000u))

// Same lexicographic tie-break contract as greedy_topk_partial.
#define BETTER(bv, bt, av, at) \
    (((bv) > (av)) || (((bv) == (av)) && ((bt) < (at))))

extern "C" __global__ void greedy_topk_final(
    float* partial_scores,
    const unsigned int* partial_ids,
    float* result_scores,
    unsigned int* result_ids,
    const unsigned int num_partials,
    const unsigned int k)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int block_dim = blockDim.x;

    // k sequential passes over the partial buffer. Each pass: every thread
    // strides over the buffer keeping its register-local best (score +
    // global token id + partial-buffer position), then a 256-way shared
    // reduction picks the global winner with the lexicographic comparator.
    // Thread 0 records the winner and masks it (set to NEG_INF) for the next
    // pass. The global token id drives the tie-break; the partial-buffer
    // position is carried only so the winner can be masked.
    __shared__ float red_val[GREEDY_BLK];
    __shared__ unsigned int red_pos[GREEDY_BLK];
    __shared__ unsigned int red_tok[GREEDY_BLK];

    for (unsigned int kk = 0u; kk < k; ++kk) {
        // Per-thread register-local best (init to the sentinel so unfilled
        // partials never win).
        float best_val = NEG_INF;
        unsigned int best_pos = 0xFFFFFFFFu;
        unsigned int best_tok = 0xFFFFFFFFu;

        for (unsigned int i = tid; i < num_partials; i += block_dim) {
            const float v = partial_scores[i];
            // Skip non-finite (NaN/±∞) — host `is_finite()` contract.
            if (isfinite(v)) {
                const unsigned int v_tok = partial_ids[i];
                if (BETTER(v, v_tok, best_val, best_tok)) {
                    best_val = v;
                    best_pos = i;
                    best_tok = v_tok;
                }
            }
        }

        red_val[tid] = best_val;
        red_pos[tid] = best_pos;
        red_tok[tid] = best_tok;
        __syncthreads();

        // 256-way tree reduction keeping the lexicographic best.
        for (unsigned int off = block_dim / 2u; off > 0u; off >>= 1u) {
            if (tid < off) {
                const float av = red_val[tid];
                const float bv = red_val[tid + off];
                const unsigned int at = red_tok[tid];
                const unsigned int bt = red_tok[tid + off];
                if (BETTER(bv, bt, av, at)) {
                    red_val[tid] = bv;
                    red_pos[tid] = red_pos[tid + off];
                    red_tok[tid] = bt;
                }
            }
            __syncthreads();
        }

        // Thread 0 records the winner and masks it in the partial buffer.
        if (tid == 0u) {
            const float win_val = red_val[0u];
            const unsigned int win_pos = red_pos[0u];
            const unsigned int win_tok = red_tok[0u];
            if (win_pos != 0xFFFFFFFFu && win_val > NEG_INF) {
                result_scores[kk] = win_val;
                result_ids[kk] = win_tok;
                // Mask the winner so it can't win the next pass.
                partial_scores[win_pos] = NEG_INF;
            } else {
                result_scores[kk] = NEG_INF;
                result_ids[kk] = 0xFFFFFFFFu;
            }
        }
        __syncthreads();
    }
}
"#;

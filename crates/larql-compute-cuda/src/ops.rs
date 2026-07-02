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

pub const Q4K_MATVEC_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>

union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}

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

union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}

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

union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}

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

union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}

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

union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}

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

union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}

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

union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}

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

union larql_half_bits {
    unsigned short bits;
    __half half;
};

__device__ __forceinline__ float larql_decode_f16(unsigned short bits) {
    larql_half_bits value;
    value.bits = bits;
    return __half2float(value.half);
}

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

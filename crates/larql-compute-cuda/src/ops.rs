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

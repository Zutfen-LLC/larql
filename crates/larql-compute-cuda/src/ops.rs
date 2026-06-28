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
        threads_per_group: [256, 1, 1],
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

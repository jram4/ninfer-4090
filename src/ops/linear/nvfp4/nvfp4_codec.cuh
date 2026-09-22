#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ float decode_nvfp4_e2m1(std::uint8_t code) {
    const unsigned magnitude = code & 0x7U;
    float value = 0.0F;
    switch (magnitude) {
    case 0: value = 0.0F; break;
    case 1: value = 0.5F; break;
    case 2: value = 1.0F; break;
    case 3: value = 1.5F; break;
    case 4: value = 2.0F; break;
    case 5: value = 3.0F; break;
    case 6: value = 4.0F; break;
    default: value = 6.0F; break;
    }
    return (code & 0x8U) != 0U ? -value : value;
}

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    return make_float2(decode_nvfp4_e2m1(storage & 0x0FU),
                       decode_nvfp4_e2m1((storage >> 4) & 0x0FU));
}

__device__ __forceinline__ std::uint8_t encode_nvfp4_e2m1(float value) {
    if (isnan(value)) return 0x7U; // CUDA FP4 conversion maps NaN to +MAXNORM.
    const bool negative = signbit(value) && value != 0.0F;
    const float x = fabsf(value);
    unsigned magnitude;
    // Midpoint ties select the even E2M1 code, matching cudaRoundNearest.
    if (x <= 0.25F) magnitude = 0;
    else if (x < 0.75F) magnitude = 1;
    else if (x <= 1.25F) magnitude = 2;
    else if (x < 1.75F) magnitude = 3;
    else if (x <= 2.5F) magnitude = 4;
    else if (x < 3.5F) magnitude = 5;
    else if (x <= 5.0F) magnitude = 6;
    else magnitude = 7;
    return static_cast<std::uint8_t>(magnitude | (negative ? 0x8U : 0U));
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
    std::uint8_t bytes[8];
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const std::uint8_t lo = encode_nvfp4_e2m1(values[pair].x);
        const std::uint8_t hi = encode_nvfp4_e2m1(values[pair].y);
        bytes[pair] = static_cast<std::uint8_t>(lo | (hi << 4));
    }
    codes_lo = static_cast<std::uint32_t>(bytes[0]) |
               (static_cast<std::uint32_t>(bytes[1]) << 8) |
               (static_cast<std::uint32_t>(bytes[2]) << 16) |
               (static_cast<std::uint32_t>(bytes[3]) << 24);
    codes_hi = static_cast<std::uint32_t>(bytes[4]) |
               (static_cast<std::uint32_t>(bytes[5]) << 8) |
               (static_cast<std::uint32_t>(bytes[6]) << 16) |
               (static_cast<std::uint32_t>(bytes[7]) << 24);
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
        values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

} // namespace ninfer::ops::detail

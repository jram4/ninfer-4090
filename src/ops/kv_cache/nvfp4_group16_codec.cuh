#pragma once

// D256 KV-cache NVFP4 codec. Unlike weight artifacts, cache rows have no matrix-level divisor:
// every represented value is exactly E2M1(code) * E4M3(scale) for its contiguous G16 group.

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheNvfp4HeadDim        = 256;
inline constexpr int kKVCacheNvfp4Group          = 16;
inline constexpr int kKVCacheNvfp4Groups         = 16;
inline constexpr int kKVCacheNvfp4CodeBytes      = 128;
inline constexpr float kKVCacheNvfp4MaxFinite    = 6.0F;
inline constexpr float kKVCacheNvfp4ScaleMinimum = 0x1p-9F;
inline constexpr float kKVCacheNvfp4ScaleMaximum = 448.0F;

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_nvfp4_code_index(int physical_page, int kv_head,
                                                                  int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheNvfp4CodeBytes, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d >> 1);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_nvfp4_scale_index(int physical_page, int kv_head,
                                                                   int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheNvfp4Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                           page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_nvfp4_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheNvfp4HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheNvfp4QuantizedGroup16 {
    std::uint32_t codes_lo = 0;
    std::uint32_t codes_hi = 0;
    std::uint8_t scale     = 0;
};

__device__ __forceinline__ KVCacheNvfp4QuantizedGroup16
kv_cache_nvfp4_quantize_group16(const float* source) {
    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = make_float2(source[2 * pair], source[2 * pair + 1]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    KVCacheNvfp4QuantizedGroup16 result{};
    if (max_abs == 0.0F) return result;

    const float raw_scale = __fdiv_rn(max_abs, kKVCacheNvfp4MaxFinite);
    const float bounded =
        fminf(kKVCacheNvfp4ScaleMaximum, fmaxf(kKVCacheNvfp4ScaleMinimum, raw_scale));
    result.scale                  = __nv_cvt_float_to_fp8(bounded, __NV_SATFINITE, __NV_E4M3);
    const float represented_scale = detail::decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x, represented_scale);
        values[pair].y = __fdiv_rn(values[pair].y, represented_scale);
    }
    detail::pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

// Placing the three E2M1 magnitude bits in FP16 bits [11:9] yields exactly value * 2^-14 for
// every code, including the E2M1 subnormal 0.5, which lands on the FP16 subnormal 2^-15.
__device__ __forceinline__ __half2 kv_cache_nvfp4_e2m1x2_scaled_f16x2(std::uint32_t byte) {
    const std::uint32_t bits = ((byte & 0x7U) << 9) | ((byte & 0x8U) << 12) |
                               ((byte & 0x70U) << 21) | ((byte & 0x80U) << 24);
    return *reinterpret_cast<const __half2*>(&bits);
}

__device__ __forceinline__ __half2 kv_cache_nvfp4_scale_f16x2(std::uint8_t scale_code) {
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x  = scale_code;
    const __half scale = static_cast<__half>(encoded_scale);
    return __halves2half2(scale, scale);
}

__device__ __forceinline__ unsigned kv_cache_nvfp4_dequant_f16x2(std::uint32_t byte,
                                                                 __half2 scale2) {
    const __half2 unit = __float2half2_rn(0x1p14F);
    const __half2 value =
        __hmul2(__hmul2(kv_cache_nvfp4_e2m1x2_scaled_f16x2(byte), unit), scale2);
    return *reinterpret_cast<const unsigned*>(&value);
}

__device__ __forceinline__ int4 kv_cache_nvfp4_dequant_f16x8(const std::uint8_t* codes,
                                                             std::uint8_t scale_code) {
    // Every finite E2M1 value times a legal nonnegative E4M3 cache scale is exactly representable
    // in FP16 (at most four product fraction bits and magnitude <= 2688). Half2 multiplication is
    // therefore the exact FP16 expansion boundary, not an additional approximation.
    const std::uint32_t packed = load_vec<std::uint32_t>(codes);
    const __half2 scale2       = kv_cache_nvfp4_scale_f16x2(scale_code);
    return make_int4(static_cast<int>(kv_cache_nvfp4_dequant_f16x2(packed, scale2)),
                     static_cast<int>(kv_cache_nvfp4_dequant_f16x2(packed >> 8, scale2)),
                     static_cast<int>(kv_cache_nvfp4_dequant_f16x2(packed >> 16, scale2)),
                     static_cast<int>(kv_cache_nvfp4_dequant_f16x2(packed >> 24, scale2)));
}

struct KVCacheNvfp4DequantizedF16x16 {
    int4 lo;
    int4 hi;
};

__device__ __forceinline__ KVCacheNvfp4DequantizedF16x16
kv_cache_nvfp4_dequant_f16x16(const std::uint8_t* codes, std::uint8_t scale_code) {
    const int2 packed    = load_vec<int2>(codes);
    const __half2 scale2 = kv_cache_nvfp4_scale_f16x2(scale_code);
    unsigned half_bits[8];
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const std::uint32_t word = static_cast<std::uint32_t>(pair < 4 ? packed.x : packed.y);
        half_bits[pair]          = kv_cache_nvfp4_dequant_f16x2(word >> (8 * (pair & 3)), scale2);
    }
    return {
        make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                  static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3])),
        make_int4(static_cast<int>(half_bits[4]), static_cast<int>(half_bits[5]),
                  static_cast<int>(half_bits[6]), static_cast<int>(half_bits[7])),
    };
}

} // namespace ninfer::ops

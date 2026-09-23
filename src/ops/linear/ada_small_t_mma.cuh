#pragma once

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

// Single-pass small-T groupwise GEMM for Ada: out^T[T x N] = x^T[T x K] * W^T[K x N] on
// mma.m16n8k16 with tokens as M (zero-padded to 16) and eight weight rows as N, for the row-split
// Q4_G64_FP16 and Q5_G64_FP16 layouts (32 code bytes per 64-weight group, weight k in byte k / 2,
// low nibble first; Q5 adds 8 high-bit bytes per group, bit b of byte j holding weight 8j + b).
//
// A CTA owns 8 * RowTiles weight rows over the whole K and streams them through a cp.async pipeline
// of GroupsPerStage groups per stage. Global reads are contiguous 16-byte chunks per row; the row
// paddings below keep the fragment reads out of shared memory bank-conflict free. x is L2-resident
// at these widths and is prefetched into registers one stage ahead instead of taking pipeline
// space. Warps split each stage's groups and reduce through shared memory at the end, so every
// weight byte is read once for any T <= 8 * TokenTiles.
//
// Per group a lane owns 16 consecutive weights (8 code bytes) of one row. MMA step s maps K slots
// {2q, 2q+1} to weights 16q + 4s + {0,1} and slots {2q+8, 2q+9} to 16q + 4s + {2,3}; x is read with
// the same permutation, so the dot product is unchanged. Codes decode to exact bf16 integers and
// the fp16 group scale is applied in fp32 after the group's four MMAs.
enum class AdaWeightFormat { Q4, Q5 };

template <AdaWeightFormat Format, int RowTiles, int Warps, int TokenTiles, int GroupsPerStage,
          int Stages>
struct AdaSmallTSchedule {
    static constexpr AdaWeightFormat kFormat = Format;
    static constexpr bool kHasHigh           = Format == AdaWeightFormat::Q5;
    static constexpr int kRowTiles           = RowTiles;
    static constexpr int kWarps              = Warps;
    static constexpr int kTokenTiles         = TokenTiles;
    static constexpr int kG                  = GroupsPerStage;
    static constexpr int kStages             = Stages;
    static constexpr int kRowsPerCta         = 8 * RowTiles;
    static constexpr int kThreads            = 32 * Warps;
    static constexpr int kMaxTokens          = 8 * TokenTiles;
    static constexpr int kGroupsPerWarp      = kG / kWarps;

    static constexpr int kCodeStride  = 32 * kG + 32;
    static constexpr int kHighStride  = kHasHigh ? 8 * kG + 16 : 0;
    static constexpr int kScaleStride = kG <= 8 ? 16 : 48;

    static constexpr int kCodeBytes  = kRowsPerCta * kCodeStride;
    static constexpr int kHighBytes  = kRowsPerCta * kHighStride;
    static constexpr int kScaleBytes = kRowsPerCta * kScaleStride;
    static constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes;
    static constexpr int kReductionBytes =
        kWarps * kRowsPerCta * kMaxTokens * static_cast<int>(sizeof(float));
    static constexpr int kSmemBytes = std::max(kStages * kStageBytes, kReductionBytes);

    static_assert(RowTiles >= 1 && Warps >= 1 && (TokenTiles == 1 || TokenTiles == 2));
    static_assert(kG == 4 || kG == 8 || kG == 16);
    static_assert(kG % kWarps == 0);
    static_assert(kStages >= 2 && kStages <= 8);
};

template <int RowTiles, int Warps, int TokenTiles, int GroupsPerStage, int Stages>
using Q5AdaSmallTSchedule =
    AdaSmallTSchedule<AdaWeightFormat::Q5, RowTiles, Warps, TokenTiles, GroupsPerStage, Stages>;
template <int RowTiles, int Warps, int TokenTiles, int GroupsPerStage, int Stages>
using Q4AdaSmallTSchedule =
    AdaSmallTSchedule<AdaWeightFormat::Q4, RowTiles, Warps, TokenTiles, GroupsPerStage, Stages>;

// CTA b owns weight rows [b * rows, (b + 1) * rows).
struct AdaContiguousRows {
    __device__ __forceinline__ static int weight_row(int block, int local, int rows) {
        return block * rows + local;
    }
};

// CTA b owns output rows [b * rows / 2, (b + 1) * rows / 2) of a gate/up parent whose up half starts
// at weight row Offset: local rows [0, rows / 2) are gate rows and the rest the matching up rows.
template <int Offset>
struct AdaPairedHalfRows {
    __device__ __forceinline__ static int weight_row(int block, int local, int rows) {
        const int half = rows / 2;
        return local < half ? block * half + local : Offset + block * half + local - half;
    }
};

// Two Q5 codes of one byte plus their inverted high bits (bits 0 and 1) -> exact bf16x2 integers.
// bf16 0x4300 is 128 with a unit mantissa step, so 128 + u is exact for u < 128; u = code +
// 16 * (1 - high) and subtracting 144 yields the signed 5-bit value code - 16 * high.
__device__ __forceinline__ unsigned q5_ada_decode_pair(unsigned byte, unsigned inv_high2) {
    const unsigned bits = 0x43004300u | (byte & 0x0fu) | ((byte & 0xf0u) << 12) |
                          ((inv_high2 & 1u) << 4) | ((inv_high2 & 2u) << 19);
    const unsigned bias = 0x43104310u;
    const __nv_bfloat162 value = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits),
                                         *reinterpret_cast<const __nv_bfloat162*>(&bias));
    return *reinterpret_cast<const unsigned*>(&value);
}

// Two signed Q4 codes of one byte -> exact bf16x2 integers: flipping the sign bit gives the offset
// value u = code + 8, and (128 + u) - 136 is the two's-complement code.
__device__ __forceinline__ unsigned q4_ada_decode_pair(unsigned byte) {
    const unsigned bits =
        0x43004300u | (((byte & 0x0fu) | ((byte & 0xf0u) << 12)) ^ 0x00080008u);
    const unsigned bias = 0x43084308u;
    const __nv_bfloat162 value = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits),
                                         *reinterpret_cast<const __nv_bfloat162*>(&bias));
    return *reinterpret_cast<const unsigned*>(&value);
}

// Epilogue::kPairedHalves == false: epilogue(weight_row, token, dot) for every CTA row and live
// token. true: epilogue(output_row, token, gate_dot, up_dot) for the CTA's AdaPairedHalfRows pairs.
template <class S, int K, class RowMap, class Epilogue>
__launch_bounds__(S::kThreads) __global__ void ada_small_t_mma_kernel(
    const __nv_bfloat16* __restrict__ x, std::int32_t x_ld, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    std::int32_t tokens, Epilogue epilogue) {
    constexpr int kGroupK = 64;
    static_assert(K % (kGroupK * S::kG) == 0);
    constexpr int kGroups     = K / kGroupK;
    constexpr int kPipeStages = kGroups / S::kG;
    constexpr int R           = S::kRowTiles;
    constexpr int kRows       = S::kRowsPerCta;

    extern __shared__ __align__(16) std::uint8_t smem[];

    const int tid      = static_cast<int>(threadIdx.x);
    const int lane     = tid & 31;
    const int warp     = tid >> 5;
    const int q        = lane & 3;
    const int group_id = lane >> 2;
    const int block    = static_cast<int>(blockIdx.x);

    const auto issue = [&](int stage) {
        std::uint8_t* code_dst  = smem + (stage % S::kStages) * S::kStageBytes;
        std::uint8_t* high_dst  = code_dst + S::kCodeBytes;
        std::uint8_t* scale_dst = high_dst + S::kHighBytes;
        const int g0            = stage * S::kG;

        constexpr int kCodeChunks = 2 * S::kG;
        for (int i = tid; i < kRows * kCodeChunks; i += S::kThreads) {
            const int row = i / kCodeChunks;
            const int c   = i % kCodeChunks;
            const std::int64_t src_row = RowMap::weight_row(block, row, kRows);
            cp_async<16, Cache::cg>(code_dst + row * S::kCodeStride + 16 * c,
                                    codes + (src_row * kGroups + g0) * 32 + 16 * c);
        }
        if constexpr (S::kHasHigh) {
            constexpr int kHighChunks = S::kG / 2;
            for (int i = tid; i < kRows * kHighChunks; i += S::kThreads) {
                const int row = i / kHighChunks;
                const int c   = i % kHighChunks;
                const std::int64_t src_row = RowMap::weight_row(block, row, kRows);
                cp_async<16, Cache::cg>(high_dst + row * S::kHighStride + 16 * c,
                                        high + (src_row * kGroups + g0) * 8 + 16 * c);
            }
        }
        if constexpr (S::kG == 4) {
            for (int row = tid; row < kRows; row += S::kThreads) {
                const std::int64_t src_row = RowMap::weight_row(block, row, kRows);
                cp_async<8>(scale_dst + row * S::kScaleStride,
                            scales + (src_row * kGroups + g0) * 2);
            }
        } else {
            constexpr int kScaleChunks = S::kG / 8;
            for (int i = tid; i < kRows * kScaleChunks; i += S::kThreads) {
                const int row = i / kScaleChunks;
                const int c   = i % kScaleChunks;
                const std::int64_t src_row = RowMap::weight_row(block, row, kRows);
                cp_async<16, Cache::cg>(scale_dst + row * S::kScaleStride + 16 * c,
                                        scales + (src_row * kGroups + g0) * 2 + 16 * c);
            }
        }
    };

    const bool lo_valid = group_id < tokens;
    const bool hi_valid = S::kTokenTiles == 2 && group_id + 8 < tokens;
    const __nv_bfloat16* x_lo_src = x + static_cast<std::int64_t>(group_id) * x_ld + 16 * q;
    const __nv_bfloat16* x_hi_src = x_lo_src + static_cast<std::int64_t>(8) * x_ld;

    struct XFragment {
        uint4 lo[2];
        uint4 hi[2];
    };
    constexpr int J = S::kGroupsPerWarp;
    const auto load_x = [&](int stage, XFragment (&dst)[J]) {
        const uint4 zero{0u, 0u, 0u, 0u};
#pragma unroll
        for (int j = 0; j < J; ++j) {
            const int k0 = (stage * S::kG + warp + j * S::kWarps) * kGroupK;
            const auto* lo = reinterpret_cast<const uint4*>(x_lo_src + k0);
            const auto* hi = reinterpret_cast<const uint4*>(x_hi_src + k0);
            dst[j].lo[0] = lo_valid ? __ldg(lo) : zero;
            dst[j].lo[1] = lo_valid ? __ldg(lo + 1) : zero;
            dst[j].hi[0] = hi_valid ? __ldg(hi) : zero;
            dst[j].hi[1] = hi_valid ? __ldg(hi + 1) : zero;
        }
    };

    float acc[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[r][i] = 0.0f; }
    }

    const auto compute = [&](int stage, const XFragment (&xf)[J]) {
        const std::uint8_t* code_s  = smem + (stage % S::kStages) * S::kStageBytes;
        const std::uint8_t* high_s  = code_s + S::kCodeBytes;
        const std::uint8_t* scale_s = high_s + S::kHighBytes;
#pragma unroll
        for (int j = 0; j < J; ++j) {
            const int gi           = warp + j * S::kWarps;
            const unsigned a_lo[8] = {xf[j].lo[0].x, xf[j].lo[0].y, xf[j].lo[0].z, xf[j].lo[0].w,
                                      xf[j].lo[1].x, xf[j].lo[1].y, xf[j].lo[1].z, xf[j].lo[1].w};
            const unsigned a_hi[8] = {xf[j].hi[0].x, xf[j].hi[0].y, xf[j].hi[0].z, xf[j].hi[0].w,
                                      xf[j].hi[1].x, xf[j].hi[1].y, xf[j].hi[1].z, xf[j].hi[1].w};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row    = 8 * r + group_id;
                const uint2 word = *reinterpret_cast<const uint2*>(code_s + row * S::kCodeStride +
                                                                   gi * 32 + 8 * q);
                unsigned inv = 0;
                if constexpr (S::kHasHigh) {
                    const unsigned hw = *reinterpret_cast<const unsigned*>(
                        high_s + row * S::kHighStride + gi * 8 + 4 * (q >> 1));
                    inv = ~(hw >> (16 * (q & 1)));
                }
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const unsigned w     = s < 2 ? word.x : word.y;
                    const unsigned shift = 16u * static_cast<unsigned>(s & 1);
                    unsigned b0;
                    unsigned b1;
                    if constexpr (S::kHasHigh) {
                        b0 = q5_ada_decode_pair((w >> shift) & 0xffu, inv >> (4 * s));
                        b1 = q5_ada_decode_pair((w >> (shift + 8u)) & 0xffu, inv >> (4 * s + 2));
                    } else {
                        b0 = q4_ada_decode_pair((w >> shift) & 0xffu);
                        b1 = q4_ada_decode_pair((w >> (shift + 8u)) & 0xffu);
                    }
                    mma_bf16(d0, d1, d2, d3, a_lo[2 * s], a_hi[2 * s], a_lo[2 * s + 1],
                             a_hi[2 * s + 1], b0, b1);
                }
                const int srow = 8 * r + 2 * q;
                const float s0 = __half2float(
                    *reinterpret_cast<const __half*>(scale_s + srow * S::kScaleStride + 2 * gi));
                const float s1 = __half2float(*reinterpret_cast<const __half*>(
                    scale_s + (srow + 1) * S::kScaleStride + 2 * gi));
                acc[r][0] = fmaf(s0, d0, acc[r][0]);
                acc[r][1] = fmaf(s1, d1, acc[r][1]);
                acc[r][2] = fmaf(s0, d2, acc[r][2]);
                acc[r][3] = fmaf(s1, d3, acc[r][3]);
            }
        }
    };

#pragma unroll
    for (int s = 0; s < S::kStages - 1; ++s) {
        if (s < kPipeStages) { issue(s); }
        cp_commit();
    }
    XFragment current[J];
    load_x(0, current);
    for (int s = 0; s < kPipeStages; ++s) {
        cp_wait<S::kStages - 2>();
        __syncthreads();
        if (s + S::kStages - 1 < kPipeStages) { issue(s + S::kStages - 1); }
        cp_commit();
        XFragment next[J];
        if (s + 1 < kPipeStages) { load_x(s + 1, next); }
        compute(s, current);
#pragma unroll
        for (int j = 0; j < J; ++j) { current[j] = next[j]; }
    }
    cp_wait<0>();
    __syncthreads();

    auto* reduction = reinterpret_cast<float*>(smem);
    const auto slot = [](int w, int row, int token) {
        return (w * kRows + row) * S::kMaxTokens + token;
    };
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row                            = 8 * r + 2 * q;
        reduction[slot(warp, row, group_id)]     = acc[r][0];
        reduction[slot(warp, row + 1, group_id)] = acc[r][1];
        if constexpr (S::kTokenTiles == 2) {
            reduction[slot(warp, row, group_id + 8)]     = acc[r][2];
            reduction[slot(warp, row + 1, group_id + 8)] = acc[r][3];
        }
    }
    __syncthreads();

    const auto reduce = [&](int row, int token) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < S::kWarps; ++w) { sum += reduction[slot(w, row, token)]; }
        return sum;
    };
    if constexpr (Epilogue::kPairedHalves) {
        constexpr int kHalf = kRows / 2;
        static_assert(kRows % 2 == 0);
        for (int index = tid; index < kHalf * S::kMaxTokens; index += S::kThreads) {
            const int row   = index % kHalf;
            const int token = index / kHalf;
            if (token >= tokens) { continue; }
            epilogue(block * kHalf + row, token, reduce(row, token), reduce(row + kHalf, token));
        }
    } else {
        for (int index = tid; index < kRows * S::kMaxTokens; index += S::kThreads) {
            const int row   = index % kRows;
            const int token = index / kRows;
            if (token >= tokens) { continue; }
            epilogue(RowMap::weight_row(block, row, kRows), token, reduce(row, token));
        }
    }
}

// Launches one CTA per 8 * RowTiles weight rows; `rows` must be a multiple of that.
template <class S, int K, class RowMap = AdaContiguousRows, class Epilogue>
void ada_small_t_mma_launch(const __nv_bfloat16* x, std::int32_t x_ld, const std::uint8_t* codes,
                            const std::uint8_t* high, const std::uint8_t* scales, std::int32_t rows,
                            std::int32_t tokens, Epilogue epilogue, cudaStream_t stream) {
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(ada_small_t_mma_kernel<S, K, RowMap, Epilogue>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        S::kSmemBytes));
        return true;
    }();
    (void)configured;
    ada_small_t_mma_kernel<S, K, RowMap, Epilogue>
        <<<rows / S::kRowsPerCta, S::kThreads, S::kSmemBytes, stream>>>(x, x_ld, codes, high,
                                                                        scales, tokens, epilogue);
}

} // namespace ninfer::ops::detail

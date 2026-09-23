#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_ksplit_strided_store.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/ada_small_t_mma.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kParentRows = 7168;
constexpr std::int32_t kSplitRow   = 6144;
constexpr std::int32_t kHidden     = 5120;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    using Schedule = Q4GemvR1Q8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule, true, kSplitRow><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(key.data), kParentRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t Capacity>
void launch_q4_ksplit_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                            cudaStream_t stream) {
    using Geometry = Q4LinearGeometry<kParentRows, kHidden>;
    constexpr std::int32_t kTileCols = (Capacity + 7) / 8 * 8;
    using Store = Q4KSplitStridedStore<true, kSplitRow>;
    // The store's live-column count is the problem's actual column count: Capacity only sizes the
    // tile, and the K-split kernel stages exactly the columns it is told are live.
    const Store store{static_cast<__nv_bfloat16*>(q.data), q.ne[0],
                      static_cast<__nv_bfloat16*>(key.data), key.ne[0], x.ne[1]};
    q4_ksplit_mma_kernel<Geometry, kTileCols, Capacity, Store, Q4KSplitIdentityRows, true>
        <<<kParentRows / Q4KSplitMmaSchedule::kRowsPerCta, Q4KSplitMmaSchedule::kThreads, 0,
           stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                     static_cast<const std::uint8_t*>(weight.qdata),
                     static_cast<const std::uint8_t*>(weight.scales),
                     static_cast<__nv_bfloat16*>(q.data), store, {}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q4_ksplit_band(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                           cudaStream_t stream) {
    if (weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("attention Q4 K-split requires padded K == hidden");
    }
    switch (x.ne[1]) {
    case 2:
        launch_q4_ksplit_exact<2>(x, weight, q, key, stream);
        return;
    case 3:
        launch_q4_ksplit_exact<3>(x, weight, q, key, stream);
        return;
    case 4:
        launch_q4_ksplit_exact<4>(x, weight, q, key, stream);
        return;
    case 5:
        launch_q4_ksplit_exact<5>(x, weight, q, key, stream);
        return;
    case 6:
        launch_q4_ksplit_exact<6>(x, weight, q, key, stream);
        return;
    case 7:
        launch_q4_ksplit_exact<7>(x, weight, q, key, stream);
        return;
    case 8:
        launch_q4_ksplit_exact<8>(x, weight, q, key, stream);
        return;
    case 9:
        launch_q4_ksplit_exact<9>(x, weight, q, key, stream);
        return;
    case 10:
        launch_q4_ksplit_exact<10>(x, weight, q, key, stream);
        return;
    case 11:
        launch_q4_ksplit_exact<11>(x, weight, q, key, stream);
        return;
    case 12:
        launch_q4_ksplit_exact<12>(x, weight, q, key, stream);
        return;
    case 13:
        launch_q4_ksplit_exact<13>(x, weight, q, key, stream);
        return;
    case 14:
        launch_q4_ksplit_exact<14>(x, weight, q, key, stream);
        return;
    case 15:
        launch_q4_ksplit_exact<15>(x, weight, q, key, stream);
        return;
    case 16:
        launch_q4_ksplit_exact<16>(x, weight, q, key, stream);
        return;
    default:
        throw std::invalid_argument("attention Q4 K-split band covers T in [2,16]");
    }
}

void launch_q4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q4_gemv(x, weight, q, key, stream);
        return;
    }
    // From T=2 the K-split MMA arms all 8 warps of a CTA onto K and reads the Q4 parent once for
    // every width up to its 16-column tile.
    launch_q4_ksplit_band(x, weight, q, key, stream);
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    constexpr int kGrid         = kParentRows / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<kParentRows, kHidden, kRowsPerBlock, 2, true, true, kSplitRow>
        <<<kGrid, kBlockThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                              static_cast<const std::uint8_t*>(weight.qdata),
                                              static_cast<const std::uint8_t*>(weight.qhigh),
                                              static_cast<const std::uint8_t*>(weight.scales),
                                              static_cast<__nv_bfloat16*>(gate.data),
                                              static_cast<__nv_bfloat16*>(value.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                      cudaStream_t stream) {
    constexpr int kThreads = 4 * 32;
    const dim3 grid(static_cast<unsigned>(kParentRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, kHidden, true, kSplitRow>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(gate.data),
                                        static_cast<__nv_bfloat16*>(value.data), kParentRows,
                                        gate.ne[0], kHidden, Cols, weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2>(x, weight, gate, value, stream);
        return;
    case 3:
        launch_q5_split4<3>(x, weight, gate, value, stream);
        return;
    case 4:
        launch_q5_split4<4>(x, weight, gate, value, stream);
        return;
    case 5:
        launch_q5_split4<5>(x, weight, gate, value, stream);
        return;
    case 6:
        launch_q5_split4<6>(x, weight, gate, value, stream);
        return;
    default:
        throw std::invalid_argument("attention Q5 split4 requires T in [2,6]");
    }
}

struct AttnGateValueEpilogue {
    static constexpr bool kPairedHalves = false;

    __nv_bfloat16* gate;
    std::int32_t gate_ld;
    __nv_bfloat16* value;
    std::int32_t value_ld;

    __device__ __forceinline__ void operator()(int row, int token, float result) const {
        if (row < kSplitRow) {
            gate[static_cast<std::int64_t>(token) * gate_ld + row] = __float2bfloat16(result);
        } else {
            value[static_cast<std::int64_t>(token) * value_ld + row - kSplitRow] =
                __float2bfloat16(result);
        }
    }
};

template <int RowTiles, int Warps, int G, int Stages>
void launch_q5_ada(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                   cudaStream_t stream) {
    if (weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("attention Q5 Ada small-T requires padded K == hidden");
    }
    const AttnGateValueEpilogue epilogue{
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<std::int32_t>(gate.nb[1] / sizeof(__nv_bfloat16)),
        static_cast<__nv_bfloat16*>(value.data),
        static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16))};
    const auto* xp   = static_cast<const __nv_bfloat16*>(x.data);
    const auto x_ld  = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const auto* code = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* hi   = static_cast<const std::uint8_t*>(weight.qhigh);
    const auto* sc   = static_cast<const std::uint8_t*>(weight.scales);
    if (x.ne[1] <= 8) {
        using S = Q5AdaSmallTSchedule<RowTiles, Warps, 1, G, Stages>;
        static_assert(kParentRows % S::kRowsPerCta == 0);
        ada_small_t_mma_launch<S, kHidden>(xp, x_ld, code, hi, sc, kParentRows, x.ne[1],
                                              epilogue, stream);
    } else {
        using S = Q5AdaSmallTSchedule<RowTiles, Warps, 2, G, Stages>;
        static_assert(kParentRows % S::kRowsPerCta == 0);
        ada_small_t_mma_launch<S, kHidden>(xp, x_ld, code, hi, sc, kParentRows, x.ne[1],
                                              epilogue, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 6) {
        launch_q5_split4_exact(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        // Every weight byte is read once for the whole verify band; the column-tiled SIMT and
        // row-block shapes re-read the Q5 parent per 4- or 8-column tile from T=7.
        launch_q5_ada<2, 8, 8, 3>(x, weight, gate, value, stream);
        return;
    }
    throw std::invalid_argument("attention Q5 split-output requires T in [1,16]");
}

} // namespace

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_q4(x, query_key_weight, q, k, stream);
    launch_q5(x, gate_value_weight, gate, v, stream);
}

} // namespace ninfer::ops::detail

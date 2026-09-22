#!/usr/bin/env python3
"""Apply the RTX 4090 / Ada port overlay to satellitedown/cinference.

Pinned source:
  satellitedown/cinference@2205806b3e9d12b23673a146eab1d36755ebc3a0

The overlay preserves Cinference's MTP-10 and captured-topology CUDA Graph work.
It retargets the engine to sm_89, uses Ada's native FP8 MMA spelling, replaces
the Blackwell-only E2M1 conversion helpers with a software-exact Ada codec so
K8V4 remains available, and replaces Blackwell-only NVFP4/TMA entry points with
fail-closed stubs. It does not requantize weights.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

BASE = "2205806b3e9d12b23673a146eab1d36755ebc3a0"
ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text()


def write(path: str, text: str) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def replace(path: str, old: str, new: str) -> None:
    text = read(path)
    if old not in text:
        raise RuntimeError(f"{path}: expected source fragment not found")
    write(path, text.replace(old, new, 1))


def remove_block(path: str, block: str) -> None:
    text = read(path)
    if block not in text:
        raise RuntimeError(f"{path}: expected block not found")
    write(path, text.replace(block, "", 1))


def main() -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
    ).strip()
    if head != BASE:
        raise SystemExit(
            f"Refusing to patch unexpected Cinference revision {head}; expected {BASE}."
        )

    # Build/runtime target: Ada AD102 / RTX 4090.
    cmake = read("CMakeLists.txt")
    cmake = cmake.replace("sm_120a", "sm_89")
    cmake = cmake.replace("CMAKE_CUDA_ARCHITECTURES 120a", "CMAKE_CUDA_ARCHITECTURES 89")
    cmake = cmake.replace('CMAKE_CUDA_ARCHITECTURES STREQUAL "120a"',
                          'CMAKE_CUDA_ARCHITECTURES STREQUAL "89"')
    cmake = cmake.replace('CMAKE_CUDA_ARCHITECTURES=120a', 'CMAKE_CUDA_ARCHITECTURES=89')
    write("CMakeLists.txt", cmake)

    clangd = read(".clangd").replace("--cuda-gpu-arch=sm_120a", "--cuda-gpu-arch=sm_89")
    write(".clangd", clangd)

    replace(
        "src/models/qwen3_5/program/planning/startup.cpp",
        'if (device.compute_capability() != 120) {\n'
        '        throw std::invalid_argument("Qwen3.5 family runtime requires compute capability 12.0");\n'
        '    }',
        'if (device.compute_capability() != 89) {\n'
        '        throw std::invalid_argument("Cinference-4090 requires compute capability 8.9");\n'
        '    }',
    )

    # Ada supports native FP8 E4M3 Tensor Core MMA, but not Blackwell's
    # kind::f8f6f4 PTX spelling. Keep identical register geometry.
    replace(
        "src/ops/common/mma.cuh",
        '"mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "',
        '"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "',
    )

    # CUDA exposes the FP4 C++ surface in recent toolkits even when targeting
    # Ada, but E2M1 conversion PTX is Blackwell-family only. K8V4 needs the
    # E2M1 *storage codec*, not FP4 Tensor Core MMA, so implement that codec
    # explicitly on sm_89. E2M1 finite magnitudes are
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6}; encode uses RN-even + satfinite.
    replace(
        "src/ops/linear/nvfp4/nvfp4_codec.cuh",
        "#include <cuda_fp4.h>\n",
        "",
    )
    replace(
        "src/ops/linear/nvfp4/nvfp4_codec.cuh",
        '''__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}''',
        '''__device__ __forceinline__ float decode_nvfp4_e2m1(std::uint8_t code) {
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
}''',
    )

    old_pack = '''__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
    asm volatile("{\\n"
                 ".reg .b8 b0;\\n"
                 ".reg .b8 b1;\\n"
                 ".reg .b8 b2;\\n"
                 ".reg .b8 b3;\\n"
                 ".reg .b8 b4;\\n"
                 ".reg .b8 b5;\\n"
                 ".reg .b8 b6;\\n"
                 ".reg .b8 b7;\\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\\n"
                 "}\\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
}'''
    new_pack = '''__device__ __forceinline__ void
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
}'''
    replace("src/ops/linear/nvfp4/nvfp4_codec.cuh", old_pack, new_pack)

    kv_codec = read("src/ops/kv_cache/nvfp4_group16_codec.cuh")
    old_decode = '''        __nv_fp4x2_e2m1 encoded;
        encoded.__x         = bytes[pair];
        const __half2 value = __hmul2(static_cast<__half2>(encoded), scale2);'''
    new_decode = '''        const float2 decoded = detail::decode_nvfp4_e2m1x2(bytes[pair]);
        const __half2 value =
            __hmul2(__floats2half2_rn(decoded.x, decoded.y), scale2);'''
    if kv_codec.count(old_decode) != 2:
        raise RuntimeError(
            "src/ops/kv_cache/nvfp4_group16_codec.cuh: expected two native FP4 decode sites"
        )
    write("src/ops/kv_cache/nvfp4_group16_codec.cuh", kv_codec.replace(old_decode, new_decode))

    # Blackwell NVFP4 W4A4 Tensor Core MMA has no Ada equivalent. Keep the
    # symbol compilable so modern Cinference source can build, but make any
    # accidental execution fail on-device. Supported 4090 model weights are
    # groupwise Q4/Q5/Q6/Q8/BF16; K8V4 cache does not use this instruction.
    old_nvfp4 = '''__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}'''
    new_nvfp4 = '''__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#else
    (void)c0; (void)c1; (void)c2; (void)c3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1; (void)sfa; (void)sfb;
    asm volatile("trap;");
#endif
}'''
    replace("src/ops/common/mma.cuh", old_nvfp4, new_nvfp4)

    # Programmatic Dependent Launch / griddepcontrol requires sm_90+.
    # Cinference uses it only as a T=4 overlap optimization for the independent
    # Q4/Q5 GDN projections. On Ada, issue the same kernels sequentially on the
    # stream; this preserves math/ordering and only gives up that overlap.
    replace(
        "src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu",
        '#include "core/pdl.cuh"\n',
        "",
    )
    old_pdl = '''void launch_t4_pdl(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                   Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    using Q4Schedule         = Q4GdnSimtR8C4Schedule;
    constexpr int kQ5Threads = 4 * 32;
    const dim3 q4_grid(kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
    const dim3 q5_grid(kValueZRows, 1u, 1u);
    const std::int32_t q4_out_ld = static_cast<std::int32_t>(qk.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t q5_out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));

    // Q5 and Q4 publish disjoint row ranges. Q4 can execute while Q5 drains and joins Q5 only at
    // exit, before the following convolution/snapshot kernel becomes runnable.
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, 4, 5, kHidden, true, kValueRows,
                                        Q5Split4StoreEpilogue, true, false>
        <<<q5_grid, kQ5Threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            kValueZRows, q5_out_ld, kHidden, 4, value_z_weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(pdl::launch_dependent(
        {q4_grid, dim3(Q4Schedule::kThreads), 0, stream},
        q4_rowsplit_gemm_simt_kernel<Q4Schedule, true, false, 0, Q4SimtStoreEpilogue, false, true>,
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(qk_weight.qdata),
        static_cast<const std::uint8_t*>(qk_weight.scales), static_cast<__nv_bfloat16*>(qk.data),
        nullptr, q4_out_ld, 0, kQkRows, kHidden, 4, qk_weight.padded_shape[1],
        Q4SimtStoreEpilogue{}));
}'''
    new_pdl = '''void launch_t4_ada(const Tensor& x, const Weight& qk_weight,
                   const Weight& value_z_weight, Tensor& qk, Tensor& value, Tensor& z,
                   cudaStream_t stream) {
    launch_q4(x, qk_weight, qk, stream);
    launch_q5(x, value_z_weight, value, z, stream);
}'''
    replace(
        "src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu",
        old_pdl,
        new_pdl,
    )
    replace(
        "src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu",
        "        launch_t4_pdl(x, qk_weight, value_z_weight, qk, value, z, stream);",
        "        launch_t4_ada(x, qk_weight, value_z_weight, qk, value, z, stream);",
    )

    # Iteration fixes for Ada (sm_89), ported from the qualified dirty-tree diff.
    # Every hunk below reproduces a fix that was built and qualified on the RTX 4090:
    # sequential Q4/Q5 and sparse-MoE fallbacks (PDL/griddepcontrol is sm_90+),
    # dynamic shared memory for Q8 kernels exceeding the 48 KiB static limit,
    # fail-closed native NVFP4 weight routes, and Q8 split-K / row-split / grouped /
    # attention / GDN / pair / add / SwiGLU launches via the typed wrappers.
    # src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu: fail-closed native NVFP4 model-weight route.
    replace(
        "src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu",
        """#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {""",
        """#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {""",
    )
    replace(
        "src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu",
        """void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    launch_nvfp4_w4a4_quantize(
        x, weight, workspace,
        w4a4_tma_route(tokens) ? Nvfp4ScaleLayout::Tiled : Nvfp4ScaleLayout::RowMajor, stream);
    if (w4a4_tma_route(tokens)) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_attention(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(v.data), tokens, alpha, stream);
    } else if (tokens <= 64) {
        launch_gemm<M32N64>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<M32N128>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<M64N128>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 512) {
        launch_gemm<M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
    } else {
        launch_gemm<M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
    }
}

} // namespace ninfer::ops::detail""",
        """void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream) {
    (void)x; (void)weight; (void)q; (void)gate; (void)k; (void)v; (void)workspace; (void)stream;
    throw std::invalid_argument(
        "Cinference-4090: native NVFP4 attention weights require Blackwell");
}

} // namespace ninfer::ops::detail""",
    )
    # src/ops/attn_input_proj/q8/q8_attn_input_gemm_mma.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/attn_input_proj/q8/q8_attn_input_gemm_mma.cu",
        """template <class Schedule, bool Full, int Rows, class Output>
void launch_variant(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    const dim3 grid(Rows / Schedule::BM, static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    q8_rowsplit_gemm_mma_kernel<Schedule, Full, Q8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(weight.qdata),
                                                 static_cast<const std::uint8_t*>(weight.scales),
                                                 output, Rows, kHidden, x.ne[1], kHidden);
}

template <class Schedule, int Rows, class Output>""",
        """template <class Schedule, bool Full, int Rows, class Output>
void launch_variant(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    const dim3 grid(Rows / Schedule::BM, static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    launch_q8_rowsplit_gemm_mma<Schedule, Full, Q8Epilogue::Store, Output>(
        grid, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, Rows, kHidden, x.ne[1], kHidden);
}

template <class Schedule, int Rows, class Output>""",
    )
    # src/ops/attn_input_proj/q8/q8_attn_input_gemm_splitk.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/attn_input_proj/q8/q8_attn_input_gemm_splitk.cu",
        """                                                : 48;
    using Geometry         = Q8LinearGeometry<Rows, kHidden>;
    using Schedule         = Q8KSplitDefaultSchedule<TileCols, ActiveCols>;
    q8_ksplit_mma_kernel<Geometry, ActiveCols, Schedule>
        <<<Rows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <int ActiveCols>""",
        """                                                : 48;
    using Geometry         = Q8LinearGeometry<Rows, kHidden>;
    using Schedule         = Q8KSplitDefaultSchedule<TileCols, ActiveCols>;
    launch_q8_ksplit_mma<Geometry, ActiveCols, Schedule, Output>(
        dim3(Rows / kRowsPerCta), stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output);
}

template <int ActiveCols>""",
    )
    replace(
        "src/ops/attn_input_proj/q8/q8_attn_input_gemm_splitk.cu",
        """    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    q8_ksplit_grouped_mma_kernel<kHidden, TileCols, KSplits, NGroups, MinBlocks>
        <<<kTargetRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>""",
        """    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_ksplit_grouped_mma<kHidden, TileCols, KSplits, NGroups, MinBlocks, TargetOutput>(
        dim3(kTargetRows / kRowsPerCta), stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>""",
    )
    replace(
        "src/ops/attn_input_proj/q8/q8_attn_input_gemm_splitk.cu",
        """    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    q8_ksplit_grouped_mma_kernel<kHidden, TileCols, KSplits, NGroups, MinBlocks>
        <<<kCompanionRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

} // namespace""",
        """    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_ksplit_grouped_mma<kHidden, TileCols, KSplits, NGroups, MinBlocks,
                                 CompanionOutput>(
        dim3(kCompanionRows / kRowsPerCta), stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

} // namespace""",
    )
    # src/ops/attn_input_proj/q8/q8_dflash2_attn_input.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/attn_input_proj/q8/q8_dflash2_attn_input.cu",
        """    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    q8_ksplit_mma_kernel<Geometry, Columns, Schedule, Output, Q8KSplitStoreEpilogue,
                         Q8KSplitIdentityRows, false, !Exact>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Q8KSplitStoreEpilogue{},
            Q8KSplitIdentityRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}""",
        """    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    launch_q8_ksplit_mma<Geometry, Columns, Schedule, Output, Q8KSplitStoreEpilogue,
                         Q8KSplitIdentityRows, false, !Exact>(
        dim3(kBlocks), stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, Q8KSplitStoreEpilogue{},
        Q8KSplitIdentityRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}""",
    )
    replace(
        "src/ops/attn_input_proj/q8/q8_dflash2_attn_input.cu",
        """                        static_cast<__nv_bfloat16*>(v.data)};
    const dim3 grid(Geometry::kOutputRows / Schedule::BM,
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    q8_rowsplit_gemm_mma_kernel<Schedule, Full, Q8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Geometry::kOutputRows,
            Geometry::kInputRows, x.ne[1], Geometry::kInputRows);
    CUDA_CHECK(cudaGetLastError());
}""",
        """                        static_cast<__nv_bfloat16*>(v.data)};
    const dim3 grid(Geometry::kOutputRows / Schedule::BM,
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    launch_q8_rowsplit_gemm_mma<Schedule, Full, Q8Epilogue::Store, Output>(
        grid, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, Geometry::kOutputRows,
        Geometry::kInputRows, x.ne[1], Geometry::kInputRows);
    CUDA_CHECK(cudaGetLastError());
}""",
    )
    # src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu",
        """    using Geometry            = Q8LinearGeometry<kRows, InputRows>;
    using Schedule            = Q8KSplitSchedule<Warps, TileColumns, Warps == 8 ? 2 : 3,
                                                 Q8KSplitScaleAccess::Shared, Activation>;
    constexpr int SharedBytes = TileColumns > 64 ? sizeof(Q8KSplitSharedStorage<Schedule>) : 0;
    if constexpr (SharedBytes > 0) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            q8_ksplit_mma_kernel<Geometry, TileColumns, Schedule, Q8ContiguousOutput,
                                 Q8KSplitStoreEpilogue, Q8KSplitIdentityRows, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes);
        CUDA_CHECK(attribute);
    }
    const int columns = x.ne[1];
    Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kRows};
    const dim3 grid(kRows / 16, (columns + TileColumns - 1) / TileColumns);
    q8_ksplit_mma_kernel<Geometry, TileColumns, Schedule, Q8ContiguousOutput, Q8KSplitStoreEpilogue,
                         Q8KSplitIdentityRows, false, true>
        <<<grid, Schedule::kThreads, SharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Q8KSplitStoreEpilogue{},
            Q8KSplitIdentityRows{}, columns);
    CUDA_CHECK(cudaGetLastError());
}""",
        """    using Geometry            = Q8LinearGeometry<kRows, InputRows>;
    using Schedule            = Q8KSplitSchedule<Warps, TileColumns, Warps == 8 ? 2 : 3,
                                                 Q8KSplitScaleAccess::Shared, Activation>;
    const int columns = x.ne[1];
    Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kRows};
    const dim3 grid(kRows / 16, (columns + TileColumns - 1) / TileColumns);
    launch_q8_ksplit_mma<Geometry, TileColumns, Schedule, Q8ContiguousOutput,
                         Q8KSplitStoreEpilogue, Q8KSplitIdentityRows, false, true>(
        grid, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, Q8KSplitStoreEpilogue{},
        Q8KSplitIdentityRows{}, columns);
    CUDA_CHECK(cudaGetLastError());
}""",
    )
    # src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu: fail-closed native NVFP4 model-weight route.
    replace(
        "src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu",
        """#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

namespace ninfer::ops::detail {
namespace {""",
        """#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {""",
    )
    replace(
        "src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu",
        """void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    launch_nvfp4_w4a4_quantize(
        x, weight, workspace,
        w4a4_tma_route(tokens) ? Nvfp4ScaleLayout::Tiled : Nvfp4ScaleLayout::RowMajor, stream);
    if (w4a4_tma_route(tokens)) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_gdn(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(qkv.data),
            static_cast<__nv_bfloat16*>(z.data), tokens, alpha, stream);
    } else if (tokens <= 64) {
        launch_gemm<M32N64>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<M32N128>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<M128N128Pipelined>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<M64N128>(weight, qkv, z, workspace, tokens, stream);
    } else {
        launch_gemm<M128N128Resident>(weight, qkv, z, workspace, tokens, stream);
    }
}

} // namespace ninfer::ops::detail""",
        """void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    (void)x; (void)weight; (void)qkv; (void)z; (void)workspace; (void)stream;
    throw std::invalid_argument(
        "Cinference-4090: native NVFP4 GDN weights require Blackwell");
}

} // namespace ninfer::ops::detail""",
    )
    # src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu: sequential Ada fallback for sm_90+ PDL.
    replace(
        "src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu",
        """               const GdnConvEpilogue<Publish>& qk_epilogue,
               const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
               Tensor& z, cudaStream_t stream) {
    // The Q4 and Q5 sides read the same activation but write disjoint output/state rows. The
    // dependent side therefore computes before waiting, then joins the producer at kernel exit.
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_t1<Publish, true, false, false>(x, value_z_weight, value_epilogue, value, z,
                                                  stream);
        launch_q4_t1<Publish, false, true, true>(x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_q4_t1<Publish, true, false, false>(x, qk_weight, qk_epilogue, query, stream);
        launch_q5_t1<Publish, false, true, true>(x, value_z_weight, value_epilogue, value, z,
                                                 stream);
    }
}

template <int Tokens, class Q4Schedule, PdlOrder Order, class Publish>""",
        """               const GdnConvEpilogue<Publish>& qk_epilogue,
               const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
               Tensor& z, cudaStream_t stream) {
    // Programmatic Dependent Launch requires sm_90+. Ada preserves the same mathematics and
    // publication ordering by running the disjoint Q4 and Q5 projections sequentially on the
    // same stream. Order remains a template parameter so the public routing contract is unchanged.
    (void)Order;
    launch_q4_t1<Publish, false, false, false>(x, qk_weight, qk_epilogue, query, stream);
    launch_q5_t1<Publish, false, false, false>(x, value_z_weight, value_epilogue, value, z, stream);
}

template <int Tokens, class Q4Schedule, PdlOrder Order, class Publish>""",
    )
    replace(
        "src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu",
        """                             const GdnConvEpilogue<Publish>& qk_epilogue,
                             const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query,
                             Tensor& value, Tensor& z, cudaStream_t stream) {
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_small_t<Tokens, Publish, true, false, false>(x, value_z_weight, value_epilogue,
                                                               value, z, stream);
        launch_q4_ksplit<Tokens, Q4Schedule, Publish, false, true, true>(x, qk_weight, qk_epilogue,
                                                                         query, stream);
    } else {
        launch_q4_ksplit<Tokens, Q4Schedule, Publish, true, false, false>(x, qk_weight, qk_epilogue,
                                                                          query, stream);
        launch_q5_small_t<Tokens, Publish, false, true, true>(x, value_z_weight, value_epilogue,
                                                              value, z, stream);
    }
}

template <int Tokens, PdlOrder Order, class Publish>""",
        """                             const GdnConvEpilogue<Publish>& qk_epilogue,
                             const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query,
                             Tensor& value, Tensor& z, cudaStream_t stream) {
    (void)Order;
    launch_q4_ksplit<Tokens, Q4Schedule, Publish, false, false, false>(
        x, qk_weight, qk_epilogue, query, stream);
    launch_q5_small_t<Tokens, Publish, false, false, false>(
        x, value_z_weight, value_epilogue, value, z, stream);
}

template <int Tokens, PdlOrder Order, class Publish>""",
    )
    # src/ops/gdn_input_proj/q8/q8_gdn_input_gemm_mma.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/gdn_input_proj/q8/q8_gdn_input_gemm_mma.cu",
        """    static_assert((8192 % Schedule::BM) == 0 && (4096 % Schedule::BM) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    const dim3 grid(kRows / Schedule::BM, static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    q8_rowsplit_gemm_mma_kernel<Schedule, Full, Q8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(weight.qdata),
                                                 static_cast<const std::uint8_t*>(weight.scales),
                                                 output, kRows, kHidden, x.ne[1], kHidden);
}

} // namespace""",
        """    static_assert((8192 % Schedule::BM) == 0 && (4096 % Schedule::BM) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    const dim3 grid(kRows / Schedule::BM, static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    launch_q8_rowsplit_gemm_mma<Schedule, Full, Q8Epilogue::Store, Output>(
        grid, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, kRows, kHidden, x.ne[1], kHidden);
}

} // namespace""",
    )
    # src/ops/gdn_input_proj/q8/q8_gdn_input_gemm_splitk.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/gdn_input_proj/q8/q8_gdn_input_gemm_splitk.cu",
        """    using Schedule = Q8KSplitDefaultSchedule<TileCols, ActiveCols>;
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    q8_ksplit_mma_kernel<Geometry, ActiveCols, Schedule>
        <<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <int ActiveCols, class Publish>""",
        """    using Schedule = Q8KSplitDefaultSchedule<TileCols, ActiveCols>;
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    launch_q8_ksplit_mma<Geometry, ActiveCols, Schedule, Output>(
        dim3(kRows / kRowsPerCta), stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output);
}

template <int ActiveCols, class Publish>""",
    )
    replace(
        "src/ops/gdn_input_proj/q8/q8_gdn_input_gemm_splitk.cu",
        """        },
        static_cast<__nv_bfloat16*>(z.data),
    };
    q8_ksplit_mma_kernel<Geometry, ActiveCols, Schedule, Output, Q8GdnSplitKConvEpilogue<Publish>>
        <<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue);
}

template <int ActiveCols>""",
        """        },
        static_cast<__nv_bfloat16*>(z.data),
    };
    launch_q8_ksplit_mma<Geometry, ActiveCols, Schedule, Output,
                         Q8GdnSplitKConvEpilogue<Publish>>(
        dim3(kRows / kRowsPerCta), stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue);
}

template <int ActiveCols>""",
    )
    # src/ops/linear/nvfp4/nvfp4_dispatch.cpp: fail-closed native NVFP4 model-weight route.
    replace(
        "src/ops/linear/nvfp4/nvfp4_dispatch.cpp",
        """void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (!allows_a4(policy) || !shape.uses_a4(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    shape.a4(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail""",
        """void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    (void)x; (void)weight; (void)out; (void)policy; (void)workspace; (void)stream;
    throw std::invalid_argument(
        "Cinference-4090: native NVFP4 model weights require Blackwell; use a groupwise artifact");
}
} // namespace ninfer::ops::detail""",
    )
    # src/ops/linear/nvfp4/shapes/n14336_k5120.cu: fail-closed native NVFP4 model-weight route.
    replace(
        "src/ops/linear/nvfp4/shapes/n14336_k5120.cu",
        """}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 1024) return nvfp4_a4_tma_route<Nvfp4GeometryId::N14336K5120>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 96) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, T64R128>();
    if (tokens <= 384) return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
    if (tokens <= 512) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 4; }""",
        """}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    (void)tokens;
    throw std::invalid_argument("Cinference-4090: native NVFP4 linear weights require Blackwell");
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 4; }""",
    )
    # src/ops/linear/nvfp4/shapes/n16384_k5120.cu: fail-closed native NVFP4 model-weight route.
    replace(
        "src/ops/linear/nvfp4/shapes/n16384_k5120.cu",
        """}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 1024) return nvfp4_a4_tma_route<Nvfp4GeometryId::N16384K5120>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 96) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, T64R128>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t) { return true; }""",
        """}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    (void)tokens;
    throw std::invalid_argument("Cinference-4090: native NVFP4 linear weights require Blackwell");
}

bool uses_a4(std::int32_t, std::int32_t) { return true; }""",
    )
    # src/ops/linear/nvfp4/shapes/n34816_k5120.cu: fail-closed native NVFP4 model-weight route.
    replace(
        "src/ops/linear/nvfp4/shapes/n34816_k5120.cu",
        """}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 256) return nvfp4_a4_tma_route<Nvfp4GeometryId::N34816K5120>();
    if (tokens <= 32) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T64R128>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t) { return true; }""",
        """}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    (void)tokens;
    throw std::invalid_argument("Cinference-4090: native NVFP4 linear weights require Blackwell");
}

bool uses_a4(std::int32_t, std::int32_t) { return true; }""",
    )
    # src/ops/linear/nvfp4/shapes/n5120_k17408.cu: fail-closed native NVFP4 model-weight route.
    replace(
        "src/ops/linear/nvfp4/shapes/n5120_k17408.cu",
        """}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 1024) return nvfp4_a4_tma_route<Nvfp4GeometryId::N5120K17408>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, T64R128>();
    if (tokens <= 384) return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
    if (tokens <= 512) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 8; }""",
        """}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    (void)tokens;
    throw std::invalid_argument("Cinference-4090: native NVFP4 linear weights require Blackwell");
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 8; }""",
    )
    # src/ops/linear/nvfp4/shapes/n5120_k6144.cu: fail-closed native NVFP4 model-weight route.
    replace(
        "src/ops/linear/nvfp4/shapes/n5120_k6144.cu",
        """}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 1024) return nvfp4_a4_tma_route<Nvfp4GeometryId::N5120K6144>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, T64R128>();
    if (tokens <= 384) return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
    if (tokens <= 512) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 8; }""",
        """}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    (void)tokens;
    throw std::invalid_argument("Cinference-4090: native NVFP4 linear weights require Blackwell");
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 8; }""",
    )
    # src/ops/linear/q8/q8_ksplit_grouped_mma.cuh: dynamic shared memory + typed launch wrapper.
    replace(
        "src/ops/linear/q8/q8_ksplit_grouped_mma.cuh",
        """#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int Hidden, int TileCols, int KSplits, int NGroups, int MinBlocks, class Output,
          bool AddResidual = false, bool TiledColumns = false>
__global__ __launch_bounds__(KSplits* NGroups * 32, MinBlocks) void q8_ksplit_grouped_mma_kernel(""",
        """#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {

template <int TileCols, int KSplits, int NGroups>
struct alignas(16) Q8KSplitGroupedSharedStorage {
    static constexpr int kTileK       = 64;
    static constexpr int kMmaRows     = 16;
    static constexpr int kKernelWarps = KSplits * NGroups;
    static constexpr int kGroupK      = KSplits * kTileK;
    static constexpr int kWarpCols    = TileCols / NGroups;

    std::uint8_t codes[kMmaRows][kGroupK];
    __nv_bfloat16 activations[kKernelWarps][kWarpCols * kTileK];
};

template <int Hidden, int TileCols, int KSplits, int NGroups, int MinBlocks, class Output,
          bool AddResidual = false, bool TiledColumns = false>
__global__ __launch_bounds__(KSplits* NGroups * 32, MinBlocks) void q8_ksplit_grouped_mma_kernel(""",
    )
    replace(
        "src/ops/linear/q8/q8_ksplit_grouped_mma.cuh",
        """    static_assert(TileCols % NGroups == 0 && kWarpCols % 8 == 0);
    static_assert(Hidden % kGroupK == 0 && kKernelWarps <= 32);

    __shared__ __align__(16) std::uint8_t code_shared[kMmaRows][kGroupK];
    __shared__ __align__(16) __nv_bfloat16 b_shared[kKernelWarps][kWarpCols * kTileK];

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;""",
        """    static_assert(TileCols % NGroups == 0 && kWarpCols % 8 == 0);
    static_assert(Hidden % kGroupK == 0 && kKernelWarps <= 32);

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto& shared = *reinterpret_cast<Q8KSplitGroupedSharedStorage<TileCols, KSplits, NGroups>*>(
        shared_raw);
    auto& code_shared = shared.codes;
    auto& b_shared    = shared.activations;

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;""",
    )
    replace(
        "src/ops/linear/q8/q8_ksplit_grouped_mma.cuh",
        """    }
}

} // namespace ninfer::ops::detail""",
        """    }
}

template <int Hidden, int TileCols, int KSplits, int NGroups, int MinBlocks, class Output,
          bool AddResidual = false, bool TiledColumns = false, class... Args>
void launch_q8_ksplit_grouped_mma(dim3 grid, cudaStream_t stream, Args&&... args) {
    constexpr int kDynamicBytes =
        static_cast<int>(sizeof(Q8KSplitGroupedSharedStorage<TileCols, KSplits, NGroups>));
    if constexpr (kDynamicBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            q8_ksplit_grouped_mma_kernel<Hidden, TileCols, KSplits, NGroups, MinBlocks, Output,
                                         AddResidual, TiledColumns>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicBytes);
        if (attribute != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(attribute)); }
    }
    q8_ksplit_grouped_mma_kernel<Hidden, TileCols, KSplits, NGroups, MinBlocks, Output,
                                 AddResidual, TiledColumns>
        <<<grid, KSplits * NGroups * 32, kDynamicBytes, stream>>>(std::forward<Args>(args)...);
}

} // namespace ninfer::ops::detail""",
    )
    # src/ops/linear/q8/q8_ksplit_launch.cuh: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear/q8/q8_ksplit_launch.cuh",
        """        throw std::invalid_argument("q8 K-split: padded K differs from the registered geometry");
    }
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    q8_ksplit_mma_kernel<Geometry, ColumnCapacity, Schedule, Q8ContiguousOutput, Epilogue,
                         Q8KSplitIdentityRows, false, true>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Epilogue{},
            Q8KSplitIdentityRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}""",
        """        throw std::invalid_argument("q8 K-split: padded K differs from the registered geometry");
    }
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    launch_q8_ksplit_mma<Geometry, ColumnCapacity, Schedule, Q8ContiguousOutput, Epilogue,
                         Q8KSplitIdentityRows, false, true>(
        dim3(Geometry::kOutputRows / Schedule::kRowsPerCta), stream,
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, Epilogue{},
        Q8KSplitIdentityRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}""",
    )
    # src/ops/linear/q8/q8_ksplit_mma.cuh: dynamic shared memory + typed launch wrapper.
    replace(
        "src/ops/linear/q8/q8_ksplit_mma.cuh",
        """#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {""",
        """#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::ops::detail {""",
    )
    replace(
        "src/ops/linear/q8/q8_ksplit_mma.cuh",
        """    float partial[Schedule::kKWarps * (Schedule::kTileTokens / 8) * 32 * 4];
};

struct Q8KSplitIdentityColumns {
    __device__ __forceinline__ int operator()(int column) const { return column; }
};""",
        """    float partial[Schedule::kKWarps * (Schedule::kTileTokens / 8) * 32 * 4];
};

template <class Schedule>
inline constexpr int kQ8KSplitDynamicSharedBytes =
    sizeof(Q8KSplitSharedStorage<Schedule>) > 48 * 1024
        ? static_cast<int>(sizeof(Q8KSplitSharedStorage<Schedule>))
        : 0;

struct Q8KSplitIdentityColumns {
    __device__ __forceinline__ int operator()(int column) const { return column; }
};""",
    )
    replace(
        "src/ops/linear/q8/q8_ksplit_mma.cuh",
        """    using SharedStorage = Q8KSplitSharedStorage<Schedule>;

    constexpr bool kDynamicShared = TiledColumns && ActiveCols > 64;
    __shared__ __align__(
        16) unsigned char static_shared[kDynamicShared ? 1 : sizeof(SharedStorage)];
    extern __shared__ __align__(16) unsigned char dynamic_shared[];""",
        """    using SharedStorage = Q8KSplitSharedStorage<Schedule>;

    constexpr bool kDynamicShared = kQ8KSplitDynamicSharedBytes<Schedule> != 0;
    __shared__ __align__(
        16) unsigned char static_shared[kDynamicShared ? 1 : sizeof(SharedStorage)];
    extern __shared__ __align__(16) unsigned char dynamic_shared[];""",
    )
    replace(
        "src/ops/linear/q8/q8_ksplit_mma.cuh",
        """                  TiledColumns>(x, codes, scales, output, epilogue, row_policy, columns);
}

} // namespace ninfer::ops::detail""",
        """                  TiledColumns>(x, codes, scales, output, epilogue, row_policy, columns);
}

template <class Geometry, int ActiveCols, class Schedule, class Output,
          class Epilogue = Q8KSplitStoreEpilogue, class RowPolicy = Q8KSplitIdentityRows,
          bool DirectPairEpilogue = false, bool TiledColumns = false, class... Args>
void launch_q8_ksplit_mma(dim3 grid, cudaStream_t stream, Args&&... args) {
    constexpr int kDynamicBytes = kQ8KSplitDynamicSharedBytes<Schedule>;
    if constexpr (kDynamicBytes != 0) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            q8_ksplit_mma_kernel<Geometry, ActiveCols, Schedule, Output, Epilogue, RowPolicy,
                                 DirectPairEpilogue, TiledColumns>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicBytes);
        if (attribute != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(attribute)); }
    }
    q8_ksplit_mma_kernel<Geometry, ActiveCols, Schedule, Output, Epilogue, RowPolicy,
                         DirectPairEpilogue, TiledColumns>
        <<<grid, Schedule::kThreads, kDynamicBytes, stream>>>(std::forward<Args>(args)...);
}

} // namespace ninfer::ops::detail""",
    )
    # src/ops/linear/q8/q8_rowsplit_gemm_mma.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear/q8/q8_rowsplit_gemm_mma.cu",
        """    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::BM)),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)), 1u);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), rows};
    q8_rowsplit_gemm_mma_kernel<Schedule, Full><<<grid, Schedule::THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), output, rows, k, cols, padded_k);
    CUDA_CHECK(cudaGetLastError());
}""",
        """    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::BM)),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)), 1u);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), rows};
    launch_q8_rowsplit_gemm_mma<Schedule, Full, Q8Epilogue::Store, Q8ContiguousOutput>(
        grid, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        output, rows, k, cols, padded_k);
    CUDA_CHECK(cudaGetLastError());
}""",
    )
    # src/ops/linear/q8/q8_rowsplit_gemm_mma.cuh: dynamic shared memory + typed launch wrapper.
    replace(
        "src/ops/linear/q8/q8_rowsplit_gemm_mma.cuh",
        """#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {""",
        """#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {""",
    )
    replace(
        "src/ops/linear/q8/q8_rowsplit_gemm_mma.cuh",
        """static_assert(sizeof(Q8Bf16x8Bits) == 16);

// The predicated loads below inherited Cache::ca from cp_async_zfill's default, while the full path
// a few lines down spells cg. This parameter makes that a choice. It defaults to ca, so adding it
// changes no instantiation, and it governs the predicated branch only - the full branch keeps its""",
        """static_assert(sizeof(Q8Bf16x8Bits) == 16);

template <class Cfg>
struct Q8RowSplitOperandStorage {
    alignas(16) __nv_bfloat16 weights[Cfg::BM * Cfg::BK];
    alignas(16) __nv_bfloat16 activations[Cfg::ACTIVATION_STAGES][Cfg::BN * Cfg::BK];
    alignas(16) std::uint8_t codes[Cfg::BM * Cfg::BK];
    alignas(16) std::uint8_t scales[Cfg::BM * Cfg::SCALE_CACHE_BYTES];
};

template <class Cfg, Q8Epilogue Epilogue>
union alignas(16) Q8RowSplitSharedStorage {
    Q8RowSplitOperandStorage<Cfg> operands;
    float projected[Epilogue == Q8Epilogue::Residual ? Cfg::BM * Cfg::BN : 1];
};

// The predicated loads below inherited Cache::ca from cp_async_zfill's default, while the full path
// a few lines down spells cg. This parameter makes that a choice. It defaults to ca, so adding it
// changes no instantiation, and it governs the predicated branch only - the full branch keeps its""",
    )
    replace(
        "src/ops/linear/q8/q8_rowsplit_gemm_mma.cuh",
        """    static_assert(!kSwiGlu || Cfg::WARPS_M == 1 || Cfg::WARPS_M == 2,
                  "SwiGLU supports warp-local or shared-memory row pairing");

    struct OperandStorage {
        alignas(16) __nv_bfloat16 weights[BM * BK];
        alignas(16) __nv_bfloat16 activations[Cfg::ACTIVATION_STAGES][BN * BK];
        alignas(16) std::uint8_t codes[BM * BK];
        alignas(16) std::uint8_t scales[BM * Cfg::SCALE_CACHE_BYTES];
    };

    union SharedStorage {
        OperandStorage operands;
        float projected[Epilogue == Q8Epilogue::Residual ? BM * BN : 1];
    };

    static_assert(sizeof(SharedStorage) <= 99 * 1024);
    __shared__ __align__(16) SharedStorage shared;
    auto& As = shared.operands.weights;
    auto& Bs = shared.operands.activations;
    auto& Cr = shared.operands.codes;""",
        """    static_assert(!kSwiGlu || Cfg::WARPS_M == 1 || Cfg::WARPS_M == 2,
                  "SwiGLU supports warp-local or shared-memory row pairing");

    using SharedStorage = Q8RowSplitSharedStorage<Cfg, Epilogue>;
    static_assert(sizeof(SharedStorage) <= 99 * 1024);
    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto& shared = *reinterpret_cast<SharedStorage*>(shared_raw);
    auto& As = shared.operands.weights;
    auto& Bs = shared.operands.activations;
    auto& Cr = shared.operands.codes;""",
    )
    replace(
        "src/ops/linear/q8/q8_rowsplit_gemm_mma.cuh",
        """    }
}

} // namespace ninfer::ops::detail""",
        """    }
}

template <class Cfg, bool Full, Q8Epilogue Epilogue = Q8Epilogue::Store,
          class Output = Q8ContiguousOutput, class... Args>
void launch_q8_rowsplit_gemm_mma(dim3 grid, cudaStream_t stream, Args&&... args) {
    constexpr int kDynamicBytes =
        static_cast<int>(sizeof(Q8RowSplitSharedStorage<Cfg, Epilogue>));
    if constexpr (kDynamicBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            q8_rowsplit_gemm_mma_kernel<Cfg, Full, Epilogue, Output>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicBytes);
        if (attribute != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(attribute)); }
    }
    q8_rowsplit_gemm_mma_kernel<Cfg, Full, Epilogue, Output>
        <<<grid, Cfg::THREADS, kDynamicBytes, stream>>>(std::forward<Args>(args)...);
}

} // namespace ninfer::ops::detail""",
    )
    # src/ops/linear/q8/shapes/n2048_k16384.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear/q8/shapes/n2048_k16384.cu",
        """            "q8 grouped K-split: padded K differs from registered geometry");
    }
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    q8_ksplit_grouped_mma_kernel<Geometry::kInputRows, Capacity, KWarps, TokenGroups, 1>
        <<<Geometry::kOutputRows / 16, KWarps * TokenGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace""",
        """            "q8 grouped K-split: padded K differs from registered geometry");
    }
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    launch_q8_ksplit_grouped_mma<Geometry::kInputRows, Capacity, KWarps, TokenGroups, 1,
                                 Q8ContiguousOutput>(
        dim3(Geometry::kOutputRows / 16), stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace""",
    )
    # src/ops/linear/q8/shapes/n5120_k25600.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear/q8/shapes/n5120_k25600.cu",
        """    using Schedule = Q8RowSplitMmaGemmSchedule<Rows, 64, 16, 16, 1, 2, 128, 1>;
    const dim3 grid(weight.n / Rows, (x.ne[1] + 63) / 64);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    q8_rowsplit_gemm_mma_kernel<Schedule, false><<<grid, Schedule::THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, weight.n, weight.k, x.ne[1],
        weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());""",
        """    using Schedule = Q8RowSplitMmaGemmSchedule<Rows, 64, 16, 16, 1, 2, 128, 1>;
    const dim3 grid(weight.n / Rows, (x.ne[1] + 63) / 64);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    launch_q8_rowsplit_gemm_mma<Schedule, false, Q8Epilogue::Store, Q8ContiguousOutput>(
        grid, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, weight.n, weight.k, x.ne[1],
        weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());""",
    )
    # src/ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu: fail-closed native NVFP4 model-weight route.
    replace(
        "src/ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu",
        """void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    launch_nvfp4_w4a4_quantize(
        x, weight, workspace,
        w4a4_tma_route(tokens) ? Nvfp4ScaleLayout::Tiled : Nvfp4ScaleLayout::RowMajor, stream);
    const Nvfp4GeometryId problem = resolve_nvfp4_geometry(weight.n, weight.k);
    if (w4a4_tma_route(tokens)) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_linear_add(problem, workspace.codes, workspace.scales,
                                         static_cast<const std::uint8_t*>(weight.qdata),
                                         static_cast<const std::uint8_t*>(weight.scales),
                                         static_cast<__nv_bfloat16*>(residual.data), tokens, alpha,
                                         stream);
        return;
    }
    switch (problem) {
    case Nvfp4GeometryId::N5120K6144:
        launch_problem<Nvfp4N5120K6144>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_problem<Nvfp4N5120K17408>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail""",
        """void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    (void)x; (void)weight; (void)residual; (void)workspace; (void)stream;
    throw std::invalid_argument(
        "Cinference-4090: native NVFP4 linear-add weights require Blackwell");
}

} // namespace ninfer::ops::detail""",
    )
    # src/ops/linear_add/q8/q8_linear_add_gemm_grouped.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear_add/q8/q8_linear_add_gemm_grouped.cu",
        """    constexpr int kTokenGroups = 4;
    const dim3 grid(5120 / 16, div_up(x.ne[1], kColumns));
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual.data), 5120};
    q8_ksplit_grouped_mma_kernel<K, kColumns, kSplits, kTokenGroups, 1, Q8ContiguousOutput, true,
                                 true><<<grid, kSplits * kTokenGroups * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}""",
        """    constexpr int kTokenGroups = 4;
    const dim3 grid(5120 / 16, div_up(x.ne[1], kColumns));
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual.data), 5120};
    launch_q8_ksplit_grouped_mma<K, kColumns, kSplits, kTokenGroups, 1, Q8ContiguousOutput, true,
                                 true>(
        grid, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}""",
    )
    # src/ops/linear_add/q8/q8_linear_add_gemm_mma.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear_add/q8/q8_linear_add_gemm_mma.cu",
        """    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::BM)),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)), 1u);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual_out.data), rows};
    q8_rowsplit_gemm_mma_kernel<Schedule, Full, Q8Epilogue::Residual>
        <<<grid, Schedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output, rows, k, cols, padded_k);
}

template <class Schedule>""",
        """    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::BM)),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)), 1u);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual_out.data), rows};
    launch_q8_rowsplit_gemm_mma<Schedule, Full, Q8Epilogue::Residual, Q8ContiguousOutput>(
        grid, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        output, rows, k, cols, padded_k);
}

template <class Schedule>""",
    )
    # src/ops/linear_add/q8/q8_linear_add_gemm_splitk.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear_add/q8/q8_linear_add_gemm_splitk.cu",
        """    static_assert((kRows % kRowsPerCta) == 0);
    auto* residual = static_cast<__nv_bfloat16*>(residual_out.data);
    const Q8ContiguousOutput output{residual, kRows};
    q8_ksplit_mma_kernel<Geometry, ActiveCols, Schedule, Q8ContiguousOutput,
                         Q8KSplitResidualEpilogue>
        <<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Q8KSplitResidualEpilogue{});
}

template <int Hidden, std::size_t... Offsets>""",
        """    static_assert((kRows % kRowsPerCta) == 0);
    auto* residual = static_cast<__nv_bfloat16*>(residual_out.data);
    const Q8ContiguousOutput output{residual, kRows};
    launch_q8_ksplit_mma<Geometry, ActiveCols, Schedule, Q8ContiguousOutput,
                         Q8KSplitResidualEpilogue>(
        dim3(kRows / kRowsPerCta), stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, Q8KSplitResidualEpilogue{});
}

template <int Hidden, std::size_t... Offsets>""",
    )
    replace(
        "src/ops/linear_add/q8/q8_linear_add_gemm_splitk.cu",
        """void launch_medium(const Tensor& x, Tensor& residual_out, const Weight& weight,
                   cudaStream_t stream) {
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual_out.data), kRows};
    q8_ksplit_grouped_mma_kernel<Hidden, TileCols, KSplits, NGroups, MinBlocks, Q8ContiguousOutput,
                                 true><<<kRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}""",
        """void launch_medium(const Tensor& x, Tensor& residual_out, const Weight& weight,
                   cudaStream_t stream) {
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual_out.data), kRows};
    launch_q8_ksplit_grouped_mma<Hidden, TileCols, KSplits, NGroups, MinBlocks, Q8ContiguousOutput,
                                 true>(
        dim3(kRows / kRowsPerCta), stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}""",
    )
    # src/ops/linear_pair/q8/q8_pair_gemm_concat.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear_pair/q8/q8_pair_gemm_concat.cu",
        """                            static_cast<__nv_bfloat16*>(second_out.data)};
    const dim3 grid(static_cast<unsigned>(2 * div_up(kRows, Schedule::BM)),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    q8_rowsplit_gemm_mma_kernel<Schedule, Full, Q8Epilogue::Store, PairOutput>
        <<<grid, Schedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(first_weight.qdata),
            static_cast<const std::uint8_t*>(first_weight.scales), output, 2 * kRows, kHidden,
            x.ne[1], kHidden);
}

template <class Schedule>""",
        """                            static_cast<__nv_bfloat16*>(second_out.data)};
    const dim3 grid(static_cast<unsigned>(2 * div_up(kRows, Schedule::BM)),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    launch_q8_rowsplit_gemm_mma<Schedule, Full, Q8Epilogue::Store, PairOutput>(
        grid, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(first_weight.qdata),
        static_cast<const std::uint8_t*>(first_weight.scales), output, 2 * kRows, kHidden,
        x.ne[1], kHidden);
}

template <class Schedule>""",
    )
    # src/ops/linear_pair/q8/q8_pair_gemm_splitk.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear_pair/q8/q8_pair_gemm_splitk.cu",
        """    const Q8ContiguousOutput ignored{static_cast<__nv_bfloat16*>(first_out.data), kRows};
    const Q8PairExactTEpilogue epilogue{static_cast<__nv_bfloat16*>(first_out.data),
                                        static_cast<__nv_bfloat16*>(second_out.data)};
    q8_ksplit_mma_kernel<Geometry, ActiveCols, Schedule, Q8ContiguousOutput, Q8PairExactTEpilogue,
                         Q8PairExactTRows><<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), first_codes, first_scales, ignored, epilogue,
        Q8PairExactTRows{});
}

template <std::size_t... Offsets>""",
        """    const Q8ContiguousOutput ignored{static_cast<__nv_bfloat16*>(first_out.data), kRows};
    const Q8PairExactTEpilogue epilogue{static_cast<__nv_bfloat16*>(first_out.data),
                                        static_cast<__nv_bfloat16*>(second_out.data)};
    launch_q8_ksplit_mma<Geometry, ActiveCols, Schedule, Q8ContiguousOutput, Q8PairExactTEpilogue,
                         Q8PairExactTRows>(
        dim3(kRows / kRowsPerCta), stream, static_cast<const __nv_bfloat16*>(x.data), first_codes,
        first_scales, ignored, epilogue, Q8PairExactTRows{});
}

template <std::size_t... Offsets>""",
    )
    replace(
        "src/ops/linear_pair/q8/q8_pair_gemm_splitk.cu",
        """    }
    const PairOutput output{static_cast<__nv_bfloat16*>(first_out.data),
                            static_cast<__nv_bfloat16*>(second_out.data)};
    q8_ksplit_grouped_mma_kernel<kHidden, TileCols, KSplits, NGroups, MinBlocks>
        <<<(2 * kRows) / 16, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), first_codes, first_scales, output, x.ne[1]);
}

} // namespace""",
        """    }
    const PairOutput output{static_cast<__nv_bfloat16*>(first_out.data),
                            static_cast<__nv_bfloat16*>(second_out.data)};
    launch_q8_ksplit_grouped_mma<kHidden, TileCols, KSplits, NGroups, MinBlocks, PairOutput>(
        dim3((2 * kRows) / 16), stream, static_cast<const __nv_bfloat16*>(x.data), first_codes,
        first_scales, output, x.ne[1]);
}

} // namespace""",
    )
    # src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu: fail-closed native NVFP4 model-weight route.
    replace(
        "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu",
        """#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {""",
        """#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {""",
    )
    replace(
        "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu",
        """void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    if (x.ne[1] <= M64N128::kBlockM) {
        launch<M64N128>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M96N128::kBlockM) {
        launch<M96N128>(x, weight, out, workspace, stream);
    } else {
        launch<M128N128>(x, weight, out, workspace, stream);
    }
}

} // namespace ninfer::ops::detail""",
        """void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    (void)x; (void)weight; (void)out; (void)workspace; (void)stream;
    throw std::invalid_argument(
        "Cinference-4090: native NVFP4 SwiGLU weights require Blackwell");
}

} // namespace ninfer::ops::detail""",
    )
    # src/ops/linear_swiglu/q8/q8_dflash2_linear_swiglu.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear_swiglu/q8/q8_dflash2_linear_swiglu.cu",
        """    const Q8SwiGluDirectEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const RowPolicy row_policy{};
    constexpr int kBlocks = kIntermediate / RowPolicy::kOutputRowsPerCta;
    q8_ksplit_mma_kernel<Geometry, Capacity, Schedule, Q8ContiguousOutput, Q8SwiGluDirectEpilogue,
                         RowPolicy, true, true><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue, row_policy,
        x.ne[1]);
    CUDA_CHECK(cudaGetLastError());""",
        """    const Q8SwiGluDirectEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const RowPolicy row_policy{};
    constexpr int kBlocks = kIntermediate / RowPolicy::kOutputRowsPerCta;
    launch_q8_ksplit_mma<Geometry, Capacity, Schedule, Q8ContiguousOutput,
                         Q8SwiGluDirectEpilogue, RowPolicy, true, true>(
        dim3(kBlocks), stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue, row_policy,
        x.ne[1]);
    CUDA_CHECK(cudaGetLastError());""",
    )
    # src/ops/linear_swiglu/q8/q8_linear_swiglu_gemm_mma.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear_swiglu/q8/q8_linear_swiglu_gemm_mma.cu",
        """    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]};
    const dim3 grid(out.ne[0] / (Schedule::BM / 2),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    q8_rowsplit_gemm_mma_kernel<Schedule, Full, Q8Epilogue::SwiGluSplitHalf>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(w.qdata),
                                                 static_cast<const std::uint8_t*>(w.scales), output,
                                                 w.n, w.k, x.ne[1], w.padded_shape[1]);
}

template <class Schedule>""",
        """    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]};
    const dim3 grid(out.ne[0] / (Schedule::BM / 2),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    launch_q8_rowsplit_gemm_mma<Schedule, Full, Q8Epilogue::SwiGluSplitHalf,
                                Q8ContiguousOutput>(
        grid, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        output, w.n, w.k, x.ne[1], w.padded_shape[1]);
}

template <class Schedule>""",
    )
    # src/ops/linear_swiglu/q8/q8_linear_swiglu_gemm_splitk.cu: Ada dynamic-shared-memory launch wrapper call site.
    replace(
        "src/ops/linear_swiglu/q8/q8_linear_swiglu_gemm_splitk.cu",
        """    const Q8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const RowPolicy row_policy{};
    q8_ksplit_mma_kernel<Geometry, ActiveCols, Schedule, Q8ContiguousOutput, Q8SwiGluDirectEpilogue,
                         RowPolicy, true>
        <<<kIntermediate / RowPolicy::kOutputRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), ignored_output, epilogue, row_policy);
}

template <std::size_t... Offsets>""",
        """    const Q8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const RowPolicy row_policy{};
    launch_q8_ksplit_mma<Geometry, ActiveCols, Schedule, Q8ContiguousOutput,
                         Q8SwiGluDirectEpilogue, RowPolicy, true>(
        dim3(kIntermediate / RowPolicy::kOutputRowsPerCta), stream,
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), ignored_output, epilogue, row_policy);
}

template <std::size_t... Offsets>""",
    )
    # src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu: sequential Ada fallback for sm_90+ PDL.
    replace(
        "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu",
        """#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh\"""",
        """#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh\"""",
    )
    replace(
        "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu",
        """                                          float* __restrict__ alpha,
                                          float* __restrict__ shared_scale) {
    __shared__ float selected_logits[kTopK];
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    sparse_moe_select_top8_warp(scores, ids, alpha, shared_scale, selected_logits);
}""",
        """                                          float* __restrict__ alpha,
                                          float* __restrict__ shared_scale) {
    __shared__ float selected_logits[kTopK];
    sparse_moe_select_top8_warp(scores, ids, alpha, shared_scale, selected_logits);
}""",
    )
    replace(
        "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu",
        """    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) { pdl::trigger_dependents(); }
    if (tid < 256) { store_vec(x_shared + tid * 8, load_vec<uint4>(x + tid * 8)); }
    __syncthreads();""",
        """    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < 256) { store_vec(x_shared + tid * 8, load_vec<uint4>(x + tid * 8)); }
    __syncthreads();""",
    )
    replace(
        "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu",
        """    float gate  = 0.0f;
    float up    = 0.0f;
    if (warp < kTopK) {
        pdl::wait_for_dependencies();
        const int expert   = ids[warp];
        const int row_base = expert * 1024;
        dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales, row_base + j,""",
        """    float gate  = 0.0f;
    float up    = 0.0f;
    if (warp < kTopK) {
        const int expert   = ids[warp];
        const int row_base = expert * 1024;
        dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales, row_base + j,""",
    )
    replace(
        "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu",
        """        }
        __syncthreads();

        constexpr int kRouterPartitions = 4;
        const int activation_begin      = (token * (kTopK + 1) + path) * kIntermediate;
        // Routed paths need S2's ids. A shared path is independent only when its output lies
        // beyond the partial-score prefix that S2 may still read from the lifetime-unioned scratch.
        const bool must_wait_for_s2 =
            path < kTopK || activation_begin < tokens * kRouterRows * kRouterPartitions;
        if constexpr (!Adaptive) {
            if (must_wait_for_s2) { pdl::wait_for_dependencies(); }
        }
        float gate = 0.0f;
        float up   = 0.0f;
        if (path < kTopK) {""",
        """        }
        __syncthreads();

        float gate = 0.0f;
        float up   = 0.0f;
        if (path < kTopK) {""",
    )
    replace(
        "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu",
        """    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination,
    const char* __restrict__ prefetch_data, unsigned long long prefetch_bytes) {
    __shared__ float paths[kTopK + 1][Rows];
    pdl::wait_for_dependencies();
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int row_base = static_cast<int>(blockIdx.x) * Rows;""",
        """    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination,
    const char* __restrict__ prefetch_data, unsigned long long prefetch_bytes) {
    __shared__ float paths[kTopK + 1][Rows];
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int row_base = static_cast<int>(blockIdx.x) * Rows;""",
    )
    replace(
        "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu",
        """template <class Codec>
void launch_d3_dependent_codec(const Tensor& x, const SparseMoeWeights& weights,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    auto* act                 = static_cast<float*>(workspace.scratch.data);""",
        """template <class Codec>
void launch_d3_dependent_codec(const Tensor& x, const SparseMoeWeights& weights,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    // Programmatic Dependent Launch is sm_90+. On Ada, ordinary launches on this stream preserve
    // the D1 -> D2 -> D3 -> D4 data dependencies exactly, at the cost of pipeline overlap.
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    auto* act                 = static_cast<float*>(workspace.scratch.data);""",
    )
    replace(
        "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu",
        """    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kIntermediate), dim3(9 * 32), 0, stream}, sparse_moe_d3_nine_warp_kernel<Codec>,
        input, ids, routed_codes, routed_high, routed_scales, shared_codes, shared_scales, act));
}

void launch_d2_d3(const Tensor& x, const SparseMoeWeights& weights,""",
        """    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    sparse_moe_d3_nine_warp_kernel<Codec><<<kIntermediate, 9 * 32, 0, stream>>>(
        input, ids, routed_codes, routed_high, routed_scales, shared_codes, shared_scales, act);
    CUDA_CHECK(cudaGetLastError());
}

void launch_d2_d3(const Tensor& x, const SparseMoeWeights& weights,""",
    )
    replace(
        "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu",
        """    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);
    auto* output              = static_cast<__nv_bfloat16*>(destination.data);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kHidden), dim3(9 * 32), 0, stream}, sparse_moe_d4_nine_warp_kernel<Codec, 1>, ids,
        alpha, shared_scale, act, routed_codes, routed_high, routed_scales, shared_codes,
        shared_scales, output, static_cast<const char*>(prefetch_data),
        static_cast<unsigned long long>(prefetch_bytes)));
}

void launch_d4_dependent(const SparseMoeWeights& weights, Tensor& destination,""",
        """    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);
    auto* output              = static_cast<__nv_bfloat16*>(destination.data);
    sparse_moe_d4_nine_warp_kernel<Codec, 1><<<kHidden, 9 * 32, 0, stream>>>(
        ids, alpha, shared_scale, act, routed_codes, routed_high, routed_scales, shared_codes,
        shared_scales, output, static_cast<const char*>(prefetch_data),
        static_cast<unsigned long long>(prefetch_bytes));
    CUDA_CHECK(cudaGetLastError());
}

void launch_d4_dependent(const SparseMoeWeights& weights, Tensor& destination,""",
    )
    replace(
        "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu",
        """                shared_scales, token_activations, tokens, adaptive_route_jobs);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(kIntermediate, tokens * kPathBlocks), dim3(PathsPerBlock * 32), 0, stream},
            sparse_moe_d3_path_tiled_kernel<Codec, PathsPerBlock, false>, input, token_ids,
            routed_codes, routed_high, routed_scales, shared_codes, shared_scales,
            token_activations, tokens, nullptr));
    }
}""",
        """                shared_scales, token_activations, tokens, adaptive_route_jobs);
        CUDA_CHECK(cudaGetLastError());
    } else {
        sparse_moe_d3_path_tiled_kernel<Codec, PathsPerBlock, false>
            <<<dim3(kIntermediate, tokens * kPathBlocks), PathsPerBlock * 32, 0, stream>>>(
                input, token_ids, routed_codes, routed_high, routed_scales, shared_codes,
                shared_scales, token_activations, tokens, nullptr);
        CUDA_CHECK(cudaGetLastError());
    }
}""",
    )
    # src/ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu: sequential Ada fallback for sm_90+ PDL.
    replace(
        "src/ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu",
        """#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/sparse_moe/decode/sparse_moe_decode.h\"""",
        """#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/sparse_moe/decode/sparse_moe_decode.h\"""",
    )
    replace(
        "src/ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu",
        """                                             float* __restrict__ partial_scores) {
    static_assert(Tokens >= 1 && Tokens <= kSparseMoeSmallTMax);
    __shared__ float partial[kRouterWarps][Tokens];
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    const int row       = static_cast<int>(blockIdx.x) / kRouterPartitions;
    const int partition = static_cast<int>(blockIdx.x) - row * kRouterPartitions;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;""",
        """                                             float* __restrict__ partial_scores) {
    static_assert(Tokens >= 1 && Tokens <= kSparseMoeSmallTMax);
    __shared__ float partial[kRouterWarps][Tokens];
    const int row       = static_cast<int>(blockIdx.x) / kRouterPartitions;
    const int partition = static_cast<int>(blockIdx.x) - row * kRouterPartitions;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;""",
    )
    replace(
        "src/ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu",
        """    __shared__ float selected_logits[kTopK];
    const int tid   = static_cast<int>(threadIdx.x);
    const int token = static_cast<int>(blockIdx.x);
    if (tid == 0) { pdl::trigger_dependents(); }
    pdl::wait_for_dependencies();
    for (int row = tid; row < kRouterRows; row += kS2Threads) {
        float sum = 0.0f;
#pragma unroll""",
        """    __shared__ float selected_logits[kTopK];
    const int tid   = static_cast<int>(threadIdx.x);
    const int token = static_cast<int>(blockIdx.x);
    for (int row = tid; row < kRouterRows; row += kS2Threads) {
        float sum = 0.0f;
#pragma unroll""",
    )
    replace(
        "src/ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu",
        """void launch_s2(const float* partial_scores, int* token_ids, float* token_alpha, float* shared_scale,
               std::int32_t tokens, cudaStream_t stream) {
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(static_cast<unsigned int>(tokens)), dim3(kS2Threads), 0, stream},
        sparse_moe_small_t_s2_kernel, partial_scores, token_ids, token_alpha, shared_scale));
}

void launch_s3_tiled(const Tensor& x, const SparseMoeWeights& weights,""",
        """void launch_s2(const float* partial_scores, int* token_ids, float* token_alpha, float* shared_scale,
               std::int32_t tokens, cudaStream_t stream) {
    // PDL is sm_90+. The surrounding S1 -> S2 -> S3 -> S4 launches all use this stream, so Ada's
    // ordinary stream ordering preserves the dependencies while giving up only cross-grid overlap.
    sparse_moe_small_t_s2_kernel<<<static_cast<unsigned int>(tokens), kS2Threads, 0, stream>>>(
        partial_scores, token_ids, token_alpha, shared_scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_s3_tiled(const Tensor& x, const SparseMoeWeights& weights,""",
    )

    # Replace the Blackwell-only non-RDC archive with Ada host-launch stubs.
    replace(
        "src/ops/CMakeLists.txt",
        '''# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
# producer to consumer warpgroups. RDC makes ptxas discard that contract. These
# self-contained kernels therefore use a non-RDC archive and host launchers.
add_library(ninfer_nvfp4_non_rdc STATIC)
ninfer_internal_includes(ninfer_nvfp4_non_rdc)
ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)
''',
        '''# RTX 4090 / Ada has no native NVFP4 W4A4/TMA route. Keep the private
# launch ABI present with fail-closed stubs so groupwise Qwen3.8 + K8V4 can use
# the same modern Cinference engine without compiling Blackwell-only PTX.
add_library(ninfer_nvfp4_non_rdc STATIC
  "${CMAKE_CURRENT_LIST_DIR}/sm89_blackwell_stubs.cu")
ninfer_internal_includes(ninfer_nvfp4_non_rdc)
ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)
''',
    )

    remove_block(
        "src/ops/linear/nvfp4/sources.cmake",
        '''\ntarget_sources(ninfer_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_w4a4_tma.cu"
)
''',
    )
    remove_block(
        "src/ops/linear_swiglu/sources.cmake",
        '''\ntarget_sources(ninfer_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu"
)
''',
    )
    remove_block(
        "src/ops/softmax_attention/sources.cmake",
        '''\ntarget_sources(ninfer_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4_non_rdc.cu"
)
''',
    )

    stub = r'''// RTX 4090 / Ada compatibility stubs for Blackwell-only NVFP4 launch routes.
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
[[noreturn]] void unsupported() {
    throw std::invalid_argument(
        "Cinference-4090: native NVFP4 weight/TMA execution requires Blackwell; "
        "use a groupwise Q4/Q5/Q6/Q8/BF16 .ninfer artifact on RTX 4090");
}
} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4GeometryId, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    unsupported();
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
                                     std::int32_t, float, cudaStream_t) {
    unsupported();
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                               __nv_bfloat16*, std::int32_t, float, cudaStream_t) {
    unsupported();
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4GeometryId, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, __nv_bfloat16*, std::int32_t,
                                      float, cudaStream_t) {
    unsupported();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float,
                                         cudaStream_t) {
    unsupported();
}

void causal_attention_prompt_nvfp4_kernel_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&, Tensor&,
                                                 cudaStream_t) {
    unsupported();
}

void causal_attention_prompt_nvfp4_batch_kernel_launch(const Tensor&, const Tensor&,
                                                       const Tensor&, const Tensor&, float,
                                                       const PagedKVBatchLayerView&, Tensor&,
                                                       cudaStream_t) {
    unsupported();
}

} // namespace ninfer::ops::detail
'''
    write("src/ops/sm89_blackwell_stubs.cu", stub)

    # Leave a durable provenance marker in the resulting source tree.
    write(
        "PORT-RTX4090.md",
        f"""# Cinference RTX 4090 port

Base: `satellitedown/cinference@{BASE}`.

This port keeps Cinference's MTP-10, expanded speculative round buffers, and
capture-derived CUDA Graph topology reuse. It retargets the runtime to NVIDIA
Ada compute capability 8.9.

Supported first target:
- Qwen3.8-27B groupwise Q4/Q5/Q6/Q8/BF16 NInfer artifacts
- MTP windows 1..10
- BF16 / INT8 / FP8-K+V4 (K8V4) KV paths
- software E2M1 pack/decode on Ada for the V4 cache plane
- Ada fallback for the sm_90+ T=4 PDL/griddepcontrol GDN overlap
- sequential Ada fallbacks for the remaining PDL/griddepcontrol routes (Q4/Q5 T=1
  and small-T GDN projections, sparse-MoE decode and small-T stages)
- dynamic shared memory for Q8 kernels exceeding Ada's 48 KiB static limit,
  issued through typed launch wrappers (split-K, row-split, grouped)
- every native NVFP4 model-weight route fails closed (attention, GDN, linear,
  linear-add, SwiGLU, dispatch, shape selection)
- single RTX 4090, Linux, CUDA 13.x

Not supported on Ada:
- native NVFP4 W4A4 model-weight execution
- Blackwell TMA/setmaxnreg NVFP4 prompt kernels

The unsupported Blackwell launch entry points fail closed. No model
requantization is performed by this port.
""",
    )

    print("Cinference RTX 4090 overlay applied.")


if __name__ == "__main__":
    main()

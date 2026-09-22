#!/usr/bin/env python3
"""Apply the RTX 4090 / Ada port overlay to satellitedown/cinference.

Pinned source:
  satellitedown/cinference@2205806b3e9d12b23673a146eab1d36755ebc3a0

The overlay preserves Cinference's MTP-10 and captured-topology CUDA Graph work.
It retargets the engine to sm_89, uses Ada's native FP8 MMA spelling so K8V4 can
remain available, and replaces Blackwell-only NVFP4/TMA entry points with
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

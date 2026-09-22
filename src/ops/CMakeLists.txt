# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops STATIC)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_nvfp4_non_rdc)

# RTX 4090 / Ada has no native NVFP4 W4A4/TMA route. Keep the private
# launch ABI present with fail-closed stubs so groupwise Qwen3.8 + K8V4 can use
# the same modern Cinference engine without compiling Blackwell-only PTX.
add_library(ninfer_nvfp4_non_rdc STATIC
  "${CMAKE_CURRENT_LIST_DIR}/sm89_blackwell_stubs.cu")
ninfer_internal_includes(ninfer_nvfp4_non_rdc)
ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")

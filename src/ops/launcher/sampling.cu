// Implements: include/ninfer/ops/sampling.h
// Match: validated contiguous BF16/I32 tensors and a shared-layout workspace.
// Algorithm assumptions: launcher and kernels use sampler_multiblock_ok() from
// the same layout authority, so exactly one finite route owns each shape.
#include "ops/launcher/sampling.h"

#include "ops/common/math.h"
#include "ops/kernel/sampling.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {

__global__ void increment_token_counts_kernel(const std::int32_t* token_ids, std::int32_t count,
                                              std::int32_t* token_counts) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) { atomicAdd(&token_counts[token_ids[index]], 1); }
}

std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain, std::int32_t columns) {
    return make_sampling_workspace_layout(token_domain, columns).bytes;
}

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t batch             = logits.ne[1];
    const auto* positions                = static_cast<const std::int32_t*>(logical_positions.data);
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, batch);
    if (!layout.multiblock) {
        sample_row_kernel<<<static_cast<unsigned int>(batch), kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            configs, positions, purpose, token_domain, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(batch));
    sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), configs, token_domain, physical_rows,
        scratch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(batch));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
        partial_blocks, groups, scratch);
    CUDA_CHECK(cudaGetLastError());
}

void sample_mtp_proposal_launch(const Tensor& logits, Tensor& out_tokens,
                                Tensor& candidate_ids, Tensor& proposal_q,
                                std::int32_t public_token_domain, const std::int32_t* id_map,
                                const SamplingConfig* configs, const Tensor& logical_positions,
                                const Tensor& round_tokens, const Tensor& round_counts,
                                const Tensor& prior_proposals, std::int32_t prior_count, std::int32_t prior_stride, std::int32_t proposal_step,
                                std::int32_t position_step, DeviceSpan workspace,
                                cudaStream_t stream) {
    const int batch = logits.ne[1];
    const int proposal_rows = id_map == nullptr ? public_token_domain : logits.ne[0];
    if (proposal_rows <= kSamplerTileItems) {
        sample_mtp_proposal_small_row_kernel<<<static_cast<unsigned int>(batch), kSamplerBlock, 0,
                                               stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<int*>(out_tokens.data),
            static_cast<int*>(candidate_ids.data), static_cast<float*>(proposal_q.data), configs,
            static_cast<const int*>(logical_positions.data),
            static_cast<const int*>(round_tokens.data), static_cast<const int*>(round_counts.data),
            static_cast<const int*>(prior_proposals.data), id_map, public_token_domain,
            proposal_rows, logits.ne[0], round_tokens.ne[0], candidate_ids.ne[1], prior_stride,
            prior_count, proposal_step, position_step);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(proposal_rows, batch);
    if (layout.multiblock) {
        const std::int32_t partial_blocks = div_up(proposal_rows, kSamplerPartialTileItems);
        const std::int32_t groups = sampler_group_count(partial_blocks);
        const SamplingWorkspace scratch = layout.bind(workspace);
        const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                                static_cast<unsigned int>(batch));
        sample_mtp_proposal_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), configs,
            static_cast<const int*>(round_tokens.data), static_cast<const int*>(round_counts.data),
            static_cast<const int*>(prior_proposals.data), id_map, proposal_rows, logits.ne[0],
            round_tokens.ne[0], prior_stride, prior_count, scratch);
        CUDA_CHECK(cudaGetLastError());
        const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(batch));
        sample_mtp_proposal_group_finalize_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
            static_cast<int*>(out_tokens.data), static_cast<int*>(candidate_ids.data),
            static_cast<float*>(proposal_q.data), configs,
            static_cast<const int*>(logical_positions.data), public_token_domain, proposal_rows,
            candidate_ids.ne[1], prior_count, proposal_step, position_step, partial_blocks, groups,
            scratch);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    sample_mtp_proposal_row_kernel<<<static_cast<unsigned int>(batch), kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<int*>(out_tokens.data),
        static_cast<int*>(candidate_ids.data), static_cast<float*>(proposal_q.data), configs,
        static_cast<const int*>(logical_positions.data),
        static_cast<const int*>(round_tokens.data), static_cast<const int*>(round_counts.data),
        static_cast<const int*>(prior_proposals.data), id_map, public_token_domain, proposal_rows,
        logits.ne[0], batch, round_tokens.ne[0], candidate_ids.ne[1], prior_stride, prior_count,
        proposal_step, position_step);
    CUDA_CHECK(cudaGetLastError());
}

void increment_token_counts_launch(const Tensor& token_ids, Tensor& token_counts,
                                   cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int count      = token_ids.ne[0];
    increment_token_counts_kernel<<<div_up(count, kBlock), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(token_ids.data), count,
        static_cast<std::int32_t*>(token_counts.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

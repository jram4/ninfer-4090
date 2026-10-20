#pragma once

// ninfer::ops::detail - private launch prototype for sample.

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream);
void sample_mtp_proposal_launch(const Tensor& logits, Tensor& out_tokens,
                                Tensor& candidate_ids, Tensor& proposal_q,
                                std::int32_t public_token_domain, const std::int32_t* id_map,
                                const SamplingConfig* configs, const Tensor& logical_positions,
                                const Tensor& round_tokens, const Tensor& round_counts,
                                const Tensor& prior_proposals, std::int32_t prior_count,
                                std::int32_t prior_stride, std::int32_t proposal_step,
                                std::int32_t position_step, DeviceSpan workspace,
                                cudaStream_t stream);

void increment_token_counts_launch(const Tensor& token_ids, Tensor& token_counts,
                                   cudaStream_t stream);

[[nodiscard]] std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain,
                                                         std::int32_t columns);

} // namespace ninfer::ops::detail

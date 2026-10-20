// ninfer::ops - sample wrapper: public api validation and dispatch.
#include "ninfer/ops/sampling.h"

#include "ops/common/sampling_workspace.h"
#include "ops/launcher/sampling.h" // detail::sample_batch_launch

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_sampling_dtype(const Tensor& tensor, DType dtype, const char* op, const char* name) {
    if (tensor.dtype != dtype || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": invalid tensor for " + name);
    }
}

void require_sampling_vector(const Tensor& tensor, DType dtype, std::int32_t count,
                            const char* op, const char* name) {
    require_sampling_dtype(tensor, dtype, op, name);
    if (count <= 0 || tensor.ne[0] != count || tensor.ne[1] != 1 || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid vector shape for " + name);
    }
}

} // namespace

std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain, std::int32_t min_lanes,
                                              std::int32_t max_lanes) {
    if (token_domain <= 0 || min_lanes <= 0 || max_lanes < min_lanes) {
        throw std::invalid_argument("sampling workspace: invalid profile or lane interval");
    }
    if (token_domain <= kSamplerTileItems || min_lanes > kSamplerMaxColumns) { return 0; }
    return detail::sampling_workspace_exact_bytes(token_domain,
                                                  std::min(max_lanes, kSamplerMaxColumns));
}

void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            WorkspaceArena& workspace, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("sample: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("sample: out must be I32"); }
    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("sample: logits must be rank-2 [physical_rows,B]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("sample: out must be rank-1 [B]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("sample: physical rows must be positive");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("sample: token_domain must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) { throw std::invalid_argument("sample: out shape must be [B]"); }
    if (logical_positions.dtype != DType::I32 || logical_positions.ne[0] != logits.ne[1] ||
        logical_positions.ne[1] != 1 || logical_positions.ne[2] != 1 ||
        logical_positions.ne[3] != 1) {
        throw std::invalid_argument("sample: logical_positions must be I32 [logits.ne[1]]");
    }
    if (logits.ne[1] <= 0) { throw std::invalid_argument("sample: B must be positive"); }
    if (!logits.is_contiguous() || !out.is_contiguous() || !logical_positions.is_contiguous()) {
        throw std::invalid_argument("sample: logits/out/logical_positions must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr || logical_positions.data == nullptr) {
        throw std::invalid_argument("sample: tensor data must be non-null");
    }
    if (configs == nullptr) { throw std::invalid_argument("sample: configs must be non-null"); }

    auto scratch_scope = workspace.scope();
    const std::size_t bytes =
        sampling_workspace_capacity_bytes(token_domain, logits.ne[1], logits.ne[1]);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::sample_batch_launch(logits, out, token_domain, configs, logical_positions, purpose,
                                scratch, stream);
}

void sample_mtp_proposal(const Tensor& logits, Tensor& out_tokens, Tensor& candidate_ids,
                         Tensor& proposal_q, std::int32_t public_token_domain,
                         const std::int32_t* id_map, const SamplingConfig* configs,
                         const Tensor& logical_positions, const Tensor& round_tokens,
                         const Tensor& round_counts, const Tensor& prior_proposals,
                         std::int32_t prior_count, std::int32_t proposal_step,
                         std::int32_t position_step, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    constexpr const char* op = "sample_mtp_proposal";
    require_sampling_dtype(logits, DType::BF16, op, "logits");
    if (logits.ne[0] < kSamplingCandidateCapacity || logits.ne[1] <= 0 || logits.ne[2] != 1 ||
        logits.ne[3] != 1) {
        throw std::invalid_argument("sample_mtp_proposal: logits must be [rows>=20,B]");
    }
    const int batch = logits.ne[1];
    require_sampling_vector(out_tokens, DType::I32, batch, op, "out_tokens");
    require_sampling_dtype(candidate_ids, DType::I32, op, "candidate_ids");
    require_sampling_dtype(proposal_q, DType::FP32, op, "proposal_q");
    if (candidate_ids.ne[0] != kSamplingCandidateCapacity || candidate_ids.ne[1] <= 0 ||
        candidate_ids.ne[2] != batch || candidate_ids.ne[3] != 1 ||
        proposal_q.ne[0] != kSamplingCandidateCapacity ||
        proposal_q.ne[1] != candidate_ids.ne[1] || proposal_q.ne[2] != batch ||
        proposal_q.ne[3] != 1 || proposal_step < 0 || proposal_step >= candidate_ids.ne[1]) {
        throw std::invalid_argument("sample_mtp_proposal: candidate outputs must be [20,K,B]");
    }
    if (public_token_domain < kSamplingCandidateCapacity ||
        (!id_map && logits.ne[0] < public_token_domain) || configs == nullptr) {
        throw std::invalid_argument("sample_mtp_proposal: unsupported vocabulary or configs");
    }
    require_sampling_vector(logical_positions, DType::I32, batch, op, "logical_positions");
    require_sampling_vector(round_counts, DType::I32, batch, op, "round_counts");
    require_sampling_dtype(round_tokens, DType::I32, op, "round_tokens");
    if (round_tokens.ne[0] <= 0 || round_tokens.ne[1] != batch || round_tokens.ne[2] != 1 ||
        round_tokens.ne[3] != 1) {
        throw std::invalid_argument("sample_mtp_proposal: round_tokens must be [K+1,B]");
    }
    if (prior_proposals.dtype != DType::I32 || prior_proposals.data == nullptr ||
        prior_proposals.ne[0] != batch || prior_proposals.ne[1] != candidate_ids.ne[1] ||
        prior_proposals.ne[2] != 1 || prior_proposals.ne[3] != 1 ||
        prior_proposals.nb[0] != static_cast<std::int64_t>(sizeof(std::int32_t)) ||
        prior_proposals.nb[1] < static_cast<std::int64_t>(batch) * sizeof(std::int32_t) ||
        prior_proposals.nb[1] % sizeof(std::int32_t) != 0 || prior_count < 0 ||
        prior_count > proposal_step || position_step < 0) {
        throw std::invalid_argument("sample_mtp_proposal: prior proposal shape or count is invalid");
    }
    const std::int32_t proposal_rows = id_map == nullptr ? public_token_domain : logits.ne[0];
    auto scratch_scope = workspace.scope();
    const std::size_t bytes =
        sampling_workspace_capacity_bytes(proposal_rows, batch, batch);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::sample_mtp_proposal_launch(
        logits, out_tokens, candidate_ids, proposal_q, public_token_domain, id_map, configs,
        logical_positions, round_tokens, round_counts, prior_proposals, prior_count,
        static_cast<std::int32_t>(prior_proposals.nb[1] / sizeof(std::int32_t)), proposal_step,
        position_step, scratch, stream);
}

void increment_token_counts(const Tensor& token_ids, Tensor& token_counts, cudaStream_t stream) {
    if (token_ids.dtype != DType::I32 || token_ids.ne[0] <= 0 || token_ids.ne[1] != 1 ||
        token_ids.ne[2] != 1 || token_ids.ne[3] != 1 || !token_ids.is_contiguous() ||
        token_ids.data == nullptr) {
        throw std::invalid_argument(
            "increment_token_counts: token_ids must be a contiguous non-empty I32 vector");
    }
    if (token_counts.dtype != DType::I32 || token_counts.ne[0] <= 0 || token_counts.ne[1] != 1 ||
        token_counts.ne[2] != 1 || token_counts.ne[3] != 1 || !token_counts.is_contiguous() ||
        token_counts.data == nullptr) {
        throw std::invalid_argument(
            "increment_token_counts: token_counts must be a contiguous non-empty I32 vector");
    }
    detail::increment_token_counts_launch(token_ids, token_counts, stream);
}

} // namespace ninfer::ops

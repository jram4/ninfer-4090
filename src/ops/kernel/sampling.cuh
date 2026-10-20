#pragma once

// Implements: include/ninfer/ops/sampling.h
// Match: contiguous BF16 logits, physical stride >= token domain, and at most
// sixteen columns on the multi-block route.
// Algorithm assumptions: 256-thread/2-item partial tiles feed bounded top-20
// group merges through caller-owned workspace; unsupported finite geometries
// use the semantically identical single-block fallback.

#include "ops/kernel/sampling_device.cuh"

namespace ninfer::ops {

__launch_bounds__(kSamplerBlock) __global__
    void sample_row_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                           const SamplingConfig* configs, const std::int32_t* logical_positions,
                           std::int32_t purpose, std::int32_t token_domain,
                           std::int32_t physical_rows) {
    const int row            = static_cast<int>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(row) * physical_rows;
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = configs[row];

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];

    // With penalties disabled this remains the exact raw-logit argmax route.
    if (!(cfg.temperature > 0.0f)) {
        float bv             = -CUDART_INF_F;
        int bi               = INT_MAX;
        const bool penalties = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
        if (!penalties) {
            for (int v = tid; v < token_domain; v += blockDim.x) {
                const float x = __bfloat162float(logits[base + v]);
                if (sampling_better(x, v, bv, bi)) {
                    bv = x;
                    bi = v;
                }
            }
        } else {
            for (int v = tid; v < token_domain; v += blockDim.x) {
                const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg);
                if (sampling_better(x, v, bv, bi)) {
                    bv = x;
                    bi = v;
                }
            }
        }
        red_val[tid] = bv;
        red_idx[tid] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s &&
                sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[row] = red_idx[0];
            if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[red_idx[0]], 1); }
        }
        return;
    }

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape (see sample_batch_launch).
    if (sampler_multiblock_ok(token_domain, static_cast<int>(gridDim.x), partial_blocks,
                              group_count)) {
        return;
    }

    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];

    if (token_domain <= kSamplerTileItems) {
        sampling_build_truncated_small(logits, base, token_domain, cfg, red_val, red_idx, cand_val,
                                       cand_idx, prob, &n_support);
    } else {
        sampling_build_truncated_block_fast(logits, base, token_domain, cfg, merge_val, merge_idx,
                                            cand_val, cand_idx, prob, &n_support);
    }

    if (tid != 0) { return; }
    const int support = n_support;
    const float u     = sampling_uniform(cfg.seed, logical_positions[row], purpose, 0u);
    float acc         = 0.0f;
    int picked        = cand_idx[support - 1];
    for (int j = 0; j < support; ++j) {
        acc += prob[j]; // prob is normalized: goal == u
        if (u < acc) {
            picked = cand_idx[j];
            break;
        }
    }
    out[row] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

__device__ __forceinline__ float sampling_mtp_adjusted_logit(
    float raw, int token_id, const SamplingConfig& cfg, const int* round_tokens,
    int round_width, int round_count, const int* prior_proposals, int prior_stride,
    int prior_count, int batch, int row) {
    if (cfg.presence_penalty == 0.0f && cfg.frequency_penalty == 0.0f) { return raw; }
    int count = cfg.token_counts != nullptr ? cfg.token_counts[token_id] : 0;
    const int safe_round_count = min(round_width, max(0, round_count));
    for (int i = 0; i < safe_round_count; ++i) {
        if (round_tokens[row * round_width + i] == token_id) { ++count; }
    }
    const int safe_prior_count = min(kSamplerCandidateCap, max(0, prior_count));
    for (int i = 0; i < safe_prior_count; ++i) {
        if (prior_proposals[i * prior_stride + row] == token_id) { ++count; }
    }
    if (count > 0) { raw -= cfg.presence_penalty; }
    if (cfg.frequency_penalty != 0.0f) {
        raw -= cfg.frequency_penalty * static_cast<float>(count);
    }
    return raw;
}

__device__ __forceinline__ int sampling_mtp_candidate_cap(const SamplingConfig& cfg,
                                                           int proposal_rows) {
    const int sample_cap = sampling_candidate_cap(cfg, proposal_rows);
    return min(sample_cap, kMtpProposalSupportCapacity);
}

__launch_bounds__(kSamplerBlock) __global__ void sample_mtp_proposal_small_row_kernel(
    const __nv_bfloat16* logits, int* out_tokens, int* candidate_ids, float* proposal_q,
    const SamplingConfig* configs, const int* logical_positions, const int* round_tokens,
    const int* round_counts, const int* prior_proposals, const int* id_map,
    int public_token_domain, int proposal_rows, int physical_rows, int round_width,
    int proposal_window, int prior_stride, int prior_count, int proposal_step, int position_step) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const SamplingConfig cfg = configs[row];
    const std::int64_t base = static_cast<std::int64_t>(row) * physical_rows;
    const int out_at = (row * proposal_window + proposal_step) * kSamplerCandidateCap;

    __shared__ float tile_val[kSamplerTileItems];
    __shared__ int tile_idx[kSamplerTileItems];
    __shared__ float cand_val[kMtpProposalSupportCapacity];
    __shared__ int cand_idx[kMtpProposalSupportCapacity];
    __shared__ float prob[kMtpProposalSupportCapacity];
    __shared__ int n_support;

    if (tid < kSamplerTileItems) {
        if (tid < proposal_rows) {
            const int token_id = id_map == nullptr ? tid : id_map[tid];
            tile_val[tid] = sampling_mtp_adjusted_logit(
                __bfloat162float(logits[base + tid]), token_id, cfg, round_tokens, round_width,
                round_counts[row], prior_proposals, prior_stride, prior_count, 0, row);
            tile_idx[tid] = token_id;
        } else {
            tile_val[tid] = -CUDART_INF_F;
            tile_idx[tid] = INT_MAX;
        }
    }
    __syncthreads();

    if (!(cfg.temperature > 0.0f)) {
        for (int stride = kSamplerTileItems / 2; stride > 0; stride >>= 1) {
            if (tid < stride && sampling_better(tile_val[tid + stride], tile_idx[tid + stride],
                                                tile_val[tid], tile_idx[tid])) {
                tile_val[tid] = tile_val[tid + stride];
                tile_idx[tid] = tile_idx[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int selected_id = tile_idx[0];
            out_tokens[row] = selected_id;
            for (int j = 0; j < kSamplerCandidateCap; ++j) {
                candidate_ids[out_at + j] = (selected_id + j) % public_token_domain;
                proposal_q[out_at + j] = j == 0 ? 1.0f : 0.0f;
            }
        }
        return;
    }

    sampling_sort_tile_desc(tile_val, tile_idx);
    const int cap = sampling_mtp_candidate_cap(cfg, proposal_rows);
    if (tid < kMtpProposalSupportCapacity) {
        if (tid < cap) {
            cand_val[tid] = tile_val[tid];
            cand_idx[tid] = tile_idx[tid];
        } else {
            cand_val[tid] = -CUDART_INF_F;
            cand_idx[tid] = INT_MAX;
        }
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        const int position = logical_positions[row] + position_step;
        const float u = sampling_uniform(cfg.seed, position, kSamplePurposeMtpProposal, 0u);
        float acc = 0.0f;
        int picked = cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
        out_tokens[row] = picked;
        for (int j = 0; j < kSamplerCandidateCap; ++j) {
            candidate_ids[out_at + j] = j < support ? cand_idx[j] : 0;
            proposal_q[out_at + j] = j < support ? prob[j] : 0.0f;
        }
    }
}

__launch_bounds__(kSamplerBlock) __global__ void sample_mtp_proposal_row_kernel(
    const __nv_bfloat16* logits, int* out_tokens, int* candidate_ids, float* proposal_q,
    const SamplingConfig* configs, const int* logical_positions, const int* round_tokens,
    const int* round_counts, const int* prior_proposals, const int* id_map,
    int public_token_domain, int proposal_rows, int physical_rows, int batch, int round_width,
    int proposal_window, int prior_stride, int prior_count, int proposal_step, int position_step) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const SamplingConfig cfg = configs[row];
    const std::int64_t base = static_cast<std::int64_t>(row) * physical_rows;
    const int out_at = (row * proposal_window + proposal_step) * kSamplerCandidateCap;

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ float cand_val[kMtpProposalSupportCapacity];
    __shared__ int cand_idx[kMtpProposalSupportCapacity];
    __shared__ float prob[kMtpProposalSupportCapacity];
    __shared__ float merge_val[kSamplerBlock * kMtpProposalSupportCapacity];
    __shared__ int merge_idx[kSamplerBlock * kMtpProposalSupportCapacity];
    __shared__ float warp_merge_val[(kSamplerBlock / 32) * kMtpProposalSupportCapacity];
    __shared__ int warp_merge_idx[(kSamplerBlock / 32) * kMtpProposalSupportCapacity];
    __shared__ int n_support;

    if (!(cfg.temperature > 0.0f)) {
        float best_value = -CUDART_INF_F;
        int best_index = INT_MAX;
        const int round_count = round_counts[row];
        for (int v = tid; v < proposal_rows; v += blockDim.x) {
            const int token_id = id_map == nullptr ? v : id_map[v];
            const float value = sampling_mtp_adjusted_logit(
                __bfloat162float(logits[base + v]), token_id, cfg, round_tokens, round_width,
                round_count, prior_proposals, prior_stride, prior_count, batch, row);
            if (sampling_better(value, token_id, best_value, best_index)) {
                best_value = value;
                best_index = token_id;
            }
        }
        red_val[tid] = best_value;
        red_idx[tid] = best_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride && sampling_better(red_val[tid + stride], red_idx[tid + stride],
                                                red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + stride];
                red_idx[tid] = red_idx[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int selected_id = red_idx[0];
            out_tokens[row] = selected_id;
            for (int j = 0; j < kSamplerCandidateCap; ++j) {
                candidate_ids[out_at + j] = (selected_id + j) % public_token_domain;
                proposal_q[out_at + j] = j == 0 ? 1.0f : 0.0f;
            }
        }
        return;
    }

    float local_val[kMtpProposalSupportCapacity];
    int local_idx[kMtpProposalSupportCapacity];
#pragma unroll
    for (int j = 0; j < kMtpProposalSupportCapacity; ++j) {
        local_val[j] = -CUDART_INF_F;
        local_idx[j] = INT_MAX;
    }
    const int round_count = round_counts[row];
    for (int v = tid; v < proposal_rows; v += blockDim.x) {
        const int token_id = id_map == nullptr ? v : id_map[v];
        const float value = sampling_mtp_adjusted_logit(
            __bfloat162float(logits[base + v]), token_id, cfg, round_tokens, round_width,
            round_count, prior_proposals, prior_stride, prior_count, batch, row);
        sampling_insert_candidate(local_val, local_idx, kMtpProposalSupportCapacity, value, token_id);
    }
    for (int j = 0; j < kMtpProposalSupportCapacity; ++j) {
        const int at = tid * kMtpProposalSupportCapacity + j;
        merge_val[at] = local_val[j];
        merge_idx[at] = local_idx[j];
    }
    __syncthreads();
    const int top_k = sampling_mtp_candidate_cap(cfg, proposal_rows);
    constexpr int kWarpSize = 32;
    constexpr int kProposalWarps = kSamplerBlock / kWarpSize;
    constexpr unsigned int kWarpMask = 0xffffffffu;
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;

    // Merge the 32 sorted lists within each warp in parallel. Each thread
    // advances only its own list; the warp reduction selects the next item.
    int local_position = 0;
    for (int rank = 0; rank < top_k; ++rank) {
        const int local_at = tid * kMtpProposalSupportCapacity + local_position;
        float best_value = merge_val[local_at];
        int best_index = merge_idx[local_at];
        int best_lane = lane;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            const float other_value = __shfl_down_sync(kWarpMask, best_value, offset);
            const int other_index = __shfl_down_sync(kWarpMask, best_index, offset);
            const int other_lane = __shfl_down_sync(kWarpMask, best_lane, offset);
            if (sampling_better(other_value, other_index, best_value, best_index)) {
                best_value = other_value;
                best_index = other_index;
                best_lane = other_lane;
            }
        }
        if (lane == 0) {
            const int out = warp * kMtpProposalSupportCapacity + rank;
            warp_merge_val[out] = best_value;
            warp_merge_idx[out] = best_index;
        }
        const int winner = __shfl_sync(kWarpMask, best_lane, 0);
        if (lane == winner) { ++local_position; }
    }
    __syncthreads();

    // Merge the eight sorted warp outputs in warp zero. This removes the serial
    // scan over all 5,120 candidates on one thread.
    if (warp == 0) {
        int warp_position = 0;
        for (int rank = 0; rank < top_k; ++rank) {
            float best_value = lane < kProposalWarps
                                   ? warp_merge_val[lane * kMtpProposalSupportCapacity + warp_position]
                                   : -CUDART_INF_F;
            int best_index = lane < kProposalWarps
                                 ? warp_merge_idx[lane * kMtpProposalSupportCapacity + warp_position]
                                 : INT_MAX;
            int best_lane = lane;
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                const float other_value = __shfl_down_sync(kWarpMask, best_value, offset);
                const int other_index = __shfl_down_sync(kWarpMask, best_index, offset);
                const int other_lane = __shfl_down_sync(kWarpMask, best_lane, offset);
                if (sampling_better(other_value, other_index, best_value, best_index)) {
                    best_value = other_value;
                    best_index = other_index;
                    best_lane = other_lane;
                }
            }
            if (lane == 0) {
                cand_val[rank] = best_value;
                cand_idx[rank] = best_index;
            }
            const int winner = __shfl_sync(kWarpMask, best_lane, 0);
            if (lane == winner) { ++warp_position; }
        }
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, top_k);
    if (tid == 0) {
        const int support = n_support;
        const int logical_position = logical_positions[row] + position_step;
        const float u = sampling_uniform(cfg.seed, logical_position,
                                         kSamplePurposeMtpProposal, 0u);
        float acc = 0.0f;
        int picked = cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
        out_tokens[row] = picked;
        for (int j = 0; j < kSamplerCandidateCap; ++j) {
            candidate_ids[out_at + j] = j < support ? cand_idx[j] : 0;
            proposal_q[out_at + j] = j < support ? prob[j] : 0.0f;
        }
    }
}

__launch_bounds__(kSamplerBlock) __global__ void sample_mtp_proposal_partial_topk_kernel(
    const __nv_bfloat16* logits, const SamplingConfig* cfg_ptr, const int* round_tokens,
    const int* round_counts, const int* prior_proposals, const int* id_map,
    int proposal_rows, int physical_rows, int round_width, int prior_stride, int prior_count,
    SamplingWorkspace workspace) {
    const int col = static_cast<int>(blockIdx.y);
    const int partial = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const SamplingConfig cfg = cfg_ptr[col];
    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_mtp_candidate_cap(cfg, proposal_rows);
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start = partial * kSamplerPartialTileItems;
    if (partial == 0 && tid == 0) { workspace.group_done[col] = 0; }

    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];
    unsigned long long keys[kSamplerItemsPerThread];

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int row = tile_start + item * blockDim.x + tid;
        if (row < proposal_rows) {
            const int token_id = id_map == nullptr ? row : id_map[row];
            const float raw = __bfloat162float(logits[base + row]);
            const float value = sampling_mtp_adjusted_logit(
                raw, token_id, cfg, round_tokens, round_width, round_counts[col],
                prior_proposals, prior_stride, prior_count, static_cast<int>(gridDim.y), col);
            keys[item] = sampling_sort_key(value, token_id);
        } else {
            keys[item] = 0ull;
        }
    }

    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < kSamplerItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            workspace.partial_keys[sampling_partial_offset(workspace, col, partial, 0)] = best;
        }
        return;
    }

    SamplingPartialSort(sort_storage).Sort(keys, SamplingKeyGreater{});
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = tid * kSamplerItemsPerThread + item;
        if (rank < cap) {
            workspace.partial_keys[sampling_partial_offset(workspace, col, partial, rank)] =
                keys[item];
        }
    }
}

__launch_bounds__(kSamplerGroupBlock) __global__ void sample_mtp_proposal_group_finalize_kernel(
    int* out_tokens, int* candidate_ids, float* proposal_q, const SamplingConfig* cfg_ptr,
    const int* logical_positions, int public_token_domain, int proposal_rows, int proposal_window,
    int prior_count, int proposal_step, int position_step, int partial_blocks, int group_count,
    SamplingWorkspace workspace) {
    const int group = static_cast<int>(blockIdx.x);
    const int col = static_cast<int>(blockIdx.y);
    const int tid = threadIdx.x;
    const SamplingConfig cfg = cfg_ptr[col];
    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_mtp_candidate_cap(cfg, proposal_rows);
    const int out_at = (col * proposal_window + proposal_step) * kSamplerCandidateCap;

    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last;
    __shared__ unsigned long long greedy_warp_keys[kSamplerGroupBlock / 32];
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) {
        group_partials = kSamplerPartialsPerGroup;
    }

    if (greedy) {
        unsigned long long best = 0ull;
        for (int p = tid; p < group_partials; p += blockDim.x) {
            const auto key = workspace.partial_keys[
                sampling_partial_offset(workspace, col, group_begin + p, 0)];
            if (key > best) { best = key; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            workspace.partial_keys[sampling_partial_offset(
                workspace, col, partial_blocks + group, 0)] = best;
            __threadfence();
            const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
            is_last = done == group_count ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { return; }

        best = 0ull;
        for (int p = tid; p < group_count; p += blockDim.x) {
            const auto key = workspace.partial_keys[
                sampling_partial_offset(workspace, col, partial_blocks + p, 0)];
            if (key > best) { best = key; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int picked = sampling_key_index(best);
            out_tokens[col] = picked;
            for (int j = 0; j < kSamplerCandidateCap; ++j) {
                candidate_ids[out_at + j] = (picked + j) % public_token_domain;
                proposal_q[out_at + j] = j == 0 ? 1.0f : 0.0f;
            }
            workspace.group_done[col] = 0;
        }
        return;
    }

    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int rank = p - (p / cap) * cap;
            keys[item] = workspace.partial_keys[
                sampling_partial_offset(workspace, col, partial, rank)];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            workspace.partial_keys[sampling_partial_offset(
                workspace, col, partial_blocks + group, rank)] = keys[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last = done == group_count ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int rank = p - (p / cap) * cap;
            keys[item] = workspace.partial_keys[
                sampling_partial_offset(workspace, col, partial, rank)];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        const int logical_position = logical_positions[col] + position_step;
        const float u = sampling_uniform(cfg.seed, logical_position,
                                         kSamplePurposeMtpProposal, 0u);
        float acc = 0.0f;
        int picked = cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
        out_tokens[col] = picked;
        for (int j = 0; j < kSamplerCandidateCap; ++j) {
            candidate_ids[out_at + j] = j < support ? cand_idx[j] : 0;
            proposal_q[out_at + j] = j < support ? prob[j] : 0.0f;
        }
        workspace.group_done[col] = 0;
    }
}

__launch_bounds__(kSamplerBlock) __global__
    void sampling_partial_topk_kernel(const __nv_bfloat16* logits, const SamplingConfig* cfg_ptr,
                                      std::int32_t token_domain, std::int32_t physical_rows,
                                      SamplingWorkspace workspace) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = cfg_ptr[col];
    if (partial == 0 && threadIdx.x == 0) { workspace.group_done[col] = 0; }

    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];
    unsigned long long keys[kSamplerItemsPerThread];

    const bool greedy       = !(cfg.temperature > 0.0f);
    const bool penalties    = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    const int cap           = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start    = partial * kSamplerPartialTileItems;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const float raw = __bfloat162float(logits[base + v]);
            const float x   = penalties ? sampling_adjusted_logit(raw, v, cfg) : raw;
            keys[item]      = sampling_sort_key(x, v);
        } else {
            keys[item] = 0ull;
        }
    }
    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < kSamplerItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (threadIdx.x == 0) {
            const int off               = sampling_partial_offset(workspace, col, partial, 0);
            workspace.partial_keys[off] = best;
        }
        return;
    }
    SamplingPartialSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < cap) {
            const int off               = sampling_partial_offset(workspace, col, partial, rank);
            workspace.partial_keys[off] = keys[item];
        }
    }
}

__launch_bounds__(kSamplerGroupBlock) __global__ void sampling_group_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* logical_positions,
    std::int32_t purpose, std::int32_t token_domain, std::int32_t partial_blocks,
    std::int32_t group_count, SamplingWorkspace workspace) {
    const int group          = static_cast<int>(blockIdx.x);
    const int col            = static_cast<int>(blockIdx.y);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = cfg_ptr[col];
    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last;
    __shared__ unsigned long long greedy_warp_keys[kSamplerGroupBlock / 32];
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap     = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes group_done[col], so caller-owned
    // workspace does not rely on prior contents or a separate memset launch.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }

    if (greedy) {
        unsigned long long best = 0ull;
        for (int p = tid; p < group_partials; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, group_begin + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int out_off = sampling_partial_offset(workspace, col, partial_blocks + group, 0);
            workspace.partial_keys[out_off] = best;
            __threadfence();
            const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
            is_last        = (done == group_count) ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { return; }

        best = 0ull;
        for (int p = tid; p < group_count; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, partial_blocks + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int picked = sampling_key_index(best);
            out[col]         = picked;
            if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
            workspace.group_done[col] = 0;
        }
        return;
    }

    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off =
                sampling_partial_offset(workspace, col, partial_blocks + group, rank);
            workspace.partial_keys[out_off] = keys[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last        = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        const float u     = sampling_uniform(cfg.seed, logical_positions[col], purpose, 0u);
        float acc         = 0.0f;
        int picked        = cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
        out[col] = picked;
        if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        workspace.group_done[col] = 0;
    }
}

} // namespace ninfer::ops

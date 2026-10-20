#pragma once

#include "models/qwen3_5/program/round_buffers.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5 {

inline constexpr std::size_t kMtpQ0ObserverMaximumTraceRounds = 4096;

struct MtpQ0DepthMetric {
    float overlap          = 0.0F;
    float support_coverage = 0.0F;
};

struct MtpQ0RoundRecord {
    std::uint32_t lane                         = 0;
    std::uint64_t round_index                  = 0;
    std::uint32_t base_frontier                = 0;
    std::uint32_t current_extent               = 0;
    std::uint32_t verifier_licensed_count      = 0;
    std::uint32_t verifier_accepted_drafts     = 0;
    std::uint32_t engine_committed_count       = 0;
    bool initial_prefill_one_hot               = false;
    bool fallback                             = false;
    bool engine_terminal                      = false;
    bool engine_cancelled                      = false;
    bool aborted                              = false;
    std::array<MtpQ0DepthMetric, kMtpDecodeMaximumDrafts> depths{};
};

[[nodiscard]] inline MtpQ0RoundRecord make_mtp_q0_round_record(
    std::uint32_t lane, std::uint32_t metrics_row, std::uint64_t round_index,
    std::uint32_t base_frontier,
    std::uint32_t current_extent, std::uint32_t verifier_licensed_count,
    std::uint32_t verifier_accepted_drafts, bool initial_prefill_one_hot,
    const float* q0_observer_metrics, std::uint32_t metric_k_stride) noexcept {
    MtpQ0RoundRecord record;
    record.lane                     = lane;
    record.round_index              = round_index;
    record.base_frontier            = base_frontier;
    record.current_extent           = current_extent;
    record.verifier_licensed_count  = verifier_licensed_count;
    record.verifier_accepted_drafts = verifier_accepted_drafts;
    record.initial_prefill_one_hot  = initial_prefill_one_hot && current_extent != 0;
    record.fallback                 = current_extent == 0;
    if (q0_observer_metrics != nullptr) {
        const std::uint32_t live_depths =
            current_extent < kMtpDecodeMaximumDrafts ? current_extent : kMtpDecodeMaximumDrafts;
        for (std::uint32_t depth = 0; depth < live_depths; ++depth) {
            const std::size_t at = 2ULL * (depth + metric_k_stride * metrics_row);
            record.depths[depth] =
                MtpQ0DepthMetric{q0_observer_metrics[at], q0_observer_metrics[at + 1]};
        }
    }
    return record;
}

[[nodiscard]] inline MtpQ0RoundRecord resolve_mtp_q0_round_record(
    MtpQ0RoundRecord record, std::uint32_t engine_committed_count, bool engine_terminal,
    bool engine_cancelled) noexcept {
    record.engine_committed_count = engine_committed_count;
    record.engine_terminal        = engine_terminal;
    record.engine_cancelled       = engine_cancelled;
    return record;
}

[[nodiscard]] inline MtpQ0RoundRecord abort_mtp_q0_round_record(
    MtpQ0RoundRecord record) noexcept {
    record.engine_committed_count = 0;
    record.engine_terminal        = false;
    record.engine_cancelled       = false;
    record.aborted                = true;
    return record;
}

struct MtpQ0ObserverTrace {
    bool enabled   = false;
    bool truncated = false;
    std::vector<MtpQ0RoundRecord> rounds;
};

// Program-owned, fixed-capacity diagnostics. The record type is allocation-free so the
// constructor's one-time reserve keeps append out of the decode/commit allocation path.
class MtpQ0ObserverTraceBuffer {
public:
    void enable() {
        rounds_.reserve(kMtpQ0ObserverMaximumTraceRounds);
        enabled_ = true;
    }

    void append(const MtpQ0RoundRecord& record) {
        if (rounds_.size() >= kMtpQ0ObserverMaximumTraceRounds) {
            truncated_ = true;
            return;
        }
        rounds_.push_back(record);
    }

    [[nodiscard]] bool truncated() const noexcept { return truncated_; }
    [[nodiscard]] const std::vector<MtpQ0RoundRecord>& rounds() const noexcept { return rounds_; }
    [[nodiscard]] MtpQ0ObserverTrace snapshot(std::size_t first_round = 0) const {
        first_round = std::min(first_round, rounds_.size());
        return MtpQ0ObserverTrace{
            .enabled = enabled_,
            .truncated = truncated_,
            .rounds = std::vector<MtpQ0RoundRecord>(rounds_.begin() +
                                                        static_cast<std::ptrdiff_t>(first_round),
                                                    rounds_.end()),
        };
    }

private:
    std::vector<MtpQ0RoundRecord> rounds_;
    bool enabled_   = false;
    bool truncated_ = false;
};

} // namespace ninfer::models::qwen3_5

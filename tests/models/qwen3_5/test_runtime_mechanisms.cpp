#include "core/layout.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_5/program/speculative/mtp_alignment.h"
#include "models/qwen3_5/program/speculative/q0_observer.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/program/vision_control.h"

#include "models/qwen3_5/program/prefix_identity.h"
#include "models/qwen3_5/program/planning/rebuild_work.h"
#include "runtime/engine/q0_observer_sink.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

namespace q36 = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q36::DecoderStateSpec decoder_spec(ninfer::KvCacheStorage storage, bool mtp) {
    return q36::DecoderStateSpec{
        .full_attention_layers     = 2,
        .mtp_layers                = 1,
        .capacity                  = 129,
        .kv_heads                  = 2,
        .attention_head_dim        = 256,
        .kv_storage                = storage,
        .enable_mtp                = mtp,
        .text_physical_page_groups = 5,
        .mtp_physical_page_groups  = mtp ? 4U : 0U,
    };
}

void test_decoder_layout() {
    ninfer::LayoutBuilder bf16_builder;
    const q36::DecoderStateLayout bf16 = q36::plan_decoder_state(
        bf16_builder, decoder_spec(ninfer::KvCacheStorage::BFloat16, false));
    (void)bf16_builder.finish(256);
    expect(bf16.text_kv.pages.planes.size() == 4, "BF16 Text KV has K/V planes per layer");
    expect(bf16.text_kv.pages.spec.page_group_count == 5 &&
               bf16.text_kv.execution_tables.spec.logical_page_capacity == 3 &&
               bf16.text_kv.execution_tables.spec.table_rows == 1,
           "Text KV separates five physical pages from three logical pages");
    expect(bf16.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::BF16 &&
               bf16.text_kv.pages.planes[1].geometry.dtype == ninfer::DType::FP16 &&
               bf16.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::BF16 &&
               bf16.text_kv.pages.planes[3].geometry.dtype == ninfer::DType::FP16,
           "BF16 KV has BF16 K and FP16 V without scale planes");
    expect(!bf16.mtp_kv.has_value(), "disabled MTP omits KV storage");
    expect(bf16.kv_payload_bytes() == bf16.text_kv.payload_bytes(), "BF16 KV payload accounting");

    ninfer::LayoutBuilder int8_builder;
    const q36::DecoderStateLayout int8 = q36::plan_decoder_state(
        int8_builder, decoder_spec(ninfer::KvCacheStorage::Int8Group64, true));
    (void)int8_builder.finish(256);
    expect(int8.text_kv.pages.planes.size() == 8 &&
               int8.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               int8.text_kv.pages.planes[3].geometry.dtype == ninfer::DType::FP16,
           "INT8 Text KV has code and scale planes per layer");
    expect(int8.mtp_kv.has_value() && int8.mtp_kv->layers == 1 &&
               int8.mtp_kv->pages.planes.size() == 4 &&
               int8.mtp_kv->pages.spec.page_group_count == 4 &&
               int8.mtp_kv->execution_tables.spec.logical_page_capacity == 3,
           "enabled MTP has one paged KV layer");
    expect(int8.mtp_kv && int8.mtp_kv->pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               int8.mtp_kv->pages.planes[3].geometry.dtype == ninfer::DType::FP16,
           "INT8 MTP KV has scale planes");
    expect(int8.kv_payload_bytes() == int8.text_kv.payload_bytes() + int8.mtp_kv->payload_bytes(),
           "INT8 Text/MTP KV payload accounting");

    q36::DecoderStateSpec fp8_spec = decoder_spec(ninfer::KvCacheStorage::Fp8E4M3Row256, true);
    ninfer::LayoutBuilder fp8_builder;
    const q36::DecoderStateLayout fp8 = q36::plan_decoder_state(fp8_builder, fp8_spec);
    (void)fp8_builder.finish(256);
    expect(fp8.text_kv.pages.planes.size() == 8 &&
               fp8.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               fp8.text_kv.pages.planes[2].geometry.leading_extent == 1,
           "FP8 Text KV has row-scaled code and scale planes per layer");
    expect(fp8.mtp_kv && fp8.mtp_kv->pages.planes.size() == 4 &&
               fp8.mtp_kv->pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.mtp_kv->pages.planes[2].geometry.leading_extent == 1,
           "FP8 MTP KV has row-scaled code and scale planes");
    expect(fp8.kv_payload_bytes() == fp8.text_kv.payload_bytes() + fp8.mtp_kv->payload_bytes(),
           "FP8 Text/MTP KV payload accounting");

    ninfer::LayoutBuilder nvfp4_builder;
    const q36::DecoderStateLayout nvfp4 = q36::plan_decoder_state(
        nvfp4_builder, decoder_spec(ninfer::KvCacheStorage::Nvfp4Group16, true));
    (void)nvfp4_builder.finish(256);
    expect(nvfp4.text_kv.pages.planes.size() == 8 &&
               nvfp4.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::U8 &&
               nvfp4.text_kv.pages.planes[0].geometry.leading_extent == 128 &&
               nvfp4.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::U8 &&
               nvfp4.text_kv.pages.planes[2].geometry.leading_extent == 16,
           "NVFP4 Text KV has packed E2M1 code and E4M3 scale planes");
    constexpr std::size_t nvfp4_vector_bytes = 128 + 16;
    constexpr std::size_t text_vectors       = 2ULL * 5ULL * 64ULL * 2ULL;
    constexpr std::size_t mtp_vectors        = 1ULL * 4ULL * 64ULL * 2ULL;
    expect(nvfp4.text_kv.payload_bytes() == 2ULL * nvfp4_vector_bytes * text_vectors &&
               nvfp4.mtp_kv &&
               nvfp4.mtp_kv->payload_bytes() == 2ULL * nvfp4_vector_bytes * mtp_vectors,
           "NVFP4 Text/MTP physical payload bytes");

    ninfer::LayoutBuilder k8v4_builder;
    const q36::DecoderStateLayout k8v4 = q36::plan_decoder_state(
        k8v4_builder, decoder_spec(ninfer::KvCacheStorage::Fp8KeyNvfp4Value, true));
    (void)k8v4_builder.finish(256);
    expect(k8v4.text_kv.pages.planes.size() == 8 &&
               k8v4.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               k8v4.text_kv.pages.planes[0].geometry.leading_extent == 256 &&
               k8v4.text_kv.pages.planes[1].geometry.dtype == ninfer::DType::U8 &&
               k8v4.text_kv.pages.planes[1].geometry.leading_extent == 128 &&
               k8v4.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               k8v4.text_kv.pages.planes[2].geometry.leading_extent == 1 &&
               k8v4.text_kv.pages.planes[3].geometry.dtype == ninfer::DType::U8 &&
               k8v4.text_kv.pages.planes[3].geometry.leading_extent == 16,
           "K8V4 Text KV preserves independent key/value code and scale geometry");
    constexpr std::size_t k8_vector_bytes = 256 + 2;
    constexpr std::size_t v4_vector_bytes = 128 + 16;
    expect(k8v4.text_kv.payload_bytes() == (k8_vector_bytes + v4_vector_bytes) * text_vectors &&
               k8v4.mtp_kv &&
               k8v4.mtp_kv->payload_bytes() == (k8_vector_bytes + v4_vector_bytes) * mtp_vectors,
           "K8V4 Text/MTP asymmetric physical payload bytes");
}

void test_round_layout() {
    ninfer::LayoutBuilder builder;
    q36::RoundStateLayout round = q36::begin_round_state_layout(
        builder, q36::RoundStateSpec{.hidden       = 32,
                                     .output_rows  = 128,
                                     .draft_window = 5,
                                     .backend      = ninfer::SpeculativeBackend::Mtp});
    const ninfer::TensorRegion exact_prefill =
        builder.add_tensor(ninfer::DType::BF16, {32, 16}, 256, "exact prefill hidden");
    q36::complete_round_state_layout(builder, round);
    (void)builder.finish(256);
    expect(round.complete, "round layout completes");
    expect(round.logits.shape[0] == 128 && round.logits.shape[1] == 1, "round logits shape");
    expect(round.mtp.has_value() && round.mtp->draft_tokens.shape[0] == 5 &&
               round.mtp->target_input_ids.shape[0] == 6,
           "MTP prefill scratch shapes");
    expect(round.logits.region.offset < exact_prefill.region.offset &&
               exact_prefill.region.offset < round.mtp->draft_tokens.region.offset,
           "exact prefill extension retains established round-region order");
    expect(round.mtp.has_value() && round.mtp->position.shape[0] == 1,
           "MTP prefill scratch is explicit");
    expect(round.mtp_decode.has_value() && round.mtp_decode->alignment_ids.shape[0] == 6 &&
               round.mtp_decode->alignment_ids.shape[1] == 1,
           "MTP decode frame is explicit");

    ninfer::LayoutBuilder speculative_builder;
    q36::RoundStateLayout dflash = q36::begin_round_state_layout(
        speculative_builder, q36::RoundStateSpec{.hidden       = 32,
                                                 .output_rows  = 128,
                                                 .draft_window = 15,
                                                 .backend = ninfer::SpeculativeBackend::DFlash});
    q36::complete_round_state_layout(speculative_builder, dflash);
    (void)speculative_builder.finish(256);
    expect(dflash.logits.shape[1] == 1 && dflash.dflash_prefill.has_value() &&
               dflash.dflash_prefill->produced_count.shape[0] == 1 &&
               dflash.dflash_decode.has_value() &&
               dflash.dflash_decode->draft_tokens.shape[0] == 15,
           "K=15 DFlash storage is backend-owned");
    expect(!dflash.mtp.has_value() && !dflash.mtp_decode.has_value(),
           "DFlash layout does not allocate MTP storage");

    ninfer::LayoutBuilder scoring_builder;
    auto scoring = q36::begin_round_state_layout(
        scoring_builder, {.hidden = 32, .output_rows = 128, .causal_scoring = true});
    q36::complete_round_state_layout(scoring_builder, scoring);
    expect(scoring.rope_delta.region.bytes != 0 && scoring.text_kv_table_row.region.bytes != 0,
           "scoring keeps its Text prefill controls");
    expect(!scoring.ordinary && !scoring.mtp_decode && !scoring.dflash_decode &&
               scoring.token.region.bytes == 0 && scoring.logits.region.bytes == 0,
           "scoring does not reserve generation frames or sampled output");
}

void test_mtp_alignment() {
    const std::vector<std::int32_t> scatter{2, 4, 7};
    const q36::MtpAlignmentWindow first = q36::plan_mtp_alignment_window(8, 0, 4);
    expect(first.hidden_begin == 0 && first.position_begin == 0 &&
               first.shifted_embedding_begin == 1 && first.columns == 4 &&
               !first.final_column_uses_generated_token,
           "non-final MTP alignment window");
    const q36::MtpVisualOverlap first_visual = q36::shifted_visual_overlap(scatter, 8, first);
    expect(first_visual.source_begin == 0 &&
               first_visual.destination_columns == std::vector<std::int32_t>({1, 3}),
           "non-final shifted visual overlap");

    const q36::MtpAlignmentWindow final = q36::plan_mtp_alignment_window(8, 4, 4);
    expect(final.shifted_embedding_begin == 5 && final.final_column_uses_generated_token,
           "final MTP alignment window");
    const q36::MtpVisualOverlap final_visual = q36::shifted_visual_overlap(scatter, 8, final);
    expect(final_visual.source_begin == 2 &&
               final_visual.destination_columns == std::vector<std::int32_t>({2}),
           "final shifted visual overlap excludes generated-token column");
}

void test_vision_control() {
    q36::PreparedPromptData prompt;
    prompt.token_ids.resize(7);
    prompt.token_types           = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0};
    prompt.prepare.media_items   = 2;
    prompt.prepare.raw_patches   = 12;
    prompt.prepare.vision_tokens = 3;
    prompt.vision_items          = {
        q36::VisionItem{.modality    = q36::PromptModality::Image,
                                 .grid        = {.temporal = 1, .height = 2, .width = 2},
                                 .patch_begin = 0,
                                 .patch_count = 4,
                                 .token_spans = {{.begin = 1, .count = 1}}},
        q36::VisionItem{.modality    = q36::PromptModality::Video,
                                 .grid        = {.temporal = 2, .height = 2, .width = 2},
                                 .patch_begin = 4,
                                 .patch_count = 8,
                                 .token_spans = {{.begin = 3, .count = 1}, {.begin = 5, .count = 1}}},
    };

    const q36::VisionControlPlan plan =
        q36::plan_vision_control(prompt, {.spatial_merge_size = 2, .position_grid_side = 48});
    const q36::VisionControl control = q36::build_vision_control(prompt, plan, 0);
    expect(control.items.size() == 2, "Vision per-item control count");
    expect(control.items[0].patch_begin == 0 && control.items[0].patch_count == 4 &&
               control.items[0].merged_count == 1 && control.items[0].segment_length == 4 &&
               control.items[0].segment_count == 1 &&
               control.items[0].scatter_indices == std::vector<std::int32_t>({1}) &&
               control.items[0].position_ids.size() == 8 &&
               control.items[0].position_table_indices.size() == 16 &&
               control.items[0].position_table_weights.size() == 16,
           "image item control offsets");
    expect(control.items[1].patch_begin == 4 && control.items[1].patch_count == 8 &&
               control.items[1].merged_count == 2 && control.items[1].segment_length == 4 &&
               control.items[1].segment_count == 2 &&
               control.items[1].scatter_indices == std::vector<std::int32_t>({3, 5}) &&
               control.items[1].position_ids.size() == 16 &&
               control.items[1].position_table_indices.size() == 32 &&
               control.items[1].position_table_weights.size() == 32,
           "video item control offsets");

    const q36::VisionControl suffix = q36::build_vision_control(prompt, plan, 1);
    expect(suffix.prepared_item_begin == 1 && suffix.items.size() == 1 &&
               suffix.items[0].patch_begin == control.items[1].patch_begin &&
               suffix.items[0].scatter_indices == control.items[1].scatter_indices &&
               suffix.items[0].position_ids == control.items[1].position_ids,
           "Vision suffix control contents");
}

q36::PreparedPromptData identity_prompt(std::uint8_t digest_byte = 1) {
    q36::PreparedPromptData prompt;
    prompt.token_ids   = {10, 248056, 248056, 11};
    prompt.token_types = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                          static_cast<std::uint8_t>(q36::PromptModality::Image), 0};
    prompt.positions   = {0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 2, 3};
    prompt.rope_delta  = 0;
    q36::VisionItem item{.modality    = q36::PromptModality::Image,
                         .grid        = {.temporal = 1, .height = 2, .width = 4},
                         .patch_begin = 0,
                         .patch_count = 8,
                         .token_spans = {{.begin = 1, .count = 2}}};
    item.content_digest.fill(digest_byte);
    prompt.vision_items.push_back(std::move(item));
    return prompt;
}

void append_text_token(q36::PreparedPromptData& prompt, ninfer::TokenId token,
                       std::int32_t position) {
    const std::size_t old_tokens = prompt.token_ids.size();
    std::vector<std::int32_t> positions;
    positions.reserve(3 * (old_tokens + 1));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * old_tokens);
        positions.insert(positions.end(), begin, begin + static_cast<std::ptrdiff_t>(old_tokens));
        positions.push_back(position);
    }
    prompt.token_ids.push_back(token);
    prompt.token_types.push_back(0);
    prompt.positions = std::move(positions);
}

void test_prefix_identity() {
    q36::PreparedPromptData original    = identity_prompt();
    std::vector<ninfer::TokenId> ledger = original.token_ids;
    q36::detail::ResidentPrefixIdentity resident;
    q36::detail::PrefixShortlistDigests digests;
    resident.reserve(16);
    resident.assign(original);
    digests.reserve(16);
    digests.assign(original);

    expect(q36::detail::prefix_matches(original, ledger, resident, original.token_ids.size()),
           "identical multimodal prefix identity");

    q36::PreparedPromptData changed_media = identity_prompt(2);
    expect(!q36::detail::prefix_matches(changed_media, ledger, resident,
                                        changed_media.token_ids.size()),
           "different media content must not reuse placeholder tokens");
    expect(q36::detail::prefix_matches(changed_media, ledger, resident, 1),
           "media wholly after the frontier does not affect prefix identity");
    expect(!q36::detail::prefix_matches(original, ledger, resident, 2),
           "frontier must not divide one Vision item");

    q36::PreparedPromptData changed_position = identity_prompt();
    changed_position.positions[0] += 1;
    expect(!q36::detail::prefix_matches(changed_position, ledger, resident,
                                        changed_position.token_ids.size()),
           "different MRoPE positions must not reuse resident state");

    q36::PreparedPromptData changed_decomposition              = identity_prompt();
    changed_decomposition.identity.rewrite_execution_frontiers = {1};
    expect(!q36::detail::prefix_matches(changed_decomposition, ledger, resident,
                                        changed_decomposition.token_ids.size()),
           "different GDN execution decomposition must not reuse resident state");
    changed_decomposition.identity.rewrite_execution_frontiers = {4};
    expect(q36::detail::prefix_matches(changed_decomposition, ledger, resident, 3),
           "execution decomposition wholly after the frontier changed prefix identity");

    q36::PreparedPromptData resident_future              = identity_prompt();
    resident_future.identity.rewrite_execution_frontiers = {1, 4};
    q36::detail::ResidentPrefixIdentity resident_with_future;
    resident_with_future.assign(resident_future);
    q36::PreparedPromptData incoming_future              = identity_prompt();
    incoming_future.identity.rewrite_execution_frontiers = {1, 3};
    expect(q36::detail::prefix_matches(incoming_future, ledger, resident_with_future, 1),
           "resident execution decomposition after the frontier changed prefix identity");
    expect(!q36::detail::prefix_matches(incoming_future, ledger, resident_with_future, 3),
           "different execution decomposition inside the frontier reused resident state");

    q36::detail::PrefixShortlistDigests future_digest;
    future_digest.assign(resident_future);
    q36::detail::PrefixShortlistDigests incoming_digest;
    incoming_digest.assign(incoming_future);
    expect(future_digest.at(1) == incoming_digest.at(1),
           "future execution boundaries changed an earlier content shortlist");
    expect(future_digest.at(3) != incoming_digest.at(3),
           "different in-prefix execution boundaries shared a shortlist digest");

    resident.append_generated(1, original.rope_delta);
    ledger.push_back(12);
    const std::array<ninfer::TokenId, 1> generated{12};
    digests.append_generated(generated, original.rope_delta);
    append_text_token(original, 12, 4);
    q36::detail::PrefixShortlistDigests rebuilt;
    rebuilt.assign(original);
    expect(digests.at(ledger.size()) == rebuilt.at(ledger.size()),
           "incremental generated-token shortlist diverged from a full rebuild");
    expect(q36::detail::prefix_matches(original, ledger, resident, ledger.size()),
           "generated multimodal continuation identity");

    q36::PreparedPromptData accepted_rebuild = identity_prompt();
    const std::array<ninfer::TokenId, 3> proposed{20, 21, 22};
    const std::span<const ninfer::TokenId> accepted(proposed.data(), 2);
    std::vector<ninfer::TokenId> accepted_ledger = accepted_rebuild.token_ids;
    accepted_ledger.insert(accepted_ledger.end(), accepted.begin(), accepted.end());
    q36::detail::ResidentPrefixIdentity accepted_resident;
    q36::detail::PrefixShortlistDigests accepted_digests;
    accepted_resident.assign(accepted_rebuild);
    accepted_digests.assign(accepted_rebuild);
    accepted_resident.append_generated(accepted.size(), accepted_rebuild.rope_delta, 1);
    accepted_digests.append_generated(accepted, accepted_rebuild.rope_delta, 1);
    append_text_token(accepted_rebuild, accepted[0], 4);
    append_text_token(accepted_rebuild, accepted[1], 5);
    accepted_rebuild.identity.rewrite_execution_frontiers = {5};
    q36::detail::PrefixShortlistDigests rebuilt_accepted_digests;
    rebuilt_accepted_digests.assign(accepted_rebuild);
    expect(q36::detail::prefix_matches(accepted_rebuild, accepted_ledger, accepted_resident,
                                       accepted_ledger.size()) &&
               accepted_digests.at(accepted_ledger.size()) ==
                   rebuilt_accepted_digests.at(accepted_ledger.size()),
           "accepted generated prefix and rebuilt history formed different cache identities");

    const q36::PreparedPromptData prompt_only = identity_prompt();
    resident.truncate(prompt_only.token_ids.size());
    digests.truncate(prompt_only.token_ids.size());
    ledger.resize(prompt_only.token_ids.size());
    q36::detail::PrefixShortlistDigests prompt_digest;
    prompt_digest.assign(prompt_only);
    expect(digests.at(ledger.size()) == prompt_digest.at(ledger.size()),
           "truncated shortlist did not restore the original frontier digest");
    expect(q36::detail::prefix_matches(prompt_only, ledger, resident, ledger.size()),
           "truncated multimodal continuation identity");
}

void test_rebuild_work_prompt_frontier_boundary() {
    constexpr std::uint32_t prompt_tokens = 100;
    constexpr std::uint32_t prefill_chunk = 2048;
    std::uint32_t tail_begin              = 0;
    q36::runtime_support::include_rebuild_boundary(tail_begin, prompt_tokens, prompt_tokens);
    expect(tail_begin == prompt_tokens,
           "prompt-frontier rebuild boundary was not retained for continuation growth");

    ninfer::runtime::PrefillWork work =
        ninfer::runtime::make_prefill_work(0, prompt_tokens, 0, 0, prefill_chunk);
    q36::runtime_support::advance_segmented_rebuild_work(work, tail_begin, prompt_tokens,
                                                         prompt_tokens + 1, prefill_chunk);
    const ninfer::runtime::PrefillWork exact =
        ninfer::runtime::make_prefill_work(0, prompt_tokens + 1, 0, 0, prefill_chunk);
    expect(work.chunks == 2 && work.tokens == exact.tokens &&
               work.attention_pairs == exact.attention_pairs,
           "continuation growth did not preserve the prompt-frontier rebuild split");
}

void test_mtp_q0_observer_trace() {
    constexpr std::uint32_t metric_k_stride = q36::kMtpDecodeMaximumDrafts;
    constexpr std::uint32_t metric_row = 1;
    std::array<float, 2 * q36::kMtpDecodeMaximumDrafts * ninfer::kMaximumConcurrency> metrics{};
    metrics[2 * (0 + metric_k_stride * metric_row)] = 0.25F;
    metrics[2 * (0 + metric_k_stride * metric_row) + 1] = 0.75F;
    metrics[2 * (1 + metric_k_stride * metric_row)] = 0.125F;
    metrics[2 * (1 + metric_k_stride * metric_row) + 1] = 0.5F;

    const q36::MtpQ0RoundRecord record = q36::make_mtp_q0_round_record(
        6, metric_row, 9, 321, 2, 3, 2, true, metrics.data(), metric_k_stride);
    expect(record.lane == 6 && record.round_index == 9 && record.base_frontier == 321 &&
               record.current_extent == 2 && record.verifier_licensed_count == 3 &&
               record.verifier_accepted_drafts == 2 && record.initial_prefill_one_hot &&
               !record.fallback && record.depths[0].overlap == 0.25F &&
               record.depths[0].support_coverage == 0.75F &&
               record.depths[1].overlap == 0.125F &&
               record.depths[1].support_coverage == 0.5F &&
               record.depths[2].overlap == 0.0F,
           "MTP Q0 record retains lane/frontier/depth metrics and the initial one-hot marker");

    const q36::MtpQ0RoundRecord fallback = q36::make_mtp_q0_round_record(
        4, 0, 10, 99, 0, 1, 0, true, metrics.data(), metric_k_stride);
    expect(fallback.fallback && !fallback.initial_prefill_one_hot &&
               fallback.depths[0].overlap == 0.0F,
           "MTP Q0 fallback record has no live proposal metrics");

    const q36::MtpQ0RoundRecord partial_commit =
        q36::resolve_mtp_q0_round_record(record, 1, true, false);
    const q36::MtpQ0RoundRecord cancelled =
        q36::resolve_mtp_q0_round_record(record, 0, true, true);
    const q36::MtpQ0RoundRecord aborted = q36::abort_mtp_q0_round_record(record);
    expect(partial_commit.verifier_licensed_count == 3 &&
               partial_commit.engine_committed_count == 1 && partial_commit.engine_terminal &&
               !partial_commit.engine_cancelled && !partial_commit.aborted &&
               cancelled.engine_committed_count == 0 && cancelled.engine_terminal &&
               cancelled.engine_cancelled && !cancelled.aborted && aborted.aborted &&
               aborted.engine_committed_count == 0 && !aborted.engine_terminal &&
               !aborted.engine_cancelled,
           "MTP Q0 trace separates verifier licensing from Engine commit, cancel, and abort");

    q36::MtpQ0ObserverTraceBuffer buffer;
    expect(!buffer.snapshot().enabled && buffer.snapshot().rounds.empty(),
           "MTP Q0 trace is disabled before opt-in");
    buffer.enable();
    buffer.append(record);
    for (std::size_t i = 1; i < q36::kMtpQ0ObserverMaximumTraceRounds; ++i) {
        buffer.append(fallback);
    }
    buffer.append(fallback);
    const q36::MtpQ0ObserverTrace trace = buffer.snapshot();
    expect(trace.enabled && trace.truncated &&
               trace.rounds.size() == q36::kMtpQ0ObserverMaximumTraceRounds &&
               trace.rounds.front().round_index == 9,
           "MTP Q0 trace retrieval is bounded and reports truncation");
    const q36::MtpQ0ObserverTrace suffix = buffer.snapshot(4095);
    expect(suffix.rounds.size() == 1 && suffix.rounds.front().round_index == fallback.round_index,
           "MTP Q0 trace snapshots can read only records not yet exported");
}

void test_token_id_identity_hash() {
    const std::array<ninfer::TokenId, 3> tokens{
        static_cast<ninfer::TokenId>(-1), 0, static_cast<ninfer::TokenId>(0x01020304)};
    expect(ninfer::models::qwen3_5::frontend::token_ids_sha256_hex(tokens) ==
               "5a5fe8cd8d307232c23740541f240323ea11d16c04c906a065f800b71f3a1801",
           "token identity hashes int32 bit patterns in little-endian order");
    expect(ninfer::models::qwen3_5::frontend::token_ids_sha256_hex(
               std::span<const ninfer::TokenId>{}) ==
               "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
           "empty committed sequences have the SHA-256 identity of empty bytes");
}

struct Q0ExportTestProgram {
    Q0ExportTestProgram() { trace.enable(); }

    [[nodiscard]] q36::MtpQ0ObserverTrace mtp_q0_observer_trace(std::size_t first_round) const {
        return trace.snapshot(first_round);
    }

    q36::MtpQ0ObserverTraceBuffer trace;
};

struct Q0ExportTestRequest {
    std::uint64_t id = 0;
};

struct Q0ExportTestResult {
    std::vector<ninfer::TokenId> tokens;
    std::uint64_t committed_tokens = 0;
};

Q0ExportTestResult run_q0_export_control(bool enabled, const std::string& path) {
    Q0ExportTestProgram program;
    std::array<std::shared_ptr<Q0ExportTestRequest>, 1> slots{};
    std::optional<ninfer::runtime::Q0ObserverExport> exporter;
    if (enabled) {
        exporter.emplace(path);
    } else {
        exporter.emplace();
    }

    Q0ExportTestResult result;
    slots[0] = std::make_shared<Q0ExportTestRequest>(Q0ExportTestRequest{.id = 101});
    q36::MtpQ0RoundRecord committed;
    committed.lane = 0;
    committed.round_index = 4;
    committed.current_extent = 1;
    committed.engine_committed_count = 2;
    result.tokens = {17, 19};
    result.committed_tokens = 2;
    const std::array<ninfer::TokenId, 2> prompt_tokens{5, 7};
    exporter->append_prompt_identity(
        101, prompt_tokens.size(),
        ninfer::models::qwen3_5::frontend::token_ids_sha256_hex(prompt_tokens));
    program.trace.append(committed);
    exporter->capture(program, slots);
    exporter->append_committed_identity(
        101, result.tokens.size(),
        ninfer::models::qwen3_5::frontend::token_ids_sha256_hex(result.tokens));

    slots[0] = std::make_shared<Q0ExportTestRequest>(Q0ExportTestRequest{.id = 102});
    q36::MtpQ0RoundRecord cancelled = committed;
    cancelled.round_index = 0;
    cancelled.engine_committed_count = 0;
    cancelled.engine_terminal = true;
    cancelled.engine_cancelled = true;
    program.trace.append(cancelled);
    exporter->capture(program, slots);

    slots[0] = std::make_shared<Q0ExportTestRequest>(Q0ExportTestRequest{.id = 103});
    q36::MtpQ0RoundRecord aborted = committed;
    aborted.round_index = 0;
    aborted.engine_committed_count = 0;
    aborted.aborted = true;
    program.trace.append(aborted);
    exporter->capture(program, slots);
    exporter->finish();
    return result;
}

void test_q0_observer_engine_export() {
    const std::filesystem::path directory = std::filesystem::temp_directory_path();
    const std::string stem = "ninfer-q0-observer-" + std::to_string(::getpid());
    const std::string disabled_path = (directory / (stem + "-disabled.jsonl")).string();
    const std::string output_path = (directory / (stem + "-export.jsonl")).string();
    const std::string truncated_path = (directory / (stem + "-truncated.jsonl")).string();
    const std::string identity_only_path = (directory / (stem + "-identity-only.jsonl")).string();
    (void)::unlink(disabled_path.c_str());
    (void)::unlink(output_path.c_str());
    (void)::unlink(truncated_path.c_str());
    (void)::unlink(identity_only_path.c_str());

    const Q0ExportTestResult disabled = run_q0_export_control(false, disabled_path);
    const Q0ExportTestResult enabled = run_q0_export_control(true, output_path);
    expect(disabled.tokens == enabled.tokens &&
               disabled.committed_tokens == enabled.committed_tokens,
           "Q0 Engine export leaves synthetic committed tokens and counters unchanged");
    expect(::access(disabled_path.c_str(), F_OK) != 0,
           "disabled Q0 observer export creates no output file");

    struct stat output_status {};
    expect(::stat(output_path.c_str(), &output_status) == 0 &&
               (output_status.st_mode & 0777) == 0600,
           "Q0 observer output is owner-only");
    std::ifstream output_file(output_path);
    const std::string output((std::istreambuf_iterator<char>(output_file)),
                             std::istreambuf_iterator<char>());
    expect(output.find("\"generation_id\":\"101\"") != std::string::npos &&
               output.find("\"generation_id\":\"102\"") != std::string::npos &&
               output.find("\"generation_id\":\"103\"") != std::string::npos &&
               output.find("\"engine_cancelled\":true") != std::string::npos &&
               output.find("\"aborted\":true") != std::string::npos &&
               output.find("\"type\":\"prompt_identity\",\"generation_id\":\"101\","
                           "\"token_count\":2") != std::string::npos &&
               output.find("\"type\":\"committed_identity\",\"generation_id\":\"101\","
                           "\"token_count\":2") != std::string::npos &&
               output.find("\"records\":3,\"truncated\":false") != std::string::npos,
           "Engine export binds token identities and Q0 rows across lane reuse");

    ninfer::runtime::Q0ObserverExport identity_only(identity_only_path, false);
    identity_only.append_prompt_identity(201, 1, "0123456789abcdef");
    identity_only.append_committed_identity(201, 0, "fedcba9876543210");
    identity_only.finish();
    struct stat identity_status {};
    expect(::stat(identity_only_path.c_str(), &identity_status) == 0 &&
               (identity_status.st_mode & 0777) == 0600,
           "observer-off identity sidecar is owner-only");
    std::ifstream identity_file(identity_only_path);
    const std::string identity_output((std::istreambuf_iterator<char>(identity_file)),
                                      std::istreambuf_iterator<char>());
    expect(identity_output.find("\"capture_rounds\":false") != std::string::npos &&
               identity_output.find("\"type\":\"prompt_identity\"") != std::string::npos &&
               identity_output.find("\"type\":\"committed_identity\"") != std::string::npos &&
               identity_output.find("\"records\":0,\"truncated\":false") !=
                   std::string::npos,
           "observer-off sidecar records identities without Q0 rounds");

    Q0ExportTestProgram program;
    std::array<std::shared_ptr<Q0ExportTestRequest>, 1> slots{
        std::make_shared<Q0ExportTestRequest>(Q0ExportTestRequest{.id = 104})};
    ninfer::runtime::Q0ObserverExport exporter(truncated_path);
    q36::MtpQ0RoundRecord record;
    record.lane = 0;
    record.current_extent = 1;
    for (std::size_t i = 0; i < q36::kMtpQ0ObserverMaximumTraceRounds; ++i) {
        program.trace.append(record);
    }
    exporter.capture(program, slots);
    program.trace.append(record);
    exporter.capture(program, slots);
    exporter.finish();
    std::ifstream truncated_file(truncated_path);
    const std::string truncated((std::istreambuf_iterator<char>(truncated_file)),
                                std::istreambuf_iterator<char>());
    expect(truncated.find("\"records\":4096,\"truncated\":true") != std::string::npos,
           "Q0 Engine export caps rows at 4096 and reports source truncation");

    (void)::unlink(output_path.c_str());
    (void)::unlink(truncated_path.c_str());
    (void)::unlink(identity_only_path.c_str());
}

} // namespace

int main() {
    test_decoder_layout();
    test_round_layout();
    test_mtp_alignment();
    test_vision_control();
    test_prefix_identity();
    test_rebuild_work_prompt_frontier_boundary();
    test_mtp_q0_observer_trace();
    test_token_id_identity_hash();
    test_q0_observer_engine_export();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.6 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.6 runtime mechanism checks passed\n";
    return 0;
}

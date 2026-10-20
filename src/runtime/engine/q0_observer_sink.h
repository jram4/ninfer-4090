#pragma once

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::runtime {

class Q0ObserverSink {
public:
    static constexpr std::size_t kMaximumRecords = 4096;
    static constexpr std::size_t kMaximumIdentityRecords = 8192;

    explicit Q0ObserverSink(const std::string& path, bool capture_rounds) :
        capture_rounds_(capture_rounds) {
        if (path.empty()) { throw std::invalid_argument("Q0 observer output path is empty"); }
        std::string metadata =
            "{\"type\":\"metadata\",\"schema_version\":2,\"maximum_records\":4096,"
            "\"maximum_identity_records\":8192,\"capture_rounds\":";
        metadata.append(capture_rounds_ ? "true" : "false");
        metadata.append(",\"token_id_encoding\":\"int32_le\","
                        "\"generation_id_scope\":\"engine_lifetime\","
                        "\"generation_id_source\":\"monotonic_request_id\","
                        "\"generation_id_format\":\"decimal_string\"}");
        int flags = O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC;
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#endif
        fd_ = ::open(path.c_str(), flags, S_IRUSR | S_IWUSR);
        if (fd_ < 0) {
            throw std::system_error(errno, std::generic_category(), "create Q0 observer output");
        }

        struct stat status {};
        int security_error = 0;
        if (::fstat(fd_, &status) != 0) {
            security_error = errno;
        } else if (status.st_uid != ::geteuid()) {
            security_error = EPERM;
        } else if (::fchmod(fd_, S_IRUSR | S_IWUSR) != 0) {
            security_error = errno;
        }
        if (security_error != 0) {
            ::close(fd_);
            fd_ = -1;
            (void)::unlink(path.c_str());
            throw std::system_error(security_error, std::generic_category(),
                                    "secure Q0 observer output");
        }
        if (!write_line(metadata)) {
            const int error = errno == 0 ? EIO : errno;
            ::close(fd_);
            fd_ = -1;
            (void)::unlink(path.c_str());
            throw std::system_error(error, std::generic_category(), "write Q0 observer metadata");
        }
    }

    ~Q0ObserverSink() { finish(); }

    Q0ObserverSink(const Q0ObserverSink&) = delete;
    Q0ObserverSink& operator=(const Q0ObserverSink&) = delete;

    template <class TraceRecord>
    void append(std::uint64_t generation_id, const TraceRecord& record) noexcept {
        if (!capture_rounds_ || fd_ < 0 || finished_ || write_failed_) { return; }
        if (record_count_ >= kMaximumRecords) {
            truncated_ = true;
            return;
        }
        try {
            std::string line;
            line.reserve(512);
            line.append("{\"type\":\"round\",\"generation_id\":\"");
            append_number(line, generation_id);
            line.append("\",\"lane\":");
            append_number(line, record.lane);
            line.append(",\"round_index\":\"");
            append_number(line, record.round_index);
            line.append("\",\"base_frontier\":");
            append_number(line, record.base_frontier);
            line.append(",\"current_extent\":");
            append_number(line, record.current_extent);
            line.append(",\"verifier_licensed_count\":");
            append_number(line, record.verifier_licensed_count);
            line.append(",\"verifier_accepted_drafts\":");
            append_number(line, record.verifier_accepted_drafts);
            line.append(",\"engine_committed_count\":");
            append_number(line, record.engine_committed_count);
            line.append(",\"initial_prefill_one_hot\":");
            append_bool(line, record.initial_prefill_one_hot);
            line.append(",\"fallback\":");
            append_bool(line, record.fallback);
            line.append(",\"engine_terminal\":");
            append_bool(line, record.engine_terminal);
            line.append(",\"engine_cancelled\":");
            append_bool(line, record.engine_cancelled);
            line.append(",\"aborted\":");
            append_bool(line, record.aborted);
            line.append(",\"depths\":[");
            const std::size_t depth_count =
                std::min<std::size_t>(record.current_extent, record.depths.size());
            for (std::size_t depth = 0; depth < depth_count; ++depth) {
                if (depth != 0) { line.push_back(','); }
                line.append("{\"overlap\":");
                append_number(line, record.depths[depth].overlap);
                line.append(",\"support_coverage\":");
                append_number(line, record.depths[depth].support_coverage);
                line.push_back('}');
            }
            line.append("]}");
            if (!write_line(line)) {
                write_failed_ = true;
                incomplete_   = true;
                return;
            }
            ++record_count_;
        } catch (...) {
            incomplete_   = true;
            write_failed_ = true;
        }
    }

    void append_prompt_identity(std::uint64_t generation_id, std::size_t token_count,
                                const std::string& token_ids_sha256) noexcept {
        append_identity("prompt_identity", generation_id, token_count, token_ids_sha256);
    }

    void append_committed_identity(std::uint64_t generation_id, std::size_t token_count,
                                   const std::string& token_ids_sha256) noexcept {
        append_identity("committed_identity", generation_id, token_count, token_ids_sha256);
    }

    void mark_truncated() noexcept { truncated_ = true; }
    void mark_incomplete() noexcept { incomplete_ = true; }

    void finish() noexcept {
        if (finished_) { return; }
        finished_ = true;
        if (fd_ < 0) { return; }
        if (!write_failed_) {
            try {
                std::string line;
                line.reserve(128);
                line.append("{\"type\":\"summary\",\"records\":");
                append_number(line, record_count_);
                line.append(",\"truncated\":");
                append_bool(line, truncated_ || identity_truncated_);
                line.append(",\"identity_records\":");
                append_number(line, identity_record_count_);
                line.append(",\"identity_truncated\":");
                append_bool(line, identity_truncated_);
                line.append(",\"incomplete\":");
                append_bool(line, incomplete_);
                line.push_back('}');
                if (!write_line(line)) { write_failed_ = true; }
            } catch (...) { write_failed_ = true; }
        }
        (void)::close(fd_);
        fd_ = -1;
    }

private:
    void append_identity(const char* type, std::uint64_t generation_id,
                         std::size_t token_count, const std::string& token_ids_sha256) noexcept {
        if (fd_ < 0 || finished_ || write_failed_) { return; }
        if (identity_record_count_ >= kMaximumIdentityRecords) {
            identity_truncated_ = true;
            return;
        }
        try {
            std::string line;
            line.reserve(256);
            line.append("{\"type\":\"");
            line.append(type);
            line.append("\",\"generation_id\":\"");
            append_number(line, generation_id);
            line.append("\",\"token_count\":");
            append_number(line, token_count);
            line.append(",\"token_ids_sha256\":\"");
            line.append(token_ids_sha256);
            line.append("\"}");
            if (!write_line(line)) {
                write_failed_ = true;
                incomplete_ = true;
                return;
            }
            ++identity_record_count_;
        } catch (...) {
            incomplete_ = true;
            write_failed_ = true;
        }
    }

    template <class Number>
    static void append_number(std::string& out, Number value) {
        if constexpr (std::is_floating_point_v<Number>) {
            if (!std::isfinite(value)) {
                out.append("null");
                return;
            }
        }
        char buffer[64];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
        if (result.ec != std::errc{}) {
            throw std::runtime_error("Q0 observer number formatting failed");
        }
        out.append(buffer, result.ptr);
    }

    static void append_bool(std::string& out, bool value) {
        out.append(value ? "true" : "false");
    }

    bool write_line(const std::string& line) noexcept {
        std::size_t offset = 0;
        while (offset < line.size()) {
            const ssize_t count = ::write(fd_, line.data() + offset, line.size() - offset);
            if (count < 0) {
                if (errno == EINTR) { continue; }
                return false;
            }
            if (count == 0) {
                errno = EIO;
                return false;
            }
            offset += static_cast<std::size_t>(count);
        }
        constexpr char newline = '\n';
        for (;;) {
            const ssize_t count = ::write(fd_, &newline, 1);
            if (count == 1) { return true; }
            if (count < 0 && errno == EINTR) { continue; }
            if (count == 0) { errno = EIO; }
            return false;
        }
    }

    int fd_ = -1;
    std::size_t record_count_ = 0;
    bool truncated_ = false;
    bool identity_truncated_ = false;
    bool incomplete_ = false;
    bool write_failed_ = false;
    bool finished_ = false;
    bool capture_rounds_ = false;
    std::size_t identity_record_count_ = 0;
};

class Q0ObserverExport {
public:
    Q0ObserverExport() = default;

    explicit Q0ObserverExport(const std::string& path, bool capture_rounds = true)
        : sink_(std::make_unique<Q0ObserverSink>(path, capture_rounds)) {}

    Q0ObserverExport(Q0ObserverExport&&) noexcept = default;
    Q0ObserverExport& operator=(Q0ObserverExport&&) noexcept = default;
    Q0ObserverExport(const Q0ObserverExport&) = delete;
    Q0ObserverExport& operator=(const Q0ObserverExport&) = delete;

    template <class Program, class Slots>
    void capture(Program& program, const Slots& slots) noexcept {
        if (!sink_ || collection_failed_) { return; }
        try {
            auto trace = program.mtp_q0_observer_trace(next_round_);
            for (const auto& record : trace.rounds) {
                if (record.lane >= slots.size() || slots[record.lane] == nullptr) {
                    sink_->mark_incomplete();
                    continue;
                }
                sink_->append(slots[record.lane]->id, record);
            }
            next_round_ += trace.rounds.size();
            if (trace.truncated) { sink_->mark_truncated(); }
        } catch (...) {
            collection_failed_ = true;
            sink_->mark_incomplete();
        }
    }

    void finish() noexcept {
        if (sink_) { sink_->finish(); }
    }

    [[nodiscard]] bool identity_enabled() const noexcept { return sink_ != nullptr; }

    void append_prompt_identity(std::uint64_t generation_id, std::size_t token_count,
                                const std::string& token_ids_sha256) noexcept {
        if (sink_) { sink_->append_prompt_identity(generation_id, token_count, token_ids_sha256); }
    }

    void append_committed_identity(std::uint64_t generation_id, std::size_t token_count,
                                   const std::string& token_ids_sha256) noexcept {
        if (sink_) {
            sink_->append_committed_identity(generation_id, token_count, token_ids_sha256);
        }
    }

    void mark_incomplete() noexcept {
        if (sink_) { sink_->mark_incomplete(); }
    }

private:
    std::unique_ptr<Q0ObserverSink> sink_;
    std::size_t next_round_ = 0;
    bool collection_failed_ = false;
};

} // namespace ninfer::runtime

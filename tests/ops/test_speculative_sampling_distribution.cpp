#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<double> filtered_distribution(const std::vector<double>& logits, double temperature,
                                          std::size_t top_k, double top_p) {
    require(temperature > 0.0, "temperature must be positive");
    std::vector<std::size_t> order(logits.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;
    });
    order.resize(std::min(top_k, order.size()));

    const double maximum = logits[order.front()] / temperature;
    std::vector<double> weights(order.size());
    double total = 0.0;
    for (std::size_t i = 0; i < order.size(); ++i) {
        weights[i] = std::exp(logits[order[i]] / temperature - maximum);
        total += weights[i];
    }
    const double cutoff = std::clamp(top_p, 0.0, 1.0) * total;
    double cumulative = 0.0;
    std::size_t support = 0;
    do {
        cumulative += weights[support++];
    } while (support < weights.size() && cumulative < cutoff);

    std::vector<double> result(logits.size(), 0.0);
    for (std::size_t i = 0; i < support; ++i) result[order[i]] = weights[i] / cumulative;
    return result;
}

std::vector<double> rejection_sampling_output(const std::vector<double>& p,
                                              const std::vector<double>& q) {
    require(p.size() == q.size(), "p and q domains differ");
    double p_sum = 0.0, q_sum = 0.0, overlap = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        p_sum += p[i];
        q_sum += q[i];
        overlap += std::min(p[i], q[i]);
    }
    require(std::abs(p_sum - 1.0) < 1e-12, "p is not normalized");
    require(std::abs(q_sum - 1.0) < 1e-12, "q is not normalized");

    const double rejection_mass = 1.0 - overlap;
    double residual_total = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i)
        residual_total += std::max(p[i] - q[i], 0.0);
    require(std::abs(residual_total - rejection_mass) < 1e-12,
            "residual mass does not match rejection probability");

    std::vector<double> output(p.size(), 0.0);
    for (std::size_t i = 0; i < p.size(); ++i) {
        output[i] = std::min(p[i], q[i]);
        if (rejection_mass > 1e-15) {
            output[i] += rejection_mass * std::max(p[i] - q[i], 0.0) / residual_total;
        }
    }
    return output;
}

std::size_t greedy_argmax(const std::vector<double>& logits) {
    require(!logits.empty(), "greedy domain is empty");
    std::size_t best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

void check_distribution(const std::vector<double>& p, const std::vector<double>& q,
                        const std::string& label) {
    const auto output = rejection_sampling_output(p, q);
    for (std::size_t i = 0; i < p.size(); ++i) {
        require(std::abs(output[i] - p[i]) < 1e-12,
                label + ": corrected output differs from target at token " + std::to_string(i));
    }
}

} // namespace

int main() {
    try {
        // Different temperatures and top-k/top-p filters create partially overlapping supports.
        const auto p = filtered_distribution({2.2, 1.5, 0.7, 0.2, -1.0}, 0.9, 4, 0.82);
        const auto q = filtered_distribution({0.4, 2.4, 1.2, -0.3, 0.8}, 1.1, 3, 0.91);
        check_distribution(p, q, "partially overlapping truncated supports");

        // q may put all its mass on a token outside p's support; residual correction must still
        // recover p, including tokens absent from q's sparse candidate list.
        const std::vector<double> disjoint_p{0.0, 0.0, 0.65, 0.35};
        const std::vector<double> disjoint_q{0.4, 0.6, 0.0, 0.0};
        check_distribution(disjoint_p, disjoint_q, "disjoint sparse supports");

        // When the proposal equals the target, overlap is one and rejection probability is zero.
        check_distribution(p, p, "identical proposal and target");

        // The greedy route retains deterministic lower-id tie breaking.
        require(greedy_argmax({1.0, 4.0, 4.0, 0.5}) == 1,
                "greedy tie did not select the lowest token id");

        std::cout << "exact speculative sampling distribution checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exact speculative sampling distribution check failed: " << error.what()
                  << '\n';
        return 1;
    }
}

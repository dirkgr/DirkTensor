#include "CrossEntropyLoss.h"

#include <cmath>

#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>

ce_loss_result ce_loss(
    const xt::xtensor<float, 3>& logits,
    const xt::xtensor<float, 2>& batch,
    const uint32_t ignore_index
) {
    // This takes the batch, not "labels" as a left-shifted version of batch, because we can avoid some
    // xtensor tensor slicing slowness by doing it this way.

    const size_t batch_size = logits.shape(0);
    const size_t seq_len = logits.shape(1);

    // Using the log sum exp trick to keep values in a nice range

    const auto max_logits = xt::eval(xt::amax(logits, {2}));
    const auto logits_minus_max = logits - xt::view(max_logits, xt::all(), xt::all(), xt::newaxis());
    const auto exp = xt::exp(logits_minus_max);
    const auto sum_exp = xt::eval(xt::sum(exp, {2}));
    const auto probs = exp / xt::view(sum_exp, xt::all(), xt::all(), xt::newaxis());

    float result = 0.0f;
    unsigned int ignored = 0;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len - 1; ++s) {
            if (batch(b, s + 1) == ignore_index) {
                ignored++;
            } else {
                result -= logits_minus_max(b, s, batch(b, s + 1));
                result += std::log(sum_exp(b, s));
            }
        }
    }
    return {
        result / (batch_size * (seq_len - 1) - ignored),
        probs
    };
}
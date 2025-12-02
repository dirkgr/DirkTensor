#pragma once

#include <xtensor/containers/xtensor.hpp>

struct ce_loss_result {
    float loss;
    xt::xtensor<float, 3> probs;
};

ce_loss_result ce_loss(
    const xt::xtensor<float, 3>& logits,
    const xt::xtensor<float, 2>& batch,
    uint32_t ignore_index);

#pragma once

#include <xtensor/containers/xtensor.hpp>

float cross_entropy_loss(
    const xt::xtensor<float, 3>& logits,
    const xt::xtensor<float, 2>& batch,
    uint32_t ignore_index
);
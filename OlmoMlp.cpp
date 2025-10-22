#include "OlmoMlp.h"

#include <cmath>
#include <format>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>

OlmoMlp::OlmoMlp(const std::string& folder, const unsigned int index) {
    m_upProjection =
        xt::load_npy<float>(
            std::format("{}/model.layers.{}.mlp.up_proj.weight.npy", folder, index));
    m_gateProjection =
        xt::load_npy<float>(
            std::format("{}/model.layers.{}.mlp.gate_proj.weight.npy", folder, index));
    m_downProjection =
        xt::load_npy<float>(
            std::format("{}/model.layers.{}.mlp.down_proj.weight.npy", folder, index));
}

xt::xtensor<float, 1> OlmoMlp::forward(const xt::xtensor<float, 1>& input) {
    const auto projected = xt::linalg::dot(m_upProjection, input);
    const auto gate = xt::linalg::dot(m_gateProjection, input);

    // SiLU activation: x * sigmoid(x) where sigmoid(x) = 1 / (1 + exp(-x))
    const auto silu = gate / (1.0f + xt::exp(-gate));
    const auto result = projected * silu;

    return xt::linalg::dot(m_downProjection, result);
}

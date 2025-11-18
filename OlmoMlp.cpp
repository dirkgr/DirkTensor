#include "OlmoMlp.h"

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

xt::xtensor<float, 3> OlmoMlp::forward(const xt::xtensor<float, 3>& input) {
    const auto projected = xt::linalg::tensordot(input, m_upProjection, {2}, {1});
    const auto gate = xt::linalg::tensordot(input, m_gateProjection, {2}, {1});
    const auto silu = gate / (1.0f + xt::exp(-gate));
    const auto result = xt::linalg::tensordot(projected * silu, m_downProjection, {2}, {1});

    return result;
}

#include "OlmoMlp.h"

#include <format>
#include <xtensor/io/xnpy.hpp>

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
    const auto sigmoid = 1.0 / (1.0 + xt::exp(-gate));
    const auto silu = gate * sigmoid;
    const auto result = projected * silu;
    return xt::linalg::dot(m_downProjection, result);
}

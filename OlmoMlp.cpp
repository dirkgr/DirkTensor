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
    const auto projected = xt::eval(xt::linalg::tensordot(input, m_upProjection, {2}, {1}));

    auto x = xt::eval(xt::linalg::tensordot(input, m_gateProjection, {2}, {1}));
    assert(x.size() == projected.size());
    for (size_t i = 0; i < x.size(); i++) {
        float v = x.flat(i);
        v = v / (1.0f + std::exp(-v));
        v *= projected.flat(i);
        x.flat(i) = v;
    }

    const auto result = xt::linalg::tensordot(x, m_downProjection, {2}, {1});

    return result;
}

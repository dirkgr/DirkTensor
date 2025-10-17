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

    // Manual SiLU implementation: result = projected * gate * sigmoid(gate)
    // This avoids xtensor expression template overhead for element-wise operations
    const size_t size = projected.size();
    xt::xtensor<float, 1> result = xt::empty<float>({size});

    const float* projected_ptr = projected.data();
    const float* gate_ptr = gate.data();
    float* result_ptr = result.data();

    for (size_t i = 0; i < size; ++i) {
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        // Combined: projected[i] * gate[i] * sigmoid(gate[i])
        const float g = gate_ptr[i];
        const float sigmoid = 1.0f / (1.0f + std::exp(-g));
        result_ptr[i] = projected_ptr[i] * g * sigmoid;
    }

    return xt::linalg::dot(m_downProjection, result);
}

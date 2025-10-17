#include "OlmoMlp.h"

#include <format>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/core/xnoalias.hpp>
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

    // Pre-allocate buffers based on intermediate dimension
    const size_t intermediate_dim = m_upProjection.shape()[0];
    m_projectedBuffer = xt::empty<float>({intermediate_dim});
    m_gateBuffer = xt::empty<float>({intermediate_dim});
    m_sigmoidBuffer = xt::empty<float>({intermediate_dim});
    m_siluBuffer = xt::empty<float>({intermediate_dim});
    m_resultBuffer = xt::empty<float>({intermediate_dim});
}

xt::xtensor<float, 1> OlmoMlp::forward(const xt::xtensor<float, 1>& input) {
    // Use pre-allocated buffers with noalias for in-place operations
    xt::noalias(m_projectedBuffer) = xt::linalg::dot(m_upProjection, input);
    xt::noalias(m_gateBuffer) = xt::linalg::dot(m_gateProjection, input);
    xt::noalias(m_sigmoidBuffer) = 1.0 / (1.0 + xt::exp(-m_gateBuffer));
    xt::noalias(m_siluBuffer) = m_gateBuffer * m_sigmoidBuffer;
    xt::noalias(m_resultBuffer) = m_projectedBuffer * m_siluBuffer;
    return xt::linalg::dot(m_downProjection, m_resultBuffer);
}

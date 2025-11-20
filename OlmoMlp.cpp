#include "OlmoMlp.h"

#include <format>
#include <iostream>
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
    // Optimize matrix multiplications by reshaping to 2D, using GEMM, then reshaping back
    // This approach is more efficient than tensordot for batched operations

    const size_t batch_size = input.shape(0);
    const size_t seq_len = input.shape(1);
    const size_t d_model = input.shape(2);
    const size_t hidden_size = m_upProjection.shape(0);  // 8192

    // Reshape input from [batch, seq, d_model] to [batch*seq, d_model]
    auto input_2d = xt::reshape_view(input, {batch_size * seq_len, d_model});

    // Perform matrix multiplications using direct BLAS GEMM via dot
    // Weights are stored as [hidden_size, d_model], so we need to transpose them
    // input_2d is [batch*seq, d_model], weights transposed become [d_model, hidden_size]
    auto projected_2d = xt::linalg::dot(input_2d, xt::transpose(m_upProjection));
    auto gate_2d = xt::linalg::dot(input_2d, xt::transpose(m_gateProjection));

    // Apply SiLU activation: gate * sigmoid(gate) * projected
    const auto silu = gate_2d / (1.0f + xt::exp(-gate_2d));
    const auto activated_2d = silu * projected_2d;

    // Final projection back to d_model
    // m_downProjection is [d_model, hidden_size], so we need to transpose it
    // activated_2d is [batch*seq, hidden_size], transposed weight is [hidden_size, d_model]
    auto result_2d = xt::linalg::dot(activated_2d, xt::transpose(m_downProjection));

    // Reshape back to 3D [batch, seq, d_model]
    auto result = xt::reshape_view(result_2d, {batch_size, seq_len, d_model});

    // Return a copy to ensure the result owns its memory
    return xt::eval(result);
}

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
}

xt::xtensor<float, 3> OlmoMlp::forward(const xt::xtensor<float, 3>& input) {
    auto output = xt::empty_like(input);

    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, input.shape(0), 0, input.shape(1)),
        [&](const tbb::blocked_range2d<size_t>& r) {
            for (size_t b = r.rows().begin(); b != r.rows().end(); ++b) {
                for (size_t i = r.cols().begin(); i != r.cols().end(); ++i) {
                    const auto input_slice = xt::view(input, b, i);
                    const auto projected = xt::linalg::dot(m_upProjection, input_slice);
                    const auto gate = xt::linalg::dot(m_gateProjection, input_slice);

                    // SiLU activation: x * sigmoid(x) where sigmoid(x) = 1 / (1 + exp(-x))
                    const auto silu = gate / (1.0f + xt::exp(-gate));
                    const auto result = projected * silu;

                    xt::noalias(xt::view(output, b, i)) = xt::linalg::dot(m_downProjection, result);
                }
            }
        },
        tbb::static_partitioner()
    );

    return output;
}

#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor-blas/xlinalg.hpp>

class OlmoMlp {
public:
    OlmoMlp(const std::string& folder, unsigned int index);

    xt::xtensor<float, 1> forward(const auto& input) {
        const auto projected = xt::linalg::dot(input, m_upProjection);
        const auto gate = xt::linalg::dot(input, m_gateProjection);
        const auto sigmoid = 1.0 / (1.0 + xt::exp(-gate));
        const auto silu = gate * sigmoid;
        const auto result = projected * silu;
        return xt::linalg::dot(result, m_downProjection);
    }

private:
    xt::xtensor<float, 2> m_upProjection;
    xt::xtensor<float, 2> m_gateProjection;
    xt::xtensor<float, 2> m_downProjection;
};

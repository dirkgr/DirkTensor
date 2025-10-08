#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor-blas/xlinalg.hpp>

class OlmoMlp {
public:
    OlmoMlp(const std::string& folder, unsigned int index);

    xt::xtensor<float, 1> forward(const xt::xtensor<float, 1>& input);

private:
    xt::xtensor<float, 2> m_upProjection;
    xt::xtensor<float, 2> m_gateProjection;
    xt::xtensor<float, 2> m_downProjection;
};

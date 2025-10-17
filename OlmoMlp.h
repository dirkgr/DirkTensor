#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>

class OlmoMlp {
public:
    OlmoMlp(const std::string& folder, unsigned int index);

    xt::xtensor<float, 1> forward(const xt::xtensor<float, 1>& input);

private:
    xt::xtensor<float, 2> m_upProjection;
    xt::xtensor<float, 2> m_gateProjection;
    xt::xtensor<float, 2> m_downProjection;

    // Pre-allocated buffers for forward pass
    mutable xt::xtensor<float, 1> m_projectedBuffer;
    mutable xt::xtensor<float, 1> m_gateBuffer;
    mutable xt::xtensor<float, 1> m_sigmoidBuffer;
    mutable xt::xtensor<float, 1> m_siluBuffer;
    mutable xt::xtensor<float, 1> m_resultBuffer;
};

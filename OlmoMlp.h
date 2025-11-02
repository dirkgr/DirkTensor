#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>

class OlmoMlp {
public:
    OlmoMlp(const std::string& folder, unsigned int index);

    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);

private:
    xt::xtensor<float, 2> m_upProjection;
    xt::xtensor<float, 2> m_gateProjection;
    xt::xtensor<float, 2> m_downProjection;
};

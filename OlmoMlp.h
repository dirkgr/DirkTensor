#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include "param.h"


class OlmoMlp {
public:
    OlmoMlp(const std::string& folder, unsigned int index);

    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);

private:
    param<2> m_upProjection;
    param<2> m_gateProjection;
    param<2> m_downProjection;
};

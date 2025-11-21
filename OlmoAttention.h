#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include "RMSNorm.h"

class OlmoAttention {
public:
    OlmoAttention(const std::string& folder, unsigned int index);
    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);

private:
    // parameters
    xt::xtensor<float, 2> m_qProj;
    RMSNorm m_qNorm;
    xt::xtensor<float, 2> m_kProj;
    RMSNorm m_kNorm;
    xt::xtensor<float, 2> m_vProj;
    xt::xtensor<float, 2> m_oProj;

    static xt::xtensor<float, 4> apply_rope(const xt::xtensor<float, 4>& input);
};

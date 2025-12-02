#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include "RMSNorm.h"
#include "param.h"


class OlmoAttention {
public:
    OlmoAttention(const std::string& folder, unsigned int index);
    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);

private:
    // parameters
    param<2> m_qProj;
    RMSNorm m_qNorm;
    param<2> m_kProj;
    RMSNorm m_kNorm;
    param<2> m_vProj;
    param<2> m_oProj;

    static xt::xtensor<float, 4> apply_rope(const xt::xtensor<float, 4>& input);
};

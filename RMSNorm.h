#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include "param.h"


class RMSNorm {
public:
    static constexpr float eps = 1e-06;

    explicit RMSNorm(const std::string& filename);

    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);
    xt::xtensor<float, 3> backward(const xt::xtensor<float, 3>& d_output);
    void step(float lr);
    void zero_grad();

private:
    param<1> m_weight;

    xt::xtensor<float, 2> m_act_rms;
    xt::xtensor<float, 3> m_output;
};

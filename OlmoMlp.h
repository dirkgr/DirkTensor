#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include "param.h"


class OlmoMlp {
public:
    OlmoMlp(const std::string& folder, unsigned int index);

    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);
    xt::xtensor<float, 3> backward(const xt::xtensor<float, 3>& d_output);
    void step(float lr);
    void zero_grad();

private:
    param<2> m_upProjection;
    param<2> m_gateProjection;
    param<2> m_downProjection;

    xt::xtensor<float, 2> m_act_input_2d;
    xt::xtensor<float, 2> m_act_activated_2d;
    xt::xtensor<float, 2> m_act_up;
    xt::xtensor<float, 2> m_act_gate;
};

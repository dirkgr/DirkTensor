#pragma once
#include <string>

#include "param.h"


class OlmoLmHead {
public:
    explicit OlmoLmHead(const std::string& filename);

    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);
    xt::xtensor<float, 3> backward(const xt::xtensor<float, 3>& grad);
    void step(float lr);
    void zero_grad();

private:
    param<2> m_lmHead;

    xt::xtensor<float, 3> m_activationsBefore;
};

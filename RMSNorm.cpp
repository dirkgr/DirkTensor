#include "RMSNorm.h"

#include <cmath>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>

#include "xtutil.h"

RMSNorm::RMSNorm(const std::string& filename) {
    m_weight.w = xt::load_npy<float>(filename);
}

xt::xtensor<float, 3> RMSNorm::forward(const xt::xtensor<float, 3>& input) {
    const auto squares = xt::pow(input, 2);
    const auto sum_squares = xt::sum(squares, {2});
    m_act_rms = xt::sqrt(sum_squares / input.shape(2) + eps);
    const auto act_normed = input / xt::view(m_act_rms, xt::all(), xt::all(), xt::newaxis());
    m_output = act_normed * m_weight.w;
    return m_output;
}

xt::xtensor<float, 3> RMSNorm::backward(const xt::xtensor<float, 3>& d_output) {
    const auto c = xt::eval(xt::mean(d_output * m_output));

    if (m_weight.grad.size() <= 0)
        m_weight.grad = xt::zeros_like(m_weight.w);
    const auto act_normed = xt::eval(m_output / m_weight.w); // recomputed from the forward
    m_weight.grad += xt::sum(d_output * act_normed, {0, 1});

    return (m_weight.w * d_output - act_normed * c) / xt::view(m_act_rms, xt::all(), xt::all(), xt::newaxis());
}
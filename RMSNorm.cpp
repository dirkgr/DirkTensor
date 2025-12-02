#include "RMSNorm.h"

#include <cmath>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>

#include "xtutil.h"

RMSNorm::RMSNorm(const std::string& filename) {
    m_weight.w = xt::load_npy<float>(filename);
}

xt::xtensor<float, 3> RMSNorm::forward(const xt::xtensor<float, 3>& input) const {
    const auto squares = xt::pow(input, 2);
    const auto sum_squares = xt::sum(squares, {2});
    const auto rms = xt::eval(xt::sqrt(sum_squares / input.shape(2) + eps));
    const auto normed = input / xt::view(rms, xt::all(), xt::all(), xt::newaxis());
    const auto scaled = normed * m_weight.w;
    return scaled;
}

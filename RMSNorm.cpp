#include "RMSNorm.h"

#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xnpy.hpp>

RMSNorm::RMSNorm(const std::string& filename) {
    m_weight = xt::load_npy<float>(filename);
}

xt::xtensor<float, 1> RMSNorm::forward(const xt::xtensor<float, 1>& input) {
    const auto rms = xt::sqrt(xt::mean(xt::square(input)) + eps);
    const auto x = xt::eval(input / rms);
    return x * m_weight;
}

#include "RMSNorm.h"

#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/core/xnoalias.hpp>

RMSNorm::RMSNorm(const std::string& filename) {
    m_weight = xt::load_npy<float>(filename);
    m_xBuffer = xt::empty<float>(m_weight.shape());
}

xt::xtensor<float, 1> RMSNorm::forward(const xt::xtensor<float, 1>& input) {
    const auto rms = xt::sqrt(xt::mean(xt::square(input)) + eps);
    xt::noalias(m_xBuffer) = input / rms;
    return m_xBuffer * m_weight;
}

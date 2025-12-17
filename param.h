#pragma once

#include <xtensor/containers/xtensor.hpp>

template<std::size_t N>
struct param {
    xt::xtensor<float, N> w;
    xt::xtensor<float, N> grad;

    param() = default;
    explicit param(const xt::xtensor<float, N>& w_) : w(w_) {}

    inline auto shape() const { return w.shape(); }
    inline auto shape(size_t i) const { return w.shape()[i]; }
};

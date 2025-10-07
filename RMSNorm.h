#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>

class RMSNorm {
public:
    static constexpr float eps = 1e-06;

    explicit RMSNorm(const std::string& filename);

    xt::xtensor<float, 1> forward(const auto& input) {
        const auto rms = xt::sqrt(xt::mean(xt::square(input)) + eps);
        const auto x = input / rms;
        return x * m_weight;
    }

private:
    xt::xtensor<float, 1> m_weight;
};

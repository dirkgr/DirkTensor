#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>

class RMSNorm {
public:
    static constexpr float eps = 1e-06;

    explicit RMSNorm(const std::string& filename);

    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);

private:
    xt::xtensor<float, 1> m_weight;
};

#include "RMSNorm.h"

#include <cmath>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xnpy.hpp>

RMSNorm::RMSNorm(const std::string& filename) {
    m_weight = xt::load_npy<float>(filename);
}

xt::xtensor<float, 1> RMSNorm::forward(const xt::xtensor<float, 1>& input) {
    // Manual implementation: compute RMS in one pass, then normalize and scale
    const size_t size = input.size();

    // First pass: compute sum of squares
    const float* input_ptr = input.data();
    double sum_squares = 0.0;
    for (size_t i = 0; i < size; ++i) {
        const float val = input_ptr[i];
        sum_squares += val * val;
    }

    // Compute RMS: sqrt(mean(square(x)) + eps)
    const float rms = std::sqrt(sum_squares / size + eps);

    // Second pass: normalize and scale by weight
    xt::xtensor<float, 1> output = xt::empty<float>({size});
    float* output_ptr = output.data();
    const float* weight_ptr = m_weight.data();

    for (size_t i = 0; i < size; ++i) {
        output_ptr[i] = (input_ptr[i] / rms) * weight_ptr[i];
    }

    return output;
}

#pragma once

#include <iostream>

#include <xtensor/core/xexpression.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor-blas/xlinalg.hpp>


// Print the shape of any xtensor expression
template<typename E>
void print_shape(const xt::xexpression<E>& expr) {
    const auto& derived = expr.derived_cast();
    const auto& shape = derived.shape();
    std::cout << "Shape: (";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
}

template<typename E>
xt::xtensor<E, 3> batched_projection(const xt::xtensor<E, 3>& input, xt::xtensor<E, 2> proj) {
    const auto b = input.shape(0);
    const auto s = input.shape(1);
    const auto d_model = input.shape(2);

    const auto proj_in = proj.shape(0);
    const auto proj_out = proj.shape(1);
    assert(proj_in == d_model);

    auto input_2d = xt::reshape_view(input, {b * s, d_model});
    auto output_2d = xt::linalg::dot(input_2d, proj);
    return xt::reshape_view(output_2d, {b, s, proj_out});
}

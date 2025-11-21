#pragma once

#include <iostream>

#include <xtensor/core/xexpression.hpp>
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

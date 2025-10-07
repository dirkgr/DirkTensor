#pragma once

#include <iostream>
#include <array>

// ostream operator for std::array (used by xtensor shapes)
template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr) {
    os << "(";
    for (size_t i = 0; i < N; ++i) {
        os << arr[i];
        if (i < N - 1) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}

// Print the shape of any xtensor expression
template<typename E>
void print_shape(const xt::xexpression<E>& expr) {
    const auto& derived = expr.derived_cast();
    std::cout << "Shape: " << derived.shape() << std::endl;
}
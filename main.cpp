#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <iterator>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "OlmoModel.h"

xt::xtensor<uint32_t, 1> read_tokens(std::istream& input) {
    std::vector<uint32_t> token_vec;
    uint32_t token;

    // Read tokens until EOF
    while (input.read(reinterpret_cast<char*>(&token), sizeof(token))) {
        token_vec.push_back(token);
    }

    // Convert to xtensor
    xt::xtensor<uint32_t, 1> result = xt::empty<uint32_t>({token_vec.size()});
    std::ranges::copy(token_vec, result.begin());
    return result;
}

int main(int argc, char* argv[]) {
    // Read tokens from binary stream
    xt::xtensor<uint32_t, 1> tokens;
    if (argc > 1) {
        // Read from file
        std::ifstream file(argv[1], std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + std::string(argv[1]));
        }
        tokens = read_tokens(file);
    } else {
        // Read from stdin
        tokens = read_tokens(std::cin);
    }

    OlmoModel model("models/OLMo-2-0425-1B");

    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << "token " << tokens(i) << std::endl;
        const xt::xtensor<float, 1> logits = -1 * model.forward(tokens(i)); // -1 to sort the highest logit first
        const auto tokens_in_order = xt::argsort(logits);

        std::cout << "Top 5 next tokens: ";
        for (size_t j = 0; j < 5; j++) {
            std::cout << tokens_in_order(j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
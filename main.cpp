#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "Detokenizer.h"
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
    Detokenizer detokenizer("models/OLMo-2-0425-1B/vocab.txt");

    unsigned int next_token_id = 0;

    for (size_t i = 0; i < 20; i++) {
        if (i < tokens.size())
            next_token_id = tokens(i);

        std::cout << i << ": token " << next_token_id << " (\"" << detokenizer.decode(next_token_id) << "\") ";
        const xt::xtensor<float, 1> logits = model.forward(next_token_id);

        // Find top 5 using partial sort instead of full argsort (much faster!)
        std::vector<size_t> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
            [&logits](size_t i1, size_t i2) { return logits(i1) > logits(i2); });

        std::cout << "Top 5 next tokens: ";
        for (size_t j = 0; j < 5; j++) {
            std::cout << indices[j] << " (\"" << detokenizer.decode(indices[j]) << "\") ";
        }
        std::cout << std::endl;

        next_token_id = indices[0];
    }

    return 0;
}
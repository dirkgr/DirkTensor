#include <algorithm>
#include <chrono>
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
    auto start_total = std::chrono::high_resolution_clock::now();

    // Read tokens from binary stream
    auto start_read_tokens = std::chrono::high_resolution_clock::now();
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
    auto end_read_tokens = std::chrono::high_resolution_clock::now();
    auto duration_read_tokens = std::chrono::duration_cast<std::chrono::milliseconds>(end_read_tokens - start_read_tokens).count();
    std::cerr << "Time to read tokens: " << duration_read_tokens << " ms" << std::endl;

    auto start_load_model = std::chrono::high_resolution_clock::now();
    OlmoModel model("models/OLMo-2-0425-1B");
    auto end_load_model = std::chrono::high_resolution_clock::now();
    auto duration_load_model = std::chrono::duration_cast<std::chrono::milliseconds>(end_load_model - start_load_model).count();
    std::cerr << "Time to load model: " << duration_load_model << " ms" << std::endl;

    auto start_load_detokenizer = std::chrono::high_resolution_clock::now();
    Detokenizer detokenizer("models/OLMo-2-0425-1B/vocab.txt");
    auto end_load_detokenizer = std::chrono::high_resolution_clock::now();
    auto duration_load_detokenizer = std::chrono::duration_cast<std::chrono::milliseconds>(end_load_detokenizer - start_load_detokenizer).count();
    std::cerr << "Time to load detokenizer: " << duration_load_detokenizer << " ms" << std::endl;

    unsigned int next_token_id = 0;

    auto start_inference = std::chrono::high_resolution_clock::now();
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
    auto end_inference = std::chrono::high_resolution_clock::now();
    auto duration_inference = std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count();
    std::cerr << "Time for inference (20 iterations): " << duration_inference << " ms" << std::endl;
    std::cerr << "Time per iteration: " << (duration_inference / 20.0) << " ms" << std::endl;

    auto end_total = std::chrono::high_resolution_clock::now();
    auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
    std::cerr << "Total time: " << duration_total << " ms" << std::endl;

    return 0;
}
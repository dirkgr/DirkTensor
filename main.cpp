#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
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

        // Find top 5 using min-heap (avoids allocating 100k+ indices array)
        using Pair = std::pair<float, size_t>;  // (logit_value, token_index)
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> min_heap;

        for (size_t idx = 0; idx < logits.size(); ++idx) {
            if (min_heap.size() < 5) {
                min_heap.push({logits(idx), idx});
            } else if (logits(idx) > min_heap.top().first) {
                min_heap.pop();
                min_heap.push({logits(idx), idx});
            }
        }

        // Extract top 5 in descending order
        std::vector<size_t> top5_indices;
        top5_indices.reserve(5);
        while (!min_heap.empty()) {
            top5_indices.push_back(min_heap.top().second);
            min_heap.pop();
        }
        std::reverse(top5_indices.begin(), top5_indices.end());

        std::cout << "Top 5 next tokens: ";
        for (size_t j = 0; j < 5; j++) {
            std::cout << top5_indices[j] << " (\"" << detokenizer.decode(top5_indices[j]) << "\") ";
        }
        std::cout << std::endl;

        next_token_id = top5_indices[0];
    }

    return 0;
}
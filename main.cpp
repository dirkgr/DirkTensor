#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>

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

float cross_entropy_loss(
    const xt::xtensor<float, 3>& logits,
    const xt::xtensor<float, 2>& batch,
    const uint32_t ignore_index
) {
    // This takes the batch, not "labels" as a left-shifted version of batch, because we can avoid some
    // xtensor tensor slicing slowness by doing it this way.

    const size_t batch_size = logits.shape(0);
    const size_t seq_len = logits.shape(1);

    const auto exp = xt::eval(xt::exp(logits));
    const auto exp_sums = xt::sum(exp, {2});

    float result = 0.0f;
    unsigned int ignored = 0;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len - 1; ++s) {
            if (batch(b, s + 1) == ignore_index)
                ignored++;
            else
                result -= std::log(exp(b, s, batch(b, s + 1)) / exp_sums(b, s));
        }
    }
    return result / (batch_size * (seq_len - 1) - ignored);
}

int main(int argc, char* argv[]) {
    const OlmoModel model("models/OLMo-2-0425-1B");
    const Detokenizer detokenizer("models/OLMo-2-0425-1B/vocab.txt");

    // Read files and determine actual max sequence length
    std::vector<xt::xtensor<uint32_t, 1>> all_tokens;
    size_t actual_max_len = 0;

    for(int i = 1; i < argc; ++i) {
        std::ifstream file(argv[i], std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot open file: " + std::string(argv[i]));
        auto tokens = read_tokens(file);
        all_tokens.push_back(tokens);
        actual_max_len = std::max(actual_max_len, tokens.size());
    }

    // Build batch with actual size needed
    auto batch = xt::empty<uint32_t>({static_cast<unsigned int>(argc - 1), static_cast<unsigned int>(actual_max_len)});

    // Fill batch with tokens and padding
    for(size_t i = 0; i < all_tokens.size(); ++i) {
        const auto& tokens = all_tokens[i];
        size_t len = std::min(tokens.size(), actual_max_len);
        xt::view(batch, i, xt::range(0, len)) = xt::view(tokens, xt::range(0, len));
        if(len < actual_max_len) {
            xt::view(batch, i, xt::range(len, actual_max_len)) = detokenizer.get_pad_token_id();
        }
    }

    // Forward pass
    xt::xtensor<float, 3> logits = model.forward(batch);

    // Compute cross entropy loss
    const float loss = cross_entropy_loss(logits, batch, detokenizer.get_pad_token_id());
    std::cout << loss << std::endl;

    return 0;
}

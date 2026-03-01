#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>

#include "CrossEntropyLoss.h"
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
    const std::string model_url = "https://dirktensor.mechanicaldirk.com/OLMo-2-0425-1B";
    OlmoModel model(model_url);
    const Detokenizer detokenizer(model_url + "/vocab.txt");

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
    auto loss_result = ce_loss(logits, batch, detokenizer.get_pad_token_id());
    std::cout << loss_result.loss << std::endl;

    // Compute gradient through CE loss.
    // We're repurposing loss_result.probs as grad because it saves us a big tensor copy.
    size_t batch_size = batch.shape(0) * batch.shape(1);
    for (size_t b = 0; b < batch.shape(0); ++b) {
        for (size_t s = 1; s < batch.shape(1); ++s) {
            const auto token_id = batch(b, s);
            if (token_id == detokenizer.get_pad_token_id()) {
                xt::view(loss_result.probs, b, s - 1, xt::all()) = 0;
                batch_size -= 1;
            } else {
                loss_result.probs(b, s - 1, token_id) -= 1.0f;
            }
        }
    }
    // last tokens never have grads, because they don't have targets
    xt::view(loss_result.probs, xt::all(), batch.shape(1) - 1, xt::all()) = 0;
    batch_size -= batch.shape(0);
    // Normalize by number of tokens
    loss_result.probs /= batch_size;

    model.backward(loss_result.probs);
    model.step(1e-4f);
    model.zero_grad();

    // Second forward pass to see if loss decreased
    logits = model.forward(batch);
    loss_result = ce_loss(logits, batch, detokenizer.get_pad_token_id());
    std::cout << loss_result.loss << std::endl;

    return 0;
}

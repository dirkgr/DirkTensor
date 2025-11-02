#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
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

int main(int argc, char* argv[]) {
    #ifdef HAVE_CBLAS
        std::cout << "HAVE_CBLAS is defined" << std::endl;
    #else
        std::cout << "HAVE_CBLAS is NOT defined - using fallback!" << std::endl;
    #endif

    OlmoModel model("models/OLMo-2-0425-1B");
    const Detokenizer detokenizer("models/OLMo-2-0425-1B/vocab.txt");

    // Build batch
    auto batch = xt::empty<uint32_t>({static_cast<unsigned int>(argc - 1), max_seq_len});
    size_t max_seq_len = 0;
    for(int i = 1; i < argc; ++i) {
        std::ifstream file(argv[i], std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot open file: " + std::string(argv[i]));
        auto tokens = read_tokens(file);
        max_seq_len = std::max(max_seq_len, tokens.size());
        if(tokens.size() > max_seq_len)
            tokens = xt::view(tokens, 0, max_seq_len);
        xt::view(batch, i - 1, xt::range(0, tokens.size())) = tokens;
        xt::view(batch, i - 1, xt::range(tokens.size(), max_seq_len)) = detokenizer.get_pad_token_id();
    }
    batch = xt::view(batch, xt::all(), xt::range(0, max_seq_len));

    auto logits = model.forward(batch);
    std::cout << logits << std::endl;

    return 0;
}
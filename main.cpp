#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <iterator>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>


xt::xtensor<uint32_t, 1> read_tokens(std::istream& input) {
    std::vector<uint32_t> token_vec;
    uint32_t token;

    // Read tokens until EOF
    while (input.read(reinterpret_cast<char*>(&token), sizeof(token))) {
        token_vec.push_back(token);
    }

    // Convert to xtensor
    xt::xtensor<uint32_t, 1> result = xt::empty<uint32_t>({token_vec.size()});
    std::copy(token_vec.begin(), token_vec.end(), result.begin());
    return result;
}

class OlmoBlock {
public:
    OlmoBlock(const std::string& folder) {

    }

private:

};


class OlmoModel {
public:
    static const size_t hidden_dim = 2048;
    static const size_t batch_size = 1;

    OlmoModel(const std::string& folder) {
        m_embeddings = xt::load_npy<float>(folder + "/model.embed_tokens.weight.npy");
        // Verify shape
        assert(m_embeddings.shape().size() == 2);
        assert(m_embeddings.shape()[1] == hidden_dim);

        m_lmHead = xt::load_npy<float>(folder + "/lm_head.weight.npy");
    }

    auto forward(const uint32_t token) {
        auto x = xt::view(m_embeddings, token, xt::all());

        return xt::linalg::dot(x, xt::transpose(m_lmHead));
    }

private:
    xt::xtensor<float, 2> m_embeddings;
    xt::xtensor<float, 2> m_lmHead;
};


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

    std::cout << "Read " << tokens.size() << " tokens" << std::endl;

    OlmoModel model("models/OLMo-2-0425-1B");
    const auto probs = model.forward(tokens(0));
    std::cout << probs << std::endl;

    return 0;
}
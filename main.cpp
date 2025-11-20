#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
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
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;

    auto total_start = Clock::now();

    // Model loading
    std::cerr << "Loading model..." << std::endl;
    auto model_start = Clock::now();
    OlmoModel model("models/OLMo-2-0425-1B");
    const Detokenizer detokenizer("models/OLMo-2-0425-1B/vocab.txt");
    Duration model_load_time = Clock::now() - model_start;
    std::cerr << "Model loaded in " << model_load_time.count() << " seconds" << std::endl;

    // File I/O and batch building
    auto io_start = Clock::now();

    // First pass: read files and determine actual max sequence length
    std::vector<xt::xtensor<uint32_t, 1>> all_tokens;
    size_t actual_max_len = 0;
    size_t total_tokens = 0;

    for(int i = 1; i < argc; ++i) {
        std::ifstream file(argv[i], std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot open file: " + std::string(argv[i]));
        auto tokens = read_tokens(file);
        all_tokens.push_back(tokens);
        total_tokens += tokens.size();
        actual_max_len = std::max(actual_max_len, tokens.size());
        std::cerr << "Read " << tokens.size() << " tokens from " << argv[i] << std::endl;
    }

    // Build batch with actual size needed (not 4096)
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

    Duration io_time = Clock::now() - io_start;
    std::cerr << "Max sequence length: " << actual_max_len << ", Total tokens: " << total_tokens << std::endl;

    // Warm-up run
    std::cerr << "\nRunning warm-up pass..." << std::endl;
    auto warmup_start = Clock::now();
    auto warmup_logits = model.forward(batch);
    Duration warmup_time = Clock::now() - warmup_start;
    std::cerr << "Warm-up completed in " << warmup_time.count() << " seconds" << std::endl;

    // Multiple timed forward passes
    const int num_runs = 5;
    std::cerr << "\nRunning " << num_runs << " timed forward passes..." << std::endl;
    std::vector<double> forward_times;
    xt::xtensor<float, 3> logits;

    for(int run = 0; run < num_runs; ++run) {
        auto forward_start = Clock::now();
        logits = model.forward(batch);
        Duration forward_time = Clock::now() - forward_start;
        forward_times.push_back(forward_time.count());
        std::cerr << "  Run " << (run + 1) << ": " << std::fixed << std::setprecision(4)
                  << forward_time.count() << " seconds" << std::endl;
    }

    // Calculate statistics
    double avg_forward_time = 0.0;
    double min_forward_time = forward_times[0];
    double max_forward_time = forward_times[0];

    for(double t : forward_times) {
        avg_forward_time += t;
        min_forward_time = std::min(min_forward_time, t);
        max_forward_time = std::max(max_forward_time, t);
    }
    avg_forward_time /= forward_times.size();

    double tokens_per_sec = total_tokens / avg_forward_time;

    Duration total_time = Clock::now() - total_start;

    // Print performance summary
    std::cerr << "\n" << std::string(60, '=') << std::endl;
    std::cerr << "C++ PERFORMANCE SUMMARY" << std::endl;
    std::cerr << std::string(60, '=') << std::endl;
    std::cerr << "Model loading:     " << std::fixed << std::setprecision(3)
              << model_load_time.count() << " seconds" << std::endl;
    std::cerr << "File I/O & batch:  " << std::fixed << std::setprecision(4)
              << io_time.count() << " seconds" << std::endl;
    std::cerr << "\nForward pass times (" << num_runs << " runs):" << std::endl;
    std::cerr << "  Average: " << std::fixed << std::setprecision(4)
              << avg_forward_time << " seconds" << std::endl;
    std::cerr << "  Min:     " << min_forward_time << " seconds" << std::endl;
    std::cerr << "  Max:     " << max_forward_time << " seconds" << std::endl;
    std::cerr << "\nThroughput: " << std::fixed << std::setprecision(1)
              << tokens_per_sec << " tokens/second" << std::endl;
    std::cerr << "Total time: " << std::fixed << std::setprecision(3)
              << total_time.count() << " seconds" << std::endl;
    std::cerr << std::string(60, '=') << "\n" << std::endl;

    // Print logits (original output)
    std::cout << logits << std::endl;

    return 0;
}
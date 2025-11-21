#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/views/xview.hpp>
#include "OlmoModel.h"

// Global model instance (loaded once for all tests)
static OlmoModel* g_model = nullptr;

class OlmoModelTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Load model once for all tests
        g_model = new OlmoModel("models/OLMo-2-0425-1B");
    }

    static void TearDownTestSuite() {
        delete g_model;
        g_model = nullptr;
    }

    std::vector<uint32_t> get_top5_predictions(const std::vector<uint32_t>& tokens) {
        // Create batch with single sequence
        xt::xtensor<uint32_t, 2> batch = xt::empty<uint32_t>({1u, static_cast<unsigned int>(tokens.size())});
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch(0, i) = tokens[i];
        }

        // Forward pass - get logits for all positions
        const xt::xtensor<float, 3> logits = g_model->forward(batch);

        // Get logits for the last position (predicting next token)
        // Extract to 1D tensor and negate for argsort
        xt::xtensor<float, 1> last_logits = xt::view(logits, 0, tokens.size() - 1, xt::all());
        xt::xtensor<float, 1> neg_logits = -1.0f * last_logits;
        const auto tokens_in_order = xt::argsort(neg_logits);

        std::vector<uint32_t> top5;
        for (size_t j = 0; j < 5; j++) {
            top5.push_back(tokens_in_order(j));
        }
        return top5;
    }
};

TEST_F(OlmoModelTest, FourscoreSequence) {
    // Test predictions for token sequences from fourscore.tokens.bin
    // Note: Batch processing doesn't have KV cache, so we pass full sequences

    // Test after seeing "Four" (token 28070)
    std::vector<uint32_t> actual = get_top5_predictions({28070});
    std::vector<uint32_t> expected = {315, 1667, 15247, 3115, 339};
    EXPECT_EQ(actual, expected);

    // Test after seeing "Four score" (tokens 28070, 5573)
    actual = get_top5_predictions({28070, 5573});
    expected = {323, 11, 1667, 279, 1752};
    EXPECT_EQ(actual, expected);

    // Test after seeing "Four score and" (tokens 28070, 5573, 323)
    actual = get_top5_predictions({28070, 5573, 323});
    expected = {810, 8208, 264, 4330, 584};
    EXPECT_EQ(actual, expected);

    // Test after seeing "Four score and seven" (tokens 28070, 5573, 323, 8254)
    actual = get_top5_predictions({28070, 5573, 323, 8254});
    expected = {1667, 5573, 4520, 198, 1};
    EXPECT_EQ(actual, expected);
}
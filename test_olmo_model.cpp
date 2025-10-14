#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/misc/xsort.hpp>
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

    std::vector<uint32_t> get_top5_predictions(const uint32_t token) {
        const xt::xtensor<float, 1> logits = -1 * g_model->forward(token);
        const auto tokens_in_order = xt::argsort(logits);

        std::vector<uint32_t> top5;
        for (size_t j = 0; j < 5; j++) {
            top5.push_back(tokens_in_order(j));
        }
        return top5;
    }
};

TEST_F(OlmoModelTest, FourscoreSequence) {
    // Test the sequence of tokens from fourscore.tokens.bin
    // Model is stateful, so tokens must be processed in order

    std::vector<uint32_t> actual = get_top5_predictions(28070);
    std::vector<uint32_t> expected = {315, 1667, 15247, 3115, 339};
    EXPECT_EQ(actual, expected);

    actual = get_top5_predictions(5573);
    expected = {323, 11, 1667, 279, 1752};
    EXPECT_EQ(actual, expected);

    actual = get_top5_predictions(323);
    expected = {810, 8208, 264, 4330, 584};
    EXPECT_EQ(actual, expected);

    actual = get_top5_predictions(8254);
    expected = {1667, 5573, 4520, 198, 1};
    EXPECT_EQ(actual, expected);
}

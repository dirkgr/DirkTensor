#include <gtest/gtest.h>
#include <fstream>
#include "Detokenizer.h"

class DetokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary vocabulary file for testing
        vocab_file = "test_vocab.txt";
        std::ofstream file(vocab_file);
        file << "0\t!\n";
        file << "1\t\"\n";
        file << "315\tĠof\n";
        file << "1667\tĠyears\n";
        file << "28070\tFour\n";
        file << "5573\tĠscore\n";
        file << "198\t\\n\n";  // Escaped newline
        file << "9\t\\t\n";   // Escaped tab
        file.close();
    }

    void TearDown() override {
        // Clean up test file
        std::remove(vocab_file.c_str());
    }

    std::string vocab_file;
};

TEST_F(DetokenizerTest, BasicDecoding) {
    Detokenizer detok(vocab_file);

    EXPECT_EQ(detok.decode(0), "!");
    EXPECT_EQ(detok.decode(1), "\"");
    EXPECT_EQ(detok.decode(28070), "Four");
}

TEST_F(DetokenizerTest, SpaceReplacement) {
    Detokenizer detok(vocab_file);

    // Ġ should be replaced with space
    EXPECT_EQ(detok.decode(315), " of");
    EXPECT_EQ(detok.decode(1667), " years");
    EXPECT_EQ(detok.decode(5573), " score");
}

TEST_F(DetokenizerTest, EscapedCharacters) {
    Detokenizer detok(vocab_file);

    // \n should be unescaped to newline
    EXPECT_EQ(detok.decode(198), "\n");

    // \t should be unescaped to tab
    EXPECT_EQ(detok.decode(9), "\t");
}

TEST_F(DetokenizerTest, OutOfBounds) {
    Detokenizer detok(vocab_file);

    // Token ID that doesn't exist should return <UNK>
    EXPECT_EQ(detok.decode(99999), "<UNK>");
}

TEST_F(DetokenizerTest, SparseVocabulary) {
    Detokenizer detok(vocab_file);

    // Test that sparse vocabulary works (token IDs with gaps)
    EXPECT_EQ(detok.decode(28070), "Four");

    // Token ID in the middle of the range but not in vocab should be empty
    EXPECT_EQ(detok.decode(100), "");
}

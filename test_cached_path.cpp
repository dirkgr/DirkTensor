#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "cached_path.h"

namespace fs = std::filesystem;

class CachedPathTest : public ::testing::Test {
protected:
    fs::path temp_cache_dir;

    void SetUp() override {
        // Create a unique temporary directory for each test
        temp_cache_dir = fs::temp_directory_path() / ("test_cached_path_" + std::to_string(getpid()));
        fs::create_directories(temp_cache_dir);
        setenv("DIRKTENSOR_CACHE", temp_cache_dir.c_str(), 1);
    }

    void TearDown() override {
        unsetenv("DIRKTENSOR_CACHE");
        std::error_code ec;
        fs::remove_all(temp_cache_dir, ec);
    }
};

TEST_F(CachedPathTest, LocalPathPassthrough) {
    // A local path should be returned unchanged
    std::string local_path = "some/local/file.txt";
    EXPECT_EQ(cached_path(local_path), local_path);
}

TEST_F(CachedPathTest, LocalPathNonexistentPassthrough) {
    // A non-existent local path should still be returned unchanged
    // (cached_path doesn't validate local paths)
    std::string local_path = "nonexistent/directory/file.npy";
    EXPECT_EQ(cached_path(local_path), local_path);
}

TEST_F(CachedPathTest, RemoteFileDownload) {
    // Download a small file from the remote server
    std::string url = "https://dirktensor.mechanicaldirk.com/OLMo-2-0425-1B/vocab.txt";
    std::string local_path = cached_path(url);

    // The returned path should exist
    EXPECT_TRUE(fs::exists(local_path));

    // It should be under our temp cache directory
    EXPECT_TRUE(local_path.starts_with(temp_cache_dir.string()));

    // The file should have some content (vocab.txt is ~1.3 MB)
    auto file_size = fs::file_size(local_path);
    EXPECT_GT(file_size, 100000u);  // At least 100 KB

    // A metadata sidecar should exist
    EXPECT_TRUE(fs::exists(local_path + ".meta"));
}

TEST_F(CachedPathTest, CacheHit) {
    // Download a file
    std::string url = "https://dirktensor.mechanicaldirk.com/OLMo-2-0425-1B/vocab.txt";
    std::string first_path = cached_path(url);
    ASSERT_TRUE(fs::exists(first_path));

    // Get the modification time
    auto first_mod_time = fs::last_write_time(first_path);

    // Call again — should return the same path without re-downloading
    std::string second_path = cached_path(url);
    EXPECT_EQ(first_path, second_path);

    // File should not have been modified (no re-download)
    auto second_mod_time = fs::last_write_time(second_path);
    EXPECT_EQ(first_mod_time, second_mod_time);
}

TEST_F(CachedPathTest, RemoteFileNotFound) {
    // Requesting a non-existent remote file should throw
    std::string url = "https://dirktensor.mechanicaldirk.com/nonexistent_file.npy";
    EXPECT_THROW(cached_path(url), std::runtime_error);
}

TEST_F(CachedPathTest, CacheExpiry) {
    // First, download a real file to get the URL hash
    std::string url = "https://dirktensor.mechanicaldirk.com/OLMo-2-0425-1B/vocab.txt";
    std::string local_path = cached_path(url);
    ASSERT_TRUE(fs::exists(local_path));

    // Now modify the metadata sidecar to have an old creation_time (31 days ago)
    std::string meta_path = local_path + ".meta";
    {
        // Read all lines
        std::ifstream in(meta_path);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line))
            lines.push_back(line);
        in.close();

        // Replace the last line (creation_time) with a time 31 days in the past
        ASSERT_GE(lines.size(), 6u);
        long old_time = static_cast<long>(std::time(nullptr)) - (31 * 86400);
        lines[5] = std::to_string(old_time);

        std::ofstream out(meta_path);
        for (const auto& l : lines)
            out << l << '\n';
    }

    // Trigger cache expiry by calling cached_path for a different URL
    // (the expiry scan happens on every call)
    std::string url2 = "https://dirktensor.mechanicaldirk.com/OLMo-2-0425-1B/model.norm.weight.npy";
    cached_path(url2);

    // The old file should have been deleted by the expiry scan
    EXPECT_FALSE(fs::exists(local_path));
    EXPECT_FALSE(fs::exists(meta_path));
}

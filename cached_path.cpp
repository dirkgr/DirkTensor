#include "cached_path.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <curl/curl.h>
#include <picosha2.h>

namespace fs = std::filesystem;

static constexpr int cache_expiry_days = 30;

// ---- Helpers ----

static bool is_url(const std::string& path) {
    return path.starts_with("https://") || path.starts_with("http://");
}

static fs::path get_cache_dir() {
    if (const char* env = std::getenv("DIRKTENSOR_CACHE"); env && env[0] != '\0')
        return fs::path(env);
    if (const char* home = std::getenv("HOME"); home && home[0] != '\0')
        return fs::path(home) / ".cache" / "dirktensor";
    throw std::runtime_error("Cannot determine cache directory: HOME is not set");
}

static std::string sha256_hex(const std::string& input) {
    return picosha2::hash256_hex_string(input);
}

// ---- Metadata sidecar ----

static constexpr int meta_format_version = 1;

struct CacheMeta {
    std::string url;
    std::string etag;
    long last_modified = -1;
    long content_length = -1;
    long creation_time = -1;
};

static void write_meta(const fs::path& meta_path, const CacheMeta& meta) {
    std::ofstream f(meta_path);
    if (!f)
        throw std::runtime_error("Cannot write metadata file: " + meta_path.string());
    f << meta_format_version << '\n'
      << meta.url << '\n'
      << meta.etag << '\n'
      << meta.last_modified << '\n'
      << meta.content_length << '\n'
      << meta.creation_time << '\n';
}

static CacheMeta read_meta(const fs::path& meta_path) {
    std::ifstream f(meta_path);
    if (!f)
        return {};
    std::string line;
    if (!std::getline(f, line) || line != std::to_string(meta_format_version))
        return {};
    CacheMeta meta;
    std::getline(f, meta.url);
    std::getline(f, meta.etag);
    if (std::getline(f, line)) meta.last_modified = std::stol(line);
    if (std::getline(f, line)) meta.content_length = std::stol(line);
    if (std::getline(f, line)) meta.creation_time = std::stol(line);
    return meta;
}

// ---- Cache expiry ----

static void expire_old_cache_entries(const fs::path& cache_dir) {
    if (!fs::exists(cache_dir))
        return;

    const auto now = std::chrono::system_clock::now();
    const auto expiry_threshold = std::chrono::duration<double>(cache_expiry_days * 86400.0);

    for (const auto& entry : fs::directory_iterator(cache_dir)) {
        if (!entry.is_regular_file())
            continue;
        if (entry.path().extension() != ".meta")
            continue;

        // Read the metadata to check creation_time
        CacheMeta meta = read_meta(entry.path());
        if (meta.creation_time < 0)
            continue;

        const auto created = std::chrono::system_clock::from_time_t(
            static_cast<std::time_t>(meta.creation_time));
        if (now - created > expiry_threshold) {
            // Delete the cached file and its metadata
            const auto cached_file = fs::path(entry.path()).replace_extension("");
            std::error_code ec;
            fs::remove(cached_file, ec);
            fs::remove(entry.path(), ec);
        }
    }
}

// ---- HTTP helpers ----

static long http_date_to_epoch(const std::string& http_date) {
    std::tm tm = {};
    std::istringstream ss(http_date);
    ss.imbue(std::locale("C"));
    ss >> std::get_time(&tm, "%a, %d %b %Y %H:%M:%S");
    if (ss.fail())
        return -1;
    return static_cast<long>(timegm(&tm));
}

struct HeadResult {
    std::string etag;
    long last_modified = -1;
    long content_length = -1;
    bool success = false;
};

static size_t header_callback(char* buffer, size_t size, size_t nitems, void* userdata) {
    auto* result = static_cast<HeadResult*>(userdata);
    std::string header(buffer, size * nitems);

    // Case-insensitive header comparison
    auto starts_with_ci = [](const std::string& str, const std::string& prefix) {
        if (str.size() < prefix.size()) return false;
        for (size_t i = 0; i < prefix.size(); i++) {
            if (std::tolower(str[i]) != std::tolower(prefix[i])) return false;
        }
        return true;
    };

    auto trim = [](const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");
        auto end = s.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return std::string{};
        return s.substr(start, end - start + 1);
    };

    if (starts_with_ci(header, "etag:")) {
        result->etag = trim(header.substr(5));
        // Strip surrounding quotes from ETag value
        if (result->etag.size() >= 2 && result->etag.front() == '"' && result->etag.back() == '"')
            result->etag = result->etag.substr(1, result->etag.size() - 2);
    } else if (starts_with_ci(header, "last-modified:")) {
        result->last_modified = http_date_to_epoch(trim(header.substr(14)));
    } else if (starts_with_ci(header, "content-length:")) {
        result->content_length = std::stol(trim(header.substr(15)));
    }

    return size * nitems;
}

static HeadResult http_head(const std::string& url) {
    HeadResult result;

    CURL* curl = curl_easy_init();
    if (!curl)
        return result;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &result);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    result.success = (res == CURLE_OK);
    return result;
}

struct DownloadState {
    std::ofstream* file;
    std::string url;
    curl_off_t total_bytes;
    std::chrono::steady_clock::time_point last_log_time;
};

static size_t write_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* state = static_cast<DownloadState*>(userdata);
    state->file->write(static_cast<char*>(ptr), static_cast<std::streamsize>(size * nmemb));
    return size * nmemb;
}

static int progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                              curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    auto* state = static_cast<DownloadState*>(clientp);
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - state->last_log_time);

    if (elapsed.count() >= 10 && dltotal > 0) {
        int pct = static_cast<int>(100 * dlnow / dltotal);
        auto dl_mb = static_cast<double>(dlnow) / (1024.0 * 1024.0);
        auto total_mb = static_cast<double>(dltotal) / (1024.0 * 1024.0);

        // Extract filename from URL
        std::string filename = state->url;
        auto last_slash = filename.rfind('/');
        if (last_slash != std::string::npos)
            filename = filename.substr(last_slash + 1);

        std::cerr << "Downloading " << filename << ": "
                  << pct << "% ("
                  << std::fixed << std::setprecision(1) << dl_mb << " MB / "
                  << total_mb << " MB)" << std::endl;

        state->last_log_time = now;
    }
    return 0;
}

static void download_file(const std::string& url, const fs::path& dest) {
    auto tmp_path = fs::path(dest).concat(".tmp");

    std::ofstream ofs(tmp_path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Cannot create file: " + tmp_path.string());

    DownloadState state;
    state.file = &ofs;
    state.url = url;
    state.total_bytes = 0;
    state.last_log_time = std::chrono::steady_clock::now();

    // Log the start of the download
    std::string filename = url;
    if (auto last_slash = filename.rfind('/'); last_slash != std::string::npos)
        filename = filename.substr(last_slash + 1);
    std::cerr << "Downloading " << filename << "..." << std::endl;

    CURL* curl = curl_easy_init();
    if (!curl) {
        ofs.close();
        fs::remove(tmp_path);
        throw std::runtime_error("Failed to initialize libcurl");
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &state);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &state);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    ofs.close();

    if (res != CURLE_OK) {
        fs::remove(tmp_path);
        throw std::runtime_error("Download failed for " + url + ": " + curl_easy_strerror(res));
    }

    fs::rename(tmp_path, dest);
}

// ---- Find existing cached version ----

static fs::path find_latest_cached(const std::string& url, const fs::path& cache_dir) {
    const std::string url_hash = sha256_hex(url);

    fs::path latest;
    long latest_time = -1;

    if (!fs::exists(cache_dir))
        return {};

    for (const auto& entry : fs::directory_iterator(cache_dir)) {
        if (!entry.is_regular_file())
            continue;
        const auto name = entry.path().filename().string();
        // Match files that start with the url hash (with or without etag suffix)
        if (!name.starts_with(url_hash))
            continue;
        if (name.ends_with(".meta") || name.ends_with(".tmp"))
            continue;

        // Check metadata for creation time
        auto meta_path = fs::path(entry.path()).concat(".meta");
        if (fs::exists(meta_path)) {
            auto meta = read_meta(meta_path);
            if (meta.creation_time > latest_time) {
                latest_time = meta.creation_time;
                latest = entry.path();
            }
        } else {
            // No metadata, still consider it
            if (latest.empty())
                latest = entry.path();
        }
    }
    return latest;
}

// ---- Main function ----

std::string cached_path(const std::string& url_or_path) {
    if (!is_url(url_or_path))
        return url_or_path;

    const fs::path cache_dir = get_cache_dir();
    fs::create_directories(cache_dir);

    // Expire old cache entries
    expire_old_cache_entries(cache_dir);

    // Try HTTP HEAD for freshness info
    HeadResult head = http_head(url_or_path);

    if (head.success) {
        // Determine cache filename
        std::string cache_name;
        if (!head.etag.empty()) {
            cache_name = sha256_hex(url_or_path) + "." + sha256_hex(head.etag);
        } else {
            cache_name = sha256_hex(url_or_path);
        }

        fs::path cached_file = cache_dir / cache_name;
        fs::path meta_path = fs::path(cached_file).concat(".meta");

        // Check if cached file exists and is valid
        if (fs::exists(cached_file) && fs::exists(meta_path)) {
            CacheMeta meta = read_meta(meta_path);

            bool valid = false;
            if (!head.etag.empty()) {
                // ETag match is sufficient
                valid = (meta.etag == head.etag);
            } else if (head.last_modified >= 0 && head.content_length >= 0) {
                // Fall back to Last-Modified + Content-Length
                valid = (meta.last_modified == head.last_modified &&
                         meta.content_length == head.content_length);
            } else {
                // No freshness headers; if file exists, it's valid
                valid = true;
            }

            if (valid)
                return cached_file.string();
        }

        // Download the file
        download_file(url_or_path, cached_file);

        // Write metadata
        CacheMeta meta;
        meta.url = url_or_path;
        meta.etag = head.etag;
        meta.last_modified = head.last_modified;
        meta.content_length = head.content_length;
        meta.creation_time = static_cast<long>(std::time(nullptr));
        write_meta(meta_path, meta);

        return cached_file.string();
    }

    // HEAD request failed — try offline fallback
    fs::path latest = find_latest_cached(url_or_path, cache_dir);
    if (!latest.empty())
        return latest.string();

    throw std::runtime_error("Cannot fetch " + url_or_path + " and no cached version exists");
}

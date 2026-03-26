#pragma once

#include <string>

// Returns a local filesystem path for the given URL or local path.
// If url_or_path is a URL (starts with "https://"), downloads and caches the file.
// If url_or_path is a local path, returns it unchanged.
// Throws std::runtime_error on download failure.
std::string cached_path(const std::string& url_or_path);

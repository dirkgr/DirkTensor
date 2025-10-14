#include "Detokenizer.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

static std::string unescape_string(const std::string& s) {
    std::string result;
    result.reserve(s.size());

    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case 'n':
                    result += '\n';
                    ++i;
                    break;
                case 'r':
                    result += '\r';
                    ++i;
                    break;
                case 't':
                    result += '\t';
                    ++i;
                    break;
                case '\\':
                    result += '\\';
                    ++i;
                    break;
                default:
                    result += s[i];
            }
        } else {
            result += s[i];
        }
    }

    return result;
}

Detokenizer::Detokenizer(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file) {
        throw std::runtime_error("Cannot open vocabulary file: " + vocab_file);
    }

    size_t max_token_id = 0;
    std::string line;
    while (std::getline(file, line)) {
        // Find the tab separator
        const size_t tab_pos = line.find('\t');
        if (tab_pos == std::string::npos) {
            throw std::runtime_error("Invalid vocabulary line (no tab): " + line);
        }

        // Parse token ID
        const std::string id_str = line.substr(0, tab_pos);
        const uint32_t token_id = std::stoul(id_str);

        // Extract token string
        std::string token = line.substr(tab_pos + 1);
        token = unescape_string(token);

        // Ensure the vector is large enough using exponential growth
        if (token_id >= m_vocab.size()) {
            size_t new_size = m_vocab.size() == 0 ? 1 : m_vocab.size();
            while (new_size <= token_id) {
                new_size *= 2;
            }
            m_vocab.resize(new_size);
        }

        m_vocab[token_id] = token;
        max_token_id = std::max(max_token_id, static_cast<size_t>(token_id));
    }

    // Shrink to actual size
    m_vocab.resize(max_token_id + 1);
}

std::string Detokenizer::decode(const uint32_t token_id) const {
    if (token_id >= m_vocab.size()) {
        return "<UNK>";
    }

    std::string token = m_vocab[token_id];

    // Replace Ġ (U+0120) with space
    // Ġ in UTF-8 is 0xC4 0xA0
    std::string result;
    for (size_t i = 0; i < token.size(); ++i) {
        if (i + 1 < token.size() &&
            static_cast<unsigned char>(token[i]) == 0xC4 &&
            static_cast<unsigned char>(token[i + 1]) == 0xA0) {
            result += ' ';
            ++i;  // Skip the next byte
        } else {
            result += token[i];
        }
    }

    return result;
}

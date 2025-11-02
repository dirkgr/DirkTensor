#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Detokenizer {
public:
    explicit Detokenizer(const std::string& vocab_file);

    std::string decode(uint32_t token_id) const;
    uint32_t get_pad_token_id() const { return m_padTokenId; }

private:
    std::vector<std::string> m_vocab;
    uint32_t m_padTokenId;
};

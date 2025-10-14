#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Detokenizer {
public:
    explicit Detokenizer(const std::string& vocab_file);

    std::string decode(uint32_t token_id) const;

private:
    std::vector<std::string> m_vocab;
};

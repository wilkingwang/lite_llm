#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace comm
{
    static const uint32_t MAX_CODEPOINTS = 0x110000;

    struct NFDRange
    {
        uint32_t first;
        uint32_t last;
        uint32_t nfd;
    };

    extern const std::vector<std::pair<uint32_t, uint16_t>> gUnicodeRangesFlags;
    extern const std::unordered_set<uint32_t> gUnicodeSetWhitespace;
    extern const std::unordered_map<uint32_t, uint32_t> gUnicodeMapLowercase;
    extern const std::unordered_map<uint32_t, uint32_t> gUnicodeMapUppercase;
    extern const std::vector<NFDRange> gUnicodeNFDRanges;

}
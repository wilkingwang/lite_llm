#pragma once

#include <cassert>
#include <limits>
#include <optional>
#include <regex>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <re2/re2.h>

#include "ic_unordered_dense.h"

namespace tiktoken
{
    class TikToken
    {
    public:
        TikToken() = default;

        TikToken(ankerl::unordered_dense::map<std::string, int> encoder,
                 ankerl::unordered_dense::map<std::string, int> specialEncoder,
                 const std::string &pattern);

        auto encode_ordinary(const std::string &text) const -> std::vector<int>;

        auto encode(const std::string &text) const -> std::vector<int>;

        auto encode_single_piece(const std::string &text) const -> std::vector<int>;

        auto decode(const std::vector<int> &tokens) const -> std::string;

    private:
        auto splitWithAllowedSpecialToken(re2::StringPiece &input,
                                          const ankerl::unordered_dense::map<std::string, int> &allowedSpecialTokens) const
            -> std::pair<std::optional<std::string>, re2::StringPiece>;

        auto ordinaryNativeEncode(const std::string &text) const -> std::vector<int>;

        auto nativeEncode(const std::string &text,
                          const ankerl::unordered_dense::map<std::string, int> &allowedSpecialTokens) const
            -> std::pair<std::vector<int>, int>;

        auto nativeDecode(const std::vector<int> &tokens) const -> std::string;

    private:
        std::unique_ptr<re2::RE2> regex;
        std::unique_ptr<re2::RE2> specialRegex;

        ankerl::unordered_dense::map<std::string, int> encoder;
        ankerl::unordered_dense::map<std::string, int> specialTokenEncoder;
        ankerl::unordered_dense::map<int, std::string> decoder;
        ankerl::unordered_dense::map<int, std::string> specialTokenDecoder;
    };
}
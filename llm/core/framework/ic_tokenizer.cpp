#include <vector>
#include <functional>

#include "ic_tokenizer.h"
#include "ic_unordered_dense.h"

namespace tiktoken
{
    static auto mergeBytePair(const std::string &piece,
                              const ankerl::unordered_dense::map<std::string, int> &ranks,
                              std::function<int(int, int)> func) -> std::vector<int>
    {
        std::vector<std::pair<int, int>> parts;

        parts.reserve(piece.size() + 1);
        for (auto idx = 0U; idx < piece.size() + 1; ++idx)
        {
            parts.emplace_back(idx, std::numeric_limits<int>::max());
        }

        auto getRank = [&piece, &ranks](
                           const std::vector<std::pair<int, int>> &parts,
                           int start_idx,
                           int skip) -> std::optional<int>
        {
            if (start_idx + skip + 2 < parts.size())
            {
                auto s = parts[start_idx].first;
                auto e = parts[start_idx + skip + 2].first;
                auto key = piece.substr(s, e - s);
                auto iter = ranks.find(key);
                if (iter != ranks.end())
                {
                    return iter->second;
                }
            }
            return std::nullopt;
        };

        for (auto i = 0U; i < parts.size() - 2; ++i)
        {
            auto rank = getRank(parts, i, 0);
            if (rank)
            {
                assert(*rank != std::numeric_limits<int>::max());
                parts[i].second = *rank;
            }
        }

        while (true)
        {
            if (parts.size() == 1)
                break;

            auto min_rank = std::make_pair<int, int>(std::numeric_limits<int>::max(), 0);
            for (auto i = 0U; i < parts.size() - 1; ++i)
            {
                auto rank = parts[i].second;
                if (rank < min_rank.first)
                {
                    min_rank = {rank, i};
                }
            }

            if (min_rank.first != std::numeric_limits<int>::max())
            {
                auto i = min_rank.second;
                auto rank = getRank(parts, i, 1);
                if (rank)
                {
                    parts[i].second = *rank;
                }
                else
                {
                    parts[i].second = std::numeric_limits<int>::max();
                }
                if (i > 0)
                {
                    auto rank = getRank(parts, i - 1, 1);
                    if (rank)
                    {
                        parts[i - 1].second = *rank;
                    }
                    else
                    {
                        parts[i - 1].second = std::numeric_limits<int>::max();
                    }
                }

                parts.erase(parts.begin() + (i + 1));
            }
            else
            {
                break;
            }
        }

        std::vector<int> out;
        out.reserve(parts.size() - 1);
        for (auto i = 0U; i < parts.size() - 1; ++i)
        {
            out.push_back(func(parts[i].first, parts[i + 1].first));
        }
        return out;
    }

    static auto encodeBytePair(const std::string &piece,
                               const ankerl::unordered_dense::map<std::string, int> &ranks) -> std::vector<int>
    {
        if (piece.size() == 1)
        {
            return {ranks.at(piece)};
        }

        auto func = [&piece, &ranks](int start, int stop) -> int
        {
            std::string key = piece.substr(start, stop - start);
            return ranks.at(key);
        };

        return mergeBytePair(piece, ranks, func);
    }

    TikToken::TikToken(ankerl::unordered_dense::map<std::string, int> encoder,
                       ankerl::unordered_dense::map<std::string, int> specialEncoder,
                       const std::string &pattern)
    {
        regex = std::make_unique<re2::RE2>("(" + pattern + ")");

        std::string specialPattern;
        for (const auto &item : specialEncoder)
        {
            if (!specialPattern.empty())
            {
                specialPattern += "|";
            }
            specialPattern += re2::RE2::QuoteMeta(item.first);
        }

        if (specialPattern.empty())
        {
            specialRegex = nullptr;
        }
        else
        {
            specialRegex = std::make_unique<re2::RE2>("(" + specialPattern + ")");
        }

        encoder = std::move(encoder);
        specialTokenEncoder = std::move(specialEncoder);

        for (const auto &[k, v] : encoder)
        {
            decoder.emplace(v, k);
        }

        for (const auto &[k, v] : specialTokenEncoder)
        {
            specialTokenDecoder.emplace(v, k);
        }
    }

    auto TikToken::encode_ordinary(const std::string &text) const -> std::vector<int>
    {
        return ordinaryNativeEncode(text);
    }

    auto TikToken::encode(const std::string &text) const -> std::vector<int>
    {
        return nativeEncode(text, specialTokenEncoder).first;
    }

    auto TikToken::encode_single_piece(const std::string &text) const -> std::vector<int>
    {
        auto iter = encoder.find(text);
        if (iter != encoder.end())
        {
            return {iter->second};
        }
        return encodeBytePair(text, encoder);
    }

    auto TikToken::decode(const std::vector<int> &tokens) const -> std::string
    {
        return nativeDecode(tokens);
    }

    auto TikToken::splitWithAllowedSpecialToken(re2::StringPiece &input,
                                                const ankerl::unordered_dense::map<std::string, int> &allowedSpecialTokens) const -> std::pair<std::optional<std::string>, re2::StringPiece>
    {
        if (specialRegex == nullptr)
            return {std::nullopt, input};

        auto start = input.begin();
        std::string special;
        while (true)
        {
            if (!re2::RE2::FindAndConsume(&input, *specialRegex, &special))
            {
                break;
            }

            if (allowedSpecialTokens.count(special) == 1)
            {
                return {std::move(special), re2::StringPiece(start, input.begin() - start - special.size())};
            }
        }

        return {std::nullopt, input};
    }

    auto TikToken::ordinaryNativeEncode(const std::string &text) const -> std::vector<int>
    {
        std::vector<int> ret;
        re2::StringPiece input(text);

        std::string piece;
        while (re2::RE2::FindAndConsume(&input, *regex, &piece))
        {
            auto iter = encoder.find(piece);
            if (iter != encoder.end())
            {
                ret.push_back(iter->second);
                continue;
            }
            auto tokens = encodeBytePair(piece, encoder);
            ret.insert(ret.end(), tokens.begin(), tokens.end());
        }
        return ret;
    }

    auto TikToken::nativeEncode(const std::string &text,
                                const ankerl::unordered_dense::map<std::string, int> &allowedSpecialTokens) const -> std::pair<std::vector<int>, int>
    {
        std::vector<int> ret;
        int lastPieceTokenLen = 0;
        re2::StringPiece input(text);

        while (true)
        {
            auto [special, sub_input] = splitWithAllowedSpecialToken(input, allowedSpecialTokens);
            std::string piece;
            while (re2::RE2::FindAndConsume(&sub_input, *regex, &piece))
            {
                auto iter = encoder.find(piece);
                if (iter != encoder.end())
                {
                    lastPieceTokenLen = 1;
                    ret.push_back(iter->second);
                    continue;
                }
                auto tokens = encodeBytePair(piece, encoder);
                lastPieceTokenLen = tokens.size();
                ret.insert(ret.end(), tokens.begin(), tokens.end());
            }

            if (special)
            {
                int token = specialTokenEncoder.at(*special);
                ret.push_back(token);
                lastPieceTokenLen = 0;
            }
            else
            {
                break;
            }
        }

        return {ret, lastPieceTokenLen};
    }

    auto TikToken::nativeDecode(const std::vector<int> &tokens) const -> std::string
    {
        std::string ret;
        ret.reserve(tokens.size() * 2);
        for (auto token : tokens)
        {
            std::string tokenBytes;
            auto iter = decoder.find(token);
            if (iter != decoder.end())
            {
                tokenBytes = iter->second;
            }
            else
            {
                iter = specialTokenDecoder.find(token);
                if (iter != specialTokenDecoder.end())
                {
                    tokenBytes = iter->second;
                }
                else
                {
                    throw std::runtime_error("unknown token: " + std::to_string(token));
                }
            }

            ret += tokenBytes;
        }

        return ret;
    }
}
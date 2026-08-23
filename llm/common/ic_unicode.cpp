#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <locale>
#include <codecvt>
#include <map>
#include <vector>
#include <string>
#include <regex>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "ic_unicode.h"
#include "ic_unicode_data.h"

namespace comm
{
    size_t getUTF8UnicodeLen(char src)
    {
        const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
        uint8_t highbits = static_cast<uint8_t>(src) >> 4;
        return lookup[highbits];
    }

    std::string getUTF8FromUnicodeCpt(uint32_t codePoint)
    {
        std::string result;

        if (codePoint <= 0x7f)
        {
            result.push_back(codePoint);
            return result;
        }

        if (0x80 <= codePoint && codePoint <= 0x7ff)
        {
            result.push_back(0xc0 | ((codePoint >> 6) & 0x1f));
            result.push_back(0x80 | (codePoint & 0x3f));
            return result;
        }

        if (0x800 <= codePoint && codePoint <= 0xffff)
        {
            result.push_back(0xe0 | ((codePoint >> 12) & 0x0f));
            result.push_back(0x80 | ((codePoint >> 6) & 0x3f));
            result.push_back(0x80 | (codePoint & 0x3f));
            return result;
        }

        if (0x10000 <= codePoint && codePoint <= 0x10ffff)
        {
            result.push_back(0xf0 | ((codePoint >> 18) & 0x07));
            result.push_back(0x80 | ((codePoint >> 12) & 0x3f));
            result.push_back(0x80 | ((codePoint >> 6) & 0x3f));
            result.push_back(0x80 | (codePoint & 0x3f));
            return result;
        }

        throw std::invalid_argument("invalid codepoint");
    }

    /**
     * @brief 构建一个长度为MAX_CODEPOINTS的查表向量，向量的每个索引对应一个 Unicode 码点，
     * 表项是CodePointFlags(表示该码点的类别（字母/数字/标点/控制 等）和辅助属性（is_whitespace、is_lowercase、is_uppercase、is_nfd 等)
     */
    static std::vector<CodePointFlags> getUnicodeCptFlags()
    {
        std::vector<CodePointFlags> cptFlags(MAX_CODEPOINTS, CodePointFlags::UNDEFINED);

        assert(gUnicodeRangesFlags.front().first == 0);
        assert(gUnicodeRangesFlags.back().first == MAX_CODEPOINTS);
        for (size_t i = 1; i < gUnicodeRangesFlags.size(); ++i)
        {
            const auto rangeStart = gUnicodeRangesFlags[i - 1]; // codepoint_start, flags
            const auto rangeEnd = gUnicodeRangesFlags[i];       // codepoint_end, flags
            for (uint32_t cpt = rangeStart.first; cpt < rangeEnd.first; ++cpt)
            {
                cptFlags[cpt] = rangeStart.second;
            }
        }

        for (auto cpt : gUnicodeSetWhitespace)
        {
            cptFlags[cpt].isWhitespace = true;
        }

        for (auto p : gUnicodeMapLowercase)
        {
            cptFlags[p.second].isLowercase = true;
        }

        for (auto p : gUnicodeMapUppercase)
        {
            cptFlags[p.second].isUppercase = true;
        }

        for (auto &range : gUnicodeNFDRanges)
        {
            cptFlags[range.nfd].isNfd = true;
        }

        return cptFlags;
    }

    /**
     * @brief 将每个可能的字节值映射到一个唯一的合法 UTF‑8 字符串表示
     */
    static std::unordered_map<uint8_t, std::string> getUnicodeByteToUTF8Map()
    {
        std::unordered_map<uint8_t, std::string> map;

        for (int ch = 0x21; ch <= 0x7E; ++ch)
        { // u'!' to u'~'
            assert(0 <= ch && ch < 256);
            map[ch] = getUTF8FromUnicodeCpt(ch);
        }

        for (int ch = 0xA1; ch <= 0xAC; ++ch)
        { // u'¡' to u'¬'
            assert(0 <= ch && ch < 256);
            map[ch] = getUTF8FromUnicodeCpt(ch);
        }

        for (int ch = 0xAE; ch <= 0xFF; ++ch)
        { // u'®' to u'ÿ'
            assert(0 <= ch && ch < 256);
            map[ch] = getUTF8FromUnicodeCpt(ch);
        }

        auto n = 0;
        for (int ch = 0; ch < 256; ++ch)
        {
            if (map.find(ch) == map.end())
            {
                map[ch] = getUTF8FromUnicodeCpt(256 + n);
                ++n;
            }
        }

        return map;
    }

    /**
     * @brief 将每个可能的字节值对应的UTF8映射回单字节Unicode码
     */
    static std::unordered_map<std::string, uint8_t> convertUnicodeCptToUTF8()
    {
        std::unordered_map<std::string, uint8_t> map;
        for (int ch = 0x21; ch <= 0x7E; ++ch)
        { // u'!' to u'~'
            assert(0 <= ch && ch < 256);
            map[getUTF8FromUnicodeCpt(ch)] = ch;
        }
        for (int ch = 0xA1; ch <= 0xAC; ++ch)
        { // u'¡' to u'¬'
            assert(0 <= ch && ch < 256);
            map[getUTF8FromUnicodeCpt(ch)] = ch;
        }
        for (int ch = 0xAE; ch <= 0xFF; ++ch)
        { // u'®' to u'ÿ'
            assert(0 <= ch && ch < 256);
            map[getUTF8FromUnicodeCpt(ch)] = ch;
        }
        auto n = 0;
        for (int ch = 0; ch < 256; ++ch)
        {
            if (map.find(getUTF8FromUnicodeCpt(ch)) == map.end())
            {
                map[getUTF8FromUnicodeCpt(256 + n)] = ch;
                ++n;
            }
        }
        return map;
    }

    static inline std::wstring convertUTF8ToUnicodeWstring(const std::string &s)
    {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

        return conv.from_bytes(s);
    }

    static std::vector<std::string> encodeUnicodeByte(const std::vector<std::string> &bpe_words)
    {
        std::vector<std::string> bpe_encoded_words;
        for (const auto &word : bpe_words)
        {
            std::string textUTF;
            auto utf_word = convertUTF8ToUnicodeCpts(word);
            for (size_t i = 0; i < utf_word.size(); ++i)
            {
                textUTF += convertUnicodeCptToUTF8(utf_word[i]);
            }

            std::string encodedToken;
            for (char &c : textUTF)
            {
                encodedToken += convertUnicodeByteToUTF8(c);
            }
            bpe_encoded_words.emplace_back(encodedToken);
        }
        return bpe_encoded_words;
    }

    static std::vector<size_t> gpt2CustomSplitUnicodeByRegex(const std::string &text, const std::vector<size_t> &offsets)
    {
        std::vector<size_t> bpeOffsets;     // store the offset of each word
        bpeOffsets.reserve(offsets.size()); // Reserve memory for the approximate size

        const auto cpts = convertUTF8ToUnicodeCpts(text);

        size_t start = 0;
        for (auto offset : offsets)
        {
            const size_t offsetStart = start;
            const size_t offsetEnd = start + offset;
            assert(offsetEnd <= cpts.size());
            start = offsetEnd;

            static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
            auto getCpt = [&](const size_t pos) -> uint32_t
            {
                return (offsetStart <= pos && pos < offsetEnd) ? cpts[pos] : OUT_OF_RANGE;
            };

            auto getCpFlags = [&](const size_t pos) -> CodePointFlags
            {
                return (offsetStart <= pos && pos < offsetEnd) ? getUnicodeCptFlags(cpts[pos]) : comm::CodePointFlags{};
            };

            size_t _prevEnd = offsetStart;
            auto _add_token = [&](const size_t end) -> size_t
            {
                assert(_prevEnd <= end && end <= offsetEnd);
                size_t len = end - _prevEnd;
                if (len > 0)
                {
                    bpeOffsets.push_back(len);
                }
                _prevEnd = end;

                return len;
            };

            for (size_t pos = offsetStart; pos < offsetEnd; /*pos++*/)
            {
                const uint32_t cpt = getCpt(pos);
                const auto flags = getCpFlags(pos);

                // regex: 's|'t|'re|'ve|'m|'ll|'d
                if (cpt == '\'' && pos + 1 < offsetEnd)
                {
                    uint32_t cptNext = getCpt(pos + 1);
                    if (cptNext == 's' || cptNext == 't' || cptNext == 'm' || cptNext == 'd')
                    {
                        pos += _add_token(pos + 2);
                        continue;
                    }
                    if (pos + 2 < offsetEnd)
                    {
                        uint32_t cptNextNext = getCpt(pos + 2);
                        if ((cptNext == 'r' && cptNextNext == 'e') ||
                            (cptNext == 'v' && cptNextNext == 'e') ||
                            (cptNext == 'l' && cptNextNext == 'l'))
                        {
                            pos += _add_token(pos + 3);
                            continue;
                        }
                    }
                }

                auto flags2 = (cpt == ' ' ? getCpFlags(pos + 1) : flags);
                // regex: <space>?\p{L}+
                if (flags2.isLetter)
                {
                    pos += (cpt == ' ');
                    while (flags2.isLetter)
                    {
                        flags2 = getCpFlags(++pos);
                    }
                    _add_token(pos);
                    continue;
                }
                // regex: <space>?\p{N}+
                if (flags2.isNumber)
                {
                    pos += (cpt == ' ');
                    while (flags2.isNumber)
                    {
                        flags2 = getCpFlags(++pos);
                    }
                    _add_token(pos);
                    continue;
                }
                // regex: <space>?[^\s\p{L}\p{N}]+
                if (!(flags2.isWhitespace | flags2.isLetter | flags2.isNumber) && flags2.asUint())
                {
                    pos += (cpt == ' ');
                    while (!(flags2.isWhitespace | flags2.isLetter | flags2.isNumber) && flags2.asUint())
                    {
                        flags2 = getCpFlags(++pos);
                    }
                    _add_token(pos);
                    continue;
                }

                size_t numWhitespaces = 0;
                while (getCpFlags(pos + numWhitespaces).isWhitespace)
                {
                    numWhitespaces++;
                }

                // regex: \s+(?!\S)
                if (numWhitespaces > 1 && getCpt(pos + numWhitespaces) != OUT_OF_RANGE)
                {
                    pos += numWhitespaces - 1;
                    _add_token(pos);
                    continue;
                }

                // regex: \s+
                if (numWhitespaces > 0)
                {
                    pos += numWhitespaces;
                    _add_token(pos);
                    continue;
                }

                // no matches
                _add_token(++pos);
            }
        }

        return bpeOffsets;
    }

    static std::vector<size_t> llama3CustomSplitUnicodeByregex(const std::string &text, const std::vector<size_t> &offsets)
    {
        std::vector<size_t> bpeOffsets;     // store the offset of each word
        bpeOffsets.reserve(offsets.size()); // Reserve memory for the approximate size

        const auto cpts = convertUTF8ToUnicodeCpts(text);

        size_t start = 0;
        for (auto offset : offsets)
        {
            const size_t offsetStart = start;
            const size_t offsetEnd = start + offset;
            assert(offsetEnd <= cpts.size());
            start = offsetEnd;

            static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
            auto getCpt = [&](const size_t pos) -> uint32_t
            {
                return (offsetStart <= pos && pos < offsetEnd) ? cpts[pos] : OUT_OF_RANGE;
            };

            auto getCpFlags = [&](const size_t pos) -> CodePointFlags
            {
                return (offsetStart <= pos && pos < offsetEnd) ? getUnicodeCptFlags(cpts[pos]) : CodePointFlags{};
            };

            size_t prevEnd = offsetStart;
            auto addToken = [&](const size_t end) -> size_t
            {
                assert(prevEnd <= end && end <= offsetEnd);
                size_t len = end - prevEnd;
                if (len > 0)
                {
                    bpeOffsets.push_back(len);
                }
                prevEnd = end;

                return len;
            };

            for (size_t pos = offsetStart; pos < offsetEnd; /*pos++*/)
            {
                const uint32_t cpt = getCpt(pos);
                const auto flags = getCpFlags(pos);

                // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
                if (cpt == '\'' && pos + 1 < offsetEnd)
                {
                    uint32_t cptNext = convertUnicodeToLower(getCpt(pos + 1));
                    if (cptNext == 's' || cptNext == 't' || cptNext == 'm' || cptNext == 'd')
                    {
                        pos += addToken(pos + 2);
                        continue;
                    }
                    if (pos + 2 < offsetEnd)
                    {
                        uint32_t cptNextNext = convertUnicodeToLower(getCpt(pos + 2));
                        if ((cptNext == 'r' && cptNextNext == 'e') ||
                            (cptNext == 'v' && cptNextNext == 'e') ||
                            (cptNext == 'l' && cptNextNext == 'l'))
                        {
                            pos += addToken(pos + 3);
                            continue;
                        }
                    }
                }

                // regex: [^\r\n\p{L}\p{N}]?\p{L}+
                if (!(cpt == '\r' || cpt == '\n' || flags.isNumber))
                {
                    if (flags.isLetter || getCpFlags(pos + 1).isLetter)
                    { // one or more letters
                        pos++;
                        while (getCpFlags(pos).isLetter)
                        {
                            pos++;
                        }
                        addToken(pos);
                        continue;
                    }
                }

                // regex: \p{N}{1,3}
                if (flags.isNumber)
                {
                    size_t ini = pos;
                    while (getCpFlags(pos).isNumber)
                    {
                        if (++pos - ini >= 3)
                        {
                            addToken(pos);
                            ini = pos;
                        }
                    }
                    addToken(pos);
                    continue;
                }

                // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
                auto flags2 = (cpt == ' ' ? getCpFlags(pos + 1) : flags);
                if (!(flags2.isWhitespace | flags2.isLetter | flags2.isNumber) && flags.asUint())
                {
                    pos += (cpt == ' ');
                    while (!(flags2.isWhitespace | flags2.isLetter | flags2.isNumber) && flags2.asUint())
                    {
                        flags2 = getCpFlags(++pos);
                    }
                    uint32_t cpt2 = getCpt(pos);
                    while (cpt2 == '\r' || cpt2 == '\n')
                    {
                        cpt2 = getCpt(++pos);
                    }
                    addToken(pos);
                    continue;
                }

                size_t numWhitespaces = 0;
                size_t last_end_r_or_n = 0;
                while (getCpFlags(pos + numWhitespaces).isWhitespace)
                {
                    uint32_t cpt2 = getCpt(pos + numWhitespaces);
                    if (cpt2 == '\r' || cpt2 == '\n')
                    {
                        last_end_r_or_n = pos + numWhitespaces + 1;
                    }
                    numWhitespaces++;
                }

                // regex: \s*[\r\n]+
                if (last_end_r_or_n > 0)
                {
                    pos = last_end_r_or_n;
                    addToken(pos);
                    continue;
                }

                // regex: \s+(?!\S)
                if (numWhitespaces > 1 && getCpt(pos + numWhitespaces) != OUT_OF_RANGE)
                {
                    pos += numWhitespaces - 1;
                    addToken(pos);
                    continue;
                }

                // regex: \s+
                if (numWhitespaces > 0)
                {
                    pos += numWhitespaces;
                    addToken(pos);
                    continue;
                }

                // no matches
                addToken(++pos);
            }
        }

        return bpeOffsets;
    }

    // use std::wregex to split the text
    static std::vector<size_t> unicode_regex_split_stl(const std::wstring &wtext, const std::wstring &regex_expr, const std::vector<size_t> &offsets)
    {
        std::wregex expr(regex_expr);
        std::vector<size_t> bpeOffsets;     // store the offset of each word
        bpeOffsets.reserve(offsets.size()); // Reserve memory for the approximate size
        size_t start = 0;
        for (auto offset : offsets)
        {
            std::wcregex_iterator it(wtext.data() + start, wtext.data() + start + offset, expr);
            std::wcregex_iterator end;

            int64_t start_idx = 0;
            while (it != end)
            {
                std::wcmatch match = *it;
                if (match.position() > start_idx)
                {
                    bpeOffsets.emplace_back(match.position() - start_idx);
                }
                bpeOffsets.emplace_back(match.length());
                start_idx = match.position() + match.length();
                ++it;
            }

            if (start_idx < (int64_t)offset)
            {
                bpeOffsets.emplace_back(offset - start_idx);
            }
            start += offset;
        }

        return bpeOffsets;
    }

    static std::vector<size_t> unicode_regex_split_stl(const std::string &text, const std::string &regex_expr, const std::vector<size_t> &offsets)
    {
        std::regex expr(regex_expr);
        std::vector<size_t> bpeOffsets;     // store the offset of each word
        bpeOffsets.reserve(offsets.size()); // Reserve memory for the approximate size
        size_t start = 0;
        for (auto offset : offsets)
        {
            std::cregex_iterator it(text.data() + start, text.data() + start + offset, expr);
            std::cregex_iterator end;

            int64_t start_idx = 0;
            while (it != end)
            {
                std::cmatch match = *it;
                if (match.position() > start_idx)
                {
                    bpeOffsets.emplace_back(match.position() - start_idx);
                }
                bpeOffsets.emplace_back(match.length());
                start_idx = match.position() + match.length();
                ++it;
            }

            if (start_idx < (int64_t)offset)
            {
                bpeOffsets.emplace_back(offset - start_idx);
            }
            start += offset;
        }

        return bpeOffsets;
    }

    static std::vector<size_t> customSplitUnicodeByRegex(const std::string &text, const std::string &regex_expr, const std::vector<size_t> &offsets)
    {
        std::vector<size_t> bpeOffsets;

        if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)")
        {
            bpeOffsets = gpt2CustomSplitUnicodeByRegex(text, offsets);
        }
        else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
        {

            bpeOffsets = llama3CustomSplitUnicodeByregex(text, offsets);
        }

        return bpeOffsets;
    }

    uint32_t convertUTF8ToUnicodeCpt(const std::string &utf8, size_t &offset)
    {
        assert(offset < utf8.size());
        if (!(utf8[offset + 0] & 0x80))
        {
            auto result = utf8[offset + 0];
            offset += 1;
            return result;
        }

        if (!(utf8[offset + 0] & 0x40))
        {
            throw std::invalid_argument("invalid character");
        }

        if (!(utf8[offset + 0] & 0x20))
        {
            if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80))
            {
                throw std::invalid_argument("invalid character");
            }

            auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
            offset += 2;
            return result;
        }

        if (!(utf8[offset + 0] & 0x10))
        {
            if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80))
            {
                throw std::invalid_argument("invalid character");
            }

            auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
            offset += 3;
            return result;
        }

        if (!(utf8[offset + 0] & 0x08))
        {
            if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80))
            {
                throw std::invalid_argument("invalid character");
            }

            auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
            offset += 4;
            return result;
        }

        throw std::invalid_argument("failed to convert utf8 to codepoint");
    }

    std::vector<uint32_t> convertUTF8ToUnicodeCpts(const std::string &utf8)
    {
        size_t offset = 0;
        std::vector<uint32_t> result;

        result.reserve(utf8.size());
        while (offset < utf8.size())
        {
            result.push_back(convertUTF8ToUnicodeCpt(utf8, offset));
        }

        return result;
    }

    std::vector<uint32_t> getUnicodeCptsByNormalizeNFD(const std::vector<uint32_t> &cpts)
    {
        auto comp = [](const uint32_t cpt, const NFDRange &range)
        {
            return cpt < range.first;
        };

        std::vector<uint32_t> result(cpts.size());
        for (size_t i = 0; i < cpts.size(); ++i)
        {
            const uint32_t cpt = cpts[i];
            auto it = std::upper_bound(gUnicodeNFDRanges.cbegin(), gUnicodeNFDRanges.cend(), cpt, comp) - 1;
            result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
        }

        return result;
    }

    CodePointFlags getUnicodeCptFlags(const uint32_t cp)
    {
        static const CodePointFlags undef(CodePointFlags::UNDEFINED);
        static const auto cptFlags = getUnicodeCptFlags();
        return cp < cptFlags.size() ? cptFlags[cp] : undef;
    }

    CodePointFlags getUnicodeCptFlags(const std::string &utf8)
    {
        static const CodePointFlags undef(CodePointFlags::UNDEFINED);
        if (utf8.empty())
        {
            return undef; // undefined
        }

        size_t offset = 0;
        return getUnicodeCptFlags(convertUTF8ToUnicodeCpt(utf8, offset));
    }

    std::string convertUnicodeByteToUTF8(uint8_t byte)
    {
        static std::unordered_map<uint8_t, std::string> map = getUnicodeByteToUTF8Map();
        return map.at(byte);
    }

    uint8_t convertUTF8ToUnicodeByte(const std::string &utf8)
    {
        static std::unordered_map<std::string, uint8_t> map = convertUnicodeCptToUTF8();
        return map.at(utf8);
    }

    uint32_t convertUnicodeToLower(uint32_t cp)
    {
        auto it = gUnicodeMapLowercase.find(cp);
        return it == gUnicodeMapLowercase.end() ? cp : it->second;
    }

    std::vector<std::string> splitUnicodeByRegex(const std::string &text, const std::vector<std::string> &regexExprs)
    {
        // unicode categories
        static const std::map<std::string, int> k_ucat_enum = {
            {"\\p{N}", CodePointFlags::NUMBER},
            {"\\p{L}", CodePointFlags::LETTER},
            {"\\p{P}", CodePointFlags::PUNCTUATION},
        };

        static const std::map<int, int> k_ucat_cpt = {
            {CodePointFlags::NUMBER, 0xD1},
            {CodePointFlags::LETTER, 0xD2},
            {CodePointFlags::PUNCTUATION, 0xD3},
        };

        static const std::map<int, std::string> k_ucat_map = {
            {CodePointFlags::NUMBER, "\x30-\x39"},                                                                       // 0-9
            {CodePointFlags::LETTER, "\x41-\x5A\x61-\x7A"},                                                              // A-Za-z
            {CodePointFlags::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D"}, // !-#%-*,-/:-;?-@\[-\]_\{\}
        };

        // compute collapsed codepoints only if needed by at least one regex
        bool bNeedCollapse = false;
        for (auto &regexExpr : regexExprs)
        {
            // search for unicode categories
            for (const auto &ucat : k_ucat_enum)
            {
                if (std::string::npos != regexExpr.find(ucat.first))
                {
                    bNeedCollapse = true;
                    break;
                }
            }
        }

        const auto cpts = convertUTF8ToUnicodeCpts(text);

        // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
        // ref: https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2081479935
        std::string textCollapsed;
        if (bNeedCollapse)
        {
            // collapse all unicode categories
            textCollapsed.resize(cpts.size());

            for (size_t i = 0; i < cpts.size(); ++i)
            {
                // keep single-byte codepoints as is
                if (cpts[i] < 128)
                {
                    textCollapsed[i] = cpts[i];
                    continue;
                }

                const auto flags = getUnicodeCptFlags(cpts[i]);

                if (flags.isWhitespace)
                {
                    // NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
                    // text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
                    textCollapsed[i] = (char)0x0B; // <vertical tab> as whitespace fallback
                }
                else if (k_ucat_cpt.find(flags.getCategoryFlag()) != k_ucat_cpt.end())
                {
                    textCollapsed[i] = k_ucat_cpt.at(flags.getCategoryFlag());
                }
                else
                {
                    textCollapsed[i] = (char)0xD0; // fallback
                }
            }
        }

        std::vector<size_t> bpeOffsets = {cpts.size()};
        for (auto &regexExpr : regexExprs)
        {
            // first, see if we have an efficient custom regex implementation
            auto tmp = customSplitUnicodeByRegex(text, regexExpr, bpeOffsets);

            if (!tmp.empty())
            {
                bpeOffsets = std::move(tmp);
                continue;
            }

            // fallback to general-purpose std::regex / std::wregex
            try
            {
                // if a unicode category is used in the regex, we use the collapsed text and replace the unicode category
                // with the corresponding collapsed representation
                bool bUseCollapsed = false;
                for (auto &ucat : k_ucat_enum)
                {
                    if (std::string::npos != regexExpr.find(ucat.first))
                    {
                        bUseCollapsed = true;
                        break;
                    }
                }

                if (bUseCollapsed)
                {
                    // sanity-check that the original regex does not contain any non-ASCII characters
                    const auto cpts_regex = convertUTF8ToUnicodeCpts(regexExpr);
                    for (size_t i = 0; i < cpts_regex.size(); ++i)
                    {
                        if (cpts_regex[i] >= 128)
                        {
                            throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
                        }
                    }

                    // generate a collapsed representation of the regex
                    std::string regexExprCollapsed;

                    // track if we are inside [], because nested [] are not allowed
                    bool inside = false;
                    for (size_t i = 0; i < regexExpr.size(); ++i)
                    {
                        if (regexExpr[i] == '[' && (i == 0 || regexExpr[i - 1] != '\\'))
                        {
                            regexExprCollapsed += '[';
                            inside = true;
                            continue;
                        }

                        if (inside && regexExpr[i] == ']' && regexExpr[i - 1] != '\\')
                        {
                            regexExprCollapsed += ']';
                            inside = false;
                            continue;
                        }

                        if (regexExpr[i + 0] == '\\' && i + 4 < regexExpr.size() &&
                            regexExpr[i + 1] == 'p' &&
                            regexExpr[i + 2] == '{' &&
                            regexExpr[i + 4] == '}')
                        {
                            const std::string pat = regexExpr.substr(i, 5);
                            if (k_ucat_enum.find(pat) != k_ucat_enum.end())
                            {
                                if (!inside)
                                {
                                    regexExprCollapsed += '[';
                                }
                                regexExprCollapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
                                regexExprCollapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                                if (!inside)
                                {
                                    regexExprCollapsed += ']';
                                }
                                i += 4;
                                continue;
                            }
                        }

                        regexExprCollapsed += regexExpr[i];
                    }

                    bpeOffsets = unicode_regex_split_stl(textCollapsed, regexExprCollapsed, bpeOffsets);
                }
                else
                {
                    // no unicode category used, we can use std::wregex directly
                    const std::wstring wregex_expr = convertUTF8ToUnicodeWstring(regexExpr);

                    // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as fallback
                    std::wstring wtext(cpts.begin(), cpts.end());
                    for (size_t i = 0; i < wtext.size(); ++i)
                    {
                        if (wtext[i] > 0x7F && getUnicodeCptFlags(wtext[i]).isWhitespace)
                        {
                            wtext[i] = 0x0B;
                        }
                    }

                    // printf("text: %s\n", text.c_str());
                    // printf("regex_expr: %s\n", regex_expr.c_str());
                    bpeOffsets = unicode_regex_split_stl(wtext, wregex_expr, bpeOffsets);
                }
            }
            catch (std::regex_error &e)
            {
                fprintf(stderr, "Failed to process regex: '%s'\n", regexExpr.c_str());
                fprintf(stderr, "Regex error: %s\n", e.what());
                throw std::runtime_error("Failed to process regex");
            }
        }

        std::vector<std::string> bpe_words;
        bpe_words.reserve(bpeOffsets.size()); // reserve memory for the approximate size

        size_t start = 0;
        for (size_t &offset : bpeOffsets)
        {
            bpe_words.emplace_back();
            for (size_t i = start; i < start + offset; ++i)
            {
                bpe_words.back() += getUTF8FromUnicodeCpt(cpts[i]);
            }
            start += offset;
        }

        return encodeUnicodeByte(bpe_words);
    }

}
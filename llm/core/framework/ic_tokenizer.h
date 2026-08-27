#pragma once

#include <cassert>
#include <optional>
#include <re2/re2.h>
#include <string>
#include <vector>

#include "ic_unordered_dense.h"

namespace tiktoken
{
    class TikToken
    {
      public:
        TikToken() = default;

        /**
         * @param encoder 普通token的bytes对应token id映射表
         * @param specialEncoder 特殊token映射表
         * @param pattern 正则与切分模型
         */
        TikToken(ankerl::unordered_dense::map<std::string, int> encoder,
                 ankerl::unordered_dense::map<std::string, int> specialEncoder,
                 const std::string &pattern);

        /**
         * @brief 对普通token进行正则切分，BPE编码
         */
        auto encodeOrdinary(const std::string &text) const -> std::vector<int>;

        /**
         * @brief 编码并识别特殊token
         */
        auto encode(const std::string &text) const -> std::vector<int>;

        /**
         * @brief 单个piece编码（先查表，查不到时再BPE编码）
         */
        auto
        encodeSinglePiece(const std::string &text) const -> std::vector<int>;

        /**
         * @brief 将token id还原为字符串
         */
        auto decode(const std::vector<int> &tokens) const -> std::string;

      private:
        /**
         * @brief 在输入中扫描特殊token，把文本切成普通段 + 特殊token交替的片段
         */
        auto splitWithAllowedSpecialToken(
            re2::StringPiece &input,
            const ankerl::unordered_dense::map<std::string, int>
                &allowedSpecialTokens) const
            -> std::pair<std::optional<std::string>, re2::StringPiece>;

        /**
         * @brief 正则逐段切分，查表命中直接转token id，否则走Merge
         */
        auto
        ordinaryNativeEncode(const std::string &text) const -> std::vector<int>;

        /**
         * @brief 字符串转token ids
         */
        auto nativeEncode(const std::string &text,
                          const ankerl::unordered_dense::map<std::string, int>
                              &allowedSpecialTokens) const
            -> std::pair<std::vector<int>, int>;

        /**
         * @brief 逐个token查表拼接字符串
         */
        auto nativeDecode(const std::vector<int> &tokens) const -> std::string;

      private:
        std::unique_ptr<re2::RE2> regex;
        std::unique_ptr<re2::RE2> specialRegex;

        ankerl::unordered_dense::map<std::string, int> encoder;
        ankerl::unordered_dense::map<std::string, int> specialTokenEncoder;
        ankerl::unordered_dense::map<int, std::string> decoder;
        ankerl::unordered_dense::map<int, std::string> specialTokenDecoder;
    };
} // namespace tiktoken
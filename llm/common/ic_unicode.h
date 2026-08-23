#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace comm
{
    struct CodePointFlags
    {
        enum
        {
            UNDEFINED = 0x0001,
            NUMBER = 0x0002,      // regex: \p{N}
            LETTER = 0x0004,      // regex: \p{L}
            SEPARATOR = 0x0008,   // regex: \p{Z}
            ACCENT_MARK = 0x0010, // regex: \p{M}
            PUNCTUATION = 0x0020, // regex: \p{P}
            SYMBOL = 0x0040,      // regex: \p{S}
            CONTROL = 0x0080,     // regex: \p{C}
            MASK_CATEGORIES = 0x00FF,
        };

        uint16_t isUndefined : 1;
        uint16_t isNumber : 1;      // regex: \p{N}
        uint16_t isLetter : 1;      // regex: \p{L}
        uint16_t isSeparator : 1;   // regex: \p{Z}
        uint16_t isAccentMark : 1;  // regex: \p{M}
        uint16_t isPunctuation : 1; // regex: \p{P}
        uint16_t isSymbol : 1;      // regex: \p{S}
        uint16_t isControl : 1;     // regex: \p{C}
        // helper flags
        uint16_t isWhitespace : 1; // regex: \s
        uint16_t isLowercase : 1;
        uint16_t isUppercase : 1;
        uint16_t isNfd : 1;

        inline CodePointFlags(uint16_t flags = 0)
        {
            setCodePointFlags(flags);
        }

        inline void setCodePointFlags(const uint16_t flags = 0)
        {
            *reinterpret_cast<uint16_t *>(this) = flags;
        }

        inline uint16_t asUint() const
        {
            return *reinterpret_cast<const uint16_t *>(this);
        }

        inline uint16_t getCategoryFlag() const
        {
            return this->asUint() & MASK_CATEGORIES;
        }
    };

    /**
     * @brief 根据传入字节的高 4 位快速查表，返回该字节作为 UTF‑8 首字节时表示的 UTF‑8 字符长度（以字节为单位）
     * @param src UTF-8字节
     * @return 返回该字节作为UTF-8首字节时对应的UTF‑8字符长度
     */
    size_t getUTF8UnicodeLen(char src);

    /**
     * @brief 将unicode code pont转为UTF‑8字节序列
     * @param codePoint Unicode Code Point
     * @return UTF-8字符串
     */
    std::string convertUnicodeCptToUTF8(uint32_t codePoint);

    /**
     * @brief 从给定UTF‑8字符串的指定偏移处解码出一个 Unicode Code Point
     * @param utf8 UTF-8字符串
     * @param offset 偏移量
     * @return Unicode Code Point，并把 offset 向前推进已消费的字节数
     */
    uint32_t convertUTF8ToUnicodeCpt(const std::string &utf8, size_t &offset);

    /**
     * @brief 把UTF-8编码的字符串解析为Unicode Code Point列表
     * @param utf8 UTF-8编码的字符串
     * @return 解析后的Unicode Code Point列表
     */
    std::vector<uint32_t> convertUTF8ToUnicodeCpts(const std::string &utf8);

    /**
     * @brief 将Code Point序列按预先生成的NFD（规范分析序列）映射做归一替换
     * @param cpts Code Point序列
     * @return 输入Code Point经过NFD映射表替换后的Code Point(大小不变)
     */
    std::vector<uint32_t> getUnicodeCptsByNormalizeNFD(const std::vector<uint32_t> &cpts);

    /**
     * @brief 返回给定Unicode Code Point的标志，标志描述该Code Point类型（字母/数字/标点/控制等）和一些辅助属性（是否空白、是否小写/大写、是否有 NFD 映射）
     * @param Unicode Code Point
     * @return Unicode Code Point对应的类型
     */
    CodePointFlags getUnicodeCptFlags(const uint32_t cp);

    /**
     * @brief 返回给定UTF-8字符串的标志，标志描述该Code Point类型（字母/数字/标点/控制等）和一些辅助属性（是否空白、是否小写/大写、是否有 NFD 映射）
     * @param UTF-8字符串
     * @return 对应的Code Point类型
     */
    CodePointFlags getUnicodeCptFlags(const std::string &utf8);

    /**
     * @brief 将单字节Uncode映射为UTF-8字符串
     * @param 单字节Uncode码
     * @return 该字节的 UTF‑8 编码字符串
     */
    std::string convertUnicodeByteToUTF8(uint8_t byte);

    /**
     * @brief 将单字节Uncode映射为UTF-8字符串转回单字节Uncode码
     * @param UTF-8字符串
     * @return 单字节Uncode码
     */
    uint8_t convertUTF8ToUnicodeByte(const std::string &utf8);

    /**
     * @brief Unicode 大写到小写转换
     * @param cp Unicode Code Point
     * @return 小写Uncode码
     */
    uint32_t convertUnicodeToLower(uint32_t cp);

    /**
     * @brief splitUnicodeByRegex
     * @param text
     * @param regexExprs
     * @return
     */
    std::vector<std::string> splitUnicodeByRegex(const std::string &text, const std::vector<std::string> &regexExprs);
}

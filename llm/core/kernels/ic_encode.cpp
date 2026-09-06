#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <sentencepiece_processor.h>

#include "absl/strings/str_replace.h"
#include "ic_common.h"
#include "ic_encode.h"
#include "ic_tokenizer.h"
#include "ic_unicode.h"
#include "ic_unordered_dense.h"

namespace kernel
{
    SPEEncodeLayer::SPEEncodeLayer(std::string modelPath, bool bHasBOS, bool bHasEOS)
        : EncodeBaseLayer(std::move(modelPath), bHasBOS, bHasEOS)
    {
        using namespace sentencepiece::util;

        spe = std::make_unique<sentencepiece::SentencePieceProcessor>();
        auto ret = spe->Load(modelPath);
        if (ret.code() != sentencepiece::util::StatusCode::kOk)
        {
            LOG(FATAL) << "The model path is not valid, please check.";
        }
    }

    std::vector<int32_t> SPEEncodeLayer::Encode(const std::string &sentence) const
    {
        CHECK(this->spe != nullptr);

        std::vector<int32_t> inputIds = spe->EncodeAsIds(sentence);
        if (bHasBos)
        {
            inputIds.insert(inputIds.begin(), spe->bos_id());
        }

        if (bHasEos)
        {
            inputIds.push_back(spe->eos_id());
        }

        return inputIds;
    }

    std::string SPEEncodeLayer::Decode(const int32_t tokenId) const
    {
        CHECK(this->spe != nullptr);

        std::vector<int32_t> tokenIds(tokenId);
        return this->spe->DecodeIds(tokenIds);
    }

    std::string SPEEncodeLayer::Decode(const std::vector<int32_t> &tokenIds) const
    {
        CHECK(this->spe != nullptr);

        return this->spe->DecodeIds(tokenIds);
    }

    bool SPEEncodeLayer::IsSentenceEnding(const int32_t tokenId) const
    {
        CHECK(this->spe != nullptr);

        return tokenId == this->spe->eos_id();
    }

    int32_t SPEEncodeLayer::GetVocabSize() const
    {
        CHECK(this->spe != nullptr);

        return this->spe->GetPieceSize();
    }

    static const std::string PAT_STR =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

    BPEEncodeLayer::BPEEncodeLayer(std::string modelPath, bool bHasBOS, bool bHasEOS)
        : EncodeBaseLayer(std::move(modelPath), bHasBOS, bHasEOS)
    {
        using json = nlohmann::json;

        std::ifstream ifs(modelPath);
        CHECK(ifs.is_open()) << "The model path is not valid, please check the path and type of model.";

        json data;
        try
        {
            data = json::parse(ifs);
        }
        catch (json::parse_error &)
        {
            LOG(FATAL) << "The token model path is not valid, please check the path and type of model.";
        }

        const auto &datas = data["added_tokens"];
        ankerl::unordered_dense::map<std::string, int> specialTokens;
        for (const auto &item : datas)
        {
            int id = item["id"];
            std::string content = item["content"];
            specialTokens.insert({content, id});
        }

        ankerl::unordered_dense::map<std::string, int> encoder;
        const auto &vocabs = data["model"]["vocab"];
        const auto &vocab_items = vocabs.items();
        for (const auto &item : vocab_items)
        {
            std::string key;
            const auto cpts = convertUTF8ToUnicodeCpts(item.key());
            for (const auto cpt : cpts)
            {
                const auto utf8 = convertUnicodeCptToUTF8(cpt);
                key += convertUTF8ToUnicodeByte(utf8);
            }

            const int32_t id = item.value();
            encoder[key] = id;
        }

        bosId = specialTokens["<|begin_of_text|>"];
        eosId = specialTokens["<|end_of_text|>"];
        stopToken1 = eosId;
        stopToken2 = specialTokens["<|eot_id|>"];

        tokenNum = encoder.size() + specialTokens.size();
        tikToken = std::make_unique<tiktoken::TikToken>(encoder, specialTokens, PAT_STR);
    }

    std::vector<int32_t> BPEEncodeLayer::Encode(const std::string &sentence) const
    {
        CHECK(this->tikToken != nullptr);

        std::map<std::string, std::string> replacements;
        replacements[" "] = "Ġ";

        std::string str = absl::StrReplaceAll(sentence, replacements);
        auto inputIds = this->tikToken->encode(str);
        if (bHasBos)
        {
            inputIds.insert(inputIds.begin(), bosId);
        }

        if (bHasEos)
        {
            inputIds.push_back(eosId);
        }

        return inputIds;
    }

    std::string BPEEncodeLayer::Decode(const int32_t tokenId) const
    {
        return "";
    }

    std::string BPEEncodeLayer::Decode(const std::vector<int32_t> &tokenIds) const
    {
        CHECK(this->tikToken != nullptr);

        std::map<std::string, std::string> reverseReplacements;
        reverseReplacements["Ġ"] = " ";

        auto str = this->tikToken->decode(tokenIds);
        const std::string &sentence = absl::StrReplaceAll(str, reverseReplacements);
        return sentence;
    }

    bool BPEEncodeLayer::IsSentenceEnding(const int32_t tokenId) const
    {
        if (tokenId == stopToken1 || tokenId == stopToken2)
        {
            return true;
        }

        return false;
    }

    int32_t BPEEncodeLayer::GetVocabSize() const
    {
        CHECK(this->tikToken != nullptr);
        return tokenNum;
    }

    QwenEncodeLayer::QwenEncodeLayer(const std::string &modelPath, bool bHasBOS, bool bHasEOS)
        : BPEEncodeLayer(std::move(modelPath), bHasBOS, bHasEOS)
    {
        using json = nlohmann::json;
        std::ifstream ifs(modelPath);

        json data;
        try
        {
            data = json::parse(ifs);
        }
        catch (json::parse_error &)
        {
            LOG(FATAL) << "The token model path is not valid, please check the path and type of model.";
        }

        const auto &datas = data["added_tokens"];
        ankerl::unordered_dense::map<std::string, int> specialTokens;
        for (const auto &item : datas)
        {
            int id = item["id"];
            std::string content = item["content"];
            specialTokens.insert({content, id});
        }

        ankerl::unordered_dense::map<std::string, int> encoder;
        const auto &vocabs = data["model"]["vocab"];
        const auto &vocab_items = vocabs.items();
        for (const auto &item : vocab_items)
        {
            const auto cpts = convertUTF8ToUnicodeCpts(item.key());
            std::string key;
            for (const auto cpt : cpts)
            {
                const auto utf8 = convertUnicodeCptToUTF8(cpt);
                key += convertUTF8ToUnicodeByte(utf8);
            }

            const int32_t id = item.value();
            encoder[key] = id;
        }

        bosId = specialTokens["<|im_start|>"];
        eosId = specialTokens["<|im_end|>"];
        stopToken1 = eosId;
        stopToken2 = specialTokens["<|endoftext|>"];

        tokenNum = encoder.size() + specialTokens.size();
        tikToken = std::make_unique<tiktoken::TikToken>(encoder, specialTokens, PAT_STR);
    }
} // namespace kernel
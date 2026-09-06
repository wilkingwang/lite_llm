#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>

#include "ic_base_layer.h"
#include "ic_common.h"
#include "ic_layer.h"

#include "ic_tokenizer.h"
#include "ic_unordered_dense.h"

namespace kernel
{
    class EncodeBaseLayer : Layer
    {
      public:
        explicit EncodeBaseLayer(std::string modelPath, bool bHasBos, bool bHasEos)
            : Layer(model::DeviceType::iDeviceCPU, LayerType::iLayerEncode, "Encode"), bHasBos(bHasBos),
              bHasEos(bHasEos), modelPath(std::move(modelPath))
        {
        }

        virtual std::vector<int32_t> Encode(const std::string &sentence) const = 0;

        virtual std::string Decode(const int32_t tokenId) const = 0;

        virtual std::string Decode(const std::vector<int32_t> &tokenIds) const = 0;

        virtual bool IsSentenceEnding(const int32_t tokenId) const = 0;

        virtual int32_t GetVocabSize() const = 0;

      protected:
        bool bHasBos = true;
        bool bHasEos = true;
        std::string modelPath;
    };

    class SPEEncodeLayer : EncodeBaseLayer
    {
      public:
        explicit SPEEncodeLayer(std::string modelPath, bool bHasBOS, bool bHasEOS);

        std::vector<int32_t> Encode(const std::string &sentence) const override;

        std::string Decode(const int32_t tokenId) const override;

        std::string Decode(const std::vector<int32_t> &tokenIds) const override;

        bool IsSentenceEnding(const int32_t tokenId) const override;

        int32_t GetVocabSize() const override;

      private:
        std::unique_ptr<sentencepiece::SentencePieceProcessor> spe;
    };

    class BPEEncodeLayer : EncodeBaseLayer
    {
      public:
        BPEEncodeLayer(std::string modelPath, bool bHasBOS, bool bHasEOS);

        std::vector<int32_t> Encode(const std::string &sentence) const override;

        std::string Decode(const int32_t tokenId) const override;

        std::string Decode(const std::vector<int32_t> &tokenIds) const override;

        bool IsSentenceEnding(const int32_t tokenId) const override;

        int32_t GetVocabSize() const override;

      protected:
        int32_t bosId = -1;
        int32_t eosId = -1;
        int32_t stopToken1 = -1;
        int32_t stopToken2 = -1;
        int32_t tokenNum = 0;
        std::unique_ptr<tiktoken::TikToken> tikToken;
    };

    class QwenEncodeLayer : BPEEncodeLayer
    {
      public:
        explicit QwenEncodeLayer(const std::string &modelPath, bool bHasBOS, bool bHasEOS);
    };
} // namespace kernel
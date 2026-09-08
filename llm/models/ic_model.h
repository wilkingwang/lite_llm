#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ic_common.h"
#include "ic_config.h"
#include "ic_embedding.h"
#include "ic_encode.h"
#include "ic_model_weight.h"
#include "ic_sampler.h"
#include "ic_tensor.h"

namespace model
{
    class ICModel
    {
      public:
        explicit ICModel(model::TokenizerType tokenizerType, model::ModelType modelType, std::string tokenizerPath,
                         std::string modelPath, bool bPrompt);

        virtual comm::Status Init(model::DeviceType deviceType) = 0;

        virtual comm::Status Predict(const tensor::Tensor &input, const tensor::Tensor &positionTensor, bool bPrompt,
                                     int &next) const = 0;
        virtual comm::Status Forward(const tensor::Tensor &input, const tensor::Tensor &positionTensor,
                                     int &next) const = 0;

        model::ModelType GetModelType() const;

        const std::string GetTokenizerPath() const;

        const std::string getModelPath() const;

        virtual tensor::Tensor &GetBuffer(comm::BufferType bufferIdx);

        virtual const tensor::Tensor &GetBuffer(comm::BufferType bufferIdx) const;

        virtual bool isSentenceEnding(int32_t tokenId) const;

        virtual std::string Decode(int32_t tokenId) const;

        virtual std::string Decode(std::vector<int32_t> &tokenIds) const;

        virtual std::vector<int32_t> Encode(const std::string &sentence) const;

        virtual std::pair<tensor::Tensor, tensor::Tensor> SliceKVCache(const int32_t layerIdx,
                                                                       const int32_t tokenPos) const;

        virtual kernel::ICEmbeddingOutput Embedding(const std::vector<int> &tokenIds) const = 0;

        virtual tensor::Tensor FillInput(const tensor::Tensor &positionTensor,
                                         const kernel::ICEmbeddingOutput &embeddingOutput, bool bPrompt) const;

      protected:
        virtual comm::Status insertBuffer(comm::BufferType bufferIdx, const tensor::Tensor &tensor);

        virtual comm::Status readModelFile();

        virtual comm::Status createEncodeLayer();

        virtual comm::Status genModelFromFile();

        virtual comm::Status generateModelInfo(const ModelConfig &modelConfig) const;

        virtual int32_t postProcessing(const tensor::Tensor &position, bool bPrompt) const = 0;

      private:
        virtual void initMem() = 0;

        virtual comm::StatusCode createLayers() = 0;

        virtual void createParamLayers() = 0;

        virtual void createNonParamLayers() = 0;

        virtual void createParamQuantLayers() = 0;

      protected:
        std::string modelPath;
        std::string tokenizerPath;

        int32_t groupSize = 1;
        bool bQuantMode = false;

        std::unique_ptr<TransformerConfig> config;
        std::unique_ptr<kernel::EncodeBaseLayer> pEncodeLayer;
        std::map<comm::BufferType, tensor::Tensor> mpBufferList;
        std::unique_ptr<sampler::ICSampler> pSampler;
        std::shared_ptr<ICRawModelData> pRawModelData;

        model::DeviceType deviceType = model::DeviceType::iDeviceUnknown;
        model::ModelType modelType = model::ModelType::iModelTypeUnknown;
        model::TokenizerType tokenizerType = model::TokenizerType::iEncodeUnknown;
    };
} // namespace model
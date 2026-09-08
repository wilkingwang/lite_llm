#pragma once

#include "ic_common.h"
#include "ic_layer.h"
#include "ic_tensor.h"
#include <cstdint>
#include <utility>

namespace kernel
{
    class ICEmbeddingOutput
    {
      public:
        tensor::Tensor inputTokens;
        tensor::Tensor inputEmbeddings;
        tensor::Tensor inputTokenNum;

        explicit ICEmbeddingOutput(tensor::Tensor inputTokens, tensor::Tensor inputEmbeddings,
                                   tensor::Tensor inputTokenNum)
            : inputTokens(std::move(inputTokens)), inputEmbeddings(std::move(inputEmbeddings)),
              inputTokenNum(std::move(inputTokenNum))
        {
        }
    };

    class ICEmbeddingLayer : public LayerParam
    {
        explicit ICEmbeddingLayer(model::DeviceType deviceType, int32_t dim, int32_t seqLen, int32_t vocabSize);

        comm::Status Check() const override;

        comm::Status Forward() override;

      private:
        int32_t dim = 0;
        int32_t seqLen = 0;
        int32_t vocabSize = 0;
    };
} // namespace kernel
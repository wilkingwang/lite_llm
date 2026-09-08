#include "ic_embedding.h"
#include "ic_base_layer.h"
#include "ic_common.h"
#include "ic_layer.h"

namespace model
{
    ICEmbeddingLayer::ICEmbeddingLayer(model::DeviceType deviceType, int32_t dim, int32_t seqLen, int32_t vocabSize)
        : dim(dim), seqLen(seqLen), vocabSize(vocabSize),
          LayerParam(deviceType, LayerType::iLayerEmbedding, false, "Embeddnig")
    {
        ResetWeightSize(1);
        ResetInputSize(2);
        ResetOutputSize(1);
    }

    comm::Status ICEmbeddingLayer::Check() const
    {
        const auto &inputTensor = GetInput(0);
        const auto &tokenSize = GetInput(1).Size();

        if (tokenSize > inputTensor.Size())
        {
            return comm::InvalidArgument("The number of input tensor is greater than seq len.");
        }

        comm::Status status = CheckTensorWithDim(GetWeight(0), deviceType, dataType, vocabSize, dim);
        if (!status)
        {
            LOG(ERROR) << "The model weight tensor invalid in the embedding layer.";
            return status;
        }

        status = CheckTensorWithDim(GetOutput(0), deviceType, dataType, tokenSize, dim);
        if (!status)
        {
            LOG(ERROR) << "The output tensor invalid in the embedding layer.";
            return status;
        }

        return comm::Success();
    }

    comm::Status ICEmbeddingLayer::Forward()
    {
        comm::Status status = Check();
        if (!status)
        {
            return status;
        }

        if (deviceType == model::DeviceType::iDeviceCUDA)
        {
            CHECK(cudaConfig != nullptr);
        }

        // TODO

        return comm::Success();
    }
} // namespace model
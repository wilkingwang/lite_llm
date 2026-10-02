#pragma once
#include "ic_cuda_config.h"
#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void MHA(const tensor::Tensor &mha, const tensor::Tensor &query, const tensor::Tensor &score,
                 const tensor::Tensor &keyCache, const tensor::Tensor &valueCache, const int32_t pos,
                 const int32_t headNum, const int32_t layerIdx, const int32_t seqLen, const int32_t kvDim,
                 const int32_t kvMul, const int32_t headSize, model::DeviceType deviceType, CudaConfig *config);
    }
} // namespace kernel
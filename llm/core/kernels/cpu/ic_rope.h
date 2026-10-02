#pragma once
#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void CalcSinCosCache(const int headSize, const int maxSeqLen, float *sinCache, float *cosCache);
        void RoPE(const int dim, const int32_t kvDim, const int32_t headSize, const tensor::Tensor &inputQ,
                  const tensor::Tensor &inputK, const tensor::Tensor &inputPos, const tensor::Tensor &sinCache,
                  const tensor::Tensor &cosCache, void *stream);
    } // namespace cpu
} // namespace kernel
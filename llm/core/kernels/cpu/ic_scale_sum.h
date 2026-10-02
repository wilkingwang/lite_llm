#pragma once

#include "ic_tensor.h"
namespace kernel
{
    namespace cpu
    {
        void ScaleSum(const tensor::Tensor &value, const tensor::Tensor &scale, const tensor::Tensor &output, int pos,
                      int size, int sride, void *stream = nullptr);
    }
} // namespace kernel
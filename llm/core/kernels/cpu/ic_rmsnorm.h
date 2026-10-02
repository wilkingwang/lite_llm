#pragma once

#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void RMSNorm(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                     void *stream);
    }
} // namespace kernel
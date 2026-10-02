#pragma once
#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void Add(const tensor::Tensor &input1, const tensor::Tensor &input2, const tensor::Tensor &output,
                 void *stream = nullptr);
    }

} // namespace kernel
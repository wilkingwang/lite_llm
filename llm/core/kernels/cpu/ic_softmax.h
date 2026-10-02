#pragma once
#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void Softmax(const tensor::Tensor &input, void *stream = nullptr);
    }
} // namespace kernel
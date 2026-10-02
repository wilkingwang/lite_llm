#pragma once
#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void Scale(float scale, const tensor::Tensor &tensor, void *stream = nullptr);
    }
} // namespace kernel
#pragma once
#include "ic_cuda_config.h"
#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void Matmul(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                    const CudaConfig *config, float scale);
    }
} // namespace kernel
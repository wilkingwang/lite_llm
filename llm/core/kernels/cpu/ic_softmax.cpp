#include "ic_softmax.h"
#include "armadillo"
#include "ic_common.h"
#include "ic_mem_buffer.h"
#include "ic_tensor.h"
#include <algorithm>
#include <cstdint>
#include <memory>

namespace kernel
{
    namespace cpu
    {
        void Softmax(const tensor::Tensor &input, void *stream)
        {
            int size = static_cast<int32_t>(input.Size());
            const float *inputPtr = input.Ptr<float>();
            float maxValue = *std::max_element(inputPtr, inputPtr + size);

            arma::fvec inputMat(const_cast<float *>(inputPtr), size, false, true);
            inputMat = arma::exp(inputMat - maxValue);

            float sum = arma::sum(inputMat);
            inputMat = inputMat / sum;
        }

        void Softmax(const float *inputPtr, size_t size)
        {
            tensor::Tensor input(model::DataType::iDataTypeFP32, size);
            std::shared_ptr<comm::MemBuffer> buffer =
                std::make_shared<comm::MemBuffer>(size * sizeof(float), nullptr, (void *)inputPtr, true);

            input.Assign(buffer);
            return Softmax(input);
        }
    } // namespace cpu
} // namespace kernel
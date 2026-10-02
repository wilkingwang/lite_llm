#include <cstdint>

#include "ic_common.h"
#include "ic_embedding.h"
#include "ic_memory.h"

namespace kernel
{
    namespace cpu
    {
        void Embedding(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                       int32_t vocabSize, void *stream)
        {
            CHECK_EQ(input.IsEmpty(), false);
            CHECK_EQ(weight.IsEmpty(), false);

            const int32_t inputNum = static_cast<int32_t>(input.Size());
            const int32_t weightDim = weight.GetDim(1);
            CHECK(weight.GetDeviceType() == output.GetDeviceType());
            CHECK(input.GetDeviceType() == model::DeviceType::iDeviceCPU);

            const auto allocator = comm::CPUMemoryAllocatorFactory::getInstance();
            for (int32_t i = 0; i < inputNum; i++)
            {
                int32_t token = *input.Ptr<int32_t>(i);
                if (token > vocabSize)
                {
                    LOG(FATAL) << "The input token index is greater than vocab size.";
                    continue;
                }

                if (weight.GetDeviceType() != model::DeviceType::iDeviceCPU)
                {
                    LOG(FATAL) << "The device type of weight in the embedding layer is not on cpu.";
                    continue;
                }

                float *outputPtr = const_cast<float *>(output.Ptr<float>(i * weightDim));
                float *embPtr = const_cast<float *>(weight.Ptr<float>(token * weightDim));
                allocator->memcpy(embPtr, outputPtr, weightDim * sizeof(float), comm::MemcpyKind::iMemcpyCPU2CPU);
            }
        }
    } // namespace cpu
} // namespace kernel
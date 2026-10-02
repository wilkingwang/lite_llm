#include "ic_matmul.h"
#include "armadillo"
#include "ic_common.h"
#include <cstdint>

namespace kernel
{
    namespace cpu
    {
        void Matmul(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                    const CudaConfig *config, float scale)
        {
            CHECK_EQ(input.IsEmpty(), false);
            CHECK_EQ(weight.IsEmpty(), false);
            CHECK_EQ(output.IsEmpty(), false);
            CHECK(input.GetDeviceType() == model::DeviceType::iDeviceCPU);
            CHECK(weight.GetDeviceType() == model::DeviceType::iDeviceCPU);
            CHECK(output.GetDeviceType() == model::DeviceType::iDeviceCPU);

            CHECK_EQ(weight.GetDimsSize(), 2);

            const float *inputPtr = input.Ptr<float>();
            const float *outputPtr = output.Ptr<float>();
            const float *weightPtr = weight.Ptr<float>();

            int32_t inputDim0 = 1;
            int32_t inputDim1 = 1;
            switch (input.GetDimsSize())
            {
            case 1:
            {
                inputDim0 = input.GetDim(0);
                inputDim1 = input.GetDim(1);
            }
            break;
            case 2:
            {
                inputDim0 = input.GetDim(0);
            }
            break;
            default:
                LOG(FATAL) << "The input tensor dim is invalid.";
            }

            int32_t weightDim0 = weight.GetDim(0);
            int32_t weightDim1 = weight.GetDim(1);
            CHECK_EQ(inputDim0, weightDim1);

            CHECK_EQ(output.Size(), weightDim0 * inputDim1);
            arma::fmat inputMat(const_cast<float *>(inputPtr), inputDim1, inputDim0, false, true);
            arma::fmat weightMat(const_cast<float *>(weightPtr), weightDim1, weightDim0, false, true);
            arma::fmat outputMat(const_cast<float *>(outputPtr), inputDim1, weightDim0, false, true);

            outputMat = ((inputMat * weightMat)) * scale;
        }
    } // namespace cpu
} // namespace kernel
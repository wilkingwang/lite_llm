#include "ic_rmsnorm.h"
#include "armadillo"
#include "ic_common.h"
#include <cmath>
#include <cstdint>

namespace kernel
{
    namespace cpu
    {
        void RMSNorm(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                     void *stream)
        {
            CHECK_EQ(input.IsEmpty(), false);
            CHECK_EQ(weight.IsEmpty(), false);
            CHECK_EQ(output.IsEmpty(), false);

            CHECK(input.GetDeviceType() == model::DeviceType::iDeviceCPU &&
                  weight.GetDeviceType() == model::DeviceType::iDeviceCPU &&
                  output.GetDeviceType() == model::DeviceType::iDeviceCPU);

            const float *inputPtr = input.Ptr<float>();
            const float *weightPtr = weight.Ptr<float>();
            const float *outputPtr = output.Ptr<float>();
            const int32_t dim = static_cast<int32_t>(input.Size());

            arma::fmat inputMat(const_cast<float *>(inputPtr), dim, false, true);
            arma::fmat outputMat(const_cast<float *>(outputPtr), dim, false, true);
            arma::fmat weightMat(const_cast<float *>(weightPtr), dim, false, true);

            const float eps = 1e-5f;
            const float mean = arma::as_scalar(arma::mean(arma::pow(inputMat, 2))) + eps;
            const float rsqrt = 1.f / std::sqrt(mean);
            outputMat = weightMat % (rsqrt * inputMat);
        }
    } // namespace cpu
} // namespace kernel
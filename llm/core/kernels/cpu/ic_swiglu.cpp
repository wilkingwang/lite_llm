
#include "ic_swiglu.h"
#include "armadillo"

namespace kernel
{
    namespace cpu
    {
        void Swiglu(const tensor::Tensor &input1, const tensor::Tensor &input2, const tensor::Tensor &output,
                    void *stream)
        {
            UNUSED(stream);
            CHECK_EQ(input1.IsEmpty(), false);
            CHECK_EQ(input2.IsEmpty(), false);
            CHECK_EQ(output.IsEmpty(), false);

            CHECK(input1.GetDeviceType() == model::DeviceType::iDeviceCPU);
            CHECK(input2.GetDeviceType() == model::DeviceType::iDeviceCPU);
            CHECK(output.GetDeviceType() == model::DeviceType::iDeviceCPU);

            arma::fvec input1_vec(const_cast<float *>(input1.Ptr<float>()), input1.Size(), false, true);
            arma::fvec input2_vec(const_cast<float *>(input2.Ptr<float>()), input2.Size(), false, true);
            arma::fvec output_vec(const_cast<float *>(output.Ptr<float>()), output.Size(), false, true);

            input1_vec %= (1.0f / (1.0f + arma::exp(-input1_vec)));
            output_vec = input1_vec % input2_vec;
        }
    } // namespace cpu
} // namespace kernel
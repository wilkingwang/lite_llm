

#include "armadillo"
#include "ic_common.h"
#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void ScaleSum(const tensor::Tensor &value, const tensor::Tensor &scale, const tensor::Tensor &output, int pos,
                      int size, int stride, void *stream = nullptr)
        {
            UNUSED(stream);

            CHECK_EQ(value.IsEmpty(), false);
            CHECK_EQ(scale.IsEmpty(), false);
            CHECK_EQ(output.IsEmpty(), false);
            CHECK_EQ(size, value.Size());
            CHECK_EQ(size, output.Size());

            arma::fvec scaleVec(const_cast<float *>(scale.Ptr<float>()), scale.Size(), false, true);
            arma::fvec outputVec(const_cast<float *>(output.Ptr<float>()), output.Size(), false, true);

            for (int i = 0; i <= pos; i++)
            {
                arma::fvec valueVec(const_cast<float *>(value.Ptr<float>()) + i * stride, value.Size(), false, true);
                outputVec += scaleVec[i] * valueVec;
            }
        }
    } // namespace cpu
} // namespace kernel
#include <armadillo>

#include "ic_add.h"
#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void Add(const tensor::Tensor &input1, const tensor::Tensor &input2, const tensor::Tensor &output, void *stream)
        {
            CHECK_EQ(input1.IsEmpty(), false);
            CHECK_EQ(input2.IsEmpty(), false);
            CHECK_EQ(output.IsEmpty(), false);

            CHECK_EQ(input1.Size(), input2.Size());
            CHECK_EQ(input1.Size(), output.Size());

            arma::fvec inputVec1(const_cast<float *>(input1.Ptr<float>()), input1.Size(), false, true);
            arma::fvec inputVec2(const_cast<float *>(input2.Ptr<float>()), input2.Size(), false, true);
            arma::fvec outputVec(const_cast<float *>(output.Ptr<float>()), output.Size(), false, true);

            outputVec = inputVec1 + inputVec2;
        }
    } // namespace cpu
} // namespace kernel
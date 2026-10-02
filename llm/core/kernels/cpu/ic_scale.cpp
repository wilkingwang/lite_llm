#include "ic_scale.h"
#include "armadillo"
#include "ic_common.h"
#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void Scale(float scale, const tensor::Tensor &tensor, void *stream)
        {
            UNUSED(stream);
            CHECK_NE(tensor.IsEmpty(), true);

            arma::fvec tensorMat(const_cast<float *>(tensor.Ptr<float>()), tensor.Size(), false, true);
            tensorMat = tensorMat * scale;
        }
    } // namespace cpu
} // namespace kernel
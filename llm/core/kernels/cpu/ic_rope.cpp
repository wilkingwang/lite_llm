#include "ic_rope.h"
#include "ic_common.h"
#include <cmath>
#include <cstdint>

namespace kernel
{
    namespace cpu
    {

        void CalcSinCosCache(const int headSize, const int maxSeqLen, float *sinCache, float *cosCache)
        {
            for (int pos = 0; pos < maxSeqLen; ++pos)
            {
                for (int headDim = 0; headDim < headSize; ++headDim)
                {
                    float freq = 1.f / std::pow(500000.0f, static_cast<float>(headDim) / static_cast<float>(headSize));
                    float val = static_cast<float>(pos) * freq;
                    float fcr = cosf(val);
                    float fci = sinf(val);

                    *(sinCache + pos * headSize + headDim) = fci;
                    *(cosCache + pos * headSize + headDim) = fcr;
                }
            }
        }

        void RoPE(const int dim, const int32_t kvDim, const int32_t headSize, const tensor::Tensor &inputQ,
                  const tensor::Tensor &inputK, const tensor::Tensor &inputPos, const tensor::Tensor &sinCache,
                  const tensor::Tensor &cosCache, void *stream)
        {
            UNUSED(stream);
            const int32_t pos = *inputPos.Ptr<float>(0);

            for (int32_t i = 0; i < dim; i += headSize)
            {
                for (int32_t headDim = i % headSize; headDim < headSize / 2; headDim++)
                {
                    float fci = *(sinCache.Ptr<float>() + pos * headSize + headDim * 2);
                    float fcr = *(cosCache.Ptr<float>() + pos * headSize + headDim * 2);

                    int rotn = i < kvDim ? 2 : 1;
                    for (int32_t v = 0; v < rotn; v++)
                    {
                        float *vec = const_cast<float *>(v == 0 ? inputQ.Ptr<float>() : inputK.Ptr<float>());
                        float v0 = vec[i + headDim];
                        float v1 = vec[i + headDim + headSize / 2];

                        vec[i + headDim] = v0 * fcr - v1 * fci;
                        vec[i + headDim + headSize / 2] = v0 * fci + v1 * fcr;
                    }
                }
            }
        }
    } // namespace cpu
} // namespace kernel
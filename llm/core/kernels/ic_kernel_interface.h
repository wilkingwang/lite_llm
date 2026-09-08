#pragma once

#include "ic_common.h"
#include "ic_cuda_config.h"
#include "ic_tensor.h"
#include <cstddef>
#include <cstdint>

namespace kernel
{
    /**
     * @brief
     * @param
     * @param
     * @param
     * @param
     */
    typedef void (*Add)(const tensor::Tensor &input1, const tensor::Tensor &input2, const tensor::Tensor &output,
                        void *stream);

    /**
     * @brief
     * @param
     * @param
     * @param
     * @param
     */
    typedef void (*Matmul)(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                           const CudaConfig *config, float scale);

    /**
     * @brief
     * @param
     * @param
     * @param
     * @param
     */
    typedef void (*MatmulQuant)(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                                const tensor::Tensor &scale, const CudaConfig *config, int32_t groupSize);

    /**
     * @brief
     * @param
     * @param
     * @param
     * @param
     */
    typedef void (*Embedding)(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                              int32_t vocabSize, void *stream);

    /**
     * @brief
     * @param
     * @param
     * @param
     * @param
     */
    typedef void (*Swiglu)(const tensor::Tensor &input1, const tensor::Tensor &weight, const tensor::Tensor &output,
                           void *stream);

    /**
     * @brief
     * @param
     * @param
     * @param
     * @param
     */
    typedef void (*MHA)(const tensor::Tensor &mha, const tensor::Tensor &query, const tensor::Tensor &score,
                        const tensor::Tensor &keyCache, const tensor::Tensor &valueCache, const int32_t pos,
                        const int32_t headNum, const int32_t layerIdx, const int32_t seqLen, const int32_t kvDim,
                        const int32_t kvMul, const int32_t headSize, model::DeviceType deviceType, CudaConfig *config);

    typedef void (*RMSNorm)(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                            void *stream);

    typedef void (*RMSNormDim)(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                               int32_t dim, void *stream);

    typedef void (*RoPE)(const tensor::Tensor &input_q, const tensor::Tensor &input_k, const tensor::Tensor &inut_pos,
                         const tensor::Tensor &sinCache, const tensor::Tensor &cosCache, const int32_t dim,
                         const int32_t kvDim, const int32_t headSize, void *stream);

    typedef void (*Scale)(const tensor::Tensor &input, const float scale, void *stream);

    typedef void (*SoftmaxCuda)(const tensor::Tensor &input, void *stream);

    typedef void (*ScaleSum)(const tensor::Tensor &value, const tensor::Tensor &scale, const tensor::Tensor &output,
                             const int t, const int size, const int stride, void *stream);

    void SoftmaxCPU(const float *pInput, size_t size);

    Add GetAddKernel(model::DeviceType deviceType);

    Embedding GetEmbeddingKernel(model::DeviceType deviceType);

    Matmul GetMatmulKernel(model::DeviceType deviceType);

    MatmulQuant GetMatmulQuantKernel(model::DeviceType deviceType);

    MHA GetMHAKernel(model::DeviceType deviceType);

    RMSNorm GetRMSNormKernel(model::DeviceType deviceType);

    RoPE GetRoPEKernel(model::DeviceType deviceType);

    Scale GetScaleKernel(model::DeviceType deviceType);

    SoftmaxCuda GetSoftmaxKernle(model::DeviceType deviceType);

    Swiglu GetSwigluKernel(model::DeviceType deviceType);

    ScaleSum GetScaleSumKernel(model::DeviceType deviceType);

    RMSNormDim GetRMSNormDim(model::DeviceType deviceType);

} // namespace kernel
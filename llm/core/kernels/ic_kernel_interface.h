#pragma once
#include <cstddef>
#include <cstdint>

#include "ic_common.h"
#include "ic_cuda_config.h"
#include "ic_tensor.h"

namespace kernel
{
    /**
     * @brief 张量相加Kernel
     * @param input1 左操作数张量
     * @param input2 右操作数张量
     * @param output 输出结果张量
     * @param stream CUDA 流或平台相关的异步执行句柄
     */
    typedef void (*Add)(const tensor::Tensor &input1, const tensor::Tensor &input2, const tensor::Tensor &output,
                        void *stream);

    /**
     * @brief 浮点矩阵乘法
     * @param input 输入张量（通常为激活值/特征矩阵）
     * @param weight 权重矩阵张量
     * @param output 输出张量
     * @param config CUDA配置
     * @param scale 缩放因子
     */
    typedef void (*Matmul)(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                           const CudaConfig *config, float scale);

    /**
     * @brief 量化矩阵乘法（分组量化支持）
     * @param input 输入张量（通常为激活值/特征矩阵）
     * @param weight 量化后的权重张量
     * @param output 输出张量
     * @param scale 每组或每列的缩放因子张量
     * @param config CUDA配置
     * @param groupSize 每组的大小（用于分组量化）
     */
    typedef void (*MatmulQuant)(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                                const tensor::Tensor &scale, const CudaConfig *config, int32_t groupSize);

    /**
     * @brief Embedding
     * @param input 输入索引张量（通常为 int 类型的 token id）
     * @param weight Embedding矩阵（vocab_size x dim）
     * @param output output 输出张量（嵌入向量）
     * @param vocabSize vocab_size 词表大小
     * @param stream CUDA stream
     */
    typedef void (*Embedding)(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                              int32_t vocabSize, void *stream);

    /**
     * @brief SwiGLU 等分支激活计算
     * @param input1 第一个张量
     * @param input2 第二个张量
     * @param output 输出张量
     * @param stream CUDA stream
     */
    typedef void (*Swiglu)(const tensor::Tensor &input1, const tensor::Tensor &input2, const tensor::Tensor &output,
                           void *stream);

    /**
     * @brief 多头注意力（MHA）内核函数
     * @param mha MHA注意力输出张量
     * @param query Query张量
     * @param score Attention Score张量
     * @param keyCache Key缓存张量
     * @param valueCache Value缓存张量
     * @param pos 当前处理的位置索引
     * @param headNum 头数量
     * @param layerIdx 层索引
     * @param seqLen 当前序列长度
     * @param kvDim Key/Value的维度
     * @param kvMul 用于控制Key/Value存储布局
     * @param headSize 每个头的维度
     * @param deviceType 设别类型
     * @param config CUDA stream
     */
    typedef void (*MHA)(const tensor::Tensor &mha, const tensor::Tensor &query, const tensor::Tensor &score,
                        const tensor::Tensor &keyCache, const tensor::Tensor &valueCache, const int32_t pos,
                        const int32_t headNum, const int32_t layerIdx, const int32_t seqLen, const int32_t kvDim,
                        const int32_t kvMul, const int32_t headSize, model::DeviceType deviceType, CudaConfig *config);

    /**
     * @brief RMSNorm 内核函数
     * @param input 输入张量
     * @param weight 归一化权重张量
     * @param output 输出张量
     * @param stream CUDA stream
     */
    typedef void (*RMSNorm)(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                            void *stream);

    /**
     * @brief RMSNorm 内核函数（带显式维度参数）
     * @param input 输入张量
     * @param weight 归一化权重张量
     * @param output 输出张量
     * @param stream CUDA stream
     */
    typedef void (*RMSNormDim)(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output,
                               int32_t dim, void *stream);

    /**
     * @brief RoPE（Rotary Positional Embedding）
     * @param input_q Query输入张量
     * @param input_k Key输入张量
     * @param inut_pos 位置索引
     * @param sinCache 值缓存张量
     * @param cosCache 值缓存张量
     * @param dim 特征维度
     * @param kvDim Key/Value维度
     * @param headSize 单个头的维度
     * @param stream CUDA stream
     */
    typedef void (*RoPE)(const tensor::Tensor &input_q, const tensor::Tensor &input_k, const tensor::Tensor &inut_pos,
                         const tensor::Tensor &sinCache, const tensor::Tensor &cosCache, const int32_t dim,
                         const int32_t kvDim, const int32_t headSize, void *stream);

    /**
     * @brief 对输入按标量进行缩放的内核函数
     * @param input 要缩放的输入张量
     * @param scale 缩放因子
     * @param stream CUDA stream
     */
    typedef void (*Scale)(const tensor::Tensor &input, const float scale, void *stream);

    /**
     * @brief softmax 内核函数
     * @param input 输入张量
     * @param stream CUDA stream
     */
    typedef void (*SoftmaxCuda)(const tensor::Tensor &input, void *stream);

    /**
     * @brief 按比例相加/累加的内核函数
     * @param value 被加的值张量
     * @param scale 缩放系数张量
     * @param output 输出张量
     * @param t 时间步或索引
     * @param size 处理长度
     * @param stride 内存步幅
     * @param stream CUDA stream
     */
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
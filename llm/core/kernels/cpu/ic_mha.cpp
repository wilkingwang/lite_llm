#include <cmath>
#include <cstdint>
#include <memory>

#include "ic_common.h"
#include "ic_memory.h"
#include "ic_mha.h"
#include "ic_tensor.h"

namespace kernel
{
    namespace cpu
    {
        void MHA(const tensor::Tensor &mha, const tensor::Tensor &query, const tensor::Tensor &score,
                 const tensor::Tensor &keyCache, const tensor::Tensor &valueCache, const int32_t pos,
                 const int32_t headNum, const int32_t layerIdx, const int32_t seqLen, const int32_t kvDim,
                 const int32_t kvMul, const int32_t headSize, model::DeviceType deviceType, CudaConfig *config)
        {
            int32_t layerOffset = layerIdx * seqLen * kvDim;
            float scale = 1.f / std::sqrt(static_cast<float>(headSize));

            std::shared_ptr<comm::MemoryAllocator> allocator;
            if (deviceType == model::DeviceType::iDeviceCPU)
            {
                allocator = comm::CPUMemoryAllocatorFactory::getInstance();
            }
            else
            {
                allocator = comm::CUDAMemoryAllocatorFactory::getInstance();
            }

            for (int32_t i = 0; i < headNum; i++)
            {
                float *scoreHeadAddr = const_cast<float *>(score.Ptr<float>() + i * seqLen);
                float *queryHeadAddr = const_cast<float *>(query.Ptr<float>() + i * headSize);

                tensor::Tensor queryMat(model::DataType::iDataTypeFP32, headSize, false, nullptr, queryHeadAddr);
                queryMat.SetDeviceType(deviceType);

                for (int32_t j = 0; j <= pos; j++)
                {
                    int32_t cacheOffset = j * kvDim + (i / kvMul) * headSize;
                    const float *keyHeadAddr = keyCache.Ptr<float>() + layerOffset + cacheOffset;

                    tensor::Tensor keyMat(model::DataType::iDataTypeFP32, 1, headSize, false, nullptr,
                                          const_cast<float *>(keyHeadAddr));
                    tensor::Tensor scoreMat(model::DataType::iDataTypeFP32, 1, false, nullptr, scoreHeadAddr + j);

                    keyMat.SetDeviceType(model::DeviceType::iDeviceCPU);
                    scoreMat.SetDeviceType(model::DeviceType::iDeviceCPU);

                    GetMatmulKernel(deviceType)(queryMat, keyMat, scoreMat, scale, config);
                }

                tensor::Tensor scoreHeadTensor(model::DataType::iDataTypeFP32, pos + 1, false, nullptr, scoreHeadAddr);
                scoreHeadTensor.SetDeviceType(deviceType);
#ifdef ENABLE_CUDA
                GetSoftmaxKernle(deviceType)(scoreHeadTensor, config ? config->stream : nullptr);
#else
                GetSoftmaxKernle(deviceType)(scoreHeadTensor, nullptr);
#endif

                float *outputHeadPtr = const_cast<float *>(mha.Ptr<float>()) + i * headSize;
#ifdef ENABLE_CUDA
                allocator->memset(outputHeadPtr, sizeof(float) * headSize, config ? config->stream : nullptr, false);
#else
                allocator->memset(outputHeadPtr, sizeof(float) * headSize, nullptr, false);
#endif
                tensor::Tensor outputTensor(model::DataType::iDataTypeFP32, headSize, false, nullptr, outputHeadPtr);
                outputTensor.SetDeviceType(deviceType);

                int32_t cacheOffset = (i / kvMul) * headSize;
                float *valueHeadAddr = const_cast<float *>(valueCache.Ptr<float>() + layerOffset + cacheOffset);
                tensor::Tensor valueTensor(model::DataType::iDataTypeFP32, headSize, false, nullptr, valueHeadAddr);
#ifdef ENABLE_CUDA
                GetScaleSumKernel(deviceType)(valueTensor, scoreHeadTensor, outputTensor, pos, headSize, kvDim,
                                              config ? config->stream : nullptr);
#else
                GetScaleSumKernel(deviceType)(valueTensor, scoreHeadTensor, outputTensor, pos, headSize, kvDim,
                                              nullptr);
#endif
            }
        }
    } // namespace cpu
} // namespace kernel
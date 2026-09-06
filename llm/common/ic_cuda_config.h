#pragma once

#ifdef ENABLE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

namespace kernel
{
    struct CudaConfig
    {
#ifdef ENABLE_CUDA
        cudaStream_t stream = nullptr;
        ~CudaConfig()
        {
            if (stream)
            {
                cudaStreamDestroy(stream);
            }
        }
#endif
    };
}; // namespace kernel
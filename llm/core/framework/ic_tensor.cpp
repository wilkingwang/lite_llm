#include "ic_common.h"
#include "ic_mem_buffer.h"
#include "ic_memory.h"
#include "ic_unicode.h"
#include <alloca.h>
#include <cstddef>
#include <memory>
#include <numeric>

#include <glog/logging.h>
#include <utility>
#ifdef ENABLE_CUDA
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#endif

#include "ic_tensor.h"

using namespace comm;
using namespace model;

namespace tensor
{
    template <typename T, typename TP> static size_t reduceDim(T begin, T end, TP init)
    {
        if (begin >= end)
        {
            return 0;
        }

        size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        return size;
    }

    static size_t getDataTypeSize(model::DataType dataType)
    {
        switch (dataType)
        {
        case model::DataType::iDataTypeFP32:
        {
            return 4;
        }
        case model::DataType::iDataTypeInt8:
        {
            return 1;
        }
        case model::DataType::iDataTypeInt32:
        {
            return 4;
        }
        default:
        {
            LOG(FATAL) << "[Tensor] Unknown data type size for " << int(dataType);
            return 0;
        }
        }
    }

    Tensor::Tensor(model::DataType dataType, int32_t dim0, bool bNeedAlloc,
                   std::shared_ptr<comm::MemoryAllocator> alloc, void *ptr)
        : dataType(dataType)
    {
        size = dim0;
        dims.push_back(dim0);

        if (bNeedAlloc && alloc)
        {
            this->Allocate(alloc);
            return;
        }

        if (ptr != NULL)
        {
            CHECK(bNeedAlloc == false) << "The bNeedAlloc is true when ptr "
                                          "parameter is not a null ptr.";
            this->InitBuf(alloc, dataType, bNeedAlloc, ptr);
        }
    }

    Tensor::Tensor(model::DataType dataType, int32_t dim0, int32_t dim1, bool bNeedAlloc,
                   std::shared_ptr<comm::MemoryAllocator> alloc, void *ptr)
        : dataType(dataType)

    {
        dims.push_back(dim0);
        dims.push_back(dim1);

        size = dim0 * dim1;
        if (bNeedAlloc && alloc)
        {
            Allocate(alloc);
        }
        else
        {
            InitBuf(alloc, dataType, bNeedAlloc, ptr);
        }
    }

    Tensor::Tensor(model::DataType dataType, int32_t dim0, int32_t dim1, int32_t dim2, bool bNeedAlloc,
                   std::shared_ptr<comm::MemoryAllocator> alloc, void *ptr)
        : dataType(dataType)
    {
        dims.push_back(dim0);
        dims.push_back(dim1);
        dims.push_back(dim2);
        size = dim0 * dim1 * dim2;
        if (bNeedAlloc && alloc)
        {
            Allocate(alloc);
        }
        else
        {
            InitBuf(alloc, dataType, bNeedAlloc, ptr);
        }
    }

    Tensor::Tensor(model::DataType dataType, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3, bool bNeedAlloc,
                   std::shared_ptr<comm::MemoryAllocator> alloc, void *ptr)
        : dataType(dataType)
    {
        dims.push_back(dim0);
        dims.push_back(dim1);
        dims.push_back(dim2);
        dims.push_back(dim3);
        size = dim0 * dim1 * dim2 * dim3;
        if (bNeedAlloc && alloc)
        {
            Allocate(alloc);
        }
        else
        {
            InitBuf(alloc, dataType, bNeedAlloc, ptr);
        }
    }

    Tensor::Tensor(model::DataType dataType, std::vector<int32_t> dims, bool bNeedAlloc,
                   std::shared_ptr<comm::MemoryAllocator> alloc, void *ptr)
        : dims(std::move(dims)), dataType(dataType)
    {
        size = reduceDim(dims.begin(), dims.end(), 1);
        if (bNeedAlloc && alloc)
        {
            Allocate(alloc);
        }
        else
        {
            InitBuf(alloc, dataType, bNeedAlloc, ptr);
        }
    }

    void Tensor::cpu()
    {
        CHECK_NE(buffer, nullptr);
        const model::DeviceType deviceType = this->GetDeviceType();

        if (deviceType == model::DeviceType::iDeviceCUDA)
        {
            size_t byteSize = this->GetByteSize();

            auto cpuAllocator = comm::CPUMemoryAllocatorFactory::getInstance();
            auto cpuBuffer = std::make_shared<comm::MemBuffer>(byteSize, cpuAllocator);
            cpuAllocator->memcpy(buffer->ptr(), cpuBuffer->ptr(), byteSize, comm::MemcpyKind::iMemcpyCUDA2CPU);
            this->buffer = cpuBuffer;
        }
        else if (deviceType == model::DeviceType::iDeviceCPU)
        {
            LOG(INFO) << "The device type of tensor is already on cpu.";
        }
        else
        {
            LOG(ERROR) << "The device type of the tensor is unknown.";
        }
    }
#ifdef ENABLE_CUDA
    void Tensor::cuda(cudaStream_t stream = nullptr)
    {
        CHECK_NE(buffer, nullptr);
        const model::DeviceType deviceType = this->GetDeviceType();

        if (deviceType == model::DeviceType::iDeviceCPU)
        {
            size_t byteSize = this->GetByteSize();

            auto cuAllocator = comm::CUDAMemoryAllocatorFactory::getInstance();
            auto cuBuffer = std::make_shared<comm::MemBuffer>(byteSize, cuAllocator);
            cuAllocator->memcpy(buffer->ptr(), cuBuffer->ptr(), byteSize, comm::MemcpyKind::iMemcpyCPU2CUDA, stream);
            this->buffer = cuBuffer;
        }
        else if (deviceType == model::DeviceType::iDeviceCUDA)
        {
            LOG(INFO) << "The device type of tensor is already on cuda.";
        }
        else
        {
            LOG(ERROR) << "The device type of the tensor is unknown.";
        }
    }
#endif
} // namespace tensor
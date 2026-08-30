#include "ic_common.h"
#include "ic_mem_buffer.h"
#include "ic_memory.h"
#include "ic_unicode.h"
#include <alloca.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>

#include <glog/logging.h>
#include <utility>
#include <vector>
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

    bool Tensor::IsEmpty() const
    {
        return this->size == 0 || this->buffer == nullptr || this->buffer->ptr() == nullptr;
    }

    void Tensor::InitBuf(std::shared_ptr<comm::MemoryAllocator> alloc, model::DataType dateType, bool bNeedAlloc,
                         void *ptr)
    {
        if (!alloc && !bNeedAlloc)
        {
            std::shared_ptr<comm::MemBuffer> buf =
                std::make_shared<comm::MemBuffer>(getDataTypeSize(dataType) * this->size, nullptr, ptr, true);
            this->buffer = buf;
        }
    }

    void Reshape(const std::vector<int32_t> &dims);

    size_t Tensor::Size() const
    {
        return this->size;
    }

    size_t Tensor::GetByteSize() const
    {
        return this->Size() * getDataTypeSize(this->dataType);
    }

    int32_t Tensor::GetDimsSize() const
    {
        return static_cast<int32_t>(this->dims.size());
    }

    model::DataType Tensor::GetDataType() const
    {
        return this->dataType;
    }

    int32_t Tensor::GetDim(int32_t idx) const
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->dims.size());

        return this->dims.at(idx);
    }

    const std::vector<int32_t> &Tensor::GetDims() const
    {
        return this->dims;
    }

    std::vector<size_t> Tensor::Strides() const
    {
        std::vector<size_t> strides;

        if (this->dims.empty())
        {
            return strides;
        }

        for (int32_t i = 0; i < this->dims.size() - 1; ++i)
        {
            size_t stride = reduceDim(dims.begin() + i + 1, dims.end(), 1);
            strides.push_back(stride);
        }

        strides.push_back(1);
        return strides;
    }

    bool Tensor::Assign(std::shared_ptr<comm::MemBuffer> buf)
    {
        if (!buf)
        {
            LOG(ERROR) << "The buffer parametr in the assign function is null pointer";
            return false;
        }

        if (this->buffer)
        {
            if (this->buffer->getDeviceType() != buf->getDeviceType())
            {
                LOG(ERROR) << "The device type of the new buffer is different from the original one.";
                return false;
            }
        }

        size_t byteSize = this->GetByteSize();
        if (byteSize > buf->getByteSize())
        {
            LOG(ERROR) << "The size of buffer is too small for the tensor.";
            return false;
        }

        this->buffer = buf;
        return true;
    }

    void Tensor::Reset(model::DataType dType, const std::vector<int32_t> &dims)
    {
        this->dataType = dType;
        this->dims = dims;
        this->size = reduceDim(dims.begin(), dims.end(), 1);
        this->buffer = nullptr;
    }

    void Tensor::SetDeviceType(model::DeviceType deviceType) const
    {
        if (this->buffer)
        {
            this->buffer->setDeviceType(deviceType);
        }
    }

    model::DeviceType Tensor::GetDeviceType() const
    {
        if (!this->buffer)
        {
            return model::DeviceType::iDeviceUnknown;
        }

        return this->buffer->getDeviceType();
    }

    bool Tensor::Allocate(std::shared_ptr<comm::MemoryAllocator> allocator, bool bNeedRealloc)
    {
        if (!allocator)
        {
            LOG(ERROR) << "The allocator parameter in the allocate function is null";
            return false;
        }

        size_t byteSize = this->GetByteSize();
        if (!byteSize)
        {
            LOG(ERROR) << "The byte size parameter in the allocate function is equal to zero.";
            return false;
        }

        if (this->buffer && byteSize <= this->buffer->getByteSize())
        {
            if (!bNeedRealloc)
            {
                return true;
            }
        }

        this->buffer = std::make_shared<comm::MemBuffer>(byteSize, allocator, nullptr);
        if (!this->buffer->ptr())
        {
            LOG(ERROR) << "The memory allocated is a null pointer.";
            return false;
        }

        return true;
    }

    tensor::Tensor Tensor::Clone() const
    {
        Tensor tensor = *this;
        size_t byteSize = this->GetByteSize();

        auto allocator = buffer->getMemAllocator();
        tensor.buffer = std::make_shared<comm::MemBuffer>(byteSize, allocator);
        tensor.buffer->copyFrom(this->buffer.get());

        return tensor;
    }

} // namespace tensor
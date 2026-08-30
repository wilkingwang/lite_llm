#pragma once
#include "ic_memory.h"
#include <cstdint>
#include <memory>
#include <regex>
#include <vector>

#include <glog/logging.h>
#ifdef ENABLE_CUDA
#include <driver_types.h>
#endif

#include "ic_common.h"
#include "ic_mem_buffer.h"

using namespace comm;
using namespace model;

namespace tensor
{
    class Tensor
    {
        explicit Tensor() = default;

        explicit Tensor(model::DataType dataType, int32_t dim0, bool bNeedAlloc = false,
                        std::shared_ptr<comm::MemoryAllocator> alloc = nullptr, void *ptr = nullptr);

        explicit Tensor(model::DataType dataType, int32_t dim0, int32_t dim1, bool bNeedAlloc = false,
                        std::shared_ptr<comm::MemoryAllocator> alloc = nullptr, void *ptr = nullptr);

        explicit Tensor(model::DataType dataType, int32_t dim0, int32_t dim1, int32_t dim2, bool bNeedAlloc = false,
                        std::shared_ptr<comm::MemoryAllocator> alloc = nullptr, void *ptr = nullptr);

        explicit Tensor(model::DataType dataType, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                        bool bNeedAlloc = false, std::shared_ptr<comm::MemoryAllocator> alloc = nullptr,
                        void *ptr = nullptr);

        explicit Tensor(model::DataType dataType, std::vector<int32_t> dims, bool bNeedAlloc = false,
                        std::shared_ptr<comm::MemoryAllocator> alloc = nullptr, void *ptr = nullptr);

        void cpu();
#ifdef ENABLE_CUDA
        void cuda(cudaStream_t stream = nullptr);
#endif

        bool IsEmpty() const;

        void InitBuf(std::shared_ptr<comm::MemoryAllocator> alloc, model::DataType dateType, bool bNeedAlloc,
                     void *ptr);

        template <typename T> T *Ptr();

        template <typename T> const T *Ptr() const;

        void Reshape(const std::vector<int32_t> &dims);

        size_t Size() const;

        size_t GetByteSize() const;

        int32_t GetDimsSize() const;

        model::DataType GetDataType() const;

        int32_t GetDim(int32_t idx) const;

        const std::vector<int32_t> &GetDims() const;

        std::vector<size_t> Strides() const;

        bool Assign(std::shared_ptr<comm::MemBuffer> buf);

        void Reset(model::DataType dType, const std::vector<int32_t> &dims);

        void SetDeviceType(model::DeviceType deviceType) const;

        model::DeviceType GetDeviceType() const;

        bool Allocate(std::shared_ptr<comm::MemoryAllocator> allocator, bool bNeedRealloc = false);

        template <typename T> T *Ptr(int64_t index);

        template <typename T> const T *Ptr(int64_t index) const;

        template <typename T> T &Index(int64_t offset);

        template <typename T> const T &Index(int64_t offset) const;

        tensor::Tensor Clone() const;

      private:
        size_t size = 0;
        std::vector<int32_t> dims;
        std::shared_ptr<comm::MemBuffer> buffer;
        model::DataType dataType = model::DataType::iDataTypeUnknown;
    };

    template <typename T> T *Tensor::Ptr()
    {
        if (!buffer)
        {
            return nullptr;
        }

        return reinterpret_cast<T *>(buffer->ptr());
    }

    template <typename T> const T *Tensor::Ptr() const
    {
        if (!buffer)
        {
            return nullptr;
        }

        return const_cast<const T *>(reinterpret_cast<T *>(buffer->ptr()));
    }

    template <typename T> T *Tensor::Ptr(int64_t index)
    {
        CHECK(buffer != nullptr && buffer->ptr() != nullptr)
            << "The data area buffer of this tensor is empty or ti points to a null pointer.";
        return reinterpret_cast<const T *>(buffer->ptr()) + index;
    }

    template <typename T> const T *Tensor::Ptr(int64_t index) const
    {
        CHECK(buffer != nullptr && buffer->ptr() != nullptr)
            << "The data area buffer of this tensor is empty or ti points to a null pointer.";

        return const_cast<const T *>(reinterpret_cast<const T *>(buffer->ptr())) + index;
    }

    template <typename T> T &Tensor::Index(int64_t offset)
    {
        CHECK_GE(offset, 0);
        CHECK_LT(offset, this->Size());

        T &val = *(reinterpret_cast<T *>(buffer->ptr()) + offset);
        return val;
    }

    template <typename T> const T &Tensor::Index(int64_t offset) const
    {
        CHECK_GE(offset, 0);
        CHECK_LT(offset, this->Size());

        const T &val = *(reinterpret_cast<T *>(buffer->ptr()) + offset);
        return val;
    }
} // namespace tensor
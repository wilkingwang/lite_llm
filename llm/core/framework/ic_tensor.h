#pragma once
#include "ic_memory.h"
#include <cstdint>
#include <memory>
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
                     void *ptr) const;

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

        bool Assign(std::shared_ptr<comm::MemBuffer> buffer);

        void Reset(model::DataType data_type, const std::vector<int32_t> &dims);

        void SetDeviceType(model::DeviceType device_type) const;

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
} // namespace tensor
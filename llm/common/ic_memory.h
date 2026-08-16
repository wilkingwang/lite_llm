#pragma once
#include <map>
#include <memory>

#include "ic_common.h"

using namespace model;

namespace comm
{
    enum class MemcpyKind
    {
        iMemcpyCPU2CPU = 0,
        iMemcpyCPU2CUDA = 1,
        iMemcpyCUDA2CPU = 2,
        iMemcpyCUDA2CUDA = 3,
    };

    class MemoryAllocator
    {
    public:
        explicit MemoryAllocator(model::DeviceType deviceType)
            : deviceType(deviceType)
        {
        }

        virtual model::DeviceType getDeviceType() const
        {
            return this->deviceType;
        }

        virtual void release(void *ptr) const = 0;

        virtual void *allocate(size_t byteSize) const = 0;

        virtual void memcpy(const void *src,
                            void *dst,
                            size_t byteSize,
                            MemcpyKind memcpyKind = MemcpyKind::iMemcpyCPU2CPU,
                            void *stream = nullptr,
                            bool bSync = false) const;

        virtual void memset(void *ptr, size_t byteSize, void *stream, bool bSync = false);

    private:
        model::DeviceType deviceType = model::DeviceType::iDeviceUnknown;
    };

    struct CUDAMemoryBuffer
    {
        void *data;
        size_t byteSize;
        bool busy;
    };

    class CPUMemoryAllocator : public MemoryAllocator
    {
    public:
        explicit CPUMemoryAllocator();

        void *allocate(size_t byteSize) const override;

        void release(void *ptr) const override;
    };

    class CUDAMemoryAllocator : public MemoryAllocator
    {
    public:
        explicit CUDAMemoryAllocator();

        void *allocate(size_t byteSize) const override;

        void release(void *ptr) const override;

    private:
        mutable std::map<int, size_t> noBusyCnt;
        mutable std::map<int, std::vector<CUDAMemoryBuffer>> bigBufferMap;
        mutable std::map<int, std::vector<CUDAMemoryBuffer>> cudaBufferMap;
    };

    class CPUMemoryAllocatorFactory
    {
    public:
        static std::shared_ptr<CPUMemoryAllocator> getInstance()
        {
            if (instance == nullptr)
            {
                instance = std::make_shared<CPUMemoryAllocator>();
            }

            return instance;
        }

    private:
        static std::shared_ptr<CPUMemoryAllocator> instance;
    };

    class CUDAMemoryAllocatorFactory
    {
    public:
        static std::shared_ptr<CUDAMemoryAllocator> getInstance()
        {
            if (instance == nullptr)
            {
                instance = std::make_shared<CUDAMemoryAllocator>();
            }

            return instance;
        }

    private:
        static std::shared_ptr<CUDAMemoryAllocator> instance;
    };
}
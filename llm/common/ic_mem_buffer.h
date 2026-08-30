#pragma once
#include <memory>

#include "ic_memory.h"

using namespace model;

namespace comm
{
    class MemBuffer : public NoCopyable, std::enable_shared_from_this<MemBuffer>
    {
      public:
        explicit MemBuffer() = default;

        explicit MemBuffer(size_t byteSize, std::shared_ptr<MemoryAllocator> allocator = nullptr,
                           void *buffer = nullptr, bool bUseExternal = false);

        virtual ~MemBuffer();

        bool allocate();

        bool copyFrom(const MemBuffer &buffer) const;

        bool copyFrom(const MemBuffer *buffer) const;

        std::shared_ptr<MemoryAllocator> getMemAllocator() const
        {
            return this->allocator;
        }

        void *ptr()
        {
            return this->buffer;
        }

        const void *ptr() const
        {
            return this->buffer;
        }

        size_t getByteSize() const
        {
            return this->byteSize;
        }

        model::DeviceType getDeviceType() const
        {
            return this->deviceType;
        }

        void setDeviceType(model::DeviceType deviceType)
        {
            this->deviceType = deviceType;
        }

        std::shared_ptr<MemBuffer> getSharedFromThis()
        {
            return shared_from_this();
        }

        bool isExternal() const
        {
            return this->bUseExternal;
        }

      private:
        size_t byteSize = 0;
        void *buffer = nullptr;
        bool bUseExternal = false;
        std::shared_ptr<MemoryAllocator> allocator;
        model::DeviceType deviceType = DeviceType::iDeviceUnknown;
    };
} // namespace comm
#include "ic_mem_buffer.h"

namespace comm
{
    MemBuffer::MemBuffer(size_t byteSize, std::shared_ptr<MemoryAllocator> allocator, void *buffer, bool bUseExternal)
        : byteSize(byteSize), allocator(allocator), buffer(buffer), bUseExternal(bUseExternal)
    {
        if (!this->buffer && this->allocator)
        {
            this->bUseExternal = false;
            this->deviceType = this->allocator->getDeviceType();
            this->buffer = allocator->allocate(this->byteSize);
        }
    }

    MemBuffer::~MemBuffer()
    {
        if (this->bUseExternal)
        {
            return;
        }

        if (this->buffer && this->allocator)
        {
            this->allocator->release(this->buffer);
            this->buffer = nullptr;
        }
    }

    bool MemBuffer::allocate()
    {
        if (this->allocator && this->byteSize != 0)
        {
            this->bUseExternal = false;
            this->buffer = allocator->allocate(this->byteSize);
            if (!buffer)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else
        {
            return false;
        }
    }

    bool MemBuffer::copyFrom(const MemBuffer &memBuffer) const
    {
        CHECK(this->allocator != nullptr);
        CHECK(memBuffer.buffer != nullptr);
        return true;
    }

    bool MemBuffer::copyFrom(const MemBuffer *memBuffer) const
    {
        CHECK(this->allocator != nullptr);
        CHECK(memBuffer != nullptr || memBuffer->buffer != nullptr);
        return true;
    }
} // namespace comm

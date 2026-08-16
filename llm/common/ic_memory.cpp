#include <cstdlib>
#include <glog/logging.h>

#include "ic_memory.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define IC_HAVE_POSIX_MEMALIGN
#endif

namespace comm
{
    CPUMemoryAllocator::CPUMemoryAllocator()
        : MemoryAllocator(model::DeviceType::iDeviceCPU)
    {
    }

    void *CPUMemoryAllocator::allocate(size_t byteSize) const
    {
        if (!byteSize)
        {
            return nullptr;
        }

#ifdef IC_HAVE_POSIX_MEMALIGN
        void *data = nullptr;
        const size_t alignment = (byteSize >= size_t(1024) ? size_t(32) : size_t(16));
        int status = posix_memalign((void **)&data,
                                    ((alignment >= sizeof(void *)) ? alignment : sizeof(void *)),
                                    byteSize);
        if (status != 0)
        {
            return nullptr;
        }

        return data;
#else
        void *data = malloc(byteSize);
        return data;
#endif
    }

    void CPUMemoryAllocator::release(void *ptr) const
    {
        if (ptr == nullptr)
        {
            return;
        }

        free(ptr);
        ptr = nullptr;
    }

    CUDAMemoryAllocator::CUDAMemoryAllocator()
        : MemoryAllocator(model::DeviceType::iDeviceCUDA)
    {
    }

    void *CUDAMemoryAllocator::allocate(size_t byteSize) const
    {
        return nullptr;
    }

    void CUDAMemoryAllocator::release(void *ptr) const
    {
        return;
    }

    void MemoryAllocator::memcpy(const void *src,
                                 void *dst,
                                 size_t byteSize,
                                 MemcpyKind memcpyKind,
                                 void *stream,
                                 bool bSync) const
    {
    }

    void MemoryAllocator::memset(void *ptr, size_t byteSize, void *stream, bool bSync)
    {
    }

    std::shared_ptr<CPUMemoryAllocator> CPUMemoryAllocatorFactory::instance = nullptr;
    std::shared_ptr<CUDAMemoryAllocator> CUDAMemoryAllocatorFactory::instance = nullptr;
}
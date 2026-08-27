#include <cstddef>
#include <numeric>

#include <glog/logging.h>
#ifdef ENABLE_CUDA
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#endif

#include "ic_tensor.h"

using namespace comm;
using namespace model;

namespace tensor
{
    template <typename T, typename TP>
    static size_t reduceDim(T begin, T end, TP init)
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
            LOG(FATAL) << "[Tensor] Unknown data type size for "
                       << int(dataType);
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

    Tensor::Tensor(model::DataType dataType, int32_t dim0, int32_t dim1,
                   bool bNeedAlloc,
                   std::shared_ptr<comm::MemoryAllocator> alloc, void *ptr)

    {
        dims.push_back(dim0);
    }
} // namespace tensor
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>

#include "ic_model_weight.h"

namespace model
{
    ICRawModelData::~ICRawModelData()
    {
        if (pData != nullptr && pData != MAP_FAILED)
        {
            munmap(pData, fileSize);
        }

        if (fd != -1)
        {
            close(fd);
        }
    }

    const void *ICRawModelDataFP32::GetWeight(size_t offset) const
    {
        return static_cast<float *>(pWeightData) + offset;
    }

    const void *ICRawModelDataINT8::GetWeight(size_t offset) const
    {
        return static_cast<int8_t *>(pWeightData) + offset;
    }
} // namespace model
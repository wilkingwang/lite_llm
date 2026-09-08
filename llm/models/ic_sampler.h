#pragma once

#include "ic_common.h"
#include <cstddef>
#include <cstdint>

namespace sampler
{
    class ICSampler
    {
      public:
        explicit ICSampler(model::DeviceType deviceType) : deviceType(deviceType)
        {
        }

        virtual size_t sample(const float *logits, size_t size, void *stream = nullptr) = 0;

      protected:
        model::DeviceType deviceType;
    };
} // namespace sampler
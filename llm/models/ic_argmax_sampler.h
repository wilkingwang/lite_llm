#pragma once

#include "ic_common.h"
#include "ic_sampler.h"
#include <cstddef>
namespace sampler
{
    class ICArgmaxSampler : public ICSampler
    {
      public:
        explicit ICArgmaxSampler(model::DeviceType deviceType) : ICSampler(deviceType)
        {
        }

        size_t sample(const float *logits, size_t size, void *stream) override;
    };
} // namespace sampler
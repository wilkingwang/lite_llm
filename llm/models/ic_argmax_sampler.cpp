#include <algorithm>
#include <iterator>

#include "ic_argmax_sampler.h"
#include "ic_common.h"

namespace sampler
{
    size_t ICArgmaxSampler::sample(const float *logits, size_t size, void *stream)
    {
        size_t next = 0;
        if (deviceType == model::DeviceType::iDeviceCPU)
        {
            next = std::distance(logits, std::max_element(logits, logits + size));
            return next;
        }
        else
        {
            // TODO
            next = 0;
        }

        return next;
    }

} // namespace sampler
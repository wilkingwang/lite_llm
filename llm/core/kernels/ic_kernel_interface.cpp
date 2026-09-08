#include "ic_kernel_interface.h"
#include "ic_common.h"

namespace kernel
{
    Add GetAddKernel(model::DeviceType deviceType)
    {
        switch (deviceType)
        {
        case model::DeviceType::iDeviceCPU:
            return nullptr;
        case model::DeviceType::iDeviceCUDA:
            return nullptr;
        default:
            LOG(FATAL) << "Unknown device type for add kernel.";
            return nullptr;
        }
    }
} // namespace kernel
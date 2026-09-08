#pragma once

#include "ic_common.h"
#include "ic_layer.h"

namespace model
{
    class ICVecAddLayer : public Layer
    {
      public:
        explicit ICVecAddLayer(model::DeviceType deviceType);

        comm::Status Check() const override;

        comm::Status Forward() override;
    };
} // namespace model
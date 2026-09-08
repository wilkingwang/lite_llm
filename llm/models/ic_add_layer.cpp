#include "ic_add_layer.h"
#include "ic_base_layer.h"
#include "ic_common.h"
#include "ic_layer.h"
#include "ic_tensor.h"
#include <cstdint>

namespace model
{
    ICVecAddLayer::ICVecAddLayer(model::DeviceType deviceType) : Layer(deviceType, LayerType::iLayerAdd, "Add")
    {
        ResetInputSize(2);
        ResetOutputSize(1);
    }

    comm::Status ICVecAddLayer::Check() const
    {
        tensor::Tensor input1 = this->GetInput(0);
        tensor::Tensor input2 = this->GetInput(1);

        comm::Status status;
        int32_t size = input1.Size();
        status = this->CheckTensorWithDim(input1, deviceType, dataType, size);
        if (!status)
        {
            LOG(ERROR) << "The first input tensor check error in the add layer.";
            return status;
        }

        status = this->CheckTensorWithDim(input2, deviceType, dataType, size);
        if (!status)
        {
            LOG(ERROR) << "The second input tensor check error in the add layer.";
            return status;
        }

        status = this->CheckTensorWithDim(GetOutput(0), deviceType, dataType, size);
        if (!status)
        {
            LOG(ERROR) << "The output tensor error in the add layer.";
            return status;
        }

        return comm::Success();
    }

    comm::Status ICVecAddLayer::Forward()
    {
        auto status = this->Check();
        if (!status)
        {
            return status;
        }

        auto input1 = this->GetInput(0);
        auto input2 = this->GetInput(1);
        auto output = this->GetOutput(0);
        if (deviceType == model::DeviceType::iDeviceCUDA)
        {
            CHECK(cudaConfig != nullptr);
        }

        // TODO

        return comm::Success();
    }
} // namespace model
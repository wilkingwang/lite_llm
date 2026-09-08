#include "ic_base_layer.h"
#include "ic_common.h"

namespace model
{
    BaseLayer::BaseLayer(model::DeviceType deviceType, model::DataType dataType, LayerType layerType,
                         std::string layerNum)
        : deviceType(deviceType), layerType(layerType), dataType(dataType), layerName(layerName)
    {
    }

    model::DataType BaseLayer::GetDataType() const
    {
        return this->dataType;
    }

    LayerType BaseLayer::GetLayerType() const
    {
        return this->layerType;
    }

    comm::Status BaseLayer::SetWeight(const int32_t idx, const tensor::Tensor &weight)
    {
        return comm::FunctionUnImplement();
    }

    comm::Status BaseLayer::SetWeight(const int32_t idx, const std::vector<int32_t> &dims, const void *pWeight,
                                      model::DeviceType deviceType)
    {
        return comm::FunctionUnImplement();
    }

    const std::string &BaseLayer::GetLayerName() const
    {
        return this->layerName;
    }

    void BaseLayer::SetLayerName(const std::string &layerName)
    {
        this->layerName = layerName;
    }

    model::DeviceType BaseLayer::GetDeviceType() const
    {
        return this->deviceType;
    }

    void BaseLayer::SetDeviceType(model::DeviceType deviceType)
    {
        this->deviceType = deviceType;
    }

} // namespace model
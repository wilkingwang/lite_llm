#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ic_common.h"
#include "ic_tensor.h"

namespace kernel
{
    class Layer;

    enum class LayerType : uint8_t
    {
        iLayerUnknown = 0,
        iLayerLinear = 1,
        iLayerEncode = 2,
        iLayerEmbedding = 3,
        iLayerRMSNorm = 4,
        iLayerMatMul = 5,
        iLayerRoPE = 6,
        iLayerMHA = 7,
        iLayerSoftmax = 8,
        iLayerAdd = 9,
        iLayerSwiGLU = 10,
    };

    class BaseLayer
    {
      public:
        explicit BaseLayer(model::DeviceType deviceType, model::DataType dataType, LayerType layerType,
                           std::string layerNum = "");

        model::DataType GetDataType() const;

        LayerType GetLayerType() const;

        virtual comm::Status Init() = 0;

        virtual comm::Status Forward() = 0;

        virtual comm::Status Forward(const tensor::Tensor &input, const tensor::Tensor &output) = 0;

        virtual comm::Status Forward(const tensor::Tensor &input1, const tensor::Tensor &intpu2,
                                     const tensor::Tensor &output) = 0;

        virtual comm::Status Forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                     const tensor::Tensor &input3, const tensor::Tensor &output) = 0;

        virtual comm::Status Forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                     const tensor::Tensor &input3, const tensor::Tensor &input4,
                                     const tensor::Tensor &output) = 0;

        virtual comm::Status Forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                     const tensor::Tensor &input3, const tensor::Tensor &input4,
                                     const tensor::Tensor &input5, const tensor::Tensor &output) = 0;

        virtual void SetInput(int32_t idx, const tensor::Tensor &input) = 0;

        virtual void SetOutput(int32_t idx, const tensor::Tensor &output) = 0;

        virtual size_t GetInputSize() const = 0;

        virtual size_t GetOutputSize() const = 0;

        virtual comm::Status Check() const = 0;

        virtual tensor::Tensor &GetInput(const int32_t idx) = 0;

        virtual tensor::Tensor &GetOutput(const int32_t idx) = 0;

        virtual const tensor::Tensor &GetInput(const int32_t idx) const = 0;

        virtual const tensor::Tensor &GetOutput(const int32_t idx) const = 0;

        virtual comm::Status SetWeight(const int32_t idx, const tensor::Tensor &weight);

        virtual comm::Status SetWeight(const int32_t idx, const std::vector<int32_t> &dims, const void *pWeight,
                                       model::DeviceType deviceType = model::DeviceType::iDeviceUnknown);

        const std::string &GetLayerName() const;

        void SetLayerName(const std::string &layerName);

        model::DeviceType GetDeviceType() const;

        void SetDeviceType(model::DeviceType deviceType);

      protected:
        std::string layerName;
        LayerType layerType = LayerType::iLayerUnknown;
        model::DataType dataType = model::DataType::iDataTypeUnknown;
        model::DeviceType deviceType = model::DeviceType::iDeviceUnknown;
    };
} // namespace kernel
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "ic_base_layer.h"
#include "ic_common.h"
#include "ic_layer.h"
#include "ic_mem_buffer.h"
#include "ic_tensor.h"

namespace kernel
{
    Layer::Layer(model::DeviceType deviceType, LayerType layerType, std::string layerName)
        : BaseLayer(deviceType, model::DataType::iDataTypeFP32, layerType, std::move(layerName))
    {
    }

    comm::Status Layer::Init()
    {
        return comm::Success();
    }

    comm::Status Layer::CheckTensor(const tensor::Tensor &tensor, model::DeviceType deviceType,
                                    model::DataType dataType) const
    {
        if (tensor.IsEmpty())
        {
            return comm::InvalidArgument("The parameter of tensor is empty.");
        }

        if (tensor.GetDeviceType() != this->deviceType)
        {
            return comm::InvalidArgument("The device type of tensor is invalid.");
        }

        if (tensor.GetDataType() != this->dataType)
        {
            return comm::InvalidArgument("The data type of tensor is invalid.");
        }

        return comm::Success();
    }

    comm::Status Layer::CheckTensorWithDim(const tensor::Tensor &tensor, model::DeviceType deviceType,
                                           model::DataType dataType, ...) const
    {
        std::va_list args;

        if (tensor.IsEmpty())
        {
            return comm::InvalidArgument("The parameter of tensor is empty.");
        }

        if (tensor.GetDeviceType() != this->deviceType)
        {
            return comm::InvalidArgument("The device type of tensor is invalid.");
        }

        if (tensor.GetDataType() != this->dataType)
        {
            return comm::InvalidArgument("The data type of tensor is invalid.");
        }

        va_start(args, dataType);
        for (int32_t i = 0; i < tensor.GetDimsSize(); i++)
        {
            int32_t dim = va_arg(args, int32_t);
            if (dim != tensor.GetDim(i))
            {
                return comm::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
            }
        }
        va_end(args);
        return comm::Success();
    }

    comm::Status Layer::Check() const
    {
        return comm::FunctionUnImplement("The check function is not implement.");
    }

    comm::Status Layer::Forward()
    {
        return comm::FunctionUnImplement("The forward function is not implement.");
    }

    comm::Status Layer::Forward(const tensor::Tensor &input, const tensor::Tensor &output)
    {
        this->SetInput(0, input);

        this->SetOutput(0, output);
        return this->Forward();
    }

    comm::Status Layer::Forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                const tensor::Tensor &output)
    {
        this->SetInput(0, input1);
        this->SetInput(1, input2);

        this->SetOutput(0, output);
        return this->Forward();
    }

    comm::Status Layer::Forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                const tensor::Tensor &input3, const tensor::Tensor &output)
    {
        this->SetInput(0, input1);
        this->SetInput(1, input2);
        this->SetInput(2, input3);

        this->SetOutput(0, output);
        return this->Forward();
    }

    comm::Status Layer::Forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                const tensor::Tensor &input3, const tensor::Tensor &input4,
                                const tensor::Tensor &output)
    {
        this->SetInput(0, input1);
        this->SetInput(1, input2);
        this->SetInput(2, input3);
        this->SetInput(3, input4);

        this->SetOutput(0, output);
        return this->Forward();
    }

    comm::Status Layer::Forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                const tensor::Tensor &input3, const tensor::Tensor &input4,
                                const tensor::Tensor &input5, const tensor::Tensor &output)
    {
        this->SetInput(0, input1);
        this->SetInput(1, input2);
        this->SetInput(2, input3);
        this->SetInput(3, input4);
        this->SetInput(4, input5);

        this->SetOutput(0, output);
        return this->Forward();
    }

    void Layer::SetInput(int32_t idx, const tensor::Tensor &input)
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->inputs.size());

        this->inputs.at(idx) = input;
    }

    void Layer::SetOutput(int32_t idx, const tensor::Tensor &output)
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->outputs.size());

        this->outputs.at(idx) = output;
    }

    size_t Layer::GetInputSize() const
    {
        return this->inputs.size();
    }

    size_t Layer::GetOutputSize() const
    {
        return this->outputs.size();
    }

    tensor::Tensor &Layer::GetInput(const int32_t idx)
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->inputs.size());

        return this->inputs.at(idx);
    }

    tensor::Tensor &Layer::GetOutput(const int32_t idx)
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->outputs.size());

        return this->outputs.at(idx);
    }

    const tensor::Tensor &Layer::GetInput(const int32_t idx) const
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->outputs.size());

        return this->inputs.at(idx);
    }

    const tensor::Tensor &Layer::GetOutput(const int32_t idx) const
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->outputs.size());

        return this->outputs.at(idx);
    }

    void Layer::ResetInputSize(const size_t size)
    {
        this->inputs.resize(size);
    }

    void Layer::ResetOutputSize(const size_t size)
    {
        this->outputs.resize(size);
    }

    void Layer::Cuda()
    {
#ifdef ENABLE_CUDA
        for (auto &input : this->inputs)
        {
            if (!input.IsEmpty())
            {
                this->input.Cuda();
            }
        }

        for (auto &output : this->outputs)
        {
            if (!output.IsEmpty())
            {
                this->output.Cuda(this->cudaConfig ? this->cudaConfig : nullptr);
            }
        }
#endif
    }

    void Layer::SetCudaConfig(std::shared_ptr<kernel::CudaConfig> config)
    {
        if (!config)
        {
            return;
        }

        this->cudaConfig = config;
    }

    std::shared_ptr<kernel::CudaConfig> Layer::GetCudaConfig() const
    {
        return this->cudaConfig;
    }

    LayerParam::LayerParam(model::DeviceType deviceType, LayerType layerType, bool bQuantLayer, std::string layerName)
        : Layer(deviceType, layerType, std::move(layerName)), bQuantLayer(bQuantLayer)
    {
    }

    size_t LayerParam::GetWeightSize() const
    {
        return this->weights.size();
    }

    void LayerParam::ResetWeightSize(size_t size)
    {
        this->weights.resize(size);
    }

    tensor::Tensor &LayerParam::GetWeight(size_t idx)
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->weights.size());

        return this->weights.at(idx);
    }

    const tensor::Tensor &LayerParam::GetWeight(size_t idx) const
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->weights.size());

        return this->weights.at(idx);
    }

    void LayerParam::Cuda()
    {
        Layer::Cuda();

#ifdef ENABLE_CUDA
        for (auto &weight : this->weights)
        {
            weight.Cuda(this->cudaConfig ? this->cudaConfig->stream : nullptr);
        }

        if (!this->scales.IsEmpty())
        {
            this->scales.Cuda(this->cudaConfig ? this->cudaConfig->stream : nullptr);
        }
#endif
    }

    comm::Status LayerParam::SetWeight(const int32_t idx, const tensor::Tensor &weight)
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->weights.size());
        CHECK(weight.GetDataType() == model::DataType::iDataTypeFP32);

        if (!weight.IsEmpty())
        {
            CHECK(weight.GetDeviceType() == this->deviceType);
        }

        this->weights.at(idx) = weight;
        return comm::Success();
    }

    comm::Status LayerParam::SetWeight(const int32_t idx, const std::vector<int32_t> &dims, const void *weightPtr,
                                       model::DeviceType deviceType)
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->weights.size());
        CHECK_NE(weightPtr, nullptr);

        size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
        std::shared_ptr<comm::MemBuffer> buffer =
            std::make_shared<comm::MemBuffer>(size, nullptr, const_cast<void *>(weightPtr), true);
        if (this->deviceType != model::DeviceType::iDeviceUnknown)
        {
            buffer->setDeviceType(this->deviceType);
        }

        if (!bQuantLayer)
        {
            tensor::Tensor weight(model::DataType::iDataTypeFP32, dims);
            weight.SetDeviceType(this->deviceType);
            CHECK(weight.Assign(buffer));
            this->weights.at(idx) = weight;
        }
        else
        {
            tensor::Tensor weight(model::DataType::iDataTypeInt8, dims);
            weight.SetDeviceType(this->deviceType);
            CHECK(weight.Assign(buffer));
            this->weights.at(idx) = weight;

            const int32_t weightSize = static_cast<int32_t>(weight.Size());
            CHECK(weightSize % this->groupSize == 0);

            int32_t scaleNums = weightSize / this->groupSize;
            scales = tensor::Tensor{model::DataType::iDataTypeFP32, scaleNums, false, nullptr,
                                    reinterpret_cast<float *>((int8_t *)weightPtr + weightSize)};
            scales.SetDeviceType(this->deviceType);
        }

        return comm::Success();
    }

    void LayerParam::SetScales(const tensor::Tensor &scales)
    {
        CHECK(!scales.IsEmpty());

        this->scales = scales;
    }

    void LayerParam::SetGroupSize(const int32_t groupSize)
    {
        this->groupSize = groupSize;
    }

    int32_t LayerParam::GetScaleNum() const
    {
        CHECK(!this->scales.IsEmpty());

        return static_cast<int32_t>(this->scales.Size());
    }

} // namespace kernel
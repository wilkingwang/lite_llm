#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ic_base_layer.h"
#include "ic_common.h"
#include "ic_cuda_config.h"
#include "ic_tensor.h"

namespace kernel
{
    class Layer : public BaseLayer
    {
      public:
        explicit Layer(model::DeviceType deviceType, LayerType layerType, std::string layerName = "");

        comm::Status Init() override;

        comm::Status CheckTensor(const tensor::Tensor &tensor, model::DeviceType deviceType,
                                 model::DataType dataType) const;

        comm::Status CheckTensorWithDim(const tensor::Tensor &tensor, model::DeviceType deviceType,
                                        model::DataType dataType, ...) const;

        comm::Status Check() const override;

        comm::Status Forward() override;

        comm::Status Forward(const tensor::Tensor &input, const tensor::Tensor &output) override;

        comm::Status Forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                             const tensor::Tensor &output) override;

        comm::Status Forward(const tensor::Tensor &input1, const tensor::Tensor &input2, const tensor::Tensor &input3,
                             const tensor::Tensor &output) override;

        comm::Status Forward(const tensor::Tensor &input1, const tensor::Tensor &input2, const tensor::Tensor &input3,
                             const tensor::Tensor &input4, const tensor::Tensor &output) override;

        comm::Status Forward(const tensor::Tensor &input1, const tensor::Tensor &input2, const tensor::Tensor &input3,
                             const tensor::Tensor &input4, const tensor::Tensor &input5,
                             const tensor::Tensor &output) override;

        void SetInput(int32_t idx, const tensor::Tensor &input) override;

        void SetOutput(int32_t idx, const tensor::Tensor &output) override;

        size_t GetInputSize() const override;

        size_t GetOutputSize() const override;

        tensor::Tensor &GetInput(const int32_t idx) override;

        tensor::Tensor &GetOutput(const int32_t idx) override;

        const tensor::Tensor &GetInput(const int32_t idx) const override;

        const tensor::Tensor &GetOutput(const int32_t idx) const override;

        void ResetInputSize(const size_t size);

        void ResetOutputSize(const size_t size);

        virtual void Cuda();

        virtual void SetCudaConfig(std::shared_ptr<kernel::CudaConfig> config);

        std::shared_ptr<kernel::CudaConfig> GetCudaConfig() const;

      protected:
        std::vector<tensor::Tensor> inputs;
        std::vector<tensor::Tensor> outputs;
        std::shared_ptr<kernel::CudaConfig> cudaConfig;
    };

    class LayerParam : public Layer
    {
      public:
        explicit LayerParam(model::DeviceType deviceType, LayerType layerType, bool bQuantLayer = false,
                            std::string layerName = "");

        size_t GetWeightSize() const;

        void ResetWeightSize(size_t size);

        tensor::Tensor &GetWeight(size_t idx);

        const tensor::Tensor &GetWeight(size_t idx) const;

        void Cuda() override;

        comm::Status SetWeight(const int32_t idx, const tensor::Tensor &weight) override;

        comm::Status SetWeight(const int32_t idx, const std::vector<int32_t> &dims, const void *weightPtr,
                               model::DeviceType deviceType = model::DeviceType::iDeviceUnknown) override;

        void SetScales(const tensor::Tensor &scales);

        void SetGroupSize(const int32_t groupSize);

        int32_t GetScaleNum() const;

      protected:
        int32_t groupSize = 0;
        bool bQuantLayer = false;
        tensor::Tensor scales;
        std::vector<tensor::Tensor> weights;
    };
} // namespace kernel
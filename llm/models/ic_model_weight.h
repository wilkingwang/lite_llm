#pragma once
#include <cstddef>
#include <cstdint>

namespace model
{
    class ICRawModelData
    {
      public:
        ~ICRawModelData();

        virtual const void *GetWeight(size_t offset) const = 0;

      protected:
        int32_t fd = -1;
        size_t fileSize = 0;
        void *pData = nullptr;
        void *pWeightData = nullptr;
    };

    class ICRawModelDataFP32 : ICRawModelData
    {
      public:
        const void *GetWeight(size_t offset) const override;
    };

    class ICRawModelDataINT8 : ICRawModelData
    {
      public:
        const void *GetWeight(size_t offset) const override;
    };
} // namespace model
#pragma once
#include <cstdint>
#include <memory>
#include <string>

#include "ic_common.h"
#include "ic_config.h"

namespace model
{
    class model
    {
      private:
        virtual void initMem() = 0;
        virtual comm::StatusCode createLayers() = 0;
        virtual void createParamLayers() = 0;
        virtual void createNonParamLayers() = 0;
        virtual void createParamQuantLayers() = 0;

      private:
        std::string modelPath;
        std::string tokenizerPath;

        int32_t groupSize = 1;
        bool bQuantMode = false;

        std::unique_ptr<TransformerConfig> config;
    };
} // namespace model
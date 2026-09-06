#pragma once

#include <cstdint>

namespace model
{
    struct KVConfig
    {
        int32_t kvDim = 0;
        int32_t kvMul = 0;
        int32_t headSize = 0;
    };

    struct ModelConfig
    {
        int32_t dim = 0;
        int32_t hiddenDim = 0;
        int32_t layerNum = 0;
        int32_t headNum = 0;
        int32_t kvHeadNum = 0;
        int32_t vocabSize = 0;
        int32_t seqLen = 0;
        int32_t immediateDim = 0;
    };

    struct TransformerConfig
    {
        KVConfig kvConfig;
        ModelConfig modelConfig;
        bool bSharedWeight = false;
    };
} // namespace model
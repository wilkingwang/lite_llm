#pragma once

#include <cstdint>
#include <string>
#include <iostream>

#include <glog/logging.h>

namespace comm
{
    enum StatusCode : uint8_t
    {
        iSuccess = 0,
        iPathInValid = 1,
        iInvalidArgument = 2,
        iFunctionUnImplement = 3,
        iModelParseErr = 4,
        iInternalErr = 5,
        iAlreadyExist = 6,
    };

    enum class BufferType
    {
        iInputTokens = 0,
        iInputEmbeddings = 1,
        iOutputRMSNorm = 2,
        iKeyCache = 3,
        iValueCache = 4,
        iQuery = 5,
        iInputPos = 6,
        iScoreStorage = 7,
        iOutputMHA = 8,
        iAttenOutput = 9,
        iW1Output = 10,
        iW2Output = 11,
        iW3Output = 12,
        iFFNRMSNorm = 13,
        iForwardOutput = 14,
        iForwardOutputCPU = 15,
        iSinCache = 16,
        iCosCache = 17,
    };

    class NoCopyable
    {
    protected:
        NoCopyable() = default;

        ~NoCopyable() = default;

        NoCopyable(const NoCopyable &) = delete;

        NoCopyable &operator=(const NoCopyable) = delete;
    };

    class Status
    {
    public:
        Status(const int code = StatusCode::iSuccess, const std::string &errMsg = "");

        Status(const Status &other) = default;

        Status &operator=(const Status &other) = default;

        Status &operator=(int code);

        bool operator==(int code) const;

        bool operator!=(int code) const;

        operator int() const;

        operator bool() const;

        int32_t getErrCode() const;

        const std::string &getErrMsg() const;

        void setErrMsg(const std::string &errMsg);

    private:
        int code = StatusCode::iSuccess;
        std::string msg;
    };

#define STATUS_CHECK(call)                                                                \
    do                                                                                    \
    {                                                                                     \
        const comm::Status &status = call;                                                \
        if (!Status)                                                                      \
        {                                                                                 \
            const size_t bufSize = 512;                                                   \
            char buf[bufSize];                                                            \
            snprintf(buf, bufSize - 1,                                                    \
                     "Infer error\n File:%s, Line: %d\n Error code:%d\n Error msg: %s\n", \
                     __FILE__, __LINE__, int(status), status.getErrMsg().c_str());        \
        }                                                                                 \
    } while (0)

    Status Success(const std::string &errMsg = "");

    Status PathInvalid(const std::string &errMsg = "");

    Status InvalidArgument(const std::string &errMsg = "");

    Status FunctionUnImplement(const std::string &errMsg = "");

    Status ModelParser(const std::string &errMsg = "");

    Status InternalError(const std::string &errMsg = "");

    Status AlreadyExists(const std::string &errMsg = "");

    std::ostream &operator<<(std::ostream &os, const Status &status);

} // namespace comm

namespace model
{
    enum class DeviceType : uint8_t
    {
        iDeviceUnknown = 0,
        iDeviceCPU = 1,
        iDeviceCUDA = 2,
    };

    enum class DataType : uint8_t
    {
        iDataTypeUnknown = 0,
        iDataTypeFP32 = 1,
        iDataTypeInt8 = 2,
        iDataTypeInt32 = 3,
    };

    enum class ModelType : uint8_t
    {
        iModelTypeUnknown = 0,
        iModelTypeLLama2 = 1,
    };

    enum class TokenizerType
    {
        iEncodeUnknown = 0,
        iEncodeSPE = 0,
        iEncodeBPE = 1,
    };

    inline size_t GetDataTypeSize(DataType dataType)
    {
        size_t dataTypeSize = 0;

        switch (dataType)
        {
        case DataType::iDataTypeFP32:
            dataTypeSize = sizeof(float);
            break;
        case DataType::iDataTypeInt8:
            dataTypeSize = sizeof(int8_t);
            break;
        case DataType::iDataTypeInt32:
            dataTypeSize = sizeof(int32_t);
            break;
        default:
            break;
        }

        return dataTypeSize;
    }
} // namespace model

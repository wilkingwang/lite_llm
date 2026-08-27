#include <string>

#include "ic_common.h"

namespace comm
{
    Status::Status(const int code, const std::string &errMsg)
        : code(code), msg(errMsg)
    {
    }

    Status &Status::operator=(int code)
    {
        this->code = code;
        return *this;
    }

    bool Status::operator==(int code) const
    {
        if (this->code == code)
        {
            return true;
        }

        return false;
    }

    bool Status::operator!=(int code) const
    {
        if (this->code != code)
        {
            return true;
        }

        return false;
    }

    Status::operator int() const
    {
        return this->code;
    }

    Status::operator bool() const
    {
        return this->code == StatusCode::iSuccess;
    }

    int32_t Status::getErrCode() const
    {
        return this->code;
    }

    const std::string &Status::getErrMsg() const
    {
        return this->msg;
    }

    void Status::setErrMsg(const std::string &errMsg)
    {
        this->msg = errMsg;
    }

    Status Success(const std::string &errMsg)
    {
        return Status{StatusCode::iSuccess, errMsg};
    }

    Status PathInvalid(const std::string &errMsg)
    {
        return Status{StatusCode::iPathInValid, errMsg};
    }

    Status InvalidArgument(const std::string &errMsg)
    {
        return Status{StatusCode::iInvalidArgument, errMsg};
    }

    Status FunctionUnImplement(const std::string &errMsg)
    {
        return Status{StatusCode::iFunctionUnImplement, errMsg};
    }

    Status ModelParser(const std::string &errMsg)
    {
        return Status{StatusCode::iModelParseErr, errMsg};
    }

    Status InternalError(const std::string &errMsg)
    {
        return Status{StatusCode::iInternalErr, errMsg};
    }

    Status AlreadyExists(const std::string &errMsg)
    {
        return Status{StatusCode::iAlreadyExist, errMsg};
    }

    std::ostream &operator<<(std::ostream &os, const Status &status)
    {
        os << status.getErrMsg();
        return os;
    }
} // namespace comm
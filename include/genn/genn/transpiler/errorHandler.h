#pragma once

// Standard C++ includes
#include <string_view>

// Mini-parse includes
#include "token.h"

//---------------------------------------------------------------------------
// MiniParse::ErrorHandler
//---------------------------------------------------------------------------
namespace MiniParse
{
class ErrorHandler
{
public:
    virtual void error(size_t line, std::string_view message) = 0;
    virtual void error(const Token &token, std::string_view message) = 0;
};
}

#pragma once

// Standard C++ includes
#include <string_view>

// Transpiler includes
#include "transpiler/token.h"

//---------------------------------------------------------------------------
// GeNN::Transpiler::ErrorHandler
//---------------------------------------------------------------------------
namespace GeNN::Transpiler
{
class ErrorHandler
{
public:
    virtual void error(size_t line, std::string_view message) = 0;
    virtual void error(const Token &token, std::string_view message) = 0;
};
}

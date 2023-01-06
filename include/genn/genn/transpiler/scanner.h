#pragma once

// Standard C++ includes
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// Transpiler includes
#include "transpiler/token.h"

// Forward declarations
namespace GeNN::Transpiler
{
class ErrorHandler;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::Scanner::Error
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Scanner
{
std::vector<Token> scanSource(const std::string_view &source, ErrorHandler &errorHandler);

}   // namespace Scanner
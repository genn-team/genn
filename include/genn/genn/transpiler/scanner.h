#pragma once

// Standard C++ includes
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <variant>
#include <vector>

// Transpiler includes
#include "transpiler/token.h"

// Forward declarations
namespace GeNN::Type
{
class NumericBase;
}
namespace GeNN::Transpiler
{
class ErrorHandlerBase;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::Scanner::Error
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Scanner
{
std::vector<Token> scanSource(const std::string_view &source, const std::unordered_set<std::string> &typedefNames, ErrorHandlerBase &errorHandler);

}   // namespace Scanner

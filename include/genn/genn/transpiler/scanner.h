#pragma once

// Standard C++ includes
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <variant>
#include <vector>

// GeNN includes
#include "type.h"

// Transpiler includes
#include "transpiler/token.h"

// Forward declarations
namespace GeNN::Transpiler
{
class ErrorHandlerBase;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::Scanner::Error
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Scanner
{
std::vector<Token> scanSource(const std::string_view &source, const Type::TypeContext &context, ErrorHandlerBase &errorHandler);

}   // namespace Scanner

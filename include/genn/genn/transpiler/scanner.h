#pragma once

// Standard C++ includes
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// Mini-parse includes
#include "token.h"

// Forward declarations
namespace MiniParse
{
class ErrorHandler;
}

//---------------------------------------------------------------------------
// MiniParse::Scanner::Error
//---------------------------------------------------------------------------
namespace MiniParse::Scanner
{
std::vector<Token> scanSource(const std::string_view &source, ErrorHandler &errorHandler);

}   // namespace Scanner
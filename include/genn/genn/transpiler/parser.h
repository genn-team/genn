#pragma once

// Standard C++ includes
#include <memory>
#include <vector>

// Transpiler includes
#include "transpiler/expression.h"
#include "transpiler/statement.h"
#include "transpiler/token.h"

// Forward declarations
namespace GeNN::Transpiler
{
class ErrorHandler;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::Parser
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Parser
{
Expression::ExpressionPtr parseExpression(const std::vector<Token> &tokens, ErrorHandler &errorHandler);

Statement::StatementList parseBlockItemList(const std::vector<Token> &tokens, ErrorHandler &errorHandler);
}   // MiniParse::MiniParse
#pragma once

// Standard C++ includes
#include <memory>
#include <vector>

// Mini-parse includes
#include "expression.h"
#include "statement.h"
#include "token.h"

// Forward declarations
namespace MiniParse
{
class ErrorHandler;
}

//---------------------------------------------------------------------------
// MiniParse::Scanner::Parser
//---------------------------------------------------------------------------
namespace MiniParse::Parser
{
Expression::ExpressionPtr parseExpression(const std::vector<Token> &tokens, ErrorHandler &errorHandler);

Statement::StatementList parseBlockItemList(const std::vector<Token> &tokens, ErrorHandler &errorHandler);
}   // MiniParse::MiniParse
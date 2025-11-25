#pragma once

// Standard C++ includes
#include <memory>
#include <set>
#include <vector>

// GeNN includes
#include "type.h"

// Transpiler includes
#include "transpiler/expression.h"
#include "transpiler/statement.h"
#include "transpiler/token.h"

// Forward declarations
namespace GeNN::Transpiler
{
class ErrorHandlerBase;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::ParseError
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Parser
{
class ParseError : public std::runtime_error
{
public:
    ParseError() : std::runtime_error("")
    {
    }
};

//! Parse expression from tokens
Expression::ExpressionPtr<> parseExpression(const std::vector<Token> &tokens, const Type::TypeContext &context, ErrorHandlerBase &errorHandler);

//! Parse block item list from tokens
/*! Block item lists are function body scope list of statements */
Statement::StatementList<> parseBlockItemList(const std::vector<Token> &tokens, const Type::TypeContext &context, ErrorHandlerBase &errorHandler);

//! Parse type from tokens
const GeNN::Type::ResolvedType parseNumericType(const std::vector<Token> &tokens, const Type::TypeContext &context, ErrorHandlerBase &errorHandler);

}   // MiniParse::MiniParse

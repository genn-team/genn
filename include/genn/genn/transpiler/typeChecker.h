#pragma once

// Standard C++ includes
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

// GeNN includes
#include "type.h"

// Transpiler includes
#include "transpiler/statement.h"

// Forward declarations
namespace GeNN::Transpiler
{
class ErrorHandlerBase;
struct Token;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::TypeCheckError
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::TypeChecker
{
class TypeCheckError : public std::runtime_error
{
public:
    TypeCheckError() : std::runtime_error("")
    {
    }
};

typedef std::unordered_map<const Expression::Base*, Type::Type> ResolvedTypeMap;

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::EnvironmentBase
//---------------------------------------------------------------------------
class EnvironmentBase
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void define(const Token &name, const Type::Type &type, ErrorHandlerBase &errorHandler) = 0;
    virtual std::vector<Type::Type> getTypes(const Token &name, ErrorHandlerBase &errorHandler) = 0;

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    Type::Type getType(const Token &name, ErrorHandlerBase &errorHandler);
};

//---------------------------------------------------------------------------
// Free functions
//---------------------------------------------------------------------------
ResolvedTypeMap typeCheck(const Statement::StatementList &statements, EnvironmentBase &environment, 
                          const Type::TypeContext &context, ErrorHandlerBase &errorHandler);

const Type::Base *typeCheck(const Expression::Base *expression, EnvironmentBase &environment, 
                            const Type::TypeContext &context, ErrorHandlerBase &errorHandler);
}   // namespace MiniParse::GeNN::Transpiler

#pragma once

// Standard C++ includes
#include <functional>
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

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::EnvironmentBase
//---------------------------------------------------------------------------
class EnvironmentBase
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void define(const Token &name, const Type::ResolvedType &type, ErrorHandlerBase &errorHandler) = 0;
    virtual std::vector<Type::ResolvedType> getTypes(const Token &name, ErrorHandlerBase &errorHandler) = 0;

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    Type::ResolvedType getType(const Token &name, ErrorHandlerBase &errorHandler);
};

//---------------------------------------------------------------------------
// Typedefines
//---------------------------------------------------------------------------
typedef std::unordered_map<const Expression::Base*, Type::ResolvedType> ResolvedTypeMap;
typedef std::function<void(EnvironmentBase&, ErrorHandlerBase&)> StatementHandler;

//---------------------------------------------------------------------------
// Free functions
//---------------------------------------------------------------------------
ResolvedTypeMap typeCheck(const Statement::StatementList &statements, EnvironmentBase &environment, 
                          ErrorHandlerBase &errorHandler, StatementHandler forEachSynapseHandler = nullptr);

ResolvedTypeMap typeCheck(const Expression::Base *expression, EnvironmentBase &environment, 
                          ErrorHandlerBase &errorHandler);
}   // namespace GeNN::Transpiler::TypeChecker

#pragma once

// Standard C++ includes
#include <stdexcept>
#include <string_view>
#include <unordered_map>

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
    virtual void define(const Token &name, const Type::QualifiedType &qualifiedType, ErrorHandlerBase &errorHandler) = 0;
    virtual const Type::QualifiedType &assign(const Token &name, Token::Type op, const Type::QualifiedType &assignedType, 
                                      ErrorHandlerBase &errorHandler, bool initializer = false) = 0;
    virtual const Type::QualifiedType &incDec(const Token &name, Token::Type op, ErrorHandlerBase &errorHandler) = 0;
    virtual const Type::QualifiedType &getType(const Token &name, ErrorHandlerBase &errorHandler) = 0;

protected:
    //---------------------------------------------------------------------------
    // Protected API
    //---------------------------------------------------------------------------
    const Type::QualifiedType &assign(const Token &name, Token::Type op, 
                                      const Type::QualifiedType &existingType, const Type::QualifiedType &assignedType, 
                                      ErrorHandlerBase &errorHandler, bool initializer = false) const;
    const Type::QualifiedType &incDec(const Token &name, Token::Type op, 
                                      const Type::QualifiedType &existingType, ErrorHandlerBase &errorHandler) const;
};

//---------------------------------------------------------------------------
// Free functions
//---------------------------------------------------------------------------
void typeCheck(const Statement::StatementList &statements, EnvironmentBase &environment, 
               ErrorHandlerBase &errorHandler);

Type::QualifiedType typeCheck(const Expression::Base *expression, EnvironmentBase &environment, 
                              ErrorHandlerBase &errorHandler);
}   // namespace MiniParse::GeNN::Transpiler

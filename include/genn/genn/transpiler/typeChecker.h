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
class ErrorHandler;
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
    virtual void define(const Token &name, const Type::QualifiedType &qualifiedType, ErrorHandler &errorHandler) = 0;
    virtual const Type::QualifiedType &assign(const Token &name, Token::Type op, const Type::QualifiedType &assignedType, 
                                      ErrorHandler &errorHandler, bool initializer = false) = 0;
    virtual const Type::QualifiedType &incDec(const Token &name, Token::Type op, ErrorHandler &errorHandler) = 0;
    virtual const Type::QualifiedType &getType(const Token &name, ErrorHandler &errorHandler) = 0;

protected:
    //---------------------------------------------------------------------------
    // Protected API
    //---------------------------------------------------------------------------
    const Type::QualifiedType &assign(const Token &name, Token::Type op, 
                                      const Type::QualifiedType &existingType, const Type::QualifiedType &assignedType, 
                                      ErrorHandler &errorHandler, bool initializer = false) const;
    const Type::QualifiedType &incDec(const Token &name, Token::Type op, 
                                      const Type::QualifiedType &existingType, ErrorHandler &errorHandler) const;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::EnvironmentExternal
//---------------------------------------------------------------------------
class EnvironmentExternal : public EnvironmentBase
{
public:
    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    template<typename T>
    void define(std::string_view name, bool isConstValue = false, bool isConstPointer = false)
    {
        if(!m_Types.try_emplace(name, T::getInstance(), isConstValue, isConstPointer).second) {
            throw std::runtime_error("Redeclaration of '" + std::string{name} + "'");
        }
    }

    //---------------------------------------------------------------------------
    // EnvironmentBase virtuals
    //---------------------------------------------------------------------------
    virtual void define(const Token &name, const Type::QualifiedType &qualifiedType, ErrorHandler &errorHandler) final;
    virtual const Type::QualifiedType &assign(const Token &name, Token::Type op, const Type::QualifiedType &assignedType, 
                                              ErrorHandler &errorHandler, bool initializer = false) final;
    virtual const Type::QualifiedType &incDec(const Token &name, Token::Type op, ErrorHandler &errorHandler) final;
    virtual const Type::QualifiedType &getType(const Token &name, ErrorHandler &errorHandler) final;

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    std::unordered_map<std::string_view, Type::QualifiedType> m_Types;
};

//---------------------------------------------------------------------------
// Free functions
//---------------------------------------------------------------------------
void typeCheck(const Statement::StatementList &statements, EnvironmentExternal &environment, 
               ErrorHandler &errorHandler);

Type::QualifiedType typeCheck(const Expression::Base *expression, EnvironmentExternal &environment, 
                              ErrorHandler &errorHandler);
}   // namespace MiniParse::GeNN::Transpiler

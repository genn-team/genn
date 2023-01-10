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
// GeNN::Transpiler::TypeChecker::Environment
//---------------------------------------------------------------------------
class Environment
{
public:
    Environment(Environment *enclosing = nullptr)
    :   m_Enclosing(enclosing)
    {
    }

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
    void define(const Token &name, const Type::QualifiedType &qualifiedType, ErrorHandler &errorHandler);
    const Type::QualifiedType &assign(const Token &name, const Type::QualifiedType &assignedType, 
                                      Token::Type op, ErrorHandler &errorHandler, bool initializer = false);
    const Type::QualifiedType &incDec(const Token &name, const Token &op, ErrorHandler &errorHandler);
    const Type::QualifiedType &getType(const Token &name, ErrorHandler &errorHandler) const;

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    Environment *m_Enclosing;
    std::unordered_map<std::string_view, Type::QualifiedType> m_Types;
};

//---------------------------------------------------------------------------
// Free functions
//---------------------------------------------------------------------------
void typeCheck(const Statement::StatementList &statements, Environment &environment, 
               ErrorHandler &errorHandler);

Type::QualifiedType typeCheck(const Expression::Base *expression, Environment &environment, 
                              ErrorHandler &errorHandler);
}   // namespace MiniParse::GeNN::Transpiler

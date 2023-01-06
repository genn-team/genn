#pragma once

// Standard C++ includes
#include <stdexcept>
#include <string_view>
#include <unordered_map>

// Transpiler includes
#include "transpiler/statement.h"

// Forward declarations
namespace GeNN::Transpiler
{
class ErrorHandler;
struct Token;
}
namespace GeNN::Type
{
class Base;
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
        : m_Enclosing(enclosing)
    {
    }

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    template<typename T>
    void define(std::string_view name, bool isConst = false)
    {
        if(!m_Types.try_emplace(name, T::getInstance(), isConst).second) {
            throw std::runtime_error("Redeclaration of '" + std::string{name} + "'");
        }
    }
    void define(const Token &name, const Type::Base *type, bool isConst, ErrorHandler &errorHandler);
    const Type::Base *assign(const Token &name, const Type::Base *assignedType, bool assignedConst, 
                             Token::Type op, ErrorHandler &errorHandler);
    const Type::Base *incDec(const Token &name, const Token &op, ErrorHandler &errorHandler);
    std::tuple<const Type::Base*, bool> getType(const Token &name, ErrorHandler &errorHandler) const;

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    Environment *m_Enclosing;
    std::unordered_map<std::string_view, std::tuple<const Type::Base*, bool>> m_Types;
};

//---------------------------------------------------------------------------
// Free functions
//---------------------------------------------------------------------------
void typeCheck(const Statement::StatementList &statements, Environment &environment, 
               ErrorHandler &errorHandler);
}   // namespace MiniParse::GeNN::Transpiler

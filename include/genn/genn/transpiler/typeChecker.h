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

typedef std::unordered_map<const Expression::Base*, const Type::Base*> ResolvedTypeMap;

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::EnvironmentBase
//---------------------------------------------------------------------------
class EnvironmentBase
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void define(const Token &name, const Type::Base *type, ErrorHandlerBase &errorHandler) = 0;
    virtual const Type::Base *assign(const Token &name, Token::Type op, const Type::Base *assignedType, 
                                     const Type::TypeContext &context, ErrorHandlerBase &errorHandler, 
                                     bool initializer = false) = 0;
    virtual const Type::Base *incDec(const Token &name, Token::Type op, 
                                     const Type::TypeContext &context, ErrorHandlerBase &errorHandler) = 0;
    virtual std::vector<const Type::Base*> getTypes(const Token &name, ErrorHandlerBase &errorHandler) = 0;

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    const Type::Base *getType(const Token &name, ErrorHandlerBase &errorHandler);

protected:
    //---------------------------------------------------------------------------
    // Protected API
    //---------------------------------------------------------------------------
    const Type::Base *assign(const Token &name, Token::Type op, 
                             const Type::Base *existingType, const Type::Base *assignedType, 
                             const Type::TypeContext &context, ErrorHandlerBase &errorHandler, 
                             bool initializer = false) const;
    const Type::Base *incDec(const Token &name, Token::Type op, 
                             const Type::Base *existingType, ErrorHandlerBase &errorHandler) const;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::StandardLibraryFunctionEnvironment
//---------------------------------------------------------------------------
class StandardLibraryFunctionEnvironment : public EnvironmentBase
{
public:
    StandardLibraryFunctionEnvironment();

    //------------------------------------------------------------------------
    // EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual void define(const Token &name, const Type::Base *type, ErrorHandlerBase &errorHandler) final;
    virtual const Type::Base *assign(const Token &name, Token::Type op, const Type::Base *assignedType, 
                                     const Type::TypeContext &context, ErrorHandlerBase &errorHandler, 
                                     bool initializer = false) final;
    virtual const Type::Base *incDec(const Token &name, Token::Type op, 
                                     const Type::TypeContext &context, ErrorHandlerBase &errorHandler) final;
    virtual std::vector<const Type::Base*> getTypes(const Token &name, ErrorHandlerBase &errorHandler) final;

private:
    std::unordered_multimap<std::string, const Type::Base*> m_Types;
};

//---------------------------------------------------------------------------
// Free functions
//---------------------------------------------------------------------------
void typeCheck(const Statement::StatementList &statements, EnvironmentBase &environment, 
               const Type::TypeContext &context, ErrorHandlerBase &errorHandler);

const Type::Base *typeCheck(const Expression::Base *expression, EnvironmentBase &environment, 
                            const Type::TypeContext &context, ErrorHandlerBase &errorHandler);
}   // namespace MiniParse::GeNN::Transpiler

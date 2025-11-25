#pragma once

// Standard C++ includes
#include <functional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

// GeNN includes
#include "gennExport.h"
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
// GeNN::Transpiler::TypeChecker::TypeAnnotation
//---------------------------------------------------------------------------
class TypeAnnotation
{
public:
    TypeAnnotation(const Type::ResolvedType &type) : m_Type(type)
    {}
    const auto &getType() const{ return m_Type; }
    
private:
    Type::ResolvedType m_Type;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::EnvironmentBase
//---------------------------------------------------------------------------
class GENN_EXPORT EnvironmentBase
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
// GeNN::Transpiler::TypeChecker::EnvironmentInternal
//---------------------------------------------------------------------------
class GENN_EXPORT EnvironmentInternal : public EnvironmentBase
{
public:
    EnvironmentInternal(EnvironmentBase &enclosing)
    :   m_Enclosing(enclosing)
    {
    }

    //---------------------------------------------------------------------------
    // EnvironmentBase virtuals
    //---------------------------------------------------------------------------
    virtual void define(const Token &name, const Type::ResolvedType &type, ErrorHandlerBase &errorHandler) final;
    virtual std::vector<Type::ResolvedType> getTypes(const Token &name, ErrorHandlerBase &errorHandler) final;

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    EnvironmentBase &m_Enclosing;
    std::unordered_map<std::string, Type::ResolvedType> m_Types;
};


//---------------------------------------------------------------------------
// Typedefines
//---------------------------------------------------------------------------
typedef std::function<void(EnvironmentBase&, ErrorHandlerBase&)> StatementHandler;

//---------------------------------------------------------------------------
// Free functions
//---------------------------------------------------------------------------
Statement::StatementList<TypeAnnotation> typeCheck(const Statement::StatementList<> &statements, 
                                                   EnvironmentInternal &environment, 
                                                   const Type::TypeContext &context, 
                                                   ErrorHandlerBase &errorHandler, 
                                                   StatementHandler forEachSynapseHandler = nullptr);

Expression::ExpressionPtr<TypeAnnotation> typeCheck(const Expression::Base<> *expression, 
                                                    EnvironmentInternal &environment, 
                                                    const Type::TypeContext &context, 
                                                    ErrorHandlerBase &errorHandler);
}   // namespace GeNN::Transpiler::TypeChecker

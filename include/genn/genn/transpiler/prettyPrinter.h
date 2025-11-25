#pragma once

// Standard C++ includes
#include <functional>
#include <string>
#include <unordered_set>

// GeNN includes
#include "gennExport.h"
#include "type.h"

// Transpiler includes
#include "transpiler/statement.h"
#include "transpiler/typeChecker.h"

// Forward declarations
namespace GeNN::CodeGenerator
{
class CodeStream;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::PrettyPrinter::EnvironmentBase
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::PrettyPrinter
{
class GENN_EXPORT EnvironmentBase
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Define identifier and return the name as it should be used in code
    virtual std::string define(const std::string &name) = 0;
    
    //! Get the name to use in code for the named identifier
    virtual std::string getName(const std::string &name, std::optional<Type::ResolvedType> type = std::nullopt) = 0;
    
    //! Get stream to write code within this environment to
    virtual CodeGenerator::CodeStream &getStream() = 0;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void print(const std::string &format);
    void printLine(const std::string &format);

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    std::string operator[] (const std::string &name)
    {
        return getName(name);
    }
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::PrettyPrinter::EnvironmentInternal
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
    virtual std::string define(const std::string &name) final;
    virtual std::string getName(const std::string &name, std::optional<Type::ResolvedType> type) final;

    virtual CodeGenerator::CodeStream &getStream()
    {
        return m_Enclosing.getStream();
    }

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    EnvironmentBase &m_Enclosing;
    std::unordered_set<std::string> m_LocalVariables;
};

typedef std::function<void(EnvironmentBase&, std::function<void(EnvironmentBase&)>)> StatementHandler;

//---------------------------------------------------------------------------
// Free functions
//---------------------------------------------------------------------------
void print(const Statement::StatementList<GeNN::Transpiler::TypeChecker::TypeAnnotation> &statements, 
           EnvironmentInternal &environment, const Type::TypeContext &context, StatementHandler forEachSynapseHandler = nullptr);
void print(const Expression::ExpressionPtr<GeNN::Transpiler::TypeChecker::TypeAnnotation> &expression,
           EnvironmentInternal &environment, const Type::TypeContext &context);
}

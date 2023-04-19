#pragma once

// Standard C++ includes
#include <string>

// GeNN includes
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
class EnvironmentBase
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Define named variable and return the name as it should be used in code
    virtual std::string define(const std::string &name) = 0;
    
    //! Get the name to use in code for the variable named by token
    virtual std::string getName(const std::string &name, const Type::Base *type = nullptr) = 0;
    
    //! Get stream to write code within this environment to
    virtual CodeGenerator::CodeStream &getStream() = 0;
};

//---------------------------------------------------------------------------
// Free functions
//---------------------------------------------------------------------------
void print(const Statement::StatementList &statements, EnvironmentBase &environment, 
           const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes);
}

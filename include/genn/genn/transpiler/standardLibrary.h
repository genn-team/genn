#pragma once

// Standard C++ includes
#include <unordered_map>
#include <vector>

// Code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/environment.h"

// Transpiler includes
#include "transpiler/typeChecker.h"

//---------------------------------------------------------------------------
// GeNN::Transpiler::StandardLibrary::FunctionTypes
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::StandardLibrary
{
class FunctionTypes : public TypeChecker::EnvironmentBase
{
public:
    FunctionTypes();

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
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::StandardLibrary::FunctionEnvironment
//---------------------------------------------------------------------------
class FunctionEnvironment : public CodeGenerator::EnvironmentExternal
{
public:
    FunctionEnvironment(CodeGenerator::CodeStream &os)
    :   CodeGenerator::EnvironmentExternal(os)
    {}

     //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string getName(const std::string &name) final;
    virtual CodeGenerator::CodeStream &getStream() final;
};
}   // namespace GeNN::Transpiler::StandardLibrary

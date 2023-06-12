#pragma once

// Standard C++ includes
#include <unordered_map>
#include <vector>

// Code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/environment.h"

//---------------------------------------------------------------------------
// GeNN::CodeGenerator::StandardLibrary::FunctionTypes
//---------------------------------------------------------------------------
namespace GeNN::CodeGenerator::StandardLibrary
{
class Environment : public EnvironmentExternal
{
public:
    //------------------------------------------------------------------------
    // TypeChecker::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::vector<Type::ResolvedType> getTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler) final;

    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string getName(const std::string &name, std::optional<Type::ResolvedType> type = std::nullopt) final;
    virtual CodeGenerator::CodeStream &getStream() final;
};
}   // namespace GeNN::CodeGenerator::StandardLibrary

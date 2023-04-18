#pragma once

// Standard C++ includes
#include <unordered_map>
#include <vector>

// Transpiler includes
#include "transpiler/typeChecker.h"

//---------------------------------------------------------------------------
// GeNN::Transpiler::StandardLibrary::TypeEnvironment
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::StandardLibrary
{
class TypeEnvironment : public TypeChecker::EnvironmentBase
{
public:
    TypeEnvironment();

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
}   // namespace GeNN::Transpiler::StandardLibrary

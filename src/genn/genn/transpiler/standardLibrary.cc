#include "transpiler/standardLibrary.h"

// Standard C++ library
#include <memory>

using namespace GeNN::Transpiler::standardLibrary;

//#define ADD_FLOAT_DOUBLE(NAME, CLASS_PREFIX) {#NAME, Type::CLASS_PREFIX##F::getInstance()}, {#NAME, Type::CLASS_PREFIX##D::getInstance()}

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
const std::unordered_multimap<std::string, std::pair<std::unique_ptr<const Type::Base>, std::string>> libraryTypes{
};
}


//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::TypeEnvironment
//---------------------------------------------------------------------------
TypeEnvironment::TypeEnvironment()
{
}
//------------------------------------------------------------------------
void TypeEnvironment::define(const Token &name, const Type::Base*, ErrorHandlerBase &errorHandler)
{
    errorHandler.error(name, "Cannot declare variable in external environment");
    throw TypeCheckError();
}
//---------------------------------------------------------------------------
const Type::Base *StandardLibraryFunctionEnvironment::assign(const Token &name, Token::Type, const Type::Base*,
                                                             const Type::TypeContext&, ErrorHandlerBase &errorHandler, bool)
{
    errorHandler.error(name, "Cannot assign variable in external environment");
    throw TypeCheckError();
}
//---------------------------------------------------------------------------
const Type::Base *StandardLibraryFunctionEnvironment::incDec(const Token &name, Token::Type, const Type::TypeContext&,
                                                             ErrorHandlerBase &errorHandler)
{
    errorHandler.error(name, "Cannot increment/decrement variable in external environment");
    throw TypeCheckError();
}
//---------------------------------------------------------------------------
std::vector<const Type::Base*> StandardLibraryFunctionEnvironment::getTypes(const Token &name, ErrorHandlerBase &errorHandler)
{
    auto [typeBegin, typeEnd] = libraryTypes.equal_range(name.lexeme);
    if (typeBegin == typeEnd) {
         errorHandler.error(name, "Undefined variable");
         throw TypeCheckError();
    }
    else {
        std::vector<const Type::Base*> types;
        types.reserve(std::distance(typeBegin, typeEnd));
        std::transform(typeBegin, typeEnd, std::back_inserter(types),
                       [](auto t) { return t.second.first.get(); });
        return types;
    }
}

#include "transpiler/standardLibrary.h"

// Standard C++ library
#include <algorithm>
#include <iterator>

// GeNN includes
#include "type.h"

// Transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/typeChecker.h"

using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler::StandardLibrary;
using namespace GeNN::Transpiler::TypeChecker;
namespace Type = GeNN::Type;

//---------------------------------------------------------------------------
// Macros
//---------------------------------------------------------------------------
#define ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(NAME)                                                                                 \
    std::make_pair(#NAME, std::make_pair(Type::ResolvedType::createFunction(Type::Float, {Type::Float}), #NAME"($(0))")),     \
    std::make_pair(#NAME, std::make_pair(Type::ResolvedType::createFunction(Type::Double, {Type::Double}), #NAME"($(0))"))

#define ADD_TWO_ARG_FLOAT_DOUBLE_FUNC(NAME)                                                                                                 \
    std::make_pair(#NAME, std::make_pair(Type::ResolvedType::createFunction(Type::Float, {Type::Float, Type::Float}), #NAME"($(0), $(1))")),  \
    std::make_pair(#NAME, std::make_pair(Type::ResolvedType::createFunction(Type::Double, {Type::Double, Type::Double}), #NAME"($(0), $(1))"))

#define ADD_THREE_ARG_FLOAT_DOUBLE_FUNC(NAME)                                                                                                                   \
    std::make_pair(#NAME, std::make_pair(Type::ResolvedType::createFunction(Type::Float, {Type::Float, Type::Float, Type::Float}), #NAME"($(0), $(1), $(2))")),  \
    std::make_pair(#NAME, std::make_pair(Type::ResolvedType::createFunction(Type::Double, {Type::Double, Type::Double, Type::Double}), #NAME"($(0), $(1), $(2))"))

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
template<typename... Args>
auto initLibraryTypes(Args&&... args)
{
    std::unordered_multimap<std::string, std::pair<Type::ResolvedType, std::string>> map;
    (map.emplace(std::forward<Args>(args)), ...);
    return map;
}

const auto libraryTypes = initLibraryTypes(
    // Trigonometric functions
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(cos),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(sin),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(tan),

    // Inverse trigonometric functions
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(acos),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(asin),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(atan),
    ADD_TWO_ARG_FLOAT_DOUBLE_FUNC(atan2),

    // Hyperbolic functions
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(cosh),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(sinh),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(tanh),

    // Inverse Hyperbolic functions
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(acosh),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(asinh),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(atanh),

    // Exponential functions
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(exp),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(expm1),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(exp2),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(pow),
    std::make_pair("scalbn", std::make_pair(Type::ResolvedType::createFunction(Type::Float, {Type::Float, Type::Int32}), "scalbn($(0), $(1))")),
    std::make_pair("scalbn", std::make_pair(Type::ResolvedType::createFunction(Type::Double, {Type::Double, Type::Int32}), "scalbn($(0), $(1))")),

    // Logarithm functions
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(log),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(log1p),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(log2),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(log10),
    std::make_pair("ldexp", std::make_pair(Type::ResolvedType::createFunction(Type::Float, {Type::Float, Type::Int32}), "ldexp($(0), $(1))")),
    std::make_pair("ldexp", std::make_pair(Type::ResolvedType::createFunction(Type::Double, {Type::Double, Type::Int32}), "ldexp($(0), $(1))")),
    std::make_pair("ilogb", std::make_pair(Type::ResolvedType::createFunction(Type::Int32, {Type::Float}), "ilogb($(0))")),
    std::make_pair("ilogb", std::make_pair(Type::ResolvedType::createFunction(Type::Int32, {Type::Double}), "ilogb($(0))")),

    // Root functions
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(sqrt),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(cbrt),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(hypot),

    // Rounding functions
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(ceil),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(floor),
    ADD_TWO_ARG_FLOAT_DOUBLE_FUNC(fmod),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(round),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(rint),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(trunc),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(nearbyint),
    ADD_TWO_ARG_FLOAT_DOUBLE_FUNC(nextafter),
    ADD_TWO_ARG_FLOAT_DOUBLE_FUNC(remainder),

    // Range functions
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(fabs),
    ADD_TWO_ARG_FLOAT_DOUBLE_FUNC(fdim),
    ADD_TWO_ARG_FLOAT_DOUBLE_FUNC(fmax),
    ADD_TWO_ARG_FLOAT_DOUBLE_FUNC(fmin),

    // Other functions
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(erf),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(erfc),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(tgamma),
    ADD_ONE_ARG_FLOAT_DOUBLE_FUNC(lgamma),
    ADD_TWO_ARG_FLOAT_DOUBLE_FUNC(copysign),
    ADD_THREE_ARG_FLOAT_DOUBLE_FUNC(fma));
}


/*{,
{"frexp", "frexpf"},    // pointer arguments
{"modf", "modff"},      // pointer arguments
{"scalbln", "scalblnf"},    // long type
{"lround", "lroundf"},  // long return type
{"lrint", "lrintf"},    // long return type
{"remquo", "remquof"},  // pointer arguments
*/
//min, max, printf

//---------------------------------------------------------------------------
// GeNN::Transpiler::StandardLibrary::FunctionTypes
//---------------------------------------------------------------------------
FunctionTypes::FunctionTypes()
{
}
//------------------------------------------------------------------------
void FunctionTypes::define(const Token &name, const Type::ResolvedType&, ErrorHandlerBase &errorHandler)
{
    errorHandler.error(name, "Cannot declare variable in external environment");
    throw TypeCheckError();
}
//---------------------------------------------------------------------------
std::vector<Type::ResolvedType> FunctionTypes::getTypes(const Token &name, ErrorHandlerBase &errorHandler)
{
    const auto [typeBegin, typeEnd] = libraryTypes.equal_range(name.lexeme);
    if (typeBegin == typeEnd) {
         errorHandler.error(name, "Undefined variable");
         throw TypeCheckError();
    }
    else {
        std::vector<Type::ResolvedType> types;
        types.reserve(std::distance(typeBegin, typeEnd));
        std::transform(typeBegin, typeEnd, std::back_inserter(types),
                       [](const auto &t) { return t.second.first; });
        return types;
    }
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::StandardLibrary::FunctionEnvironment
//---------------------------------------------------------------------------
std::string FunctionEnvironment::getName(const std::string &name, std::optional<Type::ResolvedType> type)
{
    const auto [libTypeBegin, libTypeEnd] = libraryTypes.equal_range(name);
    if (libTypeBegin == libTypeEnd) {
        return getContextName(name, type);
    }
    else {
        if (!type) {
            throw std::runtime_error("Ambiguous reference to '" + name + "' but no type provided to disambiguate");
        }
        const auto libType = std::find_if(libTypeBegin, libTypeEnd,
                                          [type](const auto &t){ return t.second.first == type; });
        assert(libType != libTypeEnd);
        return libType->second.second;
    }
}
//---------------------------------------------------------------------------
CodeStream &FunctionEnvironment::getStream()
{
    return getContextStream();
}
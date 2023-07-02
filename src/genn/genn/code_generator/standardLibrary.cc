#include "code_generator/standardLibrary.h"

// Standard C++ library
#include <algorithm>
#include <iterator>

// GeNN includes
#include "type.h"

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
    GeNN::CodeGenerator::EnvironmentLibrary::Library map;
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
    ADD_TWO_ARG_FLOAT_DOUBLE_FUNC(pow),
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
    ADD_THREE_ARG_FLOAT_DOUBLE_FUNC(fma),
    
    // Printf
    std::make_pair("printf", std::make_pair(Type::ResolvedType::createFunction(Type::Int32, {Type::Int8.addQualifier(Type::Qualifier::CONSTANT).createPointer()}, true), "printf($(0), $(@))")));
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


const GeNN::CodeGenerator::EnvironmentLibrary::Library &GeNN::CodeGenerator::StandardLibrary::getFunctions()
{
    return libraryTypes;
}

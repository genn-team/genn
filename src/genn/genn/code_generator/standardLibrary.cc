#include "code_generator/standardLibrary.h"

// Standard C++ library
#include <algorithm>
#include <iterator>

// GeNN includes
#include "type.h"

namespace Type = GeNN::Type;

using namespace GeNN::CodeGenerator;

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

#define ADD_TWO_ARG_INT_FUNC(NAME)                                                                                                              \
    std::make_pair(#NAME, std::make_pair(Type::ResolvedType::createFunction(Type::Int32, {Type::Int32, Type::Int32}),#NAME"($(0), $(1))")),     \
    std::make_pair(#NAME, std::make_pair(Type::ResolvedType::createFunction(Type::Int64, {Type::Int64, Type::Int64}), #NAME"($(0), $(1))")),    \
    std::make_pair(#NAME, std::make_pair(Type::ResolvedType::createFunction(Type::Uint32, {Type::Uint32, Type::Uint32}), #NAME"($(0), $(1))")), \
    std::make_pair(#NAME, std::make_pair(Type::ResolvedType::createFunction(Type::Uint64, {Type::Uint64, Type::Uint64}), #NAME"($(0), $(1))"))


//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
template<typename... Args>
auto initLibraryTypes(Args&&... args)
{
    EnvironmentLibrary::Library map;
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

    // Integer functions
    ADD_TWO_ARG_INT_FUNC(min),
    ADD_TWO_ARG_INT_FUNC(max),
    std::make_pair("abs", std::make_pair(Type::ResolvedType::createFunction(Type::Int32, {Type::Int32}), "abs($(0))")),
    std::make_pair("abs", std::make_pair(Type::ResolvedType::createFunction(Type::Int64, {Type::Int64}), "abs($(0))")),

    // Printf
    std::make_pair("printf", std::make_pair(Type::ResolvedType::createFunction(Type::Int32, {Type::Int8.addConst().createPointer()}, Type::FunctionFlags::VARIADIC), "printf($(0), $(@))")),

    // Assert
    std::make_pair("assert", std::make_pair(Type::ResolvedType::createFunction(Type::Void, {Type::Bool}), "assert($(0))")));
}

const EnvironmentLibrary::Library floatRandomFunctions = {
    {"gennrand", {Type::ResolvedType::createFunction(Type::Uint32, {}), "hostRNG()"}},
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Float, {}), "standardUniformDistribution(hostRNG)"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Float, {}), "standardNormalDistribution(hostRNG)"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Float, {}), "standardExponentialDistribution(hostRNG)"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Float, {Type::Float, Type::Float}), "std::lognormal_distribution<float>($(0), $(1))(hostRNG)"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Float, {Type::Float}), "std::gamma_distribution<float>($(0), 1.0f)(hostRNG)"}},
    {"gennrand_binomial", {Type::ResolvedType::createFunction(Type::Uint32, {Type::Uint32, Type::Float}), "std::binomial_distribution<unsigned int>($(0), $(1))(hostRNG)"}},
};

const EnvironmentLibrary::Library doubleRandomFunctions = {
    {"gennrand", {Type::ResolvedType::createFunction(Type::Uint32, {}), "hostRNG()"}},
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Double, {}), "standardUniformDistribution(hostRNG)"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Double, {}), "standardNormalDistribution(hostRNG)"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Double, {}), "standardExponentialDistribution(hostRNG)"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Double, {Type::Double, Type::Double}), "std::lognormal_distribution<double>($(0), $(1))(hostRNG)"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Double, {Type::Double}), "std::gamma_distribution<double>($(0), 1.0)(hostRNG)"}},
    {"gennrand_binomial", {Type::ResolvedType::createFunction(Type::Uint32, {Type::Uint32, Type::Double}), "std::binomial_distribution<unsigned int>($(0), $(1))(hostRNG)"}},
};

//---------------------------------------------------------------------------
// GeNN::CodeGenerator::StandardLibrary::FunctionTypes
//---------------------------------------------------------------------------
namespace GeNN::CodeGenerator::StandardLibrary
{
const EnvironmentLibrary::Library &getMathsFunctions()
{
    return libraryTypes;
}

const EnvironmentLibrary::Library &getHostRNGFunctions(const Type::ResolvedType &precision)
{
    if(precision == Type::Float) {
        return floatRandomFunctions;
    }
    else {
        assert(precision == Type::Double);
        return doubleRandomFunctions;
    }
}
}   // namespace GeNN::CodeGenerator::StandardLibrary

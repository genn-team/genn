#pragma once

// Standard includes
#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include <regex>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "neuronGroupInternal.h"
#include "type.h"
#include "varLocation.h"

// GeNN code generator includes
#include "backendBase.h"
#include "codeStream.h"
#include "lazyString.h"
#include "teeStream.h"

// GeNN transpiler includes
#include "transpiler/prettyPrinter.h"
#include "transpiler/statement.h"
#include "transpiler/typeChecker.h"

// Forward declarations
namespace GeNN::CodeGenerator
{
class EnvironmentExternalBase;
}
namespace GeNN::Transpiler
{
class ErrorHandler;
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
//! Divide two integers, rounding up i.e. effectively taking ceil
inline size_t ceilDivide(size_t numerator, size_t denominator)
{
    return ((numerator + denominator - 1) / denominator);
}

//! Pad an integer to a multiple of another
inline size_t padSize(size_t size, size_t blockSize)
{
    return ceilDivide(size, blockSize) * blockSize;
}

GENN_EXPORT void genTypeRange(CodeStream &os, const Type::ResolvedType &type, const std::string &prefix);

//! Parse, type check and pretty print previously scanned vector of tokens representing an expression
GENN_EXPORT void prettyPrintExpression(const std::vector<Transpiler::Token> &tokens, const Type::TypeContext &typeContext, 
                                       Transpiler::TypeChecker::EnvironmentInternal &typeCheckEnv, Transpiler::PrettyPrinter::EnvironmentInternal &prettyPrintEnv,
                                       Transpiler::ErrorHandler &errorHandler);

//! Parse, type check and pretty print previously scanned vector of tokens representing an expression
GENN_EXPORT void prettyPrintExpression(const std::vector<Transpiler::Token> &tokens, const Type::TypeContext &typeContext, 
                                       EnvironmentExternalBase &env, Transpiler::ErrorHandler &errorHandler);

//! Parse, type check and pretty print previously scanned vector of tokens representing a statement
GENN_EXPORT void prettyPrintStatements(const std::vector<Transpiler::Token> &tokens, const Type::TypeContext &typeContext,
                                       Transpiler::TypeChecker::EnvironmentInternal &typeCheckEnv, Transpiler::PrettyPrinter::EnvironmentInternal &prettyPrintEnv,
                                       Transpiler::ErrorHandler &errorHandler, Transpiler::TypeChecker::StatementHandler forEachSynapseTypeCheckHandler = nullptr,
                                       Transpiler::PrettyPrinter::StatementHandler forEachSynapsePrettyPrintHandler = nullptr);

//! Parse, type check and pretty print previously scanned vector of tokens representing a statement
GENN_EXPORT void prettyPrintStatements(const std::vector<Transpiler::Token> &tokens, const Type::TypeContext &typeContext, EnvironmentExternalBase &env, 
                                       Transpiler::ErrorHandler &errorHandler, Transpiler::TypeChecker::StatementHandler forEachSynapseTypeCheckHandler = nullptr,
                                       Transpiler::PrettyPrinter::StatementHandler forEachSynapsePrettyPrintHandler = nullptr);

GENN_EXPORT std::string printSubs(const std::string &format, Transpiler::PrettyPrinter::EnvironmentBase &env);

template<typename G>
bool isKernelSizeHeterogeneous(const G &group, size_t dimensionIndex)
{
    // Get size of this kernel dimension for archetype
    const unsigned archetypeValue = group.getArchetype().getKernelSize().at(dimensionIndex);

    // Return true if any of the other groups have a different value
    return std::any_of(group.getGroups().cbegin(), group.getGroups().cend(),
                       [archetypeValue, dimensionIndex]
                       (const typename G::GroupInternal& g)
                       {
                           return (g.getKernelSize().at(dimensionIndex) != archetypeValue);
                       });
}

template<typename G>
std::string getKernelSize(const G &group, size_t dimensionIndex)
{
    // If kernel size if heterogeneous in this dimension, return group structure entry
    if (isKernelSizeHeterogeneous(group, dimensionIndex)) {
        return "$(_kernel_size_" + std::to_string(dimensionIndex) + ")";
    }
    // Otherwise, return literal
    else {
        return std::to_string(group.getArchetype().getKernelSize().at(dimensionIndex));
    }
}

template<typename G>
std::string getKernelIndex(const G &group)
{
    // Loop through kernel dimensions to calculate array index
    const auto &kernelSize = group.getArchetype().getKernelSize();
    std::ostringstream kernelIndex;
    for (size_t i = 0; i < kernelSize.size(); i++) {
        kernelIndex << "($(id_kernel_" << i << ")";
        // Loop through remainining dimensions of kernel and multiply
        for (size_t j = i + 1; j < kernelSize.size(); j++) {
            kernelIndex << " * " << getKernelSize(group, j);
        }
        kernelIndex << ")";

        // If this isn't the last dimension, add +
        if (i != (kernelSize.size() - 1)) {
            kernelIndex << " + ";
        }
    }

    return kernelIndex.str();
}
}   // namespace GeNN::CodeGenerator

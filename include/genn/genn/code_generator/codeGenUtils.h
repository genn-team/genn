#pragma once

// Standard includes
#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <regex>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "neuronGroupInternal.h"
#include "variableMode.h"

// GeNN code generator includes
#include "backendBase.h"
#include "codeStream.h"
#include "substitutions.h"
#include "teeStream.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------
GENN_EXPORT void substitute(std::string &s, const std::string &trg, const std::string &rep);

//--------------------------------------------------------------------------
//! \brief Tool for substituting variable  names in the neuron code strings or other templates using regular expressions
//--------------------------------------------------------------------------
GENN_EXPORT bool regexVarSubstitute(std::string &s, const std::string &trg, const std::string &rep);

//--------------------------------------------------------------------------
//! \brief Tool for substituting function names in the neuron code strings or other templates using regular expressions
//--------------------------------------------------------------------------
GENN_EXPORT bool regexFuncSubstitute(std::string &s, const std::string &trg, const std::string &rep);

//--------------------------------------------------------------------------
/*! \brief This function substitutes function calls in the form:
 *
 *  $(functionName, parameter1, param2Function(0.12, "string"))
 *
 * with replacement templates in the form:
 *
 *  actualFunction(CONSTANT, $(0), $(1))
 *
 */
//--------------------------------------------------------------------------
GENN_EXPORT void functionSubstitute(std::string &code, const std::string &funcName,
                                    unsigned int numParams, const std::string &replaceFuncTemplate);

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

GENN_EXPORT void genTypeRange(CodeStream &os, const std::string &precision, const std::string &prefix);

//--------------------------------------------------------------------------
/*! \brief This function implements a parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it).
 */
//--------------------------------------------------------------------------
GENN_EXPORT std::string ensureFtype(const std::string &oldcode, const std::string &type);

//--------------------------------------------------------------------------
//! \brief Get the initial value to start reduction operations from
//--------------------------------------------------------------------------
GENN_EXPORT std::string getReductionInitialValue(const BackendBase &backend, VarAccessMode access, const std::string &type);

//--------------------------------------------------------------------------
//! \brief Generate a reduction operation to reduce value into reduction
//--------------------------------------------------------------------------
GENN_EXPORT std::string getReductionOperation(const std::string &reduction, const std::string &value, VarAccessMode access, const std::string &type);

//--------------------------------------------------------------------------
/*! \brief This function checks for unknown variable definitions and returns a gennError if any are found
 */
//--------------------------------------------------------------------------
GENN_EXPORT void checkUnreplacedVariables(const std::string &code, const std::string &codeName);

//--------------------------------------------------------------------------
/*! \brief This function substitutes function names in a code with namespace as prefix of the function name for backends that do not support namespaces by checking that the function indeed exists in the support code and returns the substituted code.
 */
 //--------------------------------------------------------------------------
GENN_EXPORT std::string disambiguateNamespaceFunction(const std::string supportCode, const std::string code, std::string namespaceName);

//-------------------------------------------------------------------------
/*!
  \brief Function for performing the code and value substitutions necessary to insert neuron related variables, parameters, and extraGlobal parameters into synaptic code.
*/
//-------------------------------------------------------------------------
template<typename P, typename D, typename V, typename S>
void neuronSubstitutionsInSynapticCode(CodeGenerator::Substitutions &substitutions, const NeuronGroupInternal *archetypeNG, 
                                       const std::string &delayOffset, const std::string &sourceSuffix, const std::string &destSuffix, 
                                       const std::string &varPrefix, const std::string &varSuffix, bool useLocalNeuronVars,
                                       P isParamHeterogeneousFn, D isDerivedParamHeterogeneousFn, V getVarIndexFn, S getPrevSpikeTimeIndexFn)
{

    // Substitute spike times
    const bool delay = archetypeNG->isDelayRequired();
    const std::string spikeTimeVarIndex = getVarIndexFn(delay, VarAccessDuplication::DUPLICATE);
    const std::string prevSpikeTimeVarIndex = getPrevSpikeTimeIndexFn(delay, VarAccessDuplication::DUPLICATE);
    substitutions.addVarSubstitution("sT" + sourceSuffix,
                                     "(" + delayOffset + varPrefix + "group->sT" + destSuffix + "[" + spikeTimeVarIndex + "]" + varSuffix + ")");
    substitutions.addVarSubstitution("prev_sT" + sourceSuffix,
                                     "(" + delayOffset + varPrefix + "group->prevST" + destSuffix + "[" + prevSpikeTimeVarIndex + "]" + varSuffix + ")");

    // Substitute spike-like-event times
    substitutions.addVarSubstitution("seT" + sourceSuffix,
                                     "(" + delayOffset + varPrefix + "group->seT" + destSuffix + "[" + spikeTimeVarIndex + "]" + varSuffix + ")");
    substitutions.addVarSubstitution("prev_seT" + sourceSuffix,
                                     "(" + delayOffset + varPrefix + "group->prevSET" + destSuffix + "[" + prevSpikeTimeVarIndex + "]" + varSuffix + ")");

    // Substitute neuron variables
    const auto *nm = archetypeNG->getNeuronModel();
    if(useLocalNeuronVars) {
        substitutions.addVarNameSubstitution(nm->getVars(), sourceSuffix, "l");
    }
    else {
        for(const auto &v : nm->getVars()) {
            const std::string varIdx = getVarIndexFn(delay && archetypeNG->isVarQueueRequired(v.name),
                                                     getVarAccessDuplication(v.access));

            substitutions.addVarSubstitution(v.name + sourceSuffix,
                                             varPrefix + "group->" + v.name + destSuffix + "[" + varIdx + "]" + varSuffix);
        }
    }

    // Substitute (potentially heterogeneous) parameters and derived parameters from neuron model
    substitutions.addParamValueSubstitution(nm->getParamNames(), archetypeNG->getParams(), isParamHeterogeneousFn,
                                            sourceSuffix, "group->", destSuffix);
    substitutions.addVarValueSubstitution(nm->getDerivedParams(), archetypeNG->getDerivedParams(), isDerivedParamHeterogeneousFn,
                                          sourceSuffix, "group->", destSuffix);

    // Substitute extra global parameters from neuron model
    substitutions.addVarNameSubstitution(nm->getExtraGlobalParams(), sourceSuffix, "group->", destSuffix);
}

template<typename G, typename K>
bool isKernelSizeHeterogeneous(const G *group, size_t dimensionIndex, K getKernelSizeFn)
{
    // Get size of this kernel dimension for archetype
    const unsigned archetypeValue = getKernelSizeFn(group->getArchetype()).at(dimensionIndex);

    // Return true if any of the other groups have a different value
    return std::any_of(group->getGroups().cbegin(), group->getGroups().cend(),
                       [archetypeValue, dimensionIndex, getKernelSizeFn]
                       (const typename G::GroupInternal& g)
                       {
                           return (getKernelSizeFn(g).at(dimensionIndex) != archetypeValue);
                       });
}

template<typename G, typename K>
std::string getKernelSize(const G *group, size_t dimensionIndex, K getKernelSizeFn)
{
    // If kernel size if heterogeneous in this dimension, return group structure entry
    if (isKernelSizeHeterogeneous(group, dimensionIndex, getKernelSizeFn)) {
        return "group->kernelSize" + std::to_string(dimensionIndex);
    }
    // Otherwise, return literal
    else {
        return std::to_string(getKernelSizeFn(group->getArchetype()).at(dimensionIndex));
    }
}

template<typename G, typename K>
void genKernelIndex(const G *group, std::ostream &os, const CodeGenerator::Substitutions &subs, 
                    K getKernelSizeFn)
{
    // Loop through kernel dimensions to calculate array index
    const auto &kernelSize = getKernelSizeFn(group->getArchetype());
    for (size_t i = 0; i < kernelSize.size(); i++) {
        os << "(" << subs["id_kernel_" + std::to_string(i)];
        // Loop through remainining dimensions of kernel and multiply
        for (size_t j = i + 1; j < kernelSize.size(); j++) {
            os << " * " << getKernelSize(group, j, getKernelSizeFn);
        }
        os << ")";

        // If this isn't the last dimension, add +
        if (i != (kernelSize.size() - 1)) {
            os << " + ";
        }
    }
}
}   // namespace CodeGenerator

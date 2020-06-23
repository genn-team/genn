#pragma once

// Standard includes
#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

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
// CodeGenerator::FunctionTemplate
//--------------------------------------------------------------------------
namespace CodeGenerator
{
//--------------------------------------------------------------------------
// CodeGenerator::MergedStructData
//--------------------------------------------------------------------------
//! Class for storing data generated when writing merged
//! structures in runner and required in later code generation
class MergedStructData
{
public:
    //! Immutable structure for tracking where an extra global variable ends up after merging
    struct MergedEGP
    {
        MergedEGP(size_t m, size_t g, const std::string &t, const std::string &f)
        :   mergedGroupIndex(m), groupIndex(g), type(t), fieldName(f){}

        const size_t mergedGroupIndex;
        const size_t groupIndex;
        const std::string type;
        const std::string fieldName;
    };

    //! Map of original extra global param names to their locations within merged structures
    typedef std::map<std::string, std::unordered_multimap<std::string, MergedEGP>> MergedEGPMap;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const MergedEGPMap &getMergedEGPs() const{ return m_MergedEGPs; }

    void addMergedEGP(const std::string &variableName, const std::string &mergedGroupType,
                      size_t mergedGroupIndex, size_t groupIndex, const std::string &type, const std::string &fieldName)
    {
        m_MergedEGPs[variableName].emplace(
            std::piecewise_construct,
            std::forward_as_tuple(mergedGroupType),
            std::forward_as_tuple(mergedGroupIndex, groupIndex, type, fieldName));
    }
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    MergedEGPMap m_MergedEGPs;
};

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

template<typename T>
void genMergedGroupPush(CodeStream &os, const std::vector<T> &groups, const MergedStructData &mergedStructData,
                        const BackendBase &backend)
{
    
    if(!groups.empty()) {
        // Loop through all extra global parameters to build a set of unique filename, group index pairs
        // **YUCK** it would be much nicer if this were part of the original data structure
        // **NOTE** tuple would be nicer but doesn't define std::hash overload
        std::set<std::pair<size_t, std::pair<std::string, std::string>>> mergedGroupFields;
        for(const auto &e : mergedStructData.getMergedEGPs()) {
            const auto groupEGPs = e.second.equal_range(T::name);
            std::transform(groupEGPs.first, groupEGPs.second, std::inserter(mergedGroupFields, mergedGroupFields.end()),
                           [](const MergedStructData::MergedEGPMap::value_type::second_type::value_type &g)
                           {
                               return std::make_pair(g.second.mergedGroupIndex, 
                                                     std::make_pair(g.second.type, g.second.fieldName));
                           });
        }

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// merged extra global parameter functions" << std::endl;
        os << "// ------------------------------------------------------------------------" << std::endl;
        // Loop through resultant fields and generate push function for pointer extra global parameters
        for(auto f : mergedGroupFields) {
            // If EGP is a pointer
            // **NOTE** this is common to all references!
            if(Utils::isTypePointer(f.second.first)) {
                os << "void pushMerged" << T::name << f.first << f.second.second << "ToDevice(unsigned int idx, " << backend.getMergedGroupFieldHostType(f.second.first) << " value)";
                {
                    CodeStream::Scope b(os);
                    backend.genMergedExtraGlobalParamPush(os, T::name, f.first, "idx", f.second.second, "value");
                }
                os << std::endl;
            }
        }
    }
}


GENN_EXPORT void genScalarEGPPush(CodeStream &os, const MergedStructData &mergedStructData, const std::string &suffix, const BackendBase &backend);

GENN_EXPORT void genTypeRange(CodeStream &os, const std::string &precision, const std::string &prefix);

//--------------------------------------------------------------------------
/*! \brief This function implements a parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it).
 */
//--------------------------------------------------------------------------
GENN_EXPORT std::string ensureFtype(const std::string &oldcode, const std::string &type);


//--------------------------------------------------------------------------
/*! \brief This function checks for unknown variable definitions and returns a gennError if any are found
 */
//--------------------------------------------------------------------------
GENN_EXPORT void checkUnreplacedVariables(const std::string &code, const std::string &codeName);

//-------------------------------------------------------------------------
/*!
  \brief Function for performing the code and value substitutions necessary to insert neuron related variables, parameters, and extraGlobal parameters into synaptic code.
*/
//-------------------------------------------------------------------------
template<typename P, typename D>
void neuronSubstitutionsInSynapticCode(CodeGenerator::Substitutions &substitutions, const NeuronGroupInternal *archetypeNG, 
                                       const std::string &offset, const std::string &delayOffset, const std::string &idx, 
                                       const std::string &sourceSuffix, const std::string &destSuffix, 
                                       const std::string &varPrefix, const std::string &varSuffix,
                                       P isParamHeterogeneousFn, D isDerivedParamHeterogeneousFn)
{

    // Substitute spike times
    substitutions.addVarSubstitution("sT" + sourceSuffix,
                                     "(" + delayOffset + varPrefix + "group->sT" + destSuffix + "[" + offset + idx + "]" + varSuffix + ")");

    // Substitute neuron variables
    const auto *nm = archetypeNG->getNeuronModel();
    for(const auto &v : nm->getVars()) {
        const std::string varIdx = archetypeNG->isVarQueueRequired(v.name) ? offset + idx : idx;

        substitutions.addVarSubstitution(v.name + sourceSuffix,
                                         varPrefix + "group->" + v.name + destSuffix + "[" + varIdx + "]" + varSuffix);
    }

    // Substitute (potentially heterogeneous) parameters and derived parameters from neuron model
    substitutions.addParamValueSubstitution(nm->getParamNames(), archetypeNG->getParams(), isParamHeterogeneousFn,
                                            sourceSuffix, "group->", destSuffix);
    substitutions.addVarValueSubstitution(nm->getDerivedParams(), archetypeNG->getDerivedParams(), isDerivedParamHeterogeneousFn,
                                          sourceSuffix, "group->", destSuffix);

    // Substitute extra global parameters from neuron model
    substitutions.addVarNameSubstitution(nm->getExtraGlobalParams(), sourceSuffix, "group->", destSuffix);
}
}   // namespace CodeGenerator

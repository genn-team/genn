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

    size_t getMergedGroupSize(const std::string &mergedGroupType, size_t mergedGroupIndex) const
    {
        return m_MergedGroupSizes.at(mergedGroupType).at(mergedGroupIndex);
    }

    void addMergedGroupSize(const std::string &mergedGroupType, size_t mergedGroupIndex, size_t sizeBytes)
    {
        m_MergedGroupSizes[mergedGroupType].emplace(mergedGroupIndex, sizeBytes);
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    MergedEGPMap m_MergedEGPs;

    std::unordered_map<std::string, std::map<size_t, size_t>> m_MergedGroupSizes;
};

//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------
void substitute(std::string &s, const std::string &trg, const std::string &rep);

//--------------------------------------------------------------------------
//! \brief Tool for substituting variable  names in the neuron code strings or other templates using regular expressions
//--------------------------------------------------------------------------
bool regexVarSubstitute(std::string &s, const std::string &trg, const std::string &rep);

//--------------------------------------------------------------------------
//! \brief Tool for substituting function names in the neuron code strings or other templates using regular expressions
//--------------------------------------------------------------------------
bool regexFuncSubstitute(std::string &s, const std::string &trg, const std::string &rep);

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
void functionSubstitute(std::string &code, const std::string &funcName,
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

GENN_EXPORT void genParamValVecInit(CodeStream &os, const CodeGenerator::Substitutions &subs, const Snippet::Base::ParamValVec &paramValVec, 
                                    const std::string &errorContext, bool constant = false, bool definition = true, 
                                    const std::string &prefix = "", const std::string &suffix = "");

template<typename T>
void genMergedGroupPush(CodeStream &os, const std::vector<T> &groups, const MergedStructData &mergedStructData,
                        const std::string &suffix, const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces)
{
    // Loop through merged neuron groups
    std::stringstream mergedGroupArrayStream;
    std::stringstream mergedGroupFuncStream;
    CodeStream mergedGroupArray(mergedGroupArrayStream);
    CodeStream mergedGroupFunc(mergedGroupFuncStream);
    TeeStream mergedGroupStreams(mergedGroupArray, mergedGroupFunc);
    for(const auto &g : groups) {
        // Declare static array to hold merged neuron groups
        const size_t idx = g.getIndex();
        const size_t numGroups = g.getGroups().size();

        // Get size of group in bytes
        const size_t groupBytes = mergedStructData.getMergedGroupSize(suffix, idx);

        // Loop through memory spaces
        bool memorySpaceFound = false;
        for(auto &m : memorySpaces) {
            // If there is space in this memory space for group
            if(m.second > groupBytes) {
                // Implement merged group array in this memory space
                backend.genMergedGroupImplementation(mergedGroupArray, m.first, suffix, idx, numGroups);

                // Set flag
                memorySpaceFound = true;

                // Subtract
                m.second -= groupBytes;

                // Stop searching
                break;
            }
        }

        assert(memorySpaceFound);

        // Write function to update
        mergedGroupFunc << "void pushMerged" << suffix << "Group" << idx << "ToDevice(const Merged" << suffix << "Group" << idx << " *group)";
        {
            CodeStream::Scope b(mergedGroupFunc);
            backend.genMergedGroupPush(mergedGroupFunc, suffix, idx, numGroups);
        }
    }

    if(!groups.empty()) {
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// merged group arrays" << std::endl;
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << mergedGroupArrayStream.str();
        os << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// merged group functions" << std::endl;
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << mergedGroupFuncStream.str();
        os << std::endl;

        // Loop through all extra global parameters to build a set of unique filename, group index pairs
        // **YUCK** it would be much nicer if this were part of the original data structure
        // **NOTE** tuple would be nicer but doesn't define std::hash overload
        std::set<std::pair<size_t, std::pair<std::string, std::string>>> mergedGroupFields;
        for(const auto &e : mergedStructData.getMergedEGPs()) {
            const auto groupEGPs = e.second.equal_range(suffix);
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
                os << "void pushMerged" << suffix << f.first << f.second.second << "ToDevice(unsigned int idx, " << f.second.first << " value)";
                {
                    CodeStream::Scope b(os);
                    backend.genMergedExtraGlobalParamPush(os, suffix, f.first, "idx", f.second.second, "value");
                }
                os << std::endl;
            }
        }
    }
}


void genScalarEGPPush(CodeStream &os, const MergedStructData &mergedStructData, const std::string &suffix, const BackendBase &backend);

//--------------------------------------------------------------------------
/*! \brief This function implements a parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it).
 */
//--------------------------------------------------------------------------
std::string ensureFtype(const std::string &oldcode, const std::string &type);


//--------------------------------------------------------------------------
/*! \brief This function checks for unknown variable definitions and returns a gennError if any are found
 */
//--------------------------------------------------------------------------
void checkUnreplacedVariables(const std::string &code, const std::string &codeName);

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
                                     "(" + delayOffset + varPrefix + "group.sT" + destSuffix + "[" + offset + idx + "]" + varSuffix + ")");

    // Substitute neuron variables
    const auto *nm = archetypeNG->getNeuronModel();
    for(const auto &v : nm->getVars()) {
        const std::string varIdx = archetypeNG->isVarQueueRequired(v.name) ? offset + idx : idx;

        substitutions.addVarSubstitution(v.name + sourceSuffix,
                                         varPrefix + "group." + v.name + destSuffix + "[" + varIdx + "]" + varSuffix);
    }

    // Substitute (potentially heterogeneous) parameters and derived parameters from neuron model
    substitutions.addParamValueSubstitution(nm->getParamNames(), archetypeNG->getParams(), isParamHeterogeneousFn,
                                            sourceSuffix, "group.", destSuffix);
    substitutions.addVarValueSubstitution(nm->getDerivedParams(), archetypeNG->getDerivedParams(), isDerivedParamHeterogeneousFn,
                                          sourceSuffix, "group.", destSuffix);

    // Substitute extra global parameters from neuron model
    substitutions.addVarNameSubstitution(nm->getExtraGlobalParams(), sourceSuffix, "group.", destSuffix);
}
}   // namespace CodeGenerator

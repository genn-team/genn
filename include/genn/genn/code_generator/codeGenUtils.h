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
#include "variableMode.h"

// GeNN code generator includes
#include "backendBase.h"
#include "codeStream.h"
#include "teeStream.h"

// Forward declarations
class ModelSpecInternal;
class SynapseGroupInternal;

namespace CodeGenerator
{
class NeuronGroupMerged;
class Substitutions;
class SynapseGroupMerged;
}

//--------------------------------------------------------------------------
// CodeGenerator::FunctionTemplate
//--------------------------------------------------------------------------
namespace CodeGenerator
{
//! Immutable structure for specifying how to implement
//! a generic function e.g. gennrand_uniform
/*! **NOTE** for the sake of easy initialisation first two parameters of GenericFunction are repeated (C++17 fixes) */
struct FunctionTemplate
{
    // **HACK** while GCC and CLang automatically generate this fine/don't require it, VS2013 seems to need it
    FunctionTemplate operator = (const FunctionTemplate &o)
    {
        return FunctionTemplate{o.genericName, o.numArguments, o.doublePrecisionTemplate, o.singlePrecisionTemplate};
    }

    //! Generic name used to refer to function in user code
    const std::string genericName;

    //! Number of function arguments
    const unsigned int numArguments;

    //! The function template (for use with ::functionSubstitute) used when model uses double precision
    const std::string doublePrecisionTemplate;

    //! The function template (for use with ::functionSubstitute) used when model uses single precision
    const std::string singlePrecisionTemplate;
};

//--------------------------------------------------------------------------
// CodeGenerator::FunctionTemplate
//--------------------------------------------------------------------------
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

GENN_EXPORT void genMergedGroupSpikeCountReset(CodeStream &os, const NeuronGroupMerged &n);

template<typename T>
void genMergedGroupPush(CodeStream &os, const std::vector<T> &groups, const MergedEGPMap &mergedEGPs,
                        const std::string &suffix, const BackendBase &backend)
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

        // Implement merged group array
        backend.genMergedGroupImplementation(mergedGroupArray, suffix, idx, numGroups);

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
        for(const auto &e : mergedEGPs) {
            const auto groupEGPs = e.second.equal_range(suffix);
            std::transform(groupEGPs.first, groupEGPs.second, std::inserter(mergedGroupFields, mergedGroupFields.end()),
                           [](const MergedEGPMap::value_type::second_type::value_type &g)
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


void genScalarEGPPush(CodeStream &os, const MergedEGPMap &mergedEGPs, const std::string &suffix, const BackendBase &backend);

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

void neuronSubstitutionsInSynapticCode(
    CodeGenerator::Substitutions &substitutions,
    const NeuronGroupInternal *ng,
    const std::string &offset,
    const std::string &delayOffset,
    const std::string &idx,             //!< index of the neuron to be accessed
    const std::string &sourceSuffix,
    const std::string &destSuffix,
    const std::string &varPrefix = "",  //!< prefix to be used for variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &varSuffix = ""); //!< suffix to be used for variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)

//-------------------------------------------------------------------------
/*!
  \brief Function for performing the code and value substitutions necessary to insert neuron related variables, parameters, and extraGlobal parameters into synaptic code.
*/
//-------------------------------------------------------------------------
void neuronSubstitutionsInSynapticCode(
    Substitutions &substitutions,
    const SynapseGroupInternal &sg,          //!< the synapse group connecting the pre and postsynaptic neuron populations whose parameters might need to be substituted
    const std::string &preIdx,               //!< index of the pre-synaptic neuron to be accessed for _pre variables
    const std::string &postIdx,              //!< index of the post-synaptic neuron to be accessed for _post variables
    double dt,                               //!< simulation timestep (ms)
    const std::string &preVarPrefix = "",    //!< prefix to be used for presynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &preVarSuffix = "",    //!< suffix to be used for presynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
    const std::string &postVarPrefix = "",   //!< prefix to be used for postsynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &postVarSuffix = "");  //!< suffix to be used for postsynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
}   // namespace CodeGenerator

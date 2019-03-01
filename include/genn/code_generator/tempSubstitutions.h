#pragma once

// Standard C++ includes
#include <string>

// Forward declarations
class CurrentSource;
class NeuronGroup;
class SynapseGroup;

namespace Models
{
    class VarInit;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
// **TODO** move all of these SOMWHERE else. Into NeuronGroup and SynapseGroup?
namespace CodeGenerator
{
void applyNeuronModelSubstitutions(std::string &code, const NeuronGroup &ng,
                                   const std::string &varPrefix, const std::string &varSuffix = "", const std::string &varExt = "");

void applyPostsynapticModelSubstitutions(std::string &code, const SynapseGroup &sg, const std::string &varPrefix);

void applyWeightUpdateModelSubstitutions(std::string &code, const SynapseGroup &sg,
                                         const std::string &varPrefix, const std::string &varSuffix = "", const std::string &varExt = "");

void applyCurrentSourceSubstitutions(std::string &code, const CurrentSource &cs,
                                     const std::string &varPrefix);

void applyVarInitSnippetSubstitutions(std::string &code, const Models::VarInit &varInit);

void applySparsConnectInitSnippetSubstitutions(std::string &code, const SynapseGroup &sg);

void preNeuronSubstitutionsInSynapticCode(
    std::string &wCode,                     //!< the code string to work on
    const SynapseGroup &sg,
    const std::string &offset,
    const std::string &axonalDelayOffset,
    const std::string &postIdx,
    const std::string &devPrefix,           //!< device prefix, "dd_" for GPU, nothing for CPU
    const std::string &preVarPrefix = "",   //!< prefix to be used for presynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &preVarSuffix = "");  //!< suffix to be used for presynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)

void postNeuronSubstitutionsInSynapticCode(
    std::string &wCode,                     //!< the code string to work on
    const SynapseGroup &sg,
    const std::string &offset,
    const std::string &backPropDelayOffset,
    const std::string &preIdx,
    const std::string &devPrefix,           //!< device prefix, "dd_" for GPU, nothing for CPU
    const std::string &postVarPrefix = "",  //!< prefix to be used for postsynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &postVarSuffix = ""); //!< suffix to be used for postsynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)

//-------------------------------------------------------------------------
/*!
  \brief Function for performing the code and value substitutions necessary to insert neuron related variables, parameters, and extraGlobal parameters into synaptic code.
*/
//-------------------------------------------------------------------------
void neuronSubstitutionsInSynapticCode(
    std::string &wCode,                     //!< the code string to work on
    const SynapseGroup &sg,                 //!< the synapse group connecting the pre and postsynaptic neuron populations whose parameters might need to be substituted
    const std::string &preIdx,              //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const std::string &postIdx,             //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const std::string &devPrefix,           //!< device prefix, "dd_" for GPU, nothing for CPU
    double dt,                              //!< simulation timestep (ms)
    const std::string &preVarPrefix = "",   //!< prefix to be used for presynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &preVarSuffix = "",   //!< suffix to be used for presynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
    const std::string &postVarPrefix = "",  //!< prefix to be used for postsynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &postVarSuffix = ""); //!< suffix to be used for postsynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
}   // namespace CodeGenerator

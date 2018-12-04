#pragma once

// Standard includes
#include <string>

// GeNN includes
#include "codeGenUtils.h"
#include "initSparseConnectivitySnippet.h"
#include "newNeuronModels.h"

// Forward declarations
class NeuronGroup;
class SynapseGroup;
class CurrentSource;

//----------------------------------------------------------------------------
// NameIterCtx
//----------------------------------------------------------------------------
template<typename Container>
struct NameIterCtx
{
    typedef PairKeyConstIter<typename Container::const_iterator> NameIter;

    NameIterCtx(const Container &c) :
        container(c), nameBegin(std::begin(container)), nameEnd(std::end(container)){}

    const Container container;
    const NameIter nameBegin;
    const NameIter nameEnd;
};

//----------------------------------------------------------------------------
// Typedefines
//----------------------------------------------------------------------------
typedef NameIterCtx<NewModels::Base::StringPairVec> VarNameIterCtx;
typedef NameIterCtx<NewModels::Base::DerivedParamVec> DerivedParamNameIterCtx;
typedef NameIterCtx<NewModels::Base::StringPairVec> ExtraGlobalParamNameIterCtx;

//----------------------------------------------------------------------------
// Standard substitution functins
//----------------------------------------------------------------------------
namespace StandardSubstitutions
{
//! Applies standard set of variable substitutions to postsynaptic model's "apply input" code
void postSynapseApplyInput(
    std::string &psCode,                                    //!< the code string to work on
    const SynapseGroup *sg,                                 //! Synapse group postsynaptic model is used in
    const NeuronGroup &ng,                                  //! The postsynaptic neuron group
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams, //!
    const std::vector<FunctionTemplate> &functions,         //! Appropriate array of platform-specific function templates used to implement platform-specific functions e.g. gennrand_uniform
    const std::string &ftype,                               //! Floating point type used by model e.g. "float"
    const std::string &rng);                                //! Name of the RNG to use for any probabilistic operations

//! Applies standard set of variable substitutions to postsynaptic model's "decay" code
void postSynapseDecay(
    std::string &pdCode,
    const SynapseGroup *sg,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng);

//! Applies standard set of variable substitutions to neuron model's "threshold condition" code
void neuronThresholdCondition(
    std::string &thCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng);

void neuronSim(
    std::string &sCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng);

void neuronSpikeEventCondition(
    std::string &eCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng);

void neuronReset(
    std::string &rCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng);

void weightUpdateThresholdCondition(
    std::string &eCode,
    const SynapseGroup &sg,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    double dt);

void weightUpdateSim(
    std::string &wCode,
    const SynapseGroup &sg,
    const VarNameIterCtx &wuVars,
    const VarNameIterCtx &wuPreVars,
    const VarNameIterCtx &wuPostVars,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    double dt);

void weightUpdateDynamics(
    std::string &SDcode,
    const SynapseGroup *sg,
    const VarNameIterCtx &wuVars,
    const VarNameIterCtx &wuPreVars,
    const VarNameIterCtx &wuPostVars,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    double dt);

void weightUpdatePostLearn(
    std::string &code,
    const SynapseGroup *sg,
    const VarNameIterCtx &wuPreVars,
    const VarNameIterCtx &wuPostVars,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    double dt,
    const string &preVarPrefix = "",    //!< prefix to be used for presynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const string &preVarSuffix = "",    //!< suffix to be used for presynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
    const string &postVarPrefix = "",   //!< prefix to be used for postsynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const string &postVarSuffix = "");  //!< suffix to be used for postsynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)

void weightUpdatePreSpike(
    std::string &code,
    const SynapseGroup *sg,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &devPrefix,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype);


void weightUpdatePostSpike(
    std::string &code,
    const SynapseGroup *sg,
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype);

std::string initNeuronVariable(
    const NewModels::VarInit &varInit,
    const std::string &varName,
    const std::vector<FunctionTemplate> &functions,
    const std::string &idx,
    const std::string &ftype,
    const std::string &rng);

std::string initWeightUpdateVariable(
    const NewModels::VarInit &varInit,
    const std::string &varName,
    const std::vector<FunctionTemplate> &functions,
	const std::string &preIdx,
	const std::string &postIdx,
    const std::string &ftype,
    const std::string &rng);

std::string initSparseConnectivity(
    const SynapseGroup &sg,
    const std::string &addSynapseFunctionTemplate,
    unsigned int numTrgNeurons,
    const std::string &preIdx,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng);

void currentSourceInjection(
    std::string &code,
    const CurrentSource *sc,
    const VarNameIterCtx &scmVars,
    const DerivedParamNameIterCtx &scmDerivedParams,
    const ExtraGlobalParamNameIterCtx &scmExtraGlobalParams,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng);
}   // StandardSubstitions

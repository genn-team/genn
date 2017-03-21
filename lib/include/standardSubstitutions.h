#pragma once

// Standard includes
#include <string>

// GeNN includes
#include "codeGenUtils.h"
#include "newNeuronModels.h"

// Forward declarations
class NeuronGroup;
class NNmodel;
class SynapseGroup;

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


namespace StandardGeneratedSections
{
void neuronOutputInit(
    std::ostream &os,
    const NeuronGroup &ng,
    const std::string &varPrefix);

void neuronLocalVarInit(
    std::ostream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &varPrefix,
    const std::string &localID);

void neuronLocalVarWrite(
    std::ostream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &varPrefix,
    const std::string &localID);

void neuronSpikeEventTest(
    std::ostream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &localID,
    const std::string &ftype);
}
//----------------------------------------------------------------------------
// Standard substitution functins
//----------------------------------------------------------------------------
namespace StandardSubstitutions
{
void postSynapseCurrentConverter(
    std::string &psCode,          //!< the code string to work on
    const std::string &sgName,
    const SynapseGroup *sg,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype);

void postSynapseDecay(
    std::string &pdCode,
    const std::string &sgName,
    const SynapseGroup *sg,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype);

void neuronThresholdCondition(
    std::string &thCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype);

void neuronSim(
    std::string &sCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype);

void neuronSpikeEventCondition(
    std::string &eCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype);

void neuronReset(
    std::string &rCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype);

void weightUpdateThresholdCondition(
    std::string &eCode,
    const std::string &sgName,
    const SynapseGroup &sg,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
   const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype);

void weightUpdateSim(
    std::string &wCode,
    const std::string &sgName,
    const SynapseGroup &sg,
    const VarNameIterCtx &wuVars,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype);

void weightUpdateDynamics(
    std::string &SDcode,
    const SynapseGroup *sg,
    const VarNameIterCtx &wuVars,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype);

void weightUpdatePostLearn(
    std::string &code,
    const std::string &sgName,
    const SynapseGroup *sg,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype);
}   // StandardSubstitions
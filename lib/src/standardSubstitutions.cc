#include "standardSubstitutions.h"

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

//----------------------------------------------------------------------------
// StandardSubstitutions
//----------------------------------------------------------------------------
void StandardSubstitutions::postSynapseCurrentConverter(
    std::string &psCode,          //!< the code string to work on
    const SynapseGroup *sg,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype)
{
    // Create iterators to iterate over the names of the postsynaptic model's initial values
    auto psmVars = VarNameIterCtx(sg->getPSModel()->getVars());
    auto psmDerivedParams = DerivedParamNameIterCtx(sg->getPSModel()->getDerivedParams());

    // Substitute in time parameter
    substitute(psCode, "$(t)", "t");

    name_substitutions(psCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(psCode, ng.getNeuronModel()->getParamNames(), ng.getParams());
    value_substitutions(psCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());

    if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        name_substitutions(psCode, "lps", psmVars.nameBegin, psmVars.nameEnd, sg->getName());
    }
    else {
        value_substitutions(psCode, psmVars.nameBegin, psmVars.nameEnd, sg->getPSInitVals());
    }
    value_substitutions(psCode, sg->getPSModel()->getParamNames(), sg->getPSParams());

    // Create iterators to iterate over the names of the postsynaptic model's derived parameters
    value_substitutions(psCode, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg->getPSDerivedParams());
    name_substitutions(psCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());
    psCode = ensureFtype(psCode, ftype);
    checkUnreplacedVariables(psCode, "postSyntoCurrent");
}

void StandardSubstitutions::postSynapseDecay(
    std::string &pdCode,
    const SynapseGroup *sg,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &,
    const std::string &ftype)
{
    // Create iterators to iterate over the names of the postsynaptic model's initial values
    auto psmVars = VarNameIterCtx(sg->getPSModel()->getVars());
    auto psmDerivedParams = DerivedParamNameIterCtx(sg->getPSModel()->getDerivedParams());

    substitute(pdCode, "$(t)", "t");

    name_substitutions(pdCode, "lps", psmVars.nameBegin, psmVars.nameEnd, sg->getName());
    value_substitutions(pdCode, sg->getPSModel()->getParamNames(), sg->getPSParams());
    value_substitutions(pdCode, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg->getPSDerivedParams());
    name_substitutions(pdCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(pdCode, ng.getNeuronModel()->getParamNames(), ng.getParams());
    value_substitutions(pdCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());

    pdCode = ensureFtype(pdCode, ftype);
    checkUnreplacedVariables(pdCode, "postSynDecay");
}


void StandardSubstitutions::neuronThresholdCondition(
    std::string &thCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype)
{
    substitute(thCode, "$(t)", "t");
    name_substitutions(thCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    substitute(thCode, "$(Isyn)", "Isyn");
    substitute(thCode, "$(sT)", "lsT");
    value_substitutions(thCode, ng.getNeuronModel()->getParamNames(), ng.getParams());
    value_substitutions(thCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    name_substitutions(thCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());
    thCode= ensureFtype(thCode, ftype);
    checkUnreplacedVariables(thCode,"thresholdConditionCode");
}

void StandardSubstitutions::neuronSim(
    std::string &sCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype)
{
    substitute(sCode, "$(t)", "t");
    name_substitutions(sCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(sCode, ng.getNeuronModel()->getParamNames(), ng.getParams());
    value_substitutions(sCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    name_substitutions(sCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());
    substitute(sCode, "$(Isyn)", "Isyn");
    substitute(sCode, "$(sT)", "lsT");
    sCode = ensureFtype(sCode, ftype);
    checkUnreplacedVariables(sCode, "neuron simCode");
}

void StandardSubstitutions::neuronSpikeEventCondition(
    std::string &eCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype)
{
    // code substitutions ----
    substitute(eCode, "$(t)", "t");
    name_substitutions(eCode, "l", nmVars.nameBegin, nmVars.nameEnd, "", "_pre");
    name_substitutions(eCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());
    eCode = ensureFtype(eCode, ftype);
    checkUnreplacedVariables(eCode, "neuronSpkEvntCondition");
}

void StandardSubstitutions::neuronReset(
    std::string &rCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype)
{
    substitute(rCode, "$(t)", "t");
    name_substitutions(rCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(rCode, ng.getNeuronModel()->getParamNames(), ng.getParams());
    value_substitutions(rCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    substitute(rCode, "$(Isyn)", "Isyn");
    substitute(rCode, "$(sT)", "lsT");
    name_substitutions(rCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());
    rCode = ensureFtype(rCode, ftype);
    checkUnreplacedVariables(rCode, "resetCode");
}

void StandardSubstitutions::weightUpdateThresholdCondition(
    std::string &eCode,
    const SynapseGroup &sg,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype)
{
    value_substitutions(eCode, sg.getWUModel()->getParamNames(), sg.getWUParams());
    value_substitutions(eCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(eCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg.getName());
    neuron_substitutions_in_synaptic_code(eCode, &sg, preIdx, postIdx, devPrefix);
    eCode= ensureFtype(eCode, ftype);
    checkUnreplacedVariables(eCode, "evntThreshold");
}

void StandardSubstitutions::weightUpdateSim(
    std::string &wCode,
    const SynapseGroup &sg,
    const VarNameIterCtx &wuVars,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype)
{
    // If the synapse group has global (immutable) state, substitute in the initial values
    if (sg.getMatrixType() & SynapseMatrixWeight::GLOBAL) {
        value_substitutions(wCode, wuVars.nameBegin, wuVars.nameEnd, sg.getWUInitVals());
    }

    // substitute weight update parameter values into weight update code
    value_substitutions(wCode, sg.getWUModel()->getParamNames(), sg.getWUParams());
    value_substitutions(wCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(wCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg.getName());

    // substitute post-synaptic model parameter values into weight update code
    DerivedParamNameIterCtx psmDerivedParams(sg.getPSModel()->getDerivedParams());
    value_substitutions(wCode, sg.getPSModel()->getParamNames(), sg.getPSParams());
    value_substitutions(wCode, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg.getPSDerivedParams());

    substitute(wCode, "$(addtoinSyn)", "addtoinSyn");
    neuron_substitutions_in_synaptic_code(wCode, &sg, preIdx, postIdx, devPrefix);
    wCode= ensureFtype(wCode, ftype);
    checkUnreplacedVariables(wCode, "simCode");
}

void StandardSubstitutions::weightUpdateDynamics(
    std::string &SDcode,
    const SynapseGroup *sg,
    const VarNameIterCtx &wuVars,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype)
{
    // If the synapse group has global (immutable) state, substitute in the initial values
    if (sg->getMatrixType() & SynapseMatrixWeight::GLOBAL) {
        value_substitutions(SDcode, wuVars.nameBegin, wuVars.nameEnd, sg->getWUInitVals());
    }

    // substitute weight update parameter values into synapseDynamics code
    value_substitutions(SDcode, sg->getWUModel()->getParamNames(), sg->getWUParams());
    value_substitutions(SDcode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
    name_substitutions(SDcode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());

    // substitute post-synaptic model parameter values into weight update code
    DerivedParamNameIterCtx psmDerivedParams(sg->getPSModel()->getDerivedParams());
    value_substitutions(SDcode, sg->getPSModel()->getParamNames(), sg->getPSParams());
    value_substitutions(SDcode, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg->getPSDerivedParams());

    substitute(SDcode, "$(addtoinSyn)", "addtoinSyn");
    neuron_substitutions_in_synaptic_code(SDcode, sg, preIdx, postIdx, devPrefix);
    SDcode= ensureFtype(SDcode, ftype);
    checkUnreplacedVariables(SDcode, "synapseDynamics");
}

void StandardSubstitutions::weightUpdatePostLearn(
    std::string &code,
    const SynapseGroup *sg,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype)
{
    value_substitutions(code, sg->getWUModel()->getParamNames(), sg->getWUParams());
    value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
    name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());

    // presynaptic neuron variables and parameters
    neuron_substitutions_in_synaptic_code(code, sg, preIdx, postIdx, devPrefix);
    code= ensureFtype(code, ftype);
    checkUnreplacedVariables(code, "simLearnPost");
}
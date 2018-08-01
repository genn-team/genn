#include "standardSubstitutions.h"

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

//----------------------------------------------------------------------------
// StandardSubstitutions
//----------------------------------------------------------------------------
void StandardSubstitutions::postSynapseApplyInput(
    std::string &psCode,          //!< the code string to work on
    const SynapseGroup *sg,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
{
    // Create iterators to iterate over the names of the postsynaptic model's initial values
    auto psmVars = VarNameIterCtx(sg->getPSModel()->getVars());
    auto psmDerivedParams = DerivedParamNameIterCtx(sg->getPSModel()->getDerivedParams());

    // Substitute in time and standard Isyn parameters
    substitute(psCode, "$(t)", "t");
    substitute(psCode, "$(Isyn)", "Isyn");

    name_substitutions(psCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(psCode, ng.getNeuronModel()->getParamNames(), ng.getParams());
    value_substitutions(psCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());

    if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
        name_substitutions(psCode, "lps", psmVars.nameBegin, psmVars.nameEnd, sg->getName());
    }
    else {
        value_substitutions(psCode, psmVars.nameBegin, psmVars.nameEnd, sg->getPSConstInitVals());
    }
    value_substitutions(psCode, sg->getPSModel()->getParamNames(), sg->getPSParams());

    // Create iterators to iterate over the names of the postsynaptic model's derived parameters
    value_substitutions(psCode, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg->getPSDerivedParams());
    name_substitutions(psCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());

    functionSubstitutions(psCode, ftype, functions);
    substitute(psCode, "$(rng)", rng);
    psCode = ensureFtype(psCode, ftype);
    checkUnreplacedVariables(psCode, sg->getName() + " : postSyntoCurrent");
}

void StandardSubstitutions::postSynapseDecay(
    std::string &pdCode,
    const SynapseGroup *sg,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
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

    functionSubstitutions(pdCode, ftype, functions);
    substitute(pdCode, "$(rng)", rng);
    pdCode = ensureFtype(pdCode, ftype);
    checkUnreplacedVariables(pdCode, sg->getName() + " : postSynDecay");
}


void StandardSubstitutions::neuronThresholdCondition(
    std::string &thCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
{
    substitute(thCode, "$(t)", "t");
    name_substitutions(thCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    substitute(thCode, "$(Isyn)", "Isyn");
    substitute(thCode, "$(sT)", "lsT");
    value_substitutions(thCode, ng.getNeuronModel()->getParamNames(), ng.getParams());
    value_substitutions(thCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    name_substitutions(thCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());

    functionSubstitutions(thCode, ftype, functions);
    substitute(thCode, "$(rng)", rng);
    thCode= ensureFtype(thCode, ftype);
    checkUnreplacedVariables(thCode, ng.getName() + " : thresholdConditionCode");
}

void StandardSubstitutions::neuronSim(
    std::string &sCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
{
    substitute(sCode, "$(t)", "t");
    name_substitutions(sCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(sCode, ng.getNeuronModel()->getParamNames(), ng.getParams());
    value_substitutions(sCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    name_substitutions(sCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());
    substitute(sCode, "$(Isyn)", "Isyn");
    substitute(sCode, "$(sT)", "lsT");

    functionSubstitutions(sCode, ftype, functions);
    substitute(sCode, "$(rng)", rng);
    sCode = ensureFtype(sCode, ftype);
    checkUnreplacedVariables(sCode, ng.getName() + " : neuron simCode");
}

void StandardSubstitutions::neuronSpikeEventCondition(
    std::string &eCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
{
    // code substitutions ----
    substitute(eCode, "$(t)", "t");
    name_substitutions(eCode, "l", nmVars.nameBegin, nmVars.nameEnd, "", "_pre");
    name_substitutions(eCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());

    functionSubstitutions(eCode, ftype, functions);
    substitute(eCode, "$(rng)", rng);
    eCode = ensureFtype(eCode, ftype);
    checkUnreplacedVariables(eCode, ng.getName() + " : neuronSpkEvntCondition");
}

void StandardSubstitutions::neuronReset(
    std::string &rCode,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
{
    substitute(rCode, "$(t)", "t");
    name_substitutions(rCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(rCode, ng.getNeuronModel()->getParamNames(), ng.getParams());
    value_substitutions(rCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    substitute(rCode, "$(Isyn)", "Isyn");
    substitute(rCode, "$(sT)", "lsT");
    name_substitutions(rCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());

    functionSubstitutions(rCode, ftype, functions);
    substitute(rCode, "$(rng)", rng);
    rCode = ensureFtype(rCode, ftype);
    checkUnreplacedVariables(rCode, ng.getName() + " : resetCode");
}

void StandardSubstitutions::weightUpdateThresholdCondition(
    std::string &eCode,
    const SynapseGroup &sg,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype){
    value_substitutions(eCode, sg.getWUModel()->getParamNames(), sg.getWUParams());
    value_substitutions(eCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(eCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg.getName());
    neuron_substitutions_in_synaptic_code(eCode, &sg, preIdx, postIdx, devPrefix);

    functionSubstitutions(eCode, ftype, functions);
    eCode= ensureFtype(eCode, ftype);
    checkUnreplacedVariables(eCode, sg.getName() + " : evntThreshold");
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
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype)
{
     if (sg.getMatrixType() & SynapseMatrixWeight::GLOBAL) {
         value_substitutions(wCode, wuVars.nameBegin, wuVars.nameEnd, sg.getWUConstInitVals());
     }

    value_substitutions(wCode, sg.getWUModel()->getParamNames(), sg.getWUParams());
    value_substitutions(wCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(wCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg.getName());
    substitute(wCode, "$(addtoinSyn)", "addtoinSyn");
    neuron_substitutions_in_synaptic_code(wCode, &sg, preIdx, postIdx, devPrefix);

    functionSubstitutions(wCode, ftype, functions);
    wCode= ensureFtype(wCode, ftype);
    checkUnreplacedVariables(wCode, sg.getName() + " : simCode");
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
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype)
{
     if (sg->getMatrixType() & SynapseMatrixWeight::GLOBAL) {
         value_substitutions(SDcode, wuVars.nameBegin, wuVars.nameEnd, sg->getWUConstInitVals());
     }

    // substitute parameter values for parameters in synapseDynamics code
    value_substitutions(SDcode, sg->getWUModel()->getParamNames(), sg->getWUParams());

    // substitute values for derived parameters in synapseDynamics code
    value_substitutions(SDcode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
    name_substitutions(SDcode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());
    substitute(SDcode, "$(addtoinSyn)", "addtoinSyn");
    neuron_substitutions_in_synaptic_code(SDcode, sg, preIdx, postIdx, devPrefix);

    functionSubstitutions(SDcode, ftype, functions);
    SDcode= ensureFtype(SDcode, ftype);
    checkUnreplacedVariables(SDcode, sg->getName() + " : synapseDynamics");
}

void StandardSubstitutions::weightUpdatePostLearn(
    std::string &code,
    const SynapseGroup *sg,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype)
{
    value_substitutions(code, sg->getWUModel()->getParamNames(), sg->getWUParams());
    value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
    name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());

    // presynaptic neuron variables and parameters
    neuron_substitutions_in_synaptic_code(code, sg, preIdx, postIdx, devPrefix);

    functionSubstitutions(code, ftype, functions);
    code= ensureFtype(code, ftype);
    checkUnreplacedVariables(code, sg->getName() + " : simLearnPost");
}

std::string StandardSubstitutions::initVariable(
    const NewModels::VarInit &varInit,
    const std::string &varName,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
{
    // Get user code string
    std::string code = varInit.getSnippet()->getCode();

    // Substitue derived and standard parameters into init code
    DerivedParamNameIterCtx viDerivedParams(varInit.getSnippet()->getDerivedParams());
    value_substitutions(code, varInit.getSnippet()->getParamNames(), varInit.getParams());
    value_substitutions(code, viDerivedParams.nameBegin, viDerivedParams.nameEnd, varInit.getDerivedParams());

    // Substitute the name of the variable we're initialising
    substitute(code, "$(value)", varName);

    functionSubstitutions(code, ftype, functions);
    substitute(code, "$(rng)", rng);
    code = ensureFtype(code, ftype);
    checkUnreplacedVariables(code, "initVar");

    return code;
}

void StandardSubstitutions::currentSourceInjection(
    std::string &iCode,
    const CurrentSource *cs,
    const VarNameIterCtx &csmVars,
    const DerivedParamNameIterCtx &csmDerivedParams,
    const ExtraGlobalParamNameIterCtx &csmExtraGlobalParams,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
{
    substitute(iCode, "$(t)", "t");
    name_substitutions(iCode, "l", csmVars.nameBegin, csmVars.nameEnd);
    value_substitutions(iCode, cs->getCurrentSourceModel()->getParamNames(), cs->getParams());
    value_substitutions(iCode, csmDerivedParams.nameBegin, csmDerivedParams.nameEnd, cs->getDerivedParams());
    name_substitutions(iCode, "", csmExtraGlobalParams.nameBegin, csmExtraGlobalParams.nameEnd, cs->getName());
    functionSubstitute(iCode, "injectCurrent", 1, "Isyn += $(0)");
    functionSubstitutions(iCode, ftype, functions);
    substitute(iCode, "$(rng)", rng);
    iCode = ensureFtype(iCode, ftype);
    checkUnreplacedVariables(iCode, cs->getName() + " : current source injectionCode");
}

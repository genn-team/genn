#include "standardSubstitutions.h"

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
void initVariable(
    std::string &code,
    const NewModels::VarInit &varInit,
    const std::string &varName,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
{
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
}
}

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
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng)
{
    // Create iterators to iterate over the names of the postsynaptic model's initial values
    VarNameIterCtx psmVars(sg->getPSModel()->getVars());
    DerivedParamNameIterCtx psmDerivedParams(sg->getPSModel()->getDerivedParams());
    ExtraGlobalParamNameIterCtx psmExtraGlobalParams(sg->getPSModel()->getExtraGlobalParams());

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
    name_substitutions(psCode, "", psmExtraGlobalParams.nameBegin, psmExtraGlobalParams.nameEnd, sg->getName());

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
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng)
{
    // Create iterators to iterate over the names of the postsynaptic model's initial values
    VarNameIterCtx psmVars(sg->getPSModel()->getVars());
    DerivedParamNameIterCtx psmDerivedParams(sg->getPSModel()->getDerivedParams());
    ExtraGlobalParamNameIterCtx psmExtraGlobalParams(sg->getPSModel()->getExtraGlobalParams());

    substitute(pdCode, "$(t)", "t");

    name_substitutions(pdCode, "lps", psmVars.nameBegin, psmVars.nameEnd, sg->getName());
    value_substitutions(pdCode, sg->getPSModel()->getParamNames(), sg->getPSParams());
    value_substitutions(pdCode, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg->getPSDerivedParams());
    name_substitutions(pdCode, "", psmExtraGlobalParams.nameBegin, psmExtraGlobalParams.nameEnd, sg->getName());

    name_substitutions(pdCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(pdCode, ng.getNeuronModel()->getParamNames(), ng.getParams());
    value_substitutions(pdCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    name_substitutions(pdCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());

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
    const std::vector<FunctionTemplate> &functions,
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
    const std::vector<FunctionTemplate> &functions,
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
    const std::vector<FunctionTemplate> &functions,
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
    const std::vector<FunctionTemplate> &functions,
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
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    double dt)
{
    value_substitutions(eCode, sg.getWUModel()->getParamNames(), sg.getWUParams());
    value_substitutions(eCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(eCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg.getName());
    neuron_substitutions_in_synaptic_code(eCode, &sg, preIdx, postIdx, devPrefix, dt);
    substitute(eCode, "$(id_pre)", preIdx);
    substitute(eCode, "$(id_post)", postIdx);

    functionSubstitutions(eCode, ftype, functions);
    eCode= ensureFtype(eCode, ftype);
    checkUnreplacedVariables(eCode, sg.getName() + " : evntThreshold");
}

void StandardSubstitutions::weightUpdateSim(
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
    double dt)
{
     if (sg.getMatrixType() & SynapseMatrixWeight::GLOBAL) {
         value_substitutions(wCode, wuVars.nameBegin, wuVars.nameEnd, sg.getWUConstInitVals());
     }

    value_substitutions(wCode, sg.getWUModel()->getParamNames(), sg.getWUParams());
    value_substitutions(wCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(wCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg.getName());

    // Substitute names of pre and postsynaptic weight update variables
    const std::string delayedPreIdx = (sg.getDelaySteps() == NO_DELAY) ? preIdx : "preReadDelayOffset + " + preIdx;
    name_substitutions(wCode, devPrefix, wuPreVars.nameBegin, wuPreVars.nameEnd, sg.getName() + "[" + delayedPreIdx + "]");

    const std::string delayedPostIdx = (sg.getBackPropDelaySteps() == NO_DELAY) ? postIdx : "postReadDelayOffset + " + postIdx;
    name_substitutions(wCode, devPrefix, wuPostVars.nameBegin, wuPostVars.nameEnd, sg.getName() + "[" + delayedPostIdx + "]");

    substitute(wCode, "$(addtoinSyn)", "addtoinSyn");
    neuron_substitutions_in_synaptic_code(wCode, &sg, preIdx, postIdx, devPrefix, dt);
    substitute(wCode, "$(id_pre)", preIdx);
    substitute(wCode, "$(id_post)", postIdx);

    functionSubstitutions(wCode, ftype, functions);
    wCode= ensureFtype(wCode, ftype);
    checkUnreplacedVariables(wCode, sg.getName() + " : simCode");
}

void StandardSubstitutions::weightUpdateDynamics(
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
    double dt)
{
     if (sg->getMatrixType() & SynapseMatrixWeight::GLOBAL) {
         value_substitutions(SDcode, wuVars.nameBegin, wuVars.nameEnd, sg->getWUConstInitVals());
     }

    // substitute parameter values for parameters in synapseDynamics code
    value_substitutions(SDcode, sg->getWUModel()->getParamNames(), sg->getWUParams());

    // Substitute names of pre and postsynaptic weight update variables
    const std::string delayedPreIdx = (sg->getDelaySteps() == NO_DELAY) ? preIdx : "preReadDelayOffset + " + preIdx;
    name_substitutions(SDcode, devPrefix, wuPreVars.nameBegin, wuPreVars.nameEnd, sg->getName() + "[" + delayedPreIdx + "]");

    const std::string delayedPostIdx = (sg->getBackPropDelaySteps() == NO_DELAY) ? postIdx : "postReadDelayOffset + " + postIdx;
    name_substitutions(SDcode, devPrefix, wuPostVars.nameBegin, wuPostVars.nameEnd, sg->getName() + "[" + delayedPostIdx + "]");

    // substitute values for derived parameters in synapseDynamics code
    value_substitutions(SDcode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
    name_substitutions(SDcode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());
    substitute(SDcode, "$(addtoinSyn)", "addtoinSyn");
    neuron_substitutions_in_synaptic_code(SDcode, sg, preIdx, postIdx, devPrefix, dt);
    substitute(SDcode, "$(id_pre)", preIdx);
    substitute(SDcode, "$(id_post)", postIdx);

    functionSubstitutions(SDcode, ftype, functions);
    SDcode= ensureFtype(SDcode, ftype);
    checkUnreplacedVariables(SDcode, sg->getName() + " : synapseDynamics");
}

void StandardSubstitutions::weightUpdatePostLearn(
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
    const string &preVarPrefix,    //!< prefix to be used for presynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const string &preVarSuffix,    //!< suffix to be used for presynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
    const string &postVarPrefix,   //!< prefix to be used for postsynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const string &postVarSuffix)  //!< suffix to be used for postsynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
{
    value_substitutions(code, sg->getWUModel()->getParamNames(), sg->getWUParams());
    value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
    name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());
    substitute(code, "$(id_pre)", code);
    substitute(code, "$(id_post)", code);

    // Substitute names of pre and postsynaptic weight update variables
    const std::string delayedPreIdx = (sg->getDelaySteps() == NO_DELAY) ? preIdx : "preReadDelayOffset + " + preIdx;
    name_substitutions(code, preVarPrefix + devPrefix, wuPreVars.nameBegin, wuPreVars.nameEnd, sg->getName() + "[" + delayedPreIdx + "]" + preVarSuffix, "");

    const std::string delayedPostIdx = (sg->getBackPropDelaySteps() == NO_DELAY) ? postIdx : "postReadDelayOffset + " + postIdx;
    name_substitutions(code, postVarPrefix + devPrefix, wuPostVars.nameBegin, wuPostVars.nameEnd, sg->getName() + "[" + delayedPostIdx + "]" + postVarSuffix, "");

    // presynaptic neuron variables and parameters
    neuron_substitutions_in_synaptic_code(code, sg, preIdx, postIdx, devPrefix, dt,
                                          preVarPrefix, preVarSuffix, postVarPrefix, postVarSuffix);


    functionSubstitutions(code, ftype, functions);
    code= ensureFtype(code, ftype);
    checkUnreplacedVariables(code, sg->getName() + " : simLearnPost");
}

void StandardSubstitutions::weightUpdatePreSpike(
    std::string &code,
    const SynapseGroup *sg,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &devPrefix,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype)
{
    // Create iteration context to iterate over the weight update model
    // postsynaptic variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(sg->getWUModel()->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(sg->getWUModel()->getExtraGlobalParams());
    VarNameIterCtx wuPreVars(sg->getWUModel()->getPreVars());

    // Perform standard substitutions
    substitute(code, "$(t)", "t");

    value_substitutions(code, sg->getWUModel()->getParamNames(), sg->getWUParams());
    value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
    name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());
    name_substitutions(code, "l", wuPreVars.nameBegin, wuPreVars.nameEnd, "");

    const std::string offset = sg->getSrcNeuronGroup()->isDelayRequired() ? "readDelayOffset + " : "";
    preNeuronSubstitutionsInSynapticCode(code, sg, offset, "", preIdx, devPrefix);

    functionSubstitutions(code, ftype, functions);
    code = ensureFtype(code, ftype);
    checkUnreplacedVariables(code, sg->getName() + " : simCodePreSpike");
}

void StandardSubstitutions::weightUpdatePostSpike(
    std::string &code,
    const SynapseGroup *sg,
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype)
{
    // Create iteration context to iterate over the weight update model
    // postsynaptic variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(sg->getWUModel()->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(sg->getWUModel()->getExtraGlobalParams());
    VarNameIterCtx wuPostVars(sg->getWUModel()->getPostVars());

    // Perform standard substitutions
    substitute(code, "$(t)", "t");

    value_substitutions(code, sg->getWUModel()->getParamNames(), sg->getWUParams());
    value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
    name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());

    name_substitutions(code, "l", wuPostVars.nameBegin, wuPostVars.nameEnd, "");

    const std::string offset = sg->getTrgNeuronGroup()->isDelayRequired() ? "readDelayOffset + " : "";
    postNeuronSubstitutionsInSynapticCode(code, sg, offset, "", postIdx, devPrefix);

    functionSubstitutions(code, ftype, functions);
    code = ensureFtype(code, ftype);
    checkUnreplacedVariables(code, sg->getName() + " : simLearnPostSpike");
}

std::string StandardSubstitutions::initNeuronVariable(
    const NewModels::VarInit &varInit,
    const std::string &varName,
    const std::vector<FunctionTemplate> &functions,
    const std::string &idx,
    const std::string &ftype,
    const std::string &rng)
{
    // Get user code string
    std::string code = varInit.getSnippet()->getCode();

    // Substitute in neuron id
    substitute(code, "$(id)", idx);

    // Substitute in initalisation code
    initVariable(code, varInit, varName, functions, ftype, rng);
    return code;
}

std::string StandardSubstitutions::initWeightUpdateVariable(
    const NewModels::VarInit &varInit,
    const std::string &varName,
    const std::vector<FunctionTemplate> &functions,
    const std::string &preIdx,
    const std::string &postIdx,
    const std::string &ftype,
    const std::string &rng)
{
    // Get user code string
    std::string code = varInit.getSnippet()->getCode();

    // Substitute in pre and postsynaptic indices
    substitute(code, "$(id_pre)", preIdx);
    substitute(code, "$(id_post)", postIdx);

    // Substitute in initalisation code
    initVariable(code, varInit, varName, functions, ftype, rng);
    return code;
}

std::string StandardSubstitutions::initSparseConnectivity(
    const SynapseGroup &sg,
    const std::string &addSynapseFunctionTemplate,
    unsigned int numTrgNeurons,
    const std::string &preIdx,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng)
{
    // Get connection initialiser
    const auto &connectInit = sg.getConnectivityInitialiser();

    // Get user code string
    std::string code = connectInit.getSnippet()->getRowBuildCode();

    // Substitute presynaptic index and number of postsynaptic neurons
    substitute(code, "$(id_pre)", preIdx);
    substitute(code, "$(num_post)", std::to_string(numTrgNeurons));

    // Replace endRow() with break to stop loop
    functionSubstitute(code, "endRow", 0, "break");

    // Replace addSynapse(j) with template to increment count var
    functionSubstitute(code, "addSynapse", 1, addSynapseFunctionTemplate);

    // Substitue derived and standard parameters into init code
    DerivedParamNameIterCtx viDerivedParams(connectInit.getSnippet()->getDerivedParams());
    ExtraGlobalParamNameIterCtx viExtraGlobalParams(connectInit.getSnippet()->getExtraGlobalParams());
    value_substitutions(code, connectInit.getSnippet()->getParamNames(), connectInit.getParams());
    value_substitutions(code, viDerivedParams.nameBegin, viDerivedParams.nameEnd, connectInit.getDerivedParams());
    name_substitutions(code, "initSparseConn", viExtraGlobalParams.nameBegin, viExtraGlobalParams.nameEnd, sg.getName());

    // Perform standard substitutions
    functionSubstitutions(code, ftype, functions);
    substitute(code, "$(rng)", rng);
    code = ensureFtype(code, ftype);
    checkUnreplacedVariables(code, "initSparseConnectivity");

    return code;
}

void StandardSubstitutions::currentSourceInjection(
    std::string &iCode,
    const CurrentSource *cs,
    const VarNameIterCtx &csmVars,
    const DerivedParamNameIterCtx &csmDerivedParams,
    const ExtraGlobalParamNameIterCtx &csmExtraGlobalParams,
    const std::vector<FunctionTemplate> &functions,
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

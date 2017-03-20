#include "standardSubstitutions.h"

// GeNN includes
#include "CodeHelper.h"
#include "modelSpec.h"

//----------------------------------------------------------------------------
// StandardGeneratedSections
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronOutputInit(
    std::ostream &os,
    const std::string &ngName,
    const NeuronGroup &ng,
    const std::string &varPrefix)
{
    if (ng.isDelayRequired()) { // with delay
        os << varPrefix << "spkQuePtr" << ngName << " = (" << varPrefix << "spkQuePtr" << ngName << " + 1) % " << ng.getNumDelaySlots() << ";" << ENDL;
        if (ng.isSpikeEventRequired()) {
            os << varPrefix << "glbSpkCntEvnt" << ngName << "[" << varPrefix << "spkQuePtr" << ngName << "] = 0;" << ENDL;
        }
        if (ng.isTrueSpikeRequired()) {
            os << varPrefix << "glbSpkCnt" << ngName << "[" << varPrefix << "spkQuePtr" << ngName << "] = 0;" << ENDL;
        }
        else {
            os << varPrefix << "glbSpkCnt" << ngName << "[0] = 0;" << ENDL;
        }
    }
    else { // no delay
        if (ng.isSpikeEventRequired()) {
            os << varPrefix << "glbSpkCntEvnt" << ngName << "[0] = 0;" << ENDL;
        }
        os << varPrefix << "glbSpkCnt" << ngName << "[0] = 0;" << ENDL;
    }
}

void StandardGeneratedSections::neuronLocalVarInit(
    std::ostream &os,
     const std::string &ngName,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &varPrefix,
    const std::string &localID)
{
    for (size_t k = 0; k < nmVars.container.size(); k++) {
        os << nmVars.container[k].second << " l" << nmVars.container[k].first << " = ";
        os << varPrefix << nmVars.container[k].first << ngName << "[";
        if (ng.isVarQueueRequired(k) && ng.isDelayRequired()) {
            os << "(delaySlot * " << ng.getNumNeurons() << ") + ";
        }
        os << localID << "];" << ENDL;
    }
}

void StandardGeneratedSections::neuronLocalVarWrite(
    std::ostream &os,
    const std::string &ngName,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &varPrefix,
    const std::string &localID)
{
    // store the defined parts of the neuron state into the global state variables dd_V etc
    for (size_t k = 0, l = nmVars.container.size(); k < l; k++) {
        if (ng.isVarQueueRequired(k)) {
            os << varPrefix << nmVars.container[k].first << ngName << "[" << ng.getQueueOffset(ngName, varPrefix) << localID << "] = l" << nmVars.container[k].first << ";" << ENDL;
        }
        else {
            os << varPrefix << nmVars.container[k].first << ngName << "[" << localID << "] = l" << nmVars.container[k].first << ";" << ENDL;
        }
    }
}

void StandardGeneratedSections::neuronSpikeEventTest(
    std::ostream &os,
    const std::string &ngName,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &,
    const std::string &ftype)
{
    // Create local variable
    os << "bool spikeLikeEvent = false;" << ENDL;

    // Loop through outgoing synapse populations that will contribute to event condition code
    for(const auto &spkEventCond : ng.getSpikeEventCondition()) {
        // Replace of parameters, derived parameters and extraglobalsynapse parameters
        string eCode = spkEventCond.first;

        // code substitutions ----
        substitute(eCode, "$(id)", "n");
        StandardSubstitutions::neuronSpikeEventCondition(eCode, ngName,
                                                         nmVars, nmExtraGlobalParams,
                                                         ftype);
        // Open scope for spike-like event test
        os << OB(31);

        // Use synapse population support code namespace if required
        if (!spkEventCond.second.empty()) {
            os << " using namespace " << spkEventCond.second << ";" << ENDL;
        }

        // Combine this event threshold test with
        os << "spikeLikeEvent |= (" << eCode << ");" << ENDL;

        // Close scope for spike-like event test
        os << CB(31);
    }
}

//----------------------------------------------------------------------------
// StandardSubstitutions
//----------------------------------------------------------------------------
void StandardSubstitutions::postSynapseCurrentConverter(
    std::string &psCode,          //!< the code string to work on
    const std::string &sgName,
    const SynapseGroup *sg,
    const std::string &ngName,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype)
{
    // Create iterators to iterate over the names of the postsynaptic model's initial values
    auto psmVars = VarNameIterCtx(sg->getPSModel()->GetVars());
    auto psmDerivedParams = DerivedParamNameIterCtx(sg->getPSModel()->GetDerivedParams());

    // Substitute in time parameter
    substitute(psCode, "$(t)", "t");

    name_substitutions(psCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(psCode, ng.getNeuronModel()->GetParamNames(), ng.getParams());
    value_substitutions(psCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());

    if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        name_substitutions(psCode, "lps", psmVars.nameBegin, psmVars.nameEnd, sgName);
    }
    else {
        value_substitutions(psCode, psmVars.nameBegin, psmVars.nameEnd, sg->getPSInitVals());
    }
    value_substitutions(psCode, sg->getPSModel()->GetParamNames(), sg->getPSParams());

    // Create iterators to iterate over the names of the postsynaptic model's derived parameters
    value_substitutions(psCode, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg->getPSDerivedParams());
    name_substitutions(psCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ngName);
    psCode = ensureFtype(psCode, ftype);
    checkUnreplacedVariables(psCode, "postSyntoCurrent");
}

void StandardSubstitutions::postSynapseDecay(
    std::string &pdCode,
    const std::string &sgName,
    const SynapseGroup *sg,
    const std::string &,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &,
    const std::string &ftype)
{
    // Create iterators to iterate over the names of the postsynaptic model's initial values
    auto psmVars = VarNameIterCtx(sg->getPSModel()->GetVars());
    auto psmDerivedParams = DerivedParamNameIterCtx(sg->getPSModel()->GetDerivedParams());

    substitute(pdCode, "$(t)", "t");

    name_substitutions(pdCode, "lps", psmVars.nameBegin, psmVars.nameEnd, sgName);
    value_substitutions(pdCode, sg->getPSModel()->GetParamNames(), sg->getPSParams());
    value_substitutions(pdCode, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg->getPSDerivedParams());
    name_substitutions(pdCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(pdCode, ng.getNeuronModel()->GetParamNames(), ng.getParams());
    value_substitutions(pdCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());

    pdCode = ensureFtype(pdCode, ftype);
    checkUnreplacedVariables(pdCode, "postSynDecay");
}


void StandardSubstitutions::neuronThresholdCondition(
    std::string &thCode,
    const std::string &ngName,
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
    value_substitutions(thCode, ng.getNeuronModel()->GetParamNames(), ng.getParams());
    value_substitutions(thCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    name_substitutions(thCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ngName);
    thCode= ensureFtype(thCode, ftype);
    checkUnreplacedVariables(thCode,"thresholdConditionCode");
}

void StandardSubstitutions::neuronSim(
    std::string &sCode,
    const std::string &ngName,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype)
{
    substitute(sCode, "$(t)", "t");
    name_substitutions(sCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(sCode, ng.getNeuronModel()->GetParamNames(), ng.getParams());
    value_substitutions(sCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    name_substitutions(sCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ngName);
    substitute(sCode, "$(Isyn)", "Isyn");
    substitute(sCode, "$(sT)", "lsT");
    sCode = ensureFtype(sCode, ftype);
    checkUnreplacedVariables(sCode, "neuron simCode");
}

void StandardSubstitutions::neuronSpikeEventCondition(
    std::string &eCode,
    const std::string &ngName,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype)
{
    // code substitutions ----
    substitute(eCode, "$(t)", "t");
    name_substitutions(eCode, "l", nmVars.nameBegin, nmVars.nameEnd, "", "_pre");
    name_substitutions(eCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ngName);
    eCode = ensureFtype(eCode, ftype);
    checkUnreplacedVariables(eCode, "neuronSpkEvntCondition");
}

void StandardSubstitutions::neuronReset(
    std::string &rCode,
    const std::string &ngName,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &ftype)
{
    substitute(rCode, "$(t)", "t");
    name_substitutions(rCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(rCode, ng.getNeuronModel()->GetParamNames(), ng.getParams());
    value_substitutions(rCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    substitute(rCode, "$(Isyn)", "Isyn");
    substitute(rCode, "$(sT)", "lsT");
    name_substitutions(rCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ngName);
    rCode = ensureFtype(rCode, ftype);
    checkUnreplacedVariables(rCode, "resetCode");
}

void StandardSubstitutions::weightUpdateThresholdCondition(
    std::string &eCode,
    const std::string &sgName,
    const SynapseGroup &sg,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype)
{
    value_substitutions(eCode, sg.getWUModel()->GetParamNames(), sg.getWUParams());
    value_substitutions(eCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(eCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sgName);
    neuron_substitutions_in_synaptic_code(eCode, &sg, preIdx, postIdx, devPrefix);
    eCode= ensureFtype(eCode, ftype);
    checkUnreplacedVariables(eCode, "evntThreshold");
}

void StandardSubstitutions::weightUpdateSim(
    std::string &wCode,
    const std::string &sgName,
    const SynapseGroup &sg,
    const VarNameIterCtx &wuVars,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype)
{
     if (sg.getMatrixType() & SynapseMatrixWeight::GLOBAL) {
         value_substitutions(wCode, wuVars.nameBegin, wuVars.nameEnd, sg.getWUInitVals());
     }

    value_substitutions(wCode, sg.getWUModel()->GetParamNames(), sg.getWUParams());
    value_substitutions(wCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(wCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sgName);
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
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype)
{
     if (sg->getMatrixType() & SynapseMatrixWeight::GLOBAL) {
         value_substitutions(SDcode, wuVars.nameBegin, wuVars.nameEnd, sg->getWUInitVals());
     }

     // substitute parameter values for parameters in synapseDynamics code
    value_substitutions(SDcode, sg->getWUModel()->GetParamNames(), sg->getWUParams());

    // substitute values for derived parameters in synapseDynamics code
    value_substitutions(SDcode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
    neuron_substitutions_in_synaptic_code(SDcode, sg, preIdx, postIdx, devPrefix);
    SDcode= ensureFtype(SDcode, ftype);
    checkUnreplacedVariables(SDcode, "synapseDynamics");
}

void StandardSubstitutions::weightUpdatePostLearn(
    std::string &code,
    const std::string &sgName,
    const SynapseGroup *sg,
    const DerivedParamNameIterCtx &wuDerivedParams,
    const ExtraGlobalParamNameIterCtx &wuExtraGlobalParams,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,
    const std::string &ftype)
{
    value_substitutions(code, sg->getWUModel()->GetParamNames(), sg->getWUParams());
    value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
    name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sgName);

    // presynaptic neuron variables and parameters
    neuron_substitutions_in_synaptic_code(code, sg, preIdx, postIdx, devPrefix);
    code= ensureFtype(code, ftype);
    checkUnreplacedVariables(code, "simLearnPost");
}
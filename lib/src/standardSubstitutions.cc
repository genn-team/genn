#include "standardSubstitutions.h"

// GeNN includes
#include "modelSpec.h"

//----------------------------------------------------------------------------
// StandardSubstitutions
//----------------------------------------------------------------------------
void StandardSubstitutions::postSynapseCurrentConverter(
    std::string &psCode,          //!< the code string to work on
    const NNmodel &model,    //!< **TEMP**
    int synPopID,            //!< **TEMP**
    const std::string &ngName,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams)
{
    // **TEMP** extract PSM from model
    const auto *psm = model.postSynapseModel[synPopID];
    const std::string &sName = model.synapseName[synPopID];

    // Create iterators to iterate over the names of the postsynaptic model's initial values
    auto psmVars = VarNameIterCtx(psm->GetVars());
    auto psmDerivedParams = DerivedParamNameIterCtx(psm->GetDerivedParams());

    // Substitute in time parameter
    substitute(psCode, "$(t)", "t");

    name_substitutions(psCode, "l", nmVars.nameBegin, nmVars.nameEnd, "");
    value_substitutions(psCode, ng.getNeuronModel()->GetParamNames(), ng.getParams());
    value_substitutions(psCode, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());

    if (model.synapseMatrixType[synPopID] & SynapseMatrixWeight::INDIVIDUAL) {
        name_substitutions(psCode, "lps", psmVars.nameBegin, psmVars.nameEnd, sName);
    }
    else {
        value_substitutions(psCode, psmVars.nameBegin, psmVars.nameEnd, model.postSynIni[synPopID]);
    }
    value_substitutions(psCode, psm->GetParamNames(), model.postSynapsePara[synPopID]);

    // Create iterators to iterate over the names of the postsynaptic model's derived parameters
    value_substitutions(psCode, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, model.dpsp[synPopID]);
    name_substitutions(psCode, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ngName);
    psCode= ensureFtype(psCode, model.ftype);
    checkUnreplacedVariables(psCode, "postSyntoCurrent");
}

void StandardSubstitutions::postSynapseDecay(
    std::string &pdCode,
    const NNmodel &model,    //!< **TEMP**
    int synPopID,            //!< **TEMP**
    const std::string &,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const DerivedParamNameIterCtx &nmDerivedParams,
    const ExtraGlobalParamNameIterCtx &,
    const std::string &ftype)
{
    // **TEMP** extract PSM from model
    const auto *psm = model.postSynapseModel[synPopID];
    const std::string &sName = model.synapseName[synPopID];

    // Create iterators to iterate over the names of the postsynaptic model's initial values
    auto psmVars = VarNameIterCtx(psm->GetVars());
    auto psmDerivedParams = DerivedParamNameIterCtx(psm->GetDerivedParams());

    substitute(pdCode, "$(t)", "t");
    substitute(pdCode, "$(inSyn)", "inSyn" + sName + "[n]");

    name_substitutions(pdCode, "lps", psmVars.nameBegin, psmVars.nameEnd, sName);
    value_substitutions(pdCode, psm->GetParamNames(), model.postSynapsePara[synPopID]);
    value_substitutions(pdCode, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, model.dpsp[synPopID]);
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

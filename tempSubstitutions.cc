#include "tempSubstitutions.h"

// GeNN includes
#include "codeGenUtils.h"
#include "initSparseConnectivitySnippet.h"
#include "neuronGroup.h"
#include "synapseGroup.h"
#include "newModels.h"
#include "standardSubstitutions.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::applyNeuronModelSubstitutions(std::string &code, const NeuronGroup &ng,
                                                  const std::string &varPrefix, const std::string &varSuffix, const std::string &varExt)
{
    const NeuronModels::Base *nm = ng.getNeuronModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    VarNameIterCtx nmVars(nm->getVars());
    DerivedParamNameIterCtx nmDerivedParams(nm->getDerivedParams());
    ExtraGlobalParamNameIterCtx nmExtraGlobalParams(nm->getExtraGlobalParams());

    name_substitutions(code, varPrefix, nmVars.nameBegin, nmVars.nameEnd, varSuffix, varExt);
    value_substitutions(code, nm->getParamNames(), ng.getParams());
    value_substitutions(code, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    name_substitutions(code, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());
}
//--------------------------------------------------------------------------
void CodeGenerator::applyPostsynapticModelSubstitutions(std::string &code, const SynapseGroup &sg, const std::string &varPrefix)
{
    const auto *psm = sg.getPSModel();

    // Create iterators to iterate over the names of the postsynaptic model's initial values
    VarNameIterCtx psmVars(psm->getVars());
    DerivedParamNameIterCtx psmDerivedParams(psm->getDerivedParams());

    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
        name_substitutions(code, varPrefix, psmVars.nameBegin, psmVars.nameEnd, sg.getName());
    }
    else {
        value_substitutions(code, psmVars.nameBegin, psmVars.nameEnd, sg.getPSConstInitVals());
    }
    value_substitutions(code, psm->getParamNames(), sg.getPSParams());

    // Create iterators to iterate over the names of the postsynaptic model's derived parameters
    value_substitutions(code, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg.getPSDerivedParams());
}
//--------------------------------------------------------------------------
void CodeGenerator::applyWeightUpdateModelSubstitutions(std::string &code, const SynapseGroup &sg,
                                                        const std::string &varPrefix, const std::string &varSuffix, const std::string &varExt)
{
    const auto *wu = sg.getWUModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());
    VarNameIterCtx wuPreVars(wu->getPreVars());
    VarNameIterCtx wuPostVars(wu->getPostVars());

    value_substitutions(code, sg.getWUModel()->getParamNames(), sg.getWUParams());
    value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg.getName());

    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        name_substitutions(code, varPrefix, wuVars.nameBegin, wuVars.nameEnd, varSuffix, varExt);
    }
    else {
        value_substitutions(code, wuVars.nameBegin, wuVars.nameEnd, sg.getWUConstInitVals());
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::applyVarInitSnippetSubstitutions(std::string &code, const NewModels::VarInit &varInit)
{
    // Substitue derived and standard parameters into init code
    DerivedParamNameIterCtx viDerivedParams(varInit.getSnippet()->getDerivedParams());
    value_substitutions(code, varInit.getSnippet()->getParamNames(), varInit.getParams());
    value_substitutions(code, viDerivedParams.nameBegin, viDerivedParams.nameEnd, varInit.getDerivedParams());
}
//--------------------------------------------------------------------------
void CodeGenerator::applySparsConnectInitSnippetSubstitutions(std::string &code, const SynapseGroup &sg)
{
    const auto connectInit = sg.getConnectivityInitialiser();

    // Substitue derived and standard parameters into init code
    DerivedParamNameIterCtx viDerivedParams(connectInit.getSnippet()->getDerivedParams());
    ExtraGlobalParamNameIterCtx viExtraGlobalParams(connectInit.getSnippet()->getExtraGlobalParams());
    value_substitutions(code, connectInit.getSnippet()->getParamNames(), connectInit.getParams());
    value_substitutions(code, viDerivedParams.nameBegin, viDerivedParams.nameEnd, connectInit.getDerivedParams());
    name_substitutions(code, "initSparseConn", viExtraGlobalParams.nameBegin, viExtraGlobalParams.nameEnd, sg.getName());
}

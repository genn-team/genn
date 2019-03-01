#include "code_generator/tempSubstitutions.h"

// GeNN includes
#include "currentSource.h"
#include "initSparseConnectivitySnippet.h"
#include "neuronGroup.h"
#include "synapseGroup.h"
#include "models.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"

namespace
{
void neuronSubstitutionsInSynapticCode(
    std::string &wCode, //!< the code string to work on
    const NeuronGroup *ng,
    const std::string &offset,
    const std::string &delayOffset,
    const std::string &idx,
    const std::string &sourceSuffix,
    const std::string &devPrefix, //!< device prefix, "dd_" for GPU, nothing for CPU
    const std::string &varPrefix,
    const std::string &varSuffix)
{
    using namespace CodeGenerator;

    // presynaptic neuron variables, parameters, and global parameters
    const auto *neuronModel = ng->getNeuronModel();
    substitute(wCode, "$(sT" + sourceSuffix + ")",
               "(" + delayOffset + varPrefix + devPrefix+ "sT" + ng->getName() + "[" + offset + idx + "]" + varSuffix + ")");
    for(const auto &v : neuronModel->getVars()) {
        const std::string varIdx = ng->isVarQueueRequired(v.first) ? offset + idx : idx;

        substitute(wCode, "$(" + v.first + sourceSuffix + ")",
                   varPrefix + devPrefix + v.first + ng->getName() + "[" + varIdx + "]" + varSuffix);
    }
    value_substitutions(wCode, neuronModel->getParamNames(), ng->getParams(), sourceSuffix);

    DerivedParamNameIterCtx derivedParams(neuronModel->getDerivedParams());
    value_substitutions(wCode, derivedParams.nameBegin, derivedParams.nameEnd, ng->getDerivedParams(), sourceSuffix);

    ExtraGlobalParamNameIterCtx extraGlobalParams(neuronModel->getExtraGlobalParams());
    name_substitutions(wCode, "", extraGlobalParams.nameBegin, extraGlobalParams.nameEnd, ng->getName(), sourceSuffix);
}
}
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
    ExtraGlobalParamNameIterCtx psmExtraGlobalParams(psm->getExtraGlobalParams());

    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
        name_substitutions(code, varPrefix, psmVars.nameBegin, psmVars.nameEnd, sg.getName());
    }
    else {
        value_substitutions(code, psmVars.nameBegin, psmVars.nameEnd, sg.getPSConstInitVals());
    }
    value_substitutions(code, psm->getParamNames(), sg.getPSParams());

    // Create iterators to iterate over the names of the postsynaptic model's derived parameters
    value_substitutions(code, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg.getPSDerivedParams());
    name_substitutions(code, "", psmExtraGlobalParams.nameBegin, psmExtraGlobalParams.nameEnd, sg.getName());
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
void CodeGenerator::applyCurrentSourceSubstitutions(std::string &code, const CurrentSource &cs,
                                                    const std::string &varPrefix)
{
    const auto* csm = cs.getCurrentSourceModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    VarNameIterCtx csVars(csm->getVars());
    DerivedParamNameIterCtx csDerivedParams(csm->getDerivedParams());
    ExtraGlobalParamNameIterCtx csExtraGlobalParams(csm->getExtraGlobalParams());


    name_substitutions(code, varPrefix, csVars.nameBegin, csVars.nameEnd);
    value_substitutions(code, csm->getParamNames(), cs.getParams());
    value_substitutions(code, csDerivedParams.nameBegin, csDerivedParams.nameEnd, cs.getDerivedParams());
    name_substitutions(code, "", csExtraGlobalParams.nameBegin, csExtraGlobalParams.nameEnd, cs.getName());
}
//--------------------------------------------------------------------------
void CodeGenerator::applyVarInitSnippetSubstitutions(std::string &code, const Models::VarInit &varInit)
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
//--------------------------------------------------------------------------
void CodeGenerator::preNeuronSubstitutionsInSynapticCode(
    std::string &wCode, //!< the code string to work on
    const SynapseGroup &sg,
    const std::string &offset,
    const std::string &axonalDelayOffset,
    const std::string &preIdx,
    const std::string &devPrefix, //!< device prefix, "dd_" for GPU, nothing for CPU
    const std::string &preVarPrefix,     //!< prefix to be used for presynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &preVarSuffix)     //!< suffix to be used for presynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
{
    // presynaptic neuron variables, parameters, and global parameters
    const auto *srcNeuronModel = sg.getSrcNeuronGroup()->getNeuronModel();
    if (srcNeuronModel->isPoisson()) {
        substitute(wCode, "$(V_pre)", std::to_string(sg.getSrcNeuronGroup()->getParams()[2]));
    }

    ::neuronSubstitutionsInSynapticCode(wCode, sg.getSrcNeuronGroup(), offset, axonalDelayOffset, preIdx, "_pre", devPrefix, preVarPrefix, preVarSuffix);
}
//--------------------------------------------------------------------------
void CodeGenerator::postNeuronSubstitutionsInSynapticCode(
    std::string &wCode, //!< the code string to work on
    const SynapseGroup &sg,
    const std::string &offset,
    const std::string &backPropDelayOffset,
    const std::string &postIdx,
    const std::string &devPrefix, //!< device prefix, "dd_" for GPU, nothing for CPU
    const std::string &postVarPrefix,    //!< prefix to be used for postsynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &postVarSuffix)    //!< suffix to be used for postsynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
{
    // postsynaptic neuron variables, parameters, and global parameters
    ::neuronSubstitutionsInSynapticCode(wCode, sg.getTrgNeuronGroup(), offset, backPropDelayOffset, postIdx, "_post", devPrefix, postVarPrefix, postVarSuffix);
}
//--------------------------------------------------------------------------
void CodeGenerator::neuronSubstitutionsInSynapticCode(
    std::string &wCode,                  //!< the code string to work on
    const SynapseGroup &sg,         //!< the synapse group connecting the pre and postsynaptic neuron populations whose parameters might need to be substituted
    const std::string &preIdx,           //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const std::string &postIdx,          //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const std::string &devPrefix,        //!< device prefix, "dd_" for GPU, nothing for CPU
    double dt,                      //!< simulation timestep (ms)
    const std::string &preVarPrefix,     //!< prefix to be used for presynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &preVarSuffix,     //!< suffix to be used for presynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
    const std::string &postVarPrefix,    //!< prefix to be used for postsynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &postVarSuffix)    //!< suffix to be used for postsynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
{
    const std::string axonalDelayOffset = writePreciseString(dt * (double)(sg.getDelaySteps() + 1)) + " + ";
    const std::string preOffset = sg.getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
    preNeuronSubstitutionsInSynapticCode(wCode, sg, preOffset, axonalDelayOffset, preIdx, devPrefix, preVarPrefix, preVarSuffix);

    const std::string backPropDelayMs = writePreciseString(dt * (double)(sg.getBackPropDelaySteps() + 1)) + " + ";
    const std::string postOffset = sg.getTrgNeuronGroup()->isDelayRequired() ? "postReadDelayOffset + " : "";
    postNeuronSubstitutionsInSynapticCode(wCode, sg, postOffset, backPropDelayMs, postIdx, devPrefix, postVarPrefix, postVarSuffix);
}

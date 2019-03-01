#include "code_generator/tempSubstitutions.h"

// GeNN includes
#include "currentSource.h"
#include "initSparseConnectivitySnippet.h"
#include "neuronGroup.h"
#include "synapseGroup.h"
#include "models.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
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
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
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

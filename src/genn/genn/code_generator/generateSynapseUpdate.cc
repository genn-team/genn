#include "code_generator/generateSynapseUpdate.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/substitutions.h"
#include "code_generator/backendBase.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void applySynapseSubstitutions(CodeGenerator::CodeStream &os, std::string code, const std::string &errorSuffix, const SynapseGroupInternal &sg,
                               const CodeGenerator::Substitutions &baseSubs, const ModelSpecInternal &model, const CodeGenerator::BackendBase &backend)
{
    using namespace CodeGenerator;
    const auto *wu = sg.getWUModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    EGPNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());
    VarNameIterCtx wuPreVars(wu->getPreVars());
    VarNameIterCtx wuPostVars(wu->getPostVars());

    value_substitutions(code, sg.getWUModel()->getParamNames(), sg.getWUParams());
    value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg.getName());

    // Substitute names of pre and postsynaptic weight update variables
    const std::string delayedPreIdx = (sg.getDelaySteps() == NO_DELAY) ? baseSubs["id_pre"] : "preReadDelayOffset + " + baseSubs["id_pre"];
    name_substitutions(code, backend.getVarPrefix(), wuPreVars.nameBegin, wuPreVars.nameEnd, sg.getName() + "[" + delayedPreIdx + "]");

    const std::string delayedPostIdx = (sg.getBackPropDelaySteps() == NO_DELAY) ? baseSubs["id_post"] : "postReadDelayOffset + " + baseSubs["id_post"];
    name_substitutions(code, backend.getVarPrefix(), wuPostVars.nameBegin, wuPostVars.nameEnd, sg.getName() + "[" + delayedPostIdx + "]");

    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        name_substitutions(code, backend.getVarPrefix(), wuVars.nameBegin, wuVars.nameEnd,
                           sg.getName() + "[" + baseSubs["id_syn"] + "]", "");
    }
    else {
        value_substitutions(code, wuVars.nameBegin, wuVars.nameEnd, sg.getWUConstInitVals());
    }

    neuronSubstitutionsInSynapticCode(code, sg, baseSubs["id_pre"],
                                      baseSubs["id_post"], backend.getVarPrefix(),
                                      model.getDT());
    baseSubs.apply(code);
    code= ensureFtype(code, model.getPrecision());
    checkUnreplacedVariables(code, sg.getName() + errorSuffix);
    os << code;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateSynapseUpdate(CodeStream &os, const ModelSpecInternal &model, const BackendBase &backend,
                                          bool standaloneModules)
{
    if(standaloneModules) {
        os << "#include \"runner.cc\"" << std::endl;
    }
    else {
        os << "#include \"definitionsInternal.h\"" << std::endl;
    }
    os << "#include \"supportCode.h\"" << std::endl;
    os << std::endl;

    // Synaptic update kernels
    backend.genSynapseUpdate(os, model,
        // Presynaptic weight update threshold
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, const Substitutions &baseSubs)
        {
            // Get event threshold condition code
            std::string code = sg.getWUModel()->getEventThresholdConditionCode();

            // Make weight update model substitutions
            DerivedParamNameIterCtx wuDerivedParams(sg.getWUModel()->getDerivedParams());
            EGPNameIterCtx wuExtraGlobalParams(sg.getWUModel()->getExtraGlobalParams());
            value_substitutions(code, sg.getWUModel()->getParamNames(), sg.getWUParams());
            value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
            name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg.getName());

            // Get read offset if required
            const std::string offset = sg.getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
            preNeuronSubstitutionsInSynapticCode(code, sg, offset, "", baseSubs["id_pre"], backend.getVarPrefix());

            baseSubs.apply(code);
            code = ensureFtype(code, model.getPrecision());
            checkUnreplacedVariables(code, sg.getName() + " : evntThreshold");
            os << code;
        },
        // Presynaptic spike
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, const Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getWUModel()->getSimCode(), " : simCode",
                                      sg, baseSubs, model, backend);
        },
        // Presynaptic spike-like event
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, const Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getWUModel()->getEventCode(), " : eventCode",
                                      sg, baseSubs, model, backend);
        },
        // Postsynaptic learning code
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, const Substitutions &baseSubs)
        {
            if (!sg.getWUModel()->getLearnPostSupportCode().empty()) {
                os << " using namespace " << sg.getName() << "_weightupdate_simLearnPost;" << std::endl;
            }

            applySynapseSubstitutions(os, sg.getWUModel()->getLearnPostCode(), " : simLearnPost",
                                      sg, baseSubs, model, backend);
        },
        // Synapse dynamics
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, const Substitutions &baseSubs)
        {
            if (!sg.getWUModel()->getSynapseDynamicsSuppportCode().empty()) {
                os << " using namespace " << sg.getName() << "_weightupdate_synapseDynamics;" << std::endl;
            }

            applySynapseSubstitutions(os, sg.getWUModel()->getSynapseDynamicsCode(), " : synapseDynamics",
                                      sg, baseSubs, model, backend);
        }
    );
}

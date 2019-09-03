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
void applySynapseSubstitutions(CodeGenerator::CodeStream &os, std::string code, const std::string &errorContext, const SynapseGroupInternal &sg,
                               const CodeGenerator::Substitutions &baseSubs, const ModelSpecInternal &model, const CodeGenerator::BackendBase &backend)
{
    const auto *wu = sg.getWUModel();

    CodeGenerator::Substitutions synapseSubs(&baseSubs);

    // Substitute parameter and derived parameter names
    synapseSubs.addParamValueSubstitution(sg.getWUModel()->getParamNames(), sg.getWUParams());
    synapseSubs.addVarValueSubstitution(wu->getDerivedParams(), sg.getWUDerivedParams());
    synapseSubs.addVarNameSubstitution(wu->getExtraGlobalParams(), "", "", sg.getName());

    // Substitute names of pre and postsynaptic weight update variables
    const std::string delayedPreIdx = (sg.getDelaySteps() == NO_DELAY) ? synapseSubs["id_pre"] : "preReadDelayOffset + " + baseSubs["id_pre"];
    synapseSubs.addVarNameSubstitution(wu->getPreVars(), "", backend.getVarPrefix(),
                                       sg.getName() + "[" + delayedPreIdx + "]");

    const std::string delayedPostIdx = (sg.getBackPropDelaySteps() == NO_DELAY) ? synapseSubs["id_post"] : "postReadDelayOffset + " + baseSubs["id_post"];
    synapseSubs.addVarNameSubstitution(wu->getPostVars(), "", backend.getVarPrefix(),
                                       sg.getName() + "[" + delayedPostIdx + "]");

    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        synapseSubs.addVarNameSubstitution(wu->getVars(), "", backend.getVarPrefix(),
                                           sg.getName() + "[" + synapseSubs["id_syn"] + "]");
    }
    else {
        synapseSubs.addVarValueSubstitution(wu->getVars(), sg.getWUConstInitVals());
    }

    neuronSubstitutionsInSynapticCode(synapseSubs, sg, synapseSubs["id_pre"],
                                      synapseSubs["id_post"], backend.getVarPrefix(),
                                      model.getDT());

    synapseSubs.apply(code);
    //synapseSubs.applyCheckUnreplaced(code, errorContext + " : " + sg.getName());
    code= CodeGenerator::ensureFtype(code, model.getPrecision());
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
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, Substitutions &baseSubs)
        {
            Substitutions synapseSubs(&baseSubs);

            // Make weight update model substitutions
            synapseSubs.addParamValueSubstitution(sg.getWUModel()->getParamNames(), sg.getWUParams());
            synapseSubs.addVarValueSubstitution(sg.getWUModel()->getDerivedParams(), sg.getWUDerivedParams());
            synapseSubs.addVarNameSubstitution(sg.getWUModel()->getExtraGlobalParams(), "", sg.getName());

            // Get read offset if required
            const std::string offset = sg.getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
            preNeuronSubstitutionsInSynapticCode(synapseSubs, sg, offset, "", baseSubs["id_pre"], backend.getVarPrefix());

            // Get event threshold condition code
            std::string code = sg.getWUModel()->getEventThresholdConditionCode();
            baseSubs.applyCheckUnreplaced(code, "eventThresholdConditionCode");
            code = ensureFtype(code, model.getPrecision());
            os << code;
        },
        // Presynaptic spike
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getWUModel()->getSimCode(), "simCode",
                                      sg, baseSubs, model, backend);
        },
        // Presynaptic spike-like event
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getWUModel()->getEventCode(), "eventCode",
                                      sg, baseSubs, model, backend);
        },
        // Procedural connectivity
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, Substitutions &baseSubs)
        {
            baseSubs.addFuncSubstitution("endRow", 0, "break");

            // Initialise row building state variables for procedural connectivity
            const auto &connectInit = sg.getConnectivityInitialiser();
            for(const auto &a : connectInit.getSnippet()->getRowBuildStateVars()) {
                std::string value = a.value;
                baseSubs.apply(value);
                os << a.type << " " << a.name << " = " << value << ";" << std::endl;
            }

            // Loop through synapses in row
            os << "while(true)";
            {
                CodeStream::Scope b(os);
                Substitutions synSubs(&baseSubs);

                synSubs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams());
                synSubs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams());
                synSubs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", sg.getName());

                std::string pCode = connectInit.getSnippet()->getRowBuildCode();
                synSubs.applyCheckUnreplaced(pCode, "proceduralSparseConnectivity : " + sg.getName());
                pCode = ensureFtype(pCode, model.getPrecision());

                // Write out code
                os << pCode << std::endl;
            }
        },
        // Postsynaptic learning code
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, const Substitutions &baseSubs)
        {
            if (!sg.getWUModel()->getLearnPostSupportCode().empty()) {
                os << " using namespace " << sg.getName() << "_weightupdate_simLearnPost;" << std::endl;
            }

            applySynapseSubstitutions(os, sg.getWUModel()->getLearnPostCode(), "learnPostCode",
                                      sg, baseSubs, model, backend);
        },
        // Synapse dynamics
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, const Substitutions &baseSubs)
        {
            if (!sg.getWUModel()->getSynapseDynamicsSuppportCode().empty()) {
                os << " using namespace " << sg.getName() << "_weightupdate_synapseDynamics;" << std::endl;
            }

            applySynapseSubstitutions(os, sg.getWUModel()->getSynapseDynamicsCode(), "synapseDynamics",
                                      sg, baseSubs, model, backend);
        }
    );
}

#include "code_generator/generateSynapseUpdate.h"

// Standard C++ includes
#include <string>

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/substitutions.h"
#include "code_generator/backendBase.h"
#include "code_generator/groupMerged.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/teeStream.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void applySynapseSubstitutions(CodeGenerator::CodeStream &os, std::string code, const std::string &errorContext,
                               const SynapseGroupInternal &sg, const CodeGenerator::Substitutions &baseSubs,
                               const ModelSpecInternal &model, const CodeGenerator::BackendBase &backend)
{
    const auto *wu = sg.getWUModel();

    CodeGenerator::Substitutions synapseSubs(&baseSubs);

    // Substitute parameter and derived parameter names
    synapseSubs.addParamValueSubstitution(sg.getWUModel()->getParamNames(), sg.getWUParams());
    synapseSubs.addVarValueSubstitution(wu->getDerivedParams(), sg.getWUDerivedParams());
    synapseSubs.addVarNameSubstitution(wu->getExtraGlobalParams(), "", "group.");

    // Substitute names of pre and postsynaptic weight update variables
    const std::string delayedPreIdx = (sg.getDelaySteps() == NO_DELAY) ? synapseSubs["id_pre"] : "preReadDelayOffset + " + baseSubs["id_pre"];
    synapseSubs.addVarNameSubstitution(wu->getPreVars(), "", "group.",
                                       "[" + delayedPreIdx + "]");

    const std::string delayedPostIdx = (sg.getBackPropDelaySteps() == NO_DELAY) ? synapseSubs["id_post"] : "postReadDelayOffset + " + baseSubs["id_post"];
    synapseSubs.addVarNameSubstitution(wu->getPostVars(), "", "group.",
                                       "[" + delayedPostIdx + "]");

    // If weights are individual, substitute variables for values stored in global memory
    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        synapseSubs.addVarNameSubstitution(wu->getVars(), "", "group.",
                                           "[" + synapseSubs["id_syn"] + "]");
    }
    // Otherwise, if weights are procedual
    else if (sg.getMatrixType() & SynapseMatrixWeight::PROCEDURAL) {
        const auto vars = wu->getVars();

        // Loop through variables and their initialisers
        auto var = vars.cbegin();
        auto varInit = sg.getWUVarInitialisers().cbegin();
        for (; var != vars.cend(); var++, varInit++) {
            // Configure variable substitutions
            CodeGenerator::Substitutions varSubs(&synapseSubs);
            varSubs.addVarSubstitution("value", "l" + var->name);
            varSubs.addParamValueSubstitution(varInit->getSnippet()->getParamNames(), varInit->getParams());
            varSubs.addVarValueSubstitution(varInit->getSnippet()->getDerivedParams(), varInit->getDerivedParams());

            // Generate variable initialization code
            std::string code = varInit->getSnippet()->getCode();
            varSubs.applyCheckUnreplaced(code, "initVar : " + var->name + sg.getName());

            // Declare local variable
            os << var->type << " " << "l" << var->name << ";" << std::endl;

            // Insert code to initialize variable into scope
            {
                CodeGenerator::CodeStream::Scope b(os);
                os << code << std::endl;;
            }
        }

        // Substitute variables for newly-declared local variables
        synapseSubs.addVarNameSubstitution(vars, "", "l");
    }
    // Otherwise, substitute variables for constant values
    else {
        synapseSubs.addVarValueSubstitution(wu->getVars(), sg.getWUConstInitVals());
    }

    neuronSubstitutionsInSynapticCode(synapseSubs, sg, synapseSubs["id_pre"],
                                      synapseSubs["id_post"], model.getDT());

    synapseSubs.apply(code);
    //synapseSubs.applyCheckUnreplaced(code, errorContext + " : " + sg.getName());
    code = CodeGenerator::ensureFtype(code, model.getPrecision());
    os << code;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateSynapseUpdate(CodeStream &os, const MergedEGPMap &mergedEGPs, const ModelSpecMerged &modelMerged, const BackendBase &backend)
{
    os << "#include \"definitionsInternal.h\"" << std::endl;
    os << "#include \"supportCode.h\"" << std::endl;
    os << std::endl;

    // Generate functions to push merged synapse group structures
    const ModelSpecInternal &model = modelMerged.getModel();
    genMergedGroupPush(os, modelMerged.getMergedSynapseDendriticDelayUpdateGroups(), mergedEGPs, "SynapseDendriticDelayUpdate", backend);
    genMergedGroupPush(os, modelMerged.getMergedPresynapticUpdateGroups(), mergedEGPs, "PresynapticUpdate", backend);
    genMergedGroupPush(os, modelMerged.getMergedPostsynapticUpdateGroups(), mergedEGPs, "PostsynapticUpdate", backend);
    genMergedGroupPush(os, modelMerged.getMergedSynapseDynamicsGroups(), mergedEGPs, "SynapseDynamics", backend);

    // Synaptic update kernels
    backend.genSynapseUpdate(os, modelMerged,
        // Presynaptic weight update threshold
        [&backend, &model](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &baseSubs)
        {
            Substitutions synapseSubs(&baseSubs);

            // Make weight update model substitutions
            synapseSubs.addParamValueSubstitution(sg.getArchetype().getWUModel()->getParamNames(), sg.getArchetype().getWUParams());
            synapseSubs.addVarValueSubstitution(sg.getArchetype().getWUModel()->getDerivedParams(), sg.getArchetype().getWUDerivedParams());
            synapseSubs.addVarNameSubstitution(sg.getArchetype().getWUModel()->getExtraGlobalParams(), "", "group.");

            // Get read offset if required
            const std::string offset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
            neuronSubstitutionsInSynapticCode(synapseSubs, sg.getArchetype().getSrcNeuronGroup(), offset, "", baseSubs["id_pre"], "_pre", "Pre");

            // Get event threshold condition code
            std::string code = sg.getArchetype().getWUModel()->getEventThresholdConditionCode();
            synapseSubs.applyCheckUnreplaced(code, "eventThresholdConditionCode");
            code = ensureFtype(code, model.getPrecision());
            os << code;
        },
        // Presynaptic spike
        [&backend, &model](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getArchetype().getWUModel()->getSimCode(), "simCode",
                                      sg.getArchetype(), baseSubs, model, backend);
        },
        // Presynaptic spike-like event
        [&backend, &model](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getArchetype().getWUModel()->getEventCode(), "eventCode",
                                      sg.getArchetype(), baseSubs, model, backend);
        },
        // Procedural connectivity
        [&backend, &model](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &baseSubs)
        {
            const auto &connectInit = sg.getArchetype().getConnectivityInitialiser();

            // Add substitutions
            baseSubs.addFuncSubstitution("endRow", 0, "break");
            baseSubs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams());
            baseSubs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams());
            baseSubs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", "group.");
            
            // Initialise row building state variables for procedural connectivity
            for(const auto &a : connectInit.getSnippet()->getRowBuildStateVars()) {
                // Apply substitutions to value
                std::string value = a.value;
                baseSubs.applyCheckUnreplaced(value, "proceduralSparseConnectivity row build state var : merged" + std::to_string(sg.getIndex()));

                os << a.type << " " << a.name << " = " << value << ";" << std::endl;
            }

            // Loop through synapses in row
            os << "while(true)";
            {
                CodeStream::Scope b(os);
                
                // Apply substitutions to row building code
                std::string pCode = connectInit.getSnippet()->getRowBuildCode();
                baseSubs.addVarNameSubstitution(connectInit.getSnippet()->getRowBuildStateVars());
                baseSubs.applyCheckUnreplaced(pCode, "proceduralSparseConnectivity : merged " + std::to_string(sg.getIndex()));
                pCode = ensureFtype(pCode, model.getPrecision());

                // Write out code
                os << pCode << std::endl;
            }
        },
        // Postsynaptic learning code
        [&backend, &modelMerged](CodeStream &os, const SynapseGroupMerged &sg, const Substitutions &baseSubs)
        {
            const auto *wum = sg.getArchetype().getWUModel();
            if (!wum->getLearnPostSupportCode().empty()) {
                os << "using namespace " << modelMerged.getPostsynapticUpdateSupportCodeNamespace(wum->getLearnPostSupportCode()) <<  ";" << std::endl;
            }

            applySynapseSubstitutions(os, wum->getLearnPostCode(), "learnPostCode",
                                      sg.getArchetype(), baseSubs, modelMerged.getModel(), backend);
        },
        // Synapse dynamics
        [&backend, &modelMerged](CodeStream &os, const SynapseGroupMerged &sg, const Substitutions &baseSubs)
        {
            const auto *wum = sg.getArchetype().getWUModel();
            if (!wum->getSynapseDynamicsSuppportCode().empty()) {
                os << "using namespace " << modelMerged.getSynapseDynamicsSupportCodeNamespace(wum->getSynapseDynamicsSuppportCode()) <<  ";" << std::endl;
            }

            applySynapseSubstitutions(os, wum->getSynapseDynamicsCode(), "synapseDynamics",
                                      sg.getArchetype(), baseSubs, modelMerged.getModel(), backend);
        },
        // Push EGP handler
        [&backend, &mergedEGPs](CodeStream &os)
        {
            genScalarEGPPush(os, mergedEGPs, "PresynapticUpdate", backend);
            genScalarEGPPush(os, mergedEGPs, "PostsynapticUpdate", backend);
            genScalarEGPPush(os, mergedEGPs, "SynapseDynamics", backend);
        });
}

#include "code_generator/generateSynapseUpdate.h"

// Standard C++ includes
#include <string>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/groupMerged.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/substitutions.h"
#include "code_generator/teeStream.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void applySynapseSubstitutions(CodeGenerator::CodeStream &os, std::string code, const std::string &errorContext,
                               const CodeGenerator::SynapseGroupMergedBase &sg, const CodeGenerator::Substitutions &baseSubs,
                               const ModelSpecInternal &model)
{
    const auto *wu = sg.getArchetype().getWUModel();

    CodeGenerator::Substitutions synapseSubs(&baseSubs);

    // Substitute parameter and derived parameter names
    synapseSubs.addParamValueSubstitution(wu->getParamNames(), sg.getArchetype().getWUParams(),
                                          [&sg](size_t i) { return sg.isWUParamHeterogeneous(i);  },
                                          "", "group->");
    synapseSubs.addVarValueSubstitution(wu->getDerivedParams(), sg.getArchetype().getWUDerivedParams(),
                                        [&sg](size_t i) { return sg.isWUDerivedParamHeterogeneous(i);  },
                                        "", "group->");
    synapseSubs.addVarNameSubstitution(wu->getExtraGlobalParams(), "", "group->");

    // Substitute names of pre and postsynaptic weight update variables
    const std::string delayedPreIdx = (sg.getArchetype().getDelaySteps() == NO_DELAY) ? synapseSubs["id_pre"] : "preReadDelayOffset + " + baseSubs["id_pre"];
    synapseSubs.addVarNameSubstitution(wu->getPreVars(), "", "group->",
                                       "[" + delayedPreIdx + "]");

    const std::string delayedPostIdx = (sg.getArchetype().getBackPropDelaySteps() == NO_DELAY) ? synapseSubs["id_post"] : "postReadDelayOffset + " + baseSubs["id_post"];
    synapseSubs.addVarNameSubstitution(wu->getPostVars(), "", "group->",
                                       "[" + delayedPostIdx + "]");

    // If weights are individual, substitute variables for values stored in global memory
    if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        synapseSubs.addVarNameSubstitution(wu->getVars(), "", "group->",
                                           "[" + synapseSubs["id_syn"] + "]");
    }
    // Otherwise, if weights are procedual
    else if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL) {
        const auto vars = wu->getVars();

        for(size_t k = 0; k < vars.size(); k++) {
            const auto &varInit = sg.getArchetype().getWUVarInitialisers().at(k);

            // If this variable has any initialisation code
            if(!varInit.getSnippet()->getCode().empty()) {
                // Configure variable substitutions
                CodeGenerator::Substitutions varSubs(&synapseSubs);
                varSubs.addVarSubstitution("value", "l" + vars[k].name);
                varSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                                  [k, &sg](size_t p) { return sg.isWUVarInitParamHeterogeneous(k, p); },
                                                  "", "group->", vars[k].name);
                varSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                                [k, &sg](size_t p) { return sg.isWUVarInitDerivedParamHeterogeneous(k, p); },
                                                "", "group->", vars[k].name);
                varSubs.addVarNameSubstitution(varInit.getSnippet()->getExtraGlobalParams(),
                                               "", "group->", vars[k].name);

                // Generate variable initialization code
                std::string code = varInit.getSnippet()->getCode();
                varSubs.applyCheckUnreplaced(code, "initVar : merged" + vars[k].name + std::to_string(sg.getIndex()));

                // Declare local variable
                os << vars[k].type << " " << "l" << vars[k].name << ";" << std::endl;

                // Insert code to initialize variable into scope
                {
                    CodeGenerator::CodeStream::Scope b(os);
                    os << code << std::endl;;
                }
            }
        }

        // Substitute variables for newly-declared local variables
        synapseSubs.addVarNameSubstitution(vars, "", "l");
    }
    // Otherwise, substitute variables for constant values
    else {
        synapseSubs.addVarValueSubstitution(wu->getVars(), sg.getArchetype().getWUConstInitVals(),
                                            [&sg](size_t v) { return sg.isWUGlobalVarHeterogeneous(v); },
                                            "", "group->");
    }

    // Make presynaptic neuron substitutions
    const std::string axonalDelayOffset = Utils::writePreciseString(model.getDT() * (double)(sg.getArchetype().getDelaySteps() + 1u)) + " + ";
    const std::string preOffset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
    neuronSubstitutionsInSynapticCode(synapseSubs, sg.getArchetype().getSrcNeuronGroup(),
                                      preOffset, axonalDelayOffset, synapseSubs["id_pre"], "_pre", "Pre", "", "",
                                      [&sg](size_t paramIndex) { return sg.isSrcNeuronParamHeterogeneous(paramIndex); },
                                      [&sg](size_t derivedParamIndex) { return sg.isSrcNeuronDerivedParamHeterogeneous(derivedParamIndex); });


    // Make postsynaptic neuron substitutions
    const std::string backPropDelayMs = Utils::writePreciseString(model.getDT() * (double)(sg.getArchetype().getBackPropDelaySteps() + 1u)) + " + ";
    const std::string postOffset = sg.getArchetype().getTrgNeuronGroup()->isDelayRequired() ? "postReadDelayOffset + " : "";
    neuronSubstitutionsInSynapticCode(synapseSubs, sg.getArchetype().getTrgNeuronGroup(),
                                      postOffset, backPropDelayMs, synapseSubs["id_post"], "_post", "Post", "", "",
                                      [&sg](size_t paramIndex) { return sg.isTrgNeuronParamHeterogeneous(paramIndex); },
                                      [&sg](size_t derivedParamIndex) { return sg.isTrgNeuronDerivedParamHeterogeneous(derivedParamIndex); });

    synapseSubs.apply(code);
    //synapseSubs.applyCheckUnreplaced(code, errorContext + " : " + sg.getName());
    code = CodeGenerator::ensureFtype(code, model.getPrecision());
    os << code;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateSynapseUpdate(CodeStream &os, BackendBase::MemorySpaces &memorySpaces,
                                          const ModelSpecMerged &modelMerged, const BackendBase &backend)
{
    os << "#include \"definitionsInternal.h\"" << std::endl;
    os << "#include \"supportCode.h\"" << std::endl;
    os << std::endl;

    // Generate functions to push merged synapse group structures
    const ModelSpecInternal &model = modelMerged.getModel();

    // Synaptic update kernels
    backend.genSynapseUpdate(os, modelMerged, memorySpaces,
        // Preamble handler
        [&modelMerged, &backend](CodeStream &os)
        {
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedSynapseDendriticDelayUpdateGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedPresynapticUpdateGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedPostsynapticUpdateGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedSynapseDynamicsGroups(), backend);
        },
        // Presynaptic weight update threshold
        [&model](CodeStream &os, const PresynapticUpdateGroupMerged &sg, Substitutions &baseSubs)
        {
            Substitutions synapseSubs(&baseSubs);

            // Make weight update model substitutions
            synapseSubs.addParamValueSubstitution(sg.getArchetype().getWUModel()->getParamNames(), sg.getArchetype().getWUParams(),
                                                  [&sg](size_t i) { return sg.isWUParamHeterogeneous(i);  },
                                                  "", "group->");
            synapseSubs.addVarValueSubstitution(sg.getArchetype().getWUModel()->getDerivedParams(), sg.getArchetype().getWUDerivedParams(),
                                                [&sg](size_t i) { return sg.isWUDerivedParamHeterogeneous(i);  },
                                                "", "group->");
            synapseSubs.addVarNameSubstitution(sg.getArchetype().getWUModel()->getExtraGlobalParams(), "", "group->");

            // Get read offset if required and substitute in presynaptic neuron properties
            const std::string offset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
            neuronSubstitutionsInSynapticCode(synapseSubs, sg.getArchetype().getSrcNeuronGroup(), offset, "", baseSubs["id_pre"], "_pre", "Pre", "", "",
                                              [&sg](size_t paramIndex) { return sg.isSrcNeuronParamHeterogeneous(paramIndex); },
                                              [&sg](size_t derivedParamIndex) { return sg.isSrcNeuronDerivedParamHeterogeneous(derivedParamIndex); });
            
            // Get event threshold condition code
            std::string code = sg.getArchetype().getWUModel()->getEventThresholdConditionCode();
            synapseSubs.applyCheckUnreplaced(code, "eventThresholdConditionCode");
            code = ensureFtype(code, model.getPrecision());
            os << code;
        },
        // Presynaptic spike
        [&model](CodeStream &os, const PresynapticUpdateGroupMerged &sg, Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getArchetype().getWUModel()->getSimCode(), "simCode",
                                      sg, baseSubs, model);
        },
        // Presynaptic spike-like event
        [&model](CodeStream &os, const PresynapticUpdateGroupMerged &sg, Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getArchetype().getWUModel()->getEventCode(), "eventCode",
                                      sg, baseSubs, model);
        },
        // Procedural connectivity
        [&model](CodeStream &os, const PresynapticUpdateGroupMerged &sg, Substitutions &baseSubs)
        {
            const auto &connectInit = sg.getArchetype().getConnectivityInitialiser();

            // Add substitutions
            baseSubs.addFuncSubstitution("endRow", 0, "break");
            baseSubs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams(),
                                               [&sg](size_t i) { return sg.isConnectivityInitParamHeterogeneous(i);  },
                                               "", "group->");
            baseSubs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams(),
                                             [&sg](size_t i) { return sg.isConnectivityInitDerivedParamHeterogeneous(i);  },
                                             "", "group->");
            baseSubs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", "group->");

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
        [&modelMerged](CodeStream &os, const PostsynapticUpdateGroupMerged &sg, const Substitutions &baseSubs)
        {
            const auto *wum = sg.getArchetype().getWUModel();
            if (!wum->getLearnPostSupportCode().empty()) {
                os << "using namespace " << modelMerged.getPostsynapticUpdateSupportCodeNamespace(wum->getLearnPostSupportCode()) <<  ";" << std::endl;
            }

            applySynapseSubstitutions(os, wum->getLearnPostCode(), "learnPostCode",
                                      sg, baseSubs, modelMerged.getModel());
        },
        // Synapse dynamics
        [&modelMerged](CodeStream &os, const SynapseDynamicsGroupMerged &sg, const Substitutions &baseSubs)
        {
            const auto *wum = sg.getArchetype().getWUModel();
            if (!wum->getSynapseDynamicsSuppportCode().empty()) {
                os << "using namespace " << modelMerged.getSynapseDynamicsSupportCodeNamespace(wum->getSynapseDynamicsSuppportCode()) <<  ";" << std::endl;
            }

            applySynapseSubstitutions(os, wum->getSynapseDynamicsCode(), "synapseDynamics",
                                      sg, baseSubs, modelMerged.getModel());
        },
        // Push EGP handler
        [&backend, &modelMerged](CodeStream &os)
        {
            modelMerged.genScalarEGPPush(os, "PresynapticUpdate", backend);
            modelMerged.genScalarEGPPush(os, "PostsynapticUpdate", backend);
            modelMerged.genScalarEGPPush(os, "SynapseDynamics", backend);
        });
}

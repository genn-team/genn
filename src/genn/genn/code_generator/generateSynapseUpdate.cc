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

using namespace CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void applySynapseSubstitutions(CodeStream &os, std::string code, const std::string &errorContext,
                               const SynapseGroupMergedBase &sg, const Substitutions &baseSubs,
                               const ModelSpecMerged &modelMerged, const bool backendSupportsNamespace)
{
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const auto *wu = sg.getArchetype().getWUModel();

    Substitutions synapseSubs(&baseSubs);

    // Substitute parameter and derived parameter names
    synapseSubs.addParamValueSubstitution(wu->getParamNames(), sg.getArchetype().getWUParams(),
                                          [&sg](size_t i) { return sg.isWUParamHeterogeneous(i);  },
                                          "", "group->");
    synapseSubs.addVarValueSubstitution(wu->getDerivedParams(), sg.getArchetype().getWUDerivedParams(),
                                        [&sg](size_t i) { return sg.isWUDerivedParamHeterogeneous(i);  },
                                        "", "group->");
    synapseSubs.addVarNameSubstitution(wu->getExtraGlobalParams(), "", "group->");

    // Substitute names of pre and postsynaptic weight update variables
    synapseSubs.addVarNameSubstitution(wu->getPreVars(), "", "group->", 
                                       [&sg, &synapseSubs, batchSize](VarAccess a) 
                                       { 
                                           return "[" + sg.getPreWUVarIndex(batchSize, getVarAccessDuplication(a), synapseSubs["id_pre"]) + "]";
                                       });

    synapseSubs.addVarNameSubstitution(wu->getPostVars(), "", "group->",
                                       [&sg, &synapseSubs, batchSize](VarAccess a) 
                                       { 
                                           return "[" + sg.getPostWUVarIndex(batchSize, getVarAccessDuplication(a), synapseSubs["id_post"]) + "]";
                                       });

    // If weights are individual, substitute variables for values stored in global memory
    if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        synapseSubs.addVarNameSubstitution(wu->getVars(), "", "group->",
                                           [&sg, &synapseSubs, batchSize](VarAccess a) 
                                           { 
                                               return "[" + sg.getSynVarIndex(batchSize, getVarAccessDuplication(a), synapseSubs["id_syn"]) + "]";
                                           });
    }
    // Otherwise, if weights are procedual
    else if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL) {
        if(!sg.getArchetype().getKernelSize().empty()) {
            // Generate kernel index
            os << "const unsigned int kernelInd = ";
            genKernelIndex(os, synapseSubs, sg);
            os << ";" << std::endl;

            // Add substitution
            synapseSubs.addVarSubstitution("id_kernel", "kernelInd");
        }
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
    neuronSubstitutionsInSynapticCode(synapseSubs, sg.getArchetype().getSrcNeuronGroup(),
                                      axonalDelayOffset, "_pre", "Pre", "", "", false,
                                      [&sg](size_t paramIndex) { return sg.isSrcNeuronParamHeterogeneous(paramIndex); },
                                      [&sg](size_t derivedParamIndex) { return sg.isSrcNeuronDerivedParamHeterogeneous(derivedParamIndex); },
                                      [&synapseSubs, &sg, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                      {
                                          return sg.getPreVarIndex(delay, batchSize, varDuplication, synapseSubs["id_pre"]); 
                                      },
                                      [&synapseSubs, &sg, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                      { 
                                          return sg.getPrePrevSpikeTimeIndex(delay, batchSize, varDuplication, synapseSubs["id_pre"]); 
                                      });


    // Make postsynaptic neuron substitutions
    const std::string backPropDelayMs = Utils::writePreciseString(model.getDT() * (double)(sg.getArchetype().getBackPropDelaySteps() + 1u)) + " + ";
    neuronSubstitutionsInSynapticCode(synapseSubs, sg.getArchetype().getTrgNeuronGroup(),
                                      backPropDelayMs, "_post", "Post", "", "", false,
                                      [&sg](size_t paramIndex) { return sg.isTrgNeuronParamHeterogeneous(paramIndex); },
                                      [&sg](size_t derivedParamIndex) { return sg.isTrgNeuronDerivedParamHeterogeneous(derivedParamIndex); },
                                      [&synapseSubs, &sg, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                      {
                                          return sg.getPostVarIndex(delay, batchSize, varDuplication, synapseSubs["id_post"]); 
                                      },
                                      [&synapseSubs, &sg, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                      { 
                                          return sg.getPostPrevSpikeTimeIndex(delay, batchSize, varDuplication, synapseSubs["id_post"]); 
                                      });

    // If the backend does not support namespaces then we substitute all support code functions with namepsace as prefix
    if (!backendSupportsNamespace) {
        if (!wu->getSimSupportCode().empty()) {
            code = disambiguateNamespaceFunction(wu->getSimSupportCode(), code, modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()));
        }
        if (!wu->getLearnPostSupportCode().empty()) {
            code = disambiguateNamespaceFunction(wu->getLearnPostSupportCode(), code, modelMerged.getPostsynapticUpdateSupportCodeNamespace(wu->getLearnPostSupportCode()));
        }
        if (!wu->getSynapseDynamicsSuppportCode().empty()) {
            code = disambiguateNamespaceFunction(wu->getSynapseDynamicsSuppportCode(), code, modelMerged.getSynapseDynamicsSupportCodeNamespace(wu->getSynapseDynamicsSuppportCode()));
        }
    }

    synapseSubs.apply(code);
    //synapseSubs.applyCheckUnreplaced(code, errorContext + " : " + sg.getName());
    code = ensureFtype(code, model.getPrecision());
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
    if (backend.supportsNamespace()) {
        os << "#include \"supportCode.h\"" << std::endl;
    }
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
        [&modelMerged, &backend](CodeStream &os, const PresynapticUpdateGroupMerged &sg, Substitutions &baseSubs)
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

            // Substitute in presynaptic neuron properties
            const unsigned int batchSize = modelMerged.getModel().getBatchSize();
            neuronSubstitutionsInSynapticCode(synapseSubs, sg.getArchetype().getSrcNeuronGroup(), "", "_pre", "Pre", "", "", false,
                                              [&sg](size_t paramIndex) { return sg.isSrcNeuronParamHeterogeneous(paramIndex); },
                                              [&sg](size_t derivedParamIndex) { return sg.isSrcNeuronDerivedParamHeterogeneous(derivedParamIndex); },
                                              [&synapseSubs, &sg, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                              {
                                                  return sg.getPreVarIndex(delay, batchSize, varDuplication, synapseSubs["id_pre"]); 
                                              },
                                              [&synapseSubs, &sg, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                              { 
                                                  return sg.getPrePrevSpikeTimeIndex(delay, batchSize, varDuplication, synapseSubs["id_pre"]); 
                                              });
            
            const auto* wum = sg.getArchetype().getWUModel();

            // Get event threshold condition code
            std::string code = wum->getEventThresholdConditionCode();
            synapseSubs.applyCheckUnreplaced(code, "eventThresholdConditionCode");
            code = ensureFtype(code, modelMerged.getModel().getPrecision());

            if (!backend.supportsNamespace() && !wum->getSimSupportCode().empty()) {
                code = disambiguateNamespaceFunction(wum->getSimSupportCode(), code, modelMerged.getPresynapticUpdateSupportCodeNamespace(wum->getSimSupportCode()));
            }

            os << code;
        },
        // Presynaptic spike
        [&modelMerged, &backend](CodeStream &os, const PresynapticUpdateGroupMerged &sg, Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getArchetype().getWUModel()->getSimCode(), "simCode",
                                      sg, baseSubs, modelMerged, backend.supportsNamespace());
        },
        // Presynaptic spike-like event
        [&modelMerged, &backend](CodeStream &os, const PresynapticUpdateGroupMerged &sg, Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getArchetype().getWUModel()->getEventCode(), "eventCode",
                                      sg, baseSubs, modelMerged, backend.supportsNamespace());
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
        [&modelMerged, &backend](CodeStream &os, const PostsynapticUpdateGroupMerged &sg, const Substitutions &baseSubs)
        {
            const auto *wum = sg.getArchetype().getWUModel();
            if (!wum->getLearnPostSupportCode().empty() && backend.supportsNamespace()) {
                os << "using namespace " << modelMerged.getPostsynapticUpdateSupportCodeNamespace(wum->getLearnPostSupportCode()) <<  ";" << std::endl;
            }

            applySynapseSubstitutions(os, wum->getLearnPostCode(), "learnPostCode",
                                      sg, baseSubs, modelMerged, backend.supportsNamespace());
        },
        // Synapse dynamics
        [&modelMerged, &backend](CodeStream &os, const SynapseDynamicsGroupMerged &sg, const Substitutions &baseSubs)
        {
            const auto *wum = sg.getArchetype().getWUModel();
            if (!wum->getSynapseDynamicsSuppportCode().empty() && backend.supportsNamespace()) {
                os << "using namespace " << modelMerged.getSynapseDynamicsSupportCodeNamespace(wum->getSynapseDynamicsSuppportCode()) <<  ";" << std::endl;
            }

            applySynapseSubstitutions(os, wum->getSynapseDynamicsCode(), "synapseDynamics",
                                      sg, baseSubs, modelMerged, backend.supportsNamespace());
        },
        // Push EGP handler
        [&backend, &modelMerged](CodeStream &os)
        {
            modelMerged.genScalarEGPPush<PresynapticUpdateGroupMerged>(os, backend);
            modelMerged.genScalarEGPPush<PostsynapticUpdateGroupMerged>(os, backend);
            modelMerged.genScalarEGPPush<SynapseDynamicsGroupMerged>(os, backend);
        });
}

#include "code_generator/synapseUpdateGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

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
                                       [&sg, &synapseSubs, batchSize](VarAccess a, size_t) 
                                       { 
                                           return "[" + sg.getPreWUVarIndex(batchSize, getVarAccessDuplication(a), synapseSubs["id_pre"]) + "]";
                                       });

    synapseSubs.addVarNameSubstitution(wu->getPostVars(), "", "group->",
                                       [&sg, &synapseSubs, batchSize](VarAccess a, size_t) 
                                       { 
                                           return "[" + sg.getPostWUVarIndex(batchSize, getVarAccessDuplication(a), synapseSubs["id_post"]) + "]";
                                       });

    // If this synapse group has a kernel and weights are either procedural and kernel
    if (!sg.getArchetype().getKernelSize().empty() && (
        (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL) 
         || (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL)))
    {
        // Generate kernel index
        os << "const unsigned int kernelInd = ";
        sg.genKernelIndex(os, synapseSubs);
        os << ";" << std::endl;

        // Add substitution
        synapseSubs.addVarSubstitution("id_kernel", "kernelInd");
    }

    // If weights are individual, substitute variables for values stored in global memory
    if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        synapseSubs.addVarNameSubstitution(wu->getVars(), "", "group->",
                                           [&sg, &synapseSubs, batchSize](VarAccess a, size_t) 
                                           { 
                                               return "[" + sg.getSynVarIndex(batchSize, getVarAccessDuplication(a), synapseSubs["id_syn"]) + "]";
                                           });
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
    // Otherwise, if weights are kernels
    else if(sg.getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL) {
        assert(!sg.getArchetype().getKernelSize().empty());

        // Use kernel index to index into variables
        synapseSubs.addVarNameSubstitution(wu->getVars(), "", "group->", 
                                           [&sg, &synapseSubs, batchSize](VarAccess a, size_t) 
                                           { 
                                               return "[" + sg.getKernelVarIndex(batchSize, getVarAccessDuplication(a), synapseSubs["id_kernel"]) + "]";
                                           });
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

//----------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string PresynapticUpdateGroupMerged::name = "PresynapticUpdate";
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeEventThreshold(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    Substitutions synapseSubs(&popSubs);

    // Make weight update model substitutions
    synapseSubs.addParamValueSubstitution(getArchetype().getWUModel()->getParamNames(), getArchetype().getWUParams(),
                                         [this](size_t i) { return isWUParamHeterogeneous(i);  },
                                         "", "group->");
    synapseSubs.addVarValueSubstitution(getArchetype().getWUModel()->getDerivedParams(), getArchetype().getWUDerivedParams(),
                                        [this](size_t i) { return isWUDerivedParamHeterogeneous(i);  },
                                        "", "group->");
    synapseSubs.addVarNameSubstitution(getArchetype().getWUModel()->getExtraGlobalParams(), "", "group->");

    // Substitute in presynaptic neuron properties
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    neuronSubstitutionsInSynapticCode(synapseSubs, getArchetype().getSrcNeuronGroup(), "", "_pre", "Pre", "", "", false,
                                      [this](size_t paramIndex) { return isSrcNeuronParamHeterogeneous(paramIndex); },
                                      [this](size_t derivedParamIndex) { return isSrcNeuronDerivedParamHeterogeneous(derivedParamIndex); },
                                      [batchSize, &synapseSubs, this](bool delay, VarAccessDuplication varDuplication) 
                                      {
                                          return getPreVarIndex(delay, batchSize, varDuplication, synapseSubs["id_pre"]); 
                                      },
                                      [batchSize, &synapseSubs, this](bool delay, VarAccessDuplication varDuplication) 
                                      { 
                                          return getPrePrevSpikeTimeIndex(delay, batchSize, varDuplication, synapseSubs["id_pre"]); 
                                      });
            
    const auto* wum = getArchetype().getWUModel();

    // Get event threshold condition code
    std::string code = wum->getEventThresholdConditionCode();
    synapseSubs.applyCheckUnreplaced(code, "eventThresholdConditionCode");
    code = ensureFtype(code, modelMerged.getModel().getPrecision());

    if (!backend.supportsNamespace() && !wum->getSimSupportCode().empty()) {
        code = disambiguateNamespaceFunction(wum->getSimSupportCode(), code, modelMerged.getPresynapticUpdateSupportCodeNamespace(wum->getSimSupportCode()));
    }

    os << code;
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeEventUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    applySynapseSubstitutions(os, getArchetype().getWUModel()->getEventCode(), "eventCode",
                              *this, popSubs, modelMerged, backend.supportsNamespace());
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    applySynapseSubstitutions(os, getArchetype().getWUModel()->getSimCode(), "simCode",
                              *this, popSubs, modelMerged, backend.supportsNamespace());
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateProceduralConnectivity(const BackendBase&, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const auto &connectInit = getArchetype().getConnectivityInitialiser();

    // Add substitutions
    popSubs.addFuncSubstitution("endRow", 0, "break");
    popSubs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams(),
                                      [this](size_t i) { return isSparseConnectivityInitParamHeterogeneous(i);  },
                                      "", "group->");
    popSubs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams(),
                                    [this](size_t i) { return isSparseConnectivityInitDerivedParamHeterogeneous(i);  },
                                    "", "group->");
    popSubs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", "group->");
    popSubs.addVarNameSubstitution(connectInit.getSnippet()->getRowBuildStateVars());

    // Initialise row building state variables for procedural connectivity
    for(const auto &a : connectInit.getSnippet()->getRowBuildStateVars()) {
        // Apply substitutions to value
        std::string value = a.value;
        popSubs.applyCheckUnreplaced(value, "proceduralSparseConnectivity row build state var : merged" + std::to_string(getIndex()));
        value = ensureFtype(value, modelMerged.getModel().getPrecision());
        os << a.type << " " << a.name << " = " << value << ";" << std::endl;
    }

    // Loop through synapses in row
    os << "while(true)";
    {
        CodeStream::Scope b(os);

        // Apply substitutions to row building code
        std::string pCode = connectInit.getSnippet()->getRowBuildCode();
        
        popSubs.applyCheckUnreplaced(pCode, "proceduralSparseConnectivity : merged " + std::to_string(getIndex()));
        pCode = ensureFtype(pCode, modelMerged.getModel().getPrecision());

        // Write out code
        os << pCode << std::endl;
    }
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateToeplitzConnectivity(const BackendBase&, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const auto &connectInit = getArchetype().getToeplitzConnectivityInitialiser();
    
    // Apply substitutions to diagonal building code
    std::string pCode = connectInit.getSnippet()->getDiagonalBuildCode();
    popSubs.applyCheckUnreplaced(pCode, "toeplitzSparseConnectivity : merged " + std::to_string(getIndex()));
    pCode = ensureFtype(pCode, modelMerged.getModel().getPrecision());

    // Write out code
    os << pCode << std::endl;
}

//----------------------------------------------------------------------------
// CodeGenerator::PostsynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string PostsynapticUpdateGroupMerged::name = "PostsynapticUpdate";
//----------------------------------------------------------------------------
void PostsynapticUpdateGroupMerged::generateSynapseUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const auto *wum = getArchetype().getWUModel();
    if (!wum->getLearnPostSupportCode().empty() && backend.supportsNamespace()) {
        os << "using namespace " << modelMerged.getPostsynapticUpdateSupportCodeNamespace(wum->getLearnPostSupportCode()) <<  ";" << std::endl;
    }

    applySynapseSubstitutions(os, wum->getLearnPostCode(), "learnPostCode",
                              *this, popSubs, modelMerged, backend.supportsNamespace());
}

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDynamicsGroupMerged::name = "SynapseDynamics";
//----------------------------------------------------------------------------
void SynapseDynamicsGroupMerged::generateSynapseUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const auto *wum = getArchetype().getWUModel();
    if (!wum->getSynapseDynamicsSuppportCode().empty() && backend.supportsNamespace()) {
        os << "using namespace " << modelMerged.getSynapseDynamicsSupportCodeNamespace(wum->getSynapseDynamicsSuppportCode()) <<  ";" << std::endl;
    }

    applySynapseSubstitutions(os, wum->getSynapseDynamicsCode(), "synapseDynamics",
                              *this, popSubs, modelMerged, backend.supportsNamespace());
}


//----------------------------------------------------------------------------
// CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDendriticDelayUpdateGroupMerged::name = "SynapseDendriticDelayUpdate";
//----------------------------------------------------------------------------
SynapseDendriticDelayUpdateGroupMerged::SynapseDendriticDelayUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, precision, groups)
{
    addField("unsigned int*", "denDelayPtr", 
             [&backend](const SynapseGroupInternal &sg, size_t) 
             {
                 return backend.getScalarAddressPrefix() + "denDelayPtr" + sg.getFusedPSVarSuffix(); 
             });
}
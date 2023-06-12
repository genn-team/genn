#include "code_generator/synapseUpdateGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
template<typename G>
void applySynapseSubstitutions(const BackendBase &backend, EnvironmentExternalBase &env, std::string code, const std::string &errorContext,
                               const G &sg, const ModelSpecMerged &modelMerged, bool backendSupportsNamespace)
{
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const auto *wu = sg.getArchetype().getWUModel();

    EnvironmentGroupMergedField<G> synEnv(sg, env);

    // Substitute parameter and derived parameter names
    synEnv.addParams(wu->getParamNames(), "", &SynapseGroupInternal::getWUParams, &G::isWUParamHeterogeneous);
    synEnv.addDerivedParams(wu->getDerivedParams(), "", &SynapseGroupInternal::getWUDerivedParams, &G::isWUDerivedParamHeterogeneous);
    synEnv.addEGPs<SynapseWUEGPAdapter>(backend.getDeviceVarPrefix());

    // Substitute names of pre and postsynaptic weight update variable
    synEnv.addVars<SynapseWUPreVarAdapter>(backend.getDeviceVarPrefix());
    synapseSubs.addVarNameSubstitution(wu->getPreVars(), "", "group->", 
                                       [&sg, &synapseSubs, batchSize](VarAccess a, const std::string&) 
                                       { 
                                           return "[" + sg.getPreWUVarIndex(batchSize, getVarAccessDuplication(a), synapseSubs["id_pre"]) + "]";
                                       });

    synapseSubs.addVarNameSubstitution(wu->getPostVars(), "", "group->",
                                       [&sg, &synapseSubs, batchSize](VarAccess a, const std::string&) 
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
                                           [&sg, &synapseSubs, batchSize](VarAccess a, const std::string&) 
                                           { 
                                               return "[" + sg.getSynVarIndex(batchSize, getVarAccessDuplication(a), synapseSubs["id_syn"]) + "]";
                                           });
    }
    // Otherwise, if weights are procedual
    else if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL) {
        const auto vars = wu->getVars();
        for(const auto &var : vars) {
            const auto &varInit = sg.getArchetype().getWUVarInitialisers().at(var.name);

            // If this variable has any initialisation code
            if(!varInit.getSnippet()->getCode().empty()) {
                // Configure variable substitutions
                CodeGenerator::Substitutions varSubs(&synapseSubs);
                varSubs.addVarSubstitution("value", "l" + var.name);
                varSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                                  [&var, &sg](const std::string &p) { return sg.isWUVarInitParamHeterogeneous(var.name, p); },
                                                  "", "group->", var.name);
                varSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                                [&var, &sg](const std::string &p) { return sg.isWUVarInitDerivedParamHeterogeneous(var.name, p); },
                                                "", "group->", var.name);
                varSubs.addVarNameSubstitution(varInit.getSnippet()->getExtraGlobalParams(),
                                               "", "group->", var.name);

                // Generate variable initialization code
                std::string code = varInit.getSnippet()->getCode();
                varSubs.applyCheckUnreplaced(code, "initVar : merged" + var.name + std::to_string(sg.getIndex()));

                // Declare local variable
                os << var.type.resolve(sg.getTypeContext()).getName() << " " << "l" << var.name << ";" << std::endl;

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
                                           [&sg, &synapseSubs, batchSize](VarAccess a, const std::string&) 
                                           { 
                                               return "[" + sg.getKernelVarIndex(batchSize, getVarAccessDuplication(a), synapseSubs["id_kernel"]) + "]";
                                           });
    }
    // Otherwise, substitute variables for constant values
    else {
        synapseSubs.addVarValueSubstitution(wu->getVars(), sg.getArchetype().getWUConstInitVals(),
                                            [&sg](const std::string &v) { return sg.isWUGlobalVarHeterogeneous(v); },
                                            "", "group->");
    }

    // Make presynaptic neuron substitutions
    const std::string axonalDelayOffset = Utils::writePreciseString(model.getDT() * (double)(sg.getArchetype().getDelaySteps() + 1u)) + " + ";
    neuronSubstitutionsInSynapticCode(synapseSubs, sg.getArchetype().getSrcNeuronGroup(),
                                      axonalDelayOffset, "_pre", "Pre", "", "", false,
                                      [&sg](const std::string &p) { return sg.isSrcNeuronParamHeterogeneous(p); },
                                      [&sg](const std::string &p) { return sg.isSrcNeuronDerivedParamHeterogeneous(p); },
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
                                      [&sg](const std::string &p) { return sg.isTrgNeuronParamHeterogeneous(p); },
                                      [&sg](const std::string &p) { return sg.isTrgNeuronDerivedParamHeterogeneous(p); },
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
    //code = ensureFtype(code, model.getPrecision());
    os << code;
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string PresynapticUpdateGroupMerged::name = "PresynapticUpdate";
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeEventThreshold(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    Substitutions synapseSubs(&popSubs);

    // Make weight update model substitutions
    synapseSubs.addParamValueSubstitution(getArchetype().getWUModel()->getParamNames(), getArchetype().getWUParams(),
                                         [this](const std::string &p) { return isWUParamHeterogeneous(p);  },
                                         "", "group->");
    synapseSubs.addVarValueSubstitution(getArchetype().getWUModel()->getDerivedParams(), getArchetype().getWUDerivedParams(),
                                        [this](const std::string &p) { return isWUDerivedParamHeterogeneous(p);  },
                                        "", "group->");
    synapseSubs.addVarNameSubstitution(getArchetype().getWUModel()->getExtraGlobalParams(), "", "group->");

    // Substitute in presynaptic neuron properties
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    neuronSubstitutionsInSynapticCode(synapseSubs, getArchetype().getSrcNeuronGroup(), "", "_pre", "Pre", "", "", false,
                                      [this](const std::string &p) { return isSrcNeuronParamHeterogeneous(p); },
                                      [this](const std::string &p) { return isSrcNeuronDerivedParamHeterogeneous(p); },
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
    //code = ensureFtype(code, modelMerged.getModel().getPrecision());

    if (!backend.supportsNamespace() && !wum->getSimSupportCode().empty()) {
        code = disambiguateNamespaceFunction(wum->getSimSupportCode(), code, modelMerged.getPresynapticUpdateSupportCodeNamespace(wum->getSimSupportCode()));
    }

    os << code;
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeEventUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    applySynapseSubstitutions(backend, os, getArchetype().getWUModel()->getEventCode(), "eventCode",
                              *this, popSubs, modelMerged);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    applySynapseSubstitutions(backend, os, getArchetype().getWUModel()->getSimCode(), "simCode",
                              *this, popSubs, modelMerged);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateProceduralConnectivity(const BackendBase&, CodeStream &os, Substitutions &popSubs) const
{
    const auto &connectInit = getArchetype().getConnectivityInitialiser();

    // Add substitutions
    popSubs.addFuncSubstitution("endRow", 0, "break");
    popSubs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams(),
                                      [this](const std::string &p) { return isSparseConnectivityInitParamHeterogeneous(p);  },
                                      "", "group->");
    popSubs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams(),
                                    [this](const std::string &p) { return isSparseConnectivityInitDerivedParamHeterogeneous(p);  },
                                    "", "group->");
    popSubs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", "group->");
    popSubs.addVarNameSubstitution(connectInit.getSnippet()->getRowBuildStateVars());

    // Initialise row building state variables for procedural connectivity
    for(const auto &a : connectInit.getSnippet()->getRowBuildStateVars()) {
        // Apply substitutions to value
        std::string value = a.value;
        popSubs.applyCheckUnreplaced(value, "proceduralSparseConnectivity row build state var : merged" + std::to_string(getIndex()));
        //value = ensureFtype(value, modelMerged.getModel().getPrecision());
        os << a.type.resolve(getTypeContext()).getName() << " " << a.name << " = " << value << ";" << std::endl;
    }

    // Loop through synapses in row
    os << "while(true)";
    {
        CodeStream::Scope b(os);

        // Apply substitutions to row building code
        std::string pCode = connectInit.getSnippet()->getRowBuildCode();
        
        popSubs.applyCheckUnreplaced(pCode, "proceduralSparseConnectivity : merged " + std::to_string(getIndex()));
        //pCode = ensureFtype(pCode, modelMerged.getModel().getPrecision());

        // Write out code
        os << pCode << std::endl;
    }
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateToeplitzConnectivity(const BackendBase&, CodeStream &os, Substitutions &popSubs) const
{
    const auto &connectInit = getArchetype().getToeplitzConnectivityInitialiser();
    
    // Apply substitutions to diagonal building code
    std::string pCode = connectInit.getSnippet()->getDiagonalBuildCode();
    popSubs.applyCheckUnreplaced(pCode, "toeplitzSparseConnectivity : merged " + std::to_string(getIndex()));
    //pCode = ensureFtype(pCode, modelMerged.getModel().getPrecision());

    // Write out code
    os << pCode << std::endl;
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PostsynapticUpdateGroupMerged
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
// GeNN::CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDynamicsGroupMerged::name = "SynapseDynamics";
//----------------------------------------------------------------------------
void SynapseDynamicsGroupMerged::generateSynapseUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged) const
{
    const auto *wum = getArchetype().getWUModel();
    /*if (!wum->getSynapseDynamicsSuppportCode().empty() && backend.supportsNamespace()) {
        os << "using namespace " << modelMerged.getSynapseDynamicsSupportCodeNamespace(wum->getSynapseDynamicsSuppportCode()) <<  ";" << std::endl;
    }*/

    applySynapseSubstitutions(backend, env, wum->getSynapseDynamicsCode(), "synapseDynamics",
                              *this, modelMerged, backend.supportsNamespace());
}


//----------------------------------------------------------------------------
// CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDendriticDelayUpdateGroupMerged::name = "SynapseDendriticDelayUpdate";
//----------------------------------------------------------------------------
SynapseDendriticDelayUpdateGroupMerged::SynapseDendriticDelayUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    addField(Type::Uint32.createPointer(), "denDelayPtr", 
             [&backend](const SynapseGroupInternal &sg, size_t) 
             {
                 return backend.getScalarAddressPrefix() + "denDelayPtr" + sg.getFusedPSVarSuffix(); 
             });
}

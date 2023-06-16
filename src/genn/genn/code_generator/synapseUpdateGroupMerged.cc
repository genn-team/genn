#include "code_generator/synapseUpdateGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
template<typename G>
void applySynapseSubstitutions(const BackendBase &backend, EnvironmentExternalBase &env, std::string code, const std::string &errorContext,
                               G &sg, const ModelSpecMerged &modelMerged, bool backendSupportsNamespace)
{
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const auto *wu = sg.getArchetype().getWUModel();

    EnvironmentGroupMergedField<G> synEnv(env, sg);

    // Substitute parameter and derived parameter names
    synEnv.addParams(wu->getParamNames(), "", &SynapseGroupInternal::getWUParams, &G::isWUParamHeterogeneous);
    synEnv.addDerivedParams(wu->getDerivedParams(), "", &SynapseGroupInternal::getWUDerivedParams, &G::isWUDerivedParamHeterogeneous);
    synEnv.addExtraGlobalParams(wu->getExtraGlobalParams(), backend.getDeviceVarPrefix());

    // Substitute names of pre and postsynaptic weight update variable
    synEnv.addVars<SynapseWUPreVarAdapter>(backend.getDeviceVarPrefix(),
                                           [&sg, &synEnv, batchSize](VarAccess a, const std::string&) 
                                           { 
                                               return "[" + sg.getPreWUVarIndex(batchSize, getVarAccessDuplication(a), synEnv["id_pre"]) + "]";
                                           }, 
                                           {"id_pre"});
     synEnv.addVars<SynapseWUPostVarAdapter>(backend.getDeviceVarPrefix(),
                                             [&sg, &synEnv, batchSize](VarAccess a, const std::string&) 
                                             { 
                                                 return "[" + sg.getPostWUVarIndex(batchSize, getVarAccessDuplication(a), synEnv["id_post"]) + "]";
                                             },
                                             {"id_post"});

    
    // If this synapse group has a kernel
    if (!sg.getArchetype().getKernelSize().empty()) {
        // Add substitution
        // **TODO** dependencies on kernel fields
        synEnv.add(Type::Uint32, "id_kernel", "kernelInd", 
                   {synEnv.addInitialiser("const unsigned int kernelInd = " + sg.getKernelIndex(synEnv) + ";")});
    }

    // If weights are individual, substitute variables for values stored in global memory
    if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        synEnv.addVars<SynapseWUVarAdapter>(backend.getDeviceVarPrefix(),
                                            [&sg, &synEnv, batchSize](VarAccess a, const std::string&) 
                                            { 
                                                return "[" + sg.getSynVarIndex(batchSize, getVarAccessDuplication(a), synEnv["id_syn"]) + "]";
                                            },
                                            {"id_syn"});
    }
    // Otherwise, if weights are procedual
    else if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL) {
        assert(false);
        /*const auto vars = wu->getVars();
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
        synEnv.add(vars, "", "l");*/
    }
    // Otherwise, if weights are kernels, use kernel index to index into variables
    else if(sg.getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL) {
        assert(!sg.getArchetype().getKernelSize().empty());

        synEnv.addVars<SynapseWUVarAdapter>(backend.getDeviceVarPrefix(),
                                            [&sg, &synEnv, batchSize](VarAccess a, const std::string&) 
                                            { 
                                                return "[" + sg.getKernelVarIndex(batchSize, getVarAccessDuplication(a), synEnv["id_kernel"]) + "]";
                                            },
                                            {"id_kernel"});
    }
    // Otherwise, substitute variables for constant values
    else {
        assert(false);
        /*synapseSubs.addVarValueSubstitution(wu->getVars(), sg.getArchetype().getWUConstInitVals(),
                                            [&sg](const std::string &v) { return sg.isWUGlobalVarHeterogeneous(v); },
                                            "", "group->");*/
    }

    // Make presynaptic neuron substitutions
    /*const std::string axonalDelayOffset = Utils::writePreciseString(model.getDT() * (double)(sg.getArchetype().getDelaySteps() + 1u)) + " + ";
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
                                      });*/

    // If the backend does not support namespaces then we substitute all support code functions with namepsace as prefix
    /*if (!backendSupportsNamespace) {
        if (!wu->getSimSupportCode().empty()) {
            code = disambiguateNamespaceFunction(wu->getSimSupportCode(), code, modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()));
        }
        if (!wu->getLearnPostSupportCode().empty()) {
            code = disambiguateNamespaceFunction(wu->getLearnPostSupportCode(), code, modelMerged.getPostsynapticUpdateSupportCodeNamespace(wu->getLearnPostSupportCode()));
        }
        if (!wu->getSynapseDynamicsSuppportCode().empty()) {
            code = disambiguateNamespaceFunction(wu->getSynapseDynamicsSuppportCode(), code, modelMerged.getSynapseDynamicsSupportCodeNamespace(wu->getSynapseDynamicsSuppportCode()));
        }
    }*/

    // Pretty print code back to environment
    Transpiler::ErrorHandler errorHandler(errorContext + std::to_string(sg.getIndex()));
    prettyPrintStatements(code, sg.getTypeContext(), synEnv, errorHandler);
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string PresynapticUpdateGroupMerged::name = "PresynapticUpdate";
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeEventThreshold(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(env, *this);

    // Substitute parameter and derived parameter names
    const auto *wum = getArchetype().getWUModel();
    synEnv.addParams(wum->getParamNames(), "", &SynapseGroupInternal::getWUParams, &PresynapticUpdateGroupMerged::isWUParamHeterogeneous);
    synEnv.addDerivedParams(wum->getDerivedParams(), "", &SynapseGroupInternal::getWUDerivedParams, &PresynapticUpdateGroupMerged::isWUDerivedParamHeterogeneous);
    synEnv.addExtraGlobalParams(wum->getExtraGlobalParams(), backend.getDeviceVarPrefix());

    // Substitute in presynaptic neuron properties
    /*const unsigned int batchSize = modelMerged.getModel().getBatchSize();
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
                                      });*/

    // Pretty print code back to environment
    Transpiler::ErrorHandler errorHandler("eventThresholdConditionCode" + std::to_string(getIndex()));
    prettyPrintStatements(wum->getEventThresholdConditionCode(), getTypeContext(), synEnv, errorHandler);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeEventUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUModel()->getEventCode(), "eventCode",
                              *this, modelMerged, backend.supportsNamespace());
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUModel()->getSimCode(), "simCode",
                              *this, modelMerged, backend.supportsNamespace());
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateProceduralConnectivity(const BackendBase&, EnvironmentExternalBase &env)
{
    const auto &connectInit = getArchetype().getConnectivityInitialiser();

    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(env, *this);

    assert(false);
    // Add substitutions
    //synEnv.addParams()
    //synEnv.addParams(wu->getParamNames(), "", &SynapseGroupInternal::getWUParams, &G::isWUParamHeterogeneous);
    //synEnv.addDerivedParams(wu->getDerivedParams(), "", &SynapseGroupInternal::getWUDerivedParams, &G::isWUDerivedParamHeterogeneous);
    /*popSubs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams(),
                                      [this](const std::string &p) { return isSparseConnectivityInitParamHeterogeneous(p);  },
                                      "", "group->");
    popSubs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams(),
                                    [this](const std::string &p) { return isSparseConnectivityInitDerivedParamHeterogeneous(p);  },
                                    "", "group->");
    popSubs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", "group->");

 
    // Apply substitutions to row building code
    std::string pCode = connectInit.getSnippet()->getRowBuildCode();
        
    popSubs.applyCheckUnreplaced(pCode, "proceduralSparseConnectivity : merged " + std::to_string(getIndex()));
    //pCode = ensureFtype(pCode, modelMerged.getModel().getPrecision());

    // Write out code
    os << pCode << std::endl;*/
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateToeplitzConnectivity(const BackendBase&, EnvironmentExternalBase &env)
{
    // Pretty print code back to environment
    Transpiler::ErrorHandler errorHandler("toeplitzSparseConnectivity" + std::to_string(getIndex()));
    prettyPrintStatements(getArchetype().getToeplitzConnectivityInitialiser().getSnippet()->getDiagonalBuildCode(), 
                          getTypeContext(), env, errorHandler);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PostsynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string PostsynapticUpdateGroupMerged::name = "PostsynapticUpdate";
//----------------------------------------------------------------------------
void PostsynapticUpdateGroupMerged::generateSynapseUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    const auto *wum = getArchetype().getWUModel();
    /*if (!wum->getLearnPostSupportCode().empty() && backend.supportsNamespace()) {
        os << "using namespace " << modelMerged.getPostsynapticUpdateSupportCodeNamespace(wum->getLearnPostSupportCode()) <<  ";" << std::endl;
    }*/

    applySynapseSubstitutions(backend, env, wum->getLearnPostCode(), "synapselearnPostCodeDynamics",
                              *this, modelMerged, backend.supportsNamespace());
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDynamicsGroupMerged::name = "SynapseDynamics";
//----------------------------------------------------------------------------
void SynapseDynamicsGroupMerged::generateSynapseUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
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

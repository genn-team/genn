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
void applySynapseSubstitutions(const BackendBase &backend, EnvironmentExternalBase &env, const std::vector<Transpiler::Token> &tokens, const std::string &errorContext,
                               G &sg, unsigned int batchSize)
{
    const auto *wu = sg.getArchetype().getWUModel();

    EnvironmentGroupMergedField<G> synEnv(env, sg);

    // Substitute parameter and derived parameter names
    synEnv.addParams(wu->getParamNames(), "", &SynapseGroupInternal::getWUParams, &G::isWUParamHeterogeneous);
    synEnv.addDerivedParams(wu->getDerivedParams(), "", &SynapseGroupInternal::getWUDerivedParams, &G::isWUDerivedParamHeterogeneous);
    synEnv.addExtraGlobalParams(wu->getExtraGlobalParams(), backend.getDeviceVarPrefix());

    // Substitute names of pre and postsynaptic weight update variable
    synEnv.template addVars<SynapseWUPreVarAdapter>(
        backend.getDeviceVarPrefix(),
        [&sg, batchSize](VarAccess a, const std::string&) 
        { 
            return sg.getPreWUVarIndex(batchSize, getVarAccessDuplication(a), "$(id_pre)");
        });
    synEnv.template addVars<SynapseWUPostVarAdapter>(
        backend.getDeviceVarPrefix(),
        [&sg, batchSize](VarAccess a, const std::string&) 
        { 
            return sg.getPostWUVarIndex(batchSize, getVarAccessDuplication(a), "$(id_post)");
        });

    
    // If this synapse group has a kernel
    if (!sg.getArchetype().getKernelSize().empty()) {
        // Add substitution
        synEnv.add(Type::Uint32, "id_kernel", "kernelInd", 
                   {synEnv.addInitialiser("const unsigned int kernelInd = " + getKernelIndex(sg) + ";")});
    }

    // If weights are individual, substitute variables for values stored in global memory
    if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        synEnv.template addVars<SynapseWUVarAdapter>(
            backend.getDeviceVarPrefix(),
            [&sg, batchSize](VarAccess a, const std::string&) 
            { 
                return sg.getSynVarIndex(batchSize, getVarAccessDuplication(a), "$(id_syn)");
            });
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

        synEnv.template addVars<SynapseWUVarAdapter>(
            backend.getDeviceVarPrefix(),
            [&sg, batchSize](VarAccess a, const std::string&) 
            { 
                return sg.getKernelVarIndex(batchSize, getVarAccessDuplication(a), "$(id_kernel)");
            });
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

    // Pretty print code back to environment
    Transpiler::ErrorHandler errorHandler("Synapse group '" + sg.getArchetype().getName() + "' weight update model " + errorContext);
    prettyPrintStatements(tokens, sg.getTypeContext(), synEnv, errorHandler);
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUParams(); });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUDerivedParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [varName](const SynapseGroupInternal &sg){ return sg.getWUVarInitialisers().at(varName).getParams(); });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [varName](const SynapseGroupInternal &sg) { return sg.getWUVarInitialisers().at(varName).getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getDerivedParams(); });
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPreSlot(unsigned int batchSize) const
{
    if(getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        return  (batchSize == 1) ? "$(_pre_delay_slot)" : "$(_pre_batch_delay_slot)";
    }
    else {
        return (batchSize == 1) ? "0" : "$(batch)";
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostSlot(unsigned int batchSize) const
{
    if(getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
        return  (batchSize == 1) ? "$(_post_delay_slot)" : "$(_post_batch_delay_slot)";
    }
    else {
        return (batchSize == 1) ? "0" : "$(batch)";
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostDenDelayIndex(unsigned int batchSize, const std::string &index, const std::string &offset) const
{
    const std::string batchID = ((batchSize == 1) ? "" : "$(_post_batch_offset) + ") + index;

    if(offset.empty()) {
        return "(*$(_den_delay_ptr) * $(num_post) + " + batchID;
    }
    else {
        return "(((*$(_den_delay_ptr) + " + offset + ") % " + std::to_string(getArchetype().getMaxDendriticDelayTimesteps()) + ") * $(num_post)) + " + batchID;
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPreVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    return getVarIndex(delay, batchSize, varDuplication, index, "pre");
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
   return getVarIndex(delay, batchSize, varDuplication, index, "post");
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPrePrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
   
    if(delay) {
        return (singleBatch ? "$(_pre_prev_spike_time_delay_offset) + " : "$(_pre_prev_spike_time_batch_delay_offset) + ") + index;
    }
    else {
        return (singleBatch ? "" : "$(_pre_batch_offset) + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostPrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
   
    if(delay) {
        return (singleBatch ? "$(_post_prev_spike_time_delay_offset) + " : "$(_post_prev_spike_time_batch_delay_offset) + ") + index;
    }
    else {
        return (singleBatch ? "" : "$(_post_batch_offset) + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getSynVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
    return (singleBatch ? "" : "$(_syn_batch_offset) + ") + index;
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getKernelVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
    return (singleBatch ? "" : "$(_kern_batch_offset) + ") + index;
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication,
                                                const std::string &index, const std::string &prefix) const
{
    if (delay) {
        if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
            return ((batchSize == 1) ? "$(_" + prefix + "_delay_slot)" : "$(_" + prefix + "_batch_delay_slot)");
        }
        else if (varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
            return "$(_" + prefix + "_delay_offset) + " + index;
        }
        else {
            return "$(_" + prefix + "_batch_delay_offset) + " + index;
        }
    }
    else {
        if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
            return (batchSize == 1) ? "0" : "$(batch)";
        }
        else if (varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
            return index;
        }
        else {
            return "$(_" + prefix + "_batch_offset) + " + index;
        }
    }
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroupMergedBase::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with number of neurons in pre and postsynaptic population
    updateHash([](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxConnections(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxSourceConnections(); }, hash);

    // Update hash with weight update model parameters and derived parameters
    updateHash([](const SynapseGroupInternal &g) { return g.getWUParams(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getWUDerivedParams(); }, hash);

    // If we're updating a hash for a group with procedural connectivity or initialising connectivity
    if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) {
        updateParamHash([](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); }, hash);
        updateParamHash([](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }, hash);
    }

    // If we're updating a hash for a group with Toeplitz connectivity
    if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) {
        // Update hash with connectivity parameters and derived parameters
        updateParamHash([](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getParams(); }, hash);

        updateParamHash([](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getDerivedParams(); }, hash);
    }

    // If weights are procedural
    if(getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL)  {
        // If synapse group has a kernel, update hash with kernel size
        if(!getArchetype().getKernelSize().empty()) {
            updateHash([](const SynapseGroupInternal &g) { return g.getKernelSize(); }, hash);
        }

        // Update hash with each group's variable initialisation parameters and derived parameters
        updateVarInitParamHash<SynapseWUVarAdapter>(hash);
        updateVarInitDerivedParamHash<SynapseWUVarAdapter>(hash);
    }

    return hash.get_digest();
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string PresynapticUpdateGroupMerged::name = "PresynapticUpdate";
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeEventThreshold(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
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
    Transpiler::ErrorHandler errorHandler("Synapse group '" + getArchetype().getName() + "' weight update model event threshold code");
    prettyPrintStatements(getArchetype().getWUEventThresholdCodeTokens(), getTypeContext(), synEnv, errorHandler);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeEventUpdate(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUEventCodeTokens(), "event code", *this, batchSize);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeUpdate(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUSimCodeTokens(), "sim code", *this, batchSize);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateProceduralConnectivity(const BackendBase &backend, EnvironmentExternalBase &env)
{
    // Create environment for group
    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(env, *this);

    // Substitute in parameters and derived parameters for initialising connectivity
    const auto &connectInit = getArchetype().getConnectivityInitialiser();
    groupEnv.addConnectInitParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                  &PresynapticUpdateGroupMerged::isSparseConnectivityInitParamHeterogeneous);
    groupEnv.addConnectInitDerivedParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                         &PresynapticUpdateGroupMerged::isSparseConnectivityInitDerivedParamHeterogeneous);
    groupEnv.addExtraGlobalParams(connectInit.getSnippet()->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", "");

    Transpiler::ErrorHandler errorHandler("Synapse group procedural connectivity '" + getArchetype().getName() + "' row build code");
    prettyPrintStatements(connectInit.getRowBuildCodeTokens(), getTypeContext(), groupEnv, errorHandler);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateToeplitzConnectivity(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                                Transpiler::TypeChecker::StatementHandler forEachSynapseTypeCheckHandler,
                                                                Transpiler::PrettyPrinter::StatementHandler forEachSynapsePrettyPrintHandler)
{
    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(env, *this);

    // Substitute in parameters and derived parameters for initialising connectivity
    const auto &connectInit = getArchetype().getToeplitzConnectivityInitialiser();
    groupEnv.addConnectInitParams("", &SynapseGroupInternal::getToeplitzConnectivityInitialiser,
                                  &PresynapticUpdateGroupMerged::isToeplitzConnectivityInitParamHeterogeneous);
    groupEnv.addConnectInitDerivedParams("", &SynapseGroupInternal::getToeplitzConnectivityInitialiser,
                                         &PresynapticUpdateGroupMerged::isToeplitzConnectivityInitDerivedParamHeterogeneous);
    groupEnv.addExtraGlobalParams(connectInit.getSnippet()->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", "");

    // Pretty print code back to environment
    Transpiler::ErrorHandler errorHandler("Synapse group '" + getArchetype().getName() + "' Toeplitz connectivity diagonal build code");
    prettyPrintStatements(getArchetype().getToeplitzConnectivityInitialiser().getDiagonalBuildCodeTokens(), 
                          getTypeContext(), groupEnv, errorHandler, forEachSynapseTypeCheckHandler,
                          forEachSynapsePrettyPrintHandler);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PostsynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string PostsynapticUpdateGroupMerged::name = "PostsynapticUpdate";
//----------------------------------------------------------------------------
void PostsynapticUpdateGroupMerged::generateSynapseUpdate(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUPostLearnCodeTokens(), "learn post code", *this, batchSize);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDynamicsGroupMerged::name = "SynapseDynamics";
//----------------------------------------------------------------------------
void SynapseDynamicsGroupMerged::generateSynapseUpdate(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUSynapseDynamicsCodeTokens(), "synapse dynamics", *this, batchSize);
}


//----------------------------------------------------------------------------
// CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDendriticDelayUpdateGroupMerged::name = "SynapseDendriticDelayUpdate";
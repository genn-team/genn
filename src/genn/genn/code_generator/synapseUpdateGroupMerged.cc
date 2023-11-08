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
void applySynapseSubstitutions(EnvironmentExternalBase &env, const std::vector<Transpiler::Token> &tokens, const std::string &errorContext,
                               G &sg, unsigned int batchSize, double dt)
{
    const auto *wu = sg.getArchetype().getWUInitialiser().getSnippet();

    EnvironmentGroupMergedField<G> synEnv(env, sg);

    // Substitute parameter and derived parameter names
    synEnv.addInitialiserParams("", &SynapseGroupInternal::getWUInitialiser, &G::isWUParamHeterogeneous);
    synEnv.addInitialiserDerivedParams("", &SynapseGroupInternal::getWUInitialiser, &G::isWUDerivedParamHeterogeneous);
    synEnv.addExtraGlobalParams(wu->getExtraGlobalParams());

    // Add referenced pre and postsynaptic neuron variables
    synEnv.template addVarRefs<SynapseWUPreNeuronVarRefAdapter>(
        [&sg, batchSize](VarAccessMode, const Models::VarReference &v)
        {
            return sg.getPreVarIndex(batchSize, v.getDelayNeuronGroup() != nullptr, 
                                     v.getVarDims(), "$(id_pre)");
        }, 
        "", true);
    synEnv.template addVarRefs<SynapseWUPostNeuronVarRefAdapter>(
        [&sg, batchSize](VarAccessMode, const Models::VarReference &v)
        {
            return sg.getPostVarIndex(batchSize, v.getDelayNeuronGroup() != nullptr, 
                                      v.getVarDims(), "$(id_post)");
        }, 
        "", true);

    // Substitute names of pre and postsynaptic weight update variable
    synEnv.template addVars<SynapseWUPreVarAdapter>(
        [&sg, batchSize](VarAccess a, const std::string&) 
        { 
            return sg.getPreWUVarIndex(batchSize, getVarAccessDim(a), "$(id_pre)");
        }, "", true);
    synEnv.template addVars<SynapseWUPostVarAdapter>(
        [&sg, batchSize](VarAccess a, const std::string&) 
        { 
            return sg.getPostWUVarIndex(batchSize, getVarAccessDim(a), "$(id_post)");
        }, "", true);

    
    // If this synapse group has a kernel
    if (!sg.getArchetype().getKernelSize().empty()) {
        // Add substitution
        synEnv.add(Type::Uint32, "id_kernel", "kernelInd", 
                   {synEnv.addInitialiser("const unsigned int kernelInd = " + getKernelIndex(sg) + ";")});
    }

    // Calculate axonal delays to add to (somatic) spike times and subsitute in presynaptic spike and spike-like event times
    const std::string timeStr = sg.getTimeType().getName();
    const std::string axonalDelayMs = Type::writeNumeric(dt * (double)(sg.getArchetype().getDelaySteps() + 1u), sg.getTimeType());
    const bool preDelay = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired();
    const std::string preSTIndex = sg.getPreVarIndex(preDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_pre)");
    const std::string prevPreSTIndex = sg.getPrePrevSpikeTimeIndex(preDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_pre)");
    synEnv.add(sg.getTimeType().addConst(), "st_pre", "stPre",
               {synEnv.addInitialiser("const " + timeStr + " stPre = " + axonalDelayMs + " + $(_src_st)[" + preSTIndex + "];")});
    synEnv.add(sg.getTimeType().addConst(), "prev_st_pre", "prevSTPre",
               {synEnv.addInitialiser("const " + timeStr + " prevSTPre = " + axonalDelayMs + " + $(_src_prev_st)[" + prevPreSTIndex + "];")});
    synEnv.add(sg.getTimeType().addConst(), "set_pre", "setPre",
               {synEnv.addInitialiser("const " + timeStr + " setPre = " + axonalDelayMs + " + $(_src_set)[" + preSTIndex + "];")});
    synEnv.add(sg.getTimeType().addConst(), "prev_set_pre", "prevSETPre",
               {synEnv.addInitialiser("const " + timeStr + " prevSETPre = " + axonalDelayMs + " + $(_src_prev_set)[" + prevPreSTIndex + "];")});

    // Calculate backprop delay to add to (somatic) spike times and substitute in postsynaptic spike times
    const std::string backPropDelayMs = Type::writeNumeric(dt * (double)(sg.getArchetype().getBackPropDelaySteps() + 1u), sg.getTimeType());
    const bool postDelay = sg.getArchetype().getTrgNeuronGroup()->isDelayRequired();
    const std::string postSTIndex = sg.getPostVarIndex(postDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_post)");
    const std::string prevPostSTIndex = sg.getPostPrevSpikeTimeIndex(postDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_post)");
    synEnv.add(sg.getTimeType().addConst(), "st_post", "stPost",
               {synEnv.addInitialiser("const " + timeStr + " stPost = " + backPropDelayMs + " + $(_trg_st)[" + postSTIndex + "];")});
    synEnv.add(sg.getTimeType().addConst(), "prev_st_post", "prevSTPost",
               {synEnv.addInitialiser("const " + timeStr + " prevSTPost = " + backPropDelayMs + " + $(_trg_prev_st)[" + prevPostSTIndex + "];")});

    // If weights are individual, substitute variables for values stored in global memory
    if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        synEnv.template addVars<SynapseWUVarAdapter>(
            [&sg, batchSize](VarAccess a, const std::string&) 
            { 
                return sg.getSynVarIndex(batchSize, getVarAccessDim(a), "$(id_syn)");
            });
    }
    // Otherwise, if weights are procedual
    else if (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL) {
        for(const auto &var : wu->getVars()) {
            // If this variable has any initialisation code
            const auto &varInit = sg.getArchetype().getWUInitialiser().getVarInitialisers().at(var.name);
            if(!Utils::areTokensEmpty(varInit.getCodeTokens())) {
                // Declare variable
                const auto resolvedType = var.type.resolve(sg.getTypeContext());
                synEnv.printLine(resolvedType.getName() + " _l" + var.name + ";");
                {
                    CodeStream::Scope b(synEnv.getStream());

                    // Substitute in parameters and derived parameters for initialising variables
                    // **THINK** synEnv has quite a lot of unwanted stuff at t
                    EnvironmentGroupMergedField<G> varInitEnv(synEnv, sg);
                    varInitEnv.template addVarInitParams<SynapseWUVarAdapter>(&G::isVarInitParamHeterogeneous, var.name);
                    varInitEnv.template addVarInitDerivedParams<SynapseWUVarAdapter>(&G::isVarInitDerivedParamHeterogeneous, var.name);
                    varInitEnv.addExtraGlobalParams(varInit.getSnippet()->getExtraGlobalParams(), var.name);

                    // Add read-write environment entry for variable
                    varInitEnv.add(resolvedType, "value", "_l" + var.name);

                    // Pretty print variable initialisation code
                    Transpiler::ErrorHandler errorHandler("Synapse group '" + sg.getArchetype().getName() + "' variable '" + var.name + "' init code");
                    prettyPrintStatements(varInit.getCodeTokens(), sg.getTypeContext(), varInitEnv, errorHandler);
                }

                // Add read-only environment entry for variable
                synEnv.add(resolvedType.addConst(), var.name, "_l" + var.name);
            }
        }
    }
    // Otherwise, if weights are kernels, use kernel index to index into variables
    else if(sg.getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL) {
        assert(!sg.getArchetype().getKernelSize().empty());

        synEnv.template addVars<SynapseWUVarAdapter>(
            [&sg, batchSize](VarAccess a, const std::string&) 
            { 
                return sg.getKernelVarIndex(batchSize, getVarAccessDim(a), "$(id_kernel)");
            });
    }


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
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUDerivedParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUInitialiser().getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [varName](const SynapseGroupInternal &sg){ return sg.getWUInitialiser().getVarInitialisers().at(varName).getParams(); });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [varName](const SynapseGroupInternal &sg) { return sg.getWUInitialiser().getVarInitialisers().at(varName).getDerivedParams(); });
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
    // **TODO** this is basically VarAccessDim::BATCH
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
    // **TODO** this is basically VarAccessDim::BATCH
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
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPrePrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDim varDims, const std::string &index) const
{
    const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);

    if(delay) {
        return (batched ? "$(_pre_prev_spike_time_batch_delay_offset) + " : "$(_pre_prev_spike_time_delay_offset) + " ) + index;
    }
    else {
        return (batched ? "$(_pre_batch_offset) + " : "") + index;
    }
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostPrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDim varDims, const std::string &index) const
{
    const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);
   
    if(delay) {
        return (batched ? "$(_post_prev_spike_time_batch_delay_offset) + " : "$(_post_prev_spike_time_delay_offset) + ") + index;
    }
    else {
        return (batched ? "$(_post_batch_offset) + " : "") + index;
    }
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getSynVarIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const
{
    const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);
    return (batched ? "$(_syn_batch_offset) + " : "") + index;
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getKernelVarIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const
{
    const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);
    return (batched ? "$(_kern_batch_offset) + " : "") + index;
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPrePostVarIndex(bool delay, unsigned int batchSize, VarAccessDim varDims,
                                                       const std::string &index, const std::string &prefix) const
{
    const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);
    if (delay) {
        if (!(varDims & VarAccessDim::ELEMENT)) {
            return (batched ? "$(_" + prefix + "_batch_delay_slot)" : "$(_" + prefix + "_delay_slot)");
        }
        else if(batched) {
            return "$(_" + prefix + "_batch_delay_offset) + " + index;
        }
        else {
            return "$(_" + prefix + "_delay_offset) + " + index;
        }
    }
    else {
        if (!(varDims & VarAccessDim::ELEMENT)) {
            return batched ? "$(batch)" : "0";
        }
        else if (batched) {
            return "$(_" + prefix + "_batch_offset) + " + index;
        }
        else {
            return index;
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
    updateHash([](const SynapseGroupInternal &g) { return g.getWUInitialiser().getParams(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getWUInitialiser().getDerivedParams(); }, hash);

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
void PresynapticUpdateGroupMerged::generateSpikeEventThreshold(EnvironmentExternalBase &env, unsigned int batchSize)
{
    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(env, *this);

    // Substitute parameter and derived parameter names
    const auto *wum = getArchetype().getWUInitialiser().getSnippet();
    synEnv.addInitialiserParams("", &SynapseGroupInternal::getWUInitialiser, &PresynapticUpdateGroupMerged::isWUParamHeterogeneous);
    synEnv.addInitialiserDerivedParams("", &SynapseGroupInternal::getWUInitialiser, &PresynapticUpdateGroupMerged::isWUDerivedParamHeterogeneous);
    synEnv.addExtraGlobalParams(wum->getExtraGlobalParams());

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
    prettyPrintStatements(getArchetype().getWUInitialiser().getEventThresholdCodeTokens(), getTypeContext(), synEnv, errorHandler);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeEventUpdate(EnvironmentExternalBase &env, 
                                                            unsigned int batchSize, double dt)
{
    applySynapseSubstitutions(env, getArchetype().getWUInitialiser().getEventCodeTokens(), "event code", *this, batchSize, dt);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeUpdate(EnvironmentExternalBase &env, 
                                                       unsigned int batchSize, double dt)
{
    applySynapseSubstitutions(env, getArchetype().getWUInitialiser().getSimCodeTokens(), "sim code", *this, batchSize, dt);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateProceduralConnectivity(EnvironmentExternalBase &env)
{
    // Create environment for group
    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(env, *this);

    // Substitute in parameters and derived parameters for initialising connectivity
    const auto &connectInit = getArchetype().getConnectivityInitialiser();
    groupEnv.addInitialiserParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                  &PresynapticUpdateGroupMerged::isSparseConnectivityInitParamHeterogeneous);
    groupEnv.addInitialiserDerivedParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                         &PresynapticUpdateGroupMerged::isSparseConnectivityInitDerivedParamHeterogeneous);
    groupEnv.addExtraGlobalParams(connectInit.getSnippet()->getExtraGlobalParams(), "", "");

    Transpiler::ErrorHandler errorHandler("Synapse group procedural connectivity '" + getArchetype().getName() + "' row build code");
    prettyPrintStatements(connectInit.getRowBuildCodeTokens(), getTypeContext(), groupEnv, errorHandler);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateToeplitzConnectivity(EnvironmentExternalBase &env, 
                                                                Transpiler::TypeChecker::StatementHandler forEachSynapseTypeCheckHandler,
                                                                Transpiler::PrettyPrinter::StatementHandler forEachSynapsePrettyPrintHandler)
{
    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(env, *this);

    // Substitute in parameters and derived parameters for initialising connectivity
    const auto &connectInit = getArchetype().getToeplitzConnectivityInitialiser();
    groupEnv.addInitialiserParams("", &SynapseGroupInternal::getToeplitzConnectivityInitialiser,
                                  &PresynapticUpdateGroupMerged::isToeplitzConnectivityInitParamHeterogeneous);
    groupEnv.addInitialiserDerivedParams("", &SynapseGroupInternal::getToeplitzConnectivityInitialiser,
                                         &PresynapticUpdateGroupMerged::isToeplitzConnectivityInitDerivedParamHeterogeneous);
    groupEnv.addExtraGlobalParams(connectInit.getSnippet()->getExtraGlobalParams(), "", "");

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
void PostsynapticUpdateGroupMerged::generateSynapseUpdate(EnvironmentExternalBase &env, 
                                                          unsigned int batchSize, double dt)
{
    applySynapseSubstitutions(env, getArchetype().getWUInitialiser().getPostLearnCodeTokens(), "learn post code", *this, batchSize, dt);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDynamicsGroupMerged::name = "SynapseDynamics";
//----------------------------------------------------------------------------
void SynapseDynamicsGroupMerged::generateSynapseUpdate(EnvironmentExternalBase &env, 
                                                       unsigned int batchSize, double dt)
{
    applySynapseSubstitutions(env, getArchetype().getWUInitialiser().getSynapseDynamicsCodeTokens(), "synapse dynamics", *this, batchSize, dt);
}


//----------------------------------------------------------------------------
// CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDendriticDelayUpdateGroupMerged::name = "SynapseDendriticDelayUpdate";
//----------------------------------------------------------------------------
void SynapseDendriticDelayUpdateGroupMerged::generateSynapseUpdate(EnvironmentExternalBase &env)
{
    env.printLine("*$(_den_delay_ptr) = (*$(_den_delay_ptr) + 1) % " + std::to_string(getArchetype().getMaxDendriticDelayTimesteps()) + ";");
}
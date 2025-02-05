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
template<typename A, typename G>
void addHeterogeneousDelayPostVarRefs(EnvironmentGroupMergedField<G> &env, const std::vector<Transpiler::Token> &tokens,
                                      G &sg, unsigned int batchSize)
{
    // Loop through variable references
    const A archetypeAdaptor(sg.getArchetype());
    for(const auto &v : archetypeAdaptor.getDefs()) {
        // If variable refernce is accessed with delay in synapse code tokens
        const auto resolvedType = v.type.resolve(sg.getTypeContext());
        const Models::VarReference &archetypeVarRef = archetypeAdaptor.getInitialisers().at(v.name);
        if(Utils::isIdentifierDelayed(v.name, tokens)) {
            env.addField(Type::getArraySubscript(resolvedType.addConst()), v.name,
                         resolvedType.createPointer(), v.name,
                         [v](auto &runtime, const auto &g, size_t) 
                         { 
                             return A(g).getInitialisers().at(v.name).getTargetArray(runtime); 
                         },
                         sg.getPostVarHetDelayIndex(batchSize, archetypeVarRef.getVarDims(), "$(id_post)"));
        }
        else {
            env.addField(resolvedType.addConst(), v.name,
                         resolvedType.createPointer(), v.name,
                         [v](auto &runtime, const auto &g, size_t) 
                         { 
                             return A(g).getInitialisers().at(v.name).getTargetArray(runtime); 
                         },
                         sg.getPostVarIndex(archetypeVarRef.getDelayNeuronGroup() != nullptr, batchSize,
                                            archetypeVarRef.getVarDims(), "$(id_post)"));
        }
    }
}
//--------------------------------------------------------------------------
template<typename G>
void applySynapseSubstitutions(const BackendBase &backend, EnvironmentExternalBase &env, const std::vector<Transpiler::Token> &tokens, const std::string &errorContext,
                               G &sg, unsigned int batchSize, double dt, bool wumVarsProvided)
{
    const auto *wu = sg.getArchetype().getWUInitialiser().getSnippet();

    EnvironmentGroupMergedField<G> synEnv(env, sg);

    // Substitute parameter and derived parameter names
    synEnv.addInitialiserParams("", &SynapseGroupInternal::getWUInitialiser, &SynapseGroupInternal::isWUParamDynamic);
    synEnv.addInitialiserDerivedParams("", &SynapseGroupInternal::getWUInitialiser);
    synEnv.addExtraGlobalParams(wu->getExtraGlobalParams());

    // Add referenced presynaptic neuron variables
    synEnv.template addVarRefs<SynapseWUPreNeuronVarRefAdapter>(
        [&sg, batchSize](VarAccessMode, const Models::VarReference &v)
        {
            return sg.getPreVarIndex(v.getDelayNeuronGroup() != nullptr, batchSize, 
                                     v.getVarDims(), "$(id_pre)");
        }, 
        "", true);

    // Add, potentially heterogeneously-delayed, references to postsynaptic model and neuron variables
    addHeterogeneousDelayPostVarRefs<SynapseWUPostNeuronVarRefAdapter>(synEnv, tokens, sg, batchSize);
    addHeterogeneousDelayPostVarRefs<SynapseWUPSMVarRefAdapter>(synEnv, tokens, sg, batchSize);

    // Substitute names of preynaptic weight update variables
    synEnv.template addVars<SynapseWUPreVarAdapter>(
        [&sg, batchSize](VarAccess a, const std::string&) 
        { 
            return sg.getPreVarIndex(sg.getArchetype().getAxonalDelaySteps() != 0, batchSize, getVarAccessDim(a), "$(id_pre)");
        }, "", true);

    // Loop through postsynaptic weight update variables
    for(const auto &v : sg.getArchetype().getWUInitialiser().getSnippet()->getPostVars()) {
        // If variable is accessed with delay in synapse code tokens
        const auto resolvedType = v.type.resolve(sg.getTypeContext());
        if(Utils::isIdentifierDelayed(v.name, tokens)) {
            synEnv.addField(Type::getArraySubscript(resolvedType.addConst()), v.name,
                            resolvedType.createPointer(), v.name,
                            [v](auto &runtime, const auto &g, size_t) 
                            { 
                                return runtime.getArray(g.getFusedWUPostTarget(), v.name);
                            },
                            sg.getPostVarHetDelayIndex(batchSize, getVarAccessDim(v.access), "$(id_post)"));
        }
        else {
            const bool delayed = (sg.getArchetype().getBackPropDelaySteps() != 0
                                  || sg.getArchetype().isWUPostVarHeterogeneouslyDelayed(v.name));
            synEnv.addField(resolvedType.addConst(), v.name,
                            resolvedType.createPointer(), v.name,
                            [v](auto &runtime, const auto &g, size_t) 
                            { 
                                return runtime.getArray(g.getFusedWUPostTarget(), v.name);
                            },
                            sg.getPostVarIndex(delayed, batchSize, getVarAccessDim(v.access), "$(id_post)"));
        }
    }
    
    // If this synapse group has a kernel
    if (!sg.getArchetype().getKernelSize().empty()) {
        // Add substitution
        synEnv.add(Type::Uint32, "id_kernel", "kernelInd", 
                   {synEnv.addInitialiser("const unsigned int kernelInd = " + getKernelIndex(sg) + ";")});
    }

    // Calculate axonal delays to add to (somatic) spike times and subsitute in presynaptic spike and spike-like event times
    const std::string timeStr = sg.getTimeType().getName();
    const std::string axonalDelayMs = Type::writeNumeric(dt * (double)(sg.getArchetype().getAxonalDelaySteps() + 1u), sg.getTimeType());
    const bool preSpikeDelay = sg.getArchetype().getSrcNeuronGroup()->isSpikeDelayRequired();
    const bool preSpikeEventDelay = sg.getArchetype().getSrcNeuronGroup()->isSpikeEventDelayRequired();
    const std::string preSTIndex = sg.getPreVarIndex(preSpikeDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_pre)");
    const std::string preSETIndex = sg.getPreVarIndex(preSpikeEventDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_pre)");
    const std::string prevPreSTIndex = sg.getPrePrevSpikeTimeIndex(preSpikeDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_pre)");
    const std::string prevPreSETIndex = sg.getPrePrevSpikeTimeIndex(preSpikeEventDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_pre)");
    synEnv.add(sg.getTimeType().addConst(), "st_pre", "stPre",
               {synEnv.addInitialiser("const " + timeStr + " stPre = " + axonalDelayMs + " + $(_src_st)[" + preSTIndex + "];")});
    synEnv.add(sg.getTimeType().addConst(), "prev_st_pre", "prevSTPre",
               {synEnv.addInitialiser("const " + timeStr + " prevSTPre = " + axonalDelayMs + " + $(_src_prev_st)[" + prevPreSTIndex + "];")});
    synEnv.add(sg.getTimeType().addConst(), "set_pre", "setPre",
               {synEnv.addInitialiser("const " + timeStr + " setPre = " + axonalDelayMs + " + $(_src_set)[" + preSETIndex + "];")});
    synEnv.add(sg.getTimeType().addConst(), "prev_set_pre", "prevSETPre",
               {synEnv.addInitialiser("const " + timeStr + " prevSETPre = " + axonalDelayMs + " + $(_src_prev_set)[" + prevPreSETIndex + "];")});

    // Calculate backprop delay to add to (somatic) spike times and substitute in postsynaptic spike times
    const std::string backPropDelayMs = Type::writeNumeric(dt * (double)(sg.getArchetype().getBackPropDelaySteps() + 1u), sg.getTimeType());
    const bool postSpikeDelay = sg.getArchetype().getTrgNeuronGroup()->isSpikeDelayRequired();
    const bool postSpikeEventDelay = sg.getArchetype().getTrgNeuronGroup()->isSpikeEventDelayRequired();
    const std::string postSTIndex = sg.getPostVarIndex(postSpikeDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_post)");
    const std::string postSETIndex = sg.getPostVarIndex(postSpikeEventDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_post)");
    const std::string prevPostSTIndex = sg.getPostPrevSpikeTimeIndex(postSpikeDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_post)");
    const std::string prevPostSETIndex = sg.getPostPrevSpikeTimeIndex(postSpikeEventDelay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id_post)");
    synEnv.add(sg.getTimeType().addConst(), "st_post", "stPost",
               {synEnv.addInitialiser("const " + timeStr + " stPost = " + backPropDelayMs + " + $(_trg_st)[" + postSTIndex + "];")});
    synEnv.add(sg.getTimeType().addConst(), "prev_st_post", "prevSTPost",
               {synEnv.addInitialiser("const " + timeStr + " prevSTPost = " + backPropDelayMs + " + $(_trg_prev_st)[" + prevPostSTIndex + "];")});
    synEnv.add(sg.getTimeType().addConst(), "set_post", "setPost",
               {synEnv.addInitialiser("const " + timeStr + " setPost = " + backPropDelayMs + " + $(_trg_set)[" + postSETIndex + "];")});
    synEnv.add(sg.getTimeType().addConst(), "prev_set_post", "prevSETPost",
               {synEnv.addInitialiser("const " + timeStr + " prevSETPost = " + backPropDelayMs + " + $(_trg_prev_set)[" + prevPostSETIndex + "];")});
    
    // If weights are individual, substitute variables for values stored in global memory
    if(wumVarsProvided) {
        synEnv.addVarExposeAliases<SynapseWUVarAdapter>();
    }
    else {
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
                        varInitEnv.template addVarInitParams<SynapseWUVarAdapter>(var.name);
                        varInitEnv.template addVarInitDerivedParams<SynapseWUVarAdapter>(var.name);
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

            // Add hidden fields with pointers to weight update model variables
            synEnv.template addVarPointers<SynapseWUVarAdapter>("", true);

            // Loop through weight update model variables
            for(const auto &v : sg.getArchetype().getWUInitialiser().getSnippet()->getVars()) {
                // Resolve types
                const auto resolvedType = v.type.resolve(sg.getTypeContext());
            
                // Add read-only accessors to de-reference pointers
                const std::string var = "$(_" + v.name + ")[" + sg.getKernelVarIndex(batchSize, getVarAccessDim(v.access), "$(id_kernel)") + "]";
                synEnv.add(resolvedType.addConst(), v.name, var);

                // If variable is read-write, also add function to add to it atomically
                if(getVarAccessMode(v.access) & VarAccessModeAttribute::READ_WRITE) {
                    synEnv.add(Type::ResolvedType::createFunction(resolvedType, {resolvedType}), "atomic_add_" + v.name,
                               backend.getAtomicOperation("&" + var, "$(0)", resolvedType));
                }
            }
        }
    }

    // Pretty print code back to environment
    Transpiler::ErrorHandler errorHandler("Synapse group '" + sg.getArchetype().getName() + "' weight update model " + errorContext);
    prettyPrintStatements(tokens, sg.getTypeContext(), synEnv, errorHandler);
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPreSlot(bool delay, unsigned int batchSize) const
{
    // **TODO** this is basically VarAccessDim::BATCH
    if(delay) {
        return  (batchSize == 1) ? "$(_pre_delay_slot)" : "$(_pre_batch_delay_slot)";
    }
    else {
        return (batchSize == 1) ? "0" : "$(batch)";
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostSlot(bool delay, unsigned int batchSize) const
{
    // **TODO** this is basically VarAccessDim::BATCH
    if(delay) {
        return  (batchSize == 1) ? "$(_post_delay_slot)" : "$(_post_batch_delay_slot)";
    }
    else {
        return (batchSize == 1) ? "0" : "$(batch)";
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostDenDelayIndex(unsigned int batchSize, const std::string &index, const std::string &offset) const
{
    assert(!offset.empty());
    const std::string batchID = ((batchSize == 1) ? "" : "$(_post_batch_den_delay_offset) + ") + index;

    return "(((*$(_den_delay_ptr) + " + offset + ") % " + std::to_string(getArchetype().getMaxDendriticDelayTimesteps()) + ") * $(num_post)) + " + batchID;
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
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostVarHetDelayIndex(unsigned int batchSize, VarAccessDim varDims,
                                                            const std::string &index) const
{
    const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);
    const unsigned int numTrgDelaySlots = getArchetype().getTrgNeuronGroup()->getNumDelaySlots();

    const std::string delaySlot = "((*$(_trg_spk_que_ptr) + " + std::to_string(numTrgDelaySlots) + " - $(0)) % " + std::to_string(numTrgDelaySlots) + ")";
    if (!(varDims & VarAccessDim::ELEMENT)) {
        if(batched) {
            return "($(batch) * " + std::to_string(numTrgDelaySlots) + ") " + delaySlot;
        }
        else {
            return delaySlot;
        }
    }
    else {
        const std::string delayOffset = "(" + delaySlot + " * $(num_post))";
        if(batched) {
            return delayOffset + " + ($(_post_batch_offset) * " + std::to_string(numTrgDelaySlots) + ") + " + index;
        }
        else {
            return delayOffset + " + " + index;
        }
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

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getWUHashDigest(), hash);

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
        updateParamHash([](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getParams(); }, hash);
        updateParamHash([](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getDerivedParams(); }, hash);
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
void PresynapticUpdateGroupMerged::generateSpikeEventUpdate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                            unsigned int batchSize, double dt, bool wumVarsProvided)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUInitialiser().getPreEventSynCodeTokens(), 
                              "presynaptic event code", *this, batchSize, dt, wumVarsProvided);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateSpikeUpdate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                       unsigned int batchSize, double dt, bool wumVarsProvided)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUInitialiser().getPreSpikeSynCodeTokens(),
                              "sim code", *this, batchSize, dt, wumVarsProvided);
}
//----------------------------------------------------------------------------
void PresynapticUpdateGroupMerged::generateProceduralConnectivity(EnvironmentExternalBase &env)
{
    // Create environment for group
    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(env, *this);

    // Substitute in parameters and derived parameters for initialising connectivity
    const auto &connectInit = getArchetype().getSparseConnectivityInitialiser();
    groupEnv.addInitialiserParams("", &SynapseGroupInternal::getSparseConnectivityInitialiser);
    groupEnv.addInitialiserDerivedParams("", &SynapseGroupInternal::getSparseConnectivityInitialiser);
    groupEnv.addExtraGlobalParams(connectInit.getSnippet()->getExtraGlobalParams(), "SparseConnect", "");

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
    groupEnv.addInitialiserParams("", &SynapseGroupInternal::getToeplitzConnectivityInitialiser);
    groupEnv.addInitialiserDerivedParams("", &SynapseGroupInternal::getToeplitzConnectivityInitialiser);
    groupEnv.addExtraGlobalParams(connectInit.getSnippet()->getExtraGlobalParams(), "ToeplitzConnect", "");

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
void PostsynapticUpdateGroupMerged::generateSpikeEventUpdate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                             unsigned int batchSize, double dt)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUInitialiser().getPostEventSynCodeTokens(),
                              "postsynaptic event code", *this, batchSize, dt, false);
}
//----------------------------------------------------------------------------
void PostsynapticUpdateGroupMerged::generateSpikeUpdate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                        unsigned int batchSize, double dt)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUInitialiser().getPostSpikeSynCodeTokens(), 
                              "learn post code", *this, batchSize, dt, false);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDynamicsGroupMerged::name = "SynapseDynamics";
//----------------------------------------------------------------------------
void SynapseDynamicsGroupMerged::generateSynapseUpdate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                       unsigned int batchSize, double dt)
{
    applySynapseSubstitutions(backend, env, getArchetype().getWUInitialiser().getSynapseDynamicsCodeTokens(), 
                              "synapse dynamics", *this, batchSize, dt, false);
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
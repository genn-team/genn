#include "code_generator/backendSIMT.h"

// Standard C++ includes
#include <algorithm>

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

//-----------------------------------------------------------------------
// Anonymous namespace
//-----------------------------------------------------------------------
namespace
{
template<typename T, typename G>
size_t getNumMergedGroupThreads(const std::vector<T> &groups, G getNumThreads)
{
    // Accumulate the accumulation of all groups in merged group
    return std::accumulate(
        groups.cbegin(), groups.cend(), size_t{0},
        [getNumThreads](size_t acc, const T &n)
        {
            return std::accumulate(n.getGroups().cbegin(), n.getGroups().cend(), acc,
                                   [getNumThreads](size_t acc, std::reference_wrapper<const typename T::GroupInternal> g)
            {
                return acc + getNumThreads(g.get());
            });
        });
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::BackendSIMT
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
const char *BackendSIMT::KernelNames[KernelMax] = {
    "updateNeuronsKernel",
    "updatePresynapticKernel",
    "updatePostsynapticKernel",
    "updateSynapseDynamicsKernel",
    "initializeKernel",
    "initializeSparseKernel",
    "neuronSpikeQueueUpdateKernel",
    "neuronPrevSpikeTimeUpdateKernel",
    "synapseDendriticDelayUpdateKernel",
    "customUpdate",
    "customTransposeUpdate"};
//--------------------------------------------------------------------------
std::vector<PresynapticUpdateStrategySIMT::Base*> BackendSIMT::s_PresynapticUpdateStrategies = {
    new PresynapticUpdateStrategySIMT::PreSpan,
    new PresynapticUpdateStrategySIMT::PostSpan,
    new PresynapticUpdateStrategySIMT::PreSpanProcedural,
    new PresynapticUpdateStrategySIMT::PostSpanBitmask,
    new PresynapticUpdateStrategySIMT::PostSpanToeplitz};
//--------------------------------------------------------------------------
size_t BackendSIMT::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    return getPresynapticUpdateStrategy(sg)->getSynapticMatrixRowStride(sg);
}
//--------------------------------------------------------------------------
void BackendSIMT::genPopVariableInit(EnvironmentExternalBase &env, HandlerEnv handler) const
{
    // If this is first thread in group
    env.getStream() << "if(" << env["id"] << " == 0)";
    {
        CodeStream::Scope b(env.getStream());
        handler(env);
    }
}
//--------------------------------------------------------------------------
void BackendSIMT::genVariableInit(EnvironmentExternalBase &env, const std::string&, const std::string&, HandlerEnv handler) const
{
    // Variable should already be provided via parallelism
    //assert(kernelSubs.hasVarSubstitution(countVarName));

    handler(env);
}
//--------------------------------------------------------------------------
void BackendSIMT::genKernelSynapseVariableInit(EnvironmentExternalBase &env, SynapseInitGroupMerged&, HandlerEnv handler) const
{
    // Variable should already be provided via parallelism
    //assert(kernelSubs.hasVarSubstitution("id"));
    
    EnvironmentExternal varEnv(env);
    varEnv.add(Type::Uint32.addConst(), "id_syn", "$(id)");

    handler(varEnv);
}
//--------------------------------------------------------------------------
void BackendSIMT::genKernelCustomUpdateVariableInit(EnvironmentExternalBase &env, CustomWUUpdateInitGroupMerged &, HandlerEnv handler) const
{
    // Variable should already be provided via parallelism
    //assert(kernelSubs.hasVarSubstitution("id"));

    EnvironmentExternal varEnv(env);
    varEnv.add(Type::Uint32.addConst(), "id_syn", "$(id)");

    handler(varEnv);
}
//--------------------------------------------------------------------------
bool BackendSIMT::isGlobalHostRNGRequired(const ModelSpecInternal &model) const
{
    // Host RNG is required if any synapse groups or custom connectivity updates require a host RNG
    return (std::any_of(model.getSynapseGroups().cbegin(), model.getSynapseGroups().cend(),
                        [](const ModelSpec::SynapseGroupValueType &s){ return s.second.getConnectivityInitialiser().isHostRNGRequired(); })
            || std::any_of(model.getCustomConnectivityUpdates().cbegin(), model.getCustomConnectivityUpdates().cend(),
                           [](const ModelSpec::CustomConnectivityUpdateValueType &c){ return Utils::isRNGRequired(c.second.getHostUpdateCodeTokens()); }));
}
//--------------------------------------------------------------------------
bool BackendSIMT::isGlobalDeviceRNGRequired(const ModelSpecInternal &model) const
{
    // If any neuron groups require  RNG for initialisation, return true
    // **NOTE** this takes postsynaptic model initialisation into account
    if(std::any_of(model.getNeuronGroups().cbegin(), model.getNeuronGroups().cend(),
                   [](const ModelSpec::NeuronGroupValueType &n){ return n.second.isInitRNGRequired(); }))
    {
        return true;
    }

    // If any synapse groups require an RNG for weight update model initialisation or procedural connectivity, return true
    if(std::any_of(model.getSynapseGroups().cbegin(), model.getSynapseGroups().cend(),
                   [](const ModelSpec::SynapseGroupValueType &s)
                   {
                       return (s.second.isWUInitRNGRequired() || s.second.isProceduralConnectivityRNGRequired());
                   }))
    {
        return true;
    }

    // If any custom updates require an RNG fo initialisation, return true
    if(std::any_of(model.getCustomUpdates().cbegin(), model.getCustomUpdates().cend(),
                   [](const ModelSpec::CustomUpdateValueType &c)
                   {
                       return (c.second.isInitRNGRequired());
                   }))
    {
        return true;
    }

    // If any custom WU updates require an RNG fo initialisation, return true
    if(std::any_of(model.getCustomWUUpdates().cbegin(), model.getCustomWUUpdates().cend(),
                   [](const ModelSpec::CustomUpdateWUValueType &c)
                   {
                       return (c.second.isInitRNGRequired());
                   }))
    {
        return true;
    }

    return false;
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getNumInitialisationRNGStreams(const ModelSpecMerged &modelMerged) const
{
    // Calculate total number of threads used for neuron initialisation group
    size_t numInitThreads = getNumMergedGroupThreads(modelMerged.getMergedNeuronInitGroups(),
                                                     [this](const NeuronGroupInternal &ng)
                                                     {
                                                         return padKernelSize(ng.getNumNeurons(), KernelInitialize);
                                                     });

    // Add on total number of threads used for custom update initialisation
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedCustomUpdateInitGroups(),
                                               [this](const CustomUpdateInternal &cg)
                                               {
                                                   return padKernelSize(cg.getSize(), KernelInitialize);
                                               });
    
    // Add on total number of threads used for custom WU update initialization
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedCustomWUUpdateInitGroups(),
                                               [this](const CustomUpdateWUInternal &cg)
                                               {
                                                   return padKernelSize(getNumInitThreads(cg), KernelInitialize);
                                               });
    
    // Add on total number of threads used for synapse initialisation
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedSynapseInitGroups(),
                                               [this](const SynapseGroupInternal &sg)
                                               {
                                                   return padKernelSize(getNumInitThreads(sg), KernelInitialize);
                                               });

    // Add on total number of threads used for synapse connectivity initialisation
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedSynapseConnectivityInitGroups(),
                                               [this](const SynapseGroupInternal &sg)
                                               {
                                                   return padKernelSize(getNumConnectivityInitThreads(sg), KernelInitialize);
                                               });

    // Add on total number of threads used for sparse synapse initialisation
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedSynapseSparseInitGroups(),
                                               [this](const SynapseGroupInternal &sg)
                                               {
                                                   return padKernelSize(sg.getMaxConnections(), KernelInitializeSparse);
                                               });
    
    // Finally, add on total number of threads used for custom WU update groups with sparse connectivity
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedCustomWUUpdateSparseInitGroups(),
                                               [this](const CustomUpdateWUInternal &cg)
                                               {
                                                   return padKernelSize(cg.getSynapseGroup()->getMaxConnections(), KernelInitializeSparse);
                                               });
    return numInitThreads;
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getPaddedNumCustomUpdateThreads(const CustomUpdateInternal &cg, unsigned int batchSize) const
{
    const size_t numCopies = ((cg.getDims() & VarAccessDim::BATCH) && !cg.isBatchReduction()) ? batchSize : 1;

    if (cg.isNeuronReduction()) {
        return padKernelSize(32 * numCopies, KernelCustomUpdate);
    }
    else if (cg.getDims() & VarAccessDim::ELEMENT) {
        return numCopies * padKernelSize(cg.getSize(), KernelCustomUpdate);
    }
    else {
        return padKernelSize(numCopies, KernelCustomUpdate);
    }
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getPaddedNumCustomUpdateWUThreads(const CustomUpdateWUInternal &cg, unsigned int batchSize) const
{
    const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
    const size_t numCopies = ((cg.getDims() & VarAccessDim::BATCH) && !cg.isBatchReduction()) ? batchSize : 1;

    if(sgInternal->getMatrixType() & SynapseMatrixWeight::KERNEL) {
        return numCopies * padKernelSize(sgInternal->getKernelSizeFlattened(), KernelCustomUpdate);
    }
    else {
        return numCopies * padKernelSize((size_t)sgInternal->getSrcNeuronGroup()->getNumNeurons() * sgInternal->getMaxConnections(),
                                         KernelCustomUpdate);
    }
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getPaddedNumCustomUpdateTransposeWUThreads(const CustomUpdateWUInternal &cg, unsigned int batchSize) const
{
    assert(cg.isTransposeOperation());
    assert(cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE);
    
    const size_t paddedNumPre = padKernelSize(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(), KernelCustomTransposeUpdate);
    const size_t paddedNumPost = padKernelSize(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(), KernelCustomTransposeUpdate);
    const size_t numCopies = (cg.getDims() & VarAccessDim::BATCH) ? batchSize : 1;
    return numCopies * paddedNumPre * paddedNumPost / getKernelBlockSize(KernelCustomTransposeUpdate);
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getNumPresynapticUpdateThreads(const SynapseGroupInternal &sg, const PreferencesBase &preferences)
{
    return getPresynapticUpdateStrategy(sg, preferences)->getNumThreads(sg);
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getNumPostsynapticUpdateThreads(const SynapseGroupInternal &sg)
{
    if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxSourceConnections();
    }
    else {
        return sg.getSrcNeuronGroup()->getNumNeurons();
    }
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getNumSynapseDynamicsThreads(const SynapseGroupInternal &sg)
{
    if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        // **THINK** this isn't really correct but correct value is inaccesible
        return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * sg.getMaxConnections();
    }
    else {
        return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getNumConnectivityInitThreads(const SynapseGroupInternal &sg)
{
    // If there's row building code, return number of source neurons i.e. rows
    if(!Utils::areTokensEmpty(sg.getConnectivityInitialiser().getRowBuildCodeTokens())) {
        return sg.getSrcNeuronGroup()->getNumNeurons();
    }
    // Otherwise, if there's column building code, return number of target neurons i.e. columns
    else if(!Utils::areTokensEmpty(sg.getConnectivityInitialiser().getColBuildCodeTokens())) {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
    // Otherwise, give an error
    else {
        throw std::runtime_error("Cannot calculate number of connectivity init threads without connectivity building code");
    }
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getNumInitThreads(const SynapseGroupInternal &sg)
{
    if (sg.getMatrixType() & SynapseMatrixWeight::KERNEL) {
        return sg.getKernelSizeFlattened();
    }
    else {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getNumInitThreads(const CustomUpdateWUInternal &cg)
{
    if (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL) {
        return cg.getSynapseGroup()->getKernelSizeFlattened();
    }
    else {
        return cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons();
    }
}
//--------------------------------------------------------------------------
void BackendSIMT::addPresynapticUpdateStrategy(PresynapticUpdateStrategySIMT::Base *strategy)
{
    s_PresynapticUpdateStrategies.push_back(strategy);
}
//--------------------------------------------------------------------------
void BackendSIMT::genNeuronPrevSpikeTimeUpdateKernel(EnvironmentExternalBase &env, ModelSpecMerged &modelMerged,
                                                     BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const
{
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();

    // Parallelise over neuron groups
    idStart = 0;
    genParallelGroup<NeuronPrevSpikeTimeUpdateGroupMerged>(
        env, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedNeuronPrevSpikeTimeUpdateGroups,
        [this](const NeuronGroupInternal &ng) { return padKernelSize(ng.getNumNeurons(), KernelNeuronUpdate); },
        [batchSize, this](EnvironmentExternalBase &popEnv, NeuronPrevSpikeTimeUpdateGroupMerged &ng)
        {
            CodeStream::Scope b(popEnv.getStream());

            // Create matching environment
            EnvironmentGroupMergedField<NeuronPrevSpikeTimeUpdateGroupMerged> neuronEnv(popEnv, ng);
            buildStandardEnvironment(neuronEnv, batchSize);

            // If neuron group requires delays
            if(ng.getArchetype().isDelayRequired()) {
                if(batchSize == 1) {
                    neuronEnv.printLine("const unsigned int lastTimestepDelaySlot = *$(_spk_que_ptr);");
                }
                else {
                    neuronEnv.printLine("const unsigned int lastTimestepDelaySlot = *$(_spk_que_ptr) + ($(batch) *  " + std::to_string(ng.getArchetype().getNumDelaySlots()) + ");");
                }
                neuronEnv.printLine("const unsigned int lastTimestepDelayOffset = lastTimestepDelaySlot * $(num_neurons);");

                if(ng.getArchetype().isPrevSpikeTimeRequired()) {
                    // If there is a spike for this thread, set previous spike time to time of last timestep
                    // **NOTE** spkQuePtr is updated below so this already points to last timestep
                    neuronEnv.print("if($(id) < $(_spk_cnt)[lastTimestepDelaySlot])");
                    {
                        CodeStream::Scope b(neuronEnv.getStream());
                        neuronEnv.printLine("$(_prev_st)[lastTimestepDelayOffset + $(_spk)[lastTimestepDelayOffset + $(id)]] = $(t) - $(dt);");
                    }
                }
                if(ng.getArchetype().isPrevSpikeEventTimeRequired()) {
                    // If there is a spike-like-event for this thread, set previous spike-like-event time to time of last timestep
                    // **NOTE** spkQuePtr is updated below so this already points to last timestep
                    neuronEnv.print("if($(id) < $(_spk_cnt_envt)[lastTimestepDelaySlot])");
                    {
                        CodeStream::Scope b(neuronEnv.getStream());
                        neuronEnv.printLine("$(_prev_set)[lastTimestepDelayOffset + $(_spk_evnt)[lastTimestepDelayOffset + $(id)]] = $(t) - $(dt);");
                    }
                }
            }
            // Otherwise
            else {
                if(ng.getArchetype().isPrevSpikeTimeRequired()) {
                    // If there is a spike for this thread, set previous spike time to time of last timestep
                    neuronEnv.print("if($(id) < $(_spk_cnt)[$(batch)])");
                    {
                        CodeStream::Scope b(neuronEnv.getStream());
                        neuronEnv.print("$(_prev_st)[");
                        if (batchSize == 1) {
                            neuronEnv.print("$(_spk)[$(id)]");
                        }
                        else {
                            neuronEnv.print("$(_batch_offset) + $(_spk)[$(_batch_offset) + $(id)]");
                        }
                        neuronEnv.printLine("] = $(t) - $(dt);");
                    }
                }
                if(ng.getArchetype().isPrevSpikeEventTimeRequired()) {
                    // If there is a spike-like-event for this thread, set previous spike-like-event time to time of last timestep
                    neuronEnv.print("if($(id) < $(_spk_cnt_evnt)[$(batch)])");
                    {
                        CodeStream::Scope b(neuronEnv.getStream());
                        neuronEnv.print("$(_prev_set)[");
                        if (batchSize == 1) {
                            neuronEnv.print("$(_spk_evnt)[$(id)]");
                        }
                        else {
                            neuronEnv.print("$(_batch_offset) + $(_spk_evnt)[$(_batch_offset) + $(id)]");
                        }
                        neuronEnv.printLine("] = $(t) - $(dt);");
                    }
                }
            }
            neuronEnv.getStream() << std::endl;
        });

}
//--------------------------------------------------------------------------
void BackendSIMT::genNeuronSpikeQueueUpdateKernel(EnvironmentExternalBase &env, ModelSpecMerged &modelMerged, 
                                                  BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const
{
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();

    // Loop through local neuron groups
    idStart = 0;
    modelMerged.genMergedNeuronSpikeQueueUpdateGroups(
        *this, memorySpaces,
        [&env, &idStart, batchSize, this](auto &n)
        {
            if(idStart == 0) {
                env.getStream() << "if(id < " << n.getGroups().size() << ")";
            }
            else {
                env.getStream() << "if(id >= " << idStart << " && id < " << idStart + n.getGroups().size() << ")";
            }
            {
                CodeStream::Scope b(env.getStream());

                // Use this to get reference to merged group structure
                env.getStream() << getPointerPrefix() << "struct MergedNeuronSpikeQueueUpdateGroup" << n.getIndex() << " *group = &d_mergedNeuronSpikeQueueUpdateGroup" << n.getIndex() << "[id - " << idStart << "]; " << std::endl;
            
                // Create matching environment
                EnvironmentGroupMergedField<NeuronSpikeQueueUpdateGroupMerged> neuronEnv(env, n);
                buildStandardEnvironment(neuronEnv, batchSize);
                n.genSpikeQueueUpdate(neuronEnv, batchSize);
            }
            idStart += n.getGroups().size();
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genNeuronUpdateKernel(EnvironmentExternalBase &env, ModelSpecMerged &modelMerged,
                                        BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const
{
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();

    // Generate code to zero shared memory spike event count using thread 1
    std::ostringstream shSpkCountInitStream;
    CodeStream shSpkCountInit(shSpkCountInitStream);
    shSpkCountInit << getSharedPrefix() << "unsigned int shSpkCount;" << std::endl;
    shSpkCountInit << "if (" << getThreadID() << " == 1)";
    {
        CodeStream::Scope b(shSpkCountInit);
        shSpkCountInit << "shSpkCount = 0;" << std::endl;
    }

    // Generate code to zero shared memory spike event count using thread 1
    std::ostringstream shSpkEvntCountInitStream;
    CodeStream shSpkEvntCountInit(shSpkEvntCountInitStream);
    shSpkEvntCountInit << getSharedPrefix() << "unsigned int shSpkEvntCount;" << std::endl;
    shSpkEvntCountInit << "if (" << getThreadID() << " == 1)";
    {
        CodeStream::Scope b(shSpkEvntCountInit);
        shSpkEvntCountInit << "shSpkEvntCount = 0;" << std::endl;
    }

    // Add shared memory substitutions so they're only instantiated as required
    EnvironmentExternal neuronEnv(env);
    const std::string blockSizeStr = std::to_string(getKernelBlockSize(KernelNeuronUpdate));
    neuronEnv.add(Type::Void, "_sh_spk", "shSpk",
                  {neuronEnv.addInitialiser(getSharedPrefix() + "unsigned int shSpk[" + blockSizeStr + "];")});
    neuronEnv.add(Type::Void, "_sh_spk_pos", "shSpkPos",
                  {neuronEnv.addInitialiser(getSharedPrefix() + "unsigned int shSpkPos;")});
    neuronEnv.add(Type::Void, "_sh_spk_count", "shSpkCount",
                  {neuronEnv.addInitialiser(shSpkCountInitStream.str())});
    neuronEnv.add(Type::Void, "_sh_spk_evnt", "shSpkEvnt",
                  {neuronEnv.addInitialiser(getSharedPrefix() + "unsigned int shSpkEvnt[" + blockSizeStr + "];")});
    neuronEnv.add(Type::Void, "_sh_spk_evnt_pos", "shSpkEvntPos",
                  {neuronEnv.addInitialiser(getSharedPrefix() + "unsigned int shSpkEvntPos;")});
    neuronEnv.add(Type::Void, "_sh_spk_evnt_count", "shSpkEvntCount",
                  {neuronEnv.addInitialiser(shSpkEvntCountInitStream.str())});

    // If any neuron groups record spikes
    if(std::any_of(modelMerged.getModel().getNeuronGroups().cbegin(), modelMerged.getModel().getNeuronGroups().cend(),
                   [](const auto &n) { return n.second.isSpikeRecordingEnabled(); }))
    {
        genRecordingSharedMemInit(env.getStream(), "");
    }

    // If any neuron groups record spike-like events
    if(std::any_of(modelMerged.getModel().getNeuronGroups().cbegin(), modelMerged.getModel().getNeuronGroups().cend(),
                   [](const auto &n) { return n.second.isSpikeEventRecordingEnabled(); }))
    {
        genRecordingSharedMemInit(env.getStream(), "Evnt");
    }

    genSharedMemBarrier(neuronEnv.getStream());

    // Parallelise over neuron groups
    idStart = 0;
    genParallelGroup<NeuronUpdateGroupMerged>(
        neuronEnv, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedNeuronUpdateGroups,
        [this](const NeuronGroupInternal &ng) { return padKernelSize(ng.getNumNeurons(), KernelNeuronUpdate); },
        [batchSize, &modelMerged, this](EnvironmentExternalBase &popEnv, NeuronUpdateGroupMerged &ng)
        {
            CodeStream::Scope b(popEnv.getStream());
            EnvironmentGroupMergedField<NeuronUpdateGroupMerged> groupEnv(popEnv, ng);
            buildStandardEnvironment(groupEnv, batchSize);

            // Call handler to generate generic neuron code
            groupEnv.print("if($(id) < $(num_neurons))");
            {
                CodeStream::Scope b(groupEnv.getStream());

                // Add population RNG field
                groupEnv.addField(getPopulationRNGType().createPointer(), "_rng", "rng",
                                  [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "rng"); },
                                  ng.getVarIndex(batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)"));
                // **TODO** for OCL do genPopulationRNGPreamble(os, popSubs, "group->rng[" + ng.getVarIndex(batchSize, VarAccessDuplication::DUPLICATE, "$(id)") + "]") in initialiser

                ng.generateNeuronUpdate(*this, groupEnv, batchSize,
                                        // Emit true spikes
                                        [this](EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng)
                                        {
                                            genEmitSpike(env, "", ng.getArchetype().isTrueSpikeRequired(),
                                                         ng.getArchetype().isSpikeRecordingEnabled());
                                        },
                                        // Emit spike-like events
                                        [this](EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng)
                                        {
                                            genEmitSpike(env, "_evnt", ng.getArchetype().isSpikeEventRequired(),
                                                         ng.getArchetype().isSpikeEventRecordingEnabled());
                                        });

                // Copy local stream back to local
                // **TODO** postamble for OCL
                //if(ng.getArchetype().isSimRNGRequired()) {
                //    genPopulationRNGPostamble(neuronEnv.getStream(), rng);
                //}
            }

            genSharedMemBarrier(groupEnv.getStream());

             // Copy spikes into block of $(_spk)
            const std::string queueOffset = ng.getWriteVarIndex(ng.getArchetype().isDelayRequired(), batchSize, 
                                                                VarAccessDim::BATCH | VarAccessDim::ELEMENT, "");

            ng.generateSpikes(
                groupEnv,
                [batchSize, &ng, &queueOffset, this](EnvironmentExternalBase &env)
                {
                    // Use first thread to 'allocate' block of $(_spk) array for this block's spikes
                    env.getStream() << "if(" << getThreadID() << " == 0)";
                    {
                        CodeStream::Scope b(env.getStream());
                        env.print("if ($(_sh_spk_count) > 0)");
                        {
                            CodeStream::Scope b(env.getStream());
                            env.print("$(_sh_spk_pos) = " + getAtomic(Type::Uint32) + "(&$(_spk_cnt)");
                            if(ng.getArchetype().isDelayRequired() && ng.getArchetype().isTrueSpikeRequired()) {
                                env.print("[*$(_spk_que_ptr)");
                                if(batchSize > 1) {
                                    env.getStream() << " + (batch * " << ng.getArchetype().getNumDelaySlots() << ")";
                                }
                                env.printLine("], $(_sh_spk_count));");
                            }
                            else {
                                env.printLine("[$(batch)], $(_sh_spk_count));");
                            }
                        }
                    } 
                    genSharedMemBarrier(env.getStream());

                    // Copy spikes from shared memory into global memory using $(_sh_spk_count) threads
                    env.print("if(" + getThreadID() + " < $(_sh_spk_count))");
                    {
                        CodeStream::Scope b(env.getStream());

                        env.printLine("const unsigned int n = $(_sh_spk)[" + getThreadID() + "];");

                        // Create new substition stack and explicitly replace id with 'n' and perform WU var update
                        EnvironmentExternal wuEnv(env);
                        wuEnv.add(Type::Uint32.addConst(), "id", "n");

                        // Create an environment which caches neuron variable fields in local variables if they are accessed
                        // **NOTE** we do this right at the top so that local copies can be used by child groups
                        // **NOTE** always copy variables if variable is delayed
                        EnvironmentLocalVarCache<NeuronVarAdapter, NeuronUpdateGroupMerged> wuVarEnv(
                            ng, ng, ng.getTypeContext(), wuEnv, "", "l", true, true,
                            [batchSize, &ng](const std::string &varName, VarAccess d)
                            {
                                const bool delayed = (ng.getArchetype().isVarQueueRequired(varName) && ng.getArchetype().isDelayRequired());
                                return ng.getReadVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)") ;
                            },
                            [batchSize, &ng](const std::string &varName, VarAccess d)
                            {
                                const bool delayed = (ng.getArchetype().isVarQueueRequired(varName) && ng.getArchetype().isDelayRequired());
                                return ng.getWriteVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)") ;
                            });
                        ng.generateWUVarUpdate(wuEnv, batchSize);

                        env.printLine("$(_spk)[" + queueOffset + "$(_sh_spk_pos) + " + getThreadID() + "] = n;");
                        /*if(ng.getArchetype().isSpikeTimeRequired()) {
                            env.printLine("$(_st)[" + queueOffset + "n] = $(t);");
                        }*/
                    }
                });
            

            // Use second thread to 'allocate' block of $(_spk_evnt) array for this block's spike-like events
            if(ng.getArchetype().isSpikeEventRequired()) {
                groupEnv.getStream() << "if (" << getThreadID() << " == 1)";
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    groupEnv.print("if($(_sh_spk_evnt_count) > 0)");
                    {
                        CodeStream::Scope b(groupEnv.getStream());
                        groupEnv.print("$(_sh_spk_evnt_pos) = " + getAtomic(Type::Uint32) + "(&$(_spk_cnt_evnt)");
                        if(ng.getArchetype().isDelayRequired()) {
                            groupEnv.print("[*$(_spk_que_ptr)");
                            if(batchSize > 1) {
                                groupEnv.getStream() << " + (batch * " << ng.getArchetype().getNumDelaySlots() << ")";
                            }
                            groupEnv.printLine("], $(_sh_spk_evnt_count));");
                        }
                        else {
                            groupEnv.printLine("[$(batch)], $(_sh_spk_evnt_count));");
                        }
                    }
                } 
                genSharedMemBarrier(groupEnv.getStream());
            }

            // Copy spike-like events into block of $(_spk_evnt)
            if(ng.getArchetype().isSpikeEventRequired()) {
                groupEnv.print("if(" + getThreadID() + " < $(_sh_spk_evnt_count))");
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    groupEnv.printLine("const unsigned int n = $(_sh_spk_evnt)[" + getThreadID() + "];");

                    groupEnv.printLine("$(_spk_evnt)[" + queueOffset + "$(_sh_spk_evnt_pos) + " + getThreadID() + "] = n;");
                    if(ng.getArchetype().isSpikeEventTimeRequired()) {
                        groupEnv.printLine("$(_set)[" + queueOffset + "n] = $(t);");
                    }
                }
            }

            // If we're recording spikes or spike-like events, use enough threads to copy this block's recording words
            if(ng.getArchetype().isSpikeRecordingEnabled() || ng.getArchetype().isSpikeEventRecordingEnabled()) {
                groupEnv.getStream() << "if(" << getThreadID() << " < " << m_KernelBlockSizes[KernelNeuronUpdate] / 32 << ")";
                {
                    CodeStream::Scope b(groupEnv.getStream());

                    // Calculate number of words which will be used to record this population's spikes in each batch
                    groupEnv.printLine("const unsigned int numRecordingWords = ($(num_neurons) + 31) / 32;");
                    groupEnv.printLine("const unsigned int popWordIdx = ($(id) / 32) + " + getThreadID() + ";");

                    // Build global index
                    std::string globalIndex = "(recordingTimestep * numRecordingWords * " + std::to_string(batchSize) + ") + popWordIdx";
                    if(batchSize > 1) {
                        globalIndex += " + (batch * numRecordingWords)";
                    }

                    groupEnv.getStream() << "if(popWordIdx < numRecordingWords)";
                    {
                        CodeStream::Scope c(groupEnv.getStream());
                        // If we are recording spikes, copy word to correct location in global memory
                        if(ng.getArchetype().isSpikeRecordingEnabled()) {
                            groupEnv.print("$(_record_spk)[" + globalIndex + "] = shSpkRecord");
                            if(m_KernelBlockSizes[KernelNeuronUpdate] != 32) {
                                groupEnv.print("[" + getThreadID() + "]");
                            }
                            groupEnv.printLine(";");
                        }

                        // If we are recording spike-like events, copy word to correct location in global memory
                        if(ng.getArchetype().isSpikeEventRecordingEnabled()) {
                            groupEnv.print("$(_record_spk_evnt)[" + globalIndex + "] = shSpkEvntRecord");
                            if(m_KernelBlockSizes[KernelNeuronUpdate] != 32) {
                                groupEnv.print("[" + getThreadID() + "]");
                            }
                            groupEnv.printLine(";");
                        }
                    }
                }
            }
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genSynapseDendriticDelayUpdateKernel(EnvironmentExternalBase &env, ModelSpecMerged &modelMerged, 
                                                       BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const
{
    // Loop through merged synapse groups
    idStart = 0;
    modelMerged.genMergedSynapseDendriticDelayUpdateGroups(
        *this, memorySpaces,
        [&env, &idStart, &modelMerged, this](auto &sg)
        {
            env.getStream() << "// merged" << sg.getIndex() << std::endl;
            if(idStart == 0) {
                env.getStream() << "if(id < " << sg.getGroups().size() << ")";
            }
            else {
                env.getStream() << "if(id >= " << idStart << " && id < " << idStart + sg.getGroups().size() << ")";
            }
            {
                CodeStream::Scope b(env.getStream());

                // Use this to get reference to merged group structure
                env.getStream() << getPointerPrefix() << "struct MergedSynapseDendriticDelayUpdateGroup" << sg.getIndex() << " *group = &d_mergedSynapseDendriticDelayUpdateGroup" << sg.getIndex() << "[id - " << idStart << "]; " << std::endl;
                EnvironmentGroupMergedField<SynapseDendriticDelayUpdateGroupMerged> groupEnv(env, sg);
                buildStandardEnvironment(groupEnv, modelMerged.getModel().getBatchSize());
                sg.generateSynapseUpdate(groupEnv);
            }
            idStart += sg.getGroups().size();
        });
    env.getStream() << std::endl;
}
//--------------------------------------------------------------------------
void BackendSIMT::genPresynapticUpdateKernel(EnvironmentExternalBase &env, ModelSpecMerged &modelMerged, 
                                             BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const
{
    EnvironmentExternal kernelEnv(env);

    // We need shOutPost if any synapse groups accumulate into shared memory
    // Determine the maximum shared memory outputs 
    size_t maxSharedMemPerThread = 0;
    for(const auto &s : modelMerged.getModel().getSynapseGroups()) {
        if(s.second.isSpikeEventRequired() || s.second.isTrueSpikeRequired()) {
            maxSharedMemPerThread = std::max(maxSharedMemPerThread,
                                              getPresynapticUpdateStrategy(s.second)->getSharedMemoryPerThread(s.second, *this));
        }
    }

    // If any shared memory is required, declare array
    if(maxSharedMemPerThread > 0) {
        const std::string scalarName = modelMerged.getModel().getPrecision().getName();
        const std::string maxSharedPerBlockStr = std::to_string(maxSharedMemPerThread * getKernelBlockSize(KernelPresynapticUpdate));
        kernelEnv.add(Type::Void, "_sh_out_post", "shOutPost",
                      {kernelEnv.addInitialiser(getSharedPrefix() +" " + scalarName + " shOutPost[" + maxSharedPerBlockStr + "];")});
    }

    // Shared memory for row length
    kernelEnv.add(Type::Void, "_sh_row_length", "shRowLength",
                  {kernelEnv.addInitialiser(getSharedPrefix() + "unsigned int shRowLength[" + std::to_string(getKernelBlockSize(KernelPresynapticUpdate)) + "];")});

    // Shared memory for spikes and spike events
    kernelEnv.add(Type::Void, "_sh_spk", "shSpk",
                  {kernelEnv.addInitialiser(getSharedPrefix() + "unsigned int shSpk[" + std::to_string(getKernelBlockSize(KernelPresynapticUpdate)) + "];")});
    kernelEnv.add(Type::Void, "_sh_spk_evnt", "shSpkEvnt",
                  {kernelEnv.addInitialiser(getSharedPrefix() + "unsigned int shSpkEvnt[" + std::to_string(getKernelBlockSize(KernelPresynapticUpdate)) + "];")});

    // Parallelise over synapse groups
    idStart = 0;
    genParallelGroup<PresynapticUpdateGroupMerged>(
        kernelEnv, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedPresynapticUpdateGroups,
        [this](const SynapseGroupInternal &sg) { return padKernelSize(getNumPresynapticUpdateThreads(sg, getPreferences()), KernelPresynapticUpdate); },
        [&modelMerged, this](EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg)
        {
            EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(env, sg);

            // Get presynaptic update strategy to use for this synapse group
            const auto *presynapticUpdateStrategy = getPresynapticUpdateStrategy(sg.getArchetype());
            LOGD_BACKEND << "Using '" << typeid(*presynapticUpdateStrategy).name() << "' presynaptic update strategy for merged synapse group '" << sg.getIndex() << "'";

            // Generate index calculation code
            const unsigned int batchSize = modelMerged.getModel().getBatchSize();
            buildStandardEnvironment(groupEnv, batchSize);

            // Generate preamble
            presynapticUpdateStrategy->genPreamble(groupEnv, sg, *this);

            // If spike events should be processed
            if(sg.getArchetype().isSpikeEventRequired()) {
                CodeStream::Scope b(groupEnv.getStream());
                presynapticUpdateStrategy->genUpdate(groupEnv, sg, *this, batchSize, 
                                                     modelMerged.getModel().getDT(), false);
            }

            // If true spikes should be processed
            if(sg.getArchetype().isTrueSpikeRequired()) {
                CodeStream::Scope b(groupEnv.getStream());
                presynapticUpdateStrategy->genUpdate(groupEnv, sg, *this, batchSize, 
                                                     modelMerged.getModel().getDT(), true);
            }

            groupEnv.getStream() << std::endl;

            // Generate pre-amble
            presynapticUpdateStrategy->genPostamble(groupEnv, sg, *this, batchSize);
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genPostsynapticUpdateKernel(EnvironmentExternalBase &env, ModelSpecMerged &modelMerged, 
                                              BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const
{
    EnvironmentExternal kernelEnv(env);

    // Shared memory for column length and spikes
    kernelEnv.add(Type::Void, "_sh_col_length", "shColLength",
                  {kernelEnv.addInitialiser(getSharedPrefix() + "unsigned int shColLength[" + std::to_string(getKernelBlockSize(KernelPostsynapticUpdate)) + "];")});
    kernelEnv.add(Type::Void, "_sh_spk", "shSpk",
                  {kernelEnv.addInitialiser(getSharedPrefix() + "unsigned int shSpk[" + std::to_string(getKernelBlockSize(KernelPostsynapticUpdate)) + "];")});

    // Parallelise over postsynaptic update groups
    idStart = 0;
    genParallelGroup<PostsynapticUpdateGroupMerged>(
        kernelEnv, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedPostsynapticUpdateGroups,
        [this](const SynapseGroupInternal &sg) { return padKernelSize(getNumPostsynapticUpdateThreads(sg), KernelPostsynapticUpdate); },
        [&modelMerged, this](EnvironmentExternalBase &env, PostsynapticUpdateGroupMerged &sg)
        {
            EnvironmentGroupMergedField<PostsynapticUpdateGroupMerged> groupEnv(env, sg);

            // Generate index calculation code
            const unsigned int batchSize = modelMerged.getModel().getBatchSize();
            buildStandardEnvironment(groupEnv, batchSize);

            groupEnv.printLine("const unsigned int numSpikes = $(_trg_spk_cnt)[" + sg.getPostSlot(batchSize) + "];");
            
            groupEnv.getStream() << "const unsigned int numSpikeBlocks = (numSpikes + " << getKernelBlockSize(KernelPostsynapticUpdate) - 1 << ") / " << getKernelBlockSize(KernelPostsynapticUpdate) << ";" << std::endl;
            groupEnv.getStream() << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
            {
                CodeStream::Scope b(groupEnv.getStream());
                groupEnv.getStream() << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << getKernelBlockSize(KernelPostsynapticUpdate) << ") + 1 : " << getKernelBlockSize(KernelPostsynapticUpdate) << ";" << std::endl;

                groupEnv.getStream() << "if (" << getThreadID() << " < numSpikesInBlock)";
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    const std::string index = "(r * " + std::to_string(getKernelBlockSize(KernelPostsynapticUpdate)) + ") + " + getThreadID();
                    groupEnv.printLine("const unsigned int spk = $(_trg_spk)[" + sg.getPostVarIndex(batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, index) + "];");
                    groupEnv.getStream() << "shSpk[" << getThreadID() << "] = spk;" << std::endl;

                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        groupEnv.printLine("$(_sh_col_length)[" + getThreadID() + "] = $(_col_length)[spk];");
                    }
                }

                genSharedMemBarrier(groupEnv.getStream());
                groupEnv.getStream() << "// only work on existing neurons" << std::endl;
                groupEnv.print("if ($(id) < $(_col_stride))");
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    groupEnv.getStream() << "// loop through all incoming spikes for learning" << std::endl;
                    groupEnv.getStream() << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
                    {
                        CodeStream::Scope b(groupEnv.getStream());

                        EnvironmentGroupMergedField<PostsynapticUpdateGroupMerged> synEnv(groupEnv, sg);

                        if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            synEnv.print("if ($(id) < $(_sh_col_length)[j])");
                            synEnv.getStream() << CodeStream::OB(1540);

                            synEnv.add(Type::Uint32.addConst(), "id_syn", "synAddress",
                                       {synEnv.addInitialiser("const unsigned int synAddress = $(_remap)[($(_sh_spk)[j] * $(_col_stride)) + $(id)];")});

                            // **OPTIMIZE** we can do a fast constant divide optimization here
                            synEnv.add(Type::Uint32.addConst(), "id_pre", "idPre",
                                       {synEnv.addInitialiser("const unsigned int idPre = $(id_syn) / $(_row_stride);")});
                        }
                        else {
                            synEnv.add(Type::Uint32.addConst(), "id_syn", "synAddress",
                                       {synEnv.addInitialiser("const unsigned int synAddress = ($(id) * $(num_post)) + $(_sh_spk)[j];")});

                            synEnv.add(Type::Uint32.addConst(), "id_pre", "$(id)");
                        }

                        synEnv.add(Type::Uint32.addConst(), "id_post", "$(_sh_spk)[j]");

                        synEnv.add(Type::AddToPre, "addToPre", getAtomic(modelMerged.getModel().getPrecision()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], $(0))");

                        sg.generateSynapseUpdate(synEnv, batchSize, modelMerged.getModel().getDT());

                        if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            synEnv.getStream() << CodeStream::CB(1540);
                        }
                    }
                }
            }
        }
    );
}
//--------------------------------------------------------------------------
void BackendSIMT::genSynapseDynamicsKernel(EnvironmentExternalBase &env, ModelSpecMerged &modelMerged, 
                                           BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const
{
    // Parallelise over synapse groups whose weight update models have code for synapse dynamics
    idStart = 0;
    genParallelGroup<SynapseDynamicsGroupMerged>(
        env, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedSynapseDynamicsGroups,
        [this](const SynapseGroupInternal &sg) { return padKernelSize(getNumSynapseDynamicsThreads(sg), KernelSynapseDynamicsUpdate); },
        [&modelMerged, this](EnvironmentExternalBase &env, SynapseDynamicsGroupMerged &sg)
        {
            EnvironmentGroupMergedField<SynapseDynamicsGroupMerged> groupEnv(env, sg);

            // Generate index calculation code
            const unsigned int batchSize = modelMerged.getModel().getBatchSize();
            buildStandardEnvironment(groupEnv, batchSize);

            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                groupEnv.print("if ($(id) < ($(num_pre) * $(_row_stride)))");
            }
            else {
                groupEnv.print("if ($(id) < ($(num_pre) * $(num_post)))");
            }
            {
                CodeStream::Scope b(groupEnv.getStream());
                EnvironmentGroupMergedField<SynapseDynamicsGroupMerged> synEnv(groupEnv, sg);

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    // **OPTIMIZE * *we can do a fast constant divide optimization here and use the result to calculate the remainder
                    synEnv.printLine("const unsigned int row = $(id) / $(_row_stride);");
                    synEnv.printLine("const unsigned int col = $(id) % $(_row_stride);");

                    synEnv.add(Type::Uint32.addConst(), "id_pre", "row");
                    synEnv.add(Type::Uint32.addConst(), "id_post", "$(_ind)[$(id)]");

                    synEnv.getStream() << "if(col < " << synEnv["_row_length"] << "[row])";
                    synEnv.getStream() << CodeStream::OB(1);
                }
                else {
                    // **OPTIMIZE** we can do a fast constant divide optimization here and use the result to calculate the remainder
                    synEnv.add(Type::Uint32.addConst(), "id_pre", "idPre",
                               {synEnv.addInitialiser("const unsigned int idPre = ($(id) / $(_row_stride));")});
                    synEnv.add(Type::Uint32.addConst(), "id_post", "idPost",
                               {synEnv.addInitialiser("const unsigned int idPost = ($(id) % $(_row_stride));")});    
                }

                synEnv.add(Type::Uint32.addConst(), "id_syn", "$(id)");

                synEnv.add(Type::AddToPostDenDelay, "addToPostDelay", 
                           getAtomic(modelMerged.getModel().getPrecision()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
                synEnv.add(Type::AddToPost, "addToPost", 
                           getAtomic(modelMerged.getModel().getPrecision()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
                synEnv.add(Type::AddToPre, "addToPre",
                            getAtomic(modelMerged.getModel().getPrecision()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], $(0))");
                
                sg.generateSynapseUpdate(synEnv, batchSize, modelMerged.getModel().getDT());

                if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    synEnv.getStream() << CodeStream::CB(1);
                }
            }
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genCustomUpdateKernel(EnvironmentExternal &env, ModelSpecMerged &modelMerged, 
                                        BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroup, size_t &idStart) const
{
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    genParallelGroup<CustomUpdateGroupMerged>(
        env, modelMerged, memorySpaces, updateGroup, idStart, &ModelSpecMerged::genMergedCustomUpdateGroups,
        [batchSize, this](const CustomUpdateInternal &cu) { return getPaddedNumCustomUpdateThreads(cu, batchSize); },
        [batchSize, this](EnvironmentExternalBase &env, CustomUpdateGroupMerged &cg)
        {
            const size_t blockSize = getKernelBlockSize(KernelCustomUpdate);

            // **YUCK** add an environment with a hidden copy of ID so we 
            // can overwrite ID deeper in here without losing access to original
            EnvironmentExternal idEnv(env);
            idEnv.add(Type::Uint32.addConst(), "_id", "$(id)");

            EnvironmentGroupMergedField<CustomUpdateGroupMerged> groupEnv(idEnv, cg);
            buildSizeEnvironment(groupEnv);

            // If update is a batch reduction
            if(cg.getArchetype().isBatchReduction()) {
                groupEnv.printLine("// only do this for existing neurons");
                groupEnv.print("if($(id) < $(size))");
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    
                    // Initialise reduction targets
                    const auto reductionTargets = genInitReductionTargets(groupEnv.getStream(), cg, 
                                                                          batchSize, groupEnv["id"]);

                    // Loop through batches
                    // **TODO** this naive approach is good for reduction when there are lots of neurons/synapses but,
                    // if this isn't the case (TF uses a threshold of 4096), we should do something smarter
                    groupEnv.getStream() << "for(unsigned int batch = 0; batch < " << batchSize << "; batch++)";
                    {
                        CodeStream::Scope b(groupEnv.getStream());
                        EnvironmentGroupMergedField<CustomUpdateGroupMerged> batchEnv(groupEnv, cg);
                        batchEnv.add(Type::Uint32.addConst(), "batch", "batch");
                        buildStandardEnvironment(batchEnv, batchSize);

                        // **THINK** it would be great to 'lift' reads of SHARED variables out of this loop
                        cg.generateCustomUpdate(
                            batchEnv, batchSize,
                            [&reductionTargets, this](auto &env, const auto&)
                            {
                                // Loop through reduction targets and generate reduction
                                for(const auto &r : reductionTargets) {
                                    env.printLine(getReductionOperation("_lr" + r.name, "$(" + r.name + ")", r.access, r.type) + ";");
                                }
                            });

                        
                    }

                    // Loop through reduction targets and write reduced value back to memory
                    for(const auto &r : reductionTargets) {
                        groupEnv.printLine("group->" + r.name + "[" + r.index + "] = _lr" + r.name + ";");
                    }
                }
            }
            // Otherwise, if this is a neuron reduction
            else if (cg.getArchetype().isNeuronReduction()) {
                groupEnv.getStream() << "// only do this for existing neurons" << std::endl;
                groupEnv.getStream() << "if(" << env["id"] << " < " << (32 * batchSize) << ")";
                {
                    CodeStream::Scope b(groupEnv.getStream());

                    // Split ID into lane and batch
                    groupEnv.printLine("const unsigned int lane = $(id) % 32;");
                    groupEnv.add(Type::Uint32.addConst(), "batch", "batch",
                                 {groupEnv.addInitialiser("const unsigned int batch = $(id) / 32;")});

                    EnvironmentGroupMergedField<CustomUpdateGroupMerged> batchEnv(groupEnv, cg);
                    buildStandardEnvironment(batchEnv, batchSize);

                    // Initialise reduction targets
                    const auto reductionTargets = genInitReductionTargets(batchEnv.getStream(), cg, batchSize);

                    // Loop through warps of data
                    // **TODO** this approach is good for reductions where there are small numbers of neurons but large batches sizes but,
                    // if this isn't thsizee case (TF uses a threshold of 1024), we should do something smarter
                    batchEnv.print("for(unsigned int idx = lane; idx < $(size); idx += 32)");
                    {
                        CodeStream::Scope b(batchEnv.getStream());

                        // Re-substitute id with loop index
                        batchEnv.add(Type::Uint32.addConst(), "id", "idx");

                        // **THINK** it would be great to 'lift' reads of NEURON_SHARED variables out of this loop
                        cg.generateCustomUpdate(
                            batchEnv, batchSize,
                            [&reductionTargets, this](auto &env, const auto&)
                            {
                                // Loop through reduction targets and generate reduction
                                for(const auto &r : reductionTargets) {
                                    env.printLine(getReductionOperation("_lr" + r.name, "$(" + r.name + ")", r.access, r.type) + ";");
                                }
                            });
                    }

                    // Perform warp reduction into first lane
                    // **YUCK** CUDA-specific
                    for (unsigned int i = 16; i > 0; i /= 2) {
                        for (const auto &r : reductionTargets) {
                            batchEnv.printLine(getReductionOperation("_lr" + r.name, "__shfl_down_sync(0xFFFFFFFF, _lr" + r.name + ", " + std::to_string(i) + ")",
                                                                     r.access, r.type) + ";");
                        }
                    }

                    // In first lane, loop through reduction targets and write reduced value back to memory
                    batchEnv.getStream() << "if(lane == 0)";
                    {
                        CodeStream::Scope b(batchEnv.getStream());
                        for (const auto &r : reductionTargets) {
                            batchEnv.printLine("group->" + r.name + "[" + r.index + "] = _lr" + r.name + ";");
                        }
                    }
                }
            }
            // Otherwise, if this update is per-element
            else if (cg.getArchetype().getDims() & VarAccessDim::ELEMENT) {
                if((cg.getArchetype().getDims() & VarAccessDim::BATCH) && (batchSize > 1)) {
                    // Split ID into intra-batch ID and batch
                    // **TODO** fast-divide style optimisations here
                    const std::string blockSizeStr = std::to_string(blockSize);
                    const size_t paddedSizeInit = groupEnv.addInitialiser("const unsigned int paddedSize = " + blockSizeStr + " * (($(size) + " + blockSizeStr + " - 1) / " + blockSizeStr + ");");
    
                    // Replace id in substitution with intra-batch ID and add batch
                    groupEnv.add(Type::Uint32.addConst(), "id", "bid",
                                 {paddedSizeInit, groupEnv.addInitialiser("const unsigned int bid = $(_id) % paddedSize;")});
                    groupEnv.add(Type::Uint32.addConst(), "batch", "batch",
                                 {paddedSizeInit, groupEnv.addInitialiser("const unsigned int batch = $(_id) / paddedSize;")});
                }
                // Otherwise, just substitute "batch" for 0
                else {
                    groupEnv.add(Type::Uint32.addConst(), "batch", "0");
                }

                EnvironmentGroupMergedField<CustomUpdateGroupMerged> batchEnv(groupEnv, cg);
                buildStandardEnvironment(batchEnv, batchSize);
                
                batchEnv.getStream() << "// only do this for existing neurons" << std::endl;
                batchEnv.print("if($(id) < $(size))");
                {
                    CodeStream::Scope b(batchEnv.getStream());
                    cg.generateCustomUpdate(batchEnv, batchSize, [](auto&, auto&){});
                }
            }
            // Otherwise
            else {
                // Use local ID for batch and always use zero for ID
                groupEnv.add(Type::Uint32.addConst(), "batch", "$(_id)");
                groupEnv.add(Type::Uint32.addConst(), "id", "0");

                groupEnv.getStream() << "// only do this for existing neurons" << std::endl;
                groupEnv.getStream() << "if(" << groupEnv["batch"] << " < " << ((cg.getArchetype().getDims() & VarAccessDim::BATCH) ? batchSize : 1) << ")";
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    EnvironmentGroupMergedField<CustomUpdateGroupMerged> batchEnv(groupEnv, cg);
                    buildStandardEnvironment(batchEnv, batchSize);

                    cg.generateCustomUpdate(batchEnv, batchSize, [](auto&, auto&){});
                }
            }
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genCustomUpdateWUKernel(EnvironmentExternal &env, ModelSpecMerged &modelMerged,
                                          BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroup, size_t &idStart) const
{
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    genParallelGroup<CustomUpdateWUGroupMerged>(
        env, modelMerged, memorySpaces, updateGroup, idStart, &ModelSpecMerged::genMergedCustomUpdateWUGroups,
        [batchSize, this](const CustomUpdateWUInternal &cu) { return getPaddedNumCustomUpdateWUThreads(cu, batchSize); },
        [batchSize, this](EnvironmentExternalBase &env, CustomUpdateWUGroupMerged &cg)
        {
            const SynapseGroupInternal *sg = cg.getArchetype().getSynapseGroup();
            const size_t blockSize = getKernelBlockSize(KernelCustomUpdate);

            // **YUCK** add an environment with a hidden copy of ID so we 
            // can overwrite ID deeper in here without losing access to original
            EnvironmentExternal idEnv(env);
            idEnv.add(Type::Uint32.addConst(), "_id", "$(id)");

            EnvironmentGroupMergedField<CustomUpdateWUGroupMerged> groupEnv(idEnv, cg);
            buildSizeEnvironment(groupEnv);
 
            // If update isn't a batch reduction
            if(!cg.getArchetype().isBatchReduction()) {
                // If it's batched
                if((cg.getArchetype().getDims() & VarAccessDim::BATCH) && (batchSize > 1)) {
                    // Split ID into intra-batch ID and batch
                    // **TODO** fast-divide style optimisations here
                    const std::string blockSizeStr = std::to_string(blockSize);
                    const size_t paddedSizeInit = groupEnv.addInitialiser("const unsigned int paddedSize = " + blockSizeStr + " * (($(_size) + " + blockSizeStr + " - 1) / " + blockSizeStr + ");");
    
                    // Replace id in substitution with intra-batch ID and add batch
                    groupEnv.add(Type::Uint32.addConst(), "id", "bid",
                                 {paddedSizeInit, groupEnv.addInitialiser("const unsigned int bid = $(_id) % paddedSize;")});
                    groupEnv.add(Type::Uint32.addConst(), "batch", "batch",
                                 {paddedSizeInit, groupEnv.addInitialiser("const unsigned int batch = $(_id) / paddedSize;")});
                }
                // Otherwise, just substitute "batch" for 0
                else {
                    groupEnv.add(Type::Uint32.addConst(), "batch", "0");
                }
            }

            // if this isn't a padding thread
            groupEnv.print("if ($(id) < $(_size))");
            {
                CodeStream::Scope b(groupEnv.getStream());
                EnvironmentGroupMergedField<CustomUpdateWUGroupMerged> synEnv(groupEnv, cg);

                if (sg->getMatrixType() & SynapseMatrixWeight::KERNEL) {
                    synEnv.add(Type::Uint32.addConst(), "id_kernel", "$(id)");
                }
                else {
                    if (sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        // **OPTIMIZE * *we can do a fast constant divide optimization here and use the result to calculate the remainder
                        synEnv.printLine("const unsigned int row = $(id) / $(_row_stride);");
                        synEnv.printLine("const unsigned int col = $(id) % $(_row_stride);");

                        synEnv.add(Type::Uint32.addConst(), "id_pre", "row");
                        synEnv.add(Type::Uint32.addConst(), "id_post", "$(_ind)[$(id)]");
                        
                        synEnv.print("if(col < $(_row_length)[row])");
                        synEnv.getStream() << CodeStream::OB(2);
                    }
                    else {
                        // **OPTIMIZE** we can do a fast constant divide optimization here and use the result to calculate the remainder
                        synEnv.add(Type::Uint32.addConst(), "id_pre", "idPre",
                                   {synEnv.addInitialiser("const unsigned int idPre = $(id) / $(_row_stride)")});
                        synEnv.add(Type::Uint32.addConst(), "id_post", "idPost",
                                   {synEnv.addInitialiser("const unsigned int idPost = $(id) % $(_row_stride)")});
                    }
                }

                synEnv.add(Type::Uint32.addConst(), "id_syn", "$(id)");

                // Initialise reduction targets
                const auto reductionTargets = genInitReductionTargets(synEnv.getStream(), cg, 
                                                                      batchSize, synEnv["id_syn"]);

                // If this is a reduction
                if(cg.getArchetype().isBatchReduction()) {
                    // Loop through batches
                    // **TODO** this naive approach is good for reduction when there are lots of neurons/synapses but,
                    // if this isn't the case (TF uses a threshold of 4096), we should do something smarter
                    synEnv.getStream() << "for(unsigned int batch = 0; batch < " << batchSize << "; batch++)";
                    synEnv.getStream() << CodeStream::OB(1);
                    synEnv.add(Type::Uint32.addConst(), "batch", "batch");
                }

                // **NOTE** use scope to force batchEnv to generate all code within loop
                {
                    EnvironmentGroupMergedField<CustomUpdateWUGroupMerged> batchEnv(synEnv, cg);
                    buildStandardEnvironment(batchEnv, batchSize);

                    cg.generateCustomUpdate(
                        batchEnv, batchSize,
                        [&reductionTargets, this](auto &env, auto &cg)
                        {
                            // If this is a reduction
                            if(cg.getArchetype().isBatchReduction()) {
                                // Loop through reduction targets and generate reduction
                                for(const auto &r : reductionTargets) {
                                    env.printLine(getReductionOperation("_lr" + r.name, "$(" + r.name + ")", r.access, r.type) + ";");
                                }
                            }
                        });
                }

                // If this is a reduction
                if(cg.getArchetype().isBatchReduction()) {
                    // End for loop through batches
                    synEnv.getStream() << CodeStream::CB(1);

                    // Loop through reduction targets and write reduced value back to memory
                    for(const auto &r : reductionTargets) {
                        synEnv.printLine("group->" + r.name + "[" + r.index + "] = _lr" + r.name + ";");
                    }
                }

                if (sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    synEnv.getStream() << CodeStream::CB(2);
                }
            }
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genCustomTransposeUpdateWUKernel(EnvironmentExternal &env, ModelSpecMerged &modelMerged,
                                                   BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroup, size_t &idStart) const
{
    // Generate 2D array
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    const size_t blockSize = getKernelBlockSize(KernelCustomTransposeUpdate);
    env.getStream() << getSharedPrefix() << " float shTile[" << blockSize << "][" << (blockSize + 1) << "];" << std::endl;
    genParallelGroup<CustomUpdateTransposeWUGroupMerged>(
        env, modelMerged, memorySpaces, updateGroup, idStart, &ModelSpecMerged::genMergedCustomUpdateTransposeWUGroups,
        [batchSize, &modelMerged, this](const CustomUpdateWUInternal &cu) { return getPaddedNumCustomUpdateTransposeWUThreads(cu, batchSize); },
        [batchSize, blockSize, this](EnvironmentExternalBase &env, CustomUpdateTransposeWUGroupMerged &cg)
        {
            EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> groupEnv(env, cg);
            buildSizeEnvironment(groupEnv);

            // To allow these kernels to be batched, we turn 2D grid into wide 1D grid of 2D block so calculate size
            groupEnv.getStream() << "const unsigned int numXBlocks = (" << groupEnv["num_post"] << " + " << (blockSize - 1) << ") / " << blockSize << ";" << std::endl;

            // Calculate what block this kernel starts at (because of kernel merging, it may not start at block 0)
            groupEnv.getStream() << "const unsigned int blockStart = " << groupEnv["_group_start_id"] << " / " << blockSize << ";" << std::endl;

            if((cg.getArchetype().getDims() & VarAccessDim::BATCH) && (batchSize > 1)) {
                // If there's multiple batches we also need to know how many Y blocks and hence total blocks there are
                groupEnv.getStream() << "const unsigned int numYBlocks = (" << groupEnv["num_pre"] << " + " << (blockSize - 1) << ") / " << blockSize << ";" << std::endl;
                groupEnv.getStream() << "const unsigned int numBlocks = numXBlocks * numYBlocks;" << std::endl;

                // Therefore determine block and batch
                groupEnv.getStream() << "const unsigned int batchBlock = " << getBlockID(0) << " - blockStart;" << std::endl;
                groupEnv.getStream() << "const unsigned int block = batchBlock % numBlocks;" << std::endl;
                groupEnv.getStream() << "const unsigned int batch = batchBlock / numBlocks;" << std::endl;

                // Add batch to substitutions
                groupEnv.add(Type::Uint32.addConst(), "batch", "batch");
            }
            // Otherwise, just substitute "batch" for 0
            else {
                groupEnv.getStream() << "const unsigned int block = " << getBlockID(0) << " - blockStart;" << std::endl;
                groupEnv.add(Type::Uint32.addConst(), "batch", "0");
            }

            EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> batchEnv(groupEnv, cg);
            buildStandardEnvironment(batchEnv, batchSize);

            // Add field for transpose field and get its name
            const std::string transposeVarName = cg.addTransposeField(batchEnv);

            // Divide block index into x and y
            // **TODO** fast-divide style optimisations here
            batchEnv.getStream() << "const unsigned int blockX = (block % numXBlocks);" << std::endl;
            batchEnv.getStream() << "const unsigned int blockY = (block / numXBlocks);" << std::endl;

            {
                CodeStream::Scope b(batchEnv.getStream());
                batchEnv.getStream() << "// Calculate coordinate of thread in input matrix" << std::endl;
                batchEnv.getStream() << "const unsigned int x = (blockX * " << blockSize << ") + " << getThreadID(0) << ";" << std::endl;
                batchEnv.getStream() << "const unsigned int y = (blockY * " << blockSize << ") + " << getThreadID(1) << ";" << std::endl;

                batchEnv.printLine("// If thread isn't off the 'right' edge of the input matrix");
                batchEnv.print("if(x < $(num_post))");
                {
                    CodeStream::Scope b(batchEnv.getStream());
                    batchEnv.getStream() << "// Loop through input rows " << std::endl;
                    batchEnv.getStream() << "for (unsigned int j = 0; j < " << blockSize << "; j += 8)";
                    {
                        CodeStream::Scope b(batchEnv.getStream());
                        batchEnv.printLine("// If thread isn't off the 'bottom' edge of the input matrix");
                        batchEnv.print("if((y + j) < $(num_pre))");
                        {
                            CodeStream::Scope b(batchEnv.getStream());
                            EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> synEnv(batchEnv, cg);

                            synEnv.add(Type::Uint32.addConst(), "id_pre", "y");
                            synEnv.add(Type::Uint32.addConst(), "id_post", "x");
                            synEnv.add(Type::Uint32.addConst(), "id_syn", "idx",
                                       {synEnv.addInitialiser("const unsigned int idx = ((y + j) * $(num_post)) + x;")});
                            cg.generateCustomUpdate(
                                synEnv, batchSize,
                                [&transposeVarName, this](auto &env, const auto&)
                                {        
                                    // Write forward weight to shared memory
                                    env.printLine("shTile[" + getThreadID(1) + " + j][" + getThreadID(0) + "] = $(" + transposeVarName + ");");
                                });
                        }
                    }
                }
            }
            genSharedMemBarrier(batchEnv.getStream());
            {
                CodeStream::Scope b(batchEnv.getStream());
                batchEnv.getStream() << "// Calculate (transposed) coordinate of thread in output matrix" << std::endl;
                batchEnv.getStream() << "const unsigned int x = (blockY * " << blockSize << ") + " << getThreadID(0) << ";" << std::endl;
                batchEnv.getStream() << "const unsigned int y = (blockX * " << blockSize << ") + " << getThreadID(1) << ";" << std::endl;

                batchEnv.printLine("// If thread isn't off the 'bottom' edge of the output matrix");
                batchEnv.print("if(x < $(num_pre))");
                {
                    CodeStream::Scope b(batchEnv.getStream());
                    batchEnv.getStream() << "// Loop through output rows" << std::endl;
                    batchEnv.getStream() <<  "for(unsigned int j = 0; j < " << blockSize << "; j += 8)";
                    {
                        CodeStream::Scope b(batchEnv.getStream());
                        batchEnv.printLine("// If thread isn't off the 'right' edge of the output matrix");
                        batchEnv.print("if((y + j) < $(num_post))");
                        {
                            CodeStream::Scope b(batchEnv.getStream());
                            batchEnv.print("$(" + transposeVarName + "_transpose)[");
                            if((cg.getArchetype().getDims() & VarAccessDim::BATCH) && (batchSize > 1)) {
                                batchEnv.print("$(_batch_offset) + ");
                            }
                            batchEnv.printLine("((y + j) * $(num_pre)) + x] = shTile[" + getThreadID(0) + "][" + getThreadID(1) + " + j];");
                        }
                    }
                }
            }
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genCustomConnectivityUpdateKernel(EnvironmentExternalBase &env, ModelSpecMerged &modelMerged,
                                                    BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroup, size_t &idStart) const
{
    // Parallelise across presynaptic neurons
    genParallelGroup<CustomConnectivityUpdateGroupMerged>(
        env, modelMerged, memorySpaces, updateGroup, idStart, &ModelSpecMerged::genMergedCustomConnectivityUpdateGroups,
        [this](const CustomConnectivityUpdateInternal &cg) { return padSize(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(), KernelCustomUpdate); },
        [&modelMerged, this](EnvironmentExternalBase &env, CustomConnectivityUpdateGroupMerged &cg)
        {
            EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> groupEnv(env, cg);
            buildStandardEnvironment(groupEnv);

            groupEnv.getStream() << "// only do this for existing presynaptic neurons" << std::endl;
            groupEnv.print("if($(id) < $(num_pre))");
            {
                CodeStream::Scope b(groupEnv.getStream());

                // Configure substitutions
                groupEnv.add(Type::Uint32.addConst(), "id_pre", "$(id)");
                
                // Add population RNG field
                groupEnv.addField(getPopulationRNGType().createPointer(), "_rng", "rng",
                                  [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "rowRNG"); },
                                  "$(id)");
                // **TODO** for OCL do genPopulationRNGPreamble(os, popSubs, "$(id)") in initialiser


                cg.generateUpdate(*this, groupEnv, modelMerged.getModel().getBatchSize());
                
                // Copy local stream back to local
                /*if(Utils::isRNGRequired(cg.getArchetype().getRowUpdateCodeTokens())) {
                    genPopulationRNGPostamble(groupEnv.getStream(), rng);
                }*/
            }
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genInitializeKernel(EnvironmentExternalBase &env, ModelSpecMerged &modelMerged, 
                                      BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const
{
    env.getStream() << "// ------------------------------------------------------------------------" << std::endl;
    env.getStream() << "// Local neuron groups" << std::endl;
    idStart = 0;
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    genParallelGroup<NeuronInitGroupMerged>(
        env, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedNeuronInitGroups,
        [this](const NeuronGroupInternal &ng) { return padKernelSize(ng.getNumNeurons(), KernelInitialize); },
        [&modelMerged, batchSize, this](EnvironmentExternalBase &env, NeuronInitGroupMerged &ng)
        {
            EnvironmentGroupMergedField<NeuronInitGroupMerged> groupEnv(env, ng);
            buildStandardEnvironment(groupEnv, batchSize);

            groupEnv.getStream() << "// only do this for existing neurons" << std::endl;
            groupEnv.print("if($(id) < $(num_neurons))");
            {
                CodeStream::Scope b(groupEnv.getStream());

                // If population RNGs are initialised on device and this neuron is going to require one, 
                if(isPopulationRNGInitialisedOnDevice() && ng.getArchetype().isSimRNGRequired()) {
                    // Add field for RNG
                    EnvironmentGroupMergedField<NeuronInitGroupMerged> rngInitEnv(groupEnv, ng);
                    rngInitEnv.addField(getPopulationRNGType().createPointer(), "_rng", "rng",
                                        [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "rng"); });

                    // If batch size is 1, initialise single RNG using GLOBAL thread id for sequence
                    if(batchSize == 1) {
                        genPopulationRNGInit(rngInitEnv.getStream(), printSubs("$(_rng)[$(id)]", rngInitEnv), 
                                             "deviceRNGSeed", "id");
                    }
                    // Otherwise, loop through batches and initialise independent RNGs using GLOBAL thread id as basis of sequence
                    else {
                        env.getStream() << "for(unsigned int b = 0; b < " << batchSize << "; b++)";
                        {
                            CodeStream::Scope b(rngInitEnv.getStream());
                            genPopulationRNGInit(rngInitEnv.getStream(), printSubs("$(_rng)[(b * $(num_neurons)) + $(id)]", rngInitEnv), 
                                                 "deviceRNGSeed", "(b * " + std::to_string(getNumInitialisationRNGStreams(modelMerged)) + ") + id");
                        }
                    }
                    
                }

                // If this neuron requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(ng.getArchetype().isInitRNGRequired()) {
                    groupEnv.add(Type::Void, "_rng", genGlobalRNGSkipAhead(groupEnv.getStream(), "id"));
                }

                ng.generateInit(*this, groupEnv, batchSize);
            }
        });
    env.getStream() << std::endl;

    env.getStream() << "// ------------------------------------------------------------------------" << std::endl;
    env.getStream() << "// Synapse groups" << std::endl;
    genParallelGroup<SynapseInitGroupMerged>(
        env, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedSynapseInitGroups,
        [this](const SynapseGroupInternal &sg) { return padKernelSize(getNumInitThreads(sg), KernelInitialize); },
        [batchSize, this](EnvironmentExternalBase &env, SynapseInitGroupMerged &sg)
        {
            EnvironmentGroupMergedField<SynapseInitGroupMerged> groupEnv(env, sg);
            buildStandardEnvironment(groupEnv, batchSize);
            genSynapseVarInit(groupEnv, batchSize, sg, sg.getArchetype().isWUInitRNGRequired(), 
                              (sg.getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL), 
                              sg.getArchetype().getKernelSize().size());
        });
    env.getStream() << std::endl;

    env.getStream() << "// ------------------------------------------------------------------------" << std::endl;
    env.getStream() << "// Custom update groups" << std::endl;
    genParallelGroup<CustomUpdateInitGroupMerged>(
        env, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedCustomUpdateInitGroups,
        [this](const CustomUpdateInternal &cg) { return padKernelSize(cg.getSize(), KernelInitialize); },
        [batchSize, this](EnvironmentExternalBase &env, CustomUpdateInitGroupMerged &cg)
        {
            EnvironmentGroupMergedField<CustomUpdateInitGroupMerged> groupEnv(env, cg);
            buildStandardEnvironment(groupEnv, batchSize);

            groupEnv.getStream() << "// only do this for existing variables" << std::endl;
            groupEnv.print("if($(id) < $(size))");
            {
                CodeStream::Scope b(groupEnv.getStream());

                // If this custom update requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(cg.getArchetype().isInitRNGRequired()) {
                    groupEnv.add(Type::Void, "_rng", genGlobalRNGSkipAhead(groupEnv.getStream(), "id"));
                }

                cg.generateInit(*this, groupEnv, batchSize);
            }
        });
    env.getStream() << std::endl;

    env.getStream() << "// ------------------------------------------------------------------------" << std::endl;
    env.getStream() << "// Custom WU update groups" << std::endl;
    genParallelGroup<CustomWUUpdateInitGroupMerged>(
        env, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedCustomWUUpdateInitGroups,
        [this](const CustomUpdateWUInternal &cg) { return padKernelSize(getNumInitThreads(cg), KernelInitialize); },
        [batchSize, this](EnvironmentExternalBase &env, CustomWUUpdateInitGroupMerged &cg)
        {
            EnvironmentGroupMergedField<CustomWUUpdateInitGroupMerged> groupEnv(env, cg);
            buildStandardEnvironment(groupEnv, batchSize);
            const SynapseGroup *sg = cg.getArchetype().getSynapseGroup();
            genSynapseVarInit(groupEnv, batchSize, cg, cg.getArchetype().isInitRNGRequired(), 
                              (sg->getMatrixType() & SynapseMatrixWeight::KERNEL), sg->getKernelSize().size());
        });
    env.getStream() << std::endl;

    env.getStream() << "// ------------------------------------------------------------------------" << std::endl;
    env.getStream() << "// Custom connectivity presynaptic update groups" << std::endl;
    genParallelGroup<CustomConnectivityUpdatePreInitGroupMerged>(
        env, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedCustomConnectivityUpdatePreInitGroups,
        [this](const CustomConnectivityUpdateInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(), KernelInitialize); },
        [batchSize, this](EnvironmentExternalBase &env, CustomConnectivityUpdatePreInitGroupMerged &cg)
        {
            // Create environment
            EnvironmentGroupMergedField<CustomConnectivityUpdatePreInitGroupMerged> groupEnv(env, cg);
            buildStandardEnvironment(groupEnv);
            
            groupEnv.getStream() << "// only do this for existing variables" << std::endl;
            groupEnv.print("if($(id) < $(size))");
            {
                CodeStream::Scope b(groupEnv.getStream());

                // If population RNGs are initialised on device and this custom connectivity update 
                // required one, initialise single RNG using GLOBAL thread id for sequence
                if(isPopulationRNGInitialisedOnDevice() && Utils::isRNGRequired(cg.getArchetype().getRowUpdateCodeTokens())) {
                    // Add field for RNG
                    EnvironmentGroupMergedField<CustomConnectivityUpdatePreInitGroupMerged> rngInitEnv(groupEnv, cg);
                    rngInitEnv.addField(getPopulationRNGType().createPointer(), "_rng", "rng",
                                        [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "rowRNG"); });

                    genPopulationRNGInit(rngInitEnv.getStream(), printSubs("$(_rng)[$(id)]", rngInitEnv), 
                                         "deviceRNGSeed", "id");
                }

                // If this custom update requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(Utils::isRNGRequired(cg.getArchetype().getPreVarInitialisers())) {
                    groupEnv.add(Type::Void, "_rng", genGlobalRNGSkipAhead(groupEnv.getStream(), "id"));
                }

                cg.generateInit(*this, groupEnv, batchSize);
            }
        });
    env.getStream() << std::endl;

    env.getStream() << "// ------------------------------------------------------------------------" << std::endl;
    env.getStream() << "// Custom connectivity postsynaptic update groups" << std::endl;
    genParallelGroup<CustomConnectivityUpdatePostInitGroupMerged>(
        env, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedCustomConnectivityUpdatePostInitGroups,
        [this](const CustomConnectivityUpdateInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(), KernelInitialize); },
        [batchSize, this](EnvironmentExternalBase &env, CustomConnectivityUpdatePostInitGroupMerged &cg)
        {
            // Create environment
            EnvironmentGroupMergedField<CustomConnectivityUpdatePostInitGroupMerged> groupEnv(env, cg);
            buildStandardEnvironment(groupEnv);

            groupEnv.getStream() << "// only do this for existing variables" << std::endl;
            groupEnv.print("if($(id) < $(size))");
            {
                CodeStream::Scope b(groupEnv.getStream());

                // If this custom update requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(Utils::isRNGRequired(cg.getArchetype().getPostVarInitialisers())) {
                    groupEnv.add(Type::Void, "_rng", genGlobalRNGSkipAhead(groupEnv.getStream(), "id"));
                }

                cg.generateInit(*this, groupEnv, batchSize);
            }
        });
    env.getStream() << std::endl;

    env.getStream() << "// ------------------------------------------------------------------------" << std::endl;
    env.getStream() << "// Synapse groups with sparse connectivity" << std::endl;
    genParallelGroup<SynapseConnectivityInitGroupMerged>(
        env, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedSynapseConnectivityInitGroups,
        [this](const SynapseGroupInternal &sg) { return padKernelSize(getNumConnectivityInitThreads(sg), KernelInitialize); },
        [&modelMerged, this](EnvironmentExternalBase &env, SynapseConnectivityInitGroupMerged &sg)
        {
            EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> groupEnv(env, sg);
            buildStandardEnvironment(groupEnv, modelMerged.getModel().getBatchSize());

            // If there is row-building code in this snippet
            const auto &connectInit = sg.getArchetype().getConnectivityInitialiser();
            if(!Utils::areTokensEmpty(connectInit.getRowBuildCodeTokens())) {
                groupEnv.getStream() << "// only do this for existing presynaptic neurons" << std::endl;
                groupEnv.print("if($(id) < $(num_pre))");

                // Configure substitutions
                groupEnv.add(Type::Uint32.addConst(), "id_pre", "$(id)");
                groupEnv.add(Type::Uint32.addConst(), "id_post_begin", "0");
                groupEnv.add(Type::Uint32.addConst(), "id_thread", "0");
                groupEnv.add(Type::Uint32.addConst(), "num_threads", "1");
            }
            // Otherwise
            else {
                assert(!Utils::areTokensEmpty(connectInit.getColBuildCodeTokens()));

                groupEnv.getStream() << "// only do this for existing postsynaptic neurons" << std::endl;
                groupEnv.print("if($(id) < $(num_post))");

                // Configure substitutions
                groupEnv.add(Type::Uint32.addConst(), "id_post", "$(id)");
                groupEnv.add(Type::Uint32.addConst(), "id_pre_begin", "0");
                groupEnv.add(Type::Uint32.addConst(), "id_thread", "0");
                groupEnv.add(Type::Uint32.addConst(), "num_threads", "1");
            }
            {
                CodeStream::Scope b(groupEnv.getStream());

                // Create environment for generating add synapsecode into seperate CodeStream
                std::ostringstream addSynapseStream;
                CodeStream addSynapse(addSynapseStream);
                {
                    CodeStream::Scope b(addSynapse);
                    EnvironmentExternal addSynapseEnv(groupEnv, addSynapse);

                    // Get postsynaptic/presynaptic index from first addSynapse parameter
                    // **YUCK** we need to do this in an initialiser so the $(0) doesn't get confused with those used in AddToXXXX
                    if(!Utils::areTokensEmpty(connectInit.getRowBuildCodeTokens())) {
                        addSynapseEnv.add(Type::Uint32.addConst(), "id_post", "idPost",
                                            {addSynapseEnv.addInitialiser("const unsigned int idPost = $(0);")});
                    }
                    else {
                        addSynapseEnv.add(Type::Uint32.addConst(), "id_pre", "idPre",
                                            {addSynapseEnv.addInitialiser("const unsigned int idPre = $(0);")});
                    }

                    // If matrix is sparse
                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        // If there is row-building code in this snippet
                        if(!Utils::areTokensEmpty(connectInit.getRowBuildCodeTokens())) {
                            // Calculate index of new synapse
                            addSynapseEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn",
                                              {addSynapseEnv.addInitialiser("const unsigned int idSyn = ($(id_pre) * $(_row_stride)) + $(_row_length)[$(id_pre)];")});

                            // Increment row length
                            addSynapseEnv.printLine("$(_row_length)[$(id_pre)]++;");
                        }
                        // Otherwise
                        else {
                            // Atomically increment row length and get previous value i.e. where our synapse is being inserted
                            // and, from this, calculate index of new synapse
                            addSynapseEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn",
                                              {addSynapseEnv.addInitialiser("const unsigned int prevRowLen = " + getAtomic(Type::Uint32) + "(&$(_row_length)[$(id_pre)], 1);"),
                                               addSynapseEnv.addInitialiser("const unsigned int idSyn = ($(id_pre) * $(_row_stride)) + prevRowLen;")});
                        }

                        // Set index of synapse
                        addSynapseEnv.printLine("$(_ind)[$(id_syn)] = $(id_post);");

                        // If there is a kernel
                        if(!sg.getArchetype().getKernelSize().empty()) {
                            // Create new environment
                            EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> kernelInitEnv(addSynapseEnv, sg);

                            // Replace kernel indices with the subsequent 'function' parameters
                            // **YUCK** these also need doing in initialisers so the $(1) doesn't get confused with those used in addToPostDelay
                            for(size_t i = 0; i < sg.getArchetype().getKernelSize().size(); i++) {
                                const std::string iStr = std::to_string(i);
                                kernelInitEnv.add(Type::Uint32.addConst(), "id_kernel_" + iStr, "idKernel" + iStr,
                                                  {kernelInitEnv.addInitialiser("const unsigned int idKernel" + iStr + " = $(" + std::to_string(i + 1) + ");")});
                            }

                            // Call handler to initialize variables
                            sg.generateKernelInit(kernelInitEnv, 1);
                        }
                    }
                    // Otherwise, if it's bitmask
                    else {
                        assert(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK);
                        assert(sg.getArchetype().getKernelSize().empty()) ;

                        // If there is row-building code in this snippet
                        const auto indexTypeName = getSynapseIndexType(sg).getName();
                        if(!Utils::areTokensEmpty(connectInit.getRowBuildCodeTokens())) {
                            addSynapseEnv.getStream() << "const " << indexTypeName << " rowStartGID = $(id) * (" << indexTypeName << ")($_row_stride);" << std::endl;
                            addSynapseEnv.getStream() << getAtomic(Type::Uint32, AtomicOperation::OR) << "(&$(_gp)[(rowStartGID + ($(0))) / 32], 0x80000000 >> ((rowStartGID + ($(0))) & 31));" << std::endl;
                        }
                        // Otherwise
                        else {
                            addSynapseEnv.getStream() << "const " << indexTypeName << " colStartGID = $(id);" << std::endl;
                            addSynapseEnv.getStream() << getAtomic(Type::Uint32, AtomicOperation::OR) << "(&$(_gp)[(colStartGID + (($(0)) * $(_row_stride))) / 32], 0x80000000 >> ((colStartGID + (($(0)) * $(_row_stride))) & 31));" << std::endl;
                        }
                    }
                }
               
                // Use addSynapseStream to implement addSynapse function
                const auto addSynapseType = Type::ResolvedType::createFunction(Type::Void, std::vector<Type::ResolvedType>{1ull + sg.getArchetype().getKernelSize().size(), Type::Uint32});
                groupEnv.add(addSynapseType, "addSynapse", addSynapseStream.str());

                // If this connectivity requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(connectInit.isRNGRequired()) {
                    groupEnv.add(Type::Void, "_rng", genGlobalRNGSkipAhead(groupEnv.getStream(), "id"));
                }

                // If there is row-building code in this snippet
                if(!Utils::areTokensEmpty(connectInit.getRowBuildCodeTokens())) {
                    // If this is a sparse matrix, zero row length
                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        groupEnv.printLine("$(_row_length)[$(id)] = 0;");
                    }

                    // Call row-based connectivity handler
                    sg.generateSparseRowInit(groupEnv);
                }
                // Otherwise, call column-based connectivity handler
                // **NOTE** in this case, row length gets zeroed by a memset call in backend
                else {
                    sg.generateSparseColumnInit(groupEnv);
                }
            }
        });
    env.getStream() << std::endl;
}
//--------------------------------------------------------------------------
void BackendSIMT::genInitializeSparseKernel(EnvironmentExternalBase &env, ModelSpecMerged &modelMerged,
                                            size_t numInitializeThreads, BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const
{
    EnvironmentExternal envKernel(env);
    envKernel.add(Type::Void, "_sh_row_length", "shRowLength",
                  {envKernel.addInitialiser(getSharedPrefix() + "unsigned int shRowLength[" + std::to_string(getKernelBlockSize(KernelInitializeSparse)) + "];")});
   
    // Initialise weight update variables for synapse groups with sparse connectivity
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    genParallelGroup<SynapseSparseInitGroupMerged>(
        envKernel, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedSynapseSparseInitGroups,
        [this](const SynapseGroupInternal &sg) { return padKernelSize(sg.getMaxConnections(), KernelInitializeSparse); },
        [batchSize, numInitializeThreads, this](EnvironmentExternalBase &env, SynapseSparseInitGroupMerged &sg)
        {
            EnvironmentGroupMergedField<SynapseSparseInitGroupMerged> groupEnv(env, sg);
            buildStandardEnvironment(groupEnv, batchSize);

            // If this post synapse requires an RNG for initialisation,
            // make copy of global phillox RNG and skip ahead by thread id
            // **NOTE** not LOCAL id
            if(sg.getArchetype().isWUInitRNGRequired()) {
                groupEnv.add(Type::Void, "_rng", 
                             genGlobalRNGSkipAhead(groupEnv.getStream(), std::to_string(numInitializeThreads) + " + id"));
            }

            // Generate sparse synapse variable initialisation code
            genSparseSynapseVarInit<SynapseSparseInitGroupMerged>(
                groupEnv, batchSize, sg, sg.getArchetype().isWUVarInitRequired(), 
                [this](EnvironmentExternalBase &env, SynapseSparseInitGroupMerged &sg)
                {
                    // If postsynaptic learning is required
                    if(!Utils::areTokensEmpty(sg.getArchetype().getWUInitialiser().getPostLearnCodeTokens())) {
                        CodeStream::Scope b(env.getStream());

                        // Extract index of synapse's postsynaptic target
                        env.printLine("const unsigned int postIndex = $(_ind)[idx];");

                        // Atomically increment length of column of connectivity associated with this target
                        // **NOTE** this returns previous length i.e. where to insert new entry
                        env.printLine("const unsigned int colLocation = " + getAtomic(Type::Uint32) + "(&$(_col_length)[postIndex], 1);");

                        // From this calculate index into column-major matrix
                        env.printLine("const unsigned int colMajorIndex = (postIndex * $(_col_stride)) + colLocation;");

                        // Add remapping entry at this location poining back to row-major index
                        env.printLine("$(_remap)[colMajorIndex] = idx;");
                    }
                });
        });

    // Initialise weight update variables for synapse groups with sparse connectivity
    genParallelGroup<CustomWUUpdateSparseInitGroupMerged>(
        envKernel, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedCustomWUUpdateSparseInitGroups,
        [this](const CustomUpdateWUInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getMaxConnections(), KernelInitializeSparse); },
        [batchSize, numInitializeThreads, this](EnvironmentExternalBase &env, CustomWUUpdateSparseInitGroupMerged &cg)
        {
            EnvironmentGroupMergedField<CustomWUUpdateSparseInitGroupMerged> groupEnv(env, cg);
            buildStandardEnvironment(groupEnv, batchSize);

            // If this custom update requires an RNG for initialisation,
            // make copy of global phillox RNG and skip ahead by thread id
            // **NOTE** not LOCAL id
            if(cg.getArchetype().isInitRNGRequired()) {
                groupEnv.add(Type::Void, "_rng", 
                             genGlobalRNGSkipAhead(groupEnv.getStream(), std::to_string(numInitializeThreads) + " + id"));
            }

            // Generate sparse synapse variable initialisation code
            genSparseSynapseVarInit<CustomWUUpdateSparseInitGroupMerged>(
                groupEnv, batchSize, cg, true,
                [](EnvironmentExternalBase&, CustomWUUpdateSparseInitGroupMerged&){});
        });

    // Initialise weight update variables for synapse groups with sparse connectivity
    genParallelGroup<CustomConnectivityUpdateSparseInitGroupMerged>(
        envKernel, modelMerged, memorySpaces, idStart, &ModelSpecMerged::genMergedCustomConnectivityUpdateSparseInitGroups,
        [this](const CustomConnectivityUpdateInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getMaxConnections(), KernelInitializeSparse); },
        [batchSize, numInitializeThreads, this](EnvironmentExternalBase &env, CustomConnectivityUpdateSparseInitGroupMerged &cg)
        {
            EnvironmentGroupMergedField<CustomConnectivityUpdateSparseInitGroupMerged> groupEnv(env, cg);
            buildStandardEnvironment(groupEnv);

            // If this custom update requires an RNG for initialisation,
            // make copy of global phillox RNG and skip ahead by thread id
            // **NOTE** not LOCAL id
            if(Utils::isRNGRequired(cg.getArchetype().getVarInitialisers())) {
                groupEnv.add(Type::Void, "_rng", 
                              genGlobalRNGSkipAhead(groupEnv.getStream(), std::to_string(numInitializeThreads) + " + id"));
            }
            
            // Generate sparse synapse variable initialisation code
            genSparseSynapseVarInit<CustomConnectivityUpdateSparseInitGroupMerged>(
                groupEnv, batchSize, cg, true,
                [](EnvironmentExternalBase&, CustomConnectivityUpdateSparseInitGroupMerged&){});
        });
}
//--------------------------------------------------------------------------
size_t BackendSIMT::padKernelSize(size_t size, Kernel kernel) const
{ 
    return padSize(size, getKernelBlockSize(kernel)); 
}
//--------------------------------------------------------------------------
void BackendSIMT::genEmitSpike(EnvironmentExternalBase &env, const std::string &suffix, 
                               bool spikeRequired, bool recordingEnabled) const
{
    if(spikeRequired) {
        env.printLine("const unsigned int spk" + suffix + "_idx = " + getAtomic(Type::Uint32, AtomicOperation::ADD, AtomicMemSpace::SHARED) + "(&$(_sh_spk" + suffix + "_count), 1);");
        env.printLine("$(_sh_spk" + suffix + ")[spk" + suffix + "_idx] = $(id);");
    }
    // If recording is enabled, set bit in recording word
    if(recordingEnabled) {
        if(m_KernelBlockSizes[KernelNeuronUpdate] == 32) {
            env.getStream() << getAtomic(Type::Uint32, AtomicOperation::OR, AtomicMemSpace::SHARED) << "(&shSpk" << suffix << "Record, 1 << " << getThreadID() << ");" << std::endl;
        }
        else {
            env.getStream() << getAtomic(Type::Uint32, AtomicOperation::OR, AtomicMemSpace::SHARED) << "(&shSpk" << suffix << "Record[" << getThreadID() << " / 32], 1 << (" << getThreadID() << " % 32));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void BackendSIMT::genRecordingSharedMemInit(CodeStream &os, const std::string &suffix) const
{
    if(m_KernelBlockSizes[KernelNeuronUpdate] == 32) {
        os << getSharedPrefix() << "uint32_t shSpk" << suffix << "Record;" << std::endl;
        os << "if (" << getThreadID() << " == 0)";
        {
            CodeStream::Scope b(os);
            os << "shSpk" << suffix << "Record = 0;" << std::endl;
        }
    }
    else {
        os << getSharedPrefix() << "uint32_t shSpk" << suffix << "Record[" << m_KernelBlockSizes[KernelNeuronUpdate] / 32 << "];" << std::endl;
        os << "if (" << getThreadID() << " < " << m_KernelBlockSizes[KernelNeuronUpdate] / 32 << ")";
        {
            CodeStream::Scope b(os);
            os << "shSpk" << suffix << "Record[" << getThreadID() << "] = 0;" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void BackendSIMT::genSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const
{
    EnvironmentExternal varEnv(env);

    // **TODO** 64-bit id_syn
    varEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn",
               {varEnv.addInitialiser("const unsigned int idSyn = ($(id_pre) * $(_row_stride)) + $(id);")});
    handler(varEnv);
}
//--------------------------------------------------------------------------
const PresynapticUpdateStrategySIMT::Base *BackendSIMT::getPresynapticUpdateStrategy(const SynapseGroupInternal &sg,
                                                                                     const PreferencesBase &preferences)
{
    // Loop through presynaptic update strategies until we find one that is compatible with this synapse group
    // **NOTE** this is done backwards so that user-registered strategies get first priority
    for(auto s = s_PresynapticUpdateStrategies.rbegin(); s != s_PresynapticUpdateStrategies.rend(); ++s) {
        if((*s)->isCompatible(sg, preferences)) {
            return *s;
        }
    }

    throw std::runtime_error("Unable to find a suitable presynaptic update strategy for synapse group '" + sg.getName() + "'");
    return nullptr;
}
}   // namespace GeNN::CodeGenerator

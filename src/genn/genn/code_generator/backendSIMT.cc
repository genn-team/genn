#include "code_generator/backendSIMT.h"

// Standard C++ includes
#include <algorithm>

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

using namespace CodeGenerator;

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
}

//--------------------------------------------------------------------------
// CodeGenerator::BackendSIMT
//--------------------------------------------------------------------------
namespace CodeGenerator
{
const char *BackendSIMT::KernelNames[KernelMax] = {
    "updateNeuronsKernel",
    "updatePresynapticKernel",
    "updatePostsynapticKernel",
    "updateSynapseDynamicsKernel",
    "initializeKernel",
    "initializeSparseKernel",
    "preNeuronResetKernel",
    "preSynapseResetKernel"};
//--------------------------------------------------------------------------
std::vector<PresynapticUpdateStrategySIMT::Base*> BackendSIMT::s_PresynapticUpdateStrategies = {
    new PresynapticUpdateStrategySIMT::PreSpan,
    new PresynapticUpdateStrategySIMT::PostSpan,
    new PresynapticUpdateStrategySIMT::PreSpanProcedural,
    new PresynapticUpdateStrategySIMT::PostSpanBitmask,
};
//--------------------------------------------------------------------------
size_t BackendSIMT::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    return getPresynapticUpdateStrategy(sg)->getSynapticMatrixRowStride(sg);
}
//--------------------------------------------------------------------------
void BackendSIMT::genPopVariableInit(CodeStream &os, const Substitutions &kernelSubs, Handler handler) const
{
    Substitutions varSubs(&kernelSubs);

    // If this is first thread in group
    os << "if(" << varSubs["id"] << " == 0)";
    {
        CodeStream::Scope b(os);
        handler(os, varSubs);
    }
}
//--------------------------------------------------------------------------
void BackendSIMT::genVariableInit(CodeStream &os, const std::string &, const std::string &countVarName,
                                  const Substitutions &kernelSubs, Handler handler) const
{
    // Variable should already be provided via parallelism
    assert(kernelSubs.hasVarSubstitution(countVarName));

    Substitutions varSubs(&kernelSubs);
    handler(os, varSubs);
}
//--------------------------------------------------------------------------
void BackendSIMT::genSynapseVariableRowInit(CodeStream &os, const SynapseGroupMergedBase &,
                                            const Substitutions &kernelSubs, Handler handler) const
{
    // Pre and postsynaptic ID should already be provided via parallelism
    assert(kernelSubs.hasVarSubstitution("id_pre"));
    assert(kernelSubs.hasVarSubstitution("id_post"));

    Substitutions varSubs(&kernelSubs);
    varSubs.addVarSubstitution("id_syn", "(" + kernelSubs["id_pre"] + " * group->rowStride) + " + kernelSubs["id"]);
    handler(os, varSubs);
}
//--------------------------------------------------------------------------
bool BackendSIMT::isGlobalHostRNGRequired(const ModelSpecMerged &modelMerged) const
{
    // Host RNG is required if any synapse groups require a host initialization RNG
    const ModelSpecInternal &model = modelMerged.getModel();
    return std::any_of(model.getSynapseGroups().cbegin(), model.getSynapseGroups().cend(),
                       [](const ModelSpec::SynapseGroupValueType &s)
                       {
                           return (s.second.isHostInitRNGRequired());
                       });
}
//--------------------------------------------------------------------------
bool BackendSIMT::isGlobalDeviceRNGRequired(const ModelSpecMerged &modelMerged) const
{
    // If any neuron groups require  RNG for initialisation, return true
    // **NOTE** this takes postsynaptic model initialisation into account
    const ModelSpecInternal &model = modelMerged.getModel();
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

    return false;
}
//--------------------------------------------------------------------------
size_t BackendSIMT::getNumInitialisationRNGStreams(const ModelSpecMerged &modelMerged) const
{
    // Calculate total number of threads used for neuron initialisation group
    size_t numInitThreads = getNumMergedGroupThreads(modelMerged.getMergedNeuronInitGroups(),
                                                     [this](const NeuronGroupInternal &ng)
                                                     {
                                                         return padSize(ng.getNumNeurons(), getKernelBlockSize(Kernel::KernelInitialize));
                                                     });


    // Add on total number of threads used for dense synapse initialisation
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedSynapseDenseInitGroups(),
                                               [this](const SynapseGroupInternal &sg)
                                               {
                                                   return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), getKernelBlockSize(Kernel::KernelInitialize));
                                               });

    // Add on total number of threads used for synapse connectivity initialisation
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedSynapseConnectivityInitGroups(),
                                               [this](const SynapseGroupInternal &sg)
                                               {
                                                   return padSize(sg.getSrcNeuronGroup()->getNumNeurons(), getKernelBlockSize(Kernel::KernelInitialize));
                                               });

    // Finally, add on total number of threads used for sparse synapse initialisation
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedSynapseDenseInitGroups(),
                                               [this](const SynapseGroupInternal &sg)
                                               {
                                                   return padSize(sg.getMaxConnections(), getKernelBlockSize(Kernel::KernelInitializeSparse));
                                               });

    return numInitThreads;
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
        return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * sg.getMaxConnections();
    }
    else {
        return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//--------------------------------------------------------------------------
void BackendSIMT::addPresynapticUpdateStrategy(PresynapticUpdateStrategySIMT::Base *strategy)
{
    s_PresynapticUpdateStrategies.push_back(strategy);
}
//--------------------------------------------------------------------------
void BackendSIMT::genPreNeuronResetKernel(CodeStream &os, const Substitutions &kernelSubs, const ModelSpecMerged &modelMerged, size_t &idStart) const
{
    // Loop through local neuron groups
    idStart = 0;
    for(const auto &n : modelMerged.getMergedNeuronSpikeQueueUpdateGroups()) {
        os << "// merged" << n.getIndex() << std::endl;
        
        // If group requires previous spike or spike-like-event times resetting here i.e. each group requires multiple threads
        if(n.getArchetype().isPrevSpikeTimeRequired() || n.getArchetype().isPrevSpikeEventTimeRequired()) {
            // Sum padded sizes of each group within merged group
            const size_t paddedSize = std::accumulate(
                    n.getGroups().cbegin(), n.getGroups().cend(), size_t{0},
                    [this](size_t acc, std::reference_wrapper<const NeuronGroupInternal> g)
                    {
                        return (acc + padSize(g.get().getNumNeurons(), getKernelBlockSize(KernelPreNeuronReset)));
                    });

            // If this is the first  group
            if(idStart == 0) {
                os << "if(id < " << paddedSize << ")";
            }
            else {
                os << "if(id >= " << idStart << " && id < " << idStart + paddedSize << ")";
            }
            {
                CodeStream::Scope b(os);

                // Generate code to find correct group structure for this thread
                Substitutions popSubs(&kernelSubs);
                genGroupMergedSearch(os, popSubs, n, idStart);

                // If neuron group requires delays
                if(n.getArchetype().isDelayRequired()) {
                    os << "const unsigned int lastTimestepDelayOffset = *group->spkQuePtr * group->numNeurons;" << std::endl;

                    if(n.getArchetype().isPrevSpikeTimeRequired()) {
                        // If there is a spike for this thread, set previous spike time to time of last timestep
                        // **NOTE** spkQuePtr is updated below so this already points to last timestep
                        os << "if(" << popSubs["id"] << " < group->spkCnt[*group->spkQuePtr])";
                        {
                            CodeStream::Scope b(os);
                            os << "group->prevST[lastTimestepDelayOffset + group->spk[lastTimestepDelayOffset + " << popSubs["id"] << "]] = " << popSubs["t"] << " - DT;" << std::endl;
                        }
                    }
                    if(n.getArchetype().isPrevSpikeEventTimeRequired()) {
                        // If there is a spike-like-event for this thread, set previous spike-like-event time to time of last timestep
                        // **NOTE** spkQuePtr is updated below so this already points to last timestep
                        os << "if(" << popSubs["id"] << " < group->spkCntEvnt[*group->spkQuePtr])";
                        {
                            CodeStream::Scope b(os);
                            os << "group->prevSET[lastTimestepDelayOffset + group->spkEvnt[lastTimestepDelayOffset + " << popSubs["id"] << "]] = " << popSubs["t"] << " - DT;" << std::endl;
                        }
                    }
                }
                // Otherwise
                else {
                    if(n.getArchetype().isPrevSpikeTimeRequired()) {
                        // If there is a spike for this thread, set previous spike time to time of last timestep
                        os << "if(" << popSubs["id"] << " < group->spkCnt[0])";
                        {
                            CodeStream::Scope b(os);
                            os << "group->prevST[group->spk[" << popSubs["id"] << "]] = " << popSubs["t"] << " - DT;" << std::endl;
                        }
                    }
                    if(n.getArchetype().isPrevSpikeEventTimeRequired()) {
                        // If there is a spike-like-event for this thread, set previous spike-like-event time to time of last timestep
                        os << "if(" << popSubs["id"] << " < group->spkCntEvnt[0])";
                        {
                            CodeStream::Scope b(os);
                            os << "group->prevSET[group->spkEvnt[" << popSubs["id"] << "]] = " << popSubs["t"] << " - DT;" << std::endl;
                        }
                    }
                }
                os << std::endl;

                // Use first thread to update spike queue pointer and reset spike counts
                os << "if(" << popSubs["id"] << " == 0)";
                {
                    CodeStream::Scope b(os);

                    if(n.getArchetype().isDelayRequired()) { // with delay
                        os << "*group->spkQuePtr  = (*group->spkQuePtr + 1) % " << n.getArchetype().getNumDelaySlots() << ";" << std::endl;
                    }
                    n.genMergedGroupSpikeCountReset(os);
                }
            }
            idStart += paddedSize;
        } 
        // Otherwise
        else {
            if(idStart == 0) {
                os << "if(id < " << n.getGroups().size() << ")";
            }
            else {
                os << "if(id >= " << idStart << " && id < " << idStart + n.getGroups().size() << ")";
            }
            {
                CodeStream::Scope b(os);

                // Use this to get reference to merged group structure
                os << getPointerPrefix() << "struct MergedNeuronSpikeQueueUpdateGroup" << n.getIndex() << " *group = &d_mergedNeuronSpikeQueueUpdateGroup" << n.getIndex() << "[id - " << idStart << "]; " << std::endl;

                if(n.getArchetype().isDelayRequired()) { // with delay
                    os << "*group->spkQuePtr  = (*group->spkQuePtr + 1) % " << n.getArchetype().getNumDelaySlots() << ";" << std::endl;
                }
                n.genMergedGroupSpikeCountReset(os);
            }
            idStart += n.getGroups().size();
        }
    }
}
//--------------------------------------------------------------------------
void BackendSIMT::genNeuronUpdateKernel(CodeStream &os, const Substitutions &kernelSubs, const ModelSpecMerged &modelMerged,
                                        NeuronGroupSimHandler simHandler, NeuronUpdateGroupMergedHandler wuVarUpdateHandler, size_t &idStart) const
{
    // If any neuron groups emit spike events
    if(std::any_of(modelMerged.getMergedNeuronUpdateGroups().cbegin(), modelMerged.getMergedNeuronUpdateGroups().cend(),
                   [](const NeuronUpdateGroupMerged &n) { return n.getArchetype().isSpikeEventRequired(); }))
    {
        os << getSharedPrefix() << "unsigned int shSpkEvnt[" << getKernelBlockSize(KernelNeuronUpdate) << "];" << std::endl;
        os << getSharedPrefix() << "unsigned int shPosSpkEvnt;" << std::endl;
        os << getSharedPrefix() << "unsigned int shSpkEvntCount;" << std::endl;
        os << std::endl;
        os << "if (" << getThreadID() << " == 1)";
        {
            CodeStream::Scope b(os);
            os << "shSpkEvntCount = 0;" << std::endl;
        }
        os << std::endl;
    }

    // If any neuron groups emit true spikes
    if(std::any_of(modelMerged.getMergedNeuronUpdateGroups().cbegin(), modelMerged.getMergedNeuronUpdateGroups().cend(),
                   [](const NeuronUpdateGroupMerged &n) { return !n.getArchetype().getNeuronModel()->getThresholdConditionCode().empty(); }))
    {
        os << getSharedPrefix() << "unsigned int shSpk[" << getKernelBlockSize(KernelNeuronUpdate) << "];" << std::endl;
        os << getSharedPrefix() << "unsigned int shPosSpk;" << std::endl;
        os << getSharedPrefix() << "unsigned int shSpkCount;" << std::endl;
        os << "if (" << getThreadID() << " == 0)";
        {
            CodeStream::Scope b(os);
            os << "shSpkCount = 0;" << std::endl;
        }
        os << std::endl;
    }

    // If any neuron groups record spikes
    if(std::any_of(modelMerged.getMergedNeuronUpdateGroups().cbegin(), modelMerged.getMergedNeuronUpdateGroups().cend(),
                   [](const NeuronUpdateGroupMerged &n) { return n.getArchetype().isSpikeRecordingEnabled(); }))
    {
        genRecordingSharedMemInit(os, "");
    }

    // If any neuron groups record spike-like events
    if(std::any_of(modelMerged.getMergedNeuronUpdateGroups().cbegin(), modelMerged.getMergedNeuronUpdateGroups().cend(),
                   [](const NeuronUpdateGroupMerged &n) { return n.getArchetype().isSpikeEventRecordingEnabled(); }))
    {
        genRecordingSharedMemInit(os, "Evnt");
    }

    genSharedMemBarrier(os);

    // Parallelise over neuron groups
    idStart = 0;
    genParallelGroup<NeuronUpdateGroupMerged>(
        os, kernelSubs, modelMerged.getMergedNeuronUpdateGroups(), idStart,
        [this](const NeuronGroupInternal &ng) { return padSize(ng.getNumNeurons(), getKernelBlockSize(KernelNeuronUpdate)); },
        [simHandler, wuVarUpdateHandler, this](CodeStream &os, const NeuronUpdateGroupMerged &ng, Substitutions &popSubs)
        {
            // If axonal delays are required
            if(ng.getArchetype().isDelayRequired()) {
                // We should READ from delay slot before spkQuePtr
                os << "const unsigned int readDelayOffset = " << ng.getPrevQueueOffset() << ";" << std::endl;

                // And we should WRITE to delay slot pointed to be spkQuePtr
                os << "const unsigned int writeDelayOffset = " << ng.getCurrentQueueOffset() << ";" << std::endl;
            }
            os << std::endl;

            // Call handler to generate generic neuron code
            os << "if(" << popSubs["id"] << " < group->numNeurons)";
            {
                CodeStream::Scope b(os);

                // Copy global RNG stream to local and use pointer to this for rng
                if(ng.getArchetype().isSimRNGRequired()) {
                    genPopulationRNGPreamble(os, popSubs, "group->rng[" + popSubs["id"] + "]");
                }

                simHandler(os, ng, popSubs,
                           // Emit true spikes
                           [this](CodeStream &neuronUpdateKernelsBody, const NeuronUpdateGroupMerged &ng, Substitutions &subs)
                           {
                               genEmitSpike(neuronUpdateKernelsBody, subs, "", ng.getArchetype().isSpikeRecordingEnabled());
                           },
                           // Emit spike-like events
                           [this](CodeStream &neuronUpdateKernelsBody, const NeuronUpdateGroupMerged &ng, Substitutions &subs)
                           {
                               genEmitSpike(neuronUpdateKernelsBody, subs, "Evnt", ng.getArchetype().isSpikeEventRecordingEnabled());
                           });

                // Copy local stream back to local
                if(ng.getArchetype().isSimRNGRequired()) {
                    genPopulationRNGPostamble(os, "group->rng[" + popSubs["id"] + "]");
                }
            }

            genSharedMemBarrier(os);

            if(ng.getArchetype().isSpikeEventRequired()) {
                os << "if (" << getThreadID() << " == 1)";
                {
                    CodeStream::Scope b(os);
                    os << "if (shSpkEvntCount > 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "shPosSpkEvnt = " << getAtomic("unsigned int") << "(&group->spkCntEvnt";
                        if(ng.getArchetype().isDelayRequired()) {
                            os << "[*group->spkQuePtr], shSpkEvntCount);" << std::endl;
                        }
                        else {
                            os << "[0], shSpkEvntCount);" << std::endl;
                        }
                    }
                } 
                genSharedMemBarrier(os);
            }

            if(!ng.getArchetype().getNeuronModel()->getThresholdConditionCode().empty()) {
                os << "if(" << getThreadID() << " == 0)";
                {
                    CodeStream::Scope b(os);
                    os << "if (shSpkCount > 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "shPosSpk = " << getAtomic("unsigned int") << "(&group->spkCnt";
                        if(ng.getArchetype().isDelayRequired() && ng.getArchetype().isTrueSpikeRequired()) {
                            os << "[*group->spkQuePtr], shSpkCount);" << std::endl;
                        }
                        else {
                            os << "[0], shSpkCount);" << std::endl;
                        }
                    }
                } 
                genSharedMemBarrier(os);
            }

            const std::string queueOffset = ng.getArchetype().isDelayRequired() ? "writeDelayOffset + " : "";
            if(ng.getArchetype().isSpikeEventRequired()) {
                os << "if(" << getThreadID() << " < shSpkEvntCount)";
                {
                    CodeStream::Scope b(os);
                    os << "const unsigned int n = shSpkEvnt[" << getThreadID() << "];" << std::endl;

                    os << "group->spkEvnt[" << queueOffset << "shPosSpkEvnt + " << getThreadID() << "] = n;" << std::endl;
                    if(ng.getArchetype().isSpikeEventTimeRequired()) {
                        os << "group->sleT[" << queueOffset << "n] = t;" << std::endl;
                    }
                }
            }

            if(!ng.getArchetype().getNeuronModel()->getThresholdConditionCode().empty()) {
                const std::string queueOffsetTrueSpk = ng.getArchetype().isTrueSpikeRequired() ? queueOffset : "";

                os << "if(" << getThreadID() << " < shSpkCount)";
                {
                    CodeStream::Scope b(os);

                    os << "const unsigned int n = shSpk[" << getThreadID() << "];" << std::endl;

                    // Create new substition stack and explicitly replace id with 'n' and perform WU var update
                    Substitutions wuSubs(&popSubs);
                    wuSubs.addVarSubstitution("id", "n", true);
                    wuVarUpdateHandler(os, ng, wuSubs);

                    os << "group->spk[" << queueOffsetTrueSpk << "shPosSpk + " << getThreadID() << "] = n;" << std::endl;
                    if(ng.getArchetype().isSpikeTimeRequired()) {
                        os << "group->sT[" << queueOffset << "n] = t;" << std::endl;
                    }
                }
            }

            // If we're recording spikes or spike-like events, use enough threads to copy this block's recording words
            if(ng.getArchetype().isSpikeRecordingEnabled() || ng.getArchetype().isSpikeEventRecordingEnabled()) {
                if(m_KernelBlockSizes[KernelNeuronUpdate] == 32) {
                    os << "if(" << getThreadID() << " == 0)";
                }
                else {
                    os << "if(" << getThreadID() << " < " << m_KernelBlockSizes[KernelNeuronUpdate] / 32 << ")";
                }
                {
                    CodeStream::Scope b(os);

                    // Calculate number of words which will be used to record this population's spikes
                    os << "const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;" << std::endl;

                    // Build global index
                    const std::string globalIndex = "(recordingTimestep * numRecordingWords) + (" + popSubs["id"] + " / 32) + " + getThreadID();

                    // If we are recording spikes, copy word to correct location in global memory
                    if(ng.getArchetype().isSpikeRecordingEnabled()) {
                        os << "group->recordSpk[" << globalIndex << "] = shSpkRecord";
                        if(m_KernelBlockSizes[KernelNeuronUpdate] != 32) {
                            os << "[" << getThreadID() << "]";
                        }
                        os << ";" << std::endl;
                    }

                    // If we are recording spike-like events, copy word to correct location in global memory
                    if(ng.getArchetype().isSpikeEventRecordingEnabled()) {
                        os << "group->recordSpkEvent[" << globalIndex << "] = shSpkEvntRecord";
                        if(m_KernelBlockSizes[KernelNeuronUpdate] != 32) {
                            os << "[" << getThreadID() << "]";
                        }
                        os << ";" << std::endl;
                    }
                }
            }
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genPreSynapseResetKernel(CodeStream &os, const ModelSpecMerged &modelMerged, size_t &idStart) const
{
    // Loop through merged synapse groups
    idStart = 0;
    for(const auto &n : modelMerged.getMergedSynapseDendriticDelayUpdateGroups()) {
        os << "// merged" << n.getIndex() << std::endl;
        if(idStart == 0) {
            os << "if(id < " << n.getGroups().size() << ")";
        }
        else {
            os << "if(id >= " << idStart << " && id < " << idStart + n.getGroups().size() << ")";
        }
        {
            CodeStream::Scope b(os);

            // Use this to get reference to merged group structure
            os << getPointerPrefix() << "struct MergedSynapseDendriticDelayUpdateGroup" << n.getIndex() << " *group = &d_mergedSynapseDendriticDelayUpdateGroup" << n.getIndex() << "[id - " << idStart << "]; " << std::endl;

            os << "*group->denDelayPtr = (*group->denDelayPtr + 1) % " << n.getArchetype().getMaxDendriticDelayTimesteps() << ";" << std::endl;
        }
        idStart += n.getGroups().size();
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void BackendSIMT::genPresynapticUpdateKernel(CodeStream &os, const Substitutions &kernelSubs, const ModelSpecMerged &modelMerged,
                                             PresynapticUpdateGroupMergedHandler wumThreshHandler, PresynapticUpdateGroupMergedHandler wumSimHandler,
                                             PresynapticUpdateGroupMergedHandler wumEventHandler, PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler, size_t &idStart) const
{
    // We need shLg if any synapse groups accumulate into shared memory
    // Determine the maximum shared memory outputs 
    size_t maxSharedMemPerThread = 0;
    for(const auto &s : modelMerged.getMergedPresynapticUpdateGroups()) {
        maxSharedMemPerThread = std::max(maxSharedMemPerThread,
                                         getPresynapticUpdateStrategy(s.getArchetype())->getSharedMemoryPerThread(s, *this));
    }

    // If any shared memory is required, declare array
    if(maxSharedMemPerThread > 0) {
        os << getSharedPrefix() << modelMerged.getModel().getPrecision() << " shLg[" << maxSharedMemPerThread * getKernelBlockSize(KernelPresynapticUpdate) << "];" << std::endl;
    }

    // If any of these synapse groups also have sparse connectivity, allocate shared memory for row length
    if(std::any_of(modelMerged.getMergedPresynapticUpdateGroups().cbegin(), modelMerged.getMergedPresynapticUpdateGroups().cend(),
                   [](const PresynapticUpdateGroupMerged &sg)
                   {
                       return (sg.getArchetype().getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC
                               && (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE));
                   }))
    {
        os << getSharedPrefix() << "unsigned int shRowLength[" << getKernelBlockSize(KernelPresynapticUpdate) << "];" << std::endl;
    }

    if(std::any_of(modelMerged.getMergedPresynapticUpdateGroups().cbegin(), modelMerged.getMergedPresynapticUpdateGroups().cend(),
                   [](const PresynapticUpdateGroupMerged &sg)
                   {
                       return (sg.getArchetype().isTrueSpikeRequired() || !sg.getArchetype().getWUModel()->getLearnPostCode().empty());
                   }))
    {
        os << getSharedPrefix() << "unsigned int shSpk[" << getKernelBlockSize(KernelPresynapticUpdate) << "];" << std::endl;
    }

    if(std::any_of(modelMerged.getMergedPresynapticUpdateGroups().cbegin(), modelMerged.getMergedPresynapticUpdateGroups().cend(),
                   [](const PresynapticUpdateGroupMerged &sg) { return (sg.getArchetype().isSpikeEventRequired()); }))
    {
        os << getSharedPrefix() << "unsigned int shSpkEvnt[" << getKernelBlockSize(KernelPresynapticUpdate) << "];" << std::endl;
    }

    // Parallelise over synapse groups
    idStart = 0;
    genParallelGroup<PresynapticUpdateGroupMerged>(
        os, kernelSubs, modelMerged.getMergedPresynapticUpdateGroups(), idStart,
        [this](const SynapseGroupInternal &sg) { return padSize(getNumPresynapticUpdateThreads(sg, getPreferences()), getKernelBlockSize(KernelPresynapticUpdate)); },
        [wumThreshHandler, wumSimHandler, wumEventHandler, wumProceduralConnectHandler, &modelMerged, this](CodeStream &os, const PresynapticUpdateGroupMerged &sg, const Substitutions &popSubs)
        {
            // Get presynaptic update strategy to use for this synapse group
            const auto *presynapticUpdateStrategy = getPresynapticUpdateStrategy(sg.getArchetype());
            LOGD_BACKEND << "Using '" << typeid(*presynapticUpdateStrategy).name() << "' presynaptic update strategy for merged synapse group '" << sg.getIndex() << "'";

            // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
            if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                os << "const unsigned int preReadDelaySlot = " << sg.getPresynapticAxonalDelaySlot() << ";" << std::endl;
                os << "const unsigned int preReadDelayOffset = preReadDelaySlot * group->numSrcNeurons;" << std::endl;

                if(sg.getArchetype().getWUModel()->isPrevPreSpikeTimeRequired() || sg.getArchetype().getWUModel()->isPrevPreSpikeEventTimeRequired()) {
                    os << "const unsigned int prevPreSpikeTimeReadDelayOffset = " << sg.getPrevPresynapticSpikeTimeAxonalDelaySlot() << " * group->numSrcNeurons;" << std::endl;
                }
            }

            // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
            if(sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                os << "const unsigned int postReadDelayOffset = " << sg.getPostsynapticBackPropDelaySlot() << " * group->numTrgNeurons;" << std::endl;

                if(sg.getArchetype().getWUModel()->isPrevPostSpikeTimeRequired()) {
                    os << "const unsigned int prevPostSpikeTimeReadDelayOffset = " << sg.getPrevPostsynapticSpikeTimeBackPropDelaySlot() << " * group->numTrgNeurons;" << std::endl;
                }
            }

            // Generate preamble
            presynapticUpdateStrategy->genPreamble(os, modelMerged, sg, popSubs, *this);

            // If spike events should be processed
            if(sg.getArchetype().isSpikeEventRequired()) {
                CodeStream::Scope b(os);
                presynapticUpdateStrategy->genUpdate(os, modelMerged, sg, popSubs, *this, false,
                                                     wumThreshHandler, wumEventHandler, wumProceduralConnectHandler);
            }

            // If true spikes should be processed
            if(sg.getArchetype().isTrueSpikeRequired()) {
                CodeStream::Scope b(os);
                presynapticUpdateStrategy->genUpdate(os, modelMerged, sg, popSubs, *this, true,
                                                     wumThreshHandler, wumSimHandler, wumProceduralConnectHandler);
            }

            os << std::endl;

            // Generate pre-amble
            presynapticUpdateStrategy->genPostamble(os, modelMerged, sg, popSubs, *this);
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genPostsynapticUpdateKernel(CodeStream &os, const Substitutions &kernelSubs, const ModelSpecMerged &modelMerged,
                                              PostsynapticUpdateGroupMergedHandler postLearnHandler, size_t &idStart) const
{
    os << getSharedPrefix() << "unsigned int shSpk[" << getKernelBlockSize(KernelPostsynapticUpdate) << "];" << std::endl;
    if(std::any_of(modelMerged.getModel().getSynapseGroups().cbegin(), modelMerged.getModel().getSynapseGroups().cend(),
                   [](const ModelSpec::SynapseGroupValueType &s)
                   {
                       return ((s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getLearnPostCode().empty());
                   }))
    {
        os << getSharedPrefix() << "unsigned int shColLength[" << getKernelBlockSize(KernelPostsynapticUpdate) << "];" << std::endl;
    }

    // Parallelise over postsynaptic update groups
    idStart = 0;
    genParallelGroup<PostsynapticUpdateGroupMerged>(os, kernelSubs, modelMerged.getMergedPostsynapticUpdateGroups(), idStart,
        [this](const SynapseGroupInternal &sg) { return padSize(getNumPostsynapticUpdateThreads(sg), getKernelBlockSize(KernelPostsynapticUpdate)); },
        [postLearnHandler, this](CodeStream &os, const PostsynapticUpdateGroupMerged &sg, Substitutions &popSubs)
        {
            // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
            if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                os << "const unsigned int preReadDelayOffset = " << sg.getPresynapticAxonalDelaySlot() << " * group->numSrcNeurons;" << std::endl;

                if(sg.getArchetype().getWUModel()->isPrevPreSpikeTimeRequired() || sg.getArchetype().getWUModel()->isPrevPreSpikeEventTimeRequired()) {
                    os << "const unsigned int prevPreSpikeTimeReadDelayOffset = " << sg.getPrevPresynapticSpikeTimeAxonalDelaySlot() << " * group->numSrcNeurons;" << std::endl;
                }
            }

            // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
            if(sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                os << "const unsigned int postReadDelaySlot = " << sg.getPostsynapticBackPropDelaySlot() << ";" << std::endl;
                os << "const unsigned int postReadDelayOffset = postReadDelaySlot * group->numTrgNeurons;" << std::endl;

                if(sg.getArchetype().getWUModel()->isPrevPostSpikeTimeRequired()) {
                    os << "const unsigned int prevPostSpikeTimeReadDelayOffset = " << sg.getPrevPostsynapticSpikeTimeBackPropDelaySlot() << " * group->numTrgNeurons;" << std::endl;
                }
            }

            if (sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                os << "const unsigned int numSpikes = group->trgSpkCnt[postReadDelaySlot];" << std::endl;
            }
            else {
                os << "const unsigned int numSpikes = group->trgSpkCnt[0];" << std::endl;
            }

            os << "const unsigned int numSpikeBlocks = (numSpikes + " << getKernelBlockSize(KernelPostsynapticUpdate) - 1 << ") / " << getKernelBlockSize(KernelPostsynapticUpdate) << ";" << std::endl;
            os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
            {
                CodeStream::Scope b(os);
                os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << getKernelBlockSize(KernelPostsynapticUpdate) << ") + 1 : " << getKernelBlockSize(KernelPostsynapticUpdate) << ";" << std::endl;

                os << "if (" << getThreadID() << " < numSpikesInBlock)";
                {
                    CodeStream::Scope b(os);
                    const std::string offsetTrueSpkPost = (sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) ? "postReadDelayOffset + " : "";
                    os << "const unsigned int spk = group->trgSpk[" << offsetTrueSpkPost << "(r * " << getKernelBlockSize(KernelPostsynapticUpdate) << ") + " << getThreadID() << "];" << std::endl;
                    os << "shSpk[" << getThreadID() << "] = spk;" << std::endl;

                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        os << "shColLength[" << getThreadID() << "] = group->colLength[spk];" << std::endl;
                    }
                }

                genSharedMemBarrier(os);
                os << "// only work on existing neurons" << std::endl;
                os << "if (" << popSubs["id"] << " < group->colStride)";
                {
                    CodeStream::Scope b(os);
                    os << "// loop through all incoming spikes for learning" << std::endl;
                    os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
                    {
                        CodeStream::Scope b(os);

                        Substitutions synSubs(&popSubs);
                        if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            os << "if (" << synSubs["id"] << " < shColLength[j])" << CodeStream::OB(1540);
                            os << "const unsigned int synAddress = group->remap[(shSpk[j] * group->colStride) + " << popSubs["id"] << "];" << std::endl;

                            // **OPTIMIZE** we can do a fast constant divide optimization here
                            os << "const unsigned int ipre = synAddress / group->rowStride;" << std::endl;
                            synSubs.addVarSubstitution("id_pre", "ipre");
                        }
                        else {
                            os << "const unsigned int synAddress = (" << synSubs["id"] << " * group->numTrgNeurons) + shSpk[j];" << std::endl;
                            synSubs.addVarSubstitution("id_pre", synSubs["id"]);
                        }

                        synSubs.addVarSubstitution("id_post", "shSpk[j]");
                        synSubs.addVarSubstitution("id_syn", "synAddress");

                        postLearnHandler(os, sg, synSubs);

                        if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            os << CodeStream::CB(1540);
                        }
                    }
                }
            }
        }
    );
}
//--------------------------------------------------------------------------
void BackendSIMT::genSynapseDynamicsKernel(CodeStream &os, const Substitutions &kernelSubs, const ModelSpecMerged &modelMerged,
                                           SynapseDynamicsGroupMergedHandler synapseDynamicsHandler, size_t &idStart) const
{
    // Parallelise over synapse groups whose weight update models have code for synapse dynamics
    idStart = 0;
    genParallelGroup<SynapseDynamicsGroupMerged>(
        os, kernelSubs, modelMerged.getMergedSynapseDynamicsGroups(), idStart,
        [this](const SynapseGroupInternal &sg) { return padSize(getNumSynapseDynamicsThreads(sg), getKernelBlockSize(KernelSynapseDynamicsUpdate)); },
        [synapseDynamicsHandler, &modelMerged, this](CodeStream &os, const SynapseDynamicsGroupMerged &sg, Substitutions &popSubs)
        {
            // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
            if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                os << "const unsigned int preReadDelayOffset = " << sg.getPresynapticAxonalDelaySlot() << " * group->numSrcNeurons;" << std::endl;

                if(sg.getArchetype().getWUModel()->isPrevPreSpikeTimeRequired()) {
                    os << "const unsigned int prevPreSpikeTimeReadDelayOffset = " << sg.getPrevPresynapticSpikeTimeAxonalDelaySlot() << " * group->numSrcNeurons;" << std::endl;
                }
            }

            // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
            if(sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                os << "const unsigned int postReadDelayOffset = " << sg.getPostsynapticBackPropDelaySlot() << " * group->numTrgNeurons;" << std::endl;

                if(sg.getArchetype().getWUModel()->isPrevPostSpikeTimeRequired()) {
                    os << "const unsigned int prevPostSpikeTimeReadDelayOffset = " << sg.getPrevPostsynapticSpikeTimeBackPropDelaySlot() << " * group->numTrgNeurons;" << std::endl;
                }
            }

            Substitutions synSubs(&popSubs);

            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                os << "if (" << popSubs["id"] << " < group->synRemap[0])";
            }
            else {
                os << "if (" << popSubs["id"] << " < (group->numSrcNeurons * group->numTrgNeurons))";
            }
            {
                CodeStream::Scope b(os);

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    // Determine synapse and presynaptic indices for this thread
                    os << "const unsigned int s = group->synRemap[1 + " << popSubs["id"] << "];" << std::endl;

                    synSubs.addVarSubstitution("id_pre", "(s / group->rowStride)");
                    synSubs.addVarSubstitution("id_post", "group->ind[s]");
                    synSubs.addVarSubstitution("id_syn", "s");
                }
                else {
                    // **OPTIMIZE** we can do a fast constant divide optimization here and use the result to calculate the remainder
                    synSubs.addVarSubstitution("id_pre", "(" + popSubs["id"] + " / group->rowStride)");
                    synSubs.addVarSubstitution("id_post", "(" + popSubs["id"] + " % group->rowStride)");
                    synSubs.addVarSubstitution("id_syn", popSubs["id"]);
                }

                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                if(sg.getArchetype().isDendriticDelayRequired()) {
                    synSubs.addFuncSubstitution("addToInSynDelay", 2, getAtomic(modelMerged.getModel().getPrecision()) + "(&group->denDelay[" + sg.getDendriticDelayOffset("$(1)") + synSubs["id_post"] + "], $(0))");
                }
                // Otherwise
                else {
                    synSubs.addFuncSubstitution("addToInSyn", 1, getAtomic(modelMerged.getModel().getPrecision()) + "(&group->inSyn[" + synSubs["id_post"] + "], $(0))");
                }

                synapseDynamicsHandler(os, sg, synSubs);
            }
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::genInitializeKernel(CodeStream &os, const Substitutions &kernelSubs, const ModelSpecMerged &modelMerged,
                                      NeuronInitGroupMergedHandler neuronInitHandler, SynapseDenseInitGroupMergedHandler synapseDenseInitHandler,
                                      SynapseConnectivityInitMergedGroupHandler synapseConnectivityInitHandler, 
                                      SynapseConnectivityInitMergedGroupHandler sgKernelInitHandler, size_t &idStart) const
{
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Local neuron groups" << std::endl;
    idStart = 0;
    genParallelGroup<NeuronInitGroupMerged>(
        os, kernelSubs, modelMerged.getMergedNeuronInitGroups(), idStart,
        [this](const NeuronGroupInternal &ng) { return padSize(ng.getNumNeurons(), getKernelBlockSize(KernelInitialize)); },
        [this, neuronInitHandler](CodeStream &os, const NeuronInitGroupMerged &ng, Substitutions &popSubs)
        {
            os << "// only do this for existing neurons" << std::endl;
            os << "if(" << popSubs["id"] << " < group->numNeurons)";
            {
                CodeStream::Scope b(os);

                // If population RNGs are initialised on device and this neuron is going to require one, 
                // Initialise RNG using GLOBAL thread id for sequence
                if(isPopulationRNGInitialisedOnDevice() && ng.getArchetype().isSimRNGRequired()) {
                    genPopulationRNGInit(os, "group->rng[" + popSubs["id"] + "]", "deviceRNGSeed", "id");
                }

                // If this neuron requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(ng.getArchetype().isInitRNGRequired()) {
                    genGlobalRNGSkipAhead(os, popSubs, "id");
                }

                neuronInitHandler(os, ng, popSubs);
            }
        });
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Synapse groups with dense connectivity" << std::endl;
    genParallelGroup<SynapseDenseInitGroupMerged>(
        os, kernelSubs, modelMerged.getMergedSynapseDenseInitGroups(), idStart,
        [this](const SynapseGroupInternal &sg) { return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), m_KernelBlockSizes[KernelInitialize]); },
        [this, synapseDenseInitHandler](CodeStream &os, const SynapseDenseInitGroupMerged &sg, Substitutions &popSubs)
        {
            os << "// only do this for existing postsynaptic neurons" << std::endl;
            os << "if(" << popSubs["id"] << " < group->numTrgNeurons)";
            {
                CodeStream::Scope b(os);
                // If this post synapse requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(sg.getArchetype().isWUInitRNGRequired()) {
                    genGlobalRNGSkipAhead(os, popSubs, "id");
                }

                popSubs.addVarSubstitution("id_post", popSubs["id"]);
                synapseDenseInitHandler(os, sg, popSubs);
            }
        });
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Synapse groups with sparse connectivity" << std::endl;
    genParallelGroup<SynapseConnectivityInitGroupMerged>(os, kernelSubs, modelMerged.getMergedSynapseConnectivityInitGroups(), idStart,
        [this](const SynapseGroupInternal &sg) { return padSize(sg.getSrcNeuronGroup()->getNumNeurons(), m_KernelBlockSizes[KernelInitialize]); },
        [this, synapseConnectivityInitHandler, sgKernelInitHandler](CodeStream &os, const SynapseConnectivityInitGroupMerged &sg, Substitutions &popSubs)
        {
            os << "// only do this for existing presynaptic neurons" << std::endl;
            os << "if(" << popSubs["id"] << " < group->numSrcNeurons)";
            {
                CodeStream::Scope b(os);
                popSubs.addVarSubstitution("id_pre", popSubs["id"]);
                popSubs.addVarSubstitution("id_post_begin", "0");
                popSubs.addVarSubstitution("id_thread", "0");
                popSubs.addVarSubstitution("num_threads", "1");
                popSubs.addVarSubstitution("num_post", "group->numTrgNeurons");

                // If the synapse group has bitmask connectivity
                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    // Calculate the maximum number of synapses in any groups
                    size_t maxSynapses = 0;
                    for(const auto &g : sg.getGroups()) {
                        const size_t numSynapses = (size_t)g.get().getSrcNeuronGroup()->getNumNeurons() * (size_t)getSynapticMatrixRowStride(g.get());
                        maxSynapses = std::max(maxSynapses, numSynapses);
                    }

                    // Calculate indices of bits at start and end of row
                    os << "// Calculate indices" << std::endl;
                    if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                        os << "const uint64_t rowStartGID = " << popSubs["id"] << " * (uint64_t)group->rowStride;" << std::endl;
                    }
                    else {
                        os << "const unsigned int rowStartGID = " << popSubs["id"] << " * group->rowStride;" << std::endl;
                    }

                    // Build function template to set correct bit in bitmask
                    popSubs.addFuncSubstitution("addSynapse", 1,
                                                getAtomic("unsigned int", AtomicOperation::OR) + "(&group->gp[(rowStartGID + $(0)) / 32], 0x80000000 >> ((rowStartGID + $(0)) & 31))");
                }
                // Otherwise, if synapse group has ragged connectivity
                else if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    // Zero row length
                    const std::string rowLength = "group->rowLength[" + popSubs["id"] + "]";
                    os << rowLength << " = 0;" << std::endl;

                    // If synapse group doesn't define a kernel, build function template to increment row length and insert synapse into ind array
                    if(sg.getArchetype().getKernelSize().empty()) {
                        popSubs.addFuncSubstitution("addSynapse", 1,
                                                    "group->ind[(" + popSubs["id"] + " * group->rowStride) + (" + rowLength + "++)] = $(0)");
                    }
                    // Otherwise
                    // **YUCK** there is a lot of duplication here with backendSIMT but,
                    // the double substitution business is a bit gnarly to expose
                    else {
                        // Create new stream to generate addSynapse function which initializes all kernel variables
                        std::ostringstream kernelInitStream;
                        CodeStream kernelInit(kernelInitStream);

                        // Calculate index in data structure of this synapse
                        kernelInit << "const unsigned int idx = " << "(" + popSubs["id_pre"] + " * group->rowStride) + " << rowLength << ";" << std::endl;
                        
                        // Use classic macro trick to turn block of initialization code into statement and 'eat' semicolon
                        kernelInit << "do";
                        {
                            CodeStream::Scope b(kernelInit);
                            Substitutions kernelInitSubs(&popSubs);

                            // Replace $(id_post) with first 'function' parameter as simulation code is
                            // going to be, in turn, substituted into procedural connectivity generation code
                            kernelInitSubs.addVarSubstitution("id_post", "$(0)");

                            // Add index of synapse
                            kernelInitSubs.addVarSubstitution("id_syn", "idx");

                            // Replace kernel indices with the subsequent 'function' parameters
                            for(size_t i = 0; i < sg.getArchetype().getKernelSize().size(); i++) {
                                kernelInitSubs.addVarSubstitution("id_kernel_" + std::to_string(i), "$(" + std::to_string(i + 1) + ")");
                            }

                            // Call handler to initialize variables
                            sgKernelInitHandler(kernelInit, sg, kernelInitSubs);

                            // End initialization code with code to set postsynaptic index and increment row length
                            kernelInit << "group->ind[idx] = $(0);" << std::endl;
                            kernelInit << rowLength << "++;" << std::endl;
                        }
                        kernelInit << "while(false)";
                        popSubs.addFuncSubstitution("addSynapse", 1 + (unsigned int)sg.getArchetype().getKernelSize().size(),
                                                    kernelInitStream.str());
                    }
                }
                else {
                    assert(false);
                }

                // If this connectivity requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(::Utils::isRNGRequired(sg.getArchetype().getConnectivityInitialiser().getSnippet()->getRowBuildCode())) {
                    genGlobalRNGSkipAhead(os, popSubs, "id");
                }

                synapseConnectivityInitHandler(os, sg, popSubs);
            }
        });
    os << std::endl;
}
//--------------------------------------------------------------------------
void BackendSIMT::genInitializeSparseKernel(CodeStream &os, const Substitutions &kernelSubs, const ModelSpecMerged &modelMerged,
                                            SynapseSparseInitGroupMergedHandler synapseSparseInitHandler, 
                                            size_t numInitializeThreads, size_t &idStart) const
{
    // Shared memory array so row lengths don't have to be read by EVERY postsynaptic thread
    // **TODO** check actually required
    os << getSharedPrefix() << "unsigned int shRowLength[" << getKernelBlockSize(KernelInitializeSparse) << "];" << std::endl;
    if(std::any_of(modelMerged.getModel().getSynapseGroups().cbegin(), modelMerged.getModel().getSynapseGroups().cend(),
                   [](const ModelSpec::SynapseGroupValueType &s) { return (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getSynapseDynamicsCode().empty(); }))
    {
        os << getSharedPrefix() << "unsigned int shRowStart[" << getKernelBlockSize(KernelInitializeSparse) + 1 << "];" << std::endl;
    }

    // Initialise weight update variables for synapse groups with sparse connectivity
    genParallelGroup<SynapseSparseInitGroupMerged>(os, kernelSubs, modelMerged.getMergedSynapseSparseInitGroups(), idStart,
        [this](const SynapseGroupInternal &sg) { return padSize(sg.getMaxConnections(), getKernelBlockSize(KernelInitializeSparse)); },
        [this, synapseSparseInitHandler, numInitializeThreads](CodeStream &os, const SynapseSparseInitGroupMerged &sg, Substitutions &popSubs)
        {
            // If this post synapse requires an RNG for initialisation,
            // make copy of global phillox RNG and skip ahead by thread id
            // **NOTE** not LOCAL id
            if(sg.getArchetype().isWUInitRNGRequired()) {
                genGlobalRNGSkipAhead(os, popSubs, std::to_string(numInitializeThreads) + " + id");
            }

            // Calculate how many blocks rows need to be processed in (in order to store row lengths in shared memory)
            const size_t blockSize = getKernelBlockSize(KernelInitializeSparse);
            os << "const unsigned int numBlocks = (group->numSrcNeurons + " << blockSize << " - 1) / " << blockSize << ";" << std::endl;

            os << "unsigned int idx = " << popSubs["id"] << ";" << std::endl;

            // Loop through blocks
            os << "for(unsigned int r = 0; r < numBlocks; r++)";
            {
                CodeStream::Scope b(os);

                // Calculate number of rows to process in this block
                os << "const unsigned numRowsInBlock = (r == (numBlocks - 1))";
                os << " ? ((group->numSrcNeurons - 1) % " << blockSize << ") + 1";
                os << " : " << blockSize << ";" << std::endl;

                // Use threads to copy block of sparse structure into shared memory
                genSharedMemBarrier(os);
                os << "if (" << getThreadID() << " < numRowsInBlock)";
                {
                    CodeStream::Scope b(os);
                    os << "shRowLength[" << getThreadID() << "] = group->rowLength[(r * " << blockSize << ") + " << getThreadID() << "];" << std::endl;
                }

                // If this synapse group has synapse dynamics
                if(!sg.getArchetype().getWUModel()->getSynapseDynamicsCode().empty()) {
                    genSharedMemBarrier(os);

                    // Use first thread to generate cumulative sum
                    os << "if(" << getThreadID() << " == 0)";
                    {
                        CodeStream::Scope b(os);

                        // Get index of last row in resultant synapse dynamics structure
                        // **NOTE** if there IS a previous block, it will always have had initSparseBlkSz rows in it
                        os << "unsigned int rowStart = (r == 0) ? 0 : shRowStart[" << blockSize << "];" << std::endl;
                        os << "shRowStart[0] = rowStart;" << std::endl;

                        // Loop through rows in block
                        os << "for(unsigned int i = 0; i < numRowsInBlock; i++)";
                        {
                            CodeStream::Scope b(os);

                            // Add this row's length to cumulative sum and write this to this row's end
                            os << "rowStart += shRowLength[i];" << std::endl;
                            os << "shRowStart[i + 1] = rowStart;" << std::endl;
                        }

                        // If this is the first thread block of the first block in the group AND the last block of rows,
                        // write the total cumulative sum to the first entry of the remap structure
                        os << "if(" << popSubs["id"] << " == 0 && (r == (numBlocks - 1)))";
                        {
                            CodeStream::Scope b(os);
                            os << "group->synRemap[0] = shRowStart[numRowsInBlock];" << std::endl;
                        }

                    }
                }

                genSharedMemBarrier(os);

                // Loop through rows
                os << "for(unsigned int i = 0; i < numRowsInBlock; i++)";
                {
                    CodeStream::Scope b(os);

                    // If there is a synapse for this thread to initialise
                    os << "if(" << popSubs["id"] << " < shRowLength[i])";
                    {
                        CodeStream::Scope b(os);

                        // Generate sparse initialisation code
                        if(sg.getArchetype().isWUVarInitRequired()) {
                            popSubs.addVarSubstitution("id_pre", "((r * " + std::to_string(blockSize) + ") + i)");
                            popSubs.addVarSubstitution("id_post", "group->ind[idx]");
                            synapseSparseInitHandler(os, sg, popSubs);
                        }

                        // If postsynaptic learning is required
                        if(!sg.getArchetype().getWUModel()->getLearnPostCode().empty()) {
                            CodeStream::Scope b(os);

                            // Extract index of synapse's postsynaptic target
                            os << "const unsigned int postIndex = group->ind[idx];" << std::endl;

                            // Atomically increment length of column of connectivity associated with this target
                            // **NOTE** this returns previous length i.e. where to insert new entry
                            os << "const unsigned int colLocation = " << getAtomic("unsigned int") << "(&group->colLength[postIndex], 1);" << std::endl;

                            // From this calculate index into column-major matrix
                            os << "const unsigned int colMajorIndex = (postIndex * group->colStride) + colLocation;" << std::endl;

                            // Add remapping entry at this location poining back to row-major index
                            os << "group->remap[colMajorIndex] = idx;" << std::endl;
                        }

                        // If synapse dynamics are required, copy idx into syn remap structure
                        if(!sg.getArchetype().getWUModel()->getSynapseDynamicsCode().empty()) {
                            CodeStream::Scope b(os);
                            os << "group->synRemap[shRowStart[i] + " + popSubs["id"] + " + 1] = idx;" << std::endl;
                        }
                    }

                    // If matrix is ragged, advance index to next row by adding stride
                    os << "idx += group->rowStride;" << std::endl;
                }
            }
        });
}
//--------------------------------------------------------------------------
void BackendSIMT::addDeviceType(const std::string &type, size_t size)
{
    addType(type, size);
    m_DeviceTypes.emplace(type);
}
//--------------------------------------------------------------------------
bool BackendSIMT::isDeviceType(const std::string &type) const
{
    // Get underlying type
    const std::string underlyingType = ::Utils::isTypePointer(type) ? ::Utils::getUnderlyingType(type) : type;

    // Return true if it is in device types set
    return (m_DeviceTypes.find(underlyingType) != m_DeviceTypes.cend());
}
//--------------------------------------------------------------------------
void BackendSIMT::genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix, bool recordingEnabled) const
{
    os << "const unsigned int spk" << suffix << "Idx = " << getAtomic("unsigned int", AtomicOperation::ADD, AtomicMemSpace::SHARED) << "(&shSpk" << suffix << "Count, 1);" << std::endl;
    os << "shSpk" << suffix << "[spk" << suffix << "Idx] = " << subs["id"] << ";" << std::endl;
    
    // If recording is enabled, set bit in recording word
    if(recordingEnabled) {
        if(m_KernelBlockSizes[KernelNeuronUpdate] == 32) {
            os << getAtomic("unsigned int", AtomicOperation::OR, AtomicMemSpace::SHARED) << "(&shSpk" << suffix << "Record, 1 << " << getThreadID() << ");" << std::endl;
        }
        else {
            os << getAtomic("unsigned int", AtomicOperation::OR, AtomicMemSpace::SHARED) << "(&shSpk" << suffix << "Record[" << getThreadID() << " / 32], 1 << (" << getThreadID() << " % 32));" << std::endl;
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
}   // namespace CodeGenerator

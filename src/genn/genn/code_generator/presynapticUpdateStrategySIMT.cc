#include "code_generator/presynapticUpdateStrategySIMT.h"

// Standard C++ includes
#include <numeric>

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
#include "code_generator/backendSIMT.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/groupMerged.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/substitutions.h"


using namespace CodeGenerator;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
bool isSmallSharedMemoryPop(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend)
{
    // If shared memory atomics are slow
    const size_t blockSize = backend.getKernelBlockSize(CodeGenerator::KernelPresynapticUpdate);
    if(backend.areSharedMemAtomicsSlow()) {
        return false;
    }
    // Otherwise, if dendritic delays are required, shared memory approach cannot be used so return false
    else if(sg.getArchetype().isDendriticDelayRequired()) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if all neuron groups targetted
    // by synapse groups within merged group are small enough that input to then can be stored in a shared memory array
    else if(std::all_of(sg.getGroups().cbegin(), sg.getGroups().cend(),
                        [blockSize](const SynapseGroupInternal &sg)
                        {
                            return (sg.getTrgNeuronGroup()->getNumNeurons() <= blockSize);
                        }))
    {
        return true;
    }
    else {
        return false;
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateStrategySIMT::PreSpan
//----------------------------------------------------------------------------
namespace CodeGenerator
{
namespace PresynapticUpdateStrategySIMT
{
size_t PreSpan::getNumThreads(const SynapseGroupInternal &sg) const
{
    // Use specified number of threads for each presynaptic neuron
    return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * (size_t)sg.getNumThreadsPerSpike();
}
//----------------------------------------------------------------------------
size_t PreSpan::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
bool PreSpan::isCompatible(const SynapseGroupInternal &sg, const PreferencesBase&) const
{
    // Presynaptic parallelism can be used when synapse groups request it and they have sparse connectivity
    return ((sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC)
            && (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE));
}
//----------------------------------------------------------------------------
size_t PreSpan::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged&, const BackendSIMT&) const
{
    return 0;
}
//----------------------------------------------------------------------------
void PreSpan::genPreamble(CodeStream &, const ModelSpecMerged&, const PresynapticUpdateGroupMerged&,
                          const Substitutions&, const BackendSIMT&) const
{
}
//----------------------------------------------------------------------------
void PreSpan::genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                        const Substitutions &popSubs, const BackendSIMT &backend, bool trueSpike,
                        BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, 
                        BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler,
                        BackendBase::PresynapticUpdateGroupMergedHandler) const
{
    // Get suffix based on type of events
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const auto *wu = sg.getArchetype().getWUModel();
    const size_t numThreadsPerSpike = sg.getArchetype().getNumThreadsPerSpike();

    if(numThreadsPerSpike > 1) {
        os << "const unsigned int spike = " << popSubs["id"] << " / " << numThreadsPerSpike << ";" << std::endl;
        os << "const unsigned int thread = " << popSubs["id"] << " % " << numThreadsPerSpike << ";" << std::endl;
    }
    else {
        os << "const unsigned int spike = " << popSubs["id"] << ";" << std::endl;
    }

    os << "if (spike < group->srcSpkCnt" << eventSuffix << "[" << sg.getPreSlot(batchSize) << "])";
    {
        CodeStream::Scope b(os);

        if(backend.supportsNamespace() && !wu->getSimSupportCode().empty()) {
            os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
        }

        os << "const unsigned int preInd = group->srcSpk" << eventSuffix << "[" << sg.getPreVarIndex(batchSize, VarAccessDuplication::DUPLICATE, "spike") << "];" << std::endl;

        if(numThreadsPerSpike > 1) {
            os << "unsigned int synAddress = (preInd * group->rowStride) + thread;" << std::endl;
        }
        else {
            os << "unsigned int synAddress = preInd * group->rowStride;" << std::endl;
        }
        os << "const unsigned int npost = group->rowLength[preInd];" << std::endl;

        if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << "if(";

            Substitutions threshSubs(&popSubs);
            threshSubs.addVarSubstitution("id_pre", "preInd");

            // Generate weight update threshold condition
            wumThreshHandler(os, sg, threshSubs);

            // end code substitutions ----
            os << ")";

            os << CodeStream::OB(130);
        }

        if(numThreadsPerSpike > 1) {
            os << "for(unsigned int i = thread; i < npost; i += " << numThreadsPerSpike << ", synAddress += " << numThreadsPerSpike << ")";
        }
        else {
            os << "for(unsigned int i = 0; i < npost; i++, synAddress++)";
        }
        {
            CodeStream::Scope b(os);

            // **TODO** pretty sure __ldg will boost performance here - basically will bring whole row into cache
            os << "const unsigned int ipost = group->ind[synAddress];" << std::endl;

            // Create substitution stack for presynaptic simulation code
            Substitutions synSubs(&popSubs);
            synSubs.addVarSubstitution("id_pre", "preInd");
            synSubs.addVarSubstitution("id_post", "ipost");
            synSubs.addVarSubstitution("id_syn", "synAddress");

            // If dendritic delay is required, use atomic operation to update dendritic delay buffer
            if(sg.getArchetype().isDendriticDelayRequired()) {
                synSubs.addFuncSubstitution("addToInSynDelay", 2, 
                                            backend.getAtomic(model.getPrecision()) + "(&group->denDelay[" + sg.getPostDenDelayIndex(batchSize, "ipost", "$(1)") + "], $(0))");
            }
            // Otherwise, substitute global memory array for $(inSyn)
            else {
                synSubs.addFuncSubstitution("addToInSyn", 1, 
                                            backend.getAtomic(model.getPrecision()) + "(&group->inSyn[" + sg.getPostISynIndex(batchSize, "ipost") + "], $(0))");
            }

            wumSimHandler(os, sg, synSubs);
        }

        if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }
    }
}
//----------------------------------------------------------------------------
void PreSpan::genPostamble(CodeStream&, const ModelSpecMerged&, const PresynapticUpdateGroupMerged&,
                           const Substitutions&, const BackendSIMT&) const
{
}

//----------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpan
//----------------------------------------------------------------------------
size_t PostSpan::getNumThreads(const SynapseGroupInternal &sg) const
{
    // **NOTE** we don't really care about extra padding i.e. stride here
    if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxConnections();
    }
    else {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//----------------------------------------------------------------------------
size_t PostSpan::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxConnections();
    }
    else {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//----------------------------------------------------------------------------
bool PostSpan::isCompatible(const SynapseGroupInternal &sg, const PreferencesBase&) const
{
    // Postsynatic parallelism can be used when synapse groups request it
    return ((sg.getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC)
            && !(sg.getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL));
}
//----------------------------------------------------------------------------
void PostSpan::genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &, const BackendSIMT &backend) const
{
    // If data structure is dense, we can accumulate output directly into register
    if(shouldAccumulateInRegister(sg)) {
        os << modelMerged.getModel().getPrecision() << " linSyn = 0;" << std::endl;
    }
    else if(isSmallSharedMemoryPop(sg, backend)) {
        os << "if(" << backend.getThreadID() << " < group->numTrgNeurons)";
        {
            CodeGenerator::CodeStream::Scope b(os);
            os << "shLg[" << backend.getThreadID() << "] = 0;" << std::endl;
        }
        backend.genSharedMemBarrier(os);
    }
}
//----------------------------------------------------------------------------
size_t PostSpan::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PostSpan::genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                         const Substitutions &popSubs, const BackendSIMT &backend, bool trueSpike,
                         BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, 
                         BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler,
                         BackendBase::PresynapticUpdateGroupMergedHandler) const
{
    // Get suffix based on type of events
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const std::string eventSuffix = trueSpike ? "" : "Evnt";

    os << "const unsigned int numSpikes = group->srcSpkCnt" << eventSuffix << "[" << sg.getPreSlot(batchSize) << "];" << std::endl;
    os << "const unsigned int numSpikeBlocks = (numSpikes + " << backend.getKernelBlockSize(KernelPresynapticUpdate) << " - 1) / " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

    const auto *wu = sg.getArchetype().getWUModel();
    os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(os);
        os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ") + 1 : " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

        backend.genSharedMemBarrier(os);
        os << "if (" << backend.getThreadID() << " < numSpikesInBlock)";
        {
            CodeStream::Scope b(os);
            const std::string index = "(r * " + std::to_string(backend.getKernelBlockSize(KernelPresynapticUpdate)) + ") + " + backend.getThreadID();
            os << "const unsigned int spk = group->srcSpk" << eventSuffix << "[" << sg.getPreVarIndex(batchSize, VarAccessDuplication::DUPLICATE, index) << "];" << std::endl;
            os << "shSpk" << eventSuffix << "[" << backend.getThreadID() << "] = spk;" << std::endl;
            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                os << "shRowLength[" << backend.getThreadID() << "] = group->rowLength[spk];" << std::endl;
            }
        }
        backend.genSharedMemBarrier(os);

        os << "// loop through all incoming spikes" << std::endl;
        os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(os);
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << popSubs["id"] << " < group->rowStride)";
            {
                CodeStream::Scope b(os);
                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    // If this can only be represented using a 64-bit number
                    if(backend.areSixtyFourBitSynapseIndicesRequired(sg)) {
                        os << "const uint64_t gid = (shSpk" << eventSuffix << "[j] * (uint64_t)group->rowStride) + " << popSubs["id"] << ";" << std::endl;
                    }
                    else {
                        os << "const unsigned int gid = (shSpk" << eventSuffix << "[j] * group->rowStride) + " << popSubs["id"] << ";" << std::endl;
                    }
                }

                if(backend.supportsNamespace() && !wu->getSimSupportCode().empty()) {
                    os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
                }
                if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << "if(";
                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // Note: we will just access global mem. For compute >= 1.2 simultaneous access to same global mem in the (half-)warp will be coalesced - no worries
                        os << "(B(group->gp[gid / 32], gid & 31)) && ";
                    }

                    Substitutions threshSubs(&popSubs);
                    threshSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");

                    // Generate weight update threshold condition
                    wumThreshHandler(os, sg, threshSubs);

                    // end code substitutions ----
                    os << ")";
                    os << CodeStream::OB(130);
                }
                else if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "if (B(group->gp[gid / 32], gid & 31))" << CodeStream::OB(135);
                }

                os << "const unsigned int synAddress = (shSpk" << eventSuffix << "[j] * group->rowStride) + " + popSubs["id"] + ";" << std::endl;

                Substitutions synSubs(&popSubs);
                synSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");
                synSubs.addVarSubstitution("id_syn", "synAddress");

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    os << "const unsigned int npost = shRowLength[j];" << std::endl;

                    os << "if (" << popSubs["id"] << " < npost)" << CodeStream::OB(140);
                    os << "const unsigned int ipost = group->ind[synAddress];" << std::endl;

                    synSubs.addVarSubstitution("id_post", "ipost");
                }
                else { // DENSE
                    synSubs.addVarSubstitution("id_post", popSubs["id"]);
                }

                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                if(sg.getArchetype().isDendriticDelayRequired()) {
                    synSubs.addFuncSubstitution("addToInSynDelay", 2, 
                                                backend.getAtomic(model.getPrecision()) + "(&group->denDelay[" + sg.getPostDenDelayIndex(batchSize, synSubs["id_post"], "$(1)") + "], $(0))");
                }
                // Otherwise
                else {
                    // If we should accumulate in register, add parameter to register
                    if(shouldAccumulateInRegister(sg)) {
                        synSubs.addFuncSubstitution("addToInSyn", 1, "linSyn += $(0)");
                    }
                    // Otherwise, if we should use shared memory, add to shared memory
                    // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                    else if(isSmallSharedMemoryPop(sg, backend)) {
                        synSubs.addFuncSubstitution("addToInSyn", 1, "shLg[" + synSubs["id_post"] + "] += $(0)");
                    }
                    // Otherwise, use global memory atomic
                    else {
                        synSubs.addFuncSubstitution("addToInSyn", 1, 
                                                    backend.getAtomic(model.getPrecision()) + "(&group->inSyn[" + sg.getPostISynIndex(batchSize, synSubs["id_post"]) + "], $(0))");
                    }
                }

                wumSimHandler(os, sg, synSubs);

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    os << CodeStream::CB(140); // end if (id < npost)
                }

                if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
                else if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << CodeStream::CB(135); // end if (B(dd_gp" << sg.getName() << "[gid / 32], gid
                }
            }
        }
    }
}
//----------------------------------------------------------------------------
void PostSpan::genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                            const Substitutions &popSubs, const BackendSIMT &backend) const
{
    // If we should accumulate output directly into register
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    if(shouldAccumulateInRegister(sg)) {
        os << "// only do this for existing neurons" << std::endl;
        os << "if (" << popSubs["id"] << " < group->numTrgNeurons)";
        {
            CodeStream::Scope b(os);
            const std::string inSyn = "group->inSyn[" + sg.getPostISynIndex(batchSize, popSubs["id"]) + "]";
            if(sg.getArchetype().isPSModelMerged()) {
                os << backend.getAtomic(model.getPrecision()) << "(&" << inSyn << ", linSyn);" << std::endl;
            }
            else {
                os << inSyn << " += linSyn;" << std::endl;
            }
        }
    }
    // Otherwise, if we should accumulate into shared memory
    else if(isSmallSharedMemoryPop(sg, backend)) {
        backend.genSharedMemBarrier(os);
        os << "if(" << backend.getThreadID() << " < group->numTrgNeurons)";
        {
            CodeGenerator::CodeStream::Scope b(os);
            os << backend.getAtomic(model.getPrecision()) << "(&group->inSyn[" << sg.getPostISynIndex(batchSize, backend.getThreadID()) << "], ";
            os << "shLg[" << backend.getThreadID() << "]); " << std::endl;
        }
    }
}
// ----------------------------------------------------------------------------
bool PostSpan::shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &sg) const
{
    // If no dendritic delays are required and data structure is dense, we can accumulate output directly into register
    const auto matrixType = sg.getArchetype().getMatrixType();
    return (!sg.getArchetype().isDendriticDelayRequired()
            && ((matrixType & SynapseMatrixConnectivity::DENSE) || (matrixType & SynapseMatrixConnectivity::BITMASK)));
}

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PreSpanProcedural
//--------------------------------------------------------------------------
size_t PreSpanProcedural::getNumThreads(const SynapseGroupInternal &sg) const
{
    // Use specified number of threads for each presynaptic neuron
    return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * (size_t)sg.getNumThreadsPerSpike();
}
//----------------------------------------------------------------------------
size_t PreSpanProcedural::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
bool PreSpanProcedural::isCompatible(const SynapseGroupInternal &sg, const PreferencesBase &) const
{
    // Presynaptic procedural parallelism can be used when synapse groups have 
    // procedural connectivity and weights are either GLOBAL or PROCEDURAL
    const auto matrixType = sg.getMatrixType();
    return ((matrixType & SynapseMatrixConnectivity::PROCEDURAL)
            && ((matrixType & SynapseMatrixWeight::GLOBAL) || (matrixType & SynapseMatrixWeight::PROCEDURAL)));
}
//----------------------------------------------------------------------------
size_t PreSpanProcedural::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged&, const BackendSIMT&) const
{
    return 0;
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genPreamble(CodeStream&, const ModelSpecMerged&, const PresynapticUpdateGroupMerged&,
                                    const Substitutions&, const BackendSIMT&) const
{
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                                  const Substitutions &popSubs, const BackendSIMT &backend, bool trueSpike,
                                  BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, 
                                  BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler,
                                  BackendBase::PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler) const
{
    // Get suffix based on type of events
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const auto *wu = sg.getArchetype().getWUModel();
    const size_t numThreadsPerSpike = sg.getArchetype().getNumThreadsPerSpike();

    if(numThreadsPerSpike > 1) {
        os << "const unsigned int spike = " << popSubs["id"] << " / " << numThreadsPerSpike << ";" << std::endl;
        os << "const unsigned int thread = " << popSubs["id"] << " % " << numThreadsPerSpike << ";" << std::endl;
        os << "const unsigned int numPostPerThread =  (group->numTrgNeurons + " << numThreadsPerSpike << " - 1) / " << numThreadsPerSpike << ";" << std::endl;

        // Calculate the starting position and length of the sub-row to process on this thread
        // **TODO** fast-divide style optimisations here
        os << "const unsigned int idPostStart = thread * numPostPerThread;" << std::endl;
        os << "const unsigned int postRemainder = group->numTrgNeurons % numPostPerThread;" << std::endl;
        os << "const unsigned int numPost = (postRemainder == 0 || thread < " << (numThreadsPerSpike - 1) << ") ? numPostPerThread : postRemainder;" << std::endl;
    }
    else {
        os << "const unsigned int spike = " << popSubs["id"] << ";" << std::endl;
    }

    // If there is a spike for this thread to process
    os << "if (spike < group->srcSpkCnt" << eventSuffix << "[" << sg.getPreSlot(batchSize) << "])";
    {
        CodeStream::Scope b(os);

        // Determine the index of the presynaptic neuron this thread is responsible for
        os << "const unsigned int preInd = group->srcSpk" << eventSuffix << "[" << sg.getPreVarIndex(batchSize, VarAccessDuplication::DUPLICATE, "spike") << "];" << std::endl;

        // Create substitution stack and add presynaptic index
        Substitutions synSubs(&popSubs);
        synSubs.addVarSubstitution("id_pre", "preInd");

        if(backend.supportsNamespace() && !wu->getSimSupportCode().empty()) {
            os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
        }

        if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << "if(";

            // Generate weight update threshold condition
            Substitutions threshSubs(&synSubs);
            wumThreshHandler(os, sg, threshSubs);

            // end code substitutions ----
            os << ")";

            os << CodeStream::OB(130);
        }

        // Create substitution stack for generating procedural connectivity code
        Substitutions connSubs(&synSubs);
        connSubs.addVarSubstitution("num_threads", std::to_string(numThreadsPerSpike));

        // If this connectivity requires an RNG for initialisation,
        // make copy of connect Phillox RNG and skip ahead to id that would have been used to initialize any variables associated with it
        if(::Utils::isRNGRequired(sg.getArchetype().getConnectivityInitialiser().getSnippet()->getRowBuildCode())
           || ((sg.getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL) && ::Utils::isRNGRequired(sg.getArchetype().getWUVarInitialisers())))
        {
            std::stringstream skipAhead;
            if(numThreadsPerSpike > 1) {
                skipAhead << "(preInd * " << numThreadsPerSpike << ") + thread";
            }
            else {
                skipAhead << "preInd";
            }
            skipAhead << " + " << connSubs["group_start_id"] << " + " << (backend.getNumInitialisationRNGStreams(modelMerged) * model.getBatchSize());

            // **NOTE** add RNG to synSubs so it can be correctly referenced in presynapticUpdateSubs below
            backend.genGlobalRNGSkipAhead(os, synSubs, skipAhead.str());
        }

        // If we are using more than one thread to process each row
        if(numThreadsPerSpike > 1) {
            connSubs.addVarSubstitution("id_post_begin", "idPostStart");
            connSubs.addVarSubstitution("id_thread", "thread");
            connSubs.addVarSubstitution("num_post", "numPost");
            connSubs.addVarSubstitution("num_pre", "group->numSrcNeurons");
        }
        else {
            connSubs.addVarSubstitution("id_post_begin", "0");
            connSubs.addVarSubstitution("id_thread", "0");
            connSubs.addVarSubstitution("num_post", "group->numTrgNeurons");
            connSubs.addVarSubstitution("num_pre", "group->numSrcNeurons");
        }

        // Create another substitution stack for generating presynaptic simulation code
        Substitutions presynapticUpdateSubs(&synSubs);

        // Replace $(id_post) with first 'function' parameter as simulation code is
        // going to be, in turn, substituted into procedural connectivity generation code
        presynapticUpdateSubs.addVarSubstitution("id_post", "$(0)");

        // If weights are provided by a kernel
        if(!sg.getArchetype().getKernelSize().empty()) {
            // Replace kernel indices with the subsequent 'function' parameters
            for(size_t i = 0; i < sg.getArchetype().getKernelSize().size(); i++) {
                presynapticUpdateSubs.addVarSubstitution("id_kernel_" + std::to_string(i),
                                                         "$(" + std::to_string(i + 1) + ")");
            }
        }

        // If dendritic delay is required, use atomic operation to update dendritic delay buffer
        if(sg.getArchetype().isDendriticDelayRequired()) {
            presynapticUpdateSubs.addFuncSubstitution("addToInSynDelay", 2, 
                                                      backend.getAtomic(model.getPrecision()) + "(&group->denDelay[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
        }
        // Otherwise, substitute global memory array for $(inSyn)
        else {
            presynapticUpdateSubs.addFuncSubstitution("addToInSyn", 1, 
                                                      backend.getAtomic(model.getPrecision()) + "(&group->inSyn[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
        }

        // Generate presynaptic simulation code into new stringstream-backed code stream
        std::ostringstream presynapticUpdateStream;
        CodeStream presynapticUpdate(presynapticUpdateStream);
        wumSimHandler(presynapticUpdate, sg, presynapticUpdateSubs);

        // When a synapse should be 'added', substitute in presynaptic update code
        connSubs.addFuncSubstitution("addSynapse", 1 + (unsigned int)sg.getArchetype().getKernelSize().size(), presynapticUpdateStream.str());

        // Generate procedural connectivity code
        wumProceduralConnectHandler(os, sg, connSubs);

        if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }
    }
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genPostamble(CodeStream&, const ModelSpecMerged&, const PresynapticUpdateGroupMerged&,
                                     const Substitutions&, const BackendSIMT&) const
{
}

//----------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpanBitmask
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getNumThreads(const SynapseGroupInternal &sg) const
{
    return ceilDivide(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
}
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    // Pad each row to a word boundary
    return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
}
//----------------------------------------------------------------------------
bool PostSpanBitmask::isCompatible(const SynapseGroupInternal &sg, const PreferencesBase &preferences) const
{
    // Postsynaptic bitmask parallelism can be used if bitmask optimisations are enabled and
    // if synapse groups with bitmask connectivity and no dendritic delays request postsynaptic parallelism
    return (preferences.enableBitmaskOptimisations
            && (sg.getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC)
            && (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK)
            && !sg.isDendriticDelayRequired());
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genPreamble(CodeStream &os, const ModelSpecMerged &, const PresynapticUpdateGroupMerged &,
                                  const Substitutions &, const BackendSIMT &backend) const
{
    // Loop through bits written by this thread
    for(size_t i = 0; i < 32; i++) {
        // Zero entries in this thread's shared memory array
        // **NOTE** this is ordered to prevent bank conflicts
        const std::string index = std::to_string(i * backend.getKernelBlockSize(KernelPresynapticUpdate)) + " + " + backend.getThreadID();
        os << "shLg[" << index << "] = 0;" << std::endl;
    }
    backend.genSharedMemBarrier(os);
}
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged&, const BackendSIMT&) const
{
    // Each thread sums up the input to 32 postsynaptic neurons
    return 32;
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                                const Substitutions &popSubs, const BackendSIMT &backend, bool trueSpike,
                                BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler,
                                BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler,
                                BackendBase::PresynapticUpdateGroupMergedHandler) const
{
    // Get suffix based on type of events
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    const std::string eventSuffix = trueSpike ? "" : "Evnt";

    // Get blocksize
    const size_t blockSize = backend.getKernelBlockSize(KernelPresynapticUpdate);

    os << "const unsigned int numSpikes = group->srcSpkCnt" << eventSuffix << "[" << sg.getPreSlot(batchSize) << "];" << std::endl;
    os << "const unsigned int numSpikeBlocks = (numSpikes + " << blockSize << " - 1) / " << blockSize << ";" << std::endl;


    const auto *wu = sg.getArchetype().getWUModel();
    os << "const unsigned int rowWords =  (group->numTrgNeurons + 32 - 1) / 32;" << std::endl;
    os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(os);
        os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << blockSize << ") + 1 : " << blockSize << ";" << std::endl;

        backend.genSharedMemBarrier(os);
        os << "if (" << backend.getThreadID() << " < numSpikesInBlock)";
        {
            CodeStream::Scope b(os);
            const std::string index = "(r * " + std::to_string(backend.getKernelBlockSize(KernelPresynapticUpdate)) + ") + " + backend.getThreadID();
            os << "const unsigned int spk = group->srcSpk" << eventSuffix << "[" << sg.getPreVarIndex(batchSize, VarAccessDuplication::DUPLICATE, index) << "];" << std::endl;
            os << "shSpk" << eventSuffix << "[" << backend.getThreadID() << "] = spk;" << std::endl;
        }
        backend.genSharedMemBarrier(os);

        os << "// loop through all incoming spikes" << std::endl;
        os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(os);
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << popSubs["id"] << " < rowWords)";
            {
                CodeStream::Scope b(os);

                if(backend.supportsNamespace() && !wu->getSimSupportCode().empty()) {
                    os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
                }
                if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << "if(";

                    Substitutions threshSubs(&popSubs);
                    threshSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");

                    // Generate weight update threshold condition
                    wumThreshHandler(os, sg, threshSubs);

                    os << ")";
                    os << CodeStream::OB(130);
                }

                // Read row word
                os << "uint32_t connectivityWord = group->gp[(shSpk" << eventSuffix << "[j] * rowWords) + " << popSubs["id"] << "];" << std::endl;

                // While there any bits left
                os << "unsigned int ibit = 0;" << std::endl;
                os << "while(connectivityWord != 0)";
                {
                    CodeStream::Scope b(os);

                    // Cound leading zeros (as bits are indexed backwards this is index of next synapse)
                    os << "const int numLZ = " << backend.getCLZ() << "(connectivityWord);" << std::endl;

                    // Shift off zeros and the one just discovered
                    // **NOTE** if numLZ == 31, undefined behaviour results in C++, BUT in CUDA this PRESUMABLY emits
                    // In a 'shl' PTX instruction where "Shift amounts greater than the register width N are clamped to N."
                    os << "connectivityWord <<= (numLZ + 1);" << std::endl;

                    // Add to bit index
                    os << "ibit += numLZ;" << std::endl;

                    // Calculate postsynaptic index
                    os << "const unsigned int ipost = ibit + (" << popSubs["id"] << " * 32);" << std::endl;

                    Substitutions synSubs(&popSubs);
                    synSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");
                    synSubs.addVarSubstitution("id_syn", "synAddress");
                    synSubs.addVarSubstitution("id_post", "ipost");
                    synSubs.addFuncSubstitution("addToInSyn", 1, "shLg[(ibit * " + std::to_string(blockSize) + ") + " + backend.getThreadID() + "] += $(0)");
                    wumSimHandler(os, sg, synSubs);

                    os << "ibit++;" << std::endl;
                }


                if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
            }
        }
    }
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                                   const Substitutions &popSubs, const BackendSIMT &backend) const
{
    backend.genSharedMemBarrier(os);
    const size_t blockSize = backend.getKernelBlockSize(KernelPresynapticUpdate);

    // Use first 32 threads in each block to write shared memory back to global memory
    os << "if (" << backend.getThreadID() << " < 32)";
    {
        CodeStream::Scope b(os);
        os << "unsigned int glbIdx = ((" << backend.getBlockID() << " - (" << popSubs["group_start_id"]  << " / " << blockSize << ")) * " << 32 * blockSize << ") + " << backend.getThreadID() << ";" << std::endl;
        os << "unsigned int shIdx = " << backend.getThreadID() << " * " << blockSize << ";" << std::endl;
        os << "const unsigned int endShIdx = shIdx + 32;" << std::endl;
        os << "for(;shIdx < endShIdx && glbIdx < group->numTrgNeurons; shIdx++, glbIdx += 32)";
        {
            CodeStream::Scope b(os);
            const std::string inSyn = "group->inSyn[" + sg.getPostISynIndex(modelMerged.getModel().getBatchSize(), "glbIdx") +"]";
            if(sg.getArchetype().isPSModelMerged()) {
                os << backend.getAtomic(modelMerged.getModel().getPrecision()) << "(&" << inSyn << ", shLg[shIdx]);" << std::endl;
            }
            else {
                os << inSyn << " += shLg[shIdx];" << std::endl;
            }
        }
    }
}
}   // namespace PresynapticUpdateStrategySIMT
}   // namespace CodeGenerator

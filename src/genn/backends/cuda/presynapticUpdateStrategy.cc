#include "presynapticUpdateStrategy.h"

// Standard C++ includes
#include <numeric>

// CUDA includes
#include <cuda_runtime.h>


// GeNN includes
#include "gennUtils.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/substitutions.h"

// CUDA backend includes
#include "backend.h"
#include "utils.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
bool isSmallSharedMemoryPop(const SynapseGroupMerged &sg, const CodeGenerator::CUDA::Backend &backend)
{
    // If device is older than Maxwell, we shouldn't use shared memory as atomics are emulated
    // and actually slower than global memory (see https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
    const size_t blockSize = backend.getKernelBlockSize(CodeGenerator::CUDA::KernelPresynapticUpdate);
    if(backend.getChosenCUDADevice().major < 5) {
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
//----------------------------------------------------------------------------
void genSmallSharedMemoryPopPreamble(CodeGenerator::CodeStream &os, const SynapseGroupMerged &)
{
    os << "if(threadIdx.x < synapseGroup.numTrgNeurons)";
    {
        CodeGenerator::CodeStream::Scope b(os);
        os << "shLg[threadIdx.x] = 0;" << std::endl;
    }
    os << "__syncthreads();" << std::endl;
}
//----------------------------------------------------------------------------
void genSmallSharedMemoryPopPostamble(CodeGenerator::CodeStream &os, const ModelSpecMerged &model,
                                      const SynapseGroupMerged &sg, const CodeGenerator::CUDA::Backend &backend)
{
    os << "__syncthreads();" << std::endl;
    os << "if (threadIdx.x < synapseGroup.numTrgNeurons)";
    {
        CodeGenerator::CodeStream::Scope b(os);
        const std::string inSyn = "(*synapseGroup.inSyn)[threadIdx.x]";
        if (sg.getArchetype().isPSModelMerged()) {
            os << backend.getFloatAtomicAdd(model.getPrecision()) << "(&" << inSyn << ", shLg[threadIdx.x]);" << std::endl;
        }
        else {
            os << inSyn << " += shLg[threadIdx.x];" << std::endl;
        }
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PreSpan
//----------------------------------------------------------------------------
namespace CodeGenerator
{
namespace CUDA
{
namespace PresynapticUpdateStrategy
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
bool PreSpan::isCompatible(const SynapseGroupMerged &sg, const cudaDeviceProp &, const Preferences &) const
{
    // Presynaptic parallelism can be used when synapse groups request it and they have sparse connectivity
    return ((sg.getArchetype().getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC)
            && (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE));
}
//----------------------------------------------------------------------------
size_t PreSpan::getSharedMemoryPerThread(const SynapseGroupMerged &sg, const Backend &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PreSpan::genPreamble(CodeStream &os, const ModelSpecMerged &, const SynapseGroupMerged &sg,
                          const Substitutions &, const Backend &backend, size_t) const
{
    if (isSmallSharedMemoryPop(sg, backend)) {
        genSmallSharedMemoryPopPreamble(os, sg);
    }
}
//----------------------------------------------------------------------------
void PreSpan::genUpdate(CodeStream &os, const ModelSpecMerged &model, const SynapseGroupMerged &sg,
                        const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t,
                        BackendBase::SynapseGroupMergedHandler wumThreshHandler, BackendBase::SynapseGroupMergedHandler wumSimHandler,
                        BackendBase::SynapseGroupMergedHandler) const
{
    // Get suffix based on type of events
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

    os << "if (spike < " ;
    if (sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        os << "(*synapseGroup.spkCnt" << eventSuffix << ")[preReadDelaySlot])";
    }
    else {
        os << "(*synapseGroup.spkCnt" << eventSuffix << ")[0])";
    }
    {
        CodeStream::Scope b(os);

        if (!wu->getSimSupportCode().empty()) {
            os << "using namespace " << sg.getArchetype().getName() << "_weightupdate_simCode;" << std::endl;
        }

        os << "const unsigned int preInd = (*synapseGroup.spk"  << eventSuffix << ")";
        if (sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
            os << "[(preReadDelaySlot * synapseGroup.numSrcNeurons) + spike];" << std::endl;
        }
        else {
            os << "[spike];" << std::endl;
        }

        if(numThreadsPerSpike > 1) {
            os << "unsigned int synAddress = (preInd * synapseGroup.rowStride) + thread;" << std::endl;
        }
        else {
            os << "unsigned int synAddress = preInd * synapseGroup.rowStride;" << std::endl;
        }
        os << "const unsigned int npost = (*synapseGroup.rowLength)[preInd];" << std::endl;

        if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
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
            os << "const unsigned int ipost = (*synapseGroup.ind)[synAddress];" << std::endl;

            // Create substitution stack for presynaptic simulation code
            Substitutions synSubs(&popSubs);
            synSubs.addVarSubstitution("id_pre", "preInd");
            synSubs.addVarSubstitution("id_post", "ipost");
            synSubs.addVarSubstitution("id_syn", "synAddress");

            // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
            if(sg.getArchetype().isDendriticDelayRequired()) {
                assert(false);
                //synSubs.addFuncSubstitution("addToInSynDelay", 2, backend.getFloatAtomicAdd(model.getPrecision()) + "(&(*synapsegroup.denDelay)[" + sg.getDendriticDelayOffset("dd_", "$(1)") + "ipost], $(0))");
            }
            // Otherwise
            else {
                // If postsynaptic input should be accumulated in shared memory, substitute shared memory array for $(inSyn)
                if(isSmallSharedMemoryPop(sg, backend)) {
                    synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&shLg[ipost], $(0))");
                }
                // Otherwise, substitute global memory array for $(inSyn)
                else {
                    synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&(*synapseGroup.inSyn)[ipost], $(0))");
                }
            }

            wumSimHandler(os, sg, synSubs);
        }

        if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }
    }
}
//----------------------------------------------------------------------------
void PreSpan::genPostamble(CodeStream &os, const ModelSpecMerged &model, const SynapseGroupMerged &sg,
                           const Substitutions &, const Backend &backend, size_t) const
{
    if (isSmallSharedMemoryPop(sg, backend)) {
        genSmallSharedMemoryPopPostamble(os, model, sg, backend);
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpan
//----------------------------------------------------------------------------
size_t PostSpan::getNumThreads(const SynapseGroupInternal &sg) const
{
    // **NOTE** we don't really care about extra padding i.e. stride here
    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxConnections();
    }
    else {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//----------------------------------------------------------------------------
size_t PostSpan::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxConnections();
    }
    else {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//----------------------------------------------------------------------------
bool PostSpan::isCompatible(const SynapseGroupMerged &sg, const cudaDeviceProp &, const Preferences &) const
{
    // Postsynatic parallelism can be used when synapse groups request it
    return ((sg.getArchetype().getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC)
            && !(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL));
}
//----------------------------------------------------------------------------
void PostSpan::genPreamble(CodeStream &os, const ModelSpecMerged &model, const SynapseGroupMerged &sg,
                           const Substitutions &, const Backend &backend, size_t) const
{
    // If data structure is dense, we can accumulate output directly into register
    if (shouldAccumulateInRegister(sg)) {
        os << model.getPrecision() << " linSyn = 0;" << std::endl;
    }
    else if(isSmallSharedMemoryPop(sg, backend)) {
        genSmallSharedMemoryPopPreamble(os, sg);
    }
}
//----------------------------------------------------------------------------
size_t PostSpan::getSharedMemoryPerThread(const SynapseGroupMerged &sg, const Backend &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PostSpan::genUpdate(CodeStream &os, const ModelSpecMerged &model, const SynapseGroupMerged &sg,
                         const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t,
                         BackendBase::SynapseGroupMergedHandler wumThreshHandler, BackendBase::SynapseGroupMergedHandler wumSimHandler,
                         BackendBase::SynapseGroupMergedHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "Evnt";

    os << "const unsigned int numSpikes = (*synapseGroup.spkCnt" << eventSuffix << ")";
    if (sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        os << "[preReadDelaySlot];" << std::endl;
    }
    else {
        os << "[0];" << std::endl;
    }
    os << "const unsigned int numSpikeBlocks = (numSpikes + " << backend.getKernelBlockSize(KernelPresynapticUpdate) << " - 1) / " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;


    const auto *wu = sg.getArchetype().getWUModel();
    os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(os);
        os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ") + 1 : " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

        os << "__syncthreads();" << std::endl;
        os << "if (threadIdx.x < numSpikesInBlock)";
        {
            CodeStream::Scope b(os);
            const std::string queueOffset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
            os << "const unsigned int spk = (*synapseGroup.spk" << eventSuffix << ")[" << queueOffset << "(r * " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ") + threadIdx.x];" << std::endl;
            os << "shSpk" << eventSuffix << "[threadIdx.x] = spk;" << std::endl;
            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                os << "shRowLength[threadIdx.x] = (*synapseGroup.rowLength)[spk];" << std::endl;
            }
        }
        os << "__syncthreads();" << std::endl;

        os << "// loop through all incoming spikes" << std::endl;
        os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(os);
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << popSubs["id"] << " < synapseGroup.rowStride)";
            {
                CodeStream::Scope b(os);
                if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    // Get maximum number of synapses anywhere in merged group
                    size_t maxSynapses = 0;
                    for(const auto &s : sg.getGroups()) {
                        maxSynapses = std::max(maxSynapses, (size_t)s.get().getTrgNeuronGroup()->getNumNeurons() * (size_t)s.get().getSrcNeuronGroup()->getNumNeurons());
                    }

                    // If this can only be represented using a 64-bit number
                    if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                        os << "const uint64_t gid = (shSpk" << eventSuffix << "[j] * (uint64_t)synapseGroup.rowStride) + " << popSubs["id"] << ";" << std::endl;
                    }
                    else {
                        os << "const unsigned int gid = (shSpk" << eventSuffix << "[j] * synapseGroup.rowStride) + " << popSubs["id"] << ";" << std::endl;
                    }
                }

                if (!wu->getSimSupportCode().empty()) {
                    os << "using namespace " << sg.getArchetype().getName() << "_weightupdate_simCode;" << std::endl;
                }
                if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << "if(";
                    if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // Note: we will just access global mem. For compute >= 1.2 simultaneous access to same global mem in the (half-)warp will be coalesced - no worries
                        os << "(B((*synapseGroup.gp)[gid / 32], gid & 31)) && ";
                    }

                    Substitutions threshSubs(&popSubs);
                    threshSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");

                    // Generate weight update threshold condition
                    wumThreshHandler(os, sg, threshSubs);

                    // end code substitutions ----
                    os << ")";
                    os << CodeStream::OB(130);
                }
                else if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "if (B((*synapseGroup.gp)[gid / 32], gid & 31))" << CodeStream::OB(135);
                }

                os << "const unsigned int synAddress = (shSpk" << eventSuffix << "[j] * synapseGroup.rowStride) + " + popSubs["id"] + ";" << std::endl;

                Substitutions synSubs(&popSubs);
                synSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");
                synSubs.addVarSubstitution("id_syn", "synAddress");

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    os << "const unsigned int npost = shRowLength[j];" << std::endl;

                    os << "if (" << popSubs["id"] << " < npost)" << CodeStream::OB(140);
                    os << "const unsigned int ipost = (*synapseGroup.ind)[synAddress];" << std::endl;

                    synSubs.addVarSubstitution("id_post", "ipost");
                }
                else { // DENSE
                    synSubs.addVarSubstitution("id_post", popSubs["id"]);
                }

                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                if(sg.getArchetype().isDendriticDelayRequired()) {
                    assert(false);
                    //synSubs.addFuncSubstitution("addToInSynDelay", 2, backend.getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + synSubs["id_post"] + "], $(0))");
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
                        synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&(*synapseGroup.inSyn)[" + synSubs["id_post"] + "], $(0))");
                    }
                }

                wumSimHandler(os, sg, synSubs);

                if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    os << CodeStream::CB(140); // end if (id < npost)
                }

                if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
                else if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << CodeStream::CB(135); // end if (B(dd_gp" << sg.getName() << "[gid / 32], gid
                }
            }
        }
    }
}
//----------------------------------------------------------------------------
void PostSpan::genPostamble(CodeStream &os, const ModelSpecMerged &model, const SynapseGroupMerged &sg,
                            const Substitutions &popSubs, const Backend &backend, size_t) const
{
    // If we should accumulate output directly into register
    if (shouldAccumulateInRegister(sg)) {
        os << "// only do this for existing neurons" << std::endl;
        os << "if (" << popSubs["id"] << " < synapseGroup.numTrgNeurons)";
        {
            CodeStream::Scope b(os);
            const std::string inSyn = "(*synapseGroup.inSyn)[" + popSubs["id"] + "]";
            if (sg.getArchetype().isPSModelMerged()) {
                os << backend.getFloatAtomicAdd(model.getPrecision()) << "(&" << inSyn << ", linSyn);" << std::endl;
            }
            else {
                os << inSyn << " += linSyn;" << std::endl;
            }
        }
    }
    // Otherwise, if we should accumulate into shared memory
    else if (isSmallSharedMemoryPop(sg, backend)) {
        genSmallSharedMemoryPopPostamble(os, model, sg, backend);
    }
}
// ----------------------------------------------------------------------------
bool PostSpan::shouldAccumulateInRegister(const SynapseGroupMerged &sg) const
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
bool PreSpanProcedural::isCompatible(const SynapseGroupMerged &sg, const cudaDeviceProp &, const Preferences &) const
{
    // Presynaptic procedural parallelism can be used when synapse groups have 
    // procedural connectivity and weights are either GLOBAL or PROCEDURAL
    const auto matrixType = sg.getArchetype().getMatrixType();
    return ((matrixType & SynapseMatrixConnectivity::PROCEDURAL)
            && ((matrixType & SynapseMatrixWeight::GLOBAL) || (matrixType & SynapseMatrixWeight::PROCEDURAL)));
}
//----------------------------------------------------------------------------
size_t PreSpanProcedural::getSharedMemoryPerThread(const SynapseGroupMerged &sg, const Backend &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genPreamble(CodeStream &os, const ModelSpecMerged &, const SynapseGroupMerged &sg,
                                    const Substitutions &, const Backend &backend, size_t) const
{
    if (isSmallSharedMemoryPop(sg, backend)) {
        genSmallSharedMemoryPopPreamble(os, sg);
    }
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genUpdate(CodeStream &os, const ModelSpecMerged &model, const SynapseGroupMerged &sg,
                                  const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t idStart,
                                  BackendBase::SynapseGroupMergedHandler wumThreshHandler, BackendBase::SynapseGroupMergedHandler wumSimHandler,
                                  BackendBase::SynapseGroupMergedHandler wumProceduralConnectHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const auto *wu = sg.getArchetype().getWUModel();
    const size_t numThreadsPerSpike = sg.getArchetype().getNumThreadsPerSpike();

    if(numThreadsPerSpike > 1) {
        os << "const unsigned int spike = " << popSubs["id"] << " / " << numThreadsPerSpike << ";" << std::endl;
        os << "const unsigned int thread = " << popSubs["id"] << " % " << numThreadsPerSpike << ";" << std::endl;
        os << "const unsigned int numPostPerThread =  (synapseGroup.numTrgNeurons + " << numThreadsPerSpike << " - 1) / " << numThreadsPerSpike << ";" << std::endl;

        // Calculate the starting position and length of the sub-row to process on this thread
        os << "const unsigned int idPostStart = thread * numPostPerThread;" << std::endl;
        os << "const unsigned int numPost = (thread == " << (numThreadsPerSpike - 1) << ") ? (synapseGroup.numTrgNeurons % numPostPerThread) : numPostPerThread;" << std::endl;
    }
    else {
        os << "const unsigned int spike = " << popSubs["id"] << ";" << std::endl;
    }

    // If there is a spike for this thread to process
    os << "if (spike < (*synapseGroup.spkCnt" << eventSuffix << ")" ;
    if (sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        os << "[preReadDelaySlot])";
    }
    else {
        os << "[0])";
    }
    {
        CodeStream::Scope b(os);

        // Determine the index of the presynaptic neuron this thread is responsible for
        os << "const unsigned int preInd = (*synapseGroup.spk"  << eventSuffix << ")";
        if (sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {

            os << "[(preReadDelaySlot * synapseGroup.numSrcNeurons) + spike];" << std::endl;
        }
        else {
            os << "[spike];" << std::endl;
        }

        // Create substitution stack and add presynaptic index
        Substitutions synSubs(&popSubs);
        synSubs.addVarSubstitution("id_pre", "preInd");

        if (!wu->getSimSupportCode().empty()) {
            os << "using namespace " << sg.getArchetype().getName() << "_weightupdate_simCode;" << std::endl;
        }


        if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
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
            // Only start using streams after those that may have been used for initialisation
            const size_t rngStreamOffset = idStart + backend.getNumInitialisationRNGStreams(model);

            // Get global RNG and skip ahead to subsequence unique to this subrow of this presynaptic neuron
            os << "curandStatePhilox4_32_10_t connectRNG = dd_rng[0];" << std::endl;
            os << "skipahead_sequence((unsigned long long)(";
            if(numThreadsPerSpike > 1) {
                os << "(preInd * " << numThreadsPerSpike << ") + thread + " << rngStreamOffset;
            }
            else {
                os << "preInd + " << rngStreamOffset;
            }
            os << "), &connectRNG);" << std::endl;

            // Add substitution for connection generation code
            connSubs.addVarSubstitution("rng", "&connectRNG");
        }

        // If we are using more than one thread to process each row
        if(numThreadsPerSpike > 1) {
            connSubs.addVarSubstitution("id_post_begin", "idPostStart");
            connSubs.addVarSubstitution("id_thread", "thread");
            connSubs.addVarSubstitution("num_post", "numPost");
        }
        else {
            connSubs.addVarSubstitution("id_post_begin", "0");
            connSubs.addVarSubstitution("id_thread", "0");
            connSubs.addVarSubstitution("num_post", "synapseGroup.numTrgNeurons");
        }

        // Create another substitution stack for generating presynaptic simulation code
        Substitutions presynapticUpdateSubs(&synSubs);

        // If this synapse group has procedural connectivity and any of it's variables require an RNG
        if((sg.getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL)
           && ::Utils::isRNGRequired(sg.getArchetype().getWUVarInitialisers()))
        {
            presynapticUpdateSubs.addVarSubstitution("rng", "&connectRNG");
        }

        // Replace $(id_post) with first 'function' parameter as simulation code is
        // going to be, in turn, substituted into procedural connectivity generation code
        presynapticUpdateSubs.addVarSubstitution("id_post", "$(0)");

        // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
        if(sg.getArchetype().isDendriticDelayRequired()) {
            assert(false);
            //presynapticUpdateSubs.addFuncSubstitution("addToInSynDelay", 2, backend.getFloatAtomicAdd(model.getPrecision()) + "(&(*synapseGroup.denDelay)[" + sg.getDendriticDelayOffset("dd_", "$(1)") + "$(id_post)], $(0))");
        }
        // Otherwise
        else {
            // If postsynaptic input should be accumulated in shared memory, substitute shared memory array for $(inSyn)
            if(isSmallSharedMemoryPop(sg, backend)) {
                presynapticUpdateSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&shLg[$(id_post)], $(0))");
            }
            // Otherwise, substitute global memory array for $(inSyn)
            else {
                presynapticUpdateSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&(*synapseGroup.inSyn)[$(id_post)], $(0))");
            }
        }

        // Generate presynaptic simulation code into new stringstream-backed code stream
        std::ostringstream presynapticUpdateStream;
        CodeStream presynapticUpdate(presynapticUpdateStream);
        wumSimHandler(presynapticUpdate, sg, presynapticUpdateSubs);

        // When a synapse should be 'added', substitute in presynaptic update code
        connSubs.addFuncSubstitution("addSynapse", 1, presynapticUpdateStream.str());

        // Generate procedural connectivity code
        wumProceduralConnectHandler(os, sg, connSubs);

        if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }
    }
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genPostamble(CodeStream &os, const ModelSpecMerged &model, const SynapseGroupMerged &sg,
                                     const Substitutions &, const Backend &backend, size_t) const
{
    if (isSmallSharedMemoryPop(sg, backend)) {
        genSmallSharedMemoryPopPostamble(os, model, sg, backend);
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpanBitmask
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getNumThreads(const SynapseGroupInternal & sg) const
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
bool PostSpanBitmask::isCompatible(const SynapseGroupMerged &sg, const cudaDeviceProp &, const Preferences &preferences) const
{
    // Postsynaptic bitmask parallelism can be used if bitmask optimisations are enabled and
    // if synapse groups with bitmask connectivity and no dendritic delays request postsynaptic parallelism
    return (preferences.enableBitmaskOptimisations
            && (sg.getArchetype().getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC)
            && (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK)
            && !sg.getArchetype().isDendriticDelayRequired());
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genPreamble(CodeStream &os, const ModelSpecMerged &, const SynapseGroupMerged &,
                                  const Substitutions &, const Backend &backend, size_t) const
{
    // Loop through bits written by this thread
    for (size_t i = 0; i < 32; i++) {
        // Zero entries in this thread's shared memory array
        // **NOTE** this is ordered to prevent bank conflicts
        const std::string index = std::to_string(i * backend.getKernelBlockSize(KernelPresynapticUpdate)) + " + threadIdx.x";
        os << "shLg[" << index << "] = 0;" << std::endl;
    }
    os << "__syncthreads();" << std::endl;
}
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getSharedMemoryPerThread(const SynapseGroupMerged &, const Backend &) const
{
    // Each thread sums up the input to 32 postsynaptic neurons
    return 32;
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genUpdate(CodeStream &os, const ModelSpecMerged &, const SynapseGroupMerged &sg,
                                const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t,
                                BackendBase::SynapseGroupMergedHandler wumThreshHandler, BackendBase::SynapseGroupMergedHandler wumSimHandler,
                                BackendBase::SynapseGroupMergedHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "Evnt";

    // Get blocksize
    const size_t blockSize = backend.getKernelBlockSize(KernelPresynapticUpdate);

    os << "const unsigned int numSpikes = (*synapseGroup.spkCnt" << eventSuffix << ")";
    if (sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        os << "[preReadDelaySlot];" << std::endl;
    }
    else {
        os << "[0];" << std::endl;
    }
    os << "const unsigned int numSpikeBlocks = (numSpikes + " << blockSize << " - 1) / " << blockSize << ";" << std::endl;


    const auto *wu = sg.getArchetype().getWUModel();
    os << "const unsigned int rowWords =  (synapseGroup.numTrgNeurons + 32 - 1) / 32;" << std::endl;
    os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(os);
        os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << blockSize << ") + 1 : " << blockSize << ";" << std::endl;

        os << "__syncthreads();" << std::endl;
        os << "if (threadIdx.x < numSpikesInBlock)";
        {
            CodeStream::Scope b(os);
            const std::string queueOffset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
            os << "const unsigned int spk = (*synapseGroup.spk" << eventSuffix << ")[" << queueOffset << "(r * " << blockSize << ") + threadIdx.x];" << std::endl;
            os << "shSpk" << eventSuffix << "[threadIdx.x] = spk;" << std::endl;
        }
        os << "__syncthreads();" << std::endl;

        os << "// loop through all incoming spikes" << std::endl;
        os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(os);
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << popSubs["id"] << " < rowWords)";
            {
                CodeStream::Scope b(os);

                if (!wu->getSimSupportCode().empty()) {
                    os << "using namespace " << sg.getArchetype().getName() << "_weightupdate_simCode;" << std::endl;
                }
                if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << "if(";

                    Substitutions threshSubs(&popSubs);
                    threshSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");

                    // Generate weight update threshold condition
                    wumThreshHandler(os, sg, threshSubs);

                    // end code substitutions ----
                    os << ")";
                    os << CodeStream::OB(130);
                }

                // Read row word
                os << "uint32_t connectivityWord = (*synapseGroup.gp)[(shSpk" << eventSuffix << "[j] * rowWords) + " << popSubs["id"] << "];" << std::endl;

                // While there any bits left
                os << "unsigned int ibit = 0;" << std::endl;
                os << "while(connectivityWord != 0)";
                {
                    CodeStream::Scope b(os);

                    // Cound leading zeros (as bits are indexed backwards this is index of next synapse)
                    os << "const int numLZ = __clz(connectivityWord);" << std::endl;

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
                    synSubs.addFuncSubstitution("addToInSyn", 1, "shLg[(ibit * " + std::to_string(blockSize) + ") + threadIdx.x] += $(0)");
                    wumSimHandler(os, sg, synSubs);

                    os << "ibit++;" << std::endl;
                }


                if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
            }
        }
    }
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genPostamble(CodeStream &os, const ModelSpecMerged &model, const SynapseGroupMerged &sg,
                                   const Substitutions &, const Backend &backend, size_t idStart) const
{
    os << "__syncthreads();" << std::endl;
    const size_t blockSize = backend.getKernelBlockSize(KernelPresynapticUpdate);

    // Use first 32 threads in each block to write shared memory back to global memory
    os << "if (threadIdx.x < 32)";
    {
        CodeStream::Scope b(os);
        os << "unsigned int glbIdx = ((blockIdx.x - " << idStart / blockSize << ") * " << 32 * blockSize << ") + threadIdx.x;" << std::endl;
        os << "unsigned int shIdx = threadIdx.x * " << blockSize << ";" << std::endl;
        os << "const unsigned int endShIdx = shIdx + 32;" << std::endl;
        os << "for(;shIdx < endShIdx && glbIdx < synapseGroup.numTrgNeurons; shIdx++, glbIdx += 32)";
        {
            CodeStream::Scope b(os);
            const std::string inSyn = "(*synapseGroup.inSyn)[glbIdx]";
            if (sg.getArchetype().isPSModelMerged()) {
                os << backend.getFloatAtomicAdd(model.getPrecision()) << "(&" << inSyn << ", shLg[shIdx]);" << std::endl;
            }
            else {
                os << inSyn << " += shLg[shIdx];" << std::endl;
            }
        }
    }
}
}   // namespace PresynapticUpdateStrategy
}   // namespace CUDA
}   // namespace CodeGenerator

#include "presynapticUpdateStrategy.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
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
bool isSmallSharedMemoryPop(const SynapseGroupInternal &sg, const CodeGenerator::CUDA::Backend &backend)
{
    // If device is older than Maxwell, we shouldn't use shared memory as atomics are emulated
    // and actually slower than global memory (see https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
    if(backend.getChosenCUDADevice().major < 5) {
        return false;
    }
    // Otherwise, if dendritic delays are required, shared memory approach cannot be used so return false
    else if(sg.isDendriticDelayRequired()) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if the output
    // population is small enough that input to it can be stored in a shared memory array
    else if(sg.getTrgNeuronGroup()->getNumNeurons() <= backend.getKernelBlockSize(CodeGenerator::CUDA::KernelPresynapticUpdate)) {
        return true;
    }
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
void genSmallSharedMemoryPopPreamble(CodeGenerator::CodeStream &os, const SynapseGroupInternal &sg)
{
    os << "if(threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
    {
        CodeGenerator::CodeStream::Scope b(os);
        os << "shLg[threadIdx.x] = 0;" << std::endl;
    }
    os << "__syncthreads();" << std::endl;
}
//----------------------------------------------------------------------------
void genSmallSharedMemoryPopPostamble(CodeGenerator::CodeStream &os, const ModelSpecInternal &model,
                                      const SynapseGroupInternal &sg, const CodeGenerator::CUDA::Backend &backend)
{
    os << "__syncthreads();" << std::endl;
    os << "if (threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
    {
        CodeGenerator::CodeStream::Scope b(os);
        const std::string inSyn = "dd_inSyn" + sg.getPSModelTargetName() + "[threadIdx.x]";
        if (sg.isPSModelMerged()) {
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
    // Use a thread for each presynaptic neuron
    // **YUCK** really should only launch a thread per-spike
    return sg.getSrcNeuronGroup()->getNumNeurons() * sg.getNumThreadsPerSpike();
}
//----------------------------------------------------------------------------
size_t PreSpan::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
bool PreSpan::isCompatible(const SynapseGroupInternal &sg) const
{
    // Presynaptic parallelism can be used when synapse groups request it and they have sparse connectivity
    return (sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) && (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE);
}
//----------------------------------------------------------------------------
size_t PreSpan::getSharedMemoryPerThread(const SynapseGroupInternal &sg, const Backend &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PreSpan::genPreamble(CodeStream &os, const ModelSpecInternal &, const SynapseGroupInternal &sg,
                          const Substitutions &, const Backend &backend, size_t) const
{
    if (isSmallSharedMemoryPop(sg, backend)) {
        genSmallSharedMemoryPopPreamble(os, sg);
    }
}
//----------------------------------------------------------------------------
void PreSpan::genUpdate(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, 
                        const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t,
                        BackendBase::SynapseGroupHandler wumThreshHandler, BackendBase::SynapseGroupHandler wumSimHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const auto *wu = sg.getWUModel();

    if(sg.getNumThreadsPerSpike() > 1) {
        os << "const unsigned int spike = " << popSubs["id"] << " / " << sg.getNumThreadsPerSpike() << ";" << std::endl;
        os << "const unsigned int thread = " << popSubs["id"] << " % " << sg.getNumThreadsPerSpike() << ";" << std::endl;
    }
    else {
        os << "const unsigned int spike = " << popSubs["id"] << ";" << std::endl;
    }

    os << "if (spike < " ;
    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "dd_glbSpkCnt" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[preReadDelaySlot])";
    }
    else {
        os << "dd_glbSpkCnt" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[0])";
    }
    {
        CodeStream::Scope b(os);

        if (!wu->getSimSupportCode().empty()) {
            os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
        }

        if (sg.getSrcNeuronGroup()->isDelayRequired()) {
            os << "const unsigned int preInd = dd_glbSpk"  << eventSuffix << sg.getSrcNeuronGroup()->getName();
            os << "[(preReadDelaySlot * " << sg.getSrcNeuronGroup()->getNumNeurons() << ") + spike];" << std::endl;
        }
        else {
            os << "const unsigned int preInd = dd_glbSpk"  << eventSuffix << sg.getSrcNeuronGroup()->getName();
            os << "[spike];" << std::endl;
        }

        if(sg.getNumThreadsPerSpike() > 1) {
            os << "unsigned int synAddress = (preInd * " << std::to_string(backend.getSynapticMatrixRowStride(sg)) << ") + thread;" << std::endl;
        }
        else {
            os << "unsigned int synAddress = preInd * " << std::to_string(backend.getSynapticMatrixRowStride(sg)) << ";" << std::endl;
        }
        os << "const unsigned int npost = dd_rowLength" << sg.getName() << "[preInd];" << std::endl;

        if (!trueSpike && sg.isEventThresholdReTestRequired()) {
            os << "if(";

            Substitutions threshSubs(&popSubs);
            threshSubs.addVarSubstitution("id_pre", "preInd");

            // Generate weight update threshold condition
            wumThreshHandler(os, sg, threshSubs);

            // end code substitutions ----
            os << ")";

            os << CodeStream::OB(130);
        }

        if(sg.getNumThreadsPerSpike() > 1) {
            os << "for(unsigned int i = thread; i < npost; i += " << sg.getNumThreadsPerSpike() << ", synAddress += " << sg.getNumThreadsPerSpike() << ")";
        }
        else {
            os << "for(unsigned int i = 0; i < npost; i++, synAddress++)";
        }
        {
            CodeStream::Scope b(os);

            // **TODO** pretty sure __ldg will boost performance here - basically will bring whole row into cache
            os << "const unsigned int ipost = dd_ind" <<  sg.getName() << "[synAddress];" << std::endl;

            // Code substitutions ----------------------------------------------------------------------------------
            std::string wCode = trueSpike ? wu->getSimCode() : wu->getEventCode();

            Substitutions synSubs(&popSubs);
            synSubs.addVarSubstitution("id_pre", "preInd");
            synSubs.addVarSubstitution("id_post", "ipost");
            synSubs.addVarSubstitution("id_syn", "synAddress");

            // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
            if(sg.isDendriticDelayRequired()) {
                synSubs.addFuncSubstitution("addToInSynDelay", 2, backend.getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + "ipost], $(0))");
            }
            // Otherwise
            else {
                // If postsynaptic input should be accumulated in shared memory, substitute shared memory array for $(inSyn)
                if(isSmallSharedMemoryPop(sg, backend)) {
                    synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&shLg[ipost], $(0))");
                }
                // Otherwise, substitute global memory array for $(inSyn)
                else {
                    synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[ipost], $(0))");
                }
            }

            wumSimHandler(os, sg, synSubs);
        }

        if (!trueSpike && sg.isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }
    }
}
//----------------------------------------------------------------------------
void PreSpan::genPostamble(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg,
                           const Substitutions &, const Backend &backend, size_t) const
{
    if (isSmallSharedMemoryPop(sg, backend)) {
        genSmallSharedMemoryPopPostamble(os, model, sg, backend);
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PreSpanBitmask
//----------------------------------------------------------------------------
size_t PreSpanBitmask::getNumThreads(const SynapseGroupInternal &sg) const
{
    // Use a thread for each presynaptic neuron
    // **YUCK** really should only launch a thread per-spike
    return sg.getSrcNeuronGroup()->getNumNeurons() * sg.getNumThreadsPerSpike();
}
//----------------------------------------------------------------------------
size_t PreSpanBitmask::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    // Pad each row to a word boundary
    return Utils::padSize(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
}
//----------------------------------------------------------------------------
bool PreSpanBitmask::isCompatible(const SynapseGroupInternal &sg) const
{
    // Presynaptic parallelism can be used when synapse groups request it and they have bitmask connectivity
    return (sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) && (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK);
}
//----------------------------------------------------------------------------
size_t PreSpanBitmask::getSharedMemoryPerThread(const SynapseGroupInternal &sg, const Backend &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PreSpanBitmask::genPreamble(CodeStream &os, const ModelSpecInternal &, const SynapseGroupInternal &sg,
                                 const Substitutions &, const Backend &backend, size_t) const
{
    if (isSmallSharedMemoryPop(sg, backend)) {
        genSmallSharedMemoryPopPreamble(os, sg);
    }
}
//----------------------------------------------------------------------------
void PreSpanBitmask::genUpdate(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, 
                               const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t,
                               BackendBase::SynapseGroupHandler wumThreshHandler, BackendBase::SynapseGroupHandler wumSimHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const auto *wu = sg.getWUModel();

    if (sg.getNumThreadsPerSpike() > 1) {
        os << "const unsigned int spike = " << popSubs["id"] << " / " << sg.getNumThreadsPerSpike() << ";" << std::endl;
        os << "const unsigned int thread = " << popSubs["id"] << " % " << sg.getNumThreadsPerSpike() << ";" << std::endl;
    }
    else {
        os << "const unsigned int spike = " << popSubs["id"] << ";" << std::endl;
    }

    os << "if (spike < ";
    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "dd_glbSpkCnt" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[preReadDelaySlot])";
    }
    else {
        os << "dd_glbSpkCnt" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[0])";
    }
    {
        CodeStream::Scope b(os);

        if (!wu->getSimSupportCode().empty()) {
            os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
        }

        if (sg.getSrcNeuronGroup()->isDelayRequired()) {
            os << "const unsigned int preInd = dd_glbSpk" << eventSuffix << sg.getSrcNeuronGroup()->getName();
            os << "[(preReadDelaySlot * " << sg.getSrcNeuronGroup()->getNumNeurons() << ") + spike];" << std::endl;
        }
        else {
            os << "const unsigned int preInd = dd_glbSpk" << eventSuffix << sg.getSrcNeuronGroup()->getName();
            os << "[spike];" << std::endl;
        }

/*if (sg.getNumThreadsPerSpike() > 1) {
            os << "unsigned int synAddress = (preInd * " << std::to_string(backend.getSynapticMatrixRowStride(sg)) << ") + thread;" << std::endl;
        }
        else {
            os << "unsigned int synAddress = preInd * " << std::to_string(backend.getSynapticMatrixRowStride(sg)) << ";" << std::endl;
        }*/

        // Determine the number of words in each row
        const size_t rowWords = Utils::ceilDivide(sg.getTrgNeuronGroup()->getNumNeurons(), 32);

        // Determine the number of words to process with each thread
        //const size_t numWordsPerSpike = Utils::ceilDivide(rowWords, sg.getNumThreadsPerSpike());

        if (!trueSpike && sg.isEventThresholdReTestRequired()) {
            os << "if(";

            Substitutions threshSubs(&popSubs);
            threshSubs.addVarSubstitution("id_pre", "preInd");

            // Generate weight update threshold condition
            wumThreshHandler(os, sg, threshSubs);

            // end code substitutions ----
            os << ")";

            os << CodeStream::OB(130);
        }

        // Create outer loop through words in rows
        // **NOTE** if multiple threads are used, this will results in coalesced reads
        if (sg.getNumThreadsPerSpike() > 1) {
            os << "for(unsigned int w = thread; w < " << rowWords << "; w += " << sg.getNumThreadsPerSpike() << ")";
        }
        else {
            os << "for(unsigned int w = 0; w < " << rowWords << "; w++)";
        }
        {
            CodeStream::Scope b(os);

            // Read row word
            os << "uint32_t connectivityWord = dd_gp" << sg.getName() << "[(preInd * " << rowWords << ") + w];" << std::endl;
            
            // Set ipost to first synapse in connectivity word
            os << "unsigned int ipost = w * 32;" << std::endl;

            // While there any bits left
            os << "while(connectivityWord != 0)";
            {
                CodeStream::Scope b(os);

                // Cound leading zeros (as bits are indexed backwards this is index of next synapse)
                os << "const int numLZ = __clz(connectivityWord);" << std::endl;

                // Shift off zeros and the one just discovered
                os << "connectivityWord <<= (numLZ + 1);" << std::endl;

                // Add to ipost
                os << "ipost += numLZ;" << std::endl;

                // If we aren't in padding region
                // **TODO** don't bother checking if there is no padding
                os << "if(ipost < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);
                    // Code substitutions ----------------------------------------------------------------------------------
                    std::string wCode = trueSpike ? wu->getSimCode() : wu->getEventCode();

                    Substitutions synSubs(&popSubs);
                    synSubs.addVarSubstitution("id_pre", "preInd");
                    synSubs.addVarSubstitution("id_post", "ipost");
                    //synSubs.addVarSubstitution("id_syn", "synAddress");

                    // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                    if (sg.isDendriticDelayRequired()) {
                        synSubs.addFuncSubstitution("addToInSynDelay", 2, backend.getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + "ipost], $(0))");
                    }
                    // Otherwise
                    else {
                        // If postsynaptic input should be accumulated in shared memory, substitute shared memory array for $(inSyn)
                        if (getSharedMemoryPerThread(sg, backend) > 0) {
                            synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&shLg[ipost], $(0))");
                        }
                        // Otherwise, substitute global memory array for $(inSyn)
                        else {
                            synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[ipost], $(0))");
                        }
                    }

                    wumSimHandler(os, sg, synSubs);
                }

                // Increment ipost to take into account fact the next CLZ will go from bit AFTER synapse
                os << "ipost++;" << std::endl;
            }
        }

        if (!trueSpike && sg.isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }
    }
}
//----------------------------------------------------------------------------
void PreSpanBitmask::genPostamble(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg,
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
bool PostSpan::isCompatible(const SynapseGroupInternal &sg) const
{
    // Postsynatic parallelism can be used when synapse groups request it
    return (sg.getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC);
}
//----------------------------------------------------------------------------
void PostSpan::genPreamble(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg,
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
size_t PostSpan::getSharedMemoryPerThread(const SynapseGroupInternal &sg, const Backend &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PostSpan::genUpdate(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, 
                         const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t,
                         BackendBase::SynapseGroupHandler wumThreshHandler, BackendBase::SynapseGroupHandler wumSimHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "Evnt";

    os << "const unsigned int numSpikes = dd_glbSpkCnt" << eventSuffix << sg.getSrcNeuronGroup()->getName();
    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "[preReadDelaySlot];" << std::endl;
    }
    else {
        os << "[0];" << std::endl;
    }
    os << "const unsigned int numSpikeBlocks = (numSpikes + " << backend.getKernelBlockSize(KernelPresynapticUpdate) << " - 1) / " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;


    const auto *wu = sg.getWUModel();
    os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(os);
        os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ") + 1 : " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

        os << "__syncthreads();" << std::endl;
        os << "if (threadIdx.x < numSpikesInBlock)";
        {
            CodeStream::Scope b(os);
            const std::string queueOffset = sg.getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
            os << "const unsigned int spk = dd_glbSpk" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[" << queueOffset << "(r * " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ") + threadIdx.x];" << std::endl;
            os << "shSpk" << eventSuffix << "[threadIdx.x] = spk;" << std::endl;
            if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                os << "shRowLength[threadIdx.x] = dd_rowLength" << sg.getName() << "[spk];" << std::endl;
            }
        }
        os << "__syncthreads();" << std::endl;

        os << "// loop through all incoming spikes" << std::endl;
        os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(os);
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << popSubs["id"] << " < " << sg.getMaxConnections() << ")";
            {
                CodeStream::Scope b(os);
                if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    const size_t maxSynapses = (size_t)sg.getTrgNeuronGroup()->getNumNeurons() * (size_t)sg.getSrcNeuronGroup()->getNumNeurons();
                    if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                        os << "const uint64_t gid = (shSpk" << eventSuffix << "[j] * " << sg.getTrgNeuronGroup()->getNumNeurons() << "ull + " << popSubs["id"] << ");" << std::endl;
                    }
                    else {
                        os << "const unsigned int gid = (shSpk" << eventSuffix << "[j] * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + " << popSubs["id"] << ");" << std::endl;
                    }
                }

                if (!wu->getSimSupportCode().empty()) {
                    os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
                }
                if (!trueSpike && sg.isEventThresholdReTestRequired()) {
                    os << "if(";
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // Note: we will just access global mem. For compute >= 1.2 simultaneous access to same global mem in the (half-)warp will be coalesced - no worries
                        os << "(B(dd_gp" << sg.getName() << "[gid / 32], gid & 31)) && ";
                    }

                    Substitutions threshSubs(&popSubs);
                    threshSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");

                    // Generate weight update threshold condition
                    wumThreshHandler(os, sg, threshSubs);

                    // end code substitutions ----
                    os << ")";
                    os << CodeStream::OB(130);
                }
                else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "if (B(dd_gp" << sg.getName() << "[gid / 32], gid & 31))" << CodeStream::OB(135);
                }

                os << "const unsigned int synAddress = (shSpk" << eventSuffix << "[j] * " << std::to_string(backend.getSynapticMatrixRowStride(sg)) << ") + " + popSubs["id"] + ";" << std::endl;

                Substitutions synSubs(&popSubs);
                synSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");
                synSubs.addVarSubstitution("id_syn", "synAddress");

                if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {

                    os << "const unsigned int npost = shRowLength[j];" << std::endl;

                    os << "if (" << popSubs["id"] << " < npost)" << CodeStream::OB(140);
                    os << "const unsigned int ipost = dd_ind" << sg.getName() << "[synAddress];" << std::endl;

                    synSubs.addVarSubstitution("id_post", "ipost");
                }
                else { // DENSE
                    synSubs.addVarSubstitution("id_post", popSubs["id"]);
                }

                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                if(sg.isDendriticDelayRequired()) {
                    synSubs.addFuncSubstitution("addToInSynDelay", 2, backend.getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + synSubs["id_post"] + "], $(0))");
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
                        synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[" + synSubs["id_post"] + "], $(0))");
                    }
                }

                wumSimHandler(os, sg, synSubs);

                if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    os << CodeStream::CB(140); // end if (id < npost)
                }

                if (!trueSpike && sg.isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
                else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << CodeStream::CB(135); // end if (B(dd_gp" << sg.getName() << "[gid / 32], gid
                }
            }
        }
    }
}
//----------------------------------------------------------------------------
void PostSpan::genPostamble(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg,
                            const Substitutions &popSubs, const Backend &backend, size_t) const
{
    // If we should accumulate output directly into register
    if (shouldAccumulateInRegister(sg)) {
        os << "// only do this for existing neurons" << std::endl;
        os << "if (" << popSubs["id"] << " < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
        {
            CodeStream::Scope b(os);
            const std::string inSyn = "dd_inSyn" + sg.getPSModelTargetName() + "[" + popSubs["id"] + "]";
            if (sg.isPSModelMerged()) {
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
bool PostSpan::shouldAccumulateInRegister(const SynapseGroupInternal &sg) const
{
    // If no dendritic delays are required and data structure is dense, we can accumulate output directly into register
    return (!sg.isDendriticDelayRequired()
            && ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) || (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK)));
}

// ----------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpanBitmask
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getNumThreads(const SynapseGroupInternal & sg) const
{
    return Utils::ceilDivide(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
}
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    // Pad each row to a word boundary
    return Utils::padSize(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
}
//----------------------------------------------------------------------------
bool PostSpanBitmask::isCompatible(const SynapseGroupInternal &sg) const
{
    // Postsynatic parallelism can be used when synapse groups request it
    return ((sg.getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC) 
            && (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK)
            && !sg.isDendriticDelayRequired());
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genPreamble(CodeStream &os, const ModelSpecInternal &, const SynapseGroupInternal &,
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
size_t PostSpanBitmask::getSharedMemoryPerThread(const SynapseGroupInternal &, const Backend &) const
{
    // Each thread sums up the input to 32 postsynaptic neurons
    return 32;
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genUpdate(CodeStream &os, const ModelSpecInternal &, const SynapseGroupInternal &sg,
                         const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t,
                         BackendBase::SynapseGroupHandler wumThreshHandler, BackendBase::SynapseGroupHandler wumSimHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "Evnt";

    // Get blocksize
    const size_t blockSize = backend.getKernelBlockSize(KernelPresynapticUpdate);

    os << "const unsigned int numSpikes = dd_glbSpkCnt" << eventSuffix << sg.getSrcNeuronGroup()->getName();
    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "[preReadDelaySlot];" << std::endl;
    }
    else {
        os << "[0];" << std::endl;
    }
    os << "const unsigned int numSpikeBlocks = (numSpikes + " << blockSize << " - 1) / " << blockSize << ";" << std::endl;


    const auto *wu = sg.getWUModel();
    os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(os);
        os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << blockSize << ") + 1 : " << blockSize << ";" << std::endl;

        os << "__syncthreads();" << std::endl;
        os << "if (threadIdx.x < numSpikesInBlock)";
        {
            CodeStream::Scope b(os);
            const std::string queueOffset = sg.getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
            os << "const unsigned int spk = dd_glbSpk" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[" << queueOffset << "(r * " << blockSize << ") + threadIdx.x];" << std::endl;
            os << "shSpk" << eventSuffix << "[threadIdx.x] = spk;" << std::endl;
        }
        os << "__syncthreads();" << std::endl;

        os << "// loop through all incoming spikes" << std::endl;
        os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(os);
            const size_t rowWords = Utils::ceilDivide(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << popSubs["id"] << " < " << rowWords << ")";
            {
                CodeStream::Scope b(os);

                if (!wu->getSimSupportCode().empty()) {
                    os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
                }
                if (!trueSpike && sg.isEventThresholdReTestRequired()) {
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
                os << "uint32_t connectivityWord = dd_gp" << sg.getName() << "[(shSpk" << eventSuffix << "[j] * " << rowWords << ") + " << popSubs["id"] << "];" << std::endl;

                // While there any bits left
                os << "unsigned int ibit = 0;" << std::endl;
                os << "while(connectivityWord != 0)";
                {
                    CodeStream::Scope b(os);

                    // Cound leading zeros (as bits are indexed backwards this is index of next synapse)
                    os << "const int numLZ = __clz(connectivityWord);" << std::endl;

                    // Shift off zeros and the one just discovered
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


                if (!trueSpike && sg.isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
            }
        }
    }
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genPostamble(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg,
                                   const Substitutions &popSubs, const Backend &backend, size_t idStart) const
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
        os << "for(;shIdx < endShIdx && glbIdx < " << sg.getTrgNeuronGroup()->getNumNeurons() << "; shIdx++, glbIdx += 32)";
        {
            CodeStream::Scope b(os);
            const std::string inSyn = "dd_inSyn" + sg.getPSModelTargetName() + "[glbIdx]";
            if (sg.isPSModelMerged()) {
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

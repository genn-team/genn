#include "presynapticUpdateStrategy.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/substitutions.h"

// CUDA backend includes
#include "backend.h"

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
bool PreSpan::isCompatible(const SynapseGroupInternal &sg) const
{
    // Presynaptic parallelism can be used when synapse groups request it and they have sparse connectivity
    return (sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) && (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE);
}
//----------------------------------------------------------------------------
bool PreSpan::shouldAccumulateInRegister(const SynapseGroupInternal &, const Backend &) const
{
    // When presynaptic parallelism is used
    return false;
}
//----------------------------------------------------------------------------
bool PreSpan::shouldAccumulateInSharedMemory(const SynapseGroupInternal &sg, const Backend &backend) const
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
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if matrix is sparse
    // and the output population is small enough that input to it can be stored in a shared memory array
    else {
        return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE)
                && sg.getTrgNeuronGroup()->getNumNeurons() <= backend.getKernelBlockSize(KernelPresynapticUpdate));
    }
}
//----------------------------------------------------------------------------
void PreSpan::genCode(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, const Substitutions &popSubs, const Backend &backend, bool trueSpike,
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
            os << "unsigned int synAddress = (preInd * " << std::to_string(sg.getMaxConnections()) << ") + thread;" << std::endl;
        }
        else {
            os << "unsigned int synAddress = preInd * " << std::to_string(sg.getMaxConnections()) << ";" << std::endl;
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
                if(shouldAccumulateInSharedMemory(sg, backend)) {
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
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpan
//----------------------------------------------------------------------------
size_t PostSpan::getNumThreads(const SynapseGroupInternal &sg) const
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
bool PostSpan::shouldAccumulateInRegister(const SynapseGroupInternal &sg, const Backend &) const
{
    // We should accumulate each postsynaptic neuron's input in a register if matrix is dense or bitfield
    // (where each thread represents an individual neuron)
    return ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE)
            || (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK));
}
//----------------------------------------------------------------------------
bool PostSpan::shouldAccumulateInSharedMemory(const SynapseGroupInternal &sg, const Backend &backend) const
{
    // If dendritic delays are required, shared memory approach cannot be used so return false
    if(sg.isDendriticDelayRequired()) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if matrix is sparse
    // and the output population is small enough that input to it can be stored in a shared memory array
    else {
        return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE)
                && sg.getTrgNeuronGroup()->getNumNeurons() <= backend.getKernelBlockSize(KernelPresynapticUpdate));
    }
}
//----------------------------------------------------------------------------
void PostSpan::genCode(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, const Substitutions &popSubs, const Backend &backend, bool trueSpike,
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

                Substitutions synSubs(&popSubs);
                synSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");
                if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    os << "unsigned int synAddress = shSpk" << eventSuffix << "[j] * " << std::to_string(sg.getMaxConnections()) << ";" << std::endl;
                    os << "const unsigned int npost = shRowLength[j];" << std::endl;

                    os << "if (" << popSubs["id"] << " < npost)" << CodeStream::OB(140);
                    os << "synAddress += " << popSubs["id"] << ";" << std::endl;
                    os << "const unsigned int ipost = dd_ind" << sg.getName() << "[synAddress];" << std::endl;

                    synSubs.addVarSubstitution("id_post", "ipost");
                }
                else { // DENSE
                    os << "unsigned int synAddress = (shSpk" << eventSuffix << "[j] * " << std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()) << ") + " + popSubs["id"] + ";" << std::endl;

                    synSubs.addVarSubstitution("id_post", popSubs["id"]);
                }
                synSubs.addVarSubstitution("id_syn", "synAddress");

                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                if(sg.isDendriticDelayRequired()) {
                    synSubs.addFuncSubstitution("addToInSynDelay", 2, backend.getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + synSubs["id_post"] + "], $(0))");
                }
                // Otherwise
                else {
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                        // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                        if (shouldAccumulateInSharedMemory(sg, backend)) {
                            synSubs.addFuncSubstitution("addToInSyn", 1, "shLg[" + synSubs["id_post"] + "] += $(0)");
                        }
                        else {
                            synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[" + synSubs["id_post"] + "], $(0))");
                        }
                    }
                    else {
                        synSubs.addFuncSubstitution("addToInSyn", 1, "linSyn += $(0)");
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
}   // namespace PresynapticUpdateStrategy
}   // namespace CUDA
}   // namespace CodeGenerator

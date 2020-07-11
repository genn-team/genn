#include "presynapticUpdateStrategy.h"

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/groupMerged.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/substitutions.h"

// OpenCL backend includes
#include "backend.h"

using namespace CodeGenerator;

//----------------------------------------------------------------------------
// CodeGenerator::OpenCL::PresynapticUpdateStrategy::PreSpan
//----------------------------------------------------------------------------
namespace CodeGenerator
{
namespace OpenCL
{
namespace PresynapticUpdateStrategy
{
size_t PreSpan::getNumThreads(const SynapseGroupInternal &sg) const
{
    // Use a thread for each presynaptic neuron
    // **YUCK** really should only launch a thread per-spike
    return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * sg.getNumThreadsPerSpike();
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
bool PreSpan::shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &, const Backend &) const
{
    // When presynaptic parallelism is used
    return false;
}
//----------------------------------------------------------------------------
bool PreSpan::shouldAccumulateInSharedMemory(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const
{
    // If dendritic delays are required, shared memory approach cannot be used so return false
    if(sg.getArchetype().isDendriticDelayRequired()) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if matrix is sparse
    // and the output population is small enough that input to it can be stored in a shared memory array
    else {
        return ((sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE)
                && sg.getArchetype().getTrgNeuronGroup()->getNumNeurons() <= backend.getKernelWorkGroupSize(KernelPresynapticUpdate));
    }
}
//----------------------------------------------------------------------------
void PreSpan::genCode(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg, const Substitutions &popSubs, const Backend &backend, bool trueSpike,
                      BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler) const
{
    // Get suffix based on type of events
    const ModelSpecInternal &model = modelMerged.getModel();
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const auto *wu = sg.getArchetype().getWUModel();
    const size_t numThreadsPerSpike = sg.getArchetype().getNumThreadsPerSpike();

    if(sg.getArchetype().getNumThreadsPerSpike() > 1) {
        os << "const unsigned int spike = " << popSubs["id"] << " / " << numThreadsPerSpike << ";" << std::endl;
        os << "const unsigned int thread = " << popSubs["id"] << " % " << numThreadsPerSpike << ";" << std::endl;
    }
    else {
        os << "const unsigned int spike = " << popSubs["id"] << ";" << std::endl;
    }

    os << "if (spike < " ;
    if (sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        os << "group->srcSpkCnt" << eventSuffix << "[preReadDelaySlot])";
    }
    else {
        os << "group->srcSpkCnt" << eventSuffix << "[0])";
    }
    {
        CodeStream::Scope b(os);

        if (sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
            os << "const unsigned int preInd = group->srcSpk" << eventSuffix;
            os << "[(preReadDelaySlot * group->numSrcNeurons) + spike];" << std::endl;
        }
        else {
            os << "const unsigned int preInd = group->srcSpk"  << eventSuffix;
            os << "[spike];" << std::endl;
        }

     
        if(numThreadsPerSpike > 1) {
            os << "unsigned int synAddress = (preInd * group->rowStride) + thread;" << std::endl;
        }
        else {
            os << "unsigned int synAddress = preInd * group->rowStride;" << std::endl;
        }
        os << "const unsigned int npost = group->rowLength[preInd];" << std::endl;

        if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << "if(";

            Substitutions threshSubs(&popSubs);
            threshSubs.addVarSubstitution("id_pre", "preInd");

            std::stringstream threshOsStream;
            CodeStream threshOs(threshOsStream);

            // Generate weight update threshold condition
            wumThreshHandler(threshOs, sg, threshSubs);

            std::string code = threshOsStream.str();

            if (!wu->getSimSupportCode().empty()) {
                code = substituteNamespaceFunction(wu->getSimSupportCode(), code, modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()));
            }

            os << code;

            // end code substitutions ----
            os << ")";

            os << CodeStream::OB(130);
        }

        if(sg.getArchetype().getNumThreadsPerSpike() > 1) {
            os << "for(unsigned int i = thread; i < npost; i += " << numThreadsPerSpike << ", synAddress += " << numThreadsPerSpike << ")";
        }
        else {
            os << "for(unsigned int i = 0; i < npost; i++, synAddress++)";
        }
        {
            CodeStream::Scope b(os);

            os << "const unsigned int ipost = group->ind[synAddress];" << std::endl;

            // Code substitutions ----------------------------------------------------------------------------------
            std::string wCode = trueSpike ? wu->getSimCode() : wu->getEventCode();

            Substitutions synSubs(&popSubs);
            synSubs.addVarSubstitution("id_pre", "preInd");
            synSubs.addVarSubstitution("id_post", "ipost");
            synSubs.addVarSubstitution("id_syn", "synAddress");

            // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
            if(sg.getArchetype().isDendriticDelayRequired()) {
                synSubs.addFuncSubstitution("addToInSynDelay", 2, backend.getFloatAtomicAdd(model.getPrecision()) + "(&group->denDelay[" + sg.getDendriticDelayOffset("$(1)") + "ipost], $(0))");            }
            // Otherwise
            else {
                // If postsynaptic input should be accumulated in shared memory, substitute shared memory array for $(inSyn)
                if(shouldAccumulateInSharedMemory(sg, backend)) {
                    synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision(), "local") + "(&shLg[ipost], $(0))");
                }
                // Otherwise, substitute global memory array for $(inSyn)
                else {
                    synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&group->inSyn[ipost], $(0))");
                }
            }

            std::stringstream wumSimOsStream;
            CodeStream wumSimOs(wumSimOsStream);

            wumSimHandler(wumSimOs, sg, synSubs);
            std::string code = wumSimOsStream.str();

            if (!wu->getSimSupportCode().empty()) {
                code = substituteNamespaceFunction(wu->getSimSupportCode(), code, modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()));
            }

            os << code;
        }

        if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::OpenCL::PresynapticUpdateStrategy::PostSpan
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
bool PostSpan::isCompatible(const SynapseGroupInternal &sg) const
{
    // Postsynatic parallelism can be used when synapse groups request it
    return ((sg.getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC)
            && !(sg.getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL));
}
//----------------------------------------------------------------------------
bool PostSpan::shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &sg, const Backend &) const
{
    // If no dendritic delays are required and data structure is dense, we can accumulate output directly into register
    const auto matrixType = sg.getArchetype().getMatrixType();
    return (!sg.getArchetype().isDendriticDelayRequired()
            && ((matrixType & SynapseMatrixConnectivity::DENSE) || (matrixType & SynapseMatrixConnectivity::BITMASK)));
}
//----------------------------------------------------------------------------
bool PostSpan::shouldAccumulateInSharedMemory(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const
{
    // If dendritic delays are required, shared memory approach cannot be used so return false
    if(sg.getArchetype().isDendriticDelayRequired()) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if matrix is sparse
    // and the output population is small enough that input to it can be stored in a shared memory array
    else {
        return ((sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE)
                && sg.getArchetype().getTrgNeuronGroup()->getNumNeurons() <= backend.getKernelWorkGroupSize(KernelPresynapticUpdate));
    }
}
//----------------------------------------------------------------------------
void PostSpan::genCode(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg, const Substitutions &popSubs, const Backend &backend, bool trueSpike,
                       BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler) const
{
    // Get suffix based on type of events
    const ModelSpecInternal &model = modelMerged.getModel();
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const size_t workGroupSize = backend.getKernelWorkGroupSize(KernelPresynapticUpdate);

    os << "const unsigned int numSpikes = group->srcSpkCnt" << eventSuffix;
    if (sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        os << "[preReadDelaySlot];" << std::endl;
    }
    else {
        os << "[0];" << std::endl;
    }

    os << "const unsigned int numSpikeBlocks = (numSpikes + " << workGroupSize << " - 1) / " << workGroupSize << ";" << std::endl;

    const auto *wu = sg.getArchetype().getWUModel();
    os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(os);
        os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << workGroupSize << ") + 1 : " << workGroupSize << ";" << std::endl;

        os << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        os << "if (localId < numSpikesInBlock)";
        {
            CodeStream::Scope b(os);
            const std::string queueOffset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
            os << "const unsigned int spk = group->srcSpk" << eventSuffix << "[" << queueOffset << "(r * " << workGroupSize << ") + localId];" << std::endl;
            os << "shSpk" << eventSuffix << "[localId] = spk;" << std::endl;
            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                os << "shRowLength[localId] = group->rowLength[spk];" << std::endl;
            }
        }
        os << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        os << "// loop through all incoming spikes" << std::endl;
        os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(os);
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << popSubs["id"] << " < group->rowStride)";
            {
                CodeStream::Scope b(os);
                if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    // Get maximum number of synapses anywhere in merged group
                    size_t maxSynapses = 0;
                    for(const auto &s : sg.getGroups()) {
                        maxSynapses = std::max(maxSynapses, (size_t)s.get().getTrgNeuronGroup()->getNumNeurons() * (size_t)s.get().getSrcNeuronGroup()->getNumNeurons());
                    }

                    if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                        os << "const ulong gid = (shSpk" << eventSuffix << "[j] * group->rowStride) + " << popSubs["id"] << ";" << std::endl;
                    }
                    else {
                        os << "const unsigned int gid = (shSpk" << eventSuffix << "[j] * group->rowStride) + " << popSubs["id"] << ";" << std::endl;
                    }
                }

                if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << "if(";
                    if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        os << "(B(group->gp[gid / 32], gid & 31)) && ";
                    }

                    Substitutions threshSubs(&popSubs);
                    threshSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");

                    std::stringstream threshOsStream;
                    CodeStream threshOs(threshOsStream);

                    // Generate weight update threshold condition
                    wumThreshHandler(threshOs, sg, threshSubs);
                    std::string code = threshOsStream.str();

                    if (!wu->getSimSupportCode().empty()) {
                        code = substituteNamespaceFunction(wu->getSimSupportCode(), code, modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()));
                    }

                    os << code;

                    // end code substitutions ----
                    os << ")";
                    os << CodeStream::OB(130);
                }
                else if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "if (B(group->gp[gid / 32], gid & 31))" << CodeStream::OB(135);
                }

                os << "unsigned int synAddress = (shSpk" << eventSuffix << "[j] * group->rowStride) + " << popSubs["id"] << ";" << std::endl;

                Substitutions synSubs(&popSubs);
                synSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");
                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    
                    os << "const unsigned int npost = shRowLength[j];" << std::endl;
                    os << "if (" << popSubs["id"] << " < npost)" << CodeStream::OB(140);
                    os << "const unsigned int ipost = group->ind[synAddress];" << std::endl;

                    synSubs.addVarSubstitution("id_post", "ipost");
                }
                else { // DENSE
                    synSubs.addVarSubstitution("id_post", popSubs["id"]);
                }
                synSubs.addVarSubstitution("id_syn", "synAddress");

                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                if(sg.getArchetype().isDendriticDelayRequired()) {
                    synSubs.addFuncSubstitution("addToInSynDelay", 2, backend.getFloatAtomicAdd(model.getPrecision()) + "(&group->denDelay[" + sg.getDendriticDelayOffset("$(1)") + synSubs["id_post"] + "], $(0))");
                }
                // Otherwise
                else {
                    if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                        // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                        if (shouldAccumulateInSharedMemory(sg, backend)) {
                            synSubs.addFuncSubstitution("addToInSyn", 1, "shLg[" + synSubs["id_post"] + "] += $(0)");
                        }
                        else {
                            synSubs.addFuncSubstitution("addToInSyn", 1, backend.getFloatAtomicAdd(model.getPrecision()) + "(&group->inSyn[" + synSubs["id_post"] + "], $(0))");
                        }
                    }
                    else {
                        synSubs.addFuncSubstitution("addToInSyn", 1, "linSyn += $(0)");
                    }
                }

                std::stringstream wumSimOsStream;
                CodeStream wumSimOs(wumSimOsStream);

                wumSimHandler(wumSimOs, sg, synSubs);
                std::string code = wumSimOsStream.str();

                // Substituting uses of support code with no namespace functions (if any)
                if (!wu->getSimSupportCode().empty()) {
                    code = substituteNamespaceFunction(wu->getSimSupportCode(), code, modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()));
                }

                os << code;

                if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    os << CodeStream::CB(140); // end if (id < npost)
                }

                if (!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
                else if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << CodeStream::CB(135); // end if (B(d_gp" << sg.getName() << "[gid / 32], gid
                }
            }
        }
    }
}
}   // namespace PresynapticUpdateStrategy
}   // namespace OpenCL
}   // namespace CodeGenerator

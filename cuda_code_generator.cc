#include "cuda_code_generator.h"

// Standard C++ includes
#include <algorithm>

// GeNN includes
#include "codeGenUtils.h"
#include "codeStream.h"
#include "modelSpec.h"

// NuGeNN includes
#include "substitution_stack.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
size_t padSize(size_t size, size_t blockSize)
{
    return ((size + blockSize - 1) / blockSize) * blockSize;
}

// Helpers::Variable
template<typename Type>
class Variable
{
public:
    Variable(const std::string &variableName) : m_Text(variableName)
    {
    }

    const std::string &getText() const{ return m_Text; }

private:
    const std::string m_Text;
};
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CUDA::CodeGenerator
//--------------------------------------------------------------------------
namespace CUDA
{
void CodeGenerator::genNeuronUpdateKernel(CodeStream &os, const NNmodel &model, 
                                          std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const NeuronGroup &ng, const Substitutions &)> handler) const
{
    os << "extern \"C\" __global__ void calcNeurons(float time)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "const unsigned int id = " << m_NeuronUpdateBlockSize << " * blockIdx.x + threadIdx.x; " << std::endl;

        Substitutions baseSubs;
        baseSubs.addSubstitution("t", "t");

        // If any neuron groups emit spike events
        if(std::any_of(model.getLocalNeuronGroups().cbegin(), model.getLocalNeuronGroups().cend(),
            [](const NNmodel::NeuronGroupValueType &n){ return n.second.isSpikeEventRequired(); }))
        {
            os << "__shared__ volatile unsigned int shSpkEvnt[" << m_NeuronUpdateBlockSize << "];" << std::endl;
            os << "__shared__ volatile unsigned int shPosSpkEvnt;" << std::endl;
            os << "__shared__ volatile unsigned int shSpkEvntCount;" << std::endl;
            os << std::endl;
            os << "if (threadIdx.x == 1);";
            {
                CodeStream::Scope b(os);
                os << "shSpkEvntCount = 0;" << std::endl;
            }
            os << std::endl;
        }

        // If any neuron groups emit true spikes
        if(std::any_of(model.getLocalNeuronGroups().cbegin(), model.getLocalNeuronGroups().cend(),
            [](const NNmodel::NeuronGroupValueType &n){ return !n.second.getNeuronModel()->getThresholdConditionCode().empty(); }))
        {
            os << "__shared__ volatile unsigned int shSpk[" << m_NeuronUpdateBlockSize << "];" << std::endl;
            os << "__shared__ volatile unsigned int shPosSpk;" << std::endl;
            os << "__shared__ volatile unsigned int shSpkCount;" << std::endl;
            os << "if (threadIdx.x == 0);";
            {
                CodeStream::Scope b(os);
                os << "shSpkCount = 0;" << std::endl;
            }
            os << std::endl;
        }
            
        os << "__syncthreads();" << std::endl;
            
        

        // Parallelise over neuron groups
        genParallelNeuronGroup(os, model,
            [handler, &baseSubs](CodeStream &os, const ::CodeGenerator::Base &codeGenerator, const NNmodel &model, const NeuronGroup &ng)
            {
                Substitutions subs(&baseSubs);
                
                // Neuron ID
                subs.addSubstitution("id", "lid");

                // Get name of rng to use for this neuron
                subs.addSubstitution("rng", "&dd_rng" + ng.getName() + "[lid]");
                
                // Call handler to generate generic neuron code
                handler(os, codeGenerator, model, ng, subs);

                os << "__syncthreads();" << std::endl;

                if (ng.isSpikeEventRequired()) {
                    os << "if (threadIdx.x == 1)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvnt" << ng.getName();
                        if (ng.isDelayRequired()) {
                            os << "[dd_spkQuePtr" << ng.getName() << "], spkEvntCount);" << std::endl;
                        }
                        else {
                            os << "[0], spkEvntCount);" << std::endl;
                        }
                    } // end if (threadIdx.x == 0)
                    os << "__syncthreads();" << std::endl;
                }

                if (!ng.getNeuronModel()->getThresholdConditionCode().empty()) {
                    os << "if (threadIdx.x == 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCnt" << ng.getName();
                        if (ng.isDelayRequired() && ng.isTrueSpikeRequired()) {
                            os << "[dd_spkQuePtr" << ng.getName() << "], spkCount);" << std::endl;
                        }
                        else {
                            os << "[0], spkCount);" << std::endl;
                        }
                    } // end if (threadIdx.x == 1)
                    os << "__syncthreads();" << std::endl;
                }

                const std::string queueOffset = ng.getQueueOffset("dd_");
                if (ng.isSpikeEventRequired()) {
                    os << "if (threadIdx.x < spkEvntCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_glbSpkEvnt" << ng.getName() << "[" << queueOffset << "posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << std::endl;
                    }   // end if (threadIdx.x < spkEvntCount)
                }

                if (!ng.getNeuronModel()->getThresholdConditionCode().empty()) {
                    const std::string queueOffsetTrueSpk = ng.isTrueSpikeRequired() ? queueOffset : "";

                    os << "if (threadIdx.x < spkCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_glbSpk" << ng.getName() << "[" << queueOffsetTrueSpk << "posSpk + threadIdx.x] = shSpk[threadIdx.x];" << std::endl;
                        if (ng.isSpikeTimeRequired()) {
                            os << "dd_sT" << ng.getName() << "[" << queueOffset << "shSpk[threadIdx.x]] = t;" << std::endl;
                        }
                    }   // end if (threadIdx.x < spkCount)
                }
            }
        );
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genPresynapticUpdateKernel(CodeStream &os, const NNmodel &model) const
{
    os << "extern \"C\" __global__ void calcSynapses(";
    for (const auto &p : model.getSynapseKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    os << model.getPrecision() << " t)" << std::endl; // end of synapse kernel header
    {
        CodeStream::Scope b(os);
        
        os << "const unsigned int id = " << m_PresynapticUpdateBlockSize << " * blockIdx.x + threadIdx.x; " << std::endl;

        // We need shLg if any synapse groups accumulate into shared memory
        if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [this](const NNmodel::SynapseGroupValueType &s){ return shouldAccumulateInSharedMemory(s.second); }))
        {
            os << "__shared__ " << model.getPrecision() << " shLg[" << m_PresynapticUpdateBlockSize << "];" << std::endl;
        }
        
        if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [&model](const NNmodel::SynapseGroupValueType &s)
            { 
                return (s.second.isTrueSpikeRequired() || model.isSynapseGroupPostLearningRequired(s.first));
            }))
        {
            os << "__shared__ unsigned int shSpk[" << m_PresynapticUpdateBlockSize << "];" << std::endl;
        }
        
        if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [](const NNmodel::SynapseGroupValueType &s){ return (s.second.isSpikeEventRequired()); }))
        {
            os << "__shared__ unsigned int shSpkEvnt[" << m_PresynapticUpdateBlockSize << "];" << std::endl;
        }
        
        // Parallelise over synapse groups
        genParallelSynapseGroup(os, model, 
            [this](const SynapseGroup &sg){ return getPresynapticUpdateKernelSize(sg); },
            [this](CodeStream &os, const ::CodeGenerator::Base &codeGenerator, const NNmodel &model, const SynapseGroup &sg)
            {
                if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                    os << "const unsigned int delaySlot = (dd_spkQuePtr" <<sg.getSrcNeuronGroup()->getName();
                    os << " + " << (sg.getSrcNeuronGroup()->getNumDelaySlots() - sg.getDelaySteps());
                    os << ") % " << sg.getSrcNeuronGroup()->getNumDelaySlots() << ";" << std::endl;
                }

                // If we are going to accumulate postsynaptic input into a register, copy current value into register from global memory
                if (shouldAccumulateInLinSyn(sg)) {
                    os << "// only do this for existing neurons" << std::endl;
                    os << model.getPrecision() << " linSyn;" << std::endl;
                    os << "if(lid < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "linSyn = dd_inSyn" << sg.getName() << "[lid];" << std::endl;
                    }
                }
                // Otherwise, if we are going to accumulate into shared memory, copy current value into correct array index
                // **NOTE** is ok as number of target neurons <= synapseBlkSz
                else if(shouldAccumulateInSharedMemory(sg)) {
                    os << "if(threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "shLg[threadIdx.x] = dd_inSyn" << sg.getName() << "[threadIdx.x];"<< std::endl;
                    }
                    os << "__syncthreads();" << std::endl;
                }

                if (sg.isSpikeEventRequired()) {
                    os << "const unsigned int spkCntEvent = dd_glbSpkCntEvnt" << sg.getSrcNeuronGroup()->getName();
                    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                        os << "[delaySlot];" << std::endl;
                    }
                    else {
                        os << "[0];" << std::endl;
                    }
                }

                if (sg.isTrueSpikeRequired() || model.isSynapseGroupPostLearningRequired(sg.getName())) {
                    os << "const unsigned int spkCnt = dd_glbSpkCnt" << sg.getSrcNeuronGroup()->getName();
                    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                        os << "[delaySlot];" << std::endl;
                    }
                    else {
                        os << "[0];" << std::endl;
                    }
                }
            
                // If spike events should be processed
                if (sg.isSpikeEventRequired()) {
                    if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) {
                    }
                    else {
                    }
                }

                // If true spikes should be processed
                if (sg.isTrueSpikeRequired()) {
                    if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) {
                    }
                    else {
                    }
                }
                
                os << std::endl;

                // If we have been accumulating into a register, write value back to global memory
                if (shouldAccumulateInLinSyn(sg)) {
                    os << "// only do this for existing neurons" << std::endl;
                    os << "if (lid < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_inSyn" << sg.getName() << "[lid] = linSyn;" << std::endl;
                    }
                }
                // Otherwise, if we have been accumulating into shared memory, write value back to global memory
                // **NOTE** is ok as number of target neurons <= synapseBlkSz
                else if(shouldAccumulateInSharedMemory(sg)) {
                    os << "__syncthreads();" << std::endl;
                    os << "if (threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_inSyn" << sg.getName() << "[threadIdx.x] = shLg[threadIdx.x];"<< std::endl;
                    }
                }
            }
        );
                                
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genParallelNeuronGroup(CodeStream &os, const NNmodel &model,
                                           std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel &, const NeuronGroup&)> handler) const
{
    // Populate neuron update groups
    size_t idStart = 0;
    for (const auto &ng : model.getLocalNeuronGroups()) {
        const size_t paddedSize = padSize(ng.second.getNumNeurons(), m_NeuronUpdateBlockSize);

        os << "// Neuron group " << ng.first << std::endl;

        // If this is the first  group
        if (idStart == 0) {
            os << "if(id < " << paddedSize << ")" << CodeStream::OB(1);
            os << "const unsigned int lid = id;" << std::endl;
        }
        else {
            os << "if(id >= " << idStart << " && id < " << idStart + paddedSize << ")" << CodeStream::OB(1);
            os << "const unsigned int lid = id - " << idStart << ";" << std::endl;
        }

        handler(os, *this, model, ng.second);

        idStart += paddedSize;
        os << CodeStream::CB(1) << std::endl;
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genParallelSynapseGroup(CodeStream &os, const NNmodel &model,
                                            std::function<size_t(const SynapseGroup&)> getPaddedSizeFunc,
                                            std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel &, const SynapseGroup&)> handler) const
{
    // Populate neuron update groups
    size_t idStart = 0;
    for (const auto &sg : model.getLocalSynapseGroups()) {
        const size_t paddedSize = getPaddedSizeFunc(sg.second);

        os << "// Synapse group " << sg.first << std::endl;

        // If this is the first  group
        if (idStart == 0) {
            os << "if(id < " << paddedSize << ")" << CodeStream::OB(1);
            os << "const unsigned int lid = id;" << std::endl;
        }
        else {
            os << "if(id >= " << idStart << " && id < " << idStart + paddedSize << ")" << CodeStream::OB(1);
            os << "const unsigned int lid = id - " << idStart << ";" << std::endl;
        }

        handler(os, *this, model, sg.second);

        idStart += paddedSize;
        os << CodeStream::CB(1) << std::endl;
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const
{
    os << "const unsigned int spk" << suffix << "Idx = atomicAdd((unsigned int *) &shSpk" << suffix << "Count, 1);" << std::endl;
    os << "shSpk" << suffix << "[spk" << suffix << "Idx] = " << subs.getSubstitution("id") << ";" << std::endl;
}
//--------------------------------------------------------------------------
size_t CodeGenerator::getPresynapticUpdateKernelSize(const SynapseGroup &sg) const
{
     if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        if (sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) {
            // paddedSize is the lowest multiple of blockSize >= neuronN[synapseSource[i]
            // **TODO** integer ceil trick
            return ceil((double)sg.getSrcNeuronGroup()->getNumNeurons() / (double)m_PresynapticUpdateBlockSize) * (double)m_PresynapticUpdateBlockSize;
        }
        else {
            // paddedSize is the lowest multiple of blockSize >= maxConn[i]
            // **TODO** integer ceil trick
            return ceil((double)sg.getMaxConnections() / (double) m_PresynapticUpdateBlockSize) * (double) m_PresynapticUpdateBlockSize;
        }
    }
    else {
        // paddedSize is the lowest multiple of blockSize >= neuronN[synapseTarget[i]]
        return ceil((double)sg.getTrgNeuronGroup()->getNumNeurons() / (double) m_PresynapticUpdateBlockSize) * (double) m_PresynapticUpdateBlockSize;
    }
}
//--------------------------------------------------------------------------
bool CodeGenerator::shouldAccumulateInLinSyn(const SynapseGroup &sg) const
{
    // We should accumulate each postsynaptic neuron's input in a register if matrix is dense or bitfield (where each thread represents an individual neuron)
    return ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) || (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK));
}
//--------------------------------------------------------------------------
bool CodeGenerator::shouldAccumulateInSharedMemory(const SynapseGroup &sg) const
{
    // If parallelism is presynaptic i.e. atomics are required and device is older than Maxwell, we shouldn't use shared memory as atomics are emulated
    // and actually slower than global memory (see https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
    if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC/* && deviceProp[theDevice].major < 5*/) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if matrix is sparse
    // and the output population is small enough that input to it can be stored in a shared memory array
    else {
        return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && sg.getTrgNeuronGroup()->getNumNeurons() <= m_PresynapticUpdateBlockSize);
    }
}
}   // namespace CUDA

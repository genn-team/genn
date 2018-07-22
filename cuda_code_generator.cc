#include "cuda_code_generator.h"

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

namespace CUDA
{
void CodeGenerator::genNeuronUpdateKernel(CodeStream &os, const NNmodel &model, 
                                          std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const NeuronGroup &ng, const std::string &, const std::string &)> handler) const
{
    os << "extern \"C\" __global__ void calcNeurons(float time)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "const unsigned int id = " << m_NeuronUpdateBlockSize << " * blockIdx.x + threadIdx.x; " << std::endl;

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
            
        // Parallelise over neurons
        genParallelNeuronGroup(os, model,
            [handler](CodeStream &os, const ::CodeGenerator::Base &codeGenerator, const NNmodel &model, const NeuronGroup &ng)
            {
                // Neuron ID
                const std::string neuronID = "lid"; 

                // Get name of rng to use for this neuron
                // **TODO** Phillox option
                const std::string rngName = "&dd_rng" + ng.getName() + "[" + neuronID + "]";

                // Call handler to generate generic neuron code
                handler(os, codeGenerator, model, ng, neuronID, rngName);

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


void CodeGenerator::genEmitSpike(CodeStream &os, const std::string &neuronID, const std::string &suffix) const
{
    os << "const unsigned int spk" << suffix << "Idx = atomicAdd((unsigned int *) &shSpk" << suffix << "Count, 1);" << std::endl;
    os << "shSpk" << suffix << "[spk" << suffix << "Idx] = " << neuronID << ";" << std::endl;
}
}   // namespace CUDA
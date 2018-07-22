#pragma once

// Standard C++ includes
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

// GeNN includes
#include "codeGenUtils.h"
#include "codeStream.h"
#include "modelSpec.h"

// NuGeNN includes
#include "code_generator.h"

// CUDACodeGenerator
namespace CUDA
{
namespace Helpers
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
}   // Helpers

class CodeGenerator : public ::CodeGenerator::Base
{
public:
    CodeGenerator(size_t neuronUpdateBlockSize) : m_NeuronUpdateBlockSize(neuronUpdateBlockSize)
    {
    }

    // CodeGenerator::Base virtuals
    virtual void genNeuronUpdateKernel(CodeStream &os, const NNmodel &model, 
                                       std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&)> handler) const override
    {
        os << "extern \"C\" __global__ void calcNeurons(float time)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "const unsigned int id = " << m_NeuronUpdateBlockSize << " * blockIdx.x + threadIdx.x; " << std::endl;
            os << "__shared__ volatile unsigned int shSpk[" << m_NeuronUpdateBlockSize << "];" << std::endl;
            os << "__shared__ volatile unsigned int posSpk;" << std::endl;
            os << "__shared__ volatile unsigned int shSpkCount;" << std::endl;
            os << std::endl;
            os << "if (threadIdx.x == 0);";
            {
                CodeStream::Scope b(os);
                os << "shSpkCount = 0;" << std::endl;
            }
            os << "__syncthreads();" << std::endl;
            handler(os, *this, model);
        }
    }

    virtual void genForEachNeuronGroup(CodeStream &os, const NNmodel &model,
                                       std::function<void(CodeStream &output, const ::CodeGenerator::Base &, const NNmodel &, const NeuronGroup&)> handler) const override
    {
        // Populate neuron update groups
        size_t idStart = 0;
        for (const auto &ng : model.getLocalNeuronGroups()) {
            const size_t paddedSize = Helpers::padSize(ng.second.getNumNeurons(), m_NeuronUpdateBlockSize);

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

    virtual void genForEachNeuron(CodeStream &os, const NNmodel &model, const NeuronGroup &ng, 
                                  std::function<void(CodeStream &output, const ::CodeGenerator::Base &, const NNmodel &, const NeuronGroup&, const std::string &, const std::string &)> handler) const override
    {
        // Get name of rng to use for this neuron
        // **TODO** Phillox option
        const std::string rngName = "&dd_rng" + ng.getName() + "[lid]";

        // Neurons are parallelised across threads - no need to parallelise
        handler(os, *this, model, ng, "lid", rngName);
    }

    virtual void genEmitTrueSpike(CodeStream &os, const NNmodel &, const NeuronGroup &, const std::string &neuronID) const override
    {
        genEmitSpike(os, neuronID, "");
    }
    
    virtual void genEmitSpikeLikeEvent(CodeStream &os, const NNmodel &model, const NeuronGroup &ng, const std::string &neuronID) const override
    {
        genEmitSpike(os, neuronID, "Evnt");
    }

    virtual std::string getVarPrefix() const override{ return "dd_"; }

    virtual const std::vector<FunctionTemplate> &getFunctions() const override{ return cudaFunctions; }

private:
    void genEmitSpike(CodeStream &os, const std::string &neuronID, const std::string &suffix) const
    {
        os << "const unsigned int spk" << suffix << "Idx = atomicAdd((unsigned int *) &shSpk" << suffix << "Count, 1);" << std::endl;
        os << "shSpk" << suffix << "[spk" << suffix << "Idx] = " << neuronID << ";" << std::endl;
    }

    const size_t m_NeuronUpdateBlockSize;
};
}   // CodeGenerator
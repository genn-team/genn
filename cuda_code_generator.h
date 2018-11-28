#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>

// NuGeNN includes
#include "code_generator.h"

//--------------------------------------------------------------------------
// CUDA::CodeGenerator
//--------------------------------------------------------------------------
namespace CUDA
{
class CodeGenerator : public ::CodeGenerator::Base
{
public:
    CodeGenerator(size_t neuronUpdateBlockSize, size_t presynapticUpdateBlockSize, int localHostID);

    //--------------------------------------------------------------------------
    // CodeGenerator::Base virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdateKernel(CodeStream &os, const NNmodel &model,
                                       std::function<void(CodeStream&, const ::CodeGenerator::Base&, const NNmodel&, const NeuronGroup &ng, Substitutions&)> handler) const override;

    virtual void genPresynapticUpdateKernel(CodeStream &os, const NNmodel &model,
                                            std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const SynapseGroup &, const Substitutions&)> wumThreshHandler,
                                            std::function<void(CodeStream&, const::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, const Substitutions&)> wumSimHandler) const override;

    virtual void genInitKernel(CodeStream &os, const NNmodel &model,
                               std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const NeuronGroup &, const Substitutions&)> ngHandler,
                               std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const SynapseGroup &, const Substitutions&)> sgHandler) const override;

    virtual void genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const override;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const override;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const override;

    virtual void genEmitTrueSpike(CodeStream &os, const NNmodel&, const NeuronGroup&, const Substitutions &subs) const override
    {
        genEmitSpike(os, subs, "");
    }
    
    virtual void genEmitSpikeLikeEvent(CodeStream &os, const NNmodel &, const NeuronGroup &, const Substitutions &subs) const override
    {
        genEmitSpike(os, subs, "Evnt");
    }

    virtual std::string getVarPrefix() const override{ return "dd_"; }

    virtual const std::vector<FunctionTemplate> &getFunctions() const override{ return cudaFunctions; }

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    void genParallelNeuronGroup(CodeStream &os, const Substitutions &kernelSubs,
                                const std::map<std::string, NeuronGroup> &ngs, std::function<bool(const NeuronGroup &)> filter,
                                std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NeuronGroup&, Substitutions &)> handler) const;

    void genParallelNeuronGroup(CodeStream &os, const Substitutions &kernelSubs,
                                const std::map<std::string, NeuronGroup> &ngs,
                                std::function<void(CodeStream &, const ::CodeGenerator::Base&, const NeuronGroup&, Substitutions &)> handler) const
    {
        genParallelNeuronGroup(os, kernelSubs, ngs, [](const NeuronGroup&){ return true; }, handler);
    }

    void genParallelSynapseGroup(CodeStream &os, const Substitutions &kernelSubs, const NNmodel &model, 
                                 std::function<size_t(const SynapseGroup&)> getPaddedSizeFunc,
                                 std::function<bool(const SynapseGroup &)> filter,
                                 std::function<void(CodeStream &, const ::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, Substitutions &)> handler) const;

    void genParallelSynapseGroup(CodeStream &os, const Substitutions &kernelSubs, const NNmodel &model, 
                                 std::function<size_t(const SynapseGroup&)> getPaddedSizeFunc,
                                 std::function<void(CodeStream &, const ::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, Substitutions &)> handler) const
    {
        genParallelSynapseGroup(os, kernelSubs,  model, getPaddedSizeFunc, [](const SynapseGroup&){ return true; }, handler);
    }
                                 
    void genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const;

    void genPresynapticUpdateKernelPreSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
                                           std::function<void(CodeStream&, const ::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, Substitutions&)> wumThreshHandler,
                                           std::function<void(CodeStream&, const::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, Substitutions&)> wumSimHandler) const;
    void genPresynapticUpdateKernelPostSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
                                            std::function<void(CodeStream&, const ::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, Substitutions&)> wumThreshHandler,
                                            std::function<void(CodeStream&, const::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, Substitutions&)> wumSimHandler) const;

    size_t getPresynapticUpdateKernelSize(const SynapseGroup &sg) const;
    
    bool shouldAccumulateInLinSyn(const SynapseGroup &sg) const;
    
    bool shouldAccumulateInSharedMemory(const SynapseGroup &sg) const;
    
    std::string getFloatAtomicAdd(const std::string &ftype) const;
    
    const cudaDeviceProp &getChosenCUDADevice() const{ return m_Devices[m_ChosenDevice]; }
    
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const size_t m_NeuronUpdateBlockSize;
    const size_t m_PresynapticUpdateBlockSize;
    const int m_LocalHostID;
    
    std::vector<cudaDeviceProp> m_Devices;
    int m_ChosenDevice;
};
}   // CodeGenerator

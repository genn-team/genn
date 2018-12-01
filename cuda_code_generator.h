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
    CodeGenerator(size_t neuronUpdateBlockSize, size_t presynapticUpdateBlockSize, size_t initBlockSize, int localHostID,
                  const ::CodeGenerator::Base &hostCodeGenerator);

    //--------------------------------------------------------------------------
    // CodeGenerator::Base virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdateKernel(CodeStream &os, const NNmodel &model,
                                       NeuronGroupHandler handler) const override;

    virtual void genPresynapticUpdateKernel(CodeStream &os, const NNmodel &model,
                                            SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const override;

    virtual void genInitKernel(CodeStream &os, const NNmodel &model,
                               NeuronGroupHandler ngHandler, SynapseGroupHandler sgHandler) const override;

    virtual void genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const override;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const override;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const override;

    virtual void genVariableInit(CodeStream &os, VarMode mode, size_t count, const Substitutions &kernelSubs, Handler handler) const override;

    virtual void genEmitTrueSpike(CodeStream &os, const NNmodel&, const NeuronGroup&, const Substitutions &subs) const override
    {
        genEmitSpike(os, subs, "");
    }
    
    virtual void genEmitSpikeLikeEvent(CodeStream &os, const NNmodel &, const NeuronGroup &, const Substitutions &subs) const override
    {
        genEmitSpike(os, subs, "Evnt");
    }

    virtual std::string getVarPrefix() const override{ return "dd_"; }

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    void genParallelNeuronGroup(CodeStream &os, const Substitutions &kernelSubs, const std::map<std::string, NeuronGroup> &ngs, size_t &idStart, 
                                std::function<bool(const NeuronGroup &)> filter, NeuronGroupHandler handler) const;

    void genParallelNeuronGroup(CodeStream &os, const Substitutions &kernelSubs, const std::map<std::string, NeuronGroup> &ngs, size_t &idStart,
                                NeuronGroupHandler handler) const
    {
        genParallelNeuronGroup(os, kernelSubs, ngs, idStart, [](const NeuronGroup&){ return true; }, handler);
    }

    void genParallelSynapseGroup(CodeStream &os, const Substitutions &kernelSubs, const NNmodel &model, size_t &idStart, 
                                 std::function<size_t(const SynapseGroup&)> getPaddedSizeFunc,
                                 std::function<bool(const SynapseGroup &)> filter, SynapseGroupHandler handler) const;

    void genParallelSynapseGroup(CodeStream &os, const Substitutions &kernelSubs, const NNmodel &model, size_t &idStart, 
                                 std::function<size_t(const SynapseGroup&)> getPaddedSizeFunc,
                                 SynapseGroupHandler handler) const
    {
        genParallelSynapseGroup(os, kernelSubs,  model, idStart, getPaddedSizeFunc, [](const SynapseGroup&){ return true; }, handler);
    }
                                 
    void genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const;

    void genPresynapticUpdateKernelPreSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
                                           SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const;
    void genPresynapticUpdateKernelPostSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
                                            SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const;

    size_t getPresynapticUpdateKernelSize(const SynapseGroup &sg) const;
    
    bool shouldAccumulateInLinSyn(const SynapseGroup &sg) const;
    
    bool shouldAccumulateInSharedMemory(const SynapseGroup &sg) const;
    
    std::string getFloatAtomicAdd(const std::string &ftype) const;
    
    const cudaDeviceProp &getChosenCUDADevice() const{ return m_Devices[m_ChosenDevice]; }
    
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const ::CodeGenerator::Base &m_HostCodeGenerator;
    
    const size_t m_NeuronUpdateBlockSize;
    const size_t m_PresynapticUpdateBlockSize;
    const size_t m_InitBlockSize;
    const int m_LocalHostID;
    
    std::vector<cudaDeviceProp> m_Devices;
    int m_ChosenDevice;
};
}   // CodeGenerator

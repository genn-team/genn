#pragma once

// Standard C++ includes
#include <array>
#include <functional>
#include <map>
#include <string>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>

// GeNN includes
#include "codeStream.h"

// NuGeNN includes
#include "base.h"
#include "../substitution_stack.h"

//--------------------------------------------------------------------------
// CodeGenerator::Backends::CUDA
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace Backends
{
class CUDA : public Base
{
public:
    //--------------------------------------------------------------------------
    // Enumerations
    //--------------------------------------------------------------------------
    enum Kernel
    {
        KernelNeuronUpdate,
        KernelPresynapticUpdate,
        KernelPostsynapticUpdate,
        KernelSynapseDynamicsUpdate,
        KernelInitialize,
        KernelInitializeSparse,
        KernelPreNeuronReset,
        KernelPreSynapseReset,
        KernelMax
    };

    //--------------------------------------------------------------------------
    // Type definitions
    //--------------------------------------------------------------------------
    using KernelBlockSize = std::array<size_t, KernelMax>;

    CUDA(const KernelBlockSize &kernelBlockSizes, int localHostID, int device, const Base &hostBackend);

    //--------------------------------------------------------------------------
    // CodeGenerator::Backends:: virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, const NNmodel &model, NeuronGroupHandler handler) const override;

    virtual void genSynapseUpdate(CodeStream &os, const NNmodel &model,
                                  SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler,
                                  SynapseGroupHandler postLearnHandler, SynapseGroupHandler synapseDynamicsHandler) const override;

    virtual void genInit(CodeStream &os, const NNmodel &model,
                         NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                         SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                         SynapseGroupHandler sgSparseInitHandler) const override;

    virtual void genDefinitionsPreamble(CodeStream &os) const override;
    virtual void genRunnerPreamble(CodeStream &os) const override;
    virtual void genAllocateMemPreamble(CodeStream &os, const NNmodel &model) const override;

    virtual void genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const override;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const override;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const override;
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarMode mode) const override;

    virtual void genPopVariableInit(CodeStream &os, VarMode mode, const Substitutions &kernelSubs, Handler handler) const override;
    virtual void genVariableInit(CodeStream &os, VarMode mode, size_t count, const std::string &countVarName,
                                 const Substitutions &kernelSubs, Handler handler) const override;

    virtual void genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, bool autoInitialized, size_t count) const override;
    virtual void genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const override;
    virtual void genCurrentTrueSpikePush(CodeStream &os, const NeuronGroup &ng) const override
    {
        genCurrentSpikePush(os, ng, false);
    }
    virtual void genCurrentTrueSpikePull(CodeStream &os, const NeuronGroup &ng) const override
    {
        genCurrentSpikePull(os, ng, false);
    }
    virtual void genCurrentSpikeLikeEventPush(CodeStream &os, const NeuronGroup &ng) const override
    {
        genCurrentSpikePush(os, ng, true);
    }
    virtual void genCurrentSpikeLikeEventPull(CodeStream &os, const NeuronGroup &ng) const override
    {
        genCurrentSpikePull(os, ng, true);
    }

    virtual void genGlobalRNG(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free, const NNmodel &model) const override;
    virtual void genPopulationRNG(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                                  const std::string &name, size_t count) const override;

    virtual void genMakefilePreamble(std::ostream &os) const override;
    virtual void genMakefileLinkRule(std::ostream &os) const override;
    virtual void genMakefileCompileRule(std::ostream &os) const override;

    virtual void genEmitTrueSpike(CodeStream &os, const NNmodel&, const NeuronGroup&, const Substitutions &subs) const override
    {
        genEmitSpike(os, subs, "");
    }
    
    virtual void genEmitSpikeLikeEvent(CodeStream &os, const NNmodel &, const NeuronGroup &, const Substitutions &subs) const override
    {
        genEmitSpike(os, subs, "Evnt");
    }

    virtual std::string getVarPrefix() const override{ return "dd_"; }

    virtual bool isGlobalRNGRequired(const NNmodel &model) const override;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    const cudaDeviceProp &getChosenCUDADevice() const{ return m_ChosenDevice; }
    int getChosenDeviceID() const{ return m_ChosenDeviceID; }
    std::string getNVCCFlags() const;

    //--------------------------------------------------------------------------
    // Static API
    //--------------------------------------------------------------------------
    static size_t getNumPresynapticUpdateThreads(const SynapseGroup &sg);
    static size_t getNumPostsynapticUpdateThreads(const SynapseGroup &sg);
    static size_t getNumSynapseDynamicsThreads(const SynapseGroup &sg);

    //--------------------------------------------------------------------------
    // Constants
    //--------------------------------------------------------------------------
    static const char *KernelNames[KernelMax];

private:
    //--------------------------------------------------------------------------
    // Type definitions
    //--------------------------------------------------------------------------
    template<typename T>
    using GetPaddedGroupSizeFunc = std::function<size_t(const T&)>;

    template<typename T>
    using FilterGroupFunc = std::function<bool(const T&)>;

    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    template<typename T>
    void genParallelGroup(CodeStream &os, const Substitutions &kernelSubs, const std::map<std::string, T> &groups, size_t &idStart,
                          GetPaddedGroupSizeFunc<T> getPaddedSizeFunc,
                          FilterGroupFunc<T> filter, 
                          GroupHandler<T> handler) const
    {
        // Populate neuron update groups
        for (const auto &g : groups) {
            // If this synapse group should be processed
            Substitutions popSubs(&kernelSubs);
            if(filter(g.second)) {
                const size_t paddedSize = getPaddedSizeFunc(g.second);

                os << "// " << g.first << std::endl;

                // If this is the first  group
                if (idStart == 0) {
                    os << "if(id < " << paddedSize << ")" << CodeStream::OB(1);
                    popSubs.addVarSubstitution("id", "id");
                }
                else {
                    os << "if(id >= " << idStart << " && id < " << idStart + paddedSize << ")" << CodeStream::OB(1);
                    os << "const unsigned int lid = id - " << idStart << ";" << std::endl;
                    popSubs.addVarSubstitution("id", "lid");
                }

                handler(os, g.second, popSubs);

                idStart += paddedSize;
                os << CodeStream::CB(1) << std::endl;
            }
        }
    }

    template<typename T>
    void genParallelGroup(CodeStream &os, const Substitutions &kernelSubs, const std::map<std::string, T> &groups, size_t &idStart,
                          GetPaddedGroupSizeFunc<T> getPaddedSizeFunc,
                          GroupHandler<T> handler) const
    {
        genParallelGroup<T>(os, kernelSubs, groups, idStart, getPaddedSizeFunc,
                            [](const T&){ return true; }, handler);
    }

    void genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const;

    void genCurrentSpikePush(CodeStream &os, const NeuronGroup &ng, bool spikeEvent) const;
    void genCurrentSpikePull(CodeStream &os, const NeuronGroup &ng, bool spikeEvent) const;

    void genPresynapticUpdatePreSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
                                     SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const;
    void genPresynapticUpdatePostSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
                                      SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const;

    void genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreads) const;

    bool shouldAccumulateInLinSyn(const SynapseGroup &sg) const;

    bool shouldAccumulateInSharedMemory(const SynapseGroup &sg) const;

    std::string getFloatAtomicAdd(const std::string &ftype) const;

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const Base &m_HostBackend;
    
    const KernelBlockSize m_KernelBlockSizes;
    const int m_LocalHostID;
    
    const int m_ChosenDeviceID;
    cudaDeviceProp m_ChosenDevice;

    int m_RuntimeVersion;
};
}   // Backends
}   // CodeGenerator

#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>

// NuGeNN includes
#include "code_generator.h"

//--------------------------------------------------------------------------
// SingleThreadedCPU::CodeGenerator
//--------------------------------------------------------------------------
namespace SingleThreadedCPU
{
class CodeGenerator : public ::CodeGenerator::Base
{
public:
    CodeGenerator(int localHostID) 
    :   m_LocalHostID(localHostID)
    {
    }

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

    virtual void genRaggedMatrix(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, 
                                 const SynapseGroup &sg) const override;
                                 
    virtual void genEmitTrueSpike(CodeStream &os, const NNmodel&, const NeuronGroup&, const Substitutions &subs) const override
    {
        genEmitSpike(os, subs, "");
    }
    
    virtual void genEmitSpikeLikeEvent(CodeStream &os, const NNmodel &, const NeuronGroup &, const Substitutions &subs) const override
    {
        genEmitSpike(os, subs, "Evnt");
    }

    virtual std::string getVarPrefix() const override{ return ""; }

    virtual const std::vector<FunctionTemplate> &getFunctions() const override{ return cpuFunctions; }

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    /*void genParallelNeuronGroup(CodeStream &os, const std::map<std::string, NeuronGroup> &ngs, const Substitutions &subs, 
                                std::function<bool(const NeuronGroup &)> filter,
                                std::function<void(CodeStream &, const ::CodeGenerator::Base&, const NeuronGroup&, const Substitutions &)> handler) const;

    void genParallelNeuronGroup(CodeStream &os, const std::map<std::string, NeuronGroup> &ngs, const Substitutions &subs, 
                                std::function<void(CodeStream &, const ::CodeGenerator::Base&, const NeuronGroup&, const Substitutions &)> handler) const
    {
        genParallelNeuronGroup(os, ngs, subs, [](const NeuronGroup&){ return true; }, handler);
    }

    void genParallelSynapseGroup(CodeStream &os, const NNmodel &model, const Substitutions &subs, 
                                 std::function<size_t(const SynapseGroup&)> getPaddedSizeFunc,
                                 std::function<bool(const SynapseGroup &)> filter,
                                 std::function<void(CodeStream &, const ::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, const Substitutions &)> handler) const;

    void genParallelSynapseGroup(CodeStream &os, const NNmodel &model, const Substitutions &subs, 
                                 std::function<size_t(const SynapseGroup&)> getPaddedSizeFunc,
                                 std::function<void(CodeStream &, const ::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, const Substitutions &)> handler) const
    {
        genParallelSynapseGroup(os, model, subs, getPaddedSizeFunc, [](const SynapseGroup&){ return true; }, handler);
    }*/
                                 
    void genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const;

  
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const int m_LocalHostID;
};
}   // SingleThreadedCPU

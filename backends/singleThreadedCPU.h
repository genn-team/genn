#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>

// NuGeNN includes
#include "base.h"

//--------------------------------------------------------------------------
// CodeGenerator::Backends::SingleThreadedCPU
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace Backends
{
class SingleThreadedCPU : public Base
{
public:
    SingleThreadedCPU(int localHostID)
    :   m_LocalHostID(localHostID)
    {
    }

    //--------------------------------------------------------------------------
    // CodeGenerator::Backends::Base virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, const NNmodel &model, NeuronGroupHandler handler) const override;

    virtual void genPresynapticUpdate(CodeStream &os, const NNmodel &model,
                                      SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const override;

    virtual void genInit(CodeStream &os, const NNmodel &model, NeuronGroupHandler ngHandler,
                         SynapseGroupHandler sgDenseVarHandler, SynapseGroupHandler sgSparseConnectHandler) const override;
    virtual void genInitSparse(CodeStream &os, const NNmodel &model, SynapseGroupHandler sgHandler) const override;

    virtual void genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const override;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const override;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const override;
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarMode mode) const override;

    virtual void genVariableInit(CodeStream &os, VarMode mode, size_t count, const std::string &countVarName,
                                 const Substitutions &kernelSubs, Handler handler) const override;

    virtual void genEmitTrueSpike(CodeStream &os, const NNmodel&, const NeuronGroup&, const Substitutions &subs) const override
    {
        genEmitSpike(os, subs, "");
    }
    
    virtual void genEmitSpikeLikeEvent(CodeStream &os, const NNmodel &, const NeuronGroup &, const Substitutions &subs) const override
    {
        genEmitSpike(os, subs, "Evnt");
    }

    virtual std::string getVarPrefix() const override{ return ""; }

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
}   // namespace Backends
}   // namespace CodeGenerator

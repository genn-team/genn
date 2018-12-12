#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "codeGenUtils.h"
#include "global.h"

// Forward declarations
class CodeStream;
class NeuronGroup;
class NNmodel;
class Substitutions;
class SynapseGroup;

//--------------------------------------------------------------------------
// CodeGenerator::Backends::Base
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace Backends
{
class Base
{
public:
    //--------------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------------
    typedef std::function<void(CodeStream &, Substitutions&)> Handler;
    
    template<typename T>
    using GroupHandler = std::function < void(CodeStream &, const T &, Substitutions&) > ;

    typedef GroupHandler<NeuronGroup> NeuronGroupHandler;
    typedef GroupHandler<SynapseGroup> SynapseGroupHandler;
    
    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, const NNmodel &model, NeuronGroupHandler handler) const = 0;
    virtual void genSynapseUpdate(CodeStream &os, const NNmodel &model,
                                  SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const = 0;

    virtual void genInit(CodeStream &os, const NNmodel &model,
                         NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                         SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                         SynapseGroupHandler sgSparseInitHandler) const = 0;

    virtual void genRunnerPreamble(CodeStream &os) const = 0;

    virtual void genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const = 0;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const = 0;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const = 0; 
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarMode mode) const = 0;

    virtual void genPopVariableInit(CodeStream &os, VarMode mode, const Substitutions &kernelSubs, Handler handler) const = 0;
    virtual void genVariableInit(CodeStream &os, VarMode mode, size_t count, const std::string &countVarName,
                                 const Substitutions &kernelSubs, Handler handler) const = 0;

    virtual void genEmitTrueSpike(CodeStream &os, const NNmodel &model, const NeuronGroup &ng, const Substitutions &subs) const = 0;
    
    virtual void genEmitSpikeLikeEvent(CodeStream &os, const NNmodel &model, const NeuronGroup &ng, const Substitutions &subs) const = 0;

    virtual std::string getVarPrefix() const{ return ""; }

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    void genArray(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                  const std::string &type, const std::string &name, VarMode mode, size_t count) const
    {
        genVariableDefinition(definitions, type + "*", name, mode);
        genVariableImplementation(runner, type + "*", name, mode);
        genVariableAllocation(allocations, type, name, mode, count);
        genVariableFree(free, name, mode);
    }

protected:
    std::string getVarExportPrefix() const
    {
        // In windows making variables extern isn't enough to export then as DLL symbols - you need to add __declspec(dllexport)
#ifdef _WIN32
        return GENN_PREFERENCES::buildSharedLibrary ? "__declspec(dllexport) extern" : "extern";
#else
        return "extern";
#endif
    }
};
}   // namespace Backends
}   // namespace CodeGenerator

#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "codeGenUtils.h"
#include "codeStream.h"
#include "global.h"

// Forward declarations
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
                                  SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler,
                                  SynapseGroupHandler postLearnHandler) const = 0;

    virtual void genInit(CodeStream &os, const NNmodel &model,
                         NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                         SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                         SynapseGroupHandler sgSparseInitHandler) const = 0;

    virtual void genDefinitionsPreamble(CodeStream &os) const = 0;
    virtual void genRunnerPreamble(CodeStream &os) const = 0;

    virtual void genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const = 0;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const = 0;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const = 0; 
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarMode mode) const = 0;

    virtual void genPopVariableInit(CodeStream &os, VarMode mode, const Substitutions &kernelSubs, Handler handler) const = 0;
    virtual void genVariableInit(CodeStream &os, VarMode mode, size_t count, const std::string &countVarName,
                                 const Substitutions &kernelSubs, Handler handler) const = 0;

    virtual void genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, bool autoInitialized, size_t count) const = 0;
    virtual void genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const = 0;
    virtual void genCurrentTrueSpikePush(CodeStream &os, const NeuronGroup &ng) const = 0;
    virtual void genCurrentTrueSpikePull(CodeStream &os, const NeuronGroup &ng) const = 0;
    virtual void genCurrentSpikeLikeEventPush(CodeStream &os, const NeuronGroup &ng) const = 0;
    virtual void genCurrentSpikeLikeEventPull(CodeStream &os, const NeuronGroup &ng) const = 0;

    virtual void genGlobalRNG(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free, const NNmodel &model) const = 0;
    virtual void genPopulationRNG(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                                  const std::string &name, size_t count) const = 0;

    virtual void genMakefilePreamble(std::ostream &os) const = 0;
    virtual void genMakefileLinkRule(std::ostream &os) const = 0;
    virtual void genMakefileCompileRule(std::ostream &os) const = 0;

    virtual void genEmitTrueSpike(CodeStream &os, const NNmodel &model, const NeuronGroup &ng, const Substitutions &subs) const = 0;
    
    virtual void genEmitSpikeLikeEvent(CodeStream &os, const NNmodel &model, const NeuronGroup &ng, const Substitutions &subs) const = 0;

    virtual std::string getVarPrefix() const{ return ""; }

    virtual bool isGlobalRNGRequired(const NNmodel &model) const = 0;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    void genVariablePushPull(CodeStream &push, CodeStream &pull,
                             const std::string &type, const std::string &name, VarMode mode, bool autoInitialized, size_t count) const
    {
        genVariablePush(push, type, name, mode, autoInitialized, count);
        genVariablePull(pull, type, name, mode, count);
    }

    void genArray(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                  const std::string &type, const std::string &name, VarMode mode, size_t count) const
    {
        genVariableDefinition(definitions, type + "*", name, mode);
        genVariableImplementation(runner, type + "*", name, mode);
        genVariableAllocation(allocations, type, name, mode, count);
        genVariableFree(free, name, mode);
    }

    void genVariable(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                     CodeStream &push, CodeStream &pull,
                     const std::string &type, const std::string &name, VarMode mode, bool autoInitialized, size_t count) const
    {
        genArray(definitions, runner, allocations, free, type, name, mode, count);
        genVariablePushPull(push, pull, type, name, mode, autoInitialized, count);
    }

protected:
    //--------------------------------------------------------------------------
    // Protected API
    //--------------------------------------------------------------------------
    void genGLIBCBugTest(CodeStream &os) const
    {
        // **NOTE** if we are using GCC on x86_64, bugs in some version of glibc can cause bad performance issues.
        // Best solution involves setting LD_BIND_NOW=1 so check whether this has been applied
        os << "#if defined(__GNUG__) && !defined(__clang__) && defined(__x86_64__) && __GLIBC__ == 2 && (__GLIBC_MINOR__ == 23 || __GLIBC_MINOR__ == 24)" << std::endl;
        os << "if(std::getenv(\"LD_BIND_NOW\") == NULL)";
        {
            CodeStream::Scope b(os);
            os << "std::cerr << \"Warning: a bug has been found in glibc 2.23 or glibc 2.24 (https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280) \";" << std::endl;
            os << "std::cerr << \"which results in poor CPU maths performance. We recommend setting the environment variable LD_BIND_NOW=1 to work around this issue.\" << std::endl;" << std::endl;
        }
        os << "#endif" << std::endl;
    }

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

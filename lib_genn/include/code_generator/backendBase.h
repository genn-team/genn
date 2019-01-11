#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "codeStream.h"
#include "variableMode.h"

// Forward declarations
class NeuronGroup;
class NNmodel;
class SynapseGroup;

namespace CodeGenerator
{
    class Substitutions;
}

//--------------------------------------------------------------------------
// CodeGenerator::BackendBase
//--------------------------------------------------------------------------
namespace CodeGenerator
{
class BackendBase
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
    // Preferences
    //--------------------------------------------------------------------------
    //! Base class for backend preferences - can be accessed via a global in 'classic' C++ code generator
    struct Preferences
    {
        //! Generate speed-optimized code, potentially at the expense of floating-point accuracy
        bool optimizeCode; 

        //! Generate code with debug symbols
        bool debugCode; 
        
        //! C++ compiler options to be used for building all host side code (used for unix based platforms)
        std::string userCxxFlagsGNU; 
        
        //!< NVCC compiler options they may want to use for all GPU code (used for unix based platforms)
        std::string userNvccFlagsGNU; 
    };

    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, const NNmodel &model, NeuronGroupHandler handler) const = 0;
    virtual void genSynapseUpdate(CodeStream &os, const NNmodel &model,
                                  SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler,
                                  SynapseGroupHandler postLearnHandler, SynapseGroupHandler synapseDynamicsHandler) const = 0;

    virtual void genInit(CodeStream &os, const NNmodel &model,
                         NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                         SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                         SynapseGroupHandler sgSparseInitHandler) const = 0;

    virtual void genDefinitionsPreamble(CodeStream &os) const = 0;
    virtual void genRunnerPreamble(CodeStream &os) const = 0;
    virtual void genAllocateMemPreamble(CodeStream &os, const NNmodel &model) const = 0;

    virtual void genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const = 0; 
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const = 0;

    virtual void genPopVariableInit(CodeStream &os, VarLocation loc, const Substitutions &kernelSubs, Handler handler) const = 0;
    virtual void genVariableInit(CodeStream &os, VarLocation loc, size_t count, const std::string &countVarName,
                                 const Substitutions &kernelSubs, Handler handler) const = 0;

    virtual void genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const = 0;
    virtual void genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const = 0;
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
                             const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const
    {
        genVariablePush(push, type, name, loc, autoInitialized, count);
        genVariablePull(pull, type, name, loc, count);
    }

    void genArray(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                  const std::string &type, const std::string &name, VarLocation loc, size_t count) const
    {
        genVariableDefinition(definitions, type + "*", name, loc);
        genVariableImplementation(runner, type + "*", name, loc);
        genVariableAllocation(allocations, type, name, loc, count);
        genVariableFree(free, name, loc);
    }

    void genVariable(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                     CodeStream &push, CodeStream &pull,
                     const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const
    {
        genArray(definitions, runner, allocations, free, type, name, loc, count);
        genVariablePushPull(push, pull, type, name, loc, autoInitialized, count);
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
        return "__declspec(dllexport) extern";
#else
        return "extern";
#endif
    }
};
}   // namespace CodeGenerator

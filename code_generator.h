#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "codeGenUtils.h"

// Forward declarations
class CodeStream;
class NeuronGroup;
class NNmodel;
class Substitutions;
class SynapseGroup;

//--------------------------------------------------------------------------
// CodeGenerator::Base
//--------------------------------------------------------------------------
namespace CodeGenerator
{
class Base
{
public:
    //--------------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------------
    typedef std::function<void(CodeStream &, const NeuronGroup &, Substitutions&)> NeuronGroupHandler;
    typedef std::function<void(CodeStream &, const SynapseGroup &, Substitutions&)> SynapseGroupHandler;
    
    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdateKernel(CodeStream &os, const NNmodel &model, NeuronGroupHandler handler) const = 0;
    virtual void genPresynapticUpdateKernel(CodeStream &os, const NNmodel &model,
                                            SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const = 0;

    virtual void genInitKernel(CodeStream &os, const NNmodel &model, NeuronGroupHandler ngHandler, SynapseGroupHandler sgHandler) const = 0;

    virtual void genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const = 0;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const = 0;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const = 0; 

    virtual void genEmitTrueSpike(CodeStream &os, const NNmodel &model, const NeuronGroup &ng, const Substitutions &subs) const = 0;
    
    virtual void genEmitSpikeLikeEvent(CodeStream &os, const NNmodel &model, const NeuronGroup &ng, const Substitutions &subs) const = 0;

    virtual std::string getVarPrefix() const{ return ""; }

    virtual const std::vector<FunctionTemplate> &getFunctions() const = 0;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    void genArray(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, 
                  const std::string &type, const std::string &name, VarMode mode, size_t count) const
    {
        genVariableDefinition(definitions, type + "*", name, mode);
        genVariableImplementation(runner, type + "*", name, mode);
        genVariableAllocation(allocations, type, name, mode, count);
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
    
};

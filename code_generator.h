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

//--------------------------------------------------------------------------
// CodeGenerator::Base
//--------------------------------------------------------------------------
namespace CodeGenerator
{
class Base
{
public:
    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdateKernel(CodeStream &os, const NNmodel &model,
                                       std::function<void(CodeStream &output, const Base &, const NNmodel &, const NeuronGroup&, const Substitutions&)> handler) const = 0;
    virtual void genPresynapticUpdateKernel(CodeStream &os, const NNmodel &model) const = 0;

    virtual void genEmitTrueSpike(CodeStream &os, const NNmodel &model, const NeuronGroup &ng, const Substitutions &subs) const = 0;
    
    virtual void genEmitSpikeLikeEvent(CodeStream &os, const NNmodel &model, const NeuronGroup &ng, const Substitutions &subs) const = 0;

    virtual std::string getVarPrefix() const{ return ""; }

    virtual const std::vector<FunctionTemplate> &getFunctions() const = 0;
};
    
};
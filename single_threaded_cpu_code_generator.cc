#include "single_threaded_cpu_code_generator.h"

// GeNN includes
#include "codeStream.h"
#include "global.h"
#include "modelSpec.h"
#include "utils.h"

// NuGeNN includes
#include "substitution_stack.h"

//--------------------------------------------------------------------------
// SingleThreadedCPU::CodeGenerator
//--------------------------------------------------------------------------
namespace SingleThreadedCPU
{
void CodeGenerator::genNeuronUpdateKernel(CodeStream &os, const NNmodel &model, NeuronGroupHandler handler) const
{
    USE(os);
    USE(model);
    USE(handler);
    assert(false);
}

void CodeGenerator::genPresynapticUpdateKernel(CodeStream &os, const NNmodel &model,
                                               SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const
{
    USE(os);
    USE(model);
    USE(wumThreshHandler);
    USE(wumSimHandler);
    assert(false);
}

void CodeGenerator::genInitKernel(CodeStream &os, const NNmodel &model,
                                  NeuronGroupHandler ngHandler, SynapseGroupHandler sgHandler) const
{
    USE(os);
    USE(model);
    USE(ngHandler);
    USE(sgHandler);
    assert(false);
}

void CodeGenerator::genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode) const
{
    os << getVarExportPrefix() << " " << type << " " << name << ";" << std::endl;
}
void CodeGenerator::genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode) const
{
    os << type << " " << name << ";" << std::endl;
}

void CodeGenerator::genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode, size_t count) const
{
    os << name << " = new " << type << "[" << count << "];" << std::endl;
}

void CodeGenerator::genVariableInit(CodeStream &os, VarMode mode, size_t count, const Substitutions &kernelSubs, Handler handler) const
{
    // **TODO** loops like this should be generated like CUDA threads
    os << "for (unsigned i = 0; i < " << count << "; i++)";
    {
        CodeStream::Scope b(os);

        // If variable should be initialised on device
        Substitutions varSubs(&kernelSubs);
        varSubs.addVarSubstitution("id", "i");
        handler(os, varSubs);
    }
}

void CodeGenerator::genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const
{
    USE(os);
    USE(subs);
    USE(suffix);
    assert(false);
}
}   // SingleThreadedCPU

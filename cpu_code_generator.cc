#include "cpu_code_generator.h"

// GeNN includes
#include "codeStream.h"
#include "global.h"
#include "utils.h"

// NuGeNN includes
#include "substitution_stack.h"

//--------------------------------------------------------------------------
// SingleThreadedCPU::CodeGenerator
//--------------------------------------------------------------------------
namespace SingleThreadedCPU
{
void CodeGenerator::genNeuronUpdateKernel(CodeStream &os, const NNmodel &model,
                                          std::function<void(CodeStream&, const ::CodeGenerator::Base&, const NNmodel&, const NeuronGroup &ng, Substitutions&)> handler) const
{
    USE(os);
    USE(model);
    USE(handler);
    assert(false);
}

void CodeGenerator::genPresynapticUpdateKernel(CodeStream &os, const NNmodel &model,
                                               std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const SynapseGroup &, const Substitutions&)> wumThreshHandler,
                                               std::function<void(CodeStream&, const::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, const Substitutions&)> wumSimHandler) const
{
    USE(os);
    USE(model);
    USE(wumThreshHandler);
    USE(wumSimHandler);
    assert(false);
}

void CodeGenerator::genInitKernel(CodeStream &os, const NNmodel &model,
                                  std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const NeuronGroup &, const Substitutions&)> ngHandler,
                                  std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const SynapseGroup &, const Substitutions&)> sgHandler) const
{
    USE(os);
    USE(model);
    USE(ngHandler);
    USE(sgHandler);
    assert(false);
}

void CodeGenerator::genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode) const
{
    // In windows making variables extern isn't enough to export then as DLL symbols - you need to add __declspec(dllexport)
#ifdef _WIN32
    const std::string varExportPrefix = GENN_PREFERENCES::buildSharedLibrary ? "__declspec(dllexport) extern" : "extern";
#else
    const std::string varExportPrefix = "extern";
#endif
    
    os << varExportPrefix << " " << type << " " << name << ";" << std::endl;
}
void CodeGenerator::genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode) const
{
    os << type << " " << name << ";" << std::endl;
}

void CodeGenerator::genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode, size_t count) const
{
    os << name << " = new " << type << "[" << count << "];" << std::endl;
}

void CodeGenerator::genRaggedMatrix(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, 
                                    const SynapseGroup &sg) const
{
}

void CodeGenerator::genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const
{
    USE(os);
    USE(subs);
    USE(suffix);
    assert(false);
}
}   // SingleThreadedCPU

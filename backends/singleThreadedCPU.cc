#include "singleThreadedCPU.h"

// GeNN includes
#include "codeStream.h"
#include "global.h"
#include "modelSpec.h"
#include "utils.h"

// NuGeNN includes
#include "../substitution_stack.h"

//--------------------------------------------------------------------------
// CodeGenerator::Backends::SingleThreadedCPU
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace Backends
{
void SingleThreadedCPU::genNeuronUpdate(CodeStream &os, const NNmodel &model, NeuronGroupHandler handler) const
{
    USE(os);
    USE(model);
    USE(handler);
    assert(false);
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genPresynapticUpdate(CodeStream &os, const NNmodel &model,
                                             SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const
{
    USE(os);
    USE(model);
    USE(wumThreshHandler);
    USE(wumSimHandler);
    assert(false);
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genInit(CodeStream &os, const NNmodel &model,
                                NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                                SynapseGroupHandler sgDenseVarHandler, SynapseGroupHandler sgSparseConnectHandler) const
{
    USE(os);
    USE(model);
    USE(localNGHandler);
    USE(remoteNGHandler);
    USE(sgDenseVarHandler);
    USE(sgSparseConnectHandler);
    assert(false);
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genInitSparse(CodeStream &os, const NNmodel &model, SynapseGroupHandler sgHandler) const
{
    USE(os);
    USE(model);
    USE(sgHandler);
    assert(false);
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode) const
{
    os << getVarExportPrefix() << " " << type << " " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode) const
{
    os << type << " " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode, size_t count) const
{
    os << name << " = new " << type << "[" << count << "];" << std::endl;
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genVariableFree(CodeStream &os, const std::string &name, VarMode) const
{
    os << "delete[] " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genPopVariableInit(CodeStream &os, VarMode mode, const Substitutions &kernelSubs, Handler handler) const
{
    Substitutions varSubs(&kernelSubs);
    handler(os, varSubs);
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genVariableInit(CodeStream &os, VarMode, size_t count, const std::string &countVarName,
                                        const Substitutions &kernelSubs, Handler handler) const
{
    // **TODO** loops like this should be generated like CUDA threads
    os << "for (unsigned i = 0; i < " << count << "; i++)";
    {
        CodeStream::Scope b(os);

        Substitutions varSubs(&kernelSubs);
        varSubs.addVarSubstitution(countVarName, "i");
        handler(os, varSubs);
    }
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const
{
    USE(os);
    USE(subs);
    USE(suffix);
    assert(false);
}
}   // namespace Backends
}   // namespace CodeGenerator
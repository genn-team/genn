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
void SingleThreadedCPU::genSynapseUpdate(CodeStream &os, const NNmodel &model,
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
                                SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                                SynapseGroupHandler sgSparseInitHandler) const
{
    USE(os);
    USE(model);
    USE(localNGHandler);
    USE(remoteNGHandler);
    USE(sgDenseInitHandler);
    USE(sgSparseConnectHandler);
    USE(sgSparseInitHandler);
    assert(false);
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genDefinitionsPreamble(CodeStream &) const
{
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genRunnerPreamble(CodeStream &) const
{
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
void SingleThreadedCPU::genPopVariableInit(CodeStream &os, VarMode, const Substitutions &kernelSubs, Handler handler) const
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
void SingleThreadedCPU::genVariablePush(CodeStream&, const std::string&, const std::string&, VarMode, bool, size_t) const
{
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genVariablePull(CodeStream&, const std::string&, const std::string&, VarMode, size_t) const
{
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genCurrentTrueSpikePush(CodeStream&, const NeuronGroup&) const
{
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genCurrentTrueSpikePull(CodeStream&, const NeuronGroup&) const
{
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genCurrentSpikeLikeEventPush(CodeStream&, const NeuronGroup&) const
{
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genCurrentSpikeLikeEventPull(CodeStream&, const NeuronGroup&) const
{
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genGlobalRNG(CodeStream &definitions, CodeStream &runner, CodeStream &, CodeStream &, const NNmodel &model) const
{
    definitions << getVarExportPrefix() << " " << "std::mt19937 rng;" << std::endl;
    runner << "std::mt19937 rng;" << std::endl;

    // Define and implement standard host distributions as recreating them each call is slow
    definitions << getVarExportPrefix() << " " << "std::uniform_real_distribution<" << model.getPrecision() << "> standardUniformDistribution;" << std::endl;
    definitions << getVarExportPrefix() << " " << "std::normal_distribution<" << model.getPrecision() << "> standardNormalDistribution;" << std::endl;
    definitions << getVarExportPrefix() << " " << "std::exponential_distribution<" << model.getPrecision() << "> standardExponentialDistribution;" << std::endl;
    runner << "std::uniform_real_distribution<" << model.getPrecision() << "> standardUniformDistribution(" << model.scalarExpr(0.0) << ", " << model.scalarExpr(1.0) << ");" << std::endl;
    runner << "std::normal_distribution<" << model.getPrecision() << "> standardNormalDistribution(" << model.scalarExpr(0.0) << ", " << model.scalarExpr(1.0) << ");" << std::endl;
    runner << "std::exponential_distribution<" << model.getPrecision() << "> standardExponentialDistribution(" << model.scalarExpr(1.0) << ");" << std::endl;
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genPopulationRNG(CodeStream &, CodeStream &, CodeStream &, CodeStream &,
                                         const std::string&, size_t) const
{
    // No need for population RNGs for single-threaded CPU
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const
{
    USE(os);
    USE(subs);
    USE(suffix);
    assert(false);
}
//--------------------------------------------------------------------------
bool SingleThreadedCPU::isGlobalRNGRequired(const NNmodel &model) const
{
    // **TODO** move logic from method in here as it is backend-logic specific
    return model.isHostRNGRequired();
}
}   // namespace Backends
}   // namespace CodeGenerator
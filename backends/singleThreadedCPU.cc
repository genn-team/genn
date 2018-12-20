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
    os << "void updateNeurons(" << model.getTimePrecision() << " t)";
    {
        CodeStream::Scope b(os);

        Substitutions funcSubs(cpuFunctions);
        funcSubs.addVarSubstitution("t", "t");

        // function code
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "// neuron group " << n.first << std::endl;
            {
                CodeStream::Scope b(os);

                // **TODO** this is kinda generic
                if (n.second.isDelayRequired()) { // with delay
                    if (n.second.isSpikeEventRequired()) {
                        os << "glbSpkCntEvnt" << n.first << "[spkQuePtr" << n.first << "] = 0;" << std::endl;
                    }
                    if (n.second.isTrueSpikeRequired()) {
                        os << "glbSpkCnt" << n.first << "[spkQuePtr" << n.first << "] = 0;" << std::endl;
                    }
                    else {
                        os << "glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
                    }
                }
                else { // no delay
                    if (n.second.isSpikeEventRequired()) {
                        os << "glbSpkCntEvnt" << n.first << "[0] = 0;" << std::endl;
                    }
                    os << "glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
                }

                // If axonal delays are required
                if (n.second.isDelayRequired()) {
                    // We should READ from delay slot before spkQuePtr
                    os << "const unsigned int readDelayOffset = " << n.second.getPrevQueueOffset("") << ";" << std::endl;

                    // And we should WRITE to delay slot pointed to be spkQuePtr
                    os << "const unsigned int writeDelayOffset = " << n.second.getCurrentQueueOffset("") << ";" << std::endl;
                }
                os << std::endl;

                os << "for(unsigned int i = 0; i < " <<  n.second.getNumNeurons() << "; i++)";
                {
                    CodeStream::Scope b(os);

                    Substitutions popSubs(&funcSubs);
                    popSubs.addVarSubstitution("id", "i");

                    handler(os, n.second, popSubs);
                }
            }
        }

    }
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genSynapseUpdate(CodeStream &os, const NNmodel &model,
                                         SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler,
                                         SynapseGroupHandler postLearnHandler, SynapseGroupHandler synapseDynamicsHandler) const
{
    os << "void updateSynapses(" << model.getTimePrecision() << " t)";
    {
        Substitutions funcSubs(cpuFunctions);
        funcSubs.addVarSubstitution("t", "t");

        CodeStream::Scope b(os);
        for(const auto &s : model.getLocalSynapseGroups()) {
            os << "// synapse group " << s.first << std::endl;
            {
                CodeStream::Scope b(os);

                // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                if(s.second.getSrcNeuronGroup()->isDelayRequired()) {
                    os << "const unsigned int preReadDelaySlot = " << s.second.getPresynapticAxonalDelaySlot("") << ";" << std::endl;
                    os << "const unsigned int preReadDelayOffset = preReadDelaySlot * " << s.second.getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;
                }

                // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                if(s.second.getTrgNeuronGroup()->isDelayRequired()) {
                    os << "const unsigned int postReadDelayOffset = " << s.second.getPostsynapticBackPropDelaySlot("") << " * " << s.second.getTrgNeuronGroup()->getNumNeurons() << ";" << std::endl;
                }

                // generate the code for processing spike-like events
                if (s.second.isSpikeEventRequired()) {
                    genPresynapticUpdate(os, s.second, funcSubs, false, wumThreshHandler, wumSimHandler);
                }

                // generate the code for processing true spike events
                if (s.second.isTrueSpikeRequired()) {
                    genPresynapticUpdate(os, s.second, funcSubs, true, wumThreshHandler, wumSimHandler);
                }
            }
            os << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genInit(CodeStream &os, const NNmodel &model,
                                NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                                SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                                SynapseGroupHandler sgSparseInitHandler) const
{
    os << "void initialize()";
    {
        CodeStream::Scope b(os);
        Substitutions funcSubs(cpuFunctions);

        // Generate test for GLIBC test
        genGLIBCBugTest(os);

        // If model requires a host RNG
        if(true/*model.isHostRNGRequired()*/) {
            // If no seed is specified, use system randomness to generate seed sequence
            CodeStream::Scope b(os);
            if (model.getSeed() == 0) {
                os << "uint32_t seedData[std::mt19937::state_size];" << std::endl;
                os << "std::random_device seedSource;" << std::endl;
                {
                    CodeStream::Scope b(os);
                    os << "for(int i = 0; i < std::mt19937::state_size; i++)";
                    {
                        CodeStream::Scope b(os);
                        os << "seedData[i] = seedSource();" << std::endl;
                    }
                }
                os << "std::seed_seq seeds(std::begin(seedData), std::end(seedData));" << std::endl;
            }
            // Otherwise, create a seed sequence from model seed
            // **NOTE** this is a terrible idea see http://www.pcg-random.org/posts/cpp-seeding-surprises.html
            else {
                os << "std::seed_seq seeds{" << model.getSeed() << "};" << std::endl;
            }

            // Seed RNG from seed sequence
            os << "rng.seed(seeds);" << std::endl;

            // Add RNG to substitutions
            funcSubs.addVarSubstitution("rng", "rng");
        }
        os << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Remote neuron groups" << std::endl;
        for(const auto &n : model.getRemoteNeuronGroups()) {
            if(n.second.hasOutputToHost(m_LocalHostID)) {
                os << "// neuron group " << n.first << std::endl;
                CodeStream::Scope b(os);
                remoteNGHandler(os, n.second, funcSubs);
            }
        }
        os << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Local neuron groups" << std::endl;
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "// neuron group " << n.first << std::endl;
            CodeStream::Scope b(os);
            localNGHandler(os, n.second, funcSubs);
        }

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with dense connectivity" << std::endl;
        for(const auto &s : model.getLocalSynapseGroups()) {
            if((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
                os << "// synapse group " << s.first << std::endl;
                CodeStream::Scope b(os);
                sgDenseInitHandler(os, s.second, funcSubs);
            }
        }

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with sparse connectivity" << std::endl;
        for(const auto &s : model.getLocalSynapseGroups()) {
            // If this synapse group has a connectivity initialisation snippet
            if(!s.second.getConnectivityInitialiser().getSnippet()->getRowBuildCode().empty()) {
                os << "// synapse group " << s.first << std::endl;
                const size_t numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
                const size_t numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();

                // If matrix connectivity is ragged
                if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                    const std::string rowLength = "rowLength" + s.first;
                    const std::string ind = "ind" + s.first;

                    // Zero row lengths
                    os << "memset(" << rowLength << ", 0, " << numSrcNeurons << " * sizeof(unsigned int));" << std::endl;

                    // Loop through source neurons
                    os << "for (unsigned int i = 0; i < " << numSrcNeurons << "; i++)";
                    {
                        CodeStream::Scope b(os);

                        Substitutions popSubs(&funcSubs);
                        popSubs.addVarSubstitution("id_pre", "i");

                        // Add function to increment row length and insert synapse into ind array
                        popSubs.addFuncSubstitution("addSynapse", 1,
                                                    ind + "[(i * " + std::to_string(s.second.getMaxConnections()) + ") + (" + rowLength + "[i]++)] = $(0)");

                        sgSparseConnectHandler(os, s.second, popSubs);
                    }

                }
                // Otherwise, if matrix connectivity is a bitmask
                else if(s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    // Zero memory before setting sparse bits
                    os << "memset(gp" << s.first << ", 0, " << (numSrcNeurons * numTrgNeurons) / 32 + 1 << " * sizeof(uint32_t));" << std::endl;

                    // Loop through source neurons
                    os << "for(unsigned int i = 0; i < " << numSrcNeurons << "; i++)";
                    {
                        // Calculate index of bit at start of this row
                        CodeStream::Scope b(os);
                        os << "const int64_t rowStartGID = i * " << numTrgNeurons << "ll;" << std::endl;

                        // Build function template to set correct bit in bitmask
                        Substitutions popSubs(&funcSubs);
                        popSubs.addVarSubstitution("id_pre", "i");

                        // Add function to increment row length and insert synapse into ind array
                        popSubs.addFuncSubstitution("addSynapse", 1,
                                                     "setB(gp" + s.first + "[(rowStartGID + $(0)) / 32], (rowStartGID + $(0)) & 31)");

                        sgSparseConnectHandler(os, s.second, popSubs);
                    }
                }
                else {
                    gennError("Only BITMASK and RAGGED format connectivity can be generated using a connectivity initialiser");
                }
            }
        }
    }
    os << std::endl;
    os << "void initializeSparse()";
    {
        CodeStream::Scope b(os);
        Substitutions funcSubs(cpuFunctions);

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with sparse connectivity" << std::endl;
        for(const auto &s : model.getLocalSynapseGroups()) {
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                if(model.isSynapseGroupDynamicsRequired(s.first)) {
                    assert(false);
                }
                if (model.isSynapseGroupPostLearningRequired(s.first)) {
                    assert(false);
                }

                /*popSubs.addVarSubstitution("id_syn", "idx");
                popSubs.addVarSubstitution("id_pre", "((r * " + std::to_string(m_InitSparseBlockSize) + ") + i)");
                popSubs.addVarSubstitution("id_post", "dd_ind" + sg.getName() + "[idx]");
                sgSparseInitHandler(os, sg, popSubs);*/
            }
        }
    }
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genDefinitionsPreamble(CodeStream &os) const
{
    os << "// Standard C++ includes" << std::endl;
    os << "#include <iostream>" << std::endl;
    os << "#include <random>" << std::endl;
    os << std::endl;
    os << "// Standard C includes" << std::endl;
    os << "#include <cmath>" << std::endl;
    os << "#include <cstring>" << std::endl;
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
void SingleThreadedCPU::genMakefilePreamble(std::ostream &os) const
{
    std::string linkFlags = "-shared -fPIC";
    std::string cxxFlags = "-c -fPIC -DCPU_ONLY -std=c++11 -MMD -MP";
    cxxFlags += " " + GENN_PREFERENCES::userCxxFlagsGNU;
    if (GENN_PREFERENCES::optimizeCode) {
        cxxFlags += " -O3 -ffast-math";
    }
    if (GENN_PREFERENCES::debugCode) {
        cxxFlags += " -O0 -g";
    }

#ifdef MPI_ENABLE
    // If MPI is enabled, add MPI include path
    cxxFlags +=" -I\"$(MPI_PATH)/include\"";
#endif

    // Write variables to preamble
    os << "CXXFLAGS := " << cxxFlags << std::endl;
    os << "LINKFLAGS := " << linkFlags << std::endl;

    os << std::endl;
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genMakefileLinkRule(std::ostream &os) const
{
    os << "\t$(CXX) $(LINKFLAGS) -o $@ $(OBJECTS)" << std::endl;
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genMakefileCompileRule(std::ostream &os) const
{
    os << "%.o: %.cc %.d" << std::endl;
    os << "\t$(CXX) $(CXXFLAGS) -o $@ $<" << std::endl;
}
//--------------------------------------------------------------------------
bool SingleThreadedCPU::isGlobalRNGRequired(const NNmodel &model) const
{
    // **TODO** move logic from method in here as it is backend-logic specific
    return true;//model.isHostRNGRequired();
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genPresynapticUpdate(CodeStream &os, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
                                             SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "evnt";
    const auto *wu = sg.getWUModel();

    // Detect spike events or spikes and do the update
    os << "// process presynaptic events: " << (trueSpike ? "True Spikes" : "Spike type events") << std::endl;
    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "for (unsigned int i = 0; i < glbSpkCnt" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[preReadDelaySlot]; i++)";
    }
    else {
        os << "for (unsigned int i = 0; i < glbSpkCnt" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[0]; i++)";
    }
    {
        CodeStream::Scope b(os);

        const std::string queueOffset = sg.getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
        os << "const unsigned int ipre = glbSpk" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[" << queueOffset << "i];" << std::endl;

        if (sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
            os << "const unsigned int npost = rowLength" << sg.getName() << "[ipre];" << std::endl;
            os << "for (unsigned int j = 0; j < npost; j++)";
        }
        // Otherwise (DENSE or BITMASK)
        else {
            os << "for (unsigned int ipost = 0; ipost < " << sg.getTrgNeuronGroup()->getNumNeurons() << "; ipost++)";
        }
        {
            CodeStream::Scope b(os);
            if(sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                // **TODO** seperate stride from max connections
                os << "const unsigned int ipost = ind" << sg.getName() << "[(ipre * " << sg.getMaxConnections() << ") + j];" << std::endl;
            }
            else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                os << "const uint64_t gid = (ipre * " << sg.getTrgNeuronGroup()->getNumNeurons() << "ull + ipost);" << std::endl;
            }

            if (!wu->getSimSupportCode().empty()) {
                os << " using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
            }

            if (!trueSpike) {
                os << "if(";
                if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "(B(gp" << sg.getName() << "[gid / 32], gid & 31)) && ";
                }

                Substitutions threshSubs(&popSubs);
                threshSubs.addVarSubstitution("id_pre", "ipre");
                threshSubs.addVarSubstitution("id_post", "ipost");

                // Generate weight update threshold condition
                wumThreshHandler(os, sg, threshSubs);

                os << ")";
                os << CodeStream::OB(2041);
            }
            else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                os << "if (B(gp" << sg.getName() << "[gid / 32], gid & 31))" << CodeStream::OB(2041);
            }


            if(sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                os << "const unsigned int synAddress = (ipre * " + std::to_string(sg.getMaxConnections()) + ") + j;" << std::endl;
            }
            else {
                os << "const unsigned int synAddress = (ipre * " + std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + ") + j;" << std::endl;
            }

            Substitutions synSubs(&popSubs);
            synSubs.addVarSubstitution("id_pre", "ipre");
            synSubs.addVarSubstitution("id_post", "ipost");
            synSubs.addVarSubstitution("id_syn", "synAddress");

            if(sg.isDendriticDelayRequired()) {
                synSubs.addFuncSubstitution("addToInSynDelay", 2, "denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("", "$(1)") + "ipost] += $(0)");
            }
            else {
                synSubs.addFuncSubstitution("addToInSyn", 1, "inSyn" + sg.getPSModelTargetName() + "[ipost] += $(0)");

            }
            wumSimHandler(os, sg, synSubs);

            if (!trueSpike) {
                os << CodeStream::CB(2041); // end if (eCode)
            }
            else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                os << CodeStream::CB(2041); // end if (B(gp" << sgName << "[gid / 32], gid
            }
        }
    }
}
//--------------------------------------------------------------------------
void SingleThreadedCPU::genEmitSpike(CodeStream &os, const NeuronGroup &ng, const Substitutions &subs, const std::string &suffix) const
{
    const std::string queueOffset = ng.isDelayRequired() ? "writeDelayOffset + " : "";

    os << "glbSpk" << suffix << ng.getName() << "[" << queueOffset << "glbSpkCnt" << suffix << ng.getName();
    if (ng.isDelayRequired()) { // WITH DELAY
        os << "[spkQuePtr" << ng.getName() << "]++]";
    }
    else { // NO DELAY
        os << "[0]++]";
    }
    os << " = " << subs.getVarSubstitution("id") << ";" << std::endl;
}
}   // namespace Backends
}   // namespace CodeGenerator
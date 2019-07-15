#include "backend.h"

// GeNN includes
#include "gennUtils.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/substitutions.h"
#include "code_generator/codeGenUtils.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const std::vector<CodeGenerator::FunctionTemplate> cpuFunctions = {
    {"gennrand_uniform", 0, "standardUniformDistribution($(rng))", "standardUniformDistribution($(rng))"},
    {"gennrand_normal", 0, "standardNormalDistribution($(rng))", "standardNormalDistribution($(rng))"},
    {"gennrand_exponential", 0, "standardExponentialDistribution($(rng))", "standardExponentialDistribution($(rng))"},
    {"gennrand_log_normal", 2, "std::lognormal_distribution<double>($(0), $(1))($(rng))", "std::lognormal_distribution<float>($(0), $(1))($(rng))"},
    {"gennrand_gamma", 1, "std::gamma_distribution<double>($(0), 1.0)($(rng))", "std::gamma_distribution<float>($(0), 1.0f)($(rng))"}
};

//--------------------------------------------------------------------------
// Timer
//--------------------------------------------------------------------------
class Timer
{
public:
    Timer(CodeGenerator::CodeStream &codeStream, const std::string &name, bool timingEnabled)
    :   m_CodeStream(codeStream), m_Name(name), m_TimingEnabled(timingEnabled)
    {
        // Record start event
        if(m_TimingEnabled) {
            m_CodeStream << "const auto " << m_Name << "Start = std::chrono::high_resolution_clock::now();" << std::endl;
        }
    }

    ~Timer()
    {
        // Record stop event
        if(m_TimingEnabled) {
            m_CodeStream << m_Name << "Time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - " << m_Name << "Start).count();" << std::endl;
        }
    }

private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    CodeGenerator::CodeStream &m_CodeStream;
    const std::string m_Name;
    const bool m_TimingEnabled;
};
}

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Backend
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace SingleThreadedCPU
{
void Backend::genNeuronUpdate(CodeStream &os, const ModelSpecInternal &model, NeuronGroupSimHandler simHandler, NeuronGroupHandler wuVarUpdateHandler) const
{
    os << "void updateNeurons(" << model.getTimePrecision() << " t)";
    {
        CodeStream::Scope b(os);

        Substitutions funcSubs(cpuFunctions, model.getPrecision());
        funcSubs.addVarSubstitution("t", "t");

        Timer t(os, "neuronUpdate", model.isTimingEnabled());

        // Update neurons
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

                    // If this neuron group requires a simulation RNG, substitute in global RNG
                    if(n.second.isSimRNGRequired()) {
                        popSubs.addVarSubstitution("rng", "rng");
                    }

                    simHandler(os, n.second, popSubs,
                        // Emit true spikes
                        [this, wuVarUpdateHandler](CodeStream &os, const NeuronGroupInternal &ng, Substitutions &subs)
                        {
                            // Insert code to emit true spikes
                            genEmitSpike(os, ng, subs, true);

                            // Insert code to update WU vars
                            wuVarUpdateHandler(os, ng, subs);
                        },
                        // Emit spike-like events
                        [this](CodeStream &os, const NeuronGroupInternal &ng, Substitutions &subs)
                        {
                            // Insert code to emit spike-like events
                            genEmitSpike(os, ng, subs, false);
                        });
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genSynapseUpdate(CodeStream &os, const ModelSpecInternal &model,
                               SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler, SynapseGroupHandler wumEventHandler,
                               SynapseGroupHandler postLearnHandler, SynapseGroupHandler synapseDynamicsHandler) const
{
    os << "void updateSynapses(" << model.getTimePrecision() << " t)";
    {
        Substitutions funcSubs(cpuFunctions, model.getPrecision());
        funcSubs.addVarSubstitution("t", "t");

        CodeStream::Scope b(os);

        // Synapse dynamics
        {
            Timer t(os, "synapseDynamics", model.isTimingEnabled());
            for(const auto &s : model.getLocalSynapseGroups()) {
                if(!s.second.getWUModel()->getSynapseDynamicsCode().empty()) {
                    os << "// synapse group " << s.first << std::endl;
                    CodeStream::Scope b(os);

                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(s.second.getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelayOffset = " << s.second.getPresynapticAxonalDelaySlot("") << " * " << s.second.getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(s.second.getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelayOffset = " << s.second.getPostsynapticBackPropDelaySlot("") << " * " << s.second.getTrgNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }

                    // Loop through presynaptic neurons
                    os << "for(unsigned int i = 0; i < " <<  s.second.getSrcNeuronGroup()->getNumNeurons() << "; i++)";
                    {
                        // If this synapse group has sparse connectivity, loop through length of this row
                        CodeStream::Scope b(os);
                        if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            os << "for(unsigned int s = 0; s < rowLength" << s.first << "[i]; s++)";
                        }
                        // Otherwise, if it's dense, loop through each postsynaptic neuron
                        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                            os << "for (unsigned int j = 0; j < " <<  s.second.getTrgNeuronGroup()->getNumNeurons() << "; j++)";
                        }
                        else {
                            throw std::runtime_error("Only DENSE and SPARSE format connectivity can be used for synapse dynamics");
                        }
                        {
                            CodeStream::Scope b(os);

                            Substitutions synSubs(&funcSubs);
                            if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                // Calculate index of synapse and use it to look up postsynaptic index
                                os << "const unsigned int n = (i * " << s.second.getMaxConnections() <<  ") + s;" << std::endl;
                                os << "const unsigned int j = ind" << s.first << "[n];" << std::endl;

                                synSubs.addVarSubstitution("id_syn", "n");
                            }
                            else {
                                synSubs.addVarSubstitution("id_syn", "(i * " + std::to_string(s.second.getTrgNeuronGroup()->getNumNeurons()) + ") + j");
                            }

                            // Add pre and postsynaptic indices to substitutions
                            synSubs.addVarSubstitution("id_pre", "i");
                            synSubs.addVarSubstitution("id_post", "j");

                            // Add correct functions for apply synaptic input
                            if(s.second.isDendriticDelayRequired()) {
                                synSubs.addFuncSubstitution("addToInSynDelay", 2, "denDelay" + s.second.getPSModelTargetName() + "[" + s.second.getDendriticDelayOffset("", "$(1)") + "j] += $(0)");
                            }
                            else {
                                synSubs.addFuncSubstitution("addToInSyn", 1, "inSyn" + s.second.getPSModelTargetName() + "[j] += $(0)");
                            }

                            // Call synapse dynamics handler
                            synapseDynamicsHandler(os, s.second, synSubs);
                        }
                    }
                }
            }
        }

        // Presynaptic update
        {
            Timer t(os, "presynapticUpdate", model.isTimingEnabled());
            for(const auto &s : model.getLocalSynapseGroups()) {
                if(s.second.isSpikeEventRequired() || s.second.isTrueSpikeRequired()) {
                    os << "// synapse group " << s.first << std::endl;
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
                        genPresynapticUpdate(os, s.second, funcSubs, false, wumThreshHandler, wumEventHandler);
                    }

                    // generate the code for processing true spike events
                    if (s.second.isTrueSpikeRequired()) {
                        genPresynapticUpdate(os, s.second, funcSubs, true, wumThreshHandler, wumSimHandler);
                    }
                    os << std::endl;
                }
            }
        }

        // Postsynaptic update
        {
            Timer t(os, "postsynapticUpdate", model.isTimingEnabled());
            for(const auto &s : model.getLocalSynapseGroups()) {
                if(!s.second.getWUModel()->getLearnPostCode().empty()) {
                    os << "// synapse group " << s.first << std::endl;
                    CodeStream::Scope b(os);

                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(s.second.getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelayOffset = " << s.second.getPresynapticAxonalDelaySlot("") << " * " << s.second.getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(s.second.getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelaySlot = " << s.second.getPostsynapticBackPropDelaySlot("") << ";" << std::endl;
                        os << "const unsigned int postReadDelayOffset = postReadDelaySlot * " << s.second.getTrgNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }

                    // Get number of postsynaptic spikes
                    if (s.second.getTrgNeuronGroup()->isDelayRequired() && s.second.getTrgNeuronGroup()->isTrueSpikeRequired()) {
                        os << "const unsigned int numSpikes = glbSpkCnt" << s.second.getTrgNeuronGroup()->getName() << "[postReadDelaySlot];" << std::endl;
                    }
                    else {
                        os << "const unsigned int numSpikes = glbSpkCnt" << s.second.getTrgNeuronGroup()->getName() << "[0];" << std::endl;
                    }

                    // Loop through postsynaptic spikes
                    os << "for (unsigned int j = 0; j < numSpikes; j++)";
                    {
                        CodeStream::Scope b(os);

                        const std::string offsetTrueSpkPost = (s.second.getTrgNeuronGroup()->isTrueSpikeRequired() && s.second.getTrgNeuronGroup()->isDelayRequired()) ? "postReadDelayOffset + " : "";
                        os << "const unsigned int spike = glbSpk" << s.second.getTrgNeuronGroup()->getName() << "[" << offsetTrueSpkPost << "j];" << std::endl;

                        // Loop through column of presynaptic neurons
                        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            os << "const unsigned int npre = colLength" << s.first << "[spike];" << std::endl;
                            os << "for (unsigned int i = 0; i < npre; i++)";
                        }
                        else {
                            os << "for (unsigned int i = 0; i < " << s.second.getSrcNeuronGroup()->getNumNeurons() << "; i++)";
                        }
                        {
                            CodeStream::Scope b(os);

                            Substitutions synSubs(&funcSubs);
                            if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                os << "const unsigned int colMajorIndex = (spike * " << s.second.getMaxSourceConnections() << ") + i;" << std::endl;
                                os << "const unsigned int rowMajorIndex = remap" << s.first << "[colMajorIndex];" << std::endl;
                                synSubs.addVarSubstitution("id_pre", "(rowMajorIndex / " + std::to_string(s.second.getMaxConnections()) + ")");
                                synSubs.addVarSubstitution("id_syn", "rowMajorIndex");
                            }
                            else {
                                synSubs.addVarSubstitution("id_pre", "i");
                                synSubs.addVarSubstitution("id_syn", "((" + std::to_string(s.second.getTrgNeuronGroup()->getNumNeurons()) + " * i) + spike)");
                            }
                            synSubs.addVarSubstitution("id_post", "spike");

                            postLearnHandler(os, s.second, synSubs);
                        }
                    }
                    os << std::endl;
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genInit(CodeStream &os, const ModelSpecInternal &model,
                      NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                      SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                      SynapseGroupHandler sgSparseInitHandler) const
{
    os << "void initialize()";
    {
        CodeStream::Scope b(os);
        Substitutions funcSubs(cpuFunctions, model.getPrecision());

        Timer t(os, "init", model.isTimingEnabled());

        // If model requires a host RNG
        if(isGlobalRNGRequired(model)) {
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
            if(n.second.hasOutputToHost(getLocalHostID())) {
                os << "// neuron group " << n.first << std::endl;
                CodeStream::Scope b(os);

                Substitutions popSubs(&funcSubs);
                remoteNGHandler(os, n.second, popSubs);
            }
        }
        os << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Local neuron groups" << std::endl;
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "// neuron group " << n.first << std::endl;
            CodeStream::Scope b(os);

            Substitutions popSubs(&funcSubs);
            localNGHandler(os, n.second, popSubs);
        }

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with dense connectivity" << std::endl;
        for(const auto &s : model.getLocalSynapseGroups()) {
            if((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
                os << "// synapse group " << s.first << std::endl;
                CodeStream::Scope b(os);

                Substitutions popSubs(&funcSubs);
                sgDenseInitHandler(os, s.second, popSubs);
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
                if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
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
                    throw std::runtime_error("Only BITMASK and SPARSE format connectivity can be generated using a connectivity initialiser");
                }
            }
        }
    }
    os << std::endl;
    os << "void initializeSparse()";
    {
        CodeStream::Scope b(os);
        Substitutions funcSubs(cpuFunctions, model.getPrecision());

        Timer t(os, "initSparse", model.isTimingEnabled());

        // If model requires RNG, add it to substitutions
        if(isGlobalRNGRequired(model)) {
            funcSubs.addVarSubstitution("rng", "rng");
        }

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with sparse connectivity" << std::endl;
        for(const auto &s : model.getLocalSynapseGroups()) {
            // If synapse group has sparse connectivity and either has variables that require initialising
            // or has postsynaptic learning, meaning that reverse lookup structures need building
            if ((s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE)
                && (s.second.isWUVarInitRequired() || !s.second.getWUModel()->getLearnPostCode().empty()))
            {
                CodeStream::Scope b(os);
                // If postsynaptic learning is required, initially zero column lengths
                if (!s.second.getWUModel()->getLearnPostCode().empty()) {
                    os << "// Zero column lengths" << std::endl;
                    os << "std::fill_n(colLength" << s.first << ", " << s.second.getTrgNeuronGroup()->getNumNeurons() << ", 0);" << std::endl;
                }

                os << "// Loop through presynaptic neurons" << std::endl;
                os << "for (unsigned int i = 0; i < " << s.second.getSrcNeuronGroup()->getNumNeurons() << "; i++)" << std::endl;
                {
                    CodeStream::Scope b(os);

                    // Generate sparse initialisation code
                    if(s.second.isWUVarInitRequired()) {
                        Substitutions popSubs(&funcSubs);
                        popSubs.addVarSubstitution("id_pre", "i");
                        popSubs.addVarSubstitution("row_len", "rowLength" + s.first + "[i]");
                        sgSparseInitHandler(os, s.second, popSubs);
                    }

                    // If postsynaptic learning is required
                    if(!s.second.getWUModel()->getLearnPostCode().empty()) {
                        os << "// Loop through synapses in corresponding matrix row" << std::endl;
                        os << "for(unsigned int j = 0; j < rowLength" << s.first << "[i]; j++)" << std::endl;
                        {
                            CodeStream::Scope b(os);

                            // If postsynaptic learning is required, calculate column length and remapping
                            if(!s.second.getWUModel()->getLearnPostCode().empty()) {
                                os << "// Calculate index of this synapse in the row-major matrix" << std::endl;
                                os << "const unsigned int rowMajorIndex = (i * " << s.second.getMaxConnections()  << ") + j;" << std::endl;
                                os << "// Using this, lookup postsynaptic target" << std::endl;
                                os << "const unsigned int postIndex = ind" << s.first << "[rowMajorIndex];" << std::endl;
                                os << "// From this calculate index of this synapse in the column-major matrix" << std::endl;
                                os << "const unsigned int colMajorIndex = (postIndex * " << s.second.getMaxSourceConnections() << ") + colLength" << s.first << "[postIndex];" << std::endl;
                                os << "// Increment column length corresponding to this postsynaptic neuron" << std::endl;
                                os << "colLength" << s.first << "[postIndex]++;" << std::endl;
                                os << "// Add remapping entry" << std::endl;
                                os << "remap" << s.first << "[colMajorIndex] = rowMajorIndex;" << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsPreamble(CodeStream &os) const
{
    os << "// Standard C++ includes" << std::endl;
    os << "#include <chrono>" << std::endl;
    os << "#include <iostream>" << std::endl;
    os << "#include <random>" << std::endl;
    os << std::endl;
    os << "// Standard C includes" << std::endl;
    os << "#include <cmath>" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << "#include <cstring>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsInternalPreamble(CodeStream &os) const
{
    os << "#define SUPPORT_CODE_FUNC inline" << std::endl;
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genRunnerPreamble(CodeStream &) const
{
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream &, const ModelSpecInternal &) const
{
}
//--------------------------------------------------------------------------
void Backend::genStepTimeFinalisePreamble(CodeStream &, const ModelSpecInternal &) const
{
}
//--------------------------------------------------------------------------
void Backend::genVariableDefinition(CodeStream &definitions, CodeStream &, const std::string &type, const std::string &name, VarLocation) const
{
    definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation) const
{
    os << type << " " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
MemAlloc Backend::genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation, size_t count) const
{
    os << name << " = new " << type << "[" << count << "];" << std::endl;

    return MemAlloc::host(count * getSize(type));
}
//--------------------------------------------------------------------------
void Backend::genVariableFree(CodeStream &os, const std::string &name, VarLocation) const
{
    os << "delete[] " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamDefinition(CodeStream &definitions, const std::string &type, const std::string &name, VarLocation) const
{
    definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    genVariableImplementation(os, type, name, loc);
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation) const
{
    // Get underlying type
    // **NOTE** could use std::remove_pointer but it seems unnecessarily elaborate
    const std::string underlyingType = Utils::getUnderlyingType(type);

    os << name << " = new " << underlyingType << "[count];" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPush(CodeStream &, const std::string &, const std::string &, VarLocation) const
{
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPull(CodeStream &, const std::string &, const std::string &, VarLocation) const
{
}
//--------------------------------------------------------------------------
void Backend::genPopVariableInit(CodeStream &os, VarLocation, const Substitutions &kernelSubs, Handler handler) const
{
    Substitutions varSubs(&kernelSubs);
    handler(os, varSubs);
}
//--------------------------------------------------------------------------
void Backend::genVariableInit(CodeStream &os, VarLocation, size_t count, const std::string &indexVarName,
                              const Substitutions &kernelSubs, Handler handler) const
{
     // **TODO** loops like this should be generated like CUDA threads
    os << "for (unsigned i = 0; i < " << count << "; i++)";
    {
        CodeStream::Scope b(os);

        Substitutions varSubs(&kernelSubs);
        varSubs.addVarSubstitution(indexVarName, "i");
        handler(os, varSubs);
    }
}
//--------------------------------------------------------------------------
void Backend::genSynapseVariableRowInit(CodeStream &os, VarLocation, const SynapseGroupInternal &sg,
                                        const Substitutions &kernelSubs, Handler handler) const
{
    // **TODO** loops like this should be generated like CUDA threads
    if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        os << "for (unsigned j = 0; j < rowLength" << sg.getName() << "[" << kernelSubs["id_pre"] << "]; j++)";
    }
    else {
        os << "for (unsigned j = 0; j < " << sg.getTrgNeuronGroup()->getNumNeurons() << "; j++)";
    }
    {
        CodeStream::Scope b(os);

        Substitutions varSubs(&kernelSubs);
        if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            varSubs.addVarSubstitution("id_syn", "(" + kernelSubs["id_pre"] + " * " + std::to_string(sg.getMaxConnections()) + ") + j");
            varSubs.addVarSubstitution("id_post", "ind" + sg.getName() + "[(" + kernelSubs["id_pre"] + " * " + std::to_string(sg.getMaxConnections()) + ") + j]");
        }
        else {
            varSubs.addVarSubstitution("id_syn", "(" + kernelSubs["id_pre"] + " * " + std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + ") + j");
            varSubs.addVarSubstitution("id_post", "j");
        }
        handler(os, varSubs);
    }
}
//--------------------------------------------------------------------------
void Backend::genVariablePush(CodeStream&, const std::string&, const std::string&, VarLocation, bool, size_t) const
{
}
//--------------------------------------------------------------------------
void Backend::genVariablePull(CodeStream&, const std::string&, const std::string&, VarLocation, size_t) const
{
}
//--------------------------------------------------------------------------
void Backend::genCurrentTrueSpikePush(CodeStream&, const NeuronGroupInternal&) const
{
}
//--------------------------------------------------------------------------
void Backend::genCurrentTrueSpikePull(CodeStream&, const NeuronGroupInternal&) const
{
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikeLikeEventPush(CodeStream&, const NeuronGroupInternal&) const
{
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikeLikeEventPull(CodeStream&, const NeuronGroupInternal&) const
{
}
//--------------------------------------------------------------------------
MemAlloc Backend::genGlobalRNG(CodeStream &definitions, CodeStream &, CodeStream &runner, CodeStream &, CodeStream &, const ModelSpecInternal &model) const
{
    definitions << "EXPORT_VAR " << "std::mt19937 rng;" << std::endl;
    runner << "std::mt19937 rng;" << std::endl;

    // Define and implement standard host distributions as recreating them each call is slow
    definitions << "EXPORT_VAR " << "std::uniform_real_distribution<" << model.getPrecision() << "> standardUniformDistribution;" << std::endl;
    definitions << "EXPORT_VAR " << "std::normal_distribution<" << model.getPrecision() << "> standardNormalDistribution;" << std::endl;
    definitions << "EXPORT_VAR " << "std::exponential_distribution<" << model.getPrecision() << "> standardExponentialDistribution;" << std::endl;
    runner << "std::uniform_real_distribution<" << model.getPrecision() << "> standardUniformDistribution(" << model.scalarExpr(0.0) << ", " << model.scalarExpr(1.0) << ");" << std::endl;
    runner << "std::normal_distribution<" << model.getPrecision() << "> standardNormalDistribution(" << model.scalarExpr(0.0) << ", " << model.scalarExpr(1.0) << ");" << std::endl;
    runner << "std::exponential_distribution<" << model.getPrecision() << "> standardExponentialDistribution(" << model.scalarExpr(1.0) << ");" << std::endl;

    return MemAlloc::zero();
}
//--------------------------------------------------------------------------
MemAlloc Backend::genPopulationRNG(CodeStream &, CodeStream &, CodeStream &, CodeStream &, CodeStream &,
                               const std::string&, size_t) const
{
    // No need for population RNGs for single-threaded CPU
    return MemAlloc::zero();
}
//--------------------------------------------------------------------------
void Backend::genTimer(CodeStream &, CodeStream &, CodeStream &, CodeStream &, CodeStream &, CodeStream &, const std::string &, bool) const
{
    // Timing single-threaded CPU backends don't require any additional state
}
//--------------------------------------------------------------------------
void Backend::genMakefilePreamble(std::ostream &os) const
{
    std::string linkFlags = "-shared ";
    std::string cxxFlags = "-c -fPIC -std=c++11 -MMD -MP";
    cxxFlags += " " + m_Preferences.userCxxFlagsGNU;
    if (m_Preferences.optimizeCode) {
        cxxFlags += " -O3 -ffast-math";
    }
    if (m_Preferences.debugCode) {
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
void Backend::genMakefileLinkRule(std::ostream &os) const
{
    os << "\t$(CXX) $(LINKFLAGS) -o $@ $(OBJECTS)" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileCompileRule(std::ostream &os) const
{
    os << "%.o: %.cc %.d" << std::endl;
    os << "\t$(CXX) $(CXXFLAGS) -o $@ $<" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildConfigProperties(std::ostream&) const
{
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportProps(std::ostream&) const
{
}
//--------------------------------------------------------------------------
void Backend::genMSBuildItemDefinitions(std::ostream &os) const
{
    // Add item definition for host compilation
    os << "\t\t<ClCompile>" << std::endl;
    os << "\t\t\t<WarningLevel>Level3</WarningLevel>" << std::endl;
    os << "\t\t\t<Optimization Condition=\"'$(Configuration)'=='Release'\">MaxSpeed</Optimization>" << std::endl;
    os << "\t\t\t<Optimization Condition=\"'$(Configuration)'=='Debug'\">Disabled</Optimization>" << std::endl;
    os << "\t\t\t<FunctionLevelLinking Condition=\"'$(Configuration)'=='Release'\">true</FunctionLevelLinking>" << std::endl;
    os << "\t\t\t<IntrinsicFunctions Condition=\"'$(Configuration)'=='Release'\">true</IntrinsicFunctions>" << std::endl;
    os << "\t\t\t<ExceptionHandling>SyncCThrow</ExceptionHandling>" << std::endl;
    os << "\t\t\t<PreprocessorDefinitions Condition=\"'$(Configuration)'=='Release'\">WIN32;WIN64;NDEBUG;_CONSOLE;BUILDING_GENERATED_CODE;%(PreprocessorDefinitions)</PreprocessorDefinitions>" << std::endl;
    os << "\t\t\t<PreprocessorDefinitions Condition=\"'$(Configuration)'=='Debug'\">WIN32;WIN64;_DEBUG;_CONSOLE;BUILDING_GENERATED_CODE;%(PreprocessorDefinitions)</PreprocessorDefinitions>" << std::endl;
    os << "\t\t\t<FloatingPointModel>" << (m_Preferences.optimizeCode ? "Fast" : "Precise") << "</FloatingPointModel>" << std::endl;
    os << "\t\t</ClCompile>" << std::endl;

    // Add item definition for linking
    os << "\t\t<Link>" << std::endl;
    os << "\t\t\t<GenerateDebugInformation>true</GenerateDebugInformation>" << std::endl;
    os << "\t\t\t<EnableCOMDATFolding Condition=\"'$(Configuration)'=='Release'\">true</EnableCOMDATFolding>" << std::endl;
    os << "\t\t\t<OptimizeReferences Condition=\"'$(Configuration)'=='Release'\">true</OptimizeReferences>" << std::endl;
    os << "\t\t\t<SubSystem>Console</SubSystem>" << std::endl;
    os << "\t\t\t<AdditionalDependencies>kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>" << std::endl;
    os << "\t\t</Link>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const
{
    os << "\t\t<ClCompile Include=\"" << moduleName << ".cc\" />" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportTarget(std::ostream&) const
{
}
//--------------------------------------------------------------------------
bool Backend::isGlobalRNGRequired(const ModelSpecInternal &model) const
{
    // If any neuron groups require simulation RNGs or require RNG for initialisation, return true
    // **NOTE** this takes postsynaptic model initialisation into account
    if(std::any_of(model.getLocalNeuronGroups().cbegin(), model.getLocalNeuronGroups().cend(),
        [](const ModelSpec::NeuronGroupValueType &n)
        {
            return n.second.isSimRNGRequired() || n.second.isInitRNGRequired();
        }))
    {
        return true;
    }

    // If any synapse groups require an RNG for weight update model initialisation, return true
    if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
        [](const ModelSpec::SynapseGroupValueType &s)
        {
            return s.second.isWUInitRNGRequired();
        }))
    {
        return true;
    }

    return false;
}
//--------------------------------------------------------------------------
void Backend::genPresynapticUpdate(CodeStream &os, const SynapseGroupInternal &sg, const Substitutions &popSubs, bool trueSpike,
                                   SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
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
        if (!wu->getSimSupportCode().empty()) {
            os << " using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
        }

        const std::string queueOffset = sg.getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
        os << "const unsigned int ipre = glbSpk" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[" << queueOffset << "i];" << std::endl;

        // If this is a spike-like event, insert threshold check for this presynaptic neuron
        if (!trueSpike) {
            os << "if(";

            Substitutions threshSubs(&popSubs);
            threshSubs.addVarSubstitution("id_pre", "ipre");

            // Generate weight update threshold condition
            wumThreshHandler(os, sg, threshSubs);

            os << ")";
            os << CodeStream::OB(10);
        }

        if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            os << "const unsigned int npost = rowLength" << sg.getName() << "[ipre];" << std::endl;
            os << "for (unsigned int j = 0; j < npost; j++)";
        }
        // Otherwise (DENSE or BITMASK)
        else {
            os << "for (unsigned int ipost = 0; ipost < " << sg.getTrgNeuronGroup()->getNumNeurons() << "; ipost++)";
        }
        {
            CodeStream::Scope b(os);
            if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                // **TODO** seperate stride from max connection
                os << "const unsigned int synAddress = (ipre * " + std::to_string(sg.getMaxConnections()) + ") + j;" << std::endl;
                os << "const unsigned int ipost = ind" << sg.getName() << "[synAddress];" << std::endl;
            }
            else {
                if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "const uint64_t gid = (ipre * " << sg.getTrgNeuronGroup()->getNumNeurons() << "ull + ipost);" << std::endl;
                    os << "if (B(gp" << sg.getName() << "[gid / 32], gid & 31))" << CodeStream::OB(20);
                }

                os << "const unsigned int synAddress = (ipre * " + std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + ") + ipost;" << std::endl;
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

            if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                os << CodeStream::CB(20);
            }
        }

        // If this is a spike-like event, close braces around threshold check
        if (!trueSpike) {
            os << CodeStream::CB(10);
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genEmitSpike(CodeStream &os, const NeuronGroupInternal &ng, const Substitutions &subs, bool trueSpike) const
{
    // Determine if delay is required and thus, at what offset we should write into the spike queue
    const bool spikeDelayRequired = trueSpike ? (ng.isDelayRequired() && ng.isTrueSpikeRequired()) : ng.isDelayRequired();
    const std::string spikeQueueOffset = spikeDelayRequired ? "writeDelayOffset + " : "";

    const std::string suffix = trueSpike ? "" : "Evnt";
    os << "glbSpk" << suffix << ng.getName() << "[" << spikeQueueOffset << "glbSpkCnt" << suffix << ng.getName();
    if(spikeDelayRequired) { // WITH DELAY
        os << "[spkQuePtr" << ng.getName() << "]++]";
    }
    else { // NO DELAY
        os << "[0]++]";
    }
    os << " = " << subs["id"] << ";" << std::endl;

    // Reset spike time if this is a true spike and spike time is required
    if(trueSpike && ng.isSpikeTimeRequired()) {
        const std::string queueOffset = ng.isDelayRequired() ? "writeDelayOffset + " : "";
        os << "sT" << ng.getName() << "[" << queueOffset << subs["id"] << "] = " << subs["t"] << ";" << std::endl;
    }
}
}   // namespace SingleThreadedCPU
}   // namespace CodeGenerator

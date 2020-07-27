#include "backend.h"

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/substitutions.h"

using namespace CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const std::vector<Substitutions::FunctionTemplate> cpuSinglePrecisionFunctions = {
    {"gennrand_uniform", 0, "standardUniformDistribution($(rng))"},
    {"gennrand_normal", 0, "standardNormalDistribution($(rng))"},
    {"gennrand_exponential", 0, "standardExponentialDistribution($(rng))"},
    {"gennrand_log_normal", 2, "std::lognormal_distribution<float>($(0), $(1))($(rng))"},
    {"gennrand_gamma", 1, "std::gamma_distribution<float>($(0), 1.0f)($(rng))"}
};
//--------------------------------------------------------------------------
const std::vector<Substitutions::FunctionTemplate> cpuDoublePrecisionFunctions = {
    {"gennrand_uniform", 0, "standardUniformDistribution($(rng))"},
    {"gennrand_normal", 0, "standardNormalDistribution($(rng))"},
    {"gennrand_exponential", 0, "standardExponentialDistribution($(rng))"},
    {"gennrand_log_normal", 2, "std::lognormal_distribution<double>($(0), $(1))($(rng))"},
    {"gennrand_gamma", 1, "std::gamma_distribution<double>($(0), 1.0)($(rng))"}
};

//--------------------------------------------------------------------------
// Timer
//--------------------------------------------------------------------------
class Timer
{
public:
    Timer(CodeStream &codeStream, const std::string &name, bool timingEnabled)
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
    CodeStream &m_CodeStream;
    const std::string m_Name;
    const bool m_TimingEnabled;
};

//-----------------------------------------------------------------------
const std::vector<Substitutions::FunctionTemplate> &getFunctionTemplates(const std::string &precision)
{
    return (precision == "double") ? cpuDoublePrecisionFunctions : cpuSinglePrecisionFunctions;
}
}

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Backend
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace SingleThreadedCPU
{
void Backend::genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces&,
                              HostHandler preambleHandler, NeuronGroupSimHandler simHandler, NeuronUpdateGroupMergedHandler wuVarUpdateHandler,
                              HostHandler pushEGPHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Generate struct definitions
    modelMerged.genMergedNeuronUpdateGroupStructs(os, *this);
    modelMerged.genMergedNeuronSpikeQueueUpdateStructs(os, *this);

    // Generate arrays of merged structs and functions to set them
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronSpikeQueueUpdateGroups());

    // Generate preamble
    preambleHandler(os);

    os << "void updateNeurons(" << model.getTimePrecision() << " t)";
    {
        CodeStream::Scope b(os);

        Substitutions funcSubs(getFunctionTemplates(model.getPrecision()));
        funcSubs.addVarSubstitution("t", "t");

        // Push any required EGPs
        pushEGPHandler(os);

        Timer t(os, "neuronUpdate", model.isTimingEnabled());

        // Loop through merged neuron spike queue update groups
        for(const auto &n : modelMerged.getMergedNeuronSpikeQueueUpdateGroups()) {
            CodeStream::Scope b(os);
            os << "// merged neuron spike queue update group " << n.getIndex() << std::endl;
            os << "for(unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
            {
                CodeStream::Scope b(os);

                // Get reference to group
                os << "const auto *group = &mergedNeuronSpikeQueueUpdateGroup" << n.getIndex() << "[g]; " << std::endl;

                // Generate spike count reset
                n.genMergedGroupSpikeCountReset(os);
            }
            
        }
        // Loop through merged neuron update groups
        for(const auto &n : modelMerged.getMergedNeuronUpdateGroups()) {
            CodeStream::Scope b(os);
            os << "// merged neuron update group " << n.getIndex() << std::endl;
            os << "for(unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
            {
                CodeStream::Scope b(os);

                // Get reference to group
                os << "const auto *group = &mergedNeuronUpdateGroup" << n.getIndex() << "[g]; " << std::endl;

                // If axonal delays are required
                if(n.getArchetype().isDelayRequired()) {
                    // We should READ from delay slot before spkQuePtr
                    os << "const unsigned int readDelayOffset = " << n.getPrevQueueOffset() << ";" << std::endl;

                    // And we should WRITE to delay slot pointed to be spkQuePtr
                    os << "const unsigned int writeDelayOffset = " << n.getCurrentQueueOffset() << ";" << std::endl;
                }
                os << std::endl;

                os << "for(unsigned int i = 0; i < group->numNeurons; i++)";
                {
                    CodeStream::Scope b(os);

                    Substitutions popSubs(&funcSubs);
                    popSubs.addVarSubstitution("id", "i");

                    // If this neuron group requires a simulation RNG, substitute in global RNG
                    if(n.getArchetype().isSimRNGRequired()) {
                        popSubs.addVarSubstitution("rng", "hostRNG");
                    }

                    simHandler(os, n, popSubs,
                               // Emit true spikes
                               [this, wuVarUpdateHandler](CodeStream &os, const NeuronUpdateGroupMerged &ng, Substitutions &subs)
                               {
                                   // Insert code to update WU vars
                                   wuVarUpdateHandler(os, ng, subs);

                                   // Insert code to emit true spikes
                                   genEmitSpike(os, ng, subs, true);
                               },
                               // Emit spike-like events
                                   [this](CodeStream &os, const NeuronUpdateGroupMerged &ng, Substitutions &subs)
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
void Backend::genSynapseUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces&,
                               HostHandler preambleHandler, PresynapticUpdateGroupMergedHandler wumThreshHandler, PresynapticUpdateGroupMergedHandler wumSimHandler,
                               PresynapticUpdateGroupMergedHandler wumEventHandler, PresynapticUpdateGroupMergedHandler,
                               PostsynapticUpdateGroupMergedHandler postLearnHandler, SynapseDynamicsGroupMergedHandler synapseDynamicsHandler,
                               HostHandler pushEGPHandler) const
{
    // Generate struct definitions
    // **YUCK** dendritic delay update structs not actually required
    modelMerged.genMergedSynapseDendriticDelayUpdateStructs(os, *this);
    modelMerged.genMergedPresynapticUpdateGroupStructs(os, *this);
    modelMerged.genMergedPostsynapticUpdateGroupStructs(os, *this);
    modelMerged.genMergedSynapseDynamicsGroupStructs(os, *this);

    // Generate arrays of merged structs and functions to set them
    // **YUCK** dendritic delay update structs not actually required
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseDendriticDelayUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedPresynapticUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedPostsynapticUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseDynamicsGroups());

    // Generate preamble
    preambleHandler(os);

    const ModelSpecInternal &model = modelMerged.getModel();
    os << "void updateSynapses(" << model.getTimePrecision() << " t)";
    {
        CodeStream::Scope b(os);
        Substitutions funcSubs(getFunctionTemplates(model.getPrecision()));
        funcSubs.addVarSubstitution("t", "t");

        // Push any required EGPs
        pushEGPHandler(os);

        // Synapse dynamics
        {
            // Loop through merged synapse dynamics groups
            Timer t(os, "synapseDynamics", model.isTimingEnabled());
            for(const auto &s : modelMerged.getMergedSynapseDynamicsGroups()) {
                CodeStream::Scope b(os);
                os << "// merged synapse dynamics group " << s.getIndex() << std::endl;
                os << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(os);

                    // Get reference to group
                    os << "const auto *group = &mergedSynapseDynamicsGroup" << s.getIndex() << "[g]; " << std::endl;

                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(s.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelayOffset = " << s.getPresynapticAxonalDelaySlot() << " * group->numSrcNeurons;" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(s.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelayOffset = " << s.getPostsynapticBackPropDelaySlot() << " * group->numTrgNeurons;" << std::endl;
                    }

                    // Loop through presynaptic neurons
                    os << "for(unsigned int i = 0; i < group->numSrcNeurons; i++)";
                    {
                        // If this synapse group has sparse connectivity, loop through length of this row
                        CodeStream::Scope b(os);
                        if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            os << "for(unsigned int s = 0; s < group->rowLength[i]; s++)";
                        }
                        // Otherwise, if it's dense, loop through each postsynaptic neuron
                        else if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                            os << "for (unsigned int j = 0; j < group->numTrgNeurons; j++)";
                        }
                        else {
                            throw std::runtime_error("Only DENSE and SPARSE format connectivity can be used for synapse dynamics");
                        }
                        {
                            CodeStream::Scope b(os);

                            Substitutions synSubs(&funcSubs);
                            if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                // Calculate index of synapse and use it to look up postsynaptic index
                                os << "const unsigned int n = (i * group->rowStride) + s;" << std::endl;
                                os << "const unsigned int j = group->ind[n];" << std::endl;

                                synSubs.addVarSubstitution("id_syn", "n");
                            }
                            else {
                                synSubs.addVarSubstitution("id_syn", "(i * group->numTrgNeurons) + j");
                            }

                            // Add pre and postsynaptic indices to substitutions
                            synSubs.addVarSubstitution("id_pre", "i");
                            synSubs.addVarSubstitution("id_post", "j");

                            // Add correct functions for apply synaptic input
                            if(s.getArchetype().isDendriticDelayRequired()) {
                                synSubs.addFuncSubstitution("addToInSynDelay", 2, "group->denDelay[" + s.getDendriticDelayOffset("$(1)") + "j] += $(0)");
                            }
                            else {
                                synSubs.addFuncSubstitution("addToInSyn", 1, "group->inSyn[j] += $(0)");
                            }

                            // Call synapse dynamics handler
                            synapseDynamicsHandler(os, s, synSubs);
                        }
                    }
                }
            }
        }

        // Presynaptic update
        {
            Timer t(os, "presynapticUpdate", model.isTimingEnabled());
            for(const auto &s : modelMerged.getMergedPresynapticUpdateGroups()) {
                CodeStream::Scope b(os);
                os << "// merged presynaptic update group " << s.getIndex() << std::endl;
                os << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(os);

                    // Get reference to group
                    os << "const auto *group = &mergedPresynapticUpdateGroup" << s.getIndex() << "[g]; " << std::endl;

                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(s.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelaySlot = " << s.getPresynapticAxonalDelaySlot() << ";" << std::endl;
                        os << "const unsigned int preReadDelayOffset = preReadDelaySlot * group->numSrcNeurons;" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(s.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelayOffset = " << s.getPostsynapticBackPropDelaySlot() << " * group->numTrgNeurons;" << std::endl;
                    }

                    // generate the code for processing spike-like events
                    if (s.getArchetype().isSpikeEventRequired()) {
                        genPresynapticUpdate(os, modelMerged, s, funcSubs, false, wumThreshHandler, wumEventHandler);
                    }

                    // generate the code for processing true spike events
                    if (s.getArchetype().isTrueSpikeRequired()) {
                        genPresynapticUpdate(os, modelMerged, s, funcSubs, true, wumThreshHandler, wumSimHandler);
                    }
                    os << std::endl;
                }
            }
        }

        // Postsynaptic update
        {
            Timer t(os, "postsynapticUpdate", model.isTimingEnabled());
            for(const auto &s : modelMerged.getMergedPostsynapticUpdateGroups()) {
                CodeStream::Scope b(os);
                os << "// merged postsynaptic update group " << s.getIndex() << std::endl;
                os << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(os);

                    // Get reference to group
                    os << "const auto *group = &mergedPostsynapticUpdateGroup" << s.getIndex() << "[g]; " << std::endl;

                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(s.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelayOffset = " << s.getPresynapticAxonalDelaySlot() << " * group->numSrcNeurons;" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(s.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelaySlot = " << s.getPostsynapticBackPropDelaySlot() << ";" << std::endl;
                        os << "const unsigned int postReadDelayOffset = postReadDelaySlot * group->numTrgNeurons;" << std::endl;
                    }

                    // Get number of postsynaptic spikes
                    if (s.getArchetype().getTrgNeuronGroup()->isDelayRequired() && s.getArchetype().getTrgNeuronGroup()->isTrueSpikeRequired()) {
                        os << "const unsigned int numSpikes = group->trgSpkCnt[postReadDelaySlot];" << std::endl;
                    }
                    else {
                        os << "const unsigned int numSpikes = group->trgSpkCnt[0];" << std::endl;
                    }

                    // Loop through postsynaptic spikes
                    os << "for (unsigned int j = 0; j < numSpikes; j++)";
                    {
                        CodeStream::Scope b(os);

                        const std::string offsetTrueSpkPost = (s.getArchetype().getTrgNeuronGroup()->isTrueSpikeRequired() && s.getArchetype().getTrgNeuronGroup()->isDelayRequired()) ? "postReadDelayOffset + " : "";
                        os << "const unsigned int spike = group->trgSpk[" << offsetTrueSpkPost << "j];" << std::endl;

                        // Loop through column of presynaptic neurons
                        if (s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            os << "const unsigned int npre = group->colLength[spike];" << std::endl;
                            os << "for (unsigned int i = 0; i < npre; i++)";
                        }
                        else {
                            os << "for (unsigned int i = 0; i < group->numSrcNeurons; i++)";
                        }
                        {
                            CodeStream::Scope b(os);

                            Substitutions synSubs(&funcSubs);
                            if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                os << "const unsigned int colMajorIndex = (spike * group->colStride) + i;" << std::endl;
                                os << "const unsigned int rowMajorIndex = group->remap[colMajorIndex];" << std::endl;

                                // **TODO** fast divide optimisations
                                synSubs.addVarSubstitution("id_pre", "(rowMajorIndex / group->rowStride)");
                                synSubs.addVarSubstitution("id_syn", "rowMajorIndex");
                            }
                            else {
                                synSubs.addVarSubstitution("id_pre", "i");
                                synSubs.addVarSubstitution("id_syn", "((group->numTrgNeurons * i) + spike)");
                            }
                            synSubs.addVarSubstitution("id_post", "spike");

                            postLearnHandler(os, s, synSubs);
                        }
                    }
                    os << std::endl;
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genInit(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces&,
                      HostHandler preambleHandler, NeuronInitGroupMergedHandler localNGHandler, SynapseDenseInitGroupMergedHandler sgDenseInitHandler,
                      SynapseConnectivityInitMergedGroupHandler sgSparseRowConnectHandler, SynapseConnectivityInitMergedGroupHandler sgSparseColConnectHandler, 
                      SynapseSparseInitGroupMergedHandler sgSparseInitHandler, HostHandler initPushEGPHandler, HostHandler initSparsePushEGPHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Generate struct definitions
    modelMerged.genMergedNeuronInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseDenseInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseConnectivityInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseSparseInitGroupStructs(os, *this);

    // Generate arrays of merged structs and functions to set them
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseDenseInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseConnectivityInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseSparseInitGroups());

    // Generate preamble
    preambleHandler(os);

    os << "void initialize()";
    {
        CodeStream::Scope b(os);
        Substitutions funcSubs(getFunctionTemplates(model.getPrecision()));

        // Push any required EGPs
        initPushEGPHandler(os);

        Timer t(os, "init", model.isTimingEnabled());

        // If model requires a host RNG, add RNG to substitutions
        if(isGlobalHostRNGRequired(modelMerged)) {
            funcSubs.addVarSubstitution("rng", "hostRNG");
        }

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Local neuron groups" << std::endl;
        for(const auto &n : modelMerged.getMergedNeuronInitGroups()) {
            CodeStream::Scope b(os);
            os << "// merged neuron init group " << n.getIndex() << std::endl;
            os << "for(unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
            {
                CodeStream::Scope b(os);

                // Get reference to group
                os << "const auto *group = &mergedNeuronInitGroup" << n.getIndex() << "[g]; " << std::endl;
                Substitutions popSubs(&funcSubs);
                localNGHandler(os, n, popSubs);
            }
        }

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with dense connectivity" << std::endl;
        for(const auto &s : modelMerged.getMergedSynapseDenseInitGroups()) {
            CodeStream::Scope b(os);
            os << "// merged synapse dense init group " << s.getIndex() << std::endl;
            os << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
            {
                CodeStream::Scope b(os);

                // Get reference to group
                os << "const auto *group = &mergedSynapseDenseInitGroup" << s.getIndex() << "[g]; " << std::endl;
                Substitutions popSubs(&funcSubs);
                sgDenseInitHandler(os, s, popSubs);
            }
        }

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with sparse connectivity" << std::endl;
        for(const auto &s : modelMerged.getMergedSynapseConnectivityInitGroups()) {
            CodeStream::Scope b(os);
            os << "// merged synapse connectivity init group " << s.getIndex() << std::endl;
            os << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
            {
                CodeStream::Scope b(os);

                // Get reference to group
                os << "const auto *group = &mergedSynapseConnectivityInitGroup" << s.getIndex() << "[g]; " << std::endl;

                Substitutions popSubs(&funcSubs);

                const auto *snippet = s.getArchetype().getConnectivityInitialiser().getSnippet();
               
                // If matrix connectivity is ragged
                if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    // Zero row lengths
                    os << "memset(group->rowLength, 0, group->numSrcNeurons * sizeof(unsigned int));" << std::endl;
                }
                else if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "const size_t gpSize = ((((size_t)group->numSrcNeurons * (size_t)group->rowStride) + 32 - 1) / 32);" << std::endl;
                    os << "memset(group->gp, 0, gpSize * sizeof(uint32_t));" << std::endl;
                }
                else {
                    throw std::runtime_error("Only BITMASK and SPARSE format connectivity can be generated using a connectivity initialiser");
                }

                // If there is row-building code in this snippet
                if(!snippet->getRowBuildCode().empty()) {
                    // Loop through source neurons
                    os << "for (unsigned int i = 0; i < group->numSrcNeurons; i++)";
                    {
                        CodeStream::Scope b(os);
                        popSubs.addVarSubstitution("id_pre", "i");
                        popSubs.addVarSubstitution("id_post_begin", "0");
                        popSubs.addVarSubstitution("id_thread", "0");
                        popSubs.addVarSubstitution("num_threads", "1");
                        popSubs.addVarSubstitution("num_post", "group->numTrgNeurons");

                        // If matrix is sparse, add function to increment row length and insert synapse into ind array
                        if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            popSubs.addFuncSubstitution("addSynapse", 1,
                                                        "group->ind[(i * group->rowStride) + (group->rowLength[i]++)] = $(0)");
                        }
                        // Otherwise, add function to set correct bit in bitmask
                        else {
                            os << "const int64_t rowStartGID = i * group->rowStride;" << std::endl;
                            popSubs.addFuncSubstitution("addSynapse", 1,
                                                        "setB(group->gp[(rowStartGID + $(0)) / 32], (rowStartGID + $(0)) & 31)");
                        }

                        sgSparseRowConnectHandler(os, s, popSubs);
                    }
                }
                // Otherwise, if there is column building code
                else if(!snippet->getColBuildCode().empty()) {
                    // Loop through target neurons
                    os << "for (unsigned int j = 0; j < group->numTrgNeurons; j++)";
                    {
                        CodeStream::Scope b(os);
                        popSubs.addVarSubstitution("id_post", "j");
                        popSubs.addVarSubstitution("id_pre_begin", "0");
                        popSubs.addVarSubstitution("id_thread", "0");
                        popSubs.addVarSubstitution("num_threads", "1");
                        popSubs.addVarSubstitution("num_pre", "group->numSrcNeurons");

                        // If matrix is sparse, add function to increment row length and insert synapse into ind array
                        if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            popSubs.addFuncSubstitution("addSynapse", 1,
                                                        "group->ind[(($(0)) * group->rowStride) + group->rowLength[$(0)]++] = " + popSubs["id_post"]);
                        }
                        else {
                            os << "const int64_t colStartGID = j;" << std::endl;
                            popSubs.addFuncSubstitution("addSynapse", 1,
                                                        "setB(group->gp[(colStartGID + ($(0) * group->rowStride)) / 32], ((colStartGID + ($(0) * group->rowStride)) & 31))");
                        }
                        sgSparseColConnectHandler(os, s, popSubs);
                    }
                }
            }
        }
    }
    os << std::endl;
    os << "void initializeSparse()";
    {
        CodeStream::Scope b(os);
        Substitutions funcSubs(getFunctionTemplates(model.getPrecision()));

        // Push any required EGPs
        initSparsePushEGPHandler(os);

        Timer t(os, "initSparse", model.isTimingEnabled());

        // If model requires RNG, add it to substitutions
        if(isGlobalHostRNGRequired(modelMerged)) {
            funcSubs.addVarSubstitution("rng", "hostRNG");
        }

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with sparse connectivity" << std::endl;
        for(const auto &s : modelMerged.getMergedSynapseSparseInitGroups()) {
            CodeStream::Scope b(os);
            os << "// merged sparse synapse init group " << s.getIndex() << std::endl;
            os << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
            {
                CodeStream::Scope b(os);

                // Get reference to group
                os << "const auto *group = &mergedSynapseSparseInitGroup" << s.getIndex() << "[g]; " << std::endl;

                // If postsynaptic learning is required, initially zero column lengths
                if (!s.getArchetype().getWUModel()->getLearnPostCode().empty()) {
                    os << "// Zero column lengths" << std::endl;
                    os << "std::fill_n(group->colLength, group->numTrgNeurons, 0);" << std::endl;
                }

                os << "// Loop through presynaptic neurons" << std::endl;
                os << "for (unsigned int i = 0; i < group->numSrcNeurons; i++)" << std::endl;
                {
                    CodeStream::Scope b(os);

                    // Generate sparse initialisation code
                    if(s.getArchetype().isWUVarInitRequired()) {
                        Substitutions popSubs(&funcSubs);
                        popSubs.addVarSubstitution("id_pre", "i");
                        popSubs.addVarSubstitution("row_len", "group->rowLength[i]");
                        sgSparseInitHandler(os, s, popSubs);
                    }

                    // If postsynaptic learning is required
                    if(!s.getArchetype().getWUModel()->getLearnPostCode().empty()) {
                        os << "// Loop through synapses in corresponding matrix row" << std::endl;
                        os << "for(unsigned int j = 0; j < group->rowLength[i]; j++)" << std::endl;
                        {
                            CodeStream::Scope b(os);

                            // If postsynaptic learning is required, calculate column length and remapping
                            if(!s.getArchetype().getWUModel()->getLearnPostCode().empty()) {
                                os << "// Calculate index of this synapse in the row-major matrix" << std::endl;
                                os << "const unsigned int rowMajorIndex = (i * group->rowStride) + j;" << std::endl;
                                os << "// Using this, lookup postsynaptic target" << std::endl;
                                os << "const unsigned int postIndex = group->ind[rowMajorIndex];" << std::endl;
                                os << "// From this calculate index of this synapse in the column-major matrix" << std::endl;
                                os << "const unsigned int colMajorIndex = (postIndex * group->colStride) + group->colLength[postIndex];" << std::endl;
                                os << "// Increment column length corresponding to this postsynaptic neuron" << std::endl;
                                os << "group->colLength[postIndex]++;" << std::endl;
                                os << "// Add remapping entry" << std::endl;
                                os << "group->remap[colMajorIndex] = rowMajorIndex;" << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
size_t Backend::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxConnections();
    }
    else if(m_Preferences.enableBitmaskOptimisations && (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK)) {
        return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
    }
    else {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    os << "// Standard C++ includes" << std::endl;
    os << "#include <algorithm>" << std::endl;
    os << "#include <chrono>" << std::endl;
    os << "#include <iostream>" << std::endl;
    os << "#include <random>" << std::endl;
    os << std::endl;
    os << "// Standard C includes" << std::endl;
    os << "#include <cassert>" << std::endl;
    os << "#include <cmath>" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << "#include <cstring>" << std::endl;

     // If a global RNG is required, define standard host distributions as recreating them each call is slow
    if(isGlobalHostRNGRequired(modelMerged)) {
        os << "EXPORT_VAR " << "std::uniform_real_distribution<" << model.getPrecision() << "> standardUniformDistribution;" << std::endl;
        os << "EXPORT_VAR " << "std::normal_distribution<" << model.getPrecision() << "> standardNormalDistribution;" << std::endl;
        os << "EXPORT_VAR " << "std::exponential_distribution<" << model.getPrecision() << "> standardExponentialDistribution;" << std::endl;
        os << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged &) const
{
    os << "#define SUPPORT_CODE_FUNC inline" << std::endl;

    // On windows, define an inline function, matching the signature of __builtin_clz which counts leading zeros
#ifdef _WIN32
    os << "#include <intrin.h>" << std::endl;
    os << std::endl;
    os << "int inline gennCLZ(unsigned int value)";
    {
        CodeStream::Scope b(os);
        os << "unsigned long leadingZero = 0;" << std::endl;
        os << "if( _BitScanReverse(&leadingZero, value))";
        {
            CodeStream::Scope b(os);
            os << "return 31 - leadingZero;" << std::endl;
        }
        os << "else";
        {
            CodeStream::Scope b(os);
            os << "return 32;" << std::endl;
        }
    }
    // Otherwise, on *nix, use __builtin_clz intrinsic
#else
    os << "#define gennCLZ __builtin_clz" << std::endl;
#endif
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // If a global RNG is required, implement standard host distributions as recreating them each call is slow
    if(isGlobalHostRNGRequired(modelMerged)) {
        os << "std::uniform_real_distribution<" << model.getPrecision() << "> standardUniformDistribution(" << model.scalarExpr(0.0) << ", " << model.scalarExpr(1.0) << ");" << std::endl;
        os << "std::normal_distribution<" << model.getPrecision() << "> standardNormalDistribution(" << model.scalarExpr(0.0) << ", " << model.scalarExpr(1.0) << ");" << std::endl;
        os << "std::exponential_distribution<" << model.getPrecision() << "> standardExponentialDistribution(" << model.scalarExpr(1.0) << ");" << std::endl;
        os << std::endl;
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream&, const ModelSpecMerged&) const
{
}
//--------------------------------------------------------------------------
void Backend::genStepTimeFinalisePreamble(CodeStream &, const ModelSpecMerged &) const
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
void Backend::genExtraGlobalParamDefinition(CodeStream &definitions, CodeStream &, 
                                            const std::string &type, const std::string &name, VarLocation) const
{
    definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    genVariableImplementation(os, type, name, loc);
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name, 
                                            VarLocation, const std::string &countVarName, const std::string &prefix) const
{
    // Get underlying type
    const std::string underlyingType = ::Utils::getUnderlyingType(type);
    const bool pointerToPointer = ::Utils::isTypePointerToPointer(type);

    const std::string pointer = pointerToPointer ? ("*" + prefix + name) : (prefix + name);

    os << pointer << " = new " << underlyingType << "[" << countVarName << "];" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPush(CodeStream &, const std::string &, const std::string &, 
                                      VarLocation, const std::string &, const std::string &) const
{
    assert(!m_Preferences.automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPull(CodeStream &, const std::string &, const std::string &, 
                                      VarLocation, const std::string &, const std::string &) const
{
    assert(!m_Preferences.automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genMergedExtraGlobalParamPush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                            const std::string &groupIdx, const std::string &fieldName, 
                                            const std::string &egpName) const
{
    os << "merged" << suffix << "Group" << mergedGroupIdx << "[" << groupIdx << "]." << fieldName << " = " << egpName << ";" << std::endl;
}
//--------------------------------------------------------------------------
std::string Backend::getMergedGroupFieldHostType(const std::string &type) const
{
    return type;
}
//--------------------------------------------------------------------------
std::string Backend::getMergedGroupSimRNGType() const
{
    assert(false);
    return "";
}
//--------------------------------------------------------------------------
void Backend::genPopVariableInit(CodeStream &os, const Substitutions &kernelSubs, Handler handler) const
{
    Substitutions varSubs(&kernelSubs);
    handler(os, varSubs);
}
//--------------------------------------------------------------------------
void Backend::genVariableInit(CodeStream &os, const std::string &count, const std::string &indexVarName,
                              const Substitutions &kernelSubs, Handler handler) const
{
     // **TODO** loops like this should be generated like CUDA threads
    os << "for (unsigned i = 0; i < (" << count << "); i++)";
    {
        CodeStream::Scope b(os);

        Substitutions varSubs(&kernelSubs);
        varSubs.addVarSubstitution(indexVarName, "i");
        handler(os, varSubs);
    }
}
//--------------------------------------------------------------------------
void Backend::genSynapseVariableRowInit(CodeStream &os, const SynapseGroupMergedBase &sg, 
                                        const Substitutions &kernelSubs, Handler handler) const
{
    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        os << "for (unsigned j = 0; j < group->rowLength[" << kernelSubs["id_pre"] << "]; j++)";
    }
    else {
        os << "for (unsigned j = 0; j < group->numTrgNeurons; j++)";
    }
    {
        CodeStream::Scope b(os);

        Substitutions varSubs(&kernelSubs);
        if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            varSubs.addVarSubstitution("id_syn", "(" + kernelSubs["id_pre"] + " * group->rowStride) + j");
            varSubs.addVarSubstitution("id_post", "group->ind[(" + kernelSubs["id_pre"] + " * group->rowStride) + j]");
        }
        else {
            varSubs.addVarSubstitution("id_syn", "(" + kernelSubs["id_pre"] + " * group->rowStride) + j");
            varSubs.addVarSubstitution("id_post", "j");
        }
        handler(os, varSubs);
    }
}
//--------------------------------------------------------------------------
void Backend::genVariablePush(CodeStream&, const std::string&, const std::string&, VarLocation, bool, size_t) const
{
    assert(!m_Preferences.automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genVariablePull(CodeStream&, const std::string&, const std::string&, VarLocation, size_t) const
{
    assert(!m_Preferences.automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePush(CodeStream &, const NeuronGroupInternal &, const std::string &, const std::string &, VarLocation) const
{
    assert(!m_Preferences.automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePull(CodeStream &, const NeuronGroupInternal &, const std::string &, const std::string &, VarLocation) const
{
    assert(!m_Preferences.automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genCurrentTrueSpikePush(CodeStream&, const NeuronGroupInternal&) const
{
    assert(!m_Preferences.automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genCurrentTrueSpikePull(CodeStream&, const NeuronGroupInternal&) const
{
    assert(!m_Preferences.automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikeLikeEventPush(CodeStream&, const NeuronGroupInternal&) const
{
    assert(!m_Preferences.automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikeLikeEventPull(CodeStream&, const NeuronGroupInternal&) const
{
    assert(!m_Preferences.automaticCopy);
}
//--------------------------------------------------------------------------
MemAlloc Backend::genGlobalDeviceRNG(CodeStream &, CodeStream &, CodeStream &, CodeStream &, CodeStream &) const
{
    assert(false);
    return MemAlloc::host(0);
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
void Backend::genReturnFreeDeviceMemoryBytes(CodeStream &os) const
{
    // There is no 'device' when using single-threaded CPU backend
    os << "return 0;" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefilePreamble(std::ostream &os) const
{
    std::string linkFlags = "-shared ";
    std::string cxxFlags = "-c -fPIC -std=c++11 -MMD -MP";
#ifdef __APPLE__
    cxxFlags += " -Wno-return-type-c-linkage";
#endif
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
    os << "\t@$(CXX) $(LINKFLAGS) -o $@ $(OBJECTS)" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileCompileRule(std::ostream &os) const
{
    os << "%.o: %.cc %.d" << std::endl;
    os << "\t@$(CXX) $(CXXFLAGS) -o $@ $<" << std::endl;
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
bool Backend::isGlobalHostRNGRequired(const ModelSpecMerged &modelMerged) const
{
    // If any neuron groups require simulation RNGs or require RNG for initialisation, return true
    // **NOTE** this takes postsynaptic model initialisation into account
    const ModelSpecInternal &model = modelMerged.getModel();
    if(std::any_of(model.getNeuronGroups().cbegin(), model.getNeuronGroups().cend(),
                   [](const ModelSpec::NeuronGroupValueType &n)
                   {
                       return n.second.isSimRNGRequired() || n.second.isInitRNGRequired();
                   }))
    {
        return true;
    }

    // If any synapse groups require an RNG for weight update model initialisation, return true
    if(std::any_of(model.getSynapseGroups().cbegin(), model.getSynapseGroups().cend(),
                   [](const ModelSpec::SynapseGroupValueType &s)
                   {
                       return (s.second.isWUInitRNGRequired() || s.second.isHostInitRNGRequired());
                   }))
    {
        return true;
    }

    return false;
}
//--------------------------------------------------------------------------
bool Backend::isGlobalDeviceRNGRequired(const ModelSpecMerged &) const
{
    return false;
}
//--------------------------------------------------------------------------
Backend::MemorySpaces Backend::getMergedGroupMemorySpaces(const ModelSpecMerged &) const
{
    return {{"", std::numeric_limits<size_t>::max()}};
}
//--------------------------------------------------------------------------
void Backend::genPresynapticUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg, const Substitutions &popSubs,
                                   bool trueSpike, PresynapticUpdateGroupMergedHandler wumThreshHandler, PresynapticUpdateGroupMergedHandler wumSimHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const auto *wu = sg.getArchetype().getWUModel();

    // Detect spike events or spikes and do the update
    os << "// process presynaptic events: " << (trueSpike ? "True Spikes" : "Spike type events") << std::endl;
    if (sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        os << "for (unsigned int i = 0; i < group->srcSpkCnt" << eventSuffix << "[preReadDelaySlot]; i++)";
    }
    else {
        os << "for (unsigned int i = 0; i < group->srcSpkCnt" << eventSuffix << "[0]; i++)";
    }
    {
        CodeStream::Scope b(os);
        if (!wu->getSimSupportCode().empty()) {
            os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) <<  ";" << std::endl;
        }

        const std::string queueOffset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
        os << "const unsigned int ipre = group->srcSpk" << eventSuffix << "[" << queueOffset << "i];" << std::endl;

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

        Substitutions synSubs(&popSubs);
        synSubs.addVarSubstitution("id_pre", "ipre");
        synSubs.addVarSubstitution("id_post", "ipost");
        synSubs.addVarSubstitution("id_syn", "synAddress");

        if(sg.getArchetype().isDendriticDelayRequired()) {
            synSubs.addFuncSubstitution("addToInSynDelay", 2, "group->denDelay[" + sg.getDendriticDelayOffset("$(1)") + "ipost] += $(0)");
        }
        else {
            synSubs.addFuncSubstitution("addToInSyn", 1, "group->inSyn[ipost] += $(0)");
        }

        if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            os << "const unsigned int npost = group->rowLength[ipre];" << std::endl;
            os << "for (unsigned int j = 0; j < npost; j++)";
            {
                CodeStream::Scope b(os);

                // **TODO** seperate stride from max connection
                os << "const unsigned int synAddress = (ipre * group->rowStride) + j;" << std::endl;
                os << "const unsigned int ipost = group->ind[synAddress];" << std::endl;

                wumSimHandler(os, sg, synSubs);
            }
        }
        else if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) {
            throw std::runtime_error("The single-threaded CPU backend does not support procedural connectivity.");
        }
        else if(m_Preferences.enableBitmaskOptimisations && (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK)) {
            // Determine the number of words in each row
            os << "const unsigned int rowWords = ((group->numTrgNeurons + 32 - 1) / 32);" << std::endl;
            os << "for(unsigned int w = 0; w < rowWords; w++)";
            {
                CodeStream::Scope b(os);

                // Read row word
                os << "uint32_t connectivityWord = group->gp[(ipre * rowWords) + w];" << std::endl;

                // Set ipost to first synapse in connectivity word
                os << "unsigned int ipost = w * 32;" << std::endl;

                // While there any bits left
                os << "while(connectivityWord != 0)";
                {
                    CodeStream::Scope b(os);

                    // Cound leading zeros (as bits are indexed backwards this is index of next synapse)
                    os << "const int numLZ = gennCLZ(connectivityWord);" << std::endl;

                    // Shift off zeros and the one just discovered
                    // **NOTE** << 32 appears to result in undefined behaviour
                    os << "connectivityWord = (numLZ == 31) ? 0 : (connectivityWord << (numLZ + 1));" << std::endl;

                    // Add to ipost
                    os << "ipost += numLZ;" << std::endl;

                    // If we aren't in padding region
                    // **TODO** don't bother checking if there is no padding
                    os << "if(ipost < group->numTrgNeurons)";
                    {
                        CodeStream::Scope b(os);
                        wumSimHandler(os, sg, synSubs);
                    }

                    // Increment ipost to take into account fact the next CLZ will go from bit AFTER synapse
                    os << "ipost++;" << std::endl;
                }
            }
        }
        // Otherwise (DENSE or BITMASK)
        else {
            os << "for (unsigned int ipost = 0; ipost < group->numTrgNeurons; ipost++)";
            {
                CodeStream::Scope b(os);

                if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "const uint64_t gid = (ipre * (uint64_t)group->numTrgNeurons + ipost);" << std::endl;
                    os << "if (B(group->gp[gid / 32], gid & 31))" << CodeStream::OB(20);
                }

                os << "const unsigned int synAddress = (ipre * group->numTrgNeurons) + ipost;" << std::endl;

                wumSimHandler(os, sg, synSubs);

                if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << CodeStream::CB(20);
                }
            }
        }
        // If this is a spike-like event, close braces around threshold check
        if (!trueSpike) {
            os << CodeStream::CB(10);
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genEmitSpike(CodeStream &os, const NeuronUpdateGroupMerged &ng, const Substitutions &subs, bool trueSpike) const
{
    // Determine if delay is required and thus, at what offset we should write into the spike queue
    const bool spikeDelayRequired = trueSpike ? (ng.getArchetype().isDelayRequired() && ng.getArchetype().isTrueSpikeRequired()) : ng.getArchetype().isDelayRequired();
    const std::string spikeQueueOffset = spikeDelayRequired ? "writeDelayOffset + " : "";

    const std::string suffix = trueSpike ? "" : "Evnt";
    os << "group->spk" << suffix << "[" << spikeQueueOffset << "group->spkCnt" << suffix;
    if(spikeDelayRequired) { // WITH DELAY
        os << "[*group->spkQuePtr]++]";
    }
    else { // NO DELAY
        os << "[0]++]";
    }
    os << " = " << subs["id"] << ";" << std::endl;

    // Reset spike time if this is a true spike and spike time is required
    if(trueSpike && ng.getArchetype().isSpikeTimeRequired()) {
        const std::string queueOffset = ng.getArchetype().isDelayRequired() ? "writeDelayOffset + " : "";
        os << "group->sT[" << queueOffset << subs["id"] << "] = " << subs["t"] << ";" << std::endl;
    }
}
}   // namespace SingleThreadedCPU
}   // namespace CodeGenerator

#include "backend.h"

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/environment.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/standardLibrary.h"
#include "code_generator/substitutions.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const EnvironmentLibrary::Library cpuSinglePrecisionFunctions = {
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Float, {}), "standardUniformDistribution(hostRNG)"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Float, {}), "standardNormalDistribution(hostRNG)"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Float, {}), "standardExponentialDistribution(hostRNG)"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Float, {}), "std::lognormal_distribution<float>($(0), $(1))(hostRNG)"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Float, {}), "std::gamma_distribution<float>($(0), 1.0f)(hostRNG)"}},
    {"gennrand_binomial", {Type::ResolvedType::createFunction(Type::Float, {}), "std::binomial_distribution<unsigned int>($(0), $(1))(hostRNG)"}},
};

const EnvironmentLibrary::Library cpuDoublePrecisionFunctions = {
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Float, {}), "standardUniformDistribution(hostRNG)"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Float, {}), "standardNormalDistribution(hostRNG)"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Float, {}), "standardExponentialDistribution(hostRNG)"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Float, {}), "std::lognormal_distribution<double>($(0), $(1))(hostRNG)"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Float, {}), "std::gamma_distribution<double>($(0), 1.0)(hostRNG)"}},
    {"gennrand_binomial", {Type::ResolvedType::createFunction(Type::Float, {}), "std::binomial_distribution<unsigned int>($(0), $(1))(hostRNG)"}},
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
template<typename G>
void genKernelIteration(EnvironmentExternal &env, const G &g, size_t numKernelDims, std::function<void(EnvironmentExternal&)>/*BackendBase::Handler*/ handler)
{
    EnvironmentSubstitute varEnv(env);

    // Define recursive function to generate nested kernel initialisation loops
    // **NOTE** this is a std::function as type of auto lambda couldn't be determined inside for recursive call
    std::function<void(size_t)> generateRecursive =
        [&handler, &varEnv, &g, &generateRecursive, numKernelDims]
        (size_t depth)
        {
            // Loop through this kernel dimensions
            const std::string idxVar = "k" + std::to_string(depth);
            varEnv.getStream() << "for(unsigned int " << idxVar << " = 0; " << idxVar << " < " << g.getKernelSize(depth) << "; " << idxVar << "++)";
            {
                CodeStream::Scope b(varEnv.getStream());
                EnvironmentSubstitute loopEnv(varEnv);

                // Add substitution for this kernel index
                loopEnv.addSubstitution("id_kernel_" + std::to_string(depth), idxVar);

                // If we've recursed through all dimensions
                if (depth == (numKernelDims - 1)) {
                    // Generate kernel index and use as "synapse" index
                    // **TODO** rename
                    assert(false);
                    //const size_t addSynapse = loopEnv.addInitialiser("const unsigned int kernelInd = " + g.genKernelIndex(loopEnv) + ";");
                    //loopEnv.addVarSubstitution("id_syn", "kernelInd", addSynapse);

                    // Call handler
                    handler(loopEnv);
                }
                // Otherwise, recurse
                else {
                    generateRecursive(depth + 1);
                }
            }
        };

    // Generate loops through kernel indices recursively
    generateRecursive(0);
}
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::SingleThreadedCPU::Backend
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::SingleThreadedCPU
{
void Backend::genNeuronUpdate(CodeStream &os, ModelSpecMerged &modelMerged, HostHandler preambleHandler) const
{
    if(modelMerged.getModel().getBatchSize() != 1) {
        throw std::runtime_error("The single-threaded CPU backend only supports simulations with a batch size of 1");
    }
   
    // Generate stream with neuron update code
    std::ostringstream neuronUpdateStream;
    CodeStream neuronUpdate(neuronUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary neuronUpdateEnv(neuronUpdate, StandardLibrary::getFunctions());

    neuronUpdateEnv.getStream() << "void updateNeurons(timepoint t";
    if(modelMerged.getModel().isRecordingInUse()) {
        neuronUpdateEnv.getStream() << ", unsigned int recordingTimestep";
    }
    neuronUpdateEnv.getStream() << ")";
    {
        CodeStream::Scope b(neuronUpdateEnv.getStream());

        EnvironmentExternal funcEnv(neuronUpdateEnv);
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t");
        funcEnv.add(Type::Uint32.addConst(), "batch", "0");
        
        Timer t(funcEnv.getStream(), "neuronUpdate", modelMerged.getModel().isTimingEnabled());
        modelMerged.genMergedNeuronPrevSpikeTimeUpdateGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &n)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged neuron prev spike update group " << n.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedNeuronPrevSpikeTimeUpdateGroup" << n.getIndex() << "[g]; " << std::endl;
                    
                    // Create matching environment
                    EnvironmentGroupMergedField<NeuronPrevSpikeTimeUpdateGroupMerged> groupEnv(funcEnv, n);
                    genNeuronIndexCalculation(groupEnv, 1);

                    if(n.getArchetype().isDelayRequired()) {
                        if(n.getArchetype().isPrevSpikeTimeRequired()) {
                            // Loop through neurons which spiked last timestep and set their spike time to time of previous timestep
                            groupEnv.getStream() << "for(unsigned int i = 0; i < " << groupEnv["_spk_cnt"] << "[" << groupEnv["_read_delay_slot"] << "]; i++)";
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                groupEnv.getStream() << groupEnv["_prev_spk_time"] << "[" << groupEnv["_read_delay_offset"] << " + " << groupEnv["_spk"] << "[" << groupEnv["_read_delay_offset"] << " + i]] = t - DT;" << std::endl;
                            }
                        }
                        if(n.getArchetype().isPrevSpikeEventTimeRequired()) {
                            // Loop through neurons which spiked last timestep and set their spike time to time of previous timestep
                            groupEnv.getStream() << "for(unsigned int i = 0; i < " << groupEnv["_spk_cnt_envt"] << "[" << groupEnv["_read_delay_slot"] << "]; i++)";
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                groupEnv.getStream() << groupEnv["_prev_spk_evnt_time"] << "[" << groupEnv["_read_delay_offset"] << " + " << groupEnv["_spk_evnt"] << "[" << groupEnv["_read_delay_offset"] << " + i]] = t - DT;" << std::endl;
                            }
                        }
                    }
                    else {
                        if(n.getArchetype().isPrevSpikeTimeRequired()) {
                            // Loop through neurons which spiked last timestep and set their spike time to time of previous timestep
                            groupEnv.getStream() << "for(unsigned int i = 0; i < " << groupEnv["_spk_cnt"] << "[0]; i++)";
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                groupEnv.getStream() << groupEnv["_prev_spk_time"] << "[" << groupEnv["_spk"] << "[i]] = t - DT;" << std::endl;
                            }
                        }
                        if(n.getArchetype().isPrevSpikeEventTimeRequired()) {
                            // Loop through neurons which spiked last timestep and set their spike time to time of previous timestep
                            groupEnv.getStream() << "for(unsigned int i = 0; i < " << groupEnv["_spk_cnt_evnt"] << "[0]; i++)";
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                groupEnv.getStream() << groupEnv["_prev_spk_evnt_time"] << "[" << groupEnv["_spk_evnt"] << "[i]] = t - DT;" << std::endl;
                            }
                        }
                    }
                }
            });

        // Loop through merged neuron spike queue update groups
        modelMerged.genMergedNeuronSpikeQueueUpdateGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &n)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged neuron spike queue update group " << n.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedNeuronSpikeQueueUpdateGroup" << n.getIndex() << "[g]; " << std::endl;
                    EnvironmentGroupMergedField<NeuronSpikeQueueUpdateGroupMerged> groupEnv(funcEnv, n);
                    genNeuronIndexCalculation(groupEnv, 1);

                    // Generate spike count reset
                    n.genMergedGroupSpikeCountReset(groupEnv, 1);
                }
            });

        // Loop through merged neuron update groups
        modelMerged.genMergedNeuronUpdateGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &n)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged neuron update group " << n.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedNeuronUpdateGroup" << n.getIndex() << "[g]; " << std::endl;
                    EnvironmentGroupMergedField<NeuronUpdateGroupMerged> groupEnv(funcEnv, n);

                    // If spike or spike-like event recording is in use
                    if(n.getArchetype().isSpikeRecordingEnabled() || n.getArchetype().isSpikeEventRecordingEnabled()) {
                        // Calculate number of words which will be used to record this population's spikes
                        groupEnv.getStream() << "const unsigned int numRecordingWords = (" << groupEnv["num_neurons"] << " + 31) / 32;" << std::endl;

                        // Zero spike recording buffer
                        if(n.getArchetype().isSpikeRecordingEnabled()) {
                            groupEnv.getStream() << "std::fill_n(&group->recordSpk[recordingTimestep * numRecordingWords], numRecordingWords, 0);" << std::endl;
                        }

                        // Zero spike-like-event recording buffer
                        if(n.getArchetype().isSpikeEventRecordingEnabled()) {
                            groupEnv.getStream() << "std::fill_n(&group->recordSpkEvent[recordingTimestep * numRecordingWords], numRecordingWords, 0);" << std::endl;
                        }
                    }

                    genNeuronIndexCalculation(groupEnv, 1);
                    groupEnv.getStream() << std::endl;

                    groupEnv.getStream() << "for(unsigned int i = 0; i < " << groupEnv["num_neurons"] << "; i++)";
                    {
                        CodeStream::Scope b(groupEnv.getStream());

                        groupEnv.add(Type::Uint32, "id", "i");

                        // Add RNG libray
                        EnvironmentLibrary rngEnv(groupEnv, (modelMerged.getModel().getPrecision() == Type::Float) ? cpuSinglePrecisionFunctions : cpuDoublePrecisionFunctions;

                        // Generate neuron update
                        n.generateNeuronUpdate(*this, rngEnv, modelMerged,
                                               // Emit true spikes
                                               [&modelMerged, this](EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng)
                                               {
                                                   // Insert code to update WU vars
                                                   ng.generateWUVarUpdate(*this, env, modelMerged);

                                                   // Insert code to emit true spikes
                                                   genEmitSpike(env, ng, true, ng.getArchetype().isSpikeRecordingEnabled());
                                               },
                                               // Emit spike-like events
                                               [this](EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng)
                                               {
                                                   // Insert code to emit spike-like events
                                                   genEmitSpike(env, ng, false, ng.getArchetype().isSpikeEventRecordingEnabled());
                                               });
                    }
                }
            });
    }

     // Generate struct definitions
    modelMerged.genMergedNeuronUpdateGroupStructs(os, *this);
    modelMerged.genMergedNeuronSpikeQueueUpdateStructs(os, *this);
    modelMerged.genMergedNeuronPrevSpikeTimeUpdateStructs(os, *this);

    // Generate arrays of merged structs and functions to set them
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronSpikeQueueUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups());

    // Generate preamble
    preambleHandler(os);

    os << neuronUpdateStream.str();

}
//--------------------------------------------------------------------------
void Backend::genSynapseUpdate(CodeStream &os, ModelSpecMerged &modelMerged, HostHandler preambleHandler) const
{
    if (modelMerged.getModel().getBatchSize() != 1) {
        throw std::runtime_error("The single-threaded CPU backend only supports simulations with a batch size of 1");
    }
    
    // Generate stream with synapse update code
    std::ostringstream synapseUpdateStream;
    CodeStream synapseUpdate(synapseUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary synapseUpdateEnv(synapseUpdate, StandardLibrary::getFunctions());

    synapseUpdateEnv.getStream() << "void updateSynapses(timepoint t)";
    {
        CodeStream::Scope b(synapseUpdateEnv.getStream());

        EnvironmentExternal funcEnv(synapseUpdateEnv);
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t");
        funcEnv.add(Type::Uint32.addConst(), "batch", "0");

        // Synapse dynamics
        {
            Timer t(funcEnv.getStream(), "synapseDynamics", modelMerged.getModel().isTimingEnabled());
            modelMerged.genMergedSynapseDynamicsGroups(
                *this,
                [this, &funcEnv, &modelMerged](auto &s)
                {
                    CodeStream::Scope b(funcEnv.getStream());
                    funcEnv.getStream() << "// merged synapse dynamics group " << s.getIndex() << std::endl;
                    funcEnv.getStream() << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                    {
                        CodeStream::Scope b(funcEnv.getStream());

                        // Get reference to group
                        funcEnv.getStream() << "const auto *group = &mergedSynapseDynamicsGroup" << s.getIndex() << "[g]; " << std::endl;

                        // Create matching environment
                        EnvironmentGroupMergedField<SynapseDynamicsGroupMerged> groupEnv(funcEnv, s);

                        // **TODO** rename as it does more!
                        genSynapseIndexCalculation(groupEnv, 1);

                        // Loop through presynaptic neurons
                        groupEnv.getStream() << "for(unsigned int i = 0; i < " << groupEnv["num_pre"] << "; i++)";
                        {
                            // If this synapse group has sparse connectivity, loop through length of this row
                            CodeStream::Scope b(groupEnv.getStream());
                            if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                groupEnv.getStream() << "for(unsigned int s = 0; s < " << groupEnv["_row_length"] << "[i]; s++)";
                            }
                            // Otherwise, if it's dense, loop through each postsynaptic neuron
                            else if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                                groupEnv.getStream() << "for (unsigned int j = 0; j < " << groupEnv["num_post"] << "; j++)";
                            }
                            else {
                                throw std::runtime_error("Only DENSE and SPARSE format connectivity can be used for synapse dynamics");
                            }
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                EnvironmentGroupMergedField<SynapseDynamicsGroupMerged> synEnv(groupEnv, s);

                                // Add presynaptic index to substitutions
                                synEnv.add(Type::Uint32.addConst(), "id_pre", "i");

                                if (s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    // Add initialiser strings to calculate synaptic and presynaptic index
                                    const size_t idSynInit = synEnv.addInitialiser("const unsigned int idSyn = (i * " + synEnv["_row_stride"] + ") + s;");
                                    const size_t idPostInit = synEnv.addInitialiser("const unsigned int idPost = " + synEnv["_ind"] + "[idSyn];");

                                    // **TODO** id_syn can be 64-bit
                                    synEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn", {idSynInit}, {"_row_stride"});
                                    synEnv.add(Type::Uint32.addConst(), "id_post", "idPost", {idPostInit, idSynInit}, {"_ind"});
                                }
                                else {
                                    // Add postsynaptic index to substitutions
                                    synEnv.add(Type::Uint32.addConst(), "id_post", "j");

                                    // Add initialiser to calculate synaptic index
                                    const size_t idSynInit = synEnv.addInitialiser("const unsigned int idSyn = (i * " + synEnv["num_post"] + ") + j;");

                                    // **TODO** id_syn can be 64-bit
                                    synEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn", {idSynInit});
                                }

                                // Add correct functions for apply synaptic input
                                synEnv.add(Type::AddToPostDenDelay, "addToPostDelay", synEnv["_den_delay"] + "[" + s.getPostDenDelayIndex(1, "j", "$(1)") + "] += $(0)",
                                           {}, {"_den_delay"});
                                synEnv.add(Type::AddToPost, "addToPost", synEnv["_out_post"] + "[" + s.getPostISynIndex(1, "j") + "] += $(0)",
                                           {}, {"_out_post"});
                                synEnv.add(Type::AddToPre, "addToPre", synEnv["_out_pre"] + "[" + s.getPreISynIndex(1, synEnv["id_pre"]) + "] += $(0)",
                                           {}, {"id_pre"}));
                                
                                // Call synapse dynamics handler
                                s.generateSynapseUpdate(*this, synEnv, modelMerged);
                            }
                        }
                    }
                });
        }

        // Presynaptic update
        {
            Timer t(funcEnv.getStream(), "presynapticUpdate", modelMerged.getModel().isTimingEnabled());
            modelMerged.genMergedPresynapticUpdateGroups(
                *this,
                [this, &funcEnv](auto &s)
                {
                    CodeStream::Scope b(funcEnv.getStream());
                    funcEnv.getStream() << "// merged presynaptic update group " << s.getIndex() << std::endl;
                    funcEnv.getStream() << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                    {
                        CodeStream::Scope b(funcEnv.getStream());

                        // Get reference to group
                        funcEnv.getStream() << "const auto *group = &mergedPresynapticUpdateGroup" << s.getIndex() << "[g]; " << std::endl;
                        
                        // Create matching environment
                        EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(funcEnv, s);

                        genSynapseIndexCalculation(groupEnv, 1);
                    
                        // generate the code for processing spike-like events
                        if (s.getArchetype().isSpikeEventRequired()) {
                            genPresynapticUpdate(groupEnv, modelMerged, s, false);
                        }

                        // generate the code for processing true spike events
                        if (s.getArchetype().isTrueSpikeRequired()) {
                            genPresynapticUpdate(groupEnv, modelMerged, s, true);
                        }
                        funcEnv.getStream() << std::endl;
                    }
                });
        }

        // Postsynaptic update
        {
            Timer t(funcEnv.getStream(), "postsynapticUpdate", modelMerged.getModel().isTimingEnabled());
            modelMerged.genMergedPostsynapticUpdateGroups(
                *this,
                [this, &funcEnv](auto &s)
                {
                    CodeStream::Scope b(funcEnv.getStream());
                    funcEnv.getStream() << "// merged postsynaptic update group " << s.getIndex() << std::endl;
                    funcEnv.getStream() << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                    {
                        CodeStream::Scope b(funcEnv.getStream());

                        // Get reference to group
                        funcEnv.getStream() << "const auto *group = &mergedPostsynapticUpdateGroup" << s.getIndex() << "[g]; " << std::endl;

                        // Create matching environment
                        EnvironmentGroupMergedField<PostsynapticUpdateGroupMerged> groupEnv(funcEnv, s);

                        genSynapseIndexCalculation(groupEnv, 1);

                        // Get number of postsynaptic spikes
                        if (s.getArchetype().getTrgNeuronGroup()->isDelayRequired() && s.getArchetype().getTrgNeuronGroup()->isTrueSpikeRequired()) {
                            groupEnv.getStream() << "const unsigned int numSpikes = group->trgSpkCnt[postDelaySlot];" << std::endl;
                        }
                        else {
                            groupEnv.getStream() << "const unsigned int numSpikes = group->trgSpkCnt[0];" << std::endl;
                        }

                        // Loop through postsynaptic spikes
                        groupEnv.getStream() << "for (unsigned int j = 0; j < numSpikes; j++)";
                        {
                            CodeStream::Scope b(groupEnv.getStream());

                            // **TODO** prod types
                            const std::string offsetTrueSpkPost = (s.getArchetype().getTrgNeuronGroup()->isTrueSpikeRequired() && s.getArchetype().getTrgNeuronGroup()->isDelayRequired()) ? "postDelayOffset + " : "";
                            groupEnv.getStream() << "const unsigned int spike = group->trgSpk[" << offsetTrueSpkPost << "j];" << std::endl;

                            // Loop through column of presynaptic neurons
                            if (s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                groupEnv.getStream() << "const unsigned int npre = group->colLength[spike];" << std::endl;
                                groupEnv.getStream() << "for (unsigned int i = 0; i < npre; i++)";
                            }
                            else {
                                groupEnv.getStream() << "for (unsigned int i = 0; i < " << groupEnv["num_pre"] << "; i++)";
                            }
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                EnvironmentGroupMergedField<PostsynapticUpdateGroupMerged> synEnv(groupEnv, s);

                                if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    // Add initialisers to calculate column and row-major indices
                                    // **TODO** fast divide optimisations
                                    const size_t colMajorIdxInit = synEnv.addInitialiser("const unsigned int colMajorIndex = (spike * " + synEnv["_col_stride"] + ") + i;");
                                    const size_t rowMajorIdxInit = synEnv.addInitialiser("const unsigned int rowMajorIndex = " + synEnv["_remap"] + "[colMajorIndex];");
                                    const size_t idPreInit = synEnv.addInitialiser("const unsigned int idPre = rowMajorIndex / " + synEnv["_row_stride"] + ";");

                                    // Add presynaptic and synapse index to environment
                                    synEnv.add("id_pre", "idPre", {colMajorIdxInit, rowMajorIdxInit, idPreInit}, {"_col_stride", "_row_stride", "_remap"});
                                    synEnv.add("id_syn", "rowMajorIndex", {colMajorIdxInit, rowMajorIdxInit}, {"_col_stride", "_remap"});
                                }
                                else {
                                    // Add initialiser to calculate synaptic index
                                    const size_t idSynInit = groupEnv.addInitialiser("const unsigned int idSyn = (i * " + synEnv["num_post"] + ") + spike;");

                                    // Add presynaptic and synapse index to environment
                                    synEnv.add(Type::Uint32, "id_pre", "i");
                                    synEnv.add(Type::Uint32, "id_syn", "idSyn", {idSynInit}, {"num_post"});
                                }

                                synEnv.add(Type::Uint32, "id_post", "spike");
                                synEnv.add(Type::AddToPre, "addToPre", synEnv["_out_pre"] + "[" + s.getPreISynIndex(1, synEnv["id_pre"]) + "] += $(0)");
            
                                s.generateSynapseUpdate(*this, synEnv, modelMerged);
                            }
                        }
                        groupEnv.getStream() << std::endl;
                    }
                });
        }
    }

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

    os << synapseUpdateStream.str();
}
//--------------------------------------------------------------------------
void Backend::genCustomUpdate(CodeStream &os, ModelSpecMerged &modelMerged, HostHandler preambleHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Build set containing names of all custom update groups
    std::set<std::string> customUpdateGroups;
    std::transform(model.getCustomUpdates().cbegin(), model.getCustomUpdates().cend(),
                   std::inserter(customUpdateGroups, customUpdateGroups.end()),
                   [](const ModelSpec::CustomUpdateValueType &v) { return v.second.getUpdateGroupName(); });
    std::transform(model.getCustomWUUpdates().cbegin(), model.getCustomWUUpdates().cend(),
                   std::inserter(customUpdateGroups, customUpdateGroups.end()),
                   [](const ModelSpec::CustomUpdateWUValueType &v) { return v.second.getUpdateGroupName(); });
    std::transform(model.getCustomConnectivityUpdates().cbegin(), model.getCustomConnectivityUpdates().cend(),
                   std::inserter(customUpdateGroups, customUpdateGroups.end()),
                   [](const ModelSpec::CustomConnectivityUpdateValueType &v) { return v.second.getUpdateGroupName(); });

    // Generate stream with custom update code
    std::ostringstream customUpdateStream;
    CodeStream customUpdate(customUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary customUpdateEnv(customUpdate, StandardLibrary::getFunctions());

    // Loop through custom update groups
    for(const auto &g : customUpdateGroups) {
        customUpdateEnv.getStream() << "void update" << g << "()";
        {
            CodeStream::Scope b(customUpdateEnv.getStream());

             EnvironmentExternal funcEnv(customUpdateEnv);
             funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t");
             funcEnv.add(Type::Uint32.addConst(), "batch", "0");


            // Loop through host update groups and generate code for those in this custom update group
            for (const auto &cg : modelMerged.getMergedCustomConnectivityHostUpdateGroups()) {
                if (cg.getArchetype().getUpdateGroupName() == g) {
                    assert(false);
                    //cg.generateUpdate(*this, os);
                }
            }

            {
                Timer t(funcEnv.getStream(), "customUpdate" + g, model.isTimingEnabled());
                modelMerged.genMergedCustomUpdateGroups(
                    *this, g,
                    [this, &funcEnv](auto &c)
                    {
                        CodeStream::Scope b(funcEnv.getStream());
                        funcEnv.getStream() << "// merged custom update group " << c.getIndex() << std::endl;
                        funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                        {
                            CodeStream::Scope b(funcEnv.getStream());

                            // Get reference to group
                            funcEnv.getStream() << "const auto *group = &mergedCustomUpdateGroup" << c.getIndex() << "[g]; " << std::endl;
                            
                            // Create matching environment
                            EnvironmentGroupMergedField<CustomUpdateGroupMerged> groupEnv(funcEnv, c);
                            
                            genCustomUpdateIndexCalculation(groupEnv, c);

                            if (c.getArchetype().isNeuronReduction()) {
                                // Initialise reduction targets
                                // **TODO** these should be provided with some sort of caching mechanism
                                const auto reductionTargets = genInitReductionTargets(groupEnv.getStream(), c);

                                // Loop through group members
                                groupEnv.getStream() << "for(unsigned int i = 0; i < group->size; i++)";
                                {
                                    CodeStream::Scope b(groupEnv.getStream());

                                    // Generate custom update
                                    EnvironmentGroupMergedField<CustomUpdateGroupMerged> memberEnv(groupEnv, c);
                                    memberEnv.addSubstitution("id", "i");
                                    c.generateCustomUpdate(*this, memberEnv);

                                    // Loop through reduction targets and generate reduction
                                    // **TODO** reduction should be automatically implemented by transpiler 
                                    for (const auto &r : reductionTargets) {
                                        memberEnv.getStream() << getReductionOperation("lr" + r.name, "l" + r.name, r.access, r.type) << ";" << std::endl;
                                    }
                                }

                                // Write back reductions
                                for (const auto &r : reductionTargets) {
                                    groupEnv.getStream() << "group->" << r.name << "[" << r.index << "] = lr" << r.name << ";" << std::endl;
                                }
                            }
                            else {
                                // Loop through group members
                                groupEnv.getStream() << "for(unsigned int i = 0; i < group->size; i++)";
                                {
                                    CodeStream::Scope b(groupEnv.getStream());

                                    // Generate custom update
                                    EnvironmentGroupMergedField<CustomUpdateGroupMerged> memberEnv(groupEnv, c);
                                    memberEnv.add(Type::Uint32.addConst(), "id", "i");
                                    c.generateCustomUpdate(*this, memberEnv);

                                    // Write back reductions
                                    genWriteBackReductions(memberEnv, c, "id");
                                }
                            }
                        }
                    });

                // Loop through merged custom WU update groups
                modelMerged.genMergedCustomUpdateWUGroups(
                    *this, g,
                    [this, &funcEnv](auto &c)
                    {
                        CodeStream::Scope b(funcEnv.getStream());
                        funcEnv.getStream() << "// merged custom WU update group " << c.getIndex() << std::endl;
                        funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                        {
                            CodeStream::Scope b(funcEnv.getStream());

                            // Get reference to group
                            funcEnv.getStream() << "const auto *group = &mergedCustomUpdateWUGroup" << c.getIndex() << "[g]; " << std::endl;

                            // Create matching environment
                            EnvironmentGroupMergedField<CustomUpdateWUGroupMerged> groupEnv(funcEnv, c);

                            // **TODO** add fields
                            const SynapseGroupInternal *sg = c.getArchetype().getSynapseGroup();
                            if (sg->getMatrixType() & SynapseMatrixWeight::KERNEL) {
                                genKernelIteration(groupEnv, c, c.getArchetype().getSynapseGroup()->getKernelSize().size(), 
                                                   [&c, this](EnvironmentExternalBase &env)
                                                   {
                                                       // Call custom update handler
                                                       c.generateCustomUpdate(*this, env);

                                                       // Write back reductions
                                                       genWriteBackReductions(env, c, "id_syn");
                                                   });
                            }
                            else {
                                // Loop through presynaptic neurons
                                groupEnv.getStream() << "for(unsigned int i = 0; i < group->numSrcNeurons; i++)";
                                {
                                    // If this synapse group has sparse connectivity, loop through length of this row
                                    CodeStream::Scope b(synEnv.getStream());
                                    if (sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                        groupEnv.getStream() << "for(unsigned int s = 0; s < group->rowLength[i]; s++)";
                                    }
                                    // Otherwise, if it's dense, loop through each postsynaptic neuron
                                    else if (sg->getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                                        groupEnv.getStream() << "for (unsigned int j = 0; j < group->numTrgNeurons; j++)";
                                    }
                                    else {
                                        throw std::runtime_error("Only DENSE and SPARSE format connectivity can be used for custom updates");
                                    }
                                    {
                                        CodeStream::Scope b(groupEnv.getStream());

                                        // Add presynaptic index to substitutions
                                        EnvironmentGroupMergedField<CustomUpdateGroupMerged> synEnv(groupEnv, c);
                                        synEnv.add(Type::Uint32.addConst(), "id_pre", "i");
                                        
                                        // If connectivity is sparse
                                        if (sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                            // Add initialisers to calculate synaptic index and thus lookup postsynaptic index
                                            const size_t idSynInit = synEnv.addInitialiser("const unsigned int idSyn = (i * group->rowStride) + s;");
                                            const size_t jInit = synEnv.addInitialiser("const unsigned int j = group->ind[idSyn];");

                                            // Add substitutions
                                            synEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn", {idSynInit}, {"_row_stride"});
                                            synEnv.add(Type::Uint32.addConst(), "id_post", "j", {jInit, idSynInit}, {"_ind", "_row_stride"});
                                        }
                                        else {
                                            synEnv.add(Type::Uint32.addConst(), "id_post", "j");

                                            const size_t idSynInit = ;
                                            synEnv.addSubstitution("id_syn", "idSyn", 
                                                                   {synEnv.addInitialiser("const unsigned int idSyn = (i * " + synEnv["num_post"] + ") + j;")},
                                        }

                                        // Generate custom update
                                        c.generateCustomUpdate(*this, synEnv);

                                        // Write back reductions
                                        genWriteBackReductions(synEnv, c, "id_syn");
                                    }
                                }
                            }
                        }
                    });
                
                // Loop through merged custom connectivity update groups
                modelMerged.genMergedCustomConnectivityUpdateGroups(
                    *this, g,
                    [this, &funcEnv](auto &c)
                    {
                        CodeStream::Scope b(funcEnv.getStream());
                        funcEnv.getStream() << "// merged custom connectivity update group " << c.getIndex() << std::endl;
                        funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                        {
                            CodeStream::Scope b(funcEnv.getStream());

                            // Get reference to group
                            funcEnv.getStream() << "const auto *group = &mergedCustomConnectivityUpdateGroup" << c.getIndex() << "[g]; " << std::endl;
                            
                            // Create matching environment
                            EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> groupEnv(funcEnv, c);

                            genCustomConnectivityUpdateIndexCalculation(funcEnv.getStream(), c);
                        
                            // Loop through presynaptic neurons
                            funcEnv.getStream() << "for(unsigned int i = 0; i < group->numSrcNeurons; i++)";
                            {
                                CodeStream::Scope b(funcEnv.getStream());
                            
                                // Configure substitutions
                                groupEnv.add(Type::Uint32, "id_pre", "i");
        
                                assert(false);
                                //c.generateUpdate(*this, cuEnv, model.getBatchSize());
                            }
                        }
                    });
            }

            // Loop through merged custom WU transpose update groups
            {
                Timer t(funcEnv.getStream(), "customUpdate" + g + "Transpose", model.isTimingEnabled());
                // Loop through merged custom connectivity update groups
                modelMerged.genMergedCustomUpdateTransposeWUGroups(
                    *this, g,
                    [this, &funcEnv](auto &c)
                    {
                        CodeStream::Scope b(funcEnv.getStream());
                        funcEnv.getStream() << "// merged custom WU transpose update group " << c.getIndex() << std::endl;
                        funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                        {
                            CodeStream::Scope b(funcEnv.getStream());

                            // Get reference to group
                            funcEnv.getStream() << "const auto *group = &mergedCustomUpdateTransposeWUGroup" << c.getIndex() << "[g]; " << std::endl;

                            // Create matching environment
                            EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> groupEnv(funcEnv, c);

                            // Get index of variable being transposed
                            const size_t transposeVarIdx = std::distance(c.getArchetype().getVarReferences().cbegin(),
                                                                         std::find_if(c.getArchetype().getVarReferences().cbegin(), c.getArchetype().getVarReferences().cend(),
                                                                                      [](const auto &v) { return v.second.getTransposeSynapseGroup() != nullptr; }));
                            const std::string transposeVarName = c.getArchetype().getCustomUpdateModel()->getVarRefs().at(transposeVarIdx).name;

                            // Loop through presynaptic neurons
                            groupEnv.getStream() << "for(unsigned int i = 0; i < group->numSrcNeurons; i++)";
                            {
                                CodeStream::Scope b(groupEnv.getStream());

                                // Loop through each postsynaptic neuron
                                groupEnv.getStream() << "for (unsigned int j = 0; j < group->numTrgNeurons; j++)";
                                {
                                    CodeStream::Scope b(groupEnv.getStream());

                                    // Add pre and postsynaptic indices to environment
                                    groupEnv.add(Type::Uint32, "id_pre", "i");
                                    groupEnv.add(Type::Uint32, "id_post", "j");
                                
                                    // Add conditional initialisation code to calculate synapse index
                                    groupEnv.addSubstitution(Type::Uint32, "id_syn", "idSyn", 
                                                             {groupEnv.addInitialiser("const unsigned int idSyn = (i * " + groupEnv["num_post"] + ") + j;")},
                                                             {"num_post"});
                                
                                    // Generate custom update
                                    c.generateCustomUpdate(*this, synEnv);

                                    // Update transpose variable
                                    // **YUCK** this is sorta outside scope
                                    synEnv.getStream() << groupEnv[transposeVarName + "_transpose"] << "[(j * " << groupEnv["num_pre"] << ") + i] = l" << transposeVarName << ";" << std::endl;
                                }
                            }

                        }
                    });
            }
        }
    }

    // Generate struct definitions
    modelMerged.genMergedCustomUpdateStructs(os, *this);
    modelMerged.genMergedCustomUpdateWUStructs(os, *this);
    modelMerged.genMergedCustomUpdateTransposeWUStructs(os, *this);
    modelMerged.genMergedCustomConnectivityUpdateStructs(os, *this);

    // Generate arrays of merged structs and functions to set them
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateWUGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateTransposeWUGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomConnectivityUpdateGroups());

    // Generate preamble
    preambleHandler(os);

    os << customUpdateStream.str();

}
//--------------------------------------------------------------------------
void Backend::genInit(CodeStream &_os, ModelSpecMerged &modelMerged, HostHandler preambleHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();
    if(model.getBatchSize() != 1) {
        throw std::runtime_error("The single-threaded CPU backend only supports simulations with a batch size of 1");
    }

    // Generate stream with neuron update code
    std::ostringstream initStream;
    CodeStream init(initStream);

    // Begin environment with standard library
    EnvironmentLibrary initEnv(init, StandardLibrary::getFunctions());

    initEnv.getStream() << "void initialize()";
    {
        CodeStream::Scope b(initEnv.getStream());
        EnvironmentExternal funcEnv(initEnv);

        Timer t(funcEnv.getStream(), "init", model.isTimingEnabled());

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Neuron groups" << std::endl;
        modelMerged.genMergedNeuronInitGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &n)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged neuron init group " << n.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedNeuronInitGroup" << n.getIndex() << "[g]; " << std::endl;
                    n.generateInit(*this, funcEnv, modelMerged);
                }
            });
        
        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Synapse groups" << std::endl;
        modelMerged.genMergedSynapseInitGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &s)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged synapse init group " << s.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(osfuncEnv.getStream();

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedSynapseInitGroup" << s.getIndex() << "[g]; " << std::endl;
                    s.generateInit(*this, funcEnv, modelMerged);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom update groups" << std::endl;
        modelMerged.genMergedCustomUpdateInitGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomUpdateInitGroup" << c.getIndex() << "[g]; " << std::endl;
                    c.generateInit(*this, funcEnv, modelMerged);
                }
            });
        
        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom connectivity presynaptic update groups" << std::endl;
        modelMerged.genMergedCustomConnectivityUpdatePreInitGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom connectivity presynaptic init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomConnectivityUpdatePreInitGroup" << c.getIndex() << "[g]; " << std::endl;
                    c.generateInit(*this, funcEnv, modelMerged);
                }
            });
        
        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom connectivity postsynaptic update groups" << std::endl;
        modelMerged.genMergedCustomConnectivityUpdatePostInitGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom connectivity postsynaptic init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomConnectivityUpdatePostInitGroup" << c.getIndex() << "[g]; " << std::endl;
                    c.generateInit(*this, os, modelMerged, popSubs);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom WU update groups" << std::endl;
        modelMerged.genMergedCustomWUUpdateInitGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom WU update group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(os);

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomWUUpdateInitGroup" << c.getIndex() << "[g]; " << std::endl;
                    c.generateInit(*this, os, modelMerged, popSubs);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Synapse sparse connectivity" << std::endl;
        modelMerged.genMergedSynapseConnectivityInitGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged synapse connectivity init group " << s.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedSynapseConnectivityInitGroup" << s.getIndex() << "[g]; " << std::endl;

                    // If matrix connectivity is ragged
                    if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        // Zero row lengths
                        funcEnv.getStream() << "std::fill_n(group->rowLength, group->numSrcNeurons, 0);" << std::endl;
                    }
                    else if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        funcEnv.getStream() << "const size_t gpSize = ((((size_t)group->numSrcNeurons * (size_t)group->rowStride) + 32 - 1) / 32);" << std::endl;
                        funcEnv.getStream() << "std::fill(group->gp, gpSize, 0);" << std::endl;
                    }
                    else {
                        throw std::runtime_error("Only BITMASK and SPARSE format connectivity can be generated using a connectivity initialiser");
                    }

                    // If there is row-building code in this snippet
                    EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> groupEnv(funcEnv, c);
                    const auto *snippet = s.getArchetype().getConnectivityInitialiser().getSnippet();
                    if(!snippet->getRowBuildCode().empty()) {
                        // Generate loop through source neurons
                        groupEnv.getStream() << "for (unsigned int i = 0; i < group->numSrcNeurons; i++)";

                        // Configure substitutions
                        groupEnv.add(Type::Uint32.addConst(), "id_pre", "i");
                        groupEnv.add(Type::Uint32.addConst(), "id_post_begin", "0");
                        groupEnv.add(Type::Uint32.addConst(), "id_thread", "0");
                        groupEnv.add(Type::Uint32.addConst(), "num_threads", "1");
                        //groupEnv.add("num_pre", "group->numSrcNeurons");
                        //groupEnv.add("num_post", "group->numTrgNeurons");
                    }
                    // Otherwise
                    else {
                        assert(!snippet->getColBuildCode().empty());

                        // Loop through target neurons
                        groupEnv.getStream() << "for (unsigned int j = 0; j < group->numTrgNeurons; j++)";

                        // Configure substitutions
                        groupEnv.add(Type::Uint32.addConst(), "id_post", "j");
                        groupEnv.add(Type::Uint32.addConst(), "id_pre_begin", "0");
                        groupEnv.add(Type::Uint32.addConst(), "id_thread", "0");
                        groupEnv.add(Type::Uint32.addConst(), "num_threads", "1");
                        //popSubs.addVarSubstitution("num_pre", "group->numSrcNeurons");
                        //popSubs.addVarSubstitution("num_post", "group->numTrgNeurons");
                    }
                    {
                        CodeStream::Scope b(os);

                        // Create new stream to generate addSynapse function which initializes all kernel variables
                        std::ostringstream addSynapseStream;
                        CodeStream addSynapse(addSynapseStream);

                        // Use classic macro trick to turn block of initialization code into statement and 'eat' semicolon
                        addSynapse << "do";
                        {
                            CodeStream::Scope b(addSynapse);

                            // Calculate index in data structure of this synapse
                            if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                if(!snippet->getRowBuildCode().empty()) {
                                    addSynapse << "const unsigned int idx = " << "(" + groupEnv["id_pre"] + " * " << groupEnv["_row_stride"] << ") + " << groupEnv["_row_length"] << "[i];" << std::endl;
                                }
                                else {
                                    addSynapse << "const unsigned int idx = " << "(($(0)) * " << groupEnv["_row_stride"] << ") + groupEnv["_row_length"][$(0)];" << std::endl;
                                }
                            }

                            // If there is a kernel
                            if(!s.getArchetype().getKernelSize().empty()) {
                                EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> kernelInitEnv(groupEnv, c);

                                // Replace $(id_post) with first 'function' parameter as simulation code is
                                // going to be, in turn, substituted into procedural connectivity generation code
                                assert(false);
                                if(!snippet->getRowBuildCode().empty()) {
                                    kernelInitEnv.add(Type::Uint32.addConst(), "id_post", "$(0)");
                                }
                                else {
                                     kernelInitEnv.add(Type::Uint32.addConst(), "id_pre", "$(0)");
                                }

                                // Add index of synapse
                                if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    kernelInitEnv.add(Type::Uint32.addConst(), "id_syn", "idx");
                                }

                                // Replace kernel indices with the subsequent 'function' parameters
                                for(size_t i = 0; i < s.getArchetype().getKernelSize().size(); i++) {
                                    kernelInitEnv.add(Type::Uint32.addConst(), "id_kernel_" + std::to_string(i), "$(" + std::to_string(i + 1) + ")");
                                }

                                // Call handler to initialize variables
                                s.generateKernelInit(*this, addSynapse, modelMerged, kernelInitSubs);
                            }

                            // If there is row-building code in this snippet
                            if(!snippet->getRowBuildCode().empty()) {
                                // If matrix is sparse, add function to increment row length and insert synapse into ind array
                                if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    addSynapse << groupEnv["_ind"] << "[idx] = $(0);" << std::endl;
                                    addSynapse << groupEnv["_row_length"] << "[i]++;" << std::endl;
                                }
                                // Otherwise, add function to set correct bit in bitmask
                                else {
                                    addSynapse << "const int64_t rowStartGID = i * " << groupEnv["_row_stride"] << ";" << std::endl;
                                    addSynapse << "setB(group->gp[(rowStartGID + ($(0))) / 32], (rowStartGID + $(0)) & 31);" << std::endl;
                                }
                            }
                            // Otherwise
                            else {
                                // If matrix is sparse, add function to increment row length and insert synapse into ind array
                                if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    addSynapse << groupEnv["_ind"] << "[idx] = " << groupEnv["id_post"] << ";" << std::endl;
                                    addSynapse << groupEnv["_row_length"] << "[$(0)]++;" << std::endl;
                                }
                                else {
                                    addSynapse << "const int64_t colStartGID = j;" << std::endl;
                                    addSynapse << "setB(" << groupEnv["_gp"] << "[(colStartGID + (($(0)) * " << groupEnv["_row_stride"] << ")) / 32], ((colStartGID + (($(0)) * " << groupEnv["_row_stride"] << ")) & 31));" << std::endl;
                                }
                            }
                        }
                        addSynapse << "while(false)";

                        const auto addSynapseType = Type::ResolvedType::createFunction(Type::Void, std::vector<Type::ResolvedType>{1ull + s.getArchetype().getKernelSize().size(), Type::Uint32});
                        groupEnv.add(addSynapseType, "addSynapse", addSynapseStream.str());

                        // Call appropriate connectivity handler
                        if(!snippet->getRowBuildCode().empty()) {
                            s.generateSparseRowInit(*this, groupEnv, modelMerged);
                        }
                        else {
                            s.generateSparseColumnInit(*this, groupEnv, modelMerged);
                        }
                    }
                }
            });
    }
    initEnv.getStream() << std::endl;
    initEnv.getStream() << "void initializeSparse()";
    {
        CodeStream::Scope b(initEnv.getStream());
        EnvironmentExternal funcEnv(initEnv);

        Timer t(funcEnv.getStream(), "initSparse", model.isTimingEnabled());

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Synapse groups with sparse connectivity" << std::endl;
        modelMerged.genMergedSynapseSparseInitGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &s)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged sparse synapse init group " << s.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedSynapseSparseInitGroup" << s.getIndex() << "[g]; " << std::endl;

                    // If postsynaptic learning is required, initially zero column lengths
                    if (!s.getArchetype().getWUModel()->getLearnPostCode().empty()) {
                        funcEnv.getStream() << "// Zero column lengths" << std::endl;
                        funcEnv.getStream() << "std::fill_n(group->colLength, group->numTrgNeurons, 0);" << std::endl;
                    }

                    funcEnv.getStream() << "// Loop through presynaptic neurons" << std::endl;
                    funcEnv.getStream() << "for (unsigned int i = 0; i < group->numSrcNeurons; i++)" << std::endl;
                    {
                        CodeStream::Scope b(funcEnv.getStream());

                        // Generate sparse initialisation code
                        if(s.getArchetype().isWUVarInitRequired()) {
                            Substitutions popSubs(&funcSubs);
                            popSubs.addVarSubstitution("id_pre", "i");
                            popSubs.addVarSubstitution("row_len", "group->rowLength[i]");
                            s.generateInit(*this, os, modelMerged, popSubs);
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
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom sparse WU update groups" << std::endl;
        modelMerged.genMergedCustomWUUpdateSparseInitGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom sparse WU update group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomWUUpdateSparseInitGroup" << c.getIndex() << "[g]; " << std::endl;

                    os << "// Loop through presynaptic neurons" << std::endl;
                    os << "for (unsigned int i = 0; i < group->numSrcNeurons; i++)" << std::endl;
                    {
                        CodeStream::Scope b(os);

                        // Generate initialisation code  
                        Substitutions popSubs(&funcSubs);
                        popSubs.addVarSubstitution("id_pre", "i");
                        popSubs.addVarSubstitution("row_len", "group->rowLength[i]");
                        c.generateInit(*this, os, modelMerged, popSubs);
                    }
                }
            });
        
        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom connectivity update sparse init groups" << std::endl;
         modelMerged.genMergedCustomConnectivityUpdateSparseInitGroups(
            *this,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(os);
                os << "// merged custom connectivity update sparse init group " << c.getIndex() << std::endl;
                os << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(os);

                    // Get reference to group
                    os << "const auto *group = &mergedCustomConnectivityUpdateSparseInitGroup" << c.getIndex() << "[g]; " << std::endl;

                    os << "// Loop through presynaptic neurons" << std::endl;
                    os << "for (unsigned int i = 0; i < group->numSrcNeurons; i++)" << std::endl;
                    {
                        CodeStream::Scope b(os);

                        // Generate initialisation code  
                        Substitutions popSubs(&funcSubs);
                        popSubs.addVarSubstitution("id_pre", "i");
                        popSubs.addVarSubstitution("row_len", "group->rowLength[i]");
                        c.generateInit(*this, os, modelMerged, popSubs);
                    }
                }
            });
    }

    
    // Generate struct definitions
    modelMerged.genMergedNeuronInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseInitGroupStructs(os, *this);
    modelMerged.genMergedCustomUpdateInitGroupStructs(os, *this);
    modelMerged.genMergedCustomWUUpdateInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseConnectivityInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseSparseInitGroupStructs(os, *this);
    modelMerged.genMergedCustomWUUpdateSparseInitGroupStructs(os, *this);
    modelMerged.genMergedCustomConnectivityUpdatePreInitStructs(os, *this);
    modelMerged.genMergedCustomConnectivityUpdatePostInitStructs(os, *this);
    modelMerged.genMergedCustomConnectivityUpdateSparseInitStructs(os, *this);

    // Generate arrays of merged structs and functions to set them
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomWUUpdateInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseConnectivityInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseSparseInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomWUUpdateSparseInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomConnectivityUpdatePreInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomConnectivityUpdatePostInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomConnectivityUpdateSparseInitGroups());
    
    // Generate preamble
    preambleHandler(os);

    os << initStream.str();

}
//--------------------------------------------------------------------------
size_t Backend::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    if ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) || (sg.getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ)) {
        return sg.getMaxConnections();
    }
    else if(getPreferences().enableBitmaskOptimisations && (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK)) {
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
    if(model.getBatchSize() != 1) {
        throw std::runtime_error("The single-threaded CPU backend only supports simulations with a batch size of 1");
    }

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
        os << "EXPORT_VAR " << "std::uniform_real_distribution<" << model.getPrecision().getName() << "> standardUniformDistribution;" << std::endl;
        os << "EXPORT_VAR " << "std::normal_distribution<" << model.getPrecision().getName() << "> standardNormalDistribution;" << std::endl;
        os << "EXPORT_VAR " << "std::exponential_distribution<" << model.getPrecision().getName() << "> standardExponentialDistribution;" << std::endl;
        os << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged &) const
{
    os << "#define SUPPORT_CODE_FUNC inline" << std::endl;

    // CUDA and OpenCL both provide generic min and max functions 
    // to match this, bring std::min and std::max into global namespace
    os << "using std::min;" << std::endl;
    os << "using std::max;" << std::endl;

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
void Backend::genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc&) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // If a global RNG is required, implement standard host distributions as recreating them each call is slow
    if(isGlobalHostRNGRequired(modelMerged)) {
        os << "std::uniform_real_distribution<" << model.getPrecision().getName() << "> standardUniformDistribution(" << modelMerged.scalarExpr(0.0) << ", " << modelMerged.scalarExpr(1.0) << ");" << std::endl;
        os << "std::normal_distribution<" << model.getPrecision().getName() << "> standardNormalDistribution(" << modelMerged.scalarExpr(0.0) << ", " << modelMerged.scalarExpr(1.0) << ");" << std::endl;
        os << "std::exponential_distribution<" << model.getPrecision().getName() << "> standardExponentialDistribution(" << modelMerged.scalarExpr(1.0) << ");" << std::endl;
        os << std::endl;
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream&, const ModelSpecMerged&, const MemAlloc&) const
{
}
//--------------------------------------------------------------------------
void Backend::genFreeMemPreamble(CodeStream&, const ModelSpecMerged&) const
{
}
//--------------------------------------------------------------------------
void Backend::genStepTimeFinalisePreamble(CodeStream &, const ModelSpecMerged &) const
{
}
//--------------------------------------------------------------------------
void Backend::genVariableDefinition(CodeStream &definitions, CodeStream &, 
                                    const Type::ResolvedType &type, const std::string &name, VarLocation) const
{
    definitions << "EXPORT_VAR " << type.getValue().name << "* " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genVariableInstantiation(CodeStream &os, 
                                       const Type::ResolvedType &type, const std::string &name, VarLocation) const
{
    os << type.getValue().name << "* " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genVariableAllocation(CodeStream &os, const Type::ResolvedType &type, const std::string &name, 
                                    VarLocation, size_t count, MemAlloc &memAlloc) const
{
    os << name << " = new " << type.getValue().name << "[" << count << "];" << std::endl;

    memAlloc += MemAlloc::host(count * type.getValue().size);
}
//--------------------------------------------------------------------------
void Backend::genVariableDynamicAllocation(CodeStream &os, 
                                           const Type::ResolvedType &type, const std::string &name, VarLocation, 
                                           const std::string &countVarName, const std::string &prefix) const
{
    if (type.isPointer()) {
        os << "*" << prefix << name <<  " = new " << type.getPointer().valueType->getValue().name << "[" << countVarName << "];" << std::endl;
    }
    else {
        os << prefix << name << " = new " << type.getValue().name << "[" << countVarName << "];" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableFree(CodeStream &os, const std::string &name, VarLocation) const
{
    os << "delete[] " << name << ";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genVariablePush(CodeStream&, const Type::ResolvedType&, const std::string&, VarLocation, bool, size_t) const
{
    assert(!getPreferences().automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genVariablePull(CodeStream&, const Type::ResolvedType&, const std::string&, VarLocation, size_t) const
{
    assert(!getPreferences().automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePush(CodeStream&, const NeuronGroupInternal&, 
                                     const Type::ResolvedType&, const std::string&, 
                                     VarLocation, unsigned int) const
{
    assert(!getPreferences().automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePull(CodeStream&, const NeuronGroupInternal&, 
                                     const Type::ResolvedType&, const std::string&, 
                                     VarLocation, unsigned int) const
{
    assert(!getPreferences().automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genVariableDynamicPush(CodeStream&, 
                                     const Type::ResolvedType&, const std::string&,
                                     VarLocation, const std::string&, const std::string&) const
{
     assert(!getPreferences().automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genVariableDynamicPull(CodeStream&, 
                                     const Type::ResolvedType&, const std::string&,
                                      VarLocation, const std::string&, const std::string&) const
{
    assert(!getPreferences().automaticCopy);
}
//--------------------------------------------------------------------------
void Backend::genMergedDynamicVariablePush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                           const std::string &groupIdx, const std::string &fieldName,
                                           const std::string &egpName) const
{
    os << "merged" << suffix << "Group" << mergedGroupIdx << "[" << groupIdx << "]." << fieldName << " = " << egpName << ";" << std::endl;
}

//--------------------------------------------------------------------------
std::string Backend::getMergedGroupFieldHostTypeName(const Type::ResolvedType &type) const
{
    return type.getName();
}
//--------------------------------------------------------------------------
std::optional<Type::ResolvedType> Backend::getMergedGroupSimRNGType() const
{
    assert(false);
    return std::nullopt;
}
//--------------------------------------------------------------------------
void Backend::genPopVariableInit(EnvironmentExternalBase &env, HandlerEnv handler) const
{
    handler(env);
}
//--------------------------------------------------------------------------
void Backend::genVariableInit(EnvironmentExternalBase &env, const std::string &count, const std::string &indexVarName, HandlerEnv handler) const
{
    // **TODO** loops like this should be generated like CUDA threads
    env.getStream() << "for (unsigned int i = 0; i < (" << env[count] << "); i++)";
    {
        CodeStream::Scope b(env.getStream());

        EnvironmentExternal varEnv(env);
        varEnv.add(Type::Uint32, indexVarName, "i");
        handler(varEnv);
    }
}
//--------------------------------------------------------------------------
void Backend::genSparseSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const
{
    env.getStream() << "for (unsigned int j = 0; j < group->rowLength[" << env["id_pre"] << "]; j++)";
    {
        CodeStream::Scope b(env.getStream());

        EnvironmentExternal varEnv(env);
        // **TODO** 64-bit
        varEnv.add(Type::Uint32, "id_syn", "idSyn",
                   {varEnv.addInitialiser("const unsigned int idSyn = (" + varEnv["id_pre"] + " * " + varEnv["_row_stride"] + ") + j;")},
                   {"id_pre", "_rowStride"});
        varEnv.add(Type::Uint32, "id_post", "idPost",
                   {varEnv.addInitialiser("const unsigned int idPost = (" + varEnv["_ind"] + "[(" + varEnv["id_pre"] + " * " + varEnv["_row_stride"] + ") + j]");
        handler(varEnv);
     }
}
//--------------------------------------------------------------------------
void Backend::genDenseSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const
{
    env.getStream() << "for (unsigned int j = 0; j < " << env["num_post"] << "; j++)";
    {
        CodeStream::Scope b(env.getStream());

        EnvironmentExternal varEnv(env);
        // **TODO** 64-bit
        varEnv.add(Type::Uint32, "id_syn", "idSyn",
                   {varEnv.addInitialiser("const unsigned int idSyn = (" + varEnv["id_pre"] + " * " + varEnv["_row_stride"] + ") + j;")},
                   {"id_pre", "_rowStride"});
        varEnv.add(Type::Uint32, "id_post", "j");
        handler(varEnv);
    }
}
//--------------------------------------------------------------------------
void Backend::genKernelSynapseVariableInit(EnvironmentExternalBase &env, const SynapseInitGroupMerged &sg, HandlerEnv handler) const
{
    assert(false);
    //genKernelIteration(os, sg, sg.getArchetype().getKernelSize().size(), kernelSubs, handler);
}
//--------------------------------------------------------------------------
void Backend::genKernelCustomUpdateVariableInit(EnvironmentExternalBase &env, const CustomWUUpdateInitGroupMerged &cu, HandlerEnv handler) const
{
    assert(false);
    //genKernelIteration(os, cu, cu.getArchetype().getSynapseGroup()->getKernelSize().size(), kernelSubs, handler);
}
//--------------------------------------------------------------------------
void Backend::genGlobalDeviceRNG(CodeStream&, CodeStream&, CodeStream&, CodeStream&, CodeStream&, MemAlloc&) const
{
    assert(false);
}
//--------------------------------------------------------------------------
void Backend::genPopulationRNG(CodeStream&, CodeStream&, CodeStream&, CodeStream&, CodeStream&,
                               const std::string&, size_t, MemAlloc&) const
{
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
void Backend::genAssert(CodeStream &os, const std::string &condition) const
{
    os << "assert(" << condition << ");" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefilePreamble(std::ostream &os) const
{
    std::string linkFlags = "-shared ";
    std::string cxxFlags = "-c -fPIC -std=c++11 -MMD -MP";
#ifdef __APPLE__
    cxxFlags += " -Wno-return-type-c-linkage";
#endif
    cxxFlags += " " + getPreferences().userCxxFlagsGNU;
    if (getPreferences().optimizeCode) {
        cxxFlags += " -O3 -ffast-math";
    }
    if (getPreferences().debugCode) {
        cxxFlags += " -O0 -g";
    }

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
    os << "\t\t\t<FloatingPointModel>" << (getPreferences().optimizeCode ? "Fast" : "Precise") << "</FloatingPointModel>" << std::endl;
    os << "\t\t\t<MultiProcessorCompilation>true</MultiProcessorCompilation>" << std::endl;
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

    // If any custom updates require an RNG fo initialisation, return true
    if(std::any_of(model.getCustomUpdates().cbegin(), model.getCustomUpdates().cend(),
                   [](const ModelSpec::CustomUpdateValueType &c)
                   {
                       return (c.second.isInitRNGRequired());
                   }))
    {
        return true;
    }

    // If any custom WU updates require an RNG fo initialisation, return true
    if(std::any_of(model.getCustomWUUpdates().cbegin(), model.getCustomWUUpdates().cend(),
                   [](const ModelSpec::CustomUpdateWUValueType &c)
                   {
                       return (c.second.isInitRNGRequired());
                   }))
    {
        return true;
    }

    // If any custom connectivity updates require an RNG fo initialisation or per-row, return true
    if(std::any_of(model.getCustomConnectivityUpdates().cbegin(), model.getCustomConnectivityUpdates().cend(),
                   [](const ModelSpec::CustomConnectivityUpdateValueType &c)
                   {
                       return (c.second.isVarInitRNGRequired()
                               || c.second.isPreVarInitRNGRequired()
                               || c.second.isPostVarInitRNGRequired()
                               || c.second.isRowSimRNGRequired()
                               || c.second.isHostRNGRequired());
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
    return {};
}
//--------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Backend::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash was name of backend
    Utils::updateHash("SingleThreadedCPU", hash);
    
    // Update hash with preferences
    getPreferences<Preferences>().updateHash(hash);

    return hash.get_digest();
}
//--------------------------------------------------------------------------
void Backend::genPresynapticUpdate(EnvironmentExternalBase &env, const PresynapticUpdateGroupMerged &sg, const ModelSpecMerged &modelMerged, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const auto *wu = sg.getArchetype().getWUModel();

    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) {
        const auto &connectInit = sg.getArchetype().getToeplitzConnectivityInitialiser();

        // Loop through Toeplitz matrix diagonals
        env.getStream() << "for(unsigned int j = 0; j < group->rowStride; j++)";
        {
            /*CodeStream::Scope b(env.getStream());

            // Create substitution stack for generating procedural connectivity code
            Substitutions connSubs(&popSubs);
            connSubs.addVarSubstitution("id_diag", "j");

            // Add substitutions
            connSubs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams(),
                                               [&sg](const std::string &p) { return sg.isToeplitzConnectivityInitParamHeterogeneous(p);  },
                                               "", "group->");
            connSubs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams(),
                                             [&sg](const std::string &p) { return sg.isToeplitzConnectivityInitDerivedParamHeterogeneous(p);  },
                                             "", "group->");
            connSubs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", "group->");
            connSubs.addVarNameSubstitution(connectInit.getSnippet()->getDiagonalBuildStateVars());

            // Initialise any diagonal build state variables defined
            for (const auto &d : connectInit.getSnippet()->getDiagonalBuildStateVars()) {
                // Apply substitutions to value
                std::string value = d.value;
                connSubs.applyCheckUnreplaced(value, "toeplitz diagonal build state var : merged" + std::to_string(sg.getIndex()));
                //value = ensureFtype(value, modelMerged.getModel().getPrecision());

                os << d.type.resolve(sg.getTypeContext()).getName() << " " << d.name << " = " << value << ";" << std::endl;
            }

             // Detect spike events or spikes and do the update
            os << "// process presynaptic events: " << (trueSpike ? "True Spikes" : "Spike type events") << std::endl;
            if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                os << "for (unsigned int i = 0; i < group->srcSpkCnt" << eventSuffix << "[preDelaySlot]; i++)";
            }
            else {
                os << "for (unsigned int i = 0; i < group->srcSpkCnt" << eventSuffix << "[0]; i++)";
            }
            {
                CodeStream::Scope b(os);

                const std::string queueOffset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "preDelayOffset + " : "";
                os << "const unsigned int ipre = group->srcSpk" << eventSuffix << "[" << queueOffset << "i];" << std::endl;

                // Create another substitution stack for generating presynaptic simulation code
                Substitutions presynapticUpdateSubs(&popSubs);
                connSubs.addVarSubstitution("id_pre", "ipre");
                presynapticUpdateSubs.addVarSubstitution("id_pre", "ipre");

                if(!wu->getSimSupportCode().empty()) {
                    os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
                }

                // If this is a spike-like event, insert threshold check for this presynaptic neuron
                if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << "if(";

                    // Generate weight update threshold condition
                    sg.generateSpikeEventThreshold(*this, os, modelMerged, presynapticUpdateSubs);

                    os << ")";
                    os << CodeStream::OB(10);
                }

                // Replace $(id_post) with first 'function' parameter as simulation code is
                // going to be, in turn, substituted into procedural connectivity generation code
                presynapticUpdateSubs.addVarSubstitution("id_post", "$(0)");

                // Replace kernel indices with the subsequent 'function' parameters
                for(size_t i = 0; i < sg.getArchetype().getKernelSize().size(); i++) {
                    presynapticUpdateSubs.addVarSubstitution("id_kernel_" + std::to_string(i),
                                                             "$(" + std::to_string(i + 1) + ")");
                }

                if(sg.getArchetype().isDendriticDelayRequired()) {
                    presynapticUpdateSubs.addFuncSubstitution("addToInSynDelay", 2, "group->denDelay[" + sg.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "] += $(0)");
                }
                else {
                    presynapticUpdateSubs.addFuncSubstitution("addToInSyn", 1, "group->inSyn[" + sg.getPostISynIndex(1, "$(id_post)") + "] += $(0)");
                }

                if(sg.getArchetype().isPresynapticOutputRequired()) {
                    presynapticUpdateSubs.addFuncSubstitution("addToPre", 1, "group->revInSyn[" + sg.getPreISynIndex(1, "ipre") + "] += $(0)");
                }

                // Generate presynaptic simulation code into new stringstream-backed code stream
                std::ostringstream presynapticUpdateStream;
                CodeStream presynapticUpdate(presynapticUpdateStream);
                if(trueSpike) {
                    sg.generateSpikeUpdate(*this, presynapticUpdate, modelMerged, presynapticUpdateSubs);
                }
                else {
                    sg.generateSpikeEventUpdate(*this, presynapticUpdate, modelMerged, presynapticUpdateSubs);
                }

                // When a synapse should be 'added', substitute in presynaptic update code
                connSubs.addFuncSubstitution("addSynapse", 1 + (unsigned int)sg.getArchetype().getKernelSize().size(), presynapticUpdateStream.str());

                // Generate toeplitz connectivity code
                sg.generateToeplitzConnectivity(*this, os, connSubs);

                if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
            }*/
        }
    }
    else {
        // Detect spike events or spikes and do the update
        env.getStream() << "// process presynaptic events: " << (trueSpike ? "True Spikes" : "Spike type events") << std::endl;
        if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
            env.getStream() << "for (unsigned int i = 0; i < group->srcSpkCnt" << eventSuffix << "[preDelaySlot]; i++)";
        }
        else {
            env.getStream() << "for (unsigned int i = 0; i < group->srcSpkCnt" << eventSuffix << "[0]; i++)";
        }
        {
            CodeStream::Scope b(env.getStream());
            /*if(!wu->getSimSupportCode().empty()) {
                os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
            }*/
            EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(env, sg);


            const std::string queueOffset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "preDelayOffset + " : "";
            groupEnv.add(Type::Uint32, "id_pre", "idPre",
                         {groupEnv.addInitialiser("const unsigned int ipre = group->srcSpk" + eventSuffix + "[" + queueOffset + "i];")});

            // If this is a spike-like event, insert threshold check for this presynaptic neuron
            if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                groupEnv.getStream() << "if(";

                // Generate weight update threshold condition
                sg.generateSpikeEventThreshold(*this, groupEnv, modelMerged);

                groupEnv.getStream() << ")";
                groupEnv.getStream() << CodeStream::OB(10);
            }

            // Add correct functions for apply synaptic input
            groupEnv.add(Type::AddToPostDenDelay, "addToPostDelay", env["_den_delay"] + "[" + sg.getPostDenDelayIndex(1, "j", "$(1)") + "] += $(0)",
                         {}, {"_den_delay"});
            groupEnv.add(Type::AddToPost, "addToPost", env["_out_post"] + "[" + sg.getPostISynIndex(1, "j") + "] += $(0)",
                         {}, {"_out_post"});
            groupEnv.add(Type::AddToPre, "addToPre", env["_out_pre"] + "[" + sg.getPreISynIndex(1, env["id_pre"]) + "] += $(0)",
                         {}, {"id_pre"});

            // If connectivity is sparse
            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                groupEnv.getStream() << "const unsigned int npost = group->rowLength[ipre];" << std::endl;
                groupEnv.getStream() << "for (unsigned int j = 0; j < npost; j++)";
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(groupEnv, sg);

                    // **TODO** 64-bit id_syn
                    synEnv.add(Type::Uint32, "id_syn", "idSyn",
                               {synEnv.addInitialiser("const unsigned int idSyn = (ipre * " + env["_row_stride"] + ") + j;")},
                               {"_row_stride"});
                    synEnv.add(Type::Uint32, "id_post", "idPost",
                               {synEnv.addInitialiser("const unsigned int idPost = " + env["_ind"] + "[idSyn];")},
                               {"_ind", "id_syn"});
                    
                    if(trueSpike) {
                        sg.generateSpikeUpdate(*this, synEnv, modelMerged);
                    }
                    else {
                        sg.generateSpikeEventUpdate(*this, synEnv, modelMerged);
                    }
                }
            }
            else if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) {
                throw std::runtime_error("The single-threaded CPU backend does not support procedural connectivity.");
            }
            else if(getPreferences().enableBitmaskOptimisations && (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK)) {
                // Determine the number of words in each row
                groupEnv.getStream() << "const unsigned int rowWords = ((" << env["_num_post"] << " + 32 - 1) / 32);" << std::endl;
                groupEnv.getStream() << "for(unsigned int w = 0; w < rowWords; w++)";
                {
                    CodeStream::Scope b(groupEnv.getStream());

                    // Read row word
                    groupEnv.getStream() << "uint32_t connectivityWord = group->gp[(ipre * rowWords) + w];" << std::endl;

                    // Set ipost to first synapse in connectivity word
                    groupEnv.getStream() << "unsigned int ipost = w * 32;" << std::endl;
                    groupEnv.add(Type::Uint32, "id_post", "ipost");

                    // While there any bits left
                    groupEnv.getStream() << "while(connectivityWord != 0)";
                    {
                        CodeStream::Scope b(groupEnv.getStream());

                        // Cound leading zeros (as bits are indexed backwards this is index of next synapse)
                        groupEnv.getStream() << "const int numLZ = gennCLZ(connectivityWord);" << std::endl;

                        // Shift off zeros and the one just discovered
                        // **NOTE** << 32 appears to result in undefined behaviour
                        groupEnv.getStream() << "connectivityWord = (numLZ == 31) ? 0 : (connectivityWord << (numLZ + 1));" << std::endl;

                        // Add to ipost
                        groupEnv.getStream() << "ipost += numLZ;" << std::endl;

                        // If we aren't in padding region
                        // **TODO** don't bother checking if there is no padding
                        groupEnv.getStream() << "if(ipost < group->numTrgNeurons)";
                        {
                            CodeStream::Scope b(env.getStream());
                            if(trueSpike) {
                                sg.generateSpikeUpdate(*this, groupEnv, modelMerged);
                            }
                            else {
                                sg.generateSpikeEventUpdate(*this, groupEnv, modelMerged);
                            }
                        }

                        // Increment ipost to take into account fact the next CLZ will go from bit AFTER synapse
                        groupEnv.getStream() << "ipost++;" << std::endl;
                    }
                }
            }
            // Otherwise (DENSE or BITMASK)
            else {
                groupEnv.getStream() << "for (unsigned int ipost = 0; ipost < group->numTrgNeurons; ipost++)";
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(groupEnv, sg);
                    synEnv.add(Type::Uint32, "id_post", "ipost");

                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // **TODO** 64-bit index
                        synEnv.getStream() << "const uint64_t gid = (ipre * group->numTrgNeurons + ipost);" << std::endl;

                        synEnv.getStream() << "if (B(group->gp[gid / 32], gid & 31))" << CodeStream::OB(20);
                    }
                    else {
                        synEnv.add(Type::Uint32, "id_syn", "idSyn",
                                   {synEnv.addInitialiser("const unsigned int idSyn = (ipre * " + synEnv["num_post"] + ") + ipost;")},
                                   {"num_post"});
                    }

                   
                    if(trueSpike) {
                        sg.generateSpikeUpdate(*this, synEnv, modelMerged);
                    }
                    else {
                        sg.generateSpikeEventUpdate(*this, synEnv, modelMerged);
                    }

                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        synEnv.getStream() << CodeStream::CB(20);
                    }
                }
            }
            // If this is a spike-like event, close braces around threshold check
            if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                groupEnv.getStream() << CodeStream::CB(10);
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genEmitSpike(EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng, bool trueSpike, bool recordingEnabled) const
{
    // Determine if delay is required and thus, at what offset we should write into the spike queue
    const bool spikeDelayRequired = trueSpike ? (ng.getArchetype().isDelayRequired() && ng.getArchetype().isTrueSpikeRequired()) : ng.getArchetype().isDelayRequired();
    const std::string spikeQueueOffset = spikeDelayRequired ? "writeDelayOffset + " : "";

    const std::string suffix = trueSpike ? "" : "Evnt";
    env.getStream() << "group->spk" << suffix << "[" << spikeQueueOffset << "group->spkCnt" << suffix;
    if(spikeDelayRequired) { // WITH DELAY
        env.getStream() << "[*group->spkQuePtr]++]";
    }
    else { // NO DELAY
        env.getStream() << "[0]++]";
    }
    env.getStream() << " = " << env["id"] << ";" << std::endl;

    // Reset spike and spike-like-event times
    const std::string queueOffset = ng.getArchetype().isDelayRequired() ? "writeDelayOffset + " : "";
    if(trueSpike && ng.getArchetype().isSpikeTimeRequired()) {
        env.getStream() << "group->sT[" << queueOffset << env["id"] << "] = " << env["t"] << ";" << std::endl;
    }
    else if(!trueSpike && ng.getArchetype().isSpikeEventTimeRequired()) {
        env.getStream() << "group->seT[" << queueOffset << env["id"] << "] = " << env["t"] << ";" << std::endl;
    }
    
    // If recording is enabled
    if(recordingEnabled) {
        const std::string recordSuffix = trueSpike ? "" : "Event";
        env.getStream() << "group->recordSpk" << recordSuffix << "[(recordingTimestep * numRecordingWords) + (" << env["id"] << " / 32)]";
        env.getStream() << " |= (1 << (" << env["id"] << " % 32));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genWriteBackReductions(EnvironmentExternal &env, const CustomUpdateGroupMerged &cg, const std::string &idxName) const
{
    genWriteBackReductions(env, cg, idxName,
                           [&cg](const Models::VarReference &varRef, const std::string &index)
                           {
                               return cg.getVarRefIndex(varRef.getDelayNeuronGroup() != nullptr,
                                                        getVarAccessDuplication(varRef.getVar().access),
                                                        index);
                           });
}
//--------------------------------------------------------------------------
void Backend::genWriteBackReductions(EnvironmentExternal &env, const CustomUpdateWUGroupMerged &cg, const std::string &idxName) const
{
    genWriteBackReductions(env, cg, idxName,
                           [&cg](const Models::WUVarReference &varRef, const std::string &index)
                           {
                               return cg.getVarRefIndex(getVarAccessDuplication(varRef.getVar().access),
                                                        index);
                           });
}
}   // namespace GeNN::CodeGenerator::SingleThreadedCPU

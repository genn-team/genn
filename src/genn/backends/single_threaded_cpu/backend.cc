#include "backend.h"

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/environment.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/standardLibrary.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
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
void genKernelIteration(EnvironmentExternalBase &env, G &g, size_t numKernelDims, BackendBase::HandlerEnv handler)
{
    // Define recursive function to generate nested kernel initialisation loops
    // **NOTE** this is a std::function as type of auto lambda couldn't be determined inside for recursive call
    std::function<void(EnvironmentExternalBase &env, size_t)> generateRecursive =
        [&handler, &g, &generateRecursive, numKernelDims]
        (EnvironmentExternalBase &env, size_t depth)
        {
            // Loop through this kernel dimensions
            const std::string idxVar = "k" + std::to_string(depth);
            env.print("for(unsigned int " + idxVar + " = 0; " + idxVar + " < " + getKernelSize(g, depth) + "; " + idxVar + "++)");
            {
                CodeStream::Scope b(env.getStream());
                EnvironmentGroupMergedField<G> loopEnv(env, g);

                // Add substitution for this kernel index
                loopEnv.add(Type::Uint32.addConst(), "id_kernel_" + std::to_string(depth), idxVar);

                // If we've recursed through all dimensions
                if (depth == (numKernelDims - 1)) {
                    // Generate kernel index and use as "synapse" index
                    // **TODO** rename
                    loopEnv.add(Type::Uint32.addConst(), "id_syn", "kernelInd", 
                                {loopEnv.addInitialiser("const unsigned int kernelInd = " + getKernelIndex(g) + ";")});

                    // Call handler
                    handler(loopEnv);
                }
                // Otherwise, recurse
                else {
                    generateRecursive(loopEnv, depth + 1);
                }
            }
        };

    // Generate loops through kernel indices recursively
    generateRecursive(env, 0);
}
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::SingleThreadedCPU::Backend
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::SingleThreadedCPU
{
void Backend::genNeuronUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                              HostHandler preambleHandler) const
{
    if(modelMerged.getModel().getBatchSize() != 1) {
        throw std::runtime_error("The single-threaded CPU backend only supports simulations with a batch size of 1");
    }
   
    // Generate stream with neuron update code
    std::ostringstream neuronUpdateStream;
    CodeStream neuronUpdate(neuronUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary neuronUpdateEnv(neuronUpdate, StandardLibrary::getMathsFunctions());

    neuronUpdateEnv.getStream() << "void updateNeurons(" << modelMerged.getModel().getTimePrecision().getName() << " t";
    if(modelMerged.getModel().isRecordingInUse()) {
        neuronUpdateEnv.getStream() << ", unsigned int recordingTimestep";
    }
    neuronUpdateEnv.getStream() << ")";
    {
        CodeStream::Scope b(neuronUpdateEnv.getStream());

        EnvironmentExternal funcEnv(neuronUpdateEnv);
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t");
        funcEnv.add(Type::Uint32.addConst(), "batch", "0");
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "dt", 
                    writePreciseLiteral(modelMerged.getModel().getDT(), modelMerged.getModel().getTimePrecision()));
        
        Timer t(funcEnv.getStream(), "neuronUpdate", modelMerged.getModel().isTimingEnabled());
        modelMerged.genMergedNeuronPrevSpikeTimeUpdateGroups(
            *this, memorySpaces,
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
                    buildStandardEnvironment(groupEnv, 1);

                    if(n.getArchetype().isDelayRequired()) {
                        if(n.getArchetype().isPrevSpikeTimeRequired()) {
                            // Loop through neurons which spiked last timestep and set their spike time to time of previous timestep
                            groupEnv.print("for(unsigned int i = 0; i < $(_spk_cnt)[$(_read_delay_slot)]; i++)");
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                groupEnv.printLine("$(_prev_spk_time)[$(_read_delay_offset) + $(_spk)[$(_read_delay_offset) + i]] = t - $(dt);");
                            }
                        }
                        if(n.getArchetype().isPrevSpikeEventTimeRequired()) {
                            // Loop through neurons which spiked last timestep and set their spike time to time of previous timestep
                            groupEnv.print("for(unsigned int i = 0; i < $(_spk_cnt_envt)[$(_read_delay_slot)]; i++)");
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                groupEnv.printLine("$(_prev_spk_evnt_time)[$(_read_delay_offset) + $(_spk_evnt)[$(_read_delay_offset) + i]] = t - $(dt);");
                            }
                        }
                    }
                    else {
                        if(n.getArchetype().isPrevSpikeTimeRequired()) {
                            // Loop through neurons which spiked last timestep and set their spike time to time of previous timestep
                            groupEnv.print("for(unsigned int i = 0; i < $(_spk_cnt)[0]; i++)");
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                groupEnv.printLine("$(_prev_spk_time)[$(_spk)[i]] = t - $(dt);");
                            }
                        }
                        if(n.getArchetype().isPrevSpikeEventTimeRequired()) {
                            // Loop through neurons which spiked last timestep and set their spike time to time of previous timestep
                            groupEnv.print("for(unsigned int i = 0; i < $(_spk_cnt_evnt)[0]; i++)");
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                groupEnv.printLine("$(_prev_spk_evnt_time)[$(_spk_evnt)[i]] = t - $(dt);");
                            }
                        }
                    }
                }
            });

        // Loop through merged neuron spike queue update groups
        modelMerged.genMergedNeuronSpikeQueueUpdateGroups(
            *this, memorySpaces,
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
                    buildStandardEnvironment(groupEnv, 1);

                    // Generate spike count reset
                    n.genMergedGroupSpikeCountReset(groupEnv, 1);
                }
            });

        // Loop through merged neuron update groups
        modelMerged.genMergedNeuronUpdateGroups(
            *this, memorySpaces,
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
                    buildStandardEnvironment(groupEnv, 1);

                    // If spike or spike-like event recording is in use
                    if(n.getArchetype().isSpikeRecordingEnabled() || n.getArchetype().isSpikeEventRecordingEnabled()) {
                        // Calculate number of words which will be used to record this population's spikes
                        groupEnv.printLine("const unsigned int numRecordingWords = ($(num_neurons) + 31) / 32;");

                        // Zero spike recording buffer
                        if(n.getArchetype().isSpikeRecordingEnabled()) {
                            groupEnv.printLine("std::fill_n(&$(_record_spk)[recordingTimestep * numRecordingWords], numRecordingWords, 0);");
                        }

                        // Zero spike-like-event recording buffer
                        if(n.getArchetype().isSpikeEventRecordingEnabled()) {
                            groupEnv.printLine("std::fill_n(&$(_record_spk_event)[recordingTimestep * numRecordingWords], numRecordingWords, 0);");
                        }
                    }

                    groupEnv.getStream() << std::endl;

                    groupEnv.print("for(unsigned int i = 0; i < $(num_neurons); i++)");
                    {
                        CodeStream::Scope b(groupEnv.getStream());

                        groupEnv.add(Type::Uint32, "id", "i");

                        // Add RNG libray
                        EnvironmentLibrary rngEnv(groupEnv, StandardLibrary::getHostRNGFunctions(modelMerged.getModel().getPrecision()));

                        // Generate neuron update
                        n.generateNeuronUpdate(*this, rngEnv, 1,
                                               // Emit true spikes
                                               [this](EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng)
                                               {
                                                   // Insert code to update WU vars
                                                   ng.generateWUVarUpdate(*this, env, 1);

                                                   // Insert code to emit true spikes
                                                   genEmitSpike(env, ng, true, ng.getArchetype().isSpikeRecordingEnabled());
                                               },
                                               // Emit spike-like events
                                               [this](EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng)
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
void Backend::genSynapseUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                               HostHandler preambleHandler) const
{
    if (modelMerged.getModel().getBatchSize() != 1) {
        throw std::runtime_error("The single-threaded CPU backend only supports simulations with a batch size of 1");
    }
    
    // Generate stream with synapse update code
    std::ostringstream synapseUpdateStream;
    CodeStream synapseUpdate(synapseUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary synapseUpdateEnv(synapseUpdate, StandardLibrary::getMathsFunctions());

    synapseUpdateEnv.getStream() << "void updateSynapses(" << modelMerged.getModel().getTimePrecision().getName() << " t)";
    {
        CodeStream::Scope b(synapseUpdateEnv.getStream());

        EnvironmentExternal funcEnv(synapseUpdateEnv);
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t");
        funcEnv.add(Type::Uint32.addConst(), "batch", "0");
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "dt", 
                    writePreciseLiteral(modelMerged.getModel().getDT(), modelMerged.getModel().getTimePrecision()));

        // Synapse dynamics
        {
            Timer t(funcEnv.getStream(), "synapseDynamics", modelMerged.getModel().isTimingEnabled());
            modelMerged.genMergedSynapseDynamicsGroups(
                *this, memorySpaces,
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
                        buildStandardEnvironment(groupEnv, 1);

                        // Loop through presynaptic neurons
                        groupEnv.print("for(unsigned int i = 0; i < $(num_pre); i++)");
                        {
                            // If this synapse group has sparse connectivity, loop through length of this row
                            CodeStream::Scope b(groupEnv.getStream());
                            if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                groupEnv.print("for(unsigned int s = 0; s < $(_row_length)[i]; s++)");
                            }
                            // Otherwise, if it's dense, loop through each postsynaptic neuron
                            else if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                                groupEnv.print("for (unsigned int j = 0; j < $(num_post); j++)");
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
                                    const size_t idSynInit = synEnv.addInitialiser("const unsigned int idSyn = (i * $(_row_stride)) + s;");
                                    const size_t idPostInit = synEnv.addInitialiser("const unsigned int idPost = $(_ind)[$(id_syn)];");

                                    // **TODO** id_syn can be 64-bit
                                    synEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn", {idSynInit});
                                    synEnv.add(Type::Uint32.addConst(), "id_post", "idPost", {idPostInit, idSynInit});
                                }
                                else {
                                    // Add postsynaptic index to substitutions
                                    synEnv.add(Type::Uint32.addConst(), "id_post", "j");

                                    // Add initialiser to calculate synaptic index
                                    // **TODO** id_syn can be 64-bit
                                    synEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn", 
                                               {synEnv.addInitialiser("const unsigned int idSyn = (i * $(num_post)) + j;")});
                                }

                                // Add correct functions for apply synaptic input
                                synEnv.add(Type::AddToPostDenDelay, "addToPostDelay", "$(_den_delay)[" + s.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "] += $(0)");
                                synEnv.add(Type::AddToPost, "addToPost", "$(_out_post)[" + s.getPostISynIndex(1, "$(id_post)") + "] += $(0)");
                                synEnv.add(Type::AddToPre, "addToPre", "$(_out_pre)[" + s.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");
                                
                                // Call synapse dynamics handler
                                s.generateSynapseUpdate(*this, synEnv, 1);
                            }
                        }
                    }
                });
        }

        // Presynaptic update
        {
            Timer t(funcEnv.getStream(), "presynapticUpdate", modelMerged.getModel().isTimingEnabled());
            modelMerged.genMergedPresynapticUpdateGroups(
                *this, memorySpaces,
                [this, &funcEnv, &modelMerged](auto &s)
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
                        buildStandardEnvironment(groupEnv, 1);
                    
                        // generate the code for processing spike-like events
                        if (s.getArchetype().isSpikeEventRequired()) {
                            genPresynapticUpdate(groupEnv, s, modelMerged, false);
                        }

                        // generate the code for processing true spike events
                        if (s.getArchetype().isTrueSpikeRequired()) {
                            genPresynapticUpdate(groupEnv, s, modelMerged, true);
                        }
                        funcEnv.getStream() << std::endl;
                    }
                });
        }

        // Postsynaptic update
        {
            Timer t(funcEnv.getStream(), "postsynapticUpdate", modelMerged.getModel().isTimingEnabled());
            modelMerged.genMergedPostsynapticUpdateGroups(
                *this, memorySpaces,
                [this, &funcEnv, &modelMerged](auto &s)
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
                        buildStandardEnvironment(groupEnv, 1);

                        // Get number of postsynaptic spikes
                        if (s.getArchetype().getTrgNeuronGroup()->isDelayRequired() && s.getArchetype().getTrgNeuronGroup()->isTrueSpikeRequired()) {
                            groupEnv.printLine("const unsigned int numSpikes = $(_trg_spk_cnt)[$(_post_delay_slot)];");
                        }
                        else {
                            groupEnv.printLine("const unsigned int numSpikes = $(_trg_spk_cnt)[0];");
                        }

                        // Loop through postsynaptic spikes
                        groupEnv.getStream() << "for (unsigned int j = 0; j < numSpikes; j++)";
                        {
                            CodeStream::Scope b(groupEnv.getStream());

                            // **TODO** prod types
                            const std::string offsetTrueSpkPost = (s.getArchetype().getTrgNeuronGroup()->isTrueSpikeRequired() && s.getArchetype().getTrgNeuronGroup()->isDelayRequired()) ? "$(_post_delay_offset) + " : "";
                            groupEnv.printLine("const unsigned int spike = $(_trg_spk)[" + offsetTrueSpkPost + "j];");

                            // Loop through column of presynaptic neurons
                            if (s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                groupEnv.printLine("const unsigned int npre = $(_col_length)[spike];");
                                groupEnv.getStream() << "for (unsigned int i = 0; i < npre; i++)";
                            }
                            else {
                                groupEnv.print("for (unsigned int i = 0; i < $(num_pre); i++)");
                            }
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                                EnvironmentGroupMergedField<PostsynapticUpdateGroupMerged> synEnv(groupEnv, s);

                                if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    // Add initialisers to calculate column and row-major indices
                                    // **TODO** fast divide optimisations
                                    const size_t colMajorIdxInit = synEnv.addInitialiser("const unsigned int colMajorIndex = (spike * $(_col_stride)) + i;");
                                    const size_t rowMajorIdxInit = synEnv.addInitialiser("const unsigned int rowMajorIndex = $(_remap)[colMajorIndex];");
                                    const size_t idPreInit = synEnv.addInitialiser("const unsigned int idPre = rowMajorIndex / $(_row_stride);");

                                    // Add presynaptic and synapse index to environment
                                    synEnv.add(Type::Uint32.addConst(), "id_pre", "idPre", {colMajorIdxInit, rowMajorIdxInit, idPreInit});
                                    synEnv.add(Type::Uint32.addConst(), "id_syn", "rowMajorIndex", {colMajorIdxInit, rowMajorIdxInit});
                                }
                                else {
                                    // Add presynaptic and synapse index to environment
                                    synEnv.add(Type::Uint32.addConst(), "id_pre", "i");
                                    synEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn", 
                                               {synEnv.addInitialiser("const unsigned int idSyn = (i * $(num_post)) + spike;")});
                                }

                                synEnv.add(Type::Uint32.addConst(), "id_post", "spike");
                                synEnv.add(Type::AddToPre, "addToPre", "$(_out_pre)[" + s.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");
            
                                s.generateSynapseUpdate(*this, synEnv, 1);
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
void Backend::genCustomUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                              HostHandler preambleHandler) const
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
    EnvironmentLibrary customUpdateEnv(customUpdate, StandardLibrary::getMathsFunctions());

    // Loop through custom update groups
    for(const auto &g : customUpdateGroups) {
        customUpdateEnv.getStream() << "void update" << g << "()";
        {
            CodeStream::Scope b(customUpdateEnv.getStream());

             EnvironmentExternal funcEnv(customUpdateEnv);
             funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t");
             funcEnv.add(Type::Uint32.addConst(), "batch", "0");
             funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "dt", 
                    writePreciseLiteral(modelMerged.getModel().getDT(), modelMerged.getModel().getTimePrecision()));

            // Loop through host update groups and generate code for those in this custom update group
            modelMerged.genMergedCustomConnectivityHostUpdateGroups(
                *this, memorySpaces, g, 
                [this, &customUpdateEnv, &modelMerged](auto &c)
                {
                    c.generateUpdate(*this, customUpdateEnv);
                });
            
            {
                Timer t(funcEnv.getStream(), "customUpdate" + g, model.isTimingEnabled());
                modelMerged.genMergedCustomUpdateGroups(
                    *this, memorySpaces, g,
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
                            buildStandardEnvironment(groupEnv);

                            if (c.getArchetype().isNeuronReduction()) {
                                // Initialise reduction targets
                                const auto reductionTargets = genInitReductionTargets(groupEnv.getStream(), c);

                                // Loop through group members
                                groupEnv.print("for(unsigned int i = 0; i < $(size); i++)");
                                {
                                    CodeStream::Scope b(groupEnv.getStream());
                                    EnvironmentGroupMergedField<CustomUpdateGroupMerged> memberEnv(groupEnv, c);
                                    memberEnv.add(Type::Uint32.addConst(), "id", "i");

                                    c.generateCustomUpdate(
                                        *this, memberEnv,
                                        [&reductionTargets, this](auto &env, const auto&)
                                        {        
                                            // Loop through reduction targets and generate reduction
                                            for (const auto &r : reductionTargets) {
                                                env.printLine(getReductionOperation("_lr" + r.name, "$(" + r.name + ")", r.access, r.type) + ";");
                                            }
                                        });
                                }

                                // Write back reductions
                                for (const auto &r : reductionTargets) {
                                    groupEnv.getStream() << "group->" << r.name << "[" << r.index << "] = _lr" << r.name << ";" << std::endl;
                                }
                            }
                            else {
                                // Loop through group members
                                EnvironmentGroupMergedField<CustomUpdateGroupMerged> memberEnv(groupEnv, c);
                                if (c.getArchetype().isPerNeuron()) {
                                    memberEnv.print("for(unsigned int i = 0; i < $(size); i++)");
                                    memberEnv.add(Type::Uint32.addConst(), "id", "i");
                                }
                                else {
                                    memberEnv.add(Type::Uint32.addConst(), "id", "0");
                                }
                                {
                                    CodeStream::Scope b(memberEnv.getStream());

                                    // Generate custom update
                                    c.generateCustomUpdate(
                                        *this, memberEnv,
                                        [this](auto &env, auto &c)
                                        {        
                                            // Write back reductions
                                            // **NOTE** this is just to handle batch reductions with batch size 1
                                            genWriteBackReductions(env, c, "id");
                                        });
                                }
                            }
                        }
                    });

                // Loop through merged custom WU update groups
                modelMerged.genMergedCustomUpdateWUGroups(
                    *this, memorySpaces, g,
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
                            buildStandardEnvironment(groupEnv);

                            // **TODO** add fields
                            const SynapseGroupInternal *sg = c.getArchetype().getSynapseGroup();
                            if (sg->getMatrixType() & SynapseMatrixWeight::KERNEL) {
                                genKernelIteration(groupEnv, c, c.getArchetype().getSynapseGroup()->getKernelSize().size(), 
                                                   [&c, this](EnvironmentExternalBase &env)
                                                   {
                                                       // Call custom update handler
                                                       c.generateCustomUpdate(*this, env, [](auto&, auto&){});

                                                       // Write back reductions
                                                       genWriteBackReductions(env, c, "id_syn");
                                                   });
                            }
                            else {
                                // Loop through presynaptic neurons
                                groupEnv.print("for(unsigned int i = 0; i < $(num_pre); i++)");
                                {
                                    // If this synapse group has sparse connectivity, loop through length of this row
                                    CodeStream::Scope b(groupEnv.getStream());
                                    if (sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                        groupEnv.print("for(unsigned int s = 0; s < $(_row_length)[i]; s++)");
                                    }
                                    // Otherwise, if it's dense, loop through each postsynaptic neuron
                                    else if (sg->getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                                        groupEnv.print("for (unsigned int j = 0; j < $(num_post); j++)");
                                    }
                                    else {
                                        throw std::runtime_error("Only DENSE and SPARSE format connectivity can be used for custom updates");
                                    }
                                    {
                                        CodeStream::Scope b(groupEnv.getStream());

                                        // Add presynaptic index to substitutions
                                        EnvironmentGroupMergedField<CustomUpdateWUGroupMerged> synEnv(groupEnv, c);
                                        synEnv.add(Type::Uint32.addConst(), "id_pre", "i");
                                        
                                        // If connectivity is sparse
                                        if (sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                            // Add initialisers to calculate synaptic index and thus lookup postsynaptic index
                                            const size_t idSynInit = synEnv.addInitialiser("const unsigned int idSyn = (i * $(_row_stride)) + s;");
                                            const size_t jInit = synEnv.addInitialiser("const unsigned int j = $(_ind)[idSyn];");

                                            // Add substitutions
                                            synEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn", {idSynInit});
                                            synEnv.add(Type::Uint32.addConst(), "id_post", "j", {jInit, idSynInit});
                                        }
                                        else {
                                            synEnv.add(Type::Uint32.addConst(), "id_post", "j");

                                            synEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn", 
                                                       {synEnv.addInitialiser("const unsigned int idSyn = (i * $(num_post)) + j;")});
                                        }

                                        // Generate custom update
                                        c.generateCustomUpdate(*this, synEnv, [](auto&, auto&){});

                                        // Write back reductions
                                        genWriteBackReductions(synEnv, c, "id_syn");
                                    }
                                }
                            }
                        }
                    });
                
                // Loop through merged custom connectivity update groups
                modelMerged.genMergedCustomConnectivityUpdateGroups(
                    *this, memorySpaces, g,
                    [this, &funcEnv](auto &c)
                    {
                        CodeStream::Scope b(funcEnv.getStream());
                        funcEnv.getStream() << "// merged custom connectivity update group " << c.getIndex() << std::endl;
                        funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                        {
                            CodeStream::Scope b(funcEnv.getStream());

                            // Get reference to group
                            funcEnv.getStream() << "const auto *group = &mergedCustomConnectivityUpdateGroup" << c.getIndex() << "[g]; " << std::endl;
                            
                            // Add host RNG functions
                            EnvironmentLibrary rngEnv(funcEnv, StandardLibrary::getHostRNGFunctions(c.getScalarType()));

                            // Create matching environment
                            EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> groupEnv(rngEnv, c);
                            buildStandardEnvironment(groupEnv);
           
                            // Loop through presynaptic neurons
                            groupEnv.print("for(unsigned int i = 0; i < $(num_pre); i++)");
                            {
                                CodeStream::Scope b(groupEnv.getStream());
                            
                                // Configure substitutions
                                groupEnv.add(Type::Uint32.addConst(), "id_pre", "i");
        
                                c.generateUpdate(*this, groupEnv, 1);
                            }
                        }
                    });
            }

            // Loop through merged custom WU transpose update groups
            {
                Timer t(funcEnv.getStream(), "customUpdate" + g + "Transpose", model.isTimingEnabled());
                // Loop through merged custom connectivity update groups
                modelMerged.genMergedCustomUpdateTransposeWUGroups(
                    *this, memorySpaces, g,
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
                            buildStandardEnvironment(groupEnv);

                            // Add field for transpose field and get its name
                            const std::string transposeVarName = c.addTransposeField(*this, groupEnv);

                            // Loop through presynaptic neurons
                            groupEnv.print("for(unsigned int i = 0; i < $(num_pre); i++)");
                            {
                                CodeStream::Scope b(groupEnv.getStream());

                                // Loop through each postsynaptic neuron
                                groupEnv.print("for (unsigned int j = 0; j < $(num_post); j++)");
                                {
                                    CodeStream::Scope b(groupEnv.getStream());
                                    EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> synEnv(groupEnv, c);

                                    // Add pre and postsynaptic indices to environment
                                    synEnv.add(Type::Uint32.addConst(), "id_pre", "i");
                                    synEnv.add(Type::Uint32.addConst(), "id_post", "j");
                                
                                    // Add conditional initialisation code to calculate synapse index
                                    synEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn", 
                                               {synEnv.addInitialiser("const unsigned int idSyn = (i * $(num_post)) + j;")});
                                
                                    // Generate custom update
                                    c.generateCustomUpdate(
                                        *this, synEnv,
                                        [&transposeVarName, this](auto &env, const auto&)
                                        {        
                                            // Update transpose variable
                                            env.printLine("$(" + transposeVarName + "_transpose)[(j * $(num_pre)) + i] = $(" + transposeVarName + ");");
                                        });
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
void Backend::genInit(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                      HostHandler preambleHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();
    if(model.getBatchSize() != 1) {
        throw std::runtime_error("The single-threaded CPU backend only supports simulations with a batch size of 1");
    }

    // Generate stream with neuron update code
    std::ostringstream initStream;
    CodeStream init(initStream);

    // Begin environment with RNG library and standard library
    EnvironmentLibrary rngEnv(init, StandardLibrary::getHostRNGFunctions(modelMerged.getModel().getPrecision()));
    EnvironmentLibrary initEnv(rngEnv, StandardLibrary::getMathsFunctions());


    initEnv.getStream() << "void initialize()";
    {
        CodeStream::Scope b(initEnv.getStream());
        EnvironmentExternal funcEnv(initEnv);
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "dt", 
                    writePreciseLiteral(modelMerged.getModel().getDT(), modelMerged.getModel().getTimePrecision()));

        Timer t(funcEnv.getStream(), "init", model.isTimingEnabled());

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Neuron groups" << std::endl;
        modelMerged.genMergedNeuronInitGroups(
            *this, memorySpaces,
            [this, &funcEnv, &modelMerged](auto &n)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged neuron init group " << n.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedNeuronInitGroup" << n.getIndex() << "[g]; " << std::endl;

                    EnvironmentGroupMergedField<NeuronInitGroupMerged> groupEnv(funcEnv, n);
                    buildStandardEnvironment(groupEnv, 1);
                    n.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Synapse groups" << std::endl;
        modelMerged.genMergedSynapseInitGroups(
            *this, memorySpaces,
            [this, &funcEnv, &modelMerged](auto &s)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged synapse init group " << s.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedSynapseInitGroup" << s.getIndex() << "[g]; " << std::endl;

                    EnvironmentGroupMergedField<SynapseInitGroupMerged> groupEnv(funcEnv, s);
                    buildStandardEnvironment(groupEnv, 1);
                    s.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom update groups" << std::endl;
        modelMerged.genMergedCustomUpdateInitGroups(
            *this, memorySpaces,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomUpdateInitGroup" << c.getIndex() << "[g]; " << std::endl;

                    EnvironmentGroupMergedField<CustomUpdateInitGroupMerged> groupEnv(funcEnv, c);
                    buildStandardEnvironment(groupEnv);
                    c.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom connectivity presynaptic update groups" << std::endl;
        modelMerged.genMergedCustomConnectivityUpdatePreInitGroups(
            *this, memorySpaces,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom connectivity presynaptic init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomConnectivityUpdatePreInitGroup" << c.getIndex() << "[g]; " << std::endl;

                    EnvironmentGroupMergedField<CustomConnectivityUpdatePreInitGroupMerged> groupEnv(funcEnv, c);
                    buildStandardEnvironment(groupEnv);
                    c.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom connectivity postsynaptic update groups" << std::endl;
        modelMerged.genMergedCustomConnectivityUpdatePostInitGroups(
            *this, memorySpaces,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom connectivity postsynaptic init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomConnectivityUpdatePostInitGroup" << c.getIndex() << "[g]; " << std::endl;
                    EnvironmentGroupMergedField<CustomConnectivityUpdatePostInitGroupMerged> groupEnv(funcEnv, c);
                    buildStandardEnvironment(groupEnv);
                    c.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom WU update groups" << std::endl;
        modelMerged.genMergedCustomWUUpdateInitGroups(
            *this, memorySpaces,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom WU update group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomWUUpdateInitGroup" << c.getIndex() << "[g]; " << std::endl;

                    EnvironmentGroupMergedField<CustomWUUpdateInitGroupMerged> groupEnv(funcEnv, c);
                    buildStandardEnvironment(groupEnv);
                    c.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Synapse sparse connectivity" << std::endl;
        modelMerged.genMergedSynapseConnectivityInitGroups(
            *this, memorySpaces,
            [this, &funcEnv, &modelMerged](auto &s)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged synapse connectivity init group " << s.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedSynapseConnectivityInitGroup" << s.getIndex() << "[g]; " << std::endl;
                    EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> groupEnv(funcEnv, s);
                    buildStandardEnvironment(groupEnv, modelMerged.getModel().getBatchSize());

                    // If matrix connectivity is ragged
                    if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        // Zero row lengths
                        groupEnv.printLine("std::fill_n($(_row_length), $(num_pre), 0);");
                    }
                    else if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        groupEnv.printLine("const size_t gpSize = ((((size_t)$(num_pre) * (size_t)$(_row_stride)) + 32 - 1) / 32);");
                        groupEnv.printLine("std::fill($(_gp), gpSize, 0);");
                    }
                    else {
                        throw std::runtime_error("Only BITMASK and SPARSE format connectivity can be generated using a connectivity initialiser");
                    }

                    // If there is row-building code in this snippet
                    const auto *snippet = s.getArchetype().getConnectivityInitialiser().getSnippet();
                    if(!snippet->getRowBuildCode().empty()) {
                        // Generate loop through source neurons
                        groupEnv.print("for (unsigned int i = 0; i < $(num_pre); i++)");

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
                        groupEnv.print("for (unsigned int j = 0; j < $(num_post); j++)");

                        // Configure substitutions
                        groupEnv.add(Type::Uint32.addConst(), "id_post", "j");
                        groupEnv.add(Type::Uint32.addConst(), "id_pre_begin", "0");
                        groupEnv.add(Type::Uint32.addConst(), "id_thread", "0");
                        groupEnv.add(Type::Uint32.addConst(), "num_threads", "1");
                    }
                    {
                        CodeStream::Scope b(groupEnv.getStream());

                        // Create new stream to generate addSynapse function which initializes all kernel variables
                        std::ostringstream addSynapseStream;
                        CodeStream addSynapse(addSynapseStream);

                        // Create block of code to add synapse
                        {
                            CodeStream::Scope b(addSynapse);

                            // Calculate index in data structure of this synapse
                            if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                if(!snippet->getRowBuildCode().empty()) {
                                    addSynapse << "const unsigned int idx = " << "($(id_pre) * $(_row_stride)) + $(_row_length)[i];" << std::endl;
                                }
                                else {
                                    addSynapse << "const unsigned int idx = " << "(($(0)) * $(_row_stride)) + $(_row_length)[$(0)];" << std::endl;
                                }
                            }

                            // If there is a kernel
                            if(!s.getArchetype().getKernelSize().empty()) {
                                EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> kernelInitEnv(groupEnv, s);

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
                                s.generateKernelInit(*this, kernelInitEnv, 1);
                            }

                            // If there is row-building code in this snippet
                            if(!snippet->getRowBuildCode().empty()) {
                                // If matrix is sparse, add function to increment row length and insert synapse into ind array
                                if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    addSynapse << "$(_ind)[idx] = $(0);" << std::endl;
                                    addSynapse << "$(_row_length)[i]++;" << std::endl;
                                }
                                // Otherwise, add function to set correct bit in bitmask
                                else {
                                    addSynapse << "const int64_t rowStartGID = i * $(_row_stride);" << std::endl;
                                    addSynapse << "setB(group->gp[(rowStartGID + ($(0))) / 32], (rowStartGID + $(0)) & 31);" << std::endl;
                                }
                            }
                            // Otherwise
                            else {
                                // If matrix is sparse, add function to increment row length and insert synapse into ind array
                                if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    addSynapse << "$(_ind)[idx] = $(id_post);" << std::endl;
                                    addSynapse << "$(_row_length)[$(0)]++;" << std::endl;
                                }
                                else {
                                    addSynapse << "const int64_t colStartGID = j;" << std::endl;
                                    addSynapse << "setB($(_gp)[(colStartGID + (($(0)) * $(_row_stride))) / 32], ((colStartGID + (($(0)) * $(_row_stride))) & 31));" << std::endl;
                                }
                            }
                        }

                        const auto addSynapseType = Type::ResolvedType::createFunction(Type::Void, std::vector<Type::ResolvedType>{1ull + s.getArchetype().getKernelSize().size(), Type::Uint32});
                        groupEnv.add(addSynapseType, "addSynapse", addSynapseStream.str());

                        // Call appropriate connectivity handler
                        if(!snippet->getRowBuildCode().empty()) {
                            s.generateSparseRowInit(*this, groupEnv);
                        }
                        else {
                            s.generateSparseColumnInit(*this, groupEnv);
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
            *this, memorySpaces,
            [this, &funcEnv, &modelMerged](auto &s)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged sparse synapse init group " << s.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedSynapseSparseInitGroup" << s.getIndex() << "[g]; " << std::endl;
                    EnvironmentGroupMergedField<SynapseSparseInitGroupMerged> groupEnv(funcEnv, s);
                    buildStandardEnvironment(groupEnv, modelMerged.getModel().getBatchSize());

                    // If postsynaptic learning is required, initially zero column lengths
                    if (!s.getArchetype().getWUModel()->getLearnPostCode().empty()) {
                        groupEnv.getStream() << "// Zero column lengths" << std::endl;
                        groupEnv.printLine("std::fill_n($(_col_length), $(num_post), 0);");
                    }

                    groupEnv.printLine("// Loop through presynaptic neurons");
                    groupEnv.print("for (unsigned int i = 0; i < $(num_pre); i++)");
                    {
                        CodeStream::Scope b(groupEnv.getStream());

                        // Generate sparse initialisation code
                        groupEnv.add(Type::Uint32.addConst(), "id_pre", "i");
                        if(s.getArchetype().isWUVarInitRequired()) {
                            groupEnv.add(Type::Uint32.addConst(), "row_len", "$(_row_length)[i]");
                            s.generateInit(*this, groupEnv, 1);
                        }

                        // If postsynaptic learning is required
                        if(!s.getArchetype().getWUModel()->getLearnPostCode().empty()) {
                            groupEnv.printLine("// Loop through synapses in corresponding matrix row");
                            groupEnv.print("for(unsigned int j = 0; j < $(_row_length)[i]; j++)");
                            {
                                CodeStream::Scope b(groupEnv.getStream());

                                // If postsynaptic learning is required, calculate column length and remapping
                                if(!s.getArchetype().getWUModel()->getLearnPostCode().empty()) {
                                    groupEnv.printLine("// Calculate index of this synapse in the row-major matrix");
                                    groupEnv.printLine("const unsigned int rowMajorIndex = (i * $(_row_stride)) + j;");
                                    groupEnv.printLine("// Using this, lookup postsynaptic target");
                                    groupEnv.printLine("const unsigned int postIndex = $(_ind)[rowMajorIndex];");
                                    groupEnv.printLine("// From this calculate index of this synapse in the column-major matrix)");
                                    groupEnv.printLine("const unsigned int colMajorIndex = (postIndex * $(_col_stride)) + $(_col_length)[postIndex];");
                                    groupEnv.printLine("// Increment column length corresponding to this postsynaptic neuron");
                                    groupEnv.printLine("$(_col_length)[postIndex]++;");
                                    groupEnv.printLine("// Add remapping entry");
                                    groupEnv.printLine("$(_remap)[colMajorIndex] = rowMajorIndex;");
                                }
                            }
                        }
                    }
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom sparse WU update groups" << std::endl;
        modelMerged.genMergedCustomWUUpdateSparseInitGroups(
            *this, memorySpaces,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom sparse WU update group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomWUUpdateSparseInitGroup" << c.getIndex() << "[g]; " << std::endl;
                    EnvironmentGroupMergedField<CustomWUUpdateSparseInitGroupMerged> groupEnv(funcEnv, c);
                    buildStandardEnvironment(groupEnv);

                    groupEnv.printLine("// Loop through presynaptic neurons");
                    groupEnv.print("for (unsigned int i = 0; i < $(num_pre); i++)");
                    {
                        CodeStream::Scope b(groupEnv.getStream());

                        // Generate initialisation code  
                        groupEnv.add(Type::Uint32.addConst(), "id_pre", "i");
                        groupEnv.add(Type::Uint32.addConst(), "row_len", "$(_row_length)[i]");
                        c.generateInit(*this, groupEnv, 1);
                    }
                }
            });
        
        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom connectivity update sparse init groups" << std::endl;
         modelMerged.genMergedCustomConnectivityUpdateSparseInitGroups(
            *this, memorySpaces,
            [this, &funcEnv, &modelMerged](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom connectivity update sparse init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const auto *group = &mergedCustomConnectivityUpdateSparseInitGroup" << c.getIndex() << "[g]; " << std::endl;
                    EnvironmentGroupMergedField<CustomConnectivityUpdateSparseInitGroupMerged> groupEnv(funcEnv, c);
                    buildStandardEnvironment(groupEnv);

                    groupEnv.printLine("// Loop through presynaptic neurons");
                    groupEnv.print("for (unsigned int i = 0; i < $(num_pre); i++)");
                    {
                        CodeStream::Scope b(groupEnv.getStream());

                        // Generate initialisation code  
                        groupEnv.add(Type::Uint32.addConst(), "id_pre", "i");
                        groupEnv.add(Type::Uint32.addConst(), "row_len", "$(_row_length)[i]");
                        c.generateInit(*this, groupEnv, 1);
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
void Backend::genRunnerPreamble(CodeStream&, const ModelSpecMerged&, const MemAlloc&) const
{
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
void Backend::genLazyVariableDynamicAllocation(CodeStream &os, 
                                               const Type::ResolvedType &type, const std::string &name, VarLocation, 
                                               const std::string &countVarName) const
{
    if (type.isPointer()) {
        os << "*$(_" << name <<  ") = new " << type.getPointer().valueType->getValue().name << "[" << countVarName << "];" << std::endl;
    }
    else {
        os << "$(_" << name <<  ") = new " << type.getValue().name << "[" << countVarName << "];" << std::endl;
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
void Backend::genLazyVariableDynamicPush(CodeStream&, 
                                         const Type::ResolvedType&, const std::string&,
                                         VarLocation, const std::string&) const
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
void Backend::genLazyVariableDynamicPull(CodeStream&, 
                                         const Type::ResolvedType&, const std::string&,
                                         VarLocation, const std::string&) const
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
    env.print("for (unsigned int j = 0; j < $(_row_length)[$(id_pre)]; j++)");
    {
        CodeStream::Scope b(env.getStream());

        EnvironmentExternal varEnv(env);
        // **TODO** 64-bit
        varEnv.add(Type::Uint32, "id_syn", "idSyn",
                   {varEnv.addInitialiser("const unsigned int idSyn = ($(id_pre) * $(_row_stride)) + j;")});
        varEnv.add(Type::Uint32, "id_post", "idPost",
                   {varEnv.addInitialiser("const unsigned int idPost = $(_ind)[$(id_syn)];")});
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
                   {varEnv.addInitialiser("const unsigned int idSyn = ($(id_pre) * $(_row_stride)) + j;")});
        varEnv.add(Type::Uint32, "id_post", "j");
        handler(varEnv);
    }
}
//--------------------------------------------------------------------------
void Backend::genKernelSynapseVariableInit(EnvironmentExternalBase &env, SynapseInitGroupMerged &sg, HandlerEnv handler) const
{
    genKernelIteration(env, sg, sg.getArchetype().getKernelSize().size(), handler);
}
//--------------------------------------------------------------------------
void Backend::genKernelCustomUpdateVariableInit(EnvironmentExternalBase &env, CustomWUUpdateInitGroupMerged &cu, HandlerEnv handler) const
{
    genKernelIteration(env, cu, cu.getArchetype().getSynapseGroup()->getKernelSize().size(), handler);
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
bool Backend::isGlobalHostRNGRequired(const ModelSpecInternal &model) const
{
    // If any neuron groups require simulation RNGs or require RNG for initialisation, return true
    // **NOTE** this takes postsynaptic model initialisation into account
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
                       return (s.second.isWUInitRNGRequired() || s.second.getConnectivityInitialiser().isHostRNGRequired());
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
                       return (Utils::isRNGRequired(c.second.getVarInitialisers())
                               || Utils::isRNGRequired(c.second.getPreVarInitialisers())
                               || Utils::isRNGRequired(c.second.getPostVarInitialisers())
                               || Utils::isRNGRequired(c.second.getRowUpdateCodeTokens())
                               || Utils::isRNGRequired(c.second.getHostUpdateCodeTokens()));
                   }))
    {
        return true;
    }
    return false;
}
//--------------------------------------------------------------------------
bool Backend::isGlobalDeviceRNGRequired(const ModelSpecInternal &) const
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
void Backend::genPresynapticUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const ModelSpecMerged &modelMerged, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_evnt";
    const auto *wu = sg.getArchetype().getWUModel();

    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) {
        const auto &connectInit = sg.getArchetype().getToeplitzConnectivityInitialiser();

        // Loop through Toeplitz matrix diagonals
        env.print("for(unsigned int j = 0; j < $(row_stride); j++)");
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
            env.print("for (unsigned int i = 0; i < $(_src_spk_cnt" + eventSuffix + ")[$(pre_delay_slot)]; i++)");
        }
        else {
            env.print("for (unsigned int i = 0; i < $(_src_spk_cnt" + eventSuffix + ")[0]; i++)");
        }
        {
            CodeStream::Scope b(env.getStream());
            /*if(!wu->getSimSupportCode().empty()) {
                os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
            }*/
            EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(env, sg);


            const std::string queueOffset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "$(pre_delay_offset) + " : "";
            groupEnv.add(Type::Uint32, "id_pre", "idPre",
                         {groupEnv.addInitialiser("const unsigned int idPre = $(_src_spk" + eventSuffix + ")[" + queueOffset + "i];")});

            // If this is a spike-like event, insert threshold check for this presynaptic neuron
            if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                groupEnv.getStream() << "if(";

                // Generate weight update threshold condition
                sg.generateSpikeEventThreshold(*this, groupEnv, 1);

                groupEnv.getStream() << ")";
                groupEnv.getStream() << CodeStream::OB(10);
            }

            // If connectivity is sparse
            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                groupEnv.printLine("const unsigned int npost = $(_row_length)[$(id_pre)];");
                groupEnv.getStream() << "for (unsigned int j = 0; j < npost; j++)";
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(groupEnv, sg);

                    // **TODO** 64-bit id_syn
                    synEnv.add(Type::Uint32, "id_syn", "idSyn",
                               {synEnv.addInitialiser("const unsigned int idSyn = ($(id_pre) * $(_row_stride)) + j;")});
                    synEnv.add(Type::Uint32, "id_post", "idPost",
                               {synEnv.addInitialiser("const unsigned int idPost = $(_ind)[$(id_syn)];")});
                    
                    // Add correct functions for apply synaptic input
                    synEnv.add(Type::AddToPostDenDelay, "addToPostDelay", "$(_den_delay)[" + sg.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "] += $(0)");
                    synEnv.add(Type::AddToPost, "addToPost", "$(_out_post)[" + sg.getPostISynIndex(1, "$(id_post)") + "] += $(0)");
                    synEnv.add(Type::AddToPre, "addToPre", "$(_out_pre)[" + sg.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");

                    if(trueSpike) {
                        sg.generateSpikeUpdate(*this, synEnv, 1);
                    }
                    else {
                        sg.generateSpikeEventUpdate(*this, synEnv, 1);
                    }
                }
            }
            else if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) {
                throw std::runtime_error("The single-threaded CPU backend does not support procedural connectivity.");
            }
            else if(getPreferences().enableBitmaskOptimisations && (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK)) {
                // Determine the number of words in each row
                groupEnv.printLine("const unsigned int rowWords = (($(num_post) + 32 - 1) / 32);");
                groupEnv.getStream() << "for(unsigned int w = 0; w < rowWords; w++)";
                {
                    CodeStream::Scope b(groupEnv.getStream());

                    // Read row word
                    groupEnv.printLine("uint32_t connectivityWord = $(_gp)[(ipre * rowWords) + w];");

                    // Set ipost to first synapse in connectivity word
                    groupEnv.getStream() << "unsigned int ipost = w * 32;" << std::endl;
                    groupEnv.add(Type::Uint32, "id_post", "ipost");
                    
                    // Add correct functions for apply synaptic input
                    groupEnv.add(Type::AddToPostDenDelay, "addToPostDelay", "$(_den_delay)[" + sg.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "] += $(0)");
                    groupEnv.add(Type::AddToPost, "addToPost", "$(_out_post)[" + sg.getPostISynIndex(1, "$(id_post)") + "] += $(0)");
                    groupEnv.add(Type::AddToPre, "addToPre", "$(_out_pre)[" + sg.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");

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
                        groupEnv.print("if(ipost < $(num_post))");
                        {
                            CodeStream::Scope b(env.getStream());
                            if(trueSpike) {
                                sg.generateSpikeUpdate(*this, groupEnv, 1);
                            }
                            else {
                                sg.generateSpikeEventUpdate(*this, groupEnv, 1);
                            }
                        }

                        // Increment ipost to take into account fact the next CLZ will go from bit AFTER synapse
                        groupEnv.getStream() << "ipost++;" << std::endl;
                    }
                }
            }
            // Otherwise (DENSE or BITMASK)
            else {
                groupEnv.print("for (unsigned int ipost = 0; ipost < $(num_post); ipost++)");
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(groupEnv, sg);
                    synEnv.add(Type::Uint32, "id_post", "ipost");
                    
                    // Add correct functions for apply synaptic input
                    synEnv.add(Type::AddToPostDenDelay, "addToPostDelay", "$(_den_delay)[" + sg.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "] += $(0)");
                    synEnv.add(Type::AddToPost, "addToPost", "$(_out_post)[" + sg.getPostISynIndex(1, "$(id_post)") + "] += $(0)");
                    synEnv.add(Type::AddToPre, "addToPre", "$(_out_pre)[" + sg.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");

                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // **TODO** 64-bit index
                        synEnv.printLine("const uint64_t gid = ($(id_pre) * $(num_post)) + $(id_post);");

                        synEnv.getStream() << "if (B(" << synEnv["_gp"] << "[gid / 32], gid & 31))" << CodeStream::OB(20);
                    }
                    else {
                        synEnv.add(Type::Uint32, "id_syn", "idSyn",
                                   {synEnv.addInitialiser("const unsigned int idSyn = ($(id_pre) * $(num_post)) + $(id_post);")});
                    }

                   
                    if(trueSpike) {
                        sg.generateSpikeUpdate(*this, synEnv, 1);
                    }
                    else {
                        sg.generateSpikeEventUpdate(*this, synEnv, 1);
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
void Backend::genEmitSpike(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng, bool trueSpike, bool recordingEnabled) const
{
    // Determine if delay is required and thus, at what offset we should write into the spike queue
    const bool spikeDelayRequired = trueSpike ? (ng.getArchetype().isDelayRequired() && ng.getArchetype().isTrueSpikeRequired()) : ng.getArchetype().isDelayRequired();
    const std::string spikeQueueOffset = spikeDelayRequired ? "$(_write_delay_offset) + " : "";

    const std::string suffix = trueSpike ? "" : "_evnt";
    env.print("$(_spk" + suffix + ")[" + spikeQueueOffset + "$(_spk_cnt" + suffix + ")");
    if(spikeDelayRequired) { // WITH DELAY
        env.print("[*$(_spk_que_ptr)]++]");
    }
    else { // NO DELAY
        env.getStream() << "[0]++]";
    }
    env.printLine(" = $(id);");

    // Reset spike and spike-like-event times
    const std::string queueOffset = ng.getArchetype().isDelayRequired() ? "$(_write_delay_offset) + " : "";
    if(trueSpike && ng.getArchetype().isSpikeTimeRequired()) {
        env.printLine("$(_spk_time)[" + queueOffset + "$(id)] = $(t);");
    }
    else if(!trueSpike && ng.getArchetype().isSpikeEventTimeRequired()) {
        env.printLine("$(_spk_evnt_time)[" + queueOffset + "$(id)] = $(t);");
    }
    
    // If recording is enabled
    if(recordingEnabled) {
        env.printLine("$(_record_spk" + suffix + ")[(recordingTimestep * numRecordingWords) + ($(id) / 32)] |= (1 << ($(id) % 32));");
    }
}
//--------------------------------------------------------------------------
void Backend::genWriteBackReductions(EnvironmentExternalBase &env, CustomUpdateGroupMerged &cg, const std::string &idxName) const
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
void Backend::genWriteBackReductions(EnvironmentExternalBase &env, CustomUpdateWUGroupMerged &cg, const std::string &idxName) const
{
    genWriteBackReductions(env, cg, idxName,
                           [&cg](const Models::WUVarReference &varRef, const std::string &index)
                           {
                               return cg.getVarRefIndex(getVarAccessDuplication(varRef.getVar().access),
                                                        index);
                           });
}
}   // namespace GeNN::CodeGenerator::SingleThreadedCPU

#include "backend.h"

// Standard C includes
#include <cstdlib>

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
const EnvironmentLibrary::Library backendFunctions = {
    {"clz", {Type::ResolvedType::createFunction(Type::Int32, {Type::Uint32}), "gennCLZ($(0))"}},
    {"atomic_or", {Type::ResolvedType::createFunction(Type::Void, {Type::Uint32.createPointer(), Type::Uint32}), "(*($(0)) |= ($(1)))"}}
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

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Array
//--------------------------------------------------------------------------
class Array : public Runtime::ArrayBase
{
public:
    Array(const Type::ResolvedType &type, size_t count, 
          VarLocation location, bool uninitialized)
    :   ArrayBase(type, count, location, uninitialized)
    {
        if(count > 0) {
            allocate(count);
        }
    }
    
    virtual ~Array()
    {
        if(getCount() > 0) {
            free();
        }
    }
    
    //------------------------------------------------------------------------
    // ArrayBase virtuals
    //------------------------------------------------------------------------
    //! Allocate array
    virtual void allocate(size_t count) final
    {
        // Malloc host pointer
        setCount(count);
        setHostPointer(new std::byte[getSizeBytes()]);
    }

    //! Free array
    virtual void free() final
    {
        delete [] getHostPointer();
        setHostPointer(nullptr);
        setCount(0);
    }


    //! Copy entire array to device
    virtual void pushToDevice() final
    {
    }

    //! Copy entire array from device
    virtual void pullFromDevice() final
    {
    }

    //! Copy a 1D slice of elements to device 
    /*! \param offset   Offset in elements to start copying from
        \param count    Number of elements to copy*/
    virtual void pushSlice1DToDevice(size_t, size_t) final
    {
    }

    //! Copy a 1D slice of elements from device 
    /*! \param offset   Offset in elements to start copying from
        \param count    Number of elements to copy*/
    virtual void pullSlice1DFromDevice(size_t, size_t) final
    {
    }
    
    //! Memset the host pointer
    virtual void memsetDeviceObject(int) final
    {
        throw std::runtime_error("Single-threaded CPU arrays have no device objects");
    }

    //! Serialise backend-specific device object to bytes
    virtual void serialiseDeviceObject(std::vector<std::byte>&, bool) const final
    {
        throw std::runtime_error("Single-threaded CPU arrays have no device objects");
    }

    //! Serialise backend-specific host object to bytes
    virtual void serialiseHostObject(std::vector<std::byte>&, bool) const
    {
        throw std::runtime_error("Single-threaded CPU arrays have no host objects");
    }
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
// CodeGenerator::SingleThreadedCPU::Backend
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
    EnvironmentLibrary backendEnv(neuronUpdate, backendFunctions);
    EnvironmentLibrary neuronUpdateEnv(backendEnv, StandardLibrary::getMathsFunctions());

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
                    Type::writeNumeric(modelMerged.getModel().getDT(), modelMerged.getModel().getTimePrecision()));
        
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
                        groupEnv.printLine("const unsigned int lastTimestepDelaySlot = *$(_spk_que_ptr);");
                        groupEnv.printLine("const unsigned int lastTimestepDelayOffset = lastTimestepDelaySlot * $(num_neurons);");
                    }

                    // Generate code to update previous spike times
                    if(n.getArchetype().isPrevSpikeTimeRequired()) {
                        n.generateSpikes(
                            groupEnv,
                            [&n, this](EnvironmentExternalBase &env)
                            {
                                genPrevEventTimeUpdate(env, n, true);
                            });
                    }
                    
                    // Generate code to update previous spike-event times
                    if(n.getArchetype().isPrevSpikeEventTimeRequired()) {
                        n.generateSpikeEvents(
                            groupEnv,
                            [&n, this](EnvironmentExternalBase &env)
                            {
                                genPrevEventTimeUpdate(env, n, false);
                            });
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
                    n.genSpikeQueueUpdate(groupEnv, 1);
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
                            n.generateSpikeEvents(
                                groupEnv,
                                [](EnvironmentExternalBase &env, NeuronUpdateGroupMerged::SynSpikeEvent&)
                                {
                                    env.printLine("std::fill_n(&$(_record_spk_event)[recordingTimestep * numRecordingWords], numRecordingWords, 0);");
                                });
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
                        n.generateNeuronUpdate(
                            *this, rngEnv, 1,
                            // Emit true spikes
                            [&n, this](EnvironmentExternalBase &env)
                            {
                                // Insert code to update WU vars
                                n.generateWUVarUpdate(env, 1);

                                // If recording is enabled
                                if(n.getArchetype().isSpikeRecordingEnabled()) {
                                    env.printLine("$(_record_spk)[(recordingTimestep * numRecordingWords) + ($(id) / 32)] |= (1 << ($(id) % 32));");
                                }

                                // Update event time
                                if(n.getArchetype().isSpikeTimeRequired()) {
                                    const std::string queueOffset = n.getArchetype().isDelayRequired() ? "$(_write_delay_offset) + " : "";
                                    env.printLine("$(_st)[" + queueOffset + "$(id)] = $(t);");
                                }

                                // Generate spike dagta structure updates
                                n.generateSpikes(
                                    env,
                                    [&n, this](EnvironmentExternalBase &env)
                                    {
                                        genEmitEvent(env, n, true);
                                    });
                               
                            },
                            // Emit spike-like events
                            [&n, this](EnvironmentExternalBase &env, NeuronUpdateGroupMerged::SynSpikeEvent &sg)
                            {
                                sg.generate(
                                    env, n,
                                    [&n, this](EnvironmentExternalBase &env, NeuronUpdateGroupMerged::SynSpikeEvent&)
                                    {
                                        genEmitEvent(env, n, false);

                                        if(n.getArchetype().isSpikeEventTimeRequired()) {
                                            const std::string queueOffset = n.getArchetype().isDelayRequired() ? "$(_write_delay_offset) + " : "";
                                            env.printLine("$(_set)[" + queueOffset + "$(id)] = $(t);");
                                        }

                                        // If recording is enabled
                                        if(n.getArchetype().isSpikeEventRecordingEnabled()) {
                                            env.printLine("$(_record_spk_event)[(recordingTimestep * numRecordingWords) + ($(id) / 32)] |= (1 << ($(id) % 32));");
                                        }
                                    });
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
    modelMerged.genMergedNeuronUpdateGroupHostStructArrayPush(os, *this);
    modelMerged.genMergedNeuronSpikeQueueUpdateHostStructArrayPush(os, *this);
    modelMerged.genMergedNeuronPrevSpikeTimeUpdateHostStructArrayPush(os, *this);

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
    EnvironmentLibrary backendEnv(synapseUpdate, backendFunctions);
    EnvironmentLibrary synapseUpdateEnv(backendEnv, StandardLibrary::getMathsFunctions());

    synapseUpdateEnv.getStream() << "void updateSynapses(" << modelMerged.getModel().getTimePrecision().getName() << " t)";
    {
        CodeStream::Scope b(synapseUpdateEnv.getStream());

        EnvironmentExternal funcEnv(synapseUpdateEnv);
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t");
        funcEnv.add(Type::Uint32.addConst(), "batch", "0");
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "dt", 
                    Type::writeNumeric(modelMerged.getModel().getDT(), modelMerged.getModel().getTimePrecision()));

        // Dendritic delay update
        modelMerged.genMergedSynapseDendriticDelayUpdateGroups(
            *this, memorySpaces,
            [&funcEnv, &modelMerged, this](auto &sg)
            {
                // Loop through groups
                funcEnv.getStream() << "// merged synapse dendritic delay update group " << sg.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << sg.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Use this to get reference to merged group structure
                    funcEnv.getStream() << "const auto *group = &mergedSynapseDendriticDelayUpdateGroup" << sg.getIndex() << "[g]; " << std::endl;
                    EnvironmentGroupMergedField<SynapseDendriticDelayUpdateGroupMerged> groupEnv(funcEnv, sg);
                    buildStandardEnvironment(groupEnv, 1);
                    sg.generateSynapseUpdate(groupEnv);
                }
            });

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

                                const auto indexType = getSynapseIndexType(s);
                                const auto indexTypeName = indexType.getName();
                                if (s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    // Add initialiser strings to calculate synaptic and presynaptic index
                                    const size_t idSynInit = synEnv.addInitialiser("const " + indexTypeName + " idSyn = ((" + indexTypeName + ")i * $(_row_stride)) + s;");
                                    const size_t idPostInit = synEnv.addInitialiser("const unsigned int idPost = $(_ind)[$(id_syn)];");

                                    synEnv.add(indexType.addConst(), "id_syn", "idSyn", {idSynInit});
                                    synEnv.add(Type::Uint32.addConst(), "id_post", "idPost", {idPostInit, idSynInit});
                                }
                                else {
                                    // Add postsynaptic index to substitutions
                                    synEnv.add(Type::Uint32.addConst(), "id_post", "j");

                                    // Add initialiser to calculate synaptic index
                                    synEnv.add(indexType.addConst(), "id_syn", "idSyn", 
                                               {synEnv.addInitialiser("const " + indexTypeName + " idSyn = ((" + indexTypeName + ")i * $(num_post)) + j;")});
                                }

                                // Add correct functions for apply synaptic input
                                synEnv.add(Type::AddToPostDenDelay, "addToPostDelay", "$(_den_delay)[" + s.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "] += $(0)");
                                synEnv.add(Type::AddToPost, "addToPost", "$(_out_post)[" + s.getPostISynIndex(1, "$(id_post)") + "] += $(0)");
                                synEnv.add(Type::AddToPre, "addToPre", "$(_out_pre)[" + s.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");
                                
                                // Call synapse dynamics handler
                                s.generateSynapseUpdate(synEnv, 1, modelMerged.getModel().getDT());
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
                        if (s.getArchetype().isPreSpikeEventRequired()) {
                            genPresynapticUpdate(groupEnv, s, modelMerged.getModel().getDT(), false);
                        }

                        // generate the code for processing true spike events
                        if (s.getArchetype().isPreSpikeRequired()) {
                            genPresynapticUpdate(groupEnv, s, modelMerged.getModel().getDT(), true);
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

                        // generate the code for processing spike-like events
                        if (s.getArchetype().isPostSpikeEventRequired()) {
                            genPostsynapticUpdate(groupEnv, s, modelMerged.getModel().getDT(), false);
                        }

                        // generate the code for processing true spike events
                        if (s.getArchetype().isPostSpikeRequired()) {
                            genPostsynapticUpdate(groupEnv, s, modelMerged.getModel().getDT(), true);
                        }
                        groupEnv.getStream() << std::endl;
                    }
                });
        }
    }

    // Generate struct definitions
    modelMerged.genMergedSynapseDendriticDelayUpdateStructs(os, *this);
    modelMerged.genMergedPresynapticUpdateGroupStructs(os, *this);
    modelMerged.genMergedPostsynapticUpdateGroupStructs(os, *this);
    modelMerged.genMergedSynapseDynamicsGroupStructs(os, *this);

    // Generate arrays of merged structs and functions to set them
    modelMerged.genMergedSynapseDendriticDelayUpdateHostStructArrayPush(os, *this);
    modelMerged.genMergedPresynapticUpdateGroupHostStructArrayPush(os, *this);
    modelMerged.genMergedPostsynapticUpdateGroupHostStructArrayPush(os, *this);
    modelMerged.genMergedSynapseDynamicsGroupHostStructArrayPush(os, *this);

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
    EnvironmentLibrary backendEnv(customUpdate, backendFunctions);
    EnvironmentLibrary customUpdateEnv(backendEnv, StandardLibrary::getMathsFunctions());

    // Loop through custom update groups
    for(const auto &g : customUpdateGroups) {
        customUpdateEnv.getStream() << "void update" << g << "(unsigned long long timestep)";
        {
            CodeStream::Scope b(customUpdateEnv.getStream());

             EnvironmentExternal funcEnv(customUpdateEnv);
             funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t",
                         {funcEnv.addInitialiser("const " + model.getTimePrecision().getName() + " t = timestep * " + Type::writeNumeric(model.getDT(), model.getTimePrecision()) + ";")});
             funcEnv.add(Type::Uint32.addConst(), "batch", "0");
             funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "dt", 
                    Type::writeNumeric(modelMerged.getModel().getDT(), modelMerged.getModel().getTimePrecision()));

            // Loop through host update groups and generate code for those in this custom update group
            modelMerged.genMergedCustomConnectivityHostUpdateGroups(
                *this, memorySpaces, g, 
                [this, &funcEnv, &modelMerged](auto &c)
                {
                    c.generateUpdate(*this, funcEnv);
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
                            buildSizeEnvironment(groupEnv);
                            buildStandardEnvironment(groupEnv, 1);

                            if (c.getArchetype().isNeuronReduction()) {
                                // Initialise reduction targets
                                // **TODO** these should be provided with some sort of caching mechanism
                                const auto reductionTargets = genInitReductionTargets(groupEnv.getStream(), c, 1);

                                // Loop through group members
                                EnvironmentGroupMergedField<CustomUpdateGroupMerged> memberEnv(groupEnv, c);
                                if (c.getArchetype().getDims() & VarAccessDim::ELEMENT) {
                                    memberEnv.print("for(unsigned int i = 0; i < $(num_neurons); i++)");
                                    memberEnv.add(Type::Uint32.addConst(), "id", "i");
                                }
                                else {
                                    memberEnv.add(Type::Uint32.addConst(), "id", "0");
                                }
                                {
                                    CodeStream::Scope b(memberEnv.getStream());
                                    c.generateCustomUpdate(memberEnv, 1,
                                                           [&reductionTargets, this](auto &env, auto&)
                                                           {        
                                                               // Loop through reduction targets and generate reduction
                                                               // **TODO** reduction should be automatically implemented by transpiler 
                                                               for (const auto &r : reductionTargets) {
                                                                   env.printLine(getReductionOperation("_lr" + r.name,  "$(" + r.name + ")", r.access, r.type) + ";");
                                                               }
                                                           });
                                }

                                // Write back reductions
                                for (const auto &r : reductionTargets) {
                                    memberEnv.getStream() << "group->" << r.name << "[" << r.index << "] = _lr" << r.name << ";" << std::endl;
                                }
                            }
                            else {
                                // Loop through group members
                                EnvironmentGroupMergedField<CustomUpdateGroupMerged> memberEnv(groupEnv, c);
                                if (c.getArchetype().getDims() & VarAccessDim::ELEMENT) {
                                    memberEnv.print("for(unsigned int i = 0; i < $(num_neurons); i++)");
                                    memberEnv.add(Type::Uint32.addConst(), "id", "i");
                                }
                                else {
                                    memberEnv.add(Type::Uint32.addConst(), "id", "0");
                                }
                                {
                                    CodeStream::Scope b(memberEnv.getStream());

                                    // Generate custom update
                                    c.generateCustomUpdate(memberEnv, 1,
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
                            buildSizeEnvironment(groupEnv);
                            buildStandardEnvironment(groupEnv, 1);

                            // **TODO** add fields
                            const SynapseGroupInternal *sg = c.getArchetype().getSynapseGroup();
                            if (sg->getMatrixType() & SynapseMatrixWeight::KERNEL) {
                                genKernelIteration(groupEnv, c, c.getArchetype().getSynapseGroup()->getKernelSize().size(), 
                                                   [&c, this](EnvironmentExternalBase &env)
                                                   {
                                                       // Call custom update handler
                                                       c.generateCustomUpdate(env, 1,
                                                                              [this](auto &env, CustomUpdateWUGroupMergedBase &c)
                                                                              {        
                                                                                  // Write back reductions
                                                                                  // **NOTE** this is just to handle batch reductions with batch size 1
                                                                                  genWriteBackReductions(env, c, "id_syn");
                                                                              });
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
                                        c.generateCustomUpdate(synEnv, 1,
                                                               [this](auto &env, auto &c)
                                                               {        
                                                                   // Write back reductions
                                                                   // **NOTE** this is just to handle batch reductions with batch size 1
                                                                   genWriteBackReductions(env, c, "id_syn");
                                                               });
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

            // Loop through merged custom connectivity remap update groups
            {
                Timer t(funcEnv.getStream(), "customUpdate" + g + "Remap", model.isTimingEnabled());
                modelMerged.genMergedCustomConnectivityRemapUpdateGroups(
                    *this, memorySpaces, g,
                    [this, &funcEnv](auto &c)
                    {
                        CodeStream::Scope b(funcEnv.getStream());
                        funcEnv.getStream() << "// merged custom connectivity remap update group " << c.getIndex() << std::endl;
                        funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                        {
                            CodeStream::Scope b(funcEnv.getStream());

                            // Get reference to group
                            funcEnv.getStream() << "const auto *group = &mergedCustomConnectivityRemapUpdateGroup" << c.getIndex() << "[g]; " << std::endl;
                     
                            // Create matching environment
                            EnvironmentGroupMergedField<CustomConnectivityRemapUpdateGroupMerged> groupEnv(funcEnv, c);
                            buildStandardEnvironment(groupEnv);
                            
                            groupEnv.printLine("// Loop through presynaptic neurons");
                            groupEnv.print("for (unsigned int i = 0; i < $(num_pre); i++)");
                            {
                                CodeStream::Scope b(groupEnv.getStream());

                                genRemap(groupEnv);
                            }
                        }
                    });
            }

            // Loop through merged custom WU transpose update groups
            {
                Timer t(funcEnv.getStream(), "customUpdate" + g + "Transpose", model.isTimingEnabled());
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
                            buildSizeEnvironment(groupEnv);
                            buildStandardEnvironment(groupEnv, 1);

                            // Add field for transpose field and get its name
                            const std::string transposeVarName = c.addTransposeField(groupEnv);

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
                                        synEnv, 1,
                                        [&transposeVarName](auto &env, const auto&)
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
    modelMerged.genMergedCustomConnectivityRemapUpdateStructs(os, *this);
    modelMerged.genMergedCustomConnectivityHostUpdateStructs(os, *this);

    // Generate arrays of merged structs and functions to set them
    modelMerged.genMergedCustomUpdateHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomUpdateWUHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomUpdateTransposeWUHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomConnectivityUpdateHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomConnectivityRemapUpdateHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomConnectivityHostUpdateStructArrayPush(os, *this);

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
    EnvironmentLibrary backendEnv(rngEnv, backendFunctions);
    EnvironmentLibrary initEnv(backendEnv, StandardLibrary::getMathsFunctions());

    initEnv.getStream() << "void initialize()";
    {
        CodeStream::Scope b(initEnv.getStream());
        EnvironmentExternal funcEnv(initEnv);
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "dt", 
                    Type::writeNumeric(modelMerged.getModel().getDT(), modelMerged.getModel().getTimePrecision()));

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
                    buildStandardEnvironment(groupEnv, 1);
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
                    buildStandardEnvironment(groupEnv, 1);
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

                    // If matrix connectivity is neither sparse or bitmask, give error
                    if(!(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE)
                       && !(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK)) 
                    {
                        throw std::runtime_error("Only BITMASK and SPARSE format connectivity can be generated using a connectivity initialiser");
                    }

                    // If there is row-building code in this snippet
                    const auto &connectInit = s.getArchetype().getSparseConnectivityInitialiser();
                    if(!Utils::areTokensEmpty(connectInit.getRowBuildCodeTokens())) {
                        // Generate loop through source neurons
                        groupEnv.print("for (unsigned int i = 0; i < $(num_pre); i++)");

                        // Configure substitutions
                        groupEnv.add(Type::Uint32.addConst(), "id_pre", "i");
                        groupEnv.add(Type::Uint32.addConst(), "id_post_begin", "0");
                        groupEnv.add(Type::Uint32.addConst(), "id_thread", "0");
                        groupEnv.add(Type::Uint32.addConst(), "num_threads", "1");
                    }
                    // Otherwise
                    else {
                        assert(!Utils::areTokensEmpty(connectInit.getColBuildCodeTokens()));

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

                        // Create environment for generating add synapsecode into seperate CodeStream
                        std::ostringstream addSynapseStream;
                        CodeStream addSynapse(addSynapseStream);
                        {
                            CodeStream::Scope b(addSynapse);
                            EnvironmentExternal addSynapseEnv(groupEnv, addSynapse);

                            // Get postsynaptic/presynaptic index from first addSynapse parameter
                            // **YUCK** we need to do this in an initialiser so the $(0) doesn't get confused with those used in AddToXXXX
                            if(!Utils::areTokensEmpty(connectInit.getRowBuildCodeTokens())) {
                                addSynapseEnv.add(Type::Uint32.addConst(), "id_post", "idPost",
                                                  {addSynapseEnv.addInitialiser("const unsigned int idPost = $(0);")});
                            }
                            else {
                                addSynapseEnv.add(Type::Uint32.addConst(), "id_pre", "idPre",
                                                  {addSynapseEnv.addInitialiser("const unsigned int idPre = $(0);")});
                            }

                            // If matrix is sparse
                            if(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                // Calculate index of new synapse
                                addSynapseEnv.add(Type::Uint32.addConst(), "id_syn", "idSyn",
                                                {addSynapseEnv.addInitialiser("const unsigned int idSyn = ($(id_pre) * $(_row_stride)) + $(_row_length)[$(id_pre)];")});

                                // If there is a kernel
                                if(!s.getArchetype().getKernelSize().empty()) {
                                    // Create new environment
                                    EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> kernelInitEnv(addSynapseEnv, s);

                                    // Replace kernel indices with the subsequent 'function' parameters
                                    // **YUCK** these also need doing in initialisers so the $(1) doesn't get confused with those used in addToPostDelay
                                    for(size_t i = 0; i < s.getArchetype().getKernelSize().size(); i++) {
                                        const std::string iStr = std::to_string(i);
                                        kernelInitEnv.add(Type::Uint32.addConst(), "id_kernel_" + iStr, "idKernel" + iStr,
                                                        {kernelInitEnv.addInitialiser("const unsigned int idKernel" + iStr + " = $(" + std::to_string(i + 1) + ");")});
                                    }

                                    // Call handler to initialize variables
                                    s.generateKernelInit(kernelInitEnv, 1);
                                }

                                // Add synapse to data structure
                                addSynapseEnv.printLine("$(_ind)[$(id_syn)] = $(id_post);");
                                addSynapseEnv.printLine("$(_row_length)[$(id_pre)]++;");
                            }
                            // Otherwise, if it's bitmask
                            else {
                                assert(s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK);
                                assert(s.getArchetype().getKernelSize().empty()) ;

                                 // If there is row-building code in this snippet
                                 // **THINK** why is this logic so convoluted?
                                if(!Utils::areTokensEmpty(connectInit.getRowBuildCodeTokens())) {
                                    addSynapseEnv.printLine("const int64_t rowStartGID = $(id_pre) * $(_row_stride);");
                                    addSynapseEnv.printLine("$(_gp)[(rowStartGID + $(id_post)) / 32] |= (0x80000000 >> ((rowStartGID + $(id_post)) & 31));");
                                }
                                // Otherwise
                                else {
                                    addSynapseEnv.printLine("const int64_t colStartGID = $(id_post);");
                                    addSynapseEnv.printLine("$(_gp)[(colStartGID + ($(id_pre) * $(_row_stride))) / 32] |= (0x80000000 >> ((colStartGID + ($(id_pre) * $(_row_stride))) & 31));");
                                }
                            }
                        }
                        

                        const auto addSynapseType = Type::ResolvedType::createFunction(Type::Void, std::vector<Type::ResolvedType>{1ull + s.getArchetype().getKernelSize().size(), Type::Uint32});
                        groupEnv.add(addSynapseType, "addSynapse", addSynapseStream.str());

                        // Call appropriate connectivity handler
                        if(!Utils::areTokensEmpty(connectInit.getRowBuildCodeTokens())) {
                            s.generateSparseRowInit(groupEnv);
                        }
                        else {
                            s.generateSparseColumnInit(groupEnv);
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
                        if(s.getArchetype().isPostSpikeRequired() || s.getArchetype().isPostSpikeEventRequired()) {
                            genRemap(groupEnv);
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
                    buildStandardEnvironment(groupEnv, 1);

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
    modelMerged.genMergedNeuronInitGroupHostStructArrayPush(os, *this);
    modelMerged.genMergedSynapseInitGroupHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomUpdateInitGroupHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomWUUpdateInitGroupHostStructArrayPush(os, *this);
    modelMerged.genMergedSynapseConnectivityInitGroupHostStructArrayPush(os, *this);
    modelMerged.genMergedSynapseSparseInitGroupHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomWUUpdateSparseInitGroupHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomConnectivityUpdatePreInitHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomConnectivityUpdatePostInitHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomConnectivityUpdateSparseInitHostStructArrayPush(os, *this);
    
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
    else if(sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
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

    // CUDA and OpenCL both provide generic min and max functions 
    // to match this, bring std::min and std::max into global namespace
    os << "using std::min;" << std::endl;
    os << "using std::max;" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genRunnerPreamble(CodeStream &, const ModelSpecMerged &) const
{
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream&, const ModelSpecMerged&) const
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
std::unique_ptr<GeNN::Runtime::StateBase> Backend::createState(const Runtime::Runtime&) const
{
    return std::make_unique<State>();
}
//--------------------------------------------------------------------------
std::unique_ptr<Runtime::ArrayBase> Backend::createArray(const Type::ResolvedType &type, size_t count, 
                                                         VarLocation location, bool uninitialized) const
{
    return std::make_unique<Array>(type, count, location, uninitialized);
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
void Backend::genLazyVariableDynamicPush(CodeStream&, 
                                         const Type::ResolvedType&, const std::string&,
                                         VarLocation, const std::string&) const
{
}
//--------------------------------------------------------------------------
void Backend::genLazyVariableDynamicPull(CodeStream&, 
                                         const Type::ResolvedType&, const std::string&,
                                         VarLocation, const std::string&) const
{
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
    env.print("for (unsigned int j = 0; j < $(num_post); j++)");
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
void Backend::genRemap(EnvironmentExternalBase &env) const
{
    env.printLine("// Loop through synapses in corresponding matrix row");
    env.print("for(unsigned int j = 0; j < $(_row_length)[i]; j++)");
    {
        CodeStream::Scope b(env.getStream());

        // Calculate column length and remapping
        env.printLine("// Calculate index of this synapse in the row-major matrix");
        env.printLine("const unsigned int rowMajorIndex = (i * $(_row_stride)) + j;");

        env.printLine("// Using this, lookup postsynaptic target");
        env.printLine("const unsigned int postIndex = $(_ind)[rowMajorIndex];");

        env.printLine("// From this calculate index of this synapse in the column-major matrix)");
        env.printLine("const unsigned int colMajorIndex = (postIndex * $(_col_stride)) + $(_col_length)[postIndex];");

        env.printLine("// Increment column length corresponding to this postsynaptic neuron");
        env.printLine("$(_col_length)[postIndex]++;");

        env.printLine("// Add remapping entry");
        env.printLine("$(_remap)[colMajorIndex] = rowMajorIndex;");
    }
}
//--------------------------------------------------------------------------
void Backend::genGlobalDeviceRNG(CodeStream&, CodeStream&, CodeStream&, CodeStream&) const
{
    assert(false);
}
//--------------------------------------------------------------------------
void Backend::genTimer(CodeStream &, CodeStream &, CodeStream &, CodeStream &, CodeStream &, const std::string &, bool) const
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
                       return (s.second.isWUInitRNGRequired() || s.second.getSparseConnectivityInitialiser().isHostRNGRequired());
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
void Backend::genPresynapticUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                   double dt, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_event";

    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) {
        // Create environment for generating presynaptic update code into seperate CodeStream
        std::ostringstream preUpdateStream;
        CodeStream preUpdate(preUpdateStream);
        {
            CodeStream::Scope b(preUpdate);
            EnvironmentExternal preUpdateEnv(env, preUpdate);
            preUpdateEnv.add(Type::Uint32.addConst(), "id_pre", "ipre");

            // Replace $(id_post) with first 'function' parameter as simulation code is
            // going to be, in turn, substituted into Toeplitz connectivity generation code
            // **YUCK** we need to do this in an initialiser so the $(0) doesn't get confused with those used in AddToXXXX
            preUpdateEnv.add(Type::Uint32.addConst(), "id_post", "idPost",
                             {preUpdateEnv.addInitialiser("const unsigned int idPost = $(0);")});

            // Replace kernel indices with the subsequent 'function' parameters
            // **YUCK** these also need doing in initialisers so the $(1) doesn't get confused with those used in addToPostDelay
            for(size_t i = 0; i < sg.getArchetype().getKernelSize().size(); i++) {
                const std::string iStr = std::to_string(i);
                preUpdateEnv.add(Type::Uint32.addConst(), "id_kernel_" + iStr, "idKernel" + iStr,
                                 {preUpdateEnv.addInitialiser("const unsigned int idKernel" + iStr + " = $(" + std::to_string(i + 1) + ");")});
            }
                    
            // Add correct functions for apply synaptic input
            preUpdateEnv.add(Type::AddToPostDenDelay, "addToPostDelay", "$(_den_delay)[" + sg.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "] += $(0)");
            preUpdateEnv.add(Type::AddToPost, "addToPost", "$(_out_post)[" + sg.getPostISynIndex(1, "$(id_post)") + "] += $(0)");
            preUpdateEnv.add(Type::AddToPre, "addToPre", "$(_out_pre)[" + sg.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");

            // Generate spike update
            if(trueSpike) {
                sg.generateSpikeUpdate(preUpdateEnv, 1, dt);
            }
            else {
                sg.generateSpikeEventUpdate(preUpdateEnv, 1, dt);
            }
        }

        // Loop through Toeplitz matrix diagonals
        env.print("for(unsigned int j = 0; j < $(_row_stride); j++)");
        {
            CodeStream::Scope b(env.getStream());

            // Create second environment for initialising Toeplitz connectivity
            EnvironmentExternal toeplitzEnv(env);
            toeplitzEnv.add(Type::Uint32.addConst(), "id_diag", "j");
            
            // Define type
            const auto addSynapseType = Type::ResolvedType::createFunction(
                Type::Void, std::vector<Type::ResolvedType>{1ull + sg.getArchetype().getKernelSize().size(), Type::Uint32});

            // Generate toeplitz connectivity generation code using custom for_each_synapse loop
            sg.generateToeplitzConnectivity(
                toeplitzEnv,
                // Within for_each_synapse loops, define addSynapse function and id_pre
                [addSynapseType](auto &env, auto &errorHandler)
                {
                    env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "addSynapse", 0}, addSynapseType, errorHandler);
                    env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "id_pre", 0}, Type::Uint32.addConst(), errorHandler);
                },
                [addSynapseType, trueSpike, &eventSuffix, &preUpdateStream, &sg](auto &env, auto generateBody)
                {
                    // Detect spike events or spikes and do the update
                    env.getStream() << "// process presynaptic events: " << (trueSpike ? "True Spikes" : "Spike type events") << std::endl;
                    if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                        env.print("for (unsigned int i = 0; i < $(_src_spk_cnt" + eventSuffix + ")[$(_pre_delay_slot)]; i++)");
                    }
                    else {
                        env.print("for (unsigned int i = 0; i < $(_src_spk_cnt" + eventSuffix + ")[0]; i++)");
                    }
                    {
                        CodeStream::Scope b(env.getStream());
                        EnvironmentExternal bodyEnv(env);

                        const std::string queueOffset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "$(_pre_delay_offset) + " : "";
                        bodyEnv.printLine("const unsigned int ipre = $(_src_spk" + eventSuffix + ")[" + queueOffset + "i];");
                        
                        // Add presynaptic index
                        bodyEnv.add(Type::Uint32.addConst(), "id_pre", "ipre");

                        // Add function substitution with parameters to add 
                        bodyEnv.add(addSynapseType, "addSynapse", preUpdateStream.str());

                        // Generate body of for_each_synapse loop within this new environment
                        generateBody(bodyEnv);
                    }
                });
        }
    }
    else {
        // Detect spike events or spikes and do the update
        env.getStream() << "// process presynaptic events: " << (trueSpike ? "True Spikes" : "Spike type events") << std::endl;
        if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
            env.print("for (unsigned int i = 0; i < $(_src_spk_cnt" + eventSuffix + ")[$(_pre_delay_slot)]; i++)");
        }
        else {
            env.print("for (unsigned int i = 0; i < $(_src_spk_cnt" + eventSuffix + ")[0]; i++)");
        }
        {
            CodeStream::Scope b(env.getStream());
            EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(env, sg);


            const std::string queueOffset = sg.getArchetype().getSrcNeuronGroup()->isDelayRequired() ? "$(_pre_delay_offset) + " : "";
            groupEnv.add(Type::Uint32.addConst(), "id_pre", "idPre",
                         {groupEnv.addInitialiser("const unsigned int idPre = $(_src_spk" + eventSuffix + ")[" + queueOffset + "i];")});

            // If connectivity is sparse
            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                groupEnv.printLine("const unsigned int npost = $(_row_length)[$(id_pre)];");
                groupEnv.getStream() << "for (unsigned int j = 0; j < npost; j++)";
                {
                    CodeStream::Scope b(groupEnv.getStream());
                    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(groupEnv, sg);

                    const auto indexType = getSynapseIndexType(sg);
                    const auto indexTypeName = indexType.getName();
                    synEnv.add(indexType.addConst(), "id_syn", "idSyn",
                               {synEnv.addInitialiser("const " + indexTypeName + " idSyn = ((" + indexTypeName + ")$(id_pre) * $(_row_stride)) + j;")});
                    synEnv.add(Type::Uint32.addConst(), "id_post", "idPost",
                               {synEnv.addInitialiser("const unsigned int idPost = $(_ind)[$(id_syn)];")});
                    
                    // Add correct functions for apply synaptic input
                    synEnv.add(Type::AddToPostDenDelay, "addToPostDelay", "$(_den_delay)[" + sg.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "] += $(0)");
                    synEnv.add(Type::AddToPost, "addToPost", "$(_out_post)[" + sg.getPostISynIndex(1, "$(id_post)") + "] += $(0)");
                    synEnv.add(Type::AddToPre, "addToPre", "$(_out_pre)[" + sg.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");

                    if(trueSpike) {
                        sg.generateSpikeUpdate(synEnv, 1, dt);
                    }
                    else {
                        sg.generateSpikeEventUpdate(synEnv, 1, dt);
                    }
                }
            }
            else if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) {
                throw std::runtime_error("The single-threaded CPU backend does not support procedural connectivity.");
            }
            else if((sg.getArchetype().getParallelismHint() == SynapseGroup::ParallelismHint::WORD_PACKED_BITMASK)
                    && (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK))
            {
                // Determine the number of words in each row
                groupEnv.printLine("const unsigned int rowWords = $(_row_stride) / 32;");
                groupEnv.print("for(unsigned int w = 0; w < rowWords; w++)");
                {
                    CodeStream::Scope b(groupEnv.getStream());

                    // Read row word
                    groupEnv.printLine("uint32_t connectivityWord = $(_gp)[($(id_pre) * rowWords) + w];");

                    // Set ipost to first synapse in connectivity word
                    groupEnv.getStream() << "unsigned int ipost = w * 32;" << std::endl;
                    groupEnv.add(Type::Uint32.addConst(), "id_post", "ipost");
                    
                    // Add correct functions for apply synaptic input
                    groupEnv.add(Type::AddToPostDenDelay, "addToPostDelay", "$(_den_delay)[" + sg.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "] += $(0)");
                    groupEnv.add(Type::AddToPost, "addToPost", "$(_out_post)[" + sg.getPostISynIndex(1, "$(id_post)") + "] += $(0)");
                    groupEnv.add(Type::AddToPre, "addToPre", "$(_out_pre)[" + sg.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");

                    // While there any bits left
                    groupEnv.getStream() << "while(connectivityWord != 0)";
                    {
                        CodeStream::Scope b(groupEnv.getStream());

                        // Cound leading zeros (as bits are indexed backwards this is index of next synapse)
                        groupEnv.printLine("const int numLZ = gennCLZ(connectivityWord);");

                        // Shift off zeros and the one just discovered
                        // **NOTE** << 32 appears to result in undefined behaviour
                        groupEnv.printLine("connectivityWord = (numLZ == 31) ? 0 : (connectivityWord << (numLZ + 1));");

                        // Add to ipost
                        groupEnv.printLine("ipost += numLZ;");

                        // If we aren't in padding region
                        // **TODO** don't bother checking if there is no padding
                        groupEnv.print("if(ipost < $(num_post))");
                        {
                            CodeStream::Scope b(env.getStream());
                            if(trueSpike) {
                                sg.generateSpikeUpdate(groupEnv, 1, dt);
                            }
                            else {
                                sg.generateSpikeEventUpdate(groupEnv, 1, dt);
                            }
                        }

                        // Increment ipost to take into account fact the next CLZ will go from bit AFTER synapse
                        groupEnv.printLine("ipost++;");
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

                    const auto indexType = getSynapseIndexType(sg);
                    const auto indexTypeName = indexType.getName();
                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        synEnv.printLine("const " + indexTypeName + " gid = ((" + indexTypeName + ")$(id_pre) * $(_row_stride)) + $(id_post);");

                        synEnv.print("if($(_gp)[gid / 32] & (0x80000000 >> (gid & 31)))");
                        synEnv.getStream() << CodeStream::OB(20);
                    }
                    else {
                        synEnv.add(indexType.addConst(), "id_syn", "idSyn",
                                   {synEnv.addInitialiser("const " + indexTypeName + " idSyn = ((" + indexTypeName + ")$(id_pre) * $(num_post)) + $(id_post);")});
                    }

                    if(trueSpike) {
                        sg.generateSpikeUpdate(synEnv, 1, dt);
                    }
                    else {
                        sg.generateSpikeEventUpdate(synEnv, 1, dt);
                    }

                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        synEnv.getStream() << CodeStream::CB(20);
                    }
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genPostsynapticUpdate(EnvironmentExternalBase &env, PostsynapticUpdateGroupMerged &sg, 
                                    double dt, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_event";

    // Get number of postsynaptic spikes
    if (sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
        env.printLine("const unsigned int numSpikes = $(_trg_spk_cnt" + eventSuffix + ")[$(_post_delay_slot)];");
    }
    else {
        env.printLine("const unsigned int numSpikes = $(_trg_spk_cnt" + eventSuffix + ")[0];");
    }

    // Loop through postsynaptic spikes
    env.getStream() << "for (unsigned int j = 0; j < numSpikes; j++)";
    {
        CodeStream::Scope b(env.getStream());

        // **TODO** prod types
        const std::string offsetTrueSpkPost = sg.getArchetype().getTrgNeuronGroup()->isDelayRequired() ? "$(_post_delay_offset) + " : "";
        env.printLine("const unsigned int spike = $(_trg_spk" + eventSuffix + ")[" + offsetTrueSpkPost + "j];");

        // Loop through column of presynaptic neurons
        if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            env.printLine("const unsigned int npre = $(_col_length)[spike];");
            env.getStream() << "for (unsigned int i = 0; i < npre; i++)";
        }
        else {
            env.print("for (unsigned int i = 0; i < $(num_pre); i++)");
        }
        {
            CodeStream::Scope b(env.getStream());
            EnvironmentGroupMergedField<PostsynapticUpdateGroupMerged> synEnv(env, sg);

            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
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
            synEnv.add(Type::AddToPre, "addToPre", "$(_out_pre)[" + sg.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");
            
            if(trueSpike) {
                sg.generateSpikeUpdate(synEnv, 1, dt);
            }
            else {
                sg.generateSpikeEventUpdate(synEnv, 1, dt);
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genPrevEventTimeUpdate(EnvironmentExternalBase &env, NeuronPrevSpikeTimeUpdateGroupMerged &ng, bool trueSpike) const
{
    const std::string suffix = trueSpike ? "" : "_event";
    const std::string time = trueSpike ? "st" : "set";
    if(ng.getArchetype().isDelayRequired()) {
        // Loop through neurons which spiked last timestep and set their spike time to time of previous timestep
        env.print("for(unsigned int i = 0; i < $(_spk_cnt" + suffix + ")[lastTimestepDelaySlot]; i++)");
        {
            CodeStream::Scope b(env.getStream());
            env.printLine("$(_prev_" + time + ")[lastTimestepDelayOffset + $(_spk" + suffix + ")[lastTimestepDelayOffset + i]] = $(t) - $(dt);");
        }
    }
    else {
        // Loop through neurons which spiked last timestep and set their spike time to time of previous timestep
        env.print("for(unsigned int i = 0; i < $(_spk_cnt" + suffix + ")[0]; i++)");
        {
            CodeStream::Scope b(env.getStream());
            env.printLine("$(_prev_" + time + ")[$(_spk" + suffix + ")[i]] = $(t) - $(dt);");
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genEmitEvent(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng, bool trueSpike) const
{
    const std::string queueOffset = ng.getArchetype().isDelayRequired() ? "$(_write_delay_offset) + " : "";
    const std::string suffix = trueSpike ? "" : "_event";
    env.print("$(_spk" + suffix + ")[" + queueOffset + "$(_spk_cnt" + suffix + ")");
    if(ng.getArchetype().isDelayRequired()) {
        env.print("[*$(_spk_que_ptr)]++]");
    }
    else {
        env.getStream() << "[0]++]";
    }
    env.printLine(" = $(id);");
}
//--------------------------------------------------------------------------
void Backend::genWriteBackReductions(EnvironmentExternalBase &env, CustomUpdateGroupMerged &cg, const std::string &idxName) const
{
    genWriteBackReductions(
        env, cg, idxName,
        [&cg](const Models::VarReference &varRef, const std::string &index)
        {
            return cg.getVarRefIndex(varRef.getDelayNeuronGroup() != nullptr, 1,
                                     varRef.getVarDims(), index);
        });
}
//--------------------------------------------------------------------------
void Backend::genWriteBackReductions(EnvironmentExternalBase &env, CustomUpdateWUGroupMergedBase &cg, const std::string &idxName) const
{
    genWriteBackReductions(
        env, cg, idxName,
        [&cg](const Models::WUVarReference &varRef, const std::string &index)
        {
            return cg.getVarRefIndex(1, varRef.getVarDims(), index);
        });
}
}   // namespace GeNN::CodeGenerator::SingleThreadedCPU

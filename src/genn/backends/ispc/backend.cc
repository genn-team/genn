#include "backend.h"

// Standard C includes
#include <cstdlib>

// Platform-specific includes
#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <stdlib.h>
#endif

// Standard C++ includes
#include <string>
#include <sstream>
#include <vector>

// GeNN includes
#include "gennUtils.h"
#include "path.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/standardLibrary.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler;
using namespace GeNN::CodeGenerator::ISPC;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const EnvironmentLibrary::Library backendFunctions = {
    {"clz", {Type::ResolvedType::createFunction(Type::Int32, {Type::Uint32}), "clz($(0))"}},
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
            m_CodeStream << "const uniform int64 " << m_Name << "Start = clock();" << std::endl;
        }
    }

    ~Timer()
    {
        // Record stop event
        if(m_TimingEnabled) {
            m_CodeStream << m_Name << "Time += (double)(clock() - " << m_Name << "Start);" << std::endl;
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
// TimerHost
//--------------------------------------------------------------------------
class TimerHost
{
public:
    TimerHost(CodeStream &codeStream, const std::string &name, bool timingEnabled)
    :   m_CodeStream(codeStream), m_Name(name), m_TimingEnabled(timingEnabled)
    {
        // Record start event
        if(m_TimingEnabled) {
            m_CodeStream << "const auto " << m_Name << "Start = std::chrono::high_resolution_clock::now();" << std::endl;
        }
    }

    ~TimerHost()
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
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Preferences
//--------------------------------------------------------------------------
void Preferences::updateHash(boost::uuids::detail::sha1 &hash) const
{
    PreferencesBase::updateHash(hash);
    Utils::updateHash(targetISA, hash);
}


//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::State
//--------------------------------------------------------------------------
State::State(const GeNN::Runtime::Runtime &)
{
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Array
//--------------------------------------------------------------------------
Array::Array(const Type::ResolvedType &type, size_t count, 
             VarLocation location, bool uninitialized, size_t alignment)
:   Runtime::ArrayBase(type, count, location, uninitialized), m_Alignment(alignment)
{
    if(count > 0) {
        allocate(count);
    }
}
    
Array::~Array()
{
    if(getCount() > 0) {
        free();
    }
}

void Array::allocate(size_t count)
{
    setCount(count);
    const size_t sizeBytes = getSizeBytes();
    
    // Using std::aligned_alloc
    // Size must be a multiple of alignment
    const size_t alignedSizeBytes = padSize(sizeBytes, m_Alignment);
    setHostPointer(reinterpret_cast<std::byte*>(
#ifdef _WIN32
        _aligned_malloc(alignedSizeBytes, m_Alignment)
#else
        std::aligned_alloc(m_Alignment, alignedSizeBytes)
#endif
        ));
    if (!getHostPointer()) {
        throw std::bad_alloc();
    }
}

void Array::free()
{
    // std::free is used to deallocate memory allocated by std::aligned_alloc
#ifdef _WIN32
    _aligned_free(getHostPointer());
#else
    std::free(getHostPointer());
#endif
    setHostPointer(nullptr);
    setCount(0);
}

void Array::pushToDevice()
{
    // ISPC runs on the CPU (host), so no transfer is needed
}

void Array::pullFromDevice()
{
    // ISPC runs on the CPU (host), so no transfer is needed
}

void Array::pushSlice1DToDevice(size_t, size_t)
{
    // ISPC runs on the CPU (host), so no transfer is needed
}

void Array::pullSlice1DFromDevice(size_t, size_t)
{
    // ISPC runs on the CPU (host), so no transfer is needed
}

void Array::memsetDeviceObject(int)
{
    throw std::runtime_error("ISPC arrays have no device object");
}

void Array::serialiseDeviceObject(std::vector<std::byte>&, bool) const
{
    throw std::runtime_error("ISPC arrays have no device object");
}

void Array::serialiseHostObject(std::vector<std::byte>& result, bool) const
{
    const size_t sizeBytes = getSizeBytes();
    
    result.resize(sizeBytes);
    
    if(sizeBytes > 0) {
        std::memcpy(result.data(), getHostPointer(), sizeBytes);
    }
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Backend
//--------------------------------------------------------------------------
Backend::Backend(const Preferences &preferences)
:   BackendBase(preferences)
{
}

void Backend::genNeuronUpdate(CodeStream &os, FileStreamCreator streamCreator, ModelSpecMerged &modelMerged, 
                              BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    if(modelMerged.getModel().getBatchSize() != 1) {
        throw std::runtime_error("The ISPC backend only supports simulations with a batch size of 1");
    }
    
    // Generate stream with neuron update code
    std::ostringstream neuronUpdateStream;
    CodeStream neuronUpdate(neuronUpdateStream);

    // Begin environment with standard library in ISPC file
    EnvironmentLibrary backendEnv(neuronUpdate, backendFunctions);
    EnvironmentLibrary neuronUpdateEnv(neuronUpdate, StandardLibrary::getMathsFunctions());
    
    // ISPC code for neuron update
    neuronUpdateEnv.getStream() << "export void updateNeurons(uniform " << modelMerged.getModel().getTimePrecision().getName() << " t";
    if(modelMerged.getModel().isRecordingInUse()) {
        neuronUpdateEnv.getStream() << ", uniform unsigned int recordingTimestep";
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
        
        // Process neuron spike time updates
        modelMerged.genMergedNeuronPrevSpikeTimeUpdateGroups(
            *this, memorySpaces,
            [this, &funcEnv](auto &n)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged neuron prev spike update group " << n.getIndex() << std::endl;
                funcEnv.getStream() << "for(uniform unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group -  auto replaced with uniform
                    funcEnv.getStream() << "const uniform MergedNeuronPrevSpikeTimeUpdateGroup" << n.getIndex() 
                                      << " * uniform group = &mergedNeuronPrevSpikeTimeUpdateGroup" << n.getIndex() << "[g]; " << std::endl;
                    
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
            [this, &funcEnv](auto &n)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged neuron spike queue update group " << n.getIndex() << std::endl;
                funcEnv.getStream() << "for(uniform unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group - auto replaced with uniform
                    funcEnv.getStream() << "const uniform MergedNeuronSpikeQueueUpdateGroup" << n.getIndex() 
                                      << " * uniform group = &mergedNeuronSpikeQueueUpdateGroup" << n.getIndex() << "[g]; " << std::endl;
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
                funcEnv.getStream() << "for(uniform unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group - auto replaced with uniform
                    funcEnv.getStream() << "const uniform MergedNeuronUpdateGroup" << n.getIndex() 
                                      << " * uniform group = &mergedNeuronUpdateGroup" << n.getIndex() << "[g]; " << std::endl;
                    EnvironmentGroupMergedField<NeuronUpdateGroupMerged> groupEnv(funcEnv, n);
                    buildStandardEnvironment(groupEnv, 1);

                    // Calculate number of words which will be used to record this population's spikes
                    if(n.getArchetype().isSpikeRecordingEnabled() || n.getArchetype().isSpikeEventRecordingEnabled()) {
                        groupEnv.printLine("const uniform unsigned int numRecordingWords = ($(num_neurons) + 31) / 32;");
                        
                        // Foreach to enable SIMD parallelism
	                groupEnv.print("foreach(i = 0 ... numRecordingWords)");
	                {
	                    CodeStream::Scope b(groupEnv.getStream());
	                    
	                    // Zero spike recording buffer - this goes in C++ code
                            if(n.getArchetype().isSpikeRecordingEnabled()) {
                                groupEnv.printLine("group->recordSpk[(recordingTimestep * numRecordingWords) + i] = 0;");
                            }

                            // Zero spike-like-event recording buffer - this goes in C++ code  
                            if(n.getArchetype().isSpikeEventRecordingEnabled()) {
                                groupEnv.printLine("group->recordSpkEvent[(recordingTimestep * numRecordingWords) + i] = 0;");
                            }
	                }
                    }
   

                    groupEnv.getStream() << std::endl;

                    // ISPC
                    // Foreach to enable SIMD parallelism
                    groupEnv.print("foreach(i = 0 ... $(num_neurons))");
                    {
                        CodeStream::Scope b(groupEnv.getStream());

                        groupEnv.add(Type::Uint32, "id", "i");

                        // Generate neuron update
                        n.generateNeuronUpdate(
                            groupEnv, 1,
                            // Emit true spikes
                            [&n, this](EnvironmentExternalBase &env)
                            {
                                // Insert code to update WU vars
                                n.generateWUVarUpdate(env, 1);

                                // If recording is enabled
                                if(n.getArchetype().isSpikeRecordingEnabled()) {
                                    env.printLine("atomic_or_global(&($(_record_spk)[(recordingTimestep * numRecordingWords) + ($(id) / 32)]), (1 << ($(id) % 32)));");
                                }

                                // Update event time
                                if(n.getArchetype().isSpikeTimeRequired()) {
                                    const std::string queueOffset = n.getArchetype().isSpikeDelayRequired() ? "$(_write_delay_offset) + " : "";
                                    env.printLine("$(_st)[" + queueOffset + "$(id)] = $(t);");
                                }

                                // Generate spike data structure updates
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
                                            const std::string queueOffset = n.getArchetype().isSpikeEventDelayRequired() ? "$(_write_delay_offset) + " : "";
                                            env.printLine("$(_set)[" + queueOffset + "$(id)] = $(t);");
                                        }

                                        // If recording is enabled
                                        if(n.getArchetype().isSpikeEventRecordingEnabled()) {
                                            env.printLine("atomic_or_global(&($(_record_spk_event)[(recordingTimestep * numRecordingWords) + ($(id) / 32)]), (1 << ($(id) % 32)));");
                                        }
                                    });
                            });
                    }
                }
            });
    }

    
    // Create stream for ISPC file
    CodeStream neuronUpdateISPC(streamCreator("neuronUpdate", "ispc"));
   
    // Integer type definitions for ISPC
    neuronUpdateISPC << "typedef uint8 uint8_t;" << std::endl;
    neuronUpdateISPC << "typedef uint16 uint16_t;" << std::endl;
    neuronUpdateISPC << "typedef uint32 uint32_t;" << std::endl;
    neuronUpdateISPC << "typedef uint64 uint64_t;" << std::endl;
    neuronUpdateISPC << "typedef int8 int8_t;" << std::endl;
    neuronUpdateISPC << "typedef int16 int16_t;" << std::endl;
    neuronUpdateISPC << "typedef int32 int32_t;" << std::endl;
    neuronUpdateISPC << "typedef int64 int64_t;" << std::endl;
    neuronUpdateISPC << std::endl;
    
    // Timing variables
    if(modelMerged.getModel().isTimingEnabled()) {
        neuronUpdateISPC << "extern uniform double neuronUpdateTime;" << std::endl;
    }
    neuronUpdateISPC << std::endl;
    
    // Struct definitions in the ISPC file
    neuronUpdateISPC << std::endl << "// Merged neuron group structures" << std::endl;
    modelMerged.genMergedNeuronUpdateGroupStructs(neuronUpdateISPC, *this);
    modelMerged.genMergedNeuronSpikeQueueUpdateStructs(neuronUpdateISPC, *this);
    modelMerged.genMergedNeuronPrevSpikeTimeUpdateStructs(neuronUpdateISPC, *this);
    
    genMergedStructArrayPush(neuronUpdateISPC, modelMerged.getMergedNeuronUpdateGroups());
    genMergedStructArrayPush(neuronUpdateISPC, modelMerged.getMergedNeuronSpikeQueueUpdateGroups());
    genMergedStructArrayPush(neuronUpdateISPC, modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups());
    

    neuronUpdateISPC << neuronUpdateStream.str();
    
    preambleHandler(os);
}

void Backend::genSynapseUpdate(CodeStream &os, FileStreamCreator streamCreator, ModelSpecMerged &modelMerged,
                               BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    if (modelMerged.getModel().getBatchSize() != 1) {
        throw std::runtime_error("The ISPC backend only supports simulations with a batch size of 1");
    }
    // Generate stream with synapse update code
    std::ostringstream synapseUpdateStream;
    CodeStream synapseUpdate(synapseUpdateStream);

    // Begin environment with standard library in ISPC file
    EnvironmentLibrary backendEnv(synapseUpdate, backendFunctions);
    EnvironmentLibrary synapseUpdateEnv(synapseUpdate, StandardLibrary::getMathsFunctions());

    // ISPC code for synapse update
    synapseUpdateEnv.getStream() << "export void updateSynapses(uniform " << modelMerged.getModel().getTimePrecision().getName() << " t)";
    {
        CodeStream::Scope b(synapseUpdateEnv.getStream());

        EnvironmentExternal funcEnv(synapseUpdateEnv);
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t");
        funcEnv.add(Type::Uint32.addConst(), "batch", "0");
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "dt", 
                    Type::writeNumeric(modelMerged.getModel().getDT(), modelMerged.getModel().getTimePrecision()));

        // Presynaptic update
        {
            Timer t(funcEnv.getStream(), "presynapticUpdate", modelMerged.getModel().isTimingEnabled());
            modelMerged.genMergedPresynapticUpdateGroups(
                *this, memorySpaces,
                [this, &funcEnv, &modelMerged](auto &s)
                {
                    CodeStream::Scope b(funcEnv.getStream());
                    funcEnv.getStream() << "// merged presynaptic update group " << s.getIndex() << std::endl;
                    funcEnv.getStream() << "for(uniform unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                    {
                        CodeStream::Scope b(funcEnv.getStream());

                        // Get reference to group - auto replaced with uniform
                        funcEnv.getStream() << "const uniform MergedPresynapticUpdateGroup" << s.getIndex() 
                                          << " * uniform group = &mergedPresynapticUpdateGroup" << s.getIndex() << "[g]; " << std::endl;
                        
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
    }

    // Create stream for ISPC file
    CodeStream synapseUpdateISPC(streamCreator("synapseUpdate", "ispc"));

    // Integer type definitions for ISPC
    synapseUpdateISPC << "typedef uint8 uint8_t;" << std::endl;
    synapseUpdateISPC << "typedef uint16 uint16_t;" << std::endl;
    synapseUpdateISPC << "typedef uint32 uint32_t;" << std::endl;
    synapseUpdateISPC << "typedef uint64 uint64_t;" << std::endl;
    synapseUpdateISPC << "typedef int8 int8_t;" << std::endl;
    synapseUpdateISPC << "typedef int16 int16_t;" << std::endl;
    synapseUpdateISPC << "typedef int32 int32_t;" << std::endl;
    synapseUpdateISPC << "typedef int64 int64_t;" << std::endl;
    synapseUpdateISPC << std::endl;
    
    // Timing variables
    if(modelMerged.getModel().isTimingEnabled()) {
        synapseUpdateISPC << "extern uniform double presynapticUpdateTime;" << std::endl;
    }
    synapseUpdateISPC << std::endl;

    // Struct definitions in the ISPC file
    synapseUpdateISPC << std::endl << "// Merged synapse group structures" << std::endl;
    modelMerged.genMergedPresynapticUpdateGroupStructs(synapseUpdateISPC, *this);
    modelMerged.genMergedPostsynapticUpdateGroupStructs(synapseUpdateISPC, *this);
    modelMerged.genMergedSynapseConnectivityInitGroupStructs(synapseUpdateISPC, *this);

    // Generate arrays of merged structs and functions to set them
    genMergedStructArrayPush(synapseUpdateISPC, modelMerged.getMergedPresynapticUpdateGroups());
    genMergedStructArrayPush(synapseUpdateISPC, modelMerged.getMergedPostsynapticUpdateGroups());
    genMergedStructArrayPush(synapseUpdateISPC, modelMerged.getMergedSynapseConnectivityInitGroups());

    synapseUpdateISPC << synapseUpdateStream.str();

    // Include the ISPC header
    os << "#include \"synapseUpdateISPC.h\"" << std::endl << std::endl;

    // Generate preamble
    preambleHandler(os);
}

void Backend::genCustomUpdate(CodeStream &os, FileStreamCreator streamCreator, ModelSpecMerged &modelMerged,
                              BackendBase::MemorySpaces&, HostHandler preambleHandler) const
{
    // Generate stream with custom update code
    std::ostringstream customUpdateStream;
    CodeStream customUpdate(customUpdateStream);

    // Begin environment with standard library in ISPC file
    EnvironmentLibrary backendEnv(customUpdate, backendFunctions);
    EnvironmentLibrary customUpdateEnv(customUpdate, StandardLibrary::getMathsFunctions());

    // ISPC code for custom update
    customUpdateEnv.getStream() << std::endl << "// Main ISPC entry point for custom updates" << std::endl;
    customUpdateEnv.getStream() << "export void updateCustom(uniform " << modelMerged.getModel().getTimePrecision().getName() << " t)";
    {
        CodeStream::Scope b(customUpdateEnv.getStream());
        
        customUpdateEnv.getStream() << "    // Custom update implementation" << std::endl;
    }

    // Create stream for ISPC file
    CodeStream customUpdateISPC(streamCreator("customUpdate", "ispc"));

    // Integer type definitions for ISPC
    customUpdateISPC << "typedef uint8 uint8_t;" << std::endl;
    customUpdateISPC << "typedef uint16 uint16_t;" << std::endl;
    customUpdateISPC << "typedef uint32 uint32_t;" << std::endl;
    customUpdateISPC << "typedef uint64 uint64_t;" << std::endl;
    customUpdateISPC << "typedef int8 int8_t;" << std::endl;
    customUpdateISPC << "typedef int16 int16_t;" << std::endl;
    customUpdateISPC << "typedef int32 int32_t;" << std::endl;
    customUpdateISPC << "typedef int64 int64_t;" << std::endl;
    customUpdateISPC << std::endl;

    // Generate struct definitions in the ISPC file
    customUpdateISPC << std::endl << "// Merged custom update group structures" << std::endl;
    modelMerged.genMergedCustomUpdateInitGroupStructs(customUpdateISPC, *this);
    modelMerged.genMergedCustomWUUpdateInitGroupStructs(customUpdateISPC, *this);
    modelMerged.genMergedCustomUpdateHostReductionStructs(customUpdateISPC, *this);

    // Generate arrays of merged structs and functions to set them
    genMergedStructArrayPush(customUpdateISPC, modelMerged.getMergedCustomUpdateInitGroups());
    genMergedStructArrayPush(customUpdateISPC, modelMerged.getMergedCustomWUUpdateInitGroups());
    genMergedStructArrayPush(customUpdateISPC, modelMerged.getMergedCustomUpdateHostReductionGroups());

    customUpdateISPC << customUpdateStream.str();

    // Include the ISPC header
    os << "#include \"customUpdateISPC.h\"" << std::endl << std::endl;
    
    // Generate preamble
    preambleHandler(os);
}

void Backend::genInit(CodeStream &os, FileStreamCreator streamCreator, ModelSpecMerged &modelMerged,
                      BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();
    if(model.getBatchSize() != 1) {
        throw std::runtime_error("The ISPC backend only supports simulations with a batch size of 1");
    }

    // Generate stream with initialization code
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

        TimerHost t(funcEnv.getStream(), "init", model.isTimingEnabled());

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Neuron groups" << std::endl;
        modelMerged.genMergedNeuronInitGroups(
            *this, memorySpaces,
            [this, &funcEnv](auto &n)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged neuron init group " << n.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << n.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const MergedNeuronInitGroup" << n.getIndex() << " *group = &mergedNeuronInitGroup" << n.getIndex() << "[g]; " << std::endl;

                    EnvironmentGroupMergedField<NeuronInitGroupMerged> groupEnv(funcEnv, n);
                    buildStandardEnvironment(groupEnv, 1);
                    n.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Synapse groups" << std::endl;
        modelMerged.genMergedSynapseInitGroups(
            *this, memorySpaces,
            [this, &funcEnv](auto &s)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged synapse init group " << s.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << s.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const MergedSynapseInitGroup" << s.getIndex() << " *group = &mergedSynapseInitGroup" << s.getIndex() << "[g]; " << std::endl;

                    EnvironmentGroupMergedField<SynapseInitGroupMerged> groupEnv(funcEnv, s);
                    buildStandardEnvironment(groupEnv, 1);
                    s.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom update groups" << std::endl;
        modelMerged.genMergedCustomUpdateInitGroups(
            *this, memorySpaces,
            [this, &funcEnv](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const uniform MergedCustomUpdateInitGroup" << c.getIndex() << " *group = &mergedCustomUpdateInitGroup" << c.getIndex() << "[g]; " << std::endl;

                    EnvironmentGroupMergedField<CustomUpdateInitGroupMerged> groupEnv(funcEnv, c);
                    buildStandardEnvironment(groupEnv, 1);
                    c.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom connectivity presynaptic update groups" << std::endl;
        modelMerged.genMergedCustomConnectivityUpdatePreInitGroups(
            *this, memorySpaces,
            [this, &funcEnv](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom connectivity presynaptic init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const MergedCustomConnectivityUpdatePreInitGroup" << c.getIndex() << " *group = &mergedCustomConnectivityUpdatePreInitGroup" << c.getIndex() << "[g]; " << std::endl;

                    EnvironmentGroupMergedField<CustomConnectivityUpdatePreInitGroupMerged> groupEnv(funcEnv, c);
                    buildStandardEnvironment(groupEnv);
                    c.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom connectivity postsynaptic update groups" << std::endl;
        modelMerged.genMergedCustomConnectivityUpdatePostInitGroups(
            *this, memorySpaces,
            [this, &funcEnv](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom connectivity postsynaptic init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const MergedCustomConnectivityUpdatePostInitGroup" << c.getIndex() << " *group = &mergedCustomConnectivityUpdatePostInitGroup" << c.getIndex() << "[g]; " << std::endl;
                    EnvironmentGroupMergedField<CustomConnectivityUpdatePostInitGroupMerged> groupEnv(funcEnv, c);
                    buildStandardEnvironment(groupEnv);
                    c.generateInit(*this, groupEnv, 1);
                }
            });

        funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
        funcEnv.getStream() << "// Custom WU update groups" << std::endl;
        modelMerged.genMergedCustomWUUpdateInitGroups(
            *this, memorySpaces,
            [this, &funcEnv](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom WU update group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const MergedCustomWUUpdateInitGroup" << c.getIndex() << " *group = &mergedCustomWUUpdateInitGroup" << c.getIndex() << "[g]; " << std::endl;

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
                    funcEnv.getStream() << "const MergedSynapseConnectivityInitGroup" << s.getIndex() << " *group = &mergedSynapseConnectivityInitGroup" << s.getIndex() << "[g]; " << std::endl;
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
                        

                        const Type::ResolvedType addSynapseType = Type::ResolvedType::createFunction(Type::Void, std::vector<Type::ResolvedType>{1ull + s.getArchetype().getKernelSize().size(), Type::Uint32});
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

        TimerHost t(funcEnv.getStream(), "initSparse", model.isTimingEnabled());

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
                    funcEnv.getStream() << "const MergedSynapseSparseInitGroup" << s.getIndex() << " *group = &mergedSynapseSparseInitGroup" << s.getIndex() << "[g]; " << std::endl;
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
            [this, &funcEnv](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom sparse WU update group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const MergedCustomWUUpdateSparseInitGroup" << c.getIndex() << " *group = &mergedCustomWUUpdateSparseInitGroup" << c.getIndex() << "[g]; " << std::endl;
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
            [this, &funcEnv](auto &c)
            {
                CodeStream::Scope b(funcEnv.getStream());
                funcEnv.getStream() << "// merged custom connectivity update sparse init group " << c.getIndex() << std::endl;
                funcEnv.getStream() << "for(unsigned int g = 0; g < " << c.getGroups().size() << "; g++)";
                {
                    CodeStream::Scope b(funcEnv.getStream());

                    // Get reference to group
                    funcEnv.getStream() << "const MergedCustomConnectivityUpdateSparseInitGroup" << c.getIndex() << " *group = &mergedCustomConnectivityUpdateSparseInitGroup" << c.getIndex() << "[g]; " << std::endl;
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
    
    // Struct definitions in the ISPC file
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
size_t Backend::getSynapticMatrixRowStride(const SynapseGroupInternal &) const
{
    return 0;
}

void Backend::genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &) const
{
    os << "// Standard C++ includes" << std::endl;
    os << "#include <chrono>" << std::endl;
    os << "#include <random>" << std::endl;
}
void Backend::genRunnerPreamble(CodeStream &, const ModelSpecMerged &) const
{
}
void Backend::genAllocateMemPreamble(CodeStream &, const ModelSpecMerged &) const
{
}
void Backend::genFreeMemPreamble(CodeStream &, const ModelSpecMerged &) const
{
}
void Backend::genStepTimeFinalisePreamble(CodeStream &, const ModelSpecMerged &) const
{
}

std::unique_ptr<GeNN::Runtime::StateBase> Backend::createState(const Runtime::Runtime &runtime) const
{
    return std::make_unique<State>(runtime);
}

std::unique_ptr<Runtime::ArrayBase> Backend::createArray(const Type::ResolvedType &type, size_t count, 
                                                        VarLocation location, bool uninitialized) const
{
    const auto &prefs = getPreferences<Preferences>();
    
    // Determine alignment based on target ISA
    // AVX-512 requires 64-byte alignment, AVX/AVX2 use 32, SSE uses 16
    size_t alignment = 16;
    if(prefs.targetISA.find("avx512") != std::string::npos) {
        alignment = 64;
    }
    else if(prefs.targetISA.find("avx") != std::string::npos) {
        alignment = 32;
    }
    
    return std::make_unique<Array>(type, count, location, uninitialized, alignment);
}

std::unique_ptr<Runtime::ArrayBase> Backend::createPopulationRNG(size_t) const
{
    return nullptr;
}

void Backend::genLazyVariableDynamicAllocation(CodeStream &, const Type::ResolvedType &, const std::string &, VarLocation, const std::string &) const
{
}

void Backend::genLazyVariableDynamicPush(CodeStream &, const Type::ResolvedType &, const std::string &, VarLocation, const std::string &) const
{
}

void Backend::genLazyVariableDynamicPull(CodeStream &, const Type::ResolvedType &, const std::string &, VarLocation, const std::string &) const
{
}

void Backend::genMergedDynamicVariablePush(CodeStream &, const std::string &, size_t, const std::string &, const std::string &, const std::string &) const
{
}

std::string Backend::getMergedGroupFieldHostTypeName(const Type::ResolvedType &type) const
{
    return type.getName();
}

void Backend::genPopVariableInit(EnvironmentExternalBase &, HandlerEnv) const
{
}
void Backend::genVariableInit(EnvironmentExternalBase &, const std::string &, const std::string &, HandlerEnv) const
{
}
void Backend::genSparseSynapseVariableRowInit(EnvironmentExternalBase &, HandlerEnv) const
{
}
void Backend::genDenseSynapseVariableRowInit(EnvironmentExternalBase &, HandlerEnv) const
{
}
void Backend::genKernelSynapseVariableInit(EnvironmentExternalBase &, SynapseInitGroupMerged &, HandlerEnv) const
{
}
void Backend::genKernelCustomUpdateVariableInit(EnvironmentExternalBase &, CustomWUUpdateInitGroupMerged &, HandlerEnv) const
{
}

std::string Backend::getAtomicOperation(const std::string &lhsPointer, const std::string &rhsValue,
                                       const Type::ResolvedType &type, AtomicOperation op) const
{
    if(op == AtomicOperation::ADD) {
        // atomic_add_global for different types
        if(type == Type::Float) {
            return "atomic_add_global(" + lhsPointer + ", " + rhsValue + ")";
        }
        else if(type == Type::Double) {
            return "atomic_add_global(" + lhsPointer + ", " + rhsValue + ")";
        }
        else if(type == Type::Uint32 || type == Type::Int32) {
            return "atomic_add_global(" + lhsPointer + ", " + rhsValue + ")";
        }
        else {
            throw std::runtime_error("Unsupported type for atomic add operation in ISPC backend");
        }
    }
    else if(op == AtomicOperation::OR) {
        // atomic_or_global for different types
        if(type == Type::Uint32 || type == Type::Int32) {
            return "atomic_or_global(" + lhsPointer + ", " + rhsValue + ")";
        }
        else {
            throw std::runtime_error("Atomic OR operation only supported for integer types in ISPC backend");
        }
    }
    else {
        throw std::runtime_error("Unsupported atomic operation in ISPC backend");
    }
}

void Backend::genGlobalDeviceRNG(CodeStream &, CodeStream &, CodeStream &, CodeStream &) const
{
}

void Backend::genTimer(CodeStream &, CodeStream &, CodeStream &, CodeStream &, CodeStream &, const std::string &, bool) const
{
}

void Backend::genReturnFreeDeviceMemoryBytes(CodeStream &os) const
{
    os << "return 0;" << std::endl;
}

void Backend::genAssert(CodeStream &os, const std::string &condition) const
{
    os << "assert(" << condition << ");" << std::endl;
}

void Backend::genMakefilePreamble(std::ostream &os, const std::vector<std::string> &moduleNames) const
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

    // Get ISPC preferences
    const auto &ispcPrefs = getPreferences<Preferences>();
    
    // Write variables to preamble
    os << "CXXFLAGS := " << cxxFlags << std::endl;
    os << "LINKFLAGS := " << linkFlags << std::endl;
    os << "ISPC := ispc" << std::endl;
    os << "ISPCFLAGS := -O2 --pic --target=" << ispcPrefs.targetISA << std::endl;
    
    // Add ISPC objects
    os << "OBJECTS += ";
    for(const auto &m : moduleNames) {
        if(m != "runner" && m != "init") {
            os << m << "ISPC.o ";
        }
    }
    os << std::endl;
}

void Backend::genMakefileLinkRule(std::ostream &os) const
{
    // Use the OBJECTS variable defined in preamble
    os << "\t@$(CXX) $(LINKFLAGS) -o $@ $(OBJECTS)" << std::endl;
}

void Backend::genMakefileCompileRule(std::ostream &os) const
{
    // Rule for compiling C++ files
    // **NOTE** needs top depend on ISPC for auto-generated header
    os << "%.o: %.cc %.d %ISPC.o" << std::endl;
    os << "\t@$(CXX) $(CXXFLAGS) -o $@ $<" << std::endl;
    
    // Rule for compiling ISPC files
    os << "%ISPC.o: %.ispc" << std::endl;
    os << "\t@$(ISPC) $(ISPCFLAGS) -o $@ -h $(@:.o=.h) $<" << std::endl;
    
    // Add dependency generation rule
    // **NOTE** needs top depend on ISPC for auto-generated header
    os << "%.d: %.cc %ISPC.o" << std::endl;
    os << "\t@$(CXX) $(CXXFLAGS) -MM -o $@ $<" << std::endl;
}

void Backend::genNMakefilePreamble(std::ostream &os, const std::vector<std::string> &moduleNames) const
{
    std::string cxxFlags = "/EHsc";
    std::string ispcFlags = "-O2 --dllexport --target=" + getPreferences<Preferences>().targetISA;
    std::string linkFlags = "/DLL";
    if (getPreferences().optimizeCode) {
        ispcFlags += " -O3";
        cxxFlags += " /O2";
    }
    if (getPreferences().debugCode) {
        ispcFlags += " -O0 -g";
        cxxFlags += " /Od /Zi";
        linkFlags += " /DEBUG";
    }

    // Write variables to preamble
    os << "CXXFLAGS = " << cxxFlags << std::endl;
    os << "LINKFLAGS = " << linkFlags << std::endl;
    os << "ISPCFLAGS = " << ispcFlags << std::endl;

    // Add ISPC objects
    os << "OBJECTS = $(OBJECTS) ";
    for(const auto &m : moduleNames) {
        if(m != "runner" && m != "init") {
            os << m << "ISPC.obj ";
        }
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genNMakefileLinkRule(std::ostream &os) const
{
    os << "runner.dll: $(OBJECTS)" << std::endl;
	os << "\t@link.exe /OUT:runner.dll $(LINKFLAGS) $(OBJECTS)" << std::endl;
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genNMakefileCompileRule(std::ostream &os) const
{
     // Rules for compiling C++ files
    // **NOTE** needs top depend on ISPC for auto-generated header
    os << "neuronUpdate.obj: neuronUpdate.cc neuronUpdateISPC.obj" << std::endl;
    os << "\t@$(CXX) $(CXXFLAGS) /c /Fo$@ neuronUpdate.cc" << std::endl;

    os << "synapseUpdate.obj: synapseUpdate.cc synapseUpdateISPC.obj" << std::endl;
    os << "\t@$(CXX) $(CXXFLAGS) /c /Fo$@ synapseUpdate.cc" << std::endl;

    os << "customUpdate.obj: customUpdate.cc customUpdateISPC.obj" << std::endl;
    os << "\t@$(CXX) $(CXXFLAGS) /c /Fo$@ customUpdate.cc" << std::endl;

    os << "init.obj: init.cc" << std::endl;
    os << "\t@$(CXX) $(CXXFLAGS) /c /Fo$@ init.cc" << std::endl;
    
    // Rules for compiling ISPC files
    // **YUCK** I don't think NMAKE inference rules are smart enough to do this properly
    os << "neuronUpdateISPC.obj: neuronUpdate.ispc" << std::endl;
    os << "\t@ispc.exe $(ISPCFLAGS) -o $@ -h neuronUpdateISPC.h neuronUpdate.ispc" << std::endl;
    os << std::endl;

    os << "synapseUpdateISPC.obj: synapseUpdate.ispc" << std::endl;
    os << "\t@ispc.exe $(ISPCFLAGS) -o $@ -h synapseUpdateISPC.h synapseUpdate.ispc" << std::endl;
    os << std::endl;

    os << "customUpdateISPC.obj: customUpdate.ispc" << std::endl;
    os << "\t@ispc.exe $(ISPCFLAGS) -o $@ -h customUpdateISPC.h customUpdate.ispc" << std::endl;
    os << std::endl;
}

void Backend::genMSBuildConfigProperties(std::ostream &) const
{
    assert(false);
}

void Backend::genMSBuildImportProps(std::ostream &) const
{
    assert(false);
}

void Backend::genMSBuildItemDefinitions(std::ostream &) const
{
    assert(false);
}

void Backend::genMSBuildCompileModule(const std::string &, std::ostream &) const
{
    assert(false);
}

void Backend::genMSBuildImportTarget(std::ostream &) const
{
    assert(false);
}

bool Backend::isArrayDeviceObjectRequired() const
{
    return false;
}

bool Backend::isArrayHostObjectRequired() const
{
    return false;
}

bool Backend::isGlobalHostRNGRequired(const ModelSpecInternal &) const
{
    return true;
}

bool Backend::isGlobalDeviceRNGRequired(const ModelSpecInternal &) const
{
    return false;
}

bool Backend::isPopulationRNGInitialisedOnDevice() const
{
    return false;
}

bool Backend::isPostsynapticRemapRequired() const
{
    return true;
}

bool Backend::isHostReductionRequired() const
{
    return false;
}

size_t Backend::getDeviceMemoryBytes() const
{
    return 0;
}

BackendBase::MemorySpaces Backend::getMergedGroupMemorySpaces(const ModelSpecMerged &) const
{
    return {};
}

boost::uuids::detail::sha1::digest_type Backend::getHashDigest() const
{
    return {};
}

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

        env.printLine("// From this calculate index of this synapse in the column-major matrix");
        env.printLine("const unsigned int colMajorIndex = (postIndex * $(_col_stride)) + $(_col_length)[postIndex];");

        env.printLine("// Increment column length corresponding to this postsynaptic neuron");
        env.printLine("$(_col_length)[postIndex]++;");

        env.printLine("// Add remapping entry");
        env.printLine("$(_remap)[colMajorIndex] = rowMajorIndex;");
    }
}

void Backend::genPrevEventTimeUpdate(EnvironmentExternalBase &env, NeuronPrevSpikeTimeUpdateGroupMerged &ng, bool trueSpike) const
{
    const std::string suffix = trueSpike ? "" : "_event";
    const std::string time = trueSpike ? "st" : "set";
    if(trueSpike ? ng.getArchetype().isSpikeDelayRequired() : ng.getArchetype().isSpikeEventDelayRequired()) {
        // Loop through neurons which spiked last timestep in parallel using foreach
        env.print("foreach(i = 0 ... $(_spk_cnt" + suffix + ")[lastTimestepDelaySlot])");
        {
            CodeStream::Scope b(env.getStream());
            env.printLine("$(_prev_" + time + ")[lastTimestepDelayOffset + $(_spk" + suffix + ")[lastTimestepDelayOffset + i]] = $(t) - $(dt);");
        }
    }
    else {
        // Loop through neurons which spiked last timestep in parallel using foreach
        env.print("foreach(i = 0 ... $(_spk_cnt" + suffix + ")[0])");
        {
            CodeStream::Scope b(env.getStream());
            env.printLine("$(_prev_" + time + ")[$(_spk" + suffix + ")[i]] = $(t) - $(dt);");
        }
    }
}

void Backend::genEmitEvent(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng, bool trueSpike) const
{
    const bool delayRequired = (trueSpike ? ng.getArchetype().isSpikeDelayRequired()
                                : ng.getArchetype().isSpikeEventDelayRequired());
    const std::string suffix = trueSpike ? "" : "_event";

    if(!delayRequired) {
        // Atomically increment spike counter to get a unique index for this spike
        // atomic_add_global returns the old value, so we use it directly as the index
        env.printLine("const unsigned int spkIdx = atomic_add_global(&($(_spk_cnt" + suffix + ")[0]), 1u);");
        
        // Write spike to this unique location
        env.printLine("$(_spk" + suffix + ")[spkIdx] = $(id);");
    }
    else {
        // For delayed spikes, the logic is similar but uses the spike queue pointer
        const std::string queueOffset = "$(_write_delay_offset) + ";
        
        // Atomically increment spike counter for the correct delay slot
        // atomic_add_global returns the old value, so we use it directly as the index
        env.printLine("const unsigned int spkIdx = atomic_add_global(&($(_spk_cnt" + suffix + ")[*$(_spk_que_ptr)]), 1u);");
        
        // Write spike to this unique location in the correct delay slot
        env.printLine("$(_spk" + suffix + ")[" + queueOffset + "spkIdx] = $(id);");
    }
}

void Backend::genPresynapticUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                   double dt, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_event";

    const bool delayRequired = (trueSpike ? sg.getArchetype().getSrcNeuronGroup()->isSpikeDelayRequired()
                                : sg.getArchetype().getSrcNeuronGroup()->isSpikeEventDelayRequired());

    // Asserts
    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) {
        assert(false);
    }
    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) {
        assert(false);
    }
    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        assert(false);
    }

    // SPARSE and DENSE connectivity
    if(!(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) &&
       !(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::DENSE)) {
        throw std::runtime_error("Only SPARSE and DENSE connectivity are supported");
    }

    // Detect spike events or spikes and do the update
    env.getStream() << "// process presynaptic events: " << (trueSpike ? "True Spikes" : "Spike type events") << std::endl;
    if(delayRequired) {
        env.print("for (uniform unsigned int i = 0; i < $(_src_spk_cnt" + eventSuffix + ")[$(_pre_delay_slot)]; i++)");
    }
    else {
        env.print("for (uniform unsigned int i = 0; i < $(_src_spk_cnt" + eventSuffix + ")[0]; i++)");
    }
    {
        CodeStream::Scope b(env.getStream());
        EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> groupEnv(env, sg);

        const std::string queueOffset = delayRequired ? "$(_pre_delay_offset) + " : "";
        groupEnv.add(Type::Uint32.addConst(), "id_pre", "idPre",
                     {groupEnv.addInitialiser("const uniform unsigned int idPre = $(_src_spk" + eventSuffix + ")[" + queueOffset + "i];")});

        // If connectivity is sparse
        if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            groupEnv.printLine("const uniform unsigned int npost = $(_row_length)[$(id_pre)];");
            
            // ISPC
            groupEnv.print("foreach (j = 0 ... npost)");
            {
                CodeStream::Scope b(groupEnv.getStream());
                EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(groupEnv, sg);

                const auto indexType = getSynapseIndexType(sg);
                const auto indexTypeName = indexType.getName();
                synEnv.add(indexType.addConst(), "id_syn", "idSyn",
                           {synEnv.addInitialiser("const " + indexTypeName + " idSyn = ((" + indexTypeName + ")$(id_pre) * $(_row_stride)) + j;")});
                synEnv.add(Type::Uint32.addConst(), "id_post", "idPost",
                           {synEnv.addInitialiser("const unsigned int idPost = $(_ind)[$(id_syn)];")});
                
                // Add correct functions for apply synaptic input - atomic operations
                synEnv.add(Type::getAddToPrePostDelay(sg.getScalarType()), "addToPostDelay", 
                           getAtomicOperation("&$(_den_delay)[" + sg.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "]", "$(0)", sg.getScalarType(), AtomicOperation::ADD));
                synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost", 
                           getAtomicOperation("&$(_out_post)[" + sg.getPostISynIndex(1, "$(id_post)") + "]", "$(0)", sg.getScalarType(), AtomicOperation::ADD));
                synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPre", "$(_out_pre)[" + sg.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");

                if(trueSpike) {
                    sg.generateSpikeUpdate(*this, synEnv, 1, dt);
                }
                else {
                    sg.generateSpikeEventUpdate(*this, synEnv, 1, dt);
                }
            }
        }
        // Otherwise (DENSE)
        else {
            // ISPC
            groupEnv.print("foreach (ipost = 0 ... $(num_post))");
            {
                CodeStream::Scope b(groupEnv.getStream());
                EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(groupEnv, sg);
                synEnv.add(Type::Uint32, "id_post", "ipost");
                
                // Add correct functions for apply synaptic input - use atomic operations for addToPost and addToPostDelay
                synEnv.add(Type::getAddToPrePostDelay(sg.getScalarType()), "addToPostDelay", "$(_den_delay)[" + sg.getPostDenDelayIndex(1, "$(id_post)", "$(1)") + "] += $(0)");
                synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost", "$(_out_post)[" + sg.getPostISynIndex(1, "$(id_post)") + "] += $(0)");
                synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPre", "$(_out_pre)[" + sg.getPreISynIndex(1, "$(id_pre)") + "] += $(0)");

                const auto indexType = getSynapseIndexType(sg);
                const auto indexTypeName = indexType.getName();
                synEnv.add(indexType.addConst(), "id_syn", "idSyn",
                           {synEnv.addInitialiser("const " + indexTypeName + " idSyn = ((" + indexTypeName + ")$(id_pre) * $(num_post)) + $(id_post);")});

                if(trueSpike) {
                    sg.generateSpikeUpdate(*this, synEnv, 1, dt);
                }
                else {
                    sg.generateSpikeEventUpdate(*this, synEnv, 1, dt);
                }
            }
        }
    }
}

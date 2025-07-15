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

// GeNN includes
#include "gennUtils.h"
#include "path.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/standardLibrary.h"

#include <vector>

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
    {"atomic_fetch_add", {Type::ResolvedType::createFunction(Type::Uint32, {Type::Uint32.createPointer(), Type::Uint32}), "atomic_add_global($(0), $(1))"}}
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

void Backend::build(const ModelSpecMerged &modelSpecMerged, const filesystem::path &outputPath,
                   const std::string &compiler, const std::map<std::string, size_t> &) const
{
    // Get ISPC preferences
    const auto &preferences = getPreferences<Preferences>();
    
    // Build command to compile ISPC code
    std::string ispcCommand = "ispc -O2";
    
    // Add target ISA
    ispcCommand += " --target=" + preferences.targetISA;
    
    // Convert output path to string
    #ifdef _WIN32
    const char pathSeparator = '\\';
    #else
    const char pathSeparator = '/';
    #endif
    
    // Create output path string manually
    // This is to avoid unavailable filesystem methods 
    std::stringstream ss;
    ss << outputPath;
    std::string outputPathStr = ss.str();
    
    // Remove quotes if present
    if (!outputPathStr.empty() && outputPathStr.front() == '"' && outputPathStr.back() == '"') {
        outputPathStr = outputPathStr.substr(1, outputPathStr.length() - 2);
    }
    
    // Ensure path ends with separator
    if(!outputPathStr.empty() && outputPathStr.back() != pathSeparator) {
        outputPathStr += pathSeparator;
    }
    
    // Construct file paths
    std::string neuronUpdatePath = outputPathStr + "neuronUpdate";
    
    ispcCommand += " \"" + neuronUpdatePath + ".ispc\"";
    
    // Add output object file
    ispcCommand += " -o \"" + neuronUpdatePath + ".o\"";
    
    // Add header file
    ispcCommand += " -h \"" + neuronUpdatePath + ".h\"";
    
    // Execute ISPC command
    const int ispcRetVal = system(ispcCommand.c_str());
    if(ispcRetVal != 0) {
        throw std::runtime_error("ISPC compilation failed with return code " + std::to_string(ispcRetVal));
    }
    
    // Build command to compile C++ code
    std::string cppCommand = compiler;
    cppCommand += " -shared -fPIC";
    if(getPreferences().optimizeCode) {
        cppCommand += " -O3";
    }
    
    // Add object file
    cppCommand += " \"" + neuronUpdatePath + ".o\"";
    
    // Add output shared library
    #ifdef _WIN32
    std::string libPath = outputPathStr + "runner_Release.dll";
    #else
    std::string libPath = outputPathStr + "librunner.so";
    #endif
    cppCommand += " -o \"" + libPath + "\"";
    
    // Execute C++ command
    const int cppRetVal = system(cppCommand.c_str());
    if(cppRetVal != 0) {
        throw std::runtime_error("C++ compilation failed with return code " + std::to_string(cppRetVal));
    }
}

void Backend::genNeuronUpdate(CodeStream &os, FileStreamCreator streamCreator, ModelSpecMerged &modelMerged, 
                              BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    if(modelMerged.getModel().getBatchSize() != 1) {
        throw std::runtime_error("The ISPC backend only supports simulations with a batch size of 1");
    }

    // Create seperate stream for ISPC file
    CodeStream neuronUpdateISPC(streamCreator("neuronUpdate", "ispc"));
   
    // Begin environment with standard library
    EnvironmentLibrary backendEnv(neuronUpdateISPC, backendFunctions);
    EnvironmentLibrary neuronUpdateEnv(neuronUpdateISPC, StandardLibrary::getMathsFunctions());
    
    // Create a stream for C++ wrapper code
    std::stringstream neuronUpdateStream;
    CodeStream neuronUpdateCPP(neuronUpdateStream);
    
    // Generate ISPC code for neuron update
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
        
        // Process neuron spike time updates
        modelMerged.genMergedNeuronPrevSpikeTimeUpdateGroups(
            *this, memorySpaces,
            [this, &funcEnv](auto &n)
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
            [this, &funcEnv](auto &n)
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

    // Generate struct definitions
    modelMerged.genMergedNeuronUpdateGroupStructs(neuronUpdateCPP, *this);
    modelMerged.genMergedNeuronSpikeQueueUpdateStructs(neuronUpdateCPP, *this);
    modelMerged.genMergedNeuronPrevSpikeTimeUpdateStructs(neuronUpdateCPP, *this);

    // Generate arrays of merged structs and functions to set them
    modelMerged.genMergedNeuronUpdateGroupHostStructArrayPush(neuronUpdateCPP, *this);
    modelMerged.genMergedNeuronSpikeQueueUpdateHostStructArrayPush(neuronUpdateCPP, *this);
    modelMerged.genMergedNeuronPrevSpikeTimeUpdateHostStructArrayPush(neuronUpdateCPP, *this);

    // Generate preamble
    preambleHandler(neuronUpdateCPP);

    // Write the C++ wrapper code to the main output stream
    os << neuronUpdateStream.str();
}

void Backend::genSynapseUpdate(CodeStream &os, FileStreamCreator streamCreator, ModelSpecMerged &modelMerged,
                              BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    // ISPC file stream
    CodeStream synapseUpdateISPC(streamCreator("synapseUpdate", "ispc"));
    
    // C++ wrapper code stream
    std::stringstream synapseUpdateStream;
    CodeStream synapseUpdateCPP(synapseUpdateStream);
    
    // Write C++ wrapper code to main output stream
    os << synapseUpdateStream.str();
}

void Backend::genCustomUpdate(CodeStream &os, FileStreamCreator streamCreator, ModelSpecMerged &modelMerged,
                             BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    // ISPC file stream
    CodeStream customUpdateISPC(streamCreator("customUpdate", "ispc"));
    
    // C++ wrapper code stream
    std::stringstream customUpdateStream;
    CodeStream customUpdateCPP(customUpdateStream);
    
    // Write C++ wrapper code to main output stream
    os << customUpdateStream.str();
}

void Backend::genInit(CodeStream &os, FileStreamCreator streamCreator, ModelSpecMerged &modelMerged,
                     BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    // ISPC file stream
    CodeStream initISPC(streamCreator("init", "ispc"));
    
    // C++ wrapper code stream
    std::stringstream initStream;
    CodeStream initCPP(initStream);
    
    // Write the C++ wrapper code to the main output stream
    os << initStream.str();
}

size_t Backend::getSynapticMatrixRowStride(const SynapseGroupInternal &) const
{
    return 0;
}

void Backend::genDefinitionsPreamble(CodeStream &, const ModelSpecMerged &) const
{
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

std::string Backend::getMergedGroupFieldHostTypeName(const Type::ResolvedType &) const
{
    return "";
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

std::string Backend::getAtomicOperation(const std::string &, const std::string &,
                                       const Type::ResolvedType &, AtomicOperation) const
{
    return "";
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

    // Get ISPC preferences
    const auto &ispcPrefs = getPreferences<Preferences>();
    
    // Write variables to preamble
    os << "CXXFLAGS := " << cxxFlags << std::endl;
    os << "LINKFLAGS := " << linkFlags << std::endl;
    os << "ISPC := ispc" << std::endl;
    os << "ISPCFLAGS := -O2 --target=" << ispcPrefs.targetISA << std::endl;

    os << std::endl;
}

void Backend::genMakefileLinkRule(std::ostream &os) const
{
    os << "\t@$(CXX) $(LINKFLAGS) -o $@ $(OBJECTS)" << std::endl;
}

void Backend::genMakefileCompileRule(std::ostream &os) const
{
    // Rule for compiling C++ files
    os << "%.o: %.cc %.d" << std::endl;
    os << "\t@$(CXX) $(CXXFLAGS) -o $@ $<" << std::endl;
    
    // Rule for compiling ISPC files
    os << "%.o: %.ispc" << std::endl;
    os << "\t@$(ISPC) $(ISPCFLAGS) -o $@ -h $(@:.o=.h) $<" << std::endl;
}

void Backend::genMSBuildConfigProperties(std::ostream &) const
{
}

void Backend::genMSBuildImportProps(std::ostream &) const
{
}

void Backend::genMSBuildItemDefinitions(std::ostream &) const
{
}

void Backend::genMSBuildCompileModule(const std::string &, std::ostream &) const
{
}

void Backend::genMSBuildImportTarget(std::ostream &) const
{
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
        env.printLine("const unsigned int spkIdx = atomic_fetch_add(&($(_spk_cnt" + suffix + ")[0]), 1u);");
        
        // Write spike to this unique location
        env.printLine("$(_spk" + suffix + ")[spkIdx] = $(id);");
    }
    else {
        // For delayed spikes, the logic is similar but uses the spike queue pointer
        const std::string queueOffset = "$(_write_delay_offset) + ";
        
        // Atomically increment spike counter for the correct delay slot
        env.printLine("const unsigned int spkIdx = atomic_fetch_add(&($(_spk_cnt" + suffix + ")[*$(_spk_que_ptr)]), 1u);");
        
        // Write spike to this unique location in the correct delay slot
        env.printLine("$(_spk" + suffix + ")[" + queueOffset + "spkIdx] = $(id);");
    }
}
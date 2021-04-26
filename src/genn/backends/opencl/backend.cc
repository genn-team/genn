#include "backend.h"

// Standard C++ includes
#include <algorithm>
#include <iterator>

// GeNN includes
#include "gennUtils.h"
#include "logging.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/substitutions.h"

// OpenCL backend includes
#include "utils.h"

using namespace CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace 
{
const std::vector<Substitutions::FunctionTemplate> openclLFSRFunctions = {
    {"gennrand_uniform", 0, "clrngLfsr113RandomU01($(rng))"},
    {"gennrand_normal", 0, "normalDistLfsr113($(rng))"},
    {"gennrand_exponential", 0, "exponentialDistLfsr113($(rng))"},
    {"gennrand_log_normal", 2, "logNormalDistLfsr113($(rng), $(0), $(1))"},
    {"gennrand_gamma", 1, "gammaDistLfsr113($(rng), $(0))"}
};
//-----------------------------------------------------------------------
const std::vector<Substitutions::FunctionTemplate> openclPhilloxFunctions = {
    {"gennrand_uniform", 0, "clrngPhilox432RandomU01($(rng))"},
    {"gennrand_normal", 0, "normalDistPhilox432($(rng))"},
    {"gennrand_exponential", 0, "exponentialDistPhilox432($(rng))"},
    {"gennrand_log_normal", 2, "logNormalDistPhilox432($(rng), $(0), $(1))"},
    {"gennrand_gamma", 1, "gammaDistPhilox432($(rng), $(0))"}
};
//--------------------------------------------------------------------------
template<typename T>
void genMergedGroupKernelParams(CodeStream &os, const std::vector<T> &groups, bool includeFinalComma = false)
{
    // Loop through groups and add pointer
    // **NOTE** ideally we'd use __constant here (which in OpenCL appears to be more of a hint) but seems to cause weird ptx errors
    for(size_t i = 0; i < groups.size(); i++) {
        os << "__global struct Merged" << T::name << "Group" << i << " *d_merged" << T::name << "Group" << i;
        if(includeFinalComma || i != (groups.size() - 1)) {
            os << ", ";
        }
    }
}
//--------------------------------------------------------------------------
template<typename T>
void setMergedGroupKernelParams(CodeStream &os, const std::string &kernelName, const std::vector<T> &groups, size_t &start)
{
    // Loop through groups and set as kernel arguments
    for(size_t i = 0; i < groups.size(); i++) {
        os << "CHECK_OPENCL_ERRORS(" << kernelName << ".setArg(" << start + i << ", d_merged" << T::name << "Group" << i << "));" << std::endl;
    }

    start += groups.size();
}
//--------------------------------------------------------------------------
template<typename T>
void setMergedGroupKernelParams(CodeStream &os, const std::string &kernelName, const std::vector<T> &groups)
{
    size_t start = 0;
    setMergedGroupKernelParams(os, kernelName, groups, start);
}
//-----------------------------------------------------------------------
void genGroupStartIDs(CodeStream &, size_t &)
{
}
//-----------------------------------------------------------------------
template<typename T, typename G, typename ...Args>
void genGroupStartIDs(CodeStream &os, size_t &idStart,
                      const std::vector<T> &mergedGroups, G getPaddedNumThreads,
                      Args... args)
{
    // Loop through merged groups
    for(const auto &m : mergedGroups) {
        // Declare array of starting thread indices for each neuron group
        os << "__constant unsigned int d_merged" << T::name << "GroupStartID" << m.getIndex() << "[] = {";
        for(const auto &ng : m.getGroups()) {
            os << idStart << ", ";
            idStart += getPaddedNumThreads(ng.get());
        }
        os << "};" << std::endl;
    }

    // Generate any remaining groups
    genGroupStartIDs(os, idStart, args...);
}
//-----------------------------------------------------------------------
template<typename ...Args>
void genMergedKernelDataStructures(CodeStream &os, Args... args)
{
    // Generate group start id arrays
    size_t idStart = 0;
    genGroupStartIDs(os, std::ref(idStart), args...);
}
//-----------------------------------------------------------------------
void genReadEventTiming(CodeStream &os, const std::string &name)
{
    os << "const cl_ulong tmpStart = " << name << "Event.getProfilingInfo<CL_PROFILING_COMMAND_START>();" << std::endl;
    os << "const cl_ulong tmpEnd = " << name << "Event.getProfilingInfo<CL_PROFILING_COMMAND_END>();" << std::endl;
    os << name << "Time += (double)(tmpEnd - tmpStart) / 1.0E9;" << std::endl;
}
}

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::Backend
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace OpenCL
{
//--------------------------------------------------------------------------
Backend::Backend(const KernelBlockSize &kernelBlockSizes, const Preferences &preferences,
                 const std::string &scalarType, unsigned int platformIndex, unsigned int deviceIndex)
:   BackendSIMT(kernelBlockSizes, preferences, scalarType), m_ChosenPlatformIndex(platformIndex), m_ChosenDeviceIndex(deviceIndex)
{
    // Throw exceptions if unsupported preferences are selected
    if(preferences.automaticCopy) {
        throw std::runtime_error("OpenCL backend does not currently support automatic copy mode.");
    }

    // Get platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(m_ChosenPlatformIndex < platforms.size());
    m_ChosenPlatform = platforms[m_ChosenPlatformIndex];

    // Show platform name
    LOGI_BACKEND << "Using OpenCL platform:" << m_ChosenPlatform.getInfo<CL_PLATFORM_NAME>();

    // Get platform devices
    std::vector<cl::Device> platformDevices;
    m_ChosenPlatform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
    assert(m_ChosenDeviceIndex < platformDevices.size());
    m_ChosenDevice = platformDevices[m_ChosenDeviceIndex];

    // Show device name
    LOGI_BACKEND << "Using OpenCL device:" << m_ChosenDevice.getInfo<CL_DEVICE_NAME>();

    if(isChosenDeviceAMD()) {
        LOGI_BACKEND << "Detected as an AMD device";
    }
    else if(isChosenDeviceNVIDIA()) {
        LOGI_BACKEND << "Detected as an NVIDIA device";
    }
    // Set pointer size
    const cl_int deviceAddressBytes = m_ChosenDevice.getInfo<CL_DEVICE_ADDRESS_BITS>() / 8;
    setPointerBytes(deviceAddressBytes);
    LOGI_BACKEND << "Device uses " << deviceAddressBytes * 8 << " bit pointers";

    // Determine minimum alignement
    m_AllocationAlignementBytes = m_ChosenDevice.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() / 8;
    LOGI_BACKEND << "Device uses " << m_AllocationAlignementBytes << " byte alignement";

    // Add OpenCL-specific types
    addType("clrngLfsr113Stream", 48);
    addType("clrngPhilox432Stream", 132);
}
//--------------------------------------------------------------------------
bool Backend::areSharedMemAtomicsSlow() const
{
    // If device doesn't have REAL shared memory, then atomics will definitely be slow!
    // **TODO** realistically, none of the shared memory optimizations we perform should be done on these devices
    return (m_ChosenDevice.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() != CL_LOCAL);
}
//--------------------------------------------------------------------------
std::string Backend::getAtomic(const std::string &type, AtomicOperation op, AtomicMemSpace memSpace) const
{
    // If operation is an atomic add
    if(op == AtomicOperation::ADD) {
        if(type == "float" || type == "double") {
            if(memSpace == AtomicMemSpace::GLOBAL) {
                return "atomic_add_f_global";
            }
            else {
                assert(memSpace == AtomicMemSpace::SHARED);
                return "atomic_add_f_local";
            }
        }
        else {
            return "atomic_add";
        }
    }
    // Otherwise, it's an atomic or
    else {
        assert(op == AtomicOperation::OR);
        assert(type == "unsigned int" || type == "int");
        return "atomic_or";
    }
}
//--------------------------------------------------------------------------
void Backend::genSharedMemBarrier(CodeStream &os) const
{
    os << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genPopulationRNGInit(CodeStream&, const std::string&, const std::string&, const std::string&) const
{
    assert(false);
}
//--------------------------------------------------------------------------
void Backend::genPopulationRNGPreamble(CodeStream &os, Substitutions &subs, const std::string &globalRNG, const std::string &name) const
{
    os << "clrngLfsr113Stream localStream;" << std::endl;
    os << "clrngLfsr113CopyOverStreamsFromGlobal(1, &localStream, &" << globalRNG << ");" << std::endl;
    subs.addVarSubstitution(name, "&localStream");
}
//--------------------------------------------------------------------------
void Backend::genPopulationRNGPostamble(CodeStream &os, const std::string &globalRNG) const
{
    os << "clrngLfsr113CopyOverStreamsToGlobal(1, &" << globalRNG << ", &localStream);" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genGlobalRNGSkipAhead(CodeStream &os, Substitutions &subs, const std::string &sequence, const std::string &name) const
{
    // Make local copy of host stream
    os << "clrngPhilox432Stream localStream;" << std::endl;
    os << "clrngPhilox432CopyOverStreamsFromGlobal(1, &localStream, &d_rng[0]);" << std::endl;

    // Convert id into steps, add these to steps, zero deck index
    // **NOTE** manually regenerating deck is not necessary as it is done on next call to clrngPhilox432NextState if deckIndex == 0
    os << "const clrngPhilox432Counter steps = {{0, " << sequence << "}, {0, 0}};" << std::endl;
    os << "localStream.current.ctr = clrngPhilox432Add(localStream.current.ctr, steps);" << std::endl;
    os << "localStream.current.deckIndex = 0;" << std::endl;
    subs.addVarSubstitution(name, "&localStream");
}
//--------------------------------------------------------------------------
void Backend::genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces&,
                              HostHandler preambleHandler, NeuronGroupSimHandler simHandler, NeuronUpdateGroupMergedHandler wuVarUpdateHandler,
                              HostHandler pushEGPHandler) const
{
    // Generate reset kernel to be run before the neuron kernel
    const ModelSpecInternal &model = modelMerged.getModel();

    // Atomic or into a shared memory array - which is required for recording with more than one 
    // word per workgroup - seems broken on NVIDIA OpenCL so give error if this configuration is used
    if(isChosenDeviceNVIDIA() && getKernelBlockSize(KernelNeuronUpdate) != 32 && model.isRecordingInUse()) {
        LOGE_BACKEND << "A bug in NVIDIA OpenCL drivers means that spike recording is broken with neuron workgroup size != 32" << std::endl;
    }

    os << "//--------------------------------------------------------------------------" << std::endl;
    os << "// OpenCL program and kernels" << std::endl;
    os << "//--------------------------------------------------------------------------" << std::endl;
    os << "cl::Program neuronUpdateProgram;" << std::endl;
    os << "cl::Kernel " << KernelNames[KernelNeuronSpikeQueueUpdate] << ";" << std::endl;
    os << "cl::Kernel " << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << ";" << std::endl;
    os << "cl::Kernel " << KernelNames[KernelNeuronUpdate] << ";" << std::endl;
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedNeuronSpikeQueueUpdateGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedNeuronUpdateGroups());
    os << std::endl;

    // Generate preamble
    preambleHandler(os);

    // Creating the kernel body separately so it can be split into multiple string literals
    std::stringstream neuronUpdateKernelsStream;
    CodeStream neuronUpdateKernels(neuronUpdateKernelsStream);

    // Include definitions
    genKernelPreamble(neuronUpdateKernels, modelMerged);
    neuronUpdateKernels << std::endl << std::endl;

    // Generate support code
    modelMerged.genNeuronUpdateGroupSupportCode(neuronUpdateKernels, false);
    modelMerged.genPresynapticUpdateSupportCode(neuronUpdateKernels, false);
    neuronUpdateKernels << std::endl << std::endl;

    // Generate struct definitions
    modelMerged.genMergedNeuronUpdateGroupStructs(neuronUpdateKernels, *this);
    modelMerged.genMergedNeuronSpikeQueueUpdateStructs(neuronUpdateKernels, *this);
    modelMerged.genMergedNeuronPrevSpikeTimeUpdateStructs(neuronUpdateKernels, *this);

    // Generate merged data structures
    genMergedKernelDataStructures(neuronUpdateKernels, modelMerged.getMergedNeuronUpdateGroups(),
                                  [this](const NeuronGroupInternal &ng) { return padKernelSize(ng.getNumNeurons(), KernelNeuronUpdate); });
    genMergedKernelDataStructures(neuronUpdateKernels, modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups(),
                                  [this](const NeuronGroupInternal &ng) { return padKernelSize(ng.getNumNeurons(), KernelNeuronPrevSpikeTimeUpdate); });
    neuronUpdateKernels << std::endl;

    // Generate kernels used to populate merged structs
    genMergedStructBuildKernels(neuronUpdateKernels, modelMerged, modelMerged.getMergedNeuronSpikeQueueUpdateGroups());
    genMergedStructBuildKernels(neuronUpdateKernels, modelMerged, modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups());
    genMergedStructBuildKernels(neuronUpdateKernels, modelMerged, modelMerged.getMergedNeuronUpdateGroups());

    //! KernelPreNeuronReset START
    size_t idNeuronPrevSpikeTimeUpdate = 0;

    // If any neuron groups require their previous spike times updating
    if(!modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups().empty()) {
        // Declare neuron spike queue update kernel
        neuronUpdateKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelNeuronPrevSpikeTimeUpdate) << ", 1, 1)))" << std::endl;
        neuronUpdateKernels << "__kernel void " << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << "(";
        genMergedGroupKernelParams(neuronUpdateKernels, modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups(), true);
        neuronUpdateKernels << model.getTimePrecision() << " t";
        if(shouldUseSubBufferAllocations()) {
            neuronUpdateKernels << ", __global void *d_staticBuffer"; 
        }
        neuronUpdateKernels << ")";
        {
            CodeStream::Scope b(neuronUpdateKernels);

            neuronUpdateKernels << "const unsigned int id = get_global_id(0);" << std::endl;
            Substitutions kernelSubs(openclLFSRFunctions);
            kernelSubs.addVarSubstitution("t", "t");
            if(model.getBatchSize() > 1) {
                neuronUpdateKernels << "const unsigned int batch = get_global_id(1);" << std::endl;
                kernelSubs.addVarSubstitution("batch", "batch");
            }
            else {
                kernelSubs.addVarSubstitution("batch", "0");
            }
            genNeuronPrevSpikeTimeUpdateKernel(neuronUpdateKernels, kernelSubs, modelMerged, idNeuronPrevSpikeTimeUpdate);
        }
        neuronUpdateKernels << std::endl;
    }

    //! KernelPreNeuronReset START
    size_t idNeuronSpikeQueueUpdate = 0;

    // Declare neuron spike queue update kernel
    neuronUpdateKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelNeuronSpikeQueueUpdate) << ", 1, 1)))" << std::endl;
    neuronUpdateKernels << "__kernel void " << KernelNames[KernelNeuronSpikeQueueUpdate] << "(";
    genMergedGroupKernelParams(neuronUpdateKernels, modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), shouldUseSubBufferAllocations());
    if(shouldUseSubBufferAllocations()) {
        neuronUpdateKernels << "__global void *d_staticBuffer"; 
    }
    neuronUpdateKernels << ")";
    {
        CodeStream::Scope b(neuronUpdateKernels);

        neuronUpdateKernels << "const unsigned int id = get_global_id(0);" << std::endl;

        genNeuronSpikeQueueUpdateKernel(neuronUpdateKernels, modelMerged, idNeuronSpikeQueueUpdate);
    }
    neuronUpdateKernels << std::endl;

    size_t idStart = 0;

    neuronUpdateKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelNeuronUpdate) << ", 1, 1)))" << std::endl;
    neuronUpdateKernels << "__kernel void " << KernelNames[KernelNeuronUpdate] << "(";
    genMergedGroupKernelParams(neuronUpdateKernels, modelMerged.getMergedNeuronUpdateGroups(), true);
    neuronUpdateKernels << model.getTimePrecision() << " t";
    if(model.isRecordingInUse()) {
        neuronUpdateKernels << ", unsigned int recordingTimestep";
    }
    if(shouldUseSubBufferAllocations()) {
        neuronUpdateKernels << ", __global void *d_staticBuffer"; 
    }
    neuronUpdateKernels << ")" << std::endl;
    {
        CodeStream::Scope b(neuronUpdateKernels);

        Substitutions kernelSubs(openclLFSRFunctions);
        kernelSubs.addVarSubstitution("t", "t");

        neuronUpdateKernels << "const unsigned int id = get_global_id(0);" << std::endl;
        if(model.getBatchSize() > 1) {
            neuronUpdateKernels << "const unsigned int batch = get_global_id(1);" << std::endl;
            kernelSubs.addVarSubstitution("batch", "batch");
        }
        else {
            kernelSubs.addVarSubstitution("batch", "0");
        }
        genNeuronUpdateKernel(neuronUpdateKernels, kernelSubs, modelMerged, 
                              simHandler, wuVarUpdateHandler, idStart);
    }
    neuronUpdateKernels << std::endl;

    // Write out kernel source string literal
    os << "const char* neuronUpdateSrc = ";
    divideKernelStreamInParts(os, neuronUpdateKernelsStream, 5000);
    os << ";" << std::endl;
    os << std::endl;

    // Function for initializing the KernelNeuronUpdate kernels
    os << "// Initialize the neuronUpdate kernels" << std::endl;
    os << "void buildNeuronUpdateProgram()";
    {
        CodeStream::Scope b(os);

        // If there are any kernels (some implementations complain)
        if(idNeuronPrevSpikeTimeUpdate > 0 || idNeuronSpikeQueueUpdate > 0 || idStart > 0) {
            os << "// Build program" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(neuronUpdateProgram = cl::Program(clContext, neuronUpdateSrc, false, &error));" << std::endl;
            genBuildProgramFlagsString(os);
            os << "if(neuronUpdateProgram.build(buildProgramFlags.c_str()) != CL_SUCCESS)";
            {
                CodeStream::Scope b(os);
                os << "std::cerr << neuronUpdateProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevice);" << std::endl;
                os << "throw std::runtime_error(\"Neuron update program compile error\");" << std::endl;
            }
            os << std::endl;

            os << "// Configure merged struct buffers and kernels" << std::endl;
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), "neuronUpdateProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups(), "neuronUpdateProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedNeuronUpdateGroups(), "neuronUpdateProgram");
            os << std::endl;
        }

        // KernelNeuronPrevSpikeTimeUpdate initialisation
        if(idNeuronPrevSpikeTimeUpdate > 0) {
            os << "// Configure neuron previous spike time update kernel" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << " = cl::Kernel(neuronUpdateProgram, \"" << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << "\", &error));" << std::endl;
            setMergedGroupKernelParams(os, KernelNames[KernelNeuronPrevSpikeTimeUpdate], modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups());
            if(shouldUseSubBufferAllocations()) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << ".setArg(" << (modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups().size() + 1) << ", d_staticBuffer));" << std::endl;
            }
            os << std::endl;
        }

        // KernelNeuronSpikeQueueUpdate initialization
        if(idNeuronSpikeQueueUpdate > 0) {
            os << "// Configure neuron spike queue update kernel" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelNeuronSpikeQueueUpdate] << " = cl::Kernel(neuronUpdateProgram, \"" << KernelNames[KernelNeuronSpikeQueueUpdate] << "\", &error));" << std::endl;
            setMergedGroupKernelParams(os, KernelNames[KernelNeuronSpikeQueueUpdate], modelMerged.getMergedNeuronSpikeQueueUpdateGroups());
            if(shouldUseSubBufferAllocations()) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelNeuronSpikeQueueUpdate] << ".setArg(" << modelMerged.getMergedNeuronSpikeQueueUpdateGroups().size() << ", d_staticBuffer));" << std::endl;
            }
            os << std::endl;
        }

        // KernelNeuronUpdate initialization
        if(idStart > 0) {
            os << "// Configure neuron update kernel" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelNeuronUpdate] << " = cl::Kernel(neuronUpdateProgram, \"" << KernelNames[KernelNeuronUpdate] << "\", &error));" << std::endl;
            setMergedGroupKernelParams(os, KernelNames[KernelNeuronUpdate], modelMerged.getMergedNeuronUpdateGroups());
            if(shouldUseSubBufferAllocations()) {
                const size_t start = modelMerged.getMergedNeuronUpdateGroups().size() + (model.isRecordingInUse() ? 2 : 1);
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelNeuronUpdate] << ".setArg(" << start << ", d_staticBuffer));" << std::endl;
            }
            os << std::endl;
        }
    }

    os << std::endl;

    os << "void updateNeurons(" << model.getTimePrecision() << " t";
    if(model.isRecordingInUse()) {
        os << ", unsigned int recordingTimestep";
    }
    os << ")";
    {
        CodeStream::Scope b(os);

        // Push any required EGPS
        pushEGPHandler(os);

        if (idNeuronPrevSpikeTimeUpdate > 0) {
            CodeStream::Scope b(os);
            os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << ".setArg(" << modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups().size() << ", t));" << std::endl;
            genKernelDimensions(os, KernelNeuronPrevSpikeTimeUpdate, idNeuronPrevSpikeTimeUpdate, model.getBatchSize());
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
            os << std::endl;
        }
        if (idNeuronSpikeQueueUpdate > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelNeuronSpikeQueueUpdate, idNeuronSpikeQueueUpdate, 1);
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelNeuronSpikeQueueUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
            os << std::endl;
        }
        if (idStart > 0) {
            CodeStream::Scope b(os);
            os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelNeuronUpdate] << ".setArg(" << modelMerged.getMergedNeuronUpdateGroups().size() << ", t));" << std::endl;
            if(model.isRecordingInUse()) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelNeuronUpdate] << ".setArg(" << modelMerged.getMergedNeuronUpdateGroups().size() + 1 << ", recordingTimestep));" << std::endl;
            }
            os << std::endl;
            genKernelDimensions(os, KernelNeuronUpdate, idStart, model.getBatchSize());
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelNeuronUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize";
            if(model.isTimingEnabled()) {
                os << ", nullptr, &neuronUpdateEvent";
            }
            os << "));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genSynapseUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces&,
                               HostHandler preambleHandler, PresynapticUpdateGroupMergedHandler wumThreshHandler, PresynapticUpdateGroupMergedHandler wumSimHandler,
                               PresynapticUpdateGroupMergedHandler wumEventHandler, PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler,
                               PostsynapticUpdateGroupMergedHandler postLearnHandler, SynapseDynamicsGroupMergedHandler synapseDynamicsHandler,
                               HostHandler pushEGPHandler) const
{
    // Generate reset kernel to be run before the neuron kernel
    const ModelSpecInternal &model = modelMerged.getModel();

    os << "//--------------------------------------------------------------------------" << std::endl;
    os << "// OpenCL program and kernels" << std::endl;
    os << "//--------------------------------------------------------------------------" << std::endl;
    os << "cl::Program synapseUpdateProgram;" << std::endl;
    os << "cl::Kernel " << KernelNames[KernelSynapseDendriticDelayUpdate] << ";" << std::endl;
    os << "cl::Kernel " << KernelNames[KernelPresynapticUpdate] << ";" << std::endl;
    os << "cl::Kernel " << KernelNames[KernelPostsynapticUpdate] << ";" << std::endl;
    os << "cl::Kernel " << KernelNames[KernelSynapseDynamicsUpdate] << ";" << std::endl;
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedSynapseDendriticDelayUpdateGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedPresynapticUpdateGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedPostsynapticUpdateGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedSynapseDynamicsGroups());

    // Generate preamble
    preambleHandler(os);

    // Creating the kernel body separately so it can be split into multiple string literals
    std::stringstream synapseUpdateKernelsStream;
    CodeStream synapseUpdateKernels(synapseUpdateKernelsStream);

    // Include definitions
    genKernelPreamble(synapseUpdateKernels, modelMerged);
    synapseUpdateKernels << "// ------------------------------------------------------------------------" << std::endl;
    synapseUpdateKernels << "// bit tool macros" << std::endl;
    synapseUpdateKernels << "#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x" << std::endl;
    synapseUpdateKernels << "#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1" << std::endl;
    synapseUpdateKernels << "#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0" << std::endl;
    synapseUpdateKernels << std::endl << std::endl;

    genAtomicAddFloat(synapseUpdateKernels, "local");
    genAtomicAddFloat(synapseUpdateKernels, "global");

    // Generate support code
    modelMerged.genPresynapticUpdateSupportCode(synapseUpdateKernels, false);
    modelMerged.genPostsynapticUpdateSupportCode(synapseUpdateKernels, false);
    modelMerged.genSynapseDynamicsSupportCode(synapseUpdateKernels, false);
    synapseUpdateKernels << std::endl;

    // Generate struct definitions
    modelMerged.genMergedSynapseDendriticDelayUpdateStructs(synapseUpdateKernels, *this);
    modelMerged.genMergedPresynapticUpdateGroupStructs(synapseUpdateKernels, *this);
    modelMerged.genMergedPostsynapticUpdateGroupStructs(synapseUpdateKernels, *this);
    modelMerged.genMergedSynapseDynamicsGroupStructs(synapseUpdateKernels, *this);
    synapseUpdateKernels << std::endl;

    // Generate data structure for accessing merged groups
    genMergedKernelDataStructures(
        synapseUpdateKernels,
        modelMerged.getMergedPresynapticUpdateGroups(), [this](const SynapseGroupInternal &sg){ return padKernelSize(getNumPresynapticUpdateThreads(sg, getPreferences()), KernelPresynapticUpdate); });
    genMergedKernelDataStructures(
        synapseUpdateKernels,
        modelMerged.getMergedPostsynapticUpdateGroups(), [this](const SynapseGroupInternal &sg) { return padKernelSize(getNumPostsynapticUpdateThreads(sg), KernelPostsynapticUpdate); });
    genMergedKernelDataStructures(
        synapseUpdateKernels,
        modelMerged.getMergedSynapseDynamicsGroups(), [this](const SynapseGroupInternal &sg) { return padKernelSize(getNumSynapseDynamicsThreads(sg), KernelSynapseDynamicsUpdate); });

    // Generate kernels used to populate merged structs
    genMergedStructBuildKernels(synapseUpdateKernels, modelMerged, modelMerged.getMergedSynapseDendriticDelayUpdateGroups());
    genMergedStructBuildKernels(synapseUpdateKernels, modelMerged, modelMerged.getMergedPresynapticUpdateGroups());
    genMergedStructBuildKernels(synapseUpdateKernels, modelMerged, modelMerged.getMergedPostsynapticUpdateGroups());
    genMergedStructBuildKernels(synapseUpdateKernels, modelMerged, modelMerged.getMergedSynapseDynamicsGroups());

    // Declare neuron spike queue update kernel
    size_t idSynapseDendriticDelayUpdate = 0;
    if(!modelMerged.getMergedSynapseDendriticDelayUpdateGroups().empty()) {
        synapseUpdateKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelSynapseDendriticDelayUpdate) << ", 1, 1)))" << std::endl;
        synapseUpdateKernels << "__kernel void " << KernelNames[KernelSynapseDendriticDelayUpdate] << "(";
        genMergedGroupKernelParams(synapseUpdateKernels, modelMerged.getMergedSynapseDendriticDelayUpdateGroups(), shouldUseSubBufferAllocations());
        if(shouldUseSubBufferAllocations()) {
            synapseUpdateKernels << "__global void *d_staticBuffer"; 
        }
        synapseUpdateKernels << ")";
        {
            CodeStream::Scope b(synapseUpdateKernels);
            synapseUpdateKernels << "const unsigned int id = get_global_id(0);" << std::endl;
            genSynapseDendriticDelayUpdateKernel(synapseUpdateKernels, modelMerged, idSynapseDendriticDelayUpdate);
        }
    }

    // If there are any presynaptic update groups
    size_t idPresynapticStart = 0;
    const bool globalRNGRequired = isGlobalDeviceRNGRequired(modelMerged);
    if(!modelMerged.getMergedPresynapticUpdateGroups().empty()) {
        synapseUpdateKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelPresynapticUpdate) << ", 1, 1)))" << std::endl;
        synapseUpdateKernels << "__kernel void " << KernelNames[KernelPresynapticUpdate] << "(";
        genMergedGroupKernelParams(synapseUpdateKernels, modelMerged.getMergedPresynapticUpdateGroups(), true);
        synapseUpdateKernels << model.getTimePrecision() << " t";
        if(globalRNGRequired) {
            synapseUpdateKernels << ", __global clrngPhilox432HostStream *d_rng";
        }
        if(shouldUseSubBufferAllocations()) {
            synapseUpdateKernels << ", __global void *d_staticBuffer"; 
        }
        synapseUpdateKernels << ")";
        {
            CodeStream::Scope b(synapseUpdateKernels);

            Substitutions kernelSubs(openclLFSRFunctions);
            kernelSubs.addVarSubstitution("t", "t");

            synapseUpdateKernels << "const unsigned int id = get_global_id(0);" << std::endl;
            if(model.getBatchSize() > 1) {
                synapseUpdateKernels << "const unsigned int batch = get_global_id(1);" << std::endl;
                kernelSubs.addVarSubstitution("batch", "batch");
            }
            else {
                kernelSubs.addVarSubstitution("batch", "0");
            }
            genPresynapticUpdateKernel(synapseUpdateKernels, kernelSubs, modelMerged, wumThreshHandler, 
                                       wumSimHandler, wumEventHandler, wumProceduralConnectHandler, idPresynapticStart);
        }
    }

    // If any synapse groups require postsynaptic learning
    size_t idPostsynapticStart = 0;
    if(!modelMerged.getMergedPostsynapticUpdateGroups().empty()) {
        synapseUpdateKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelPostsynapticUpdate) << ", 1, 1)))" << std::endl;
        synapseUpdateKernels << "__kernel void " << KernelNames[KernelPostsynapticUpdate] << "(";
        genMergedGroupKernelParams(synapseUpdateKernels, modelMerged.getMergedPostsynapticUpdateGroups(), true);
        synapseUpdateKernels << model.getTimePrecision() << " t";
        if(shouldUseSubBufferAllocations()) {
            synapseUpdateKernels << ", __global void *d_staticBuffer"; 
        }
        synapseUpdateKernels << ")";
        {
            CodeStream::Scope b(synapseUpdateKernels);

            Substitutions kernelSubs(openclLFSRFunctions);
            kernelSubs.addVarSubstitution("t", "t");

            synapseUpdateKernels << "const unsigned int id = get_global_id(0);" << std::endl;
            if(model.getBatchSize() > 1) {
                synapseUpdateKernels << "const unsigned int batch = get_global_id(1);" << std::endl;
                kernelSubs.addVarSubstitution("batch", "batch");
            }
            else {
                kernelSubs.addVarSubstitution("batch", "0");
            }
            genPostsynapticUpdateKernel(synapseUpdateKernels, kernelSubs, modelMerged, postLearnHandler, idPostsynapticStart);
        }
    }
 
    size_t idSynapseDynamicsStart = 0;
    if(!modelMerged.getMergedSynapseDynamicsGroups().empty()) {
        synapseUpdateKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelSynapseDynamicsUpdate) << ", 1, 1)))" << std::endl;
        synapseUpdateKernels << "__kernel void " << KernelNames[KernelSynapseDynamicsUpdate] << "(";
        genMergedGroupKernelParams(synapseUpdateKernels, modelMerged.getMergedSynapseDynamicsGroups(), true);
        synapseUpdateKernels << model.getTimePrecision() << " t";
        if(shouldUseSubBufferAllocations()) {
            synapseUpdateKernels << ", __global void *d_staticBuffer"; 
        }
        synapseUpdateKernels << ")";
        {
            CodeStream::Scope b(synapseUpdateKernels);

            Substitutions kernelSubs(openclLFSRFunctions);
            kernelSubs.addVarSubstitution("t", "t");

            synapseUpdateKernels << "const unsigned int id = get_global_id(0);" << std::endl;
            if(model.getBatchSize() > 1) {
                synapseUpdateKernels << "const unsigned int batch = get_global_id(1);" << std::endl;
                kernelSubs.addVarSubstitution("batch", "batch");
            }
            else {
                kernelSubs.addVarSubstitution("batch", "0");
            }
            genSynapseDynamicsKernel(synapseUpdateKernels, kernelSubs, modelMerged, synapseDynamicsHandler, idSynapseDynamicsStart);
        }
    }
    synapseUpdateKernels << std::endl;

    // Write out kernel source string literal
    os << "const char* synapseUpdateSrc = ";
    divideKernelStreamInParts(os, synapseUpdateKernelsStream, 5000);
    os << ";" << std::endl;
    os << std::endl;

    os << "// Initialize the synapseUpdate kernels" << std::endl;
    os << "void buildSynapseUpdateProgram()";
    {
        CodeStream::Scope b(os);

        // If there are any kernels (some implementations complain)
        if(idSynapseDendriticDelayUpdate > 0 || idPresynapticStart > 0 || idPostsynapticStart > 0 || idSynapseDynamicsStart > 0) {
            os << "// Build program" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(synapseUpdateProgram = cl::Program(clContext, synapseUpdateSrc, false, &error));" << std::endl;
            genBuildProgramFlagsString(os);
            os << "if(synapseUpdateProgram.build(buildProgramFlags.c_str()) != CL_SUCCESS)";
            {
                CodeStream::Scope b(os);
                os << "std::cerr << synapseUpdateProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevice);" << std::endl;
                os << "throw std::runtime_error(\"Synapse update program compile error\");" << std::endl;
            }
            os << std::endl;

            os << "// Configure merged struct buffers and kernels" << std::endl;
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedSynapseDendriticDelayUpdateGroups(), "synapseUpdateProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedPresynapticUpdateGroups(), "synapseUpdateProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedPostsynapticUpdateGroups(), "synapseUpdateProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedSynapseDynamicsGroups(), "synapseUpdateProgram");
            os << std::endl;
        }

        if(idSynapseDendriticDelayUpdate > 0) {
            os << "// Configure dendritic delay update kernel" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelSynapseDendriticDelayUpdate] << " = cl::Kernel(synapseUpdateProgram, \"" << KernelNames[KernelSynapseDendriticDelayUpdate] << "\", &error));" << std::endl;
            setMergedGroupKernelParams(os, KernelNames[KernelSynapseDendriticDelayUpdate], modelMerged.getMergedSynapseDendriticDelayUpdateGroups());
            if(shouldUseSubBufferAllocations()) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelSynapseDendriticDelayUpdate] << ".setArg(" << modelMerged.getMergedSynapseDendriticDelayUpdateGroups().size() << ", d_staticBuffer));" << std::endl;
            }
            os << std::endl;
        }

        if(idPresynapticStart > 0) {
            os << "// Configure presynaptic update kernel" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelPresynapticUpdate] << " = cl::Kernel(synapseUpdateProgram, \"" << KernelNames[KernelPresynapticUpdate] << "\", &error));" << std::endl;
            setMergedGroupKernelParams(os, KernelNames[KernelPresynapticUpdate], modelMerged.getMergedPresynapticUpdateGroups());
            if(globalRNGRequired) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelPresynapticUpdate] << ".setArg(" << modelMerged.getMergedPresynapticUpdateGroups().size() + 1 << ", d_rng));" << std::endl;
            }
            if(shouldUseSubBufferAllocations()) {
                const size_t start = modelMerged.getMergedPresynapticUpdateGroups().size() + (globalRNGRequired ? 2 : 1);
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelPresynapticUpdate] << ".setArg(" << start << ", d_staticBuffer));" << std::endl;
            }
            os << std::endl;
        }

        if(idPostsynapticStart > 0) {
            os << "// Configure postsynaptic update kernel" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelPostsynapticUpdate] << " = cl::Kernel(synapseUpdateProgram, \"" << KernelNames[KernelPostsynapticUpdate] << "\", &error));" << std::endl;
            setMergedGroupKernelParams(os, KernelNames[KernelPostsynapticUpdate], modelMerged.getMergedPostsynapticUpdateGroups());
            if(shouldUseSubBufferAllocations()) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelPostsynapticUpdate] << ".setArg(" << (modelMerged.getMergedPostsynapticUpdateGroups().size() + 1) << ", d_staticBuffer));" << std::endl;
            }
            os << std::endl;
        }

        if(idSynapseDynamicsStart > 0) {
            os << "// Configure postsynaptic update kernel" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelSynapseDynamicsUpdate] << " = cl::Kernel(synapseUpdateProgram, \"" << KernelNames[KernelSynapseDynamicsUpdate] << "\", &error));" << std::endl;
            setMergedGroupKernelParams(os, KernelNames[KernelSynapseDynamicsUpdate], modelMerged.getMergedSynapseDynamicsGroups());
            if(shouldUseSubBufferAllocations()) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelSynapseDynamicsUpdate] << ".setArg(" << (modelMerged.getMergedSynapseDynamicsGroups().size() + 1) << ", d_staticBuffer));" << std::endl;
            }
            os << std::endl;
        }
    }

    os << std::endl;

    os << "void updateSynapses(" << modelMerged.getModel().getTimePrecision() << " t)";
    {
        CodeStream::Scope b(os);

        // Push any required EGPs
        pushEGPHandler(os);

        // Launch pre-synapse reset kernel if required
        if (idSynapseDendriticDelayUpdate > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelSynapseDendriticDelayUpdate, idSynapseDendriticDelayUpdate, 1);
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelSynapseDendriticDelayUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
        }

        // Launch synapse dynamics kernel if required
        if (idSynapseDynamicsStart > 0) {
            CodeStream::Scope b(os);
            os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelSynapseDynamicsUpdate] << ".setArg(" << modelMerged.getMergedSynapseDynamicsGroups().size() << ", t));" << std::endl;
            os << std::endl;
            genKernelDimensions(os, KernelSynapseDynamicsUpdate, idSynapseDynamicsStart, model.getBatchSize());
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelSynapseDynamicsUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize";
            if(model.isTimingEnabled()) {
                os << ", nullptr, &synapseDynamicsEvent";
            }
            os << "));" << std::endl;
        }

        // Launch presynaptic update kernel
        if (idPresynapticStart > 0) {
            CodeStream::Scope b(os);
            os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelPresynapticUpdate] << ".setArg(" << modelMerged.getMergedPresynapticUpdateGroups().size() << ", t));" << std::endl;
            os << std::endl;
            genKernelDimensions(os, KernelPresynapticUpdate, idPresynapticStart, model.getBatchSize());
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelPresynapticUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize";
            if(model.isTimingEnabled()) {
                os << ", nullptr, &presynapticUpdateEvent";
            }
            os << "));" << std::endl;
        }

        // Launch postsynaptic update kernel
        if (idPostsynapticStart > 0) {
            CodeStream::Scope b(os);
            os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelPostsynapticUpdate] << ".setArg(" << modelMerged.getMergedPostsynapticUpdateGroups().size() << ", t));" << std::endl;
            os << std::endl;
            genKernelDimensions(os, KernelPostsynapticUpdate, idPostsynapticStart, model.getBatchSize());
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelPostsynapticUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize";
            if(model.isTimingEnabled()) {
                os << ", nullptr, &postsynapticUpdateEvent";
            }
            os << "));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genCustomUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces&, HostHandler preambleHandler,
                              CustomUpdateGroupMergedHandler customUpdateHandler, CustomUpdateWUGroupMergedHandler customWUUpdateHandler,
                              CustomUpdateTransposeWUGroupMergedHandler customWUTransposeUpdateHandler, HostHandler pushEGPHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Build map containing union of all custom update groupsnames
    std::map<std::string, std::pair<size_t, size_t>> customUpdateGroups;
    std::transform(model.getCustomUpdates().cbegin(), model.getCustomUpdates().cend(),
                   std::inserter(customUpdateGroups, customUpdateGroups.end()),
                   [](const ModelSpec::CustomUpdateValueType &v) { return std::make_pair(v.second.getUpdateGroupName(), std::make_pair(0, 0)); });
    std::transform(model.getCustomWUUpdates().cbegin(), model.getCustomWUUpdates().cend(),
                   std::inserter(customUpdateGroups, customUpdateGroups.end()),
                   [](const ModelSpec::CustomUpdateWUValueType &v) { return std::make_pair(v.second.getUpdateGroupName(), std::make_pair(0, 0)); });

    os << "//--------------------------------------------------------------------------" << std::endl;
    os << "// OpenCL program and kernels" << std::endl;
    os << "//--------------------------------------------------------------------------" << std::endl;
    os << "cl::Program customUpdateProgram;" << std::endl;
    for(const auto &g : customUpdateGroups) {
        os << "cl::Kernel " << KernelNames[KernelCustomUpdate] << g.first << ";" << std::endl;
        os << "cl::Kernel " << KernelNames[KernelCustomTransposeUpdate] << g.first << ";" << std::endl;
    }
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedCustomUpdateGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedCustomUpdateWUGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedCustomUpdateTransposeWUGroups());
    os << std::endl;

    // Generate preamble
    preambleHandler(os);

    // Creating the kernel body separately so it can be split into multiple string literals
    std::stringstream customUpdateKernelsStream;
    CodeStream customUpdateKernels(customUpdateKernelsStream);

    // Include definitions
    genKernelPreamble(customUpdateKernels, modelMerged);  

    // Generate struct definitions
    modelMerged.genMergedCustomUpdateStructs(customUpdateKernels, *this);
    modelMerged.genMergedCustomUpdateWUStructs(customUpdateKernels, *this);
    modelMerged.gemMergedCustomUpdateTransposeWUStructs(customUpdateKernels, *this);
   

    // Generate data structure for accessing merged groups from within custom update kernels
    genMergedKernelDataStructures(
        customUpdateKernels, 
        modelMerged.getMergedCustomUpdateGroups(), [this](const CustomUpdateInternal &cg) { return padKernelSize(cg.getSize(), KernelCustomUpdate); },
        modelMerged.getMergedCustomUpdateWUGroups(), [&model, this](const CustomUpdateWUInternal &cg) { return getPaddedNumCustomUpdateWUThreads(cg, model.getBatchSize()); });
    genMergedKernelDataStructures(
        customUpdateKernels,
        modelMerged.getMergedCustomUpdateTransposeWUGroups(), [&model, this](const CustomUpdateWUInternal &cg) { return getPaddedNumCustomUpdateTransposeWUThreads(cg, model.getBatchSize()); });
    customUpdateKernels << std::endl;

    // Generate kernels used to populate merged structs
    genMergedStructBuildKernels(customUpdateKernels, modelMerged, modelMerged.getMergedCustomUpdateGroups());
    genMergedStructBuildKernels(customUpdateKernels, modelMerged, modelMerged.getMergedCustomUpdateWUGroups());
    genMergedStructBuildKernels(customUpdateKernels, modelMerged, modelMerged.getMergedCustomUpdateTransposeWUGroups());

    // Loop through custom update groups
    for(auto &g : customUpdateGroups) {
        // If there are any custom updates in this group, generate kernel
        if(std::any_of(modelMerged.getMergedCustomUpdateGroups().cbegin(), modelMerged.getMergedCustomUpdateGroups().cend(),
                       [&g](const CustomUpdateGroupMerged &c) { return (c.getArchetype().getUpdateGroupName() == g.first); })
           || std::any_of(modelMerged.getMergedCustomUpdateWUGroups().cbegin(), modelMerged.getMergedCustomUpdateWUGroups().cend(),
                          [&g](const CustomUpdateWUGroupMerged &c) { return (c.getArchetype().getUpdateGroupName() == g.first); }))
        {
            customUpdateKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelCustomUpdate) << ", 1, 1)))" << std::endl;
            customUpdateKernels << "__kernel void " << KernelNames[KernelCustomUpdate] << g.first << "(";
            genMergedGroupKernelParams(customUpdateKernels, modelMerged.getMergedCustomUpdateGroups(), true);
            genMergedGroupKernelParams(customUpdateKernels, modelMerged.getMergedCustomUpdateWUGroups(), true);
            customUpdateKernels << model.getTimePrecision() << " t";
            if(shouldUseSubBufferAllocations()) {
                customUpdateKernels << ", __global void *d_staticBuffer"; 
            }
            customUpdateKernels << ")";
            {
                CodeStream::Scope b(customUpdateKernels);

                Substitutions kernelSubs(openclLFSRFunctions);
                kernelSubs.addVarSubstitution("t", "t");

                customUpdateKernels << "const unsigned int id = get_global_id(0);" << std::endl;
                customUpdateKernels << "// ------------------------------------------------------------------------" << std::endl;
                customUpdateKernels << "// Custom updates" << std::endl;
                genCustomUpdateKernel(customUpdateKernels, kernelSubs, modelMerged, g.first, customUpdateHandler, g.second.first);

                customUpdateKernels << "// ------------------------------------------------------------------------" << std::endl;
                customUpdateKernels << "// Custom WU updates" << std::endl;
                genCustomUpdateWUKernel(customUpdateKernels, kernelSubs, modelMerged, g.first, customWUUpdateHandler, g.second.first);
            }
        }

        // If there are any custom transpose weight updates in this group, generate kernel
        if(std::any_of(modelMerged.getMergedCustomUpdateTransposeWUGroups().cbegin(), modelMerged.getMergedCustomUpdateTransposeWUGroups().cend(),
                       [&g](const CustomUpdateTransposeWUGroupMerged &c) { return (c.getArchetype().getUpdateGroupName() == g.first); }))
        {
            customUpdateKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelCustomUpdate) << ", 8, 1)))" << std::endl;
            customUpdateKernels << "__kernel void " << KernelNames[KernelCustomTransposeUpdate] << g.first << "(";
            genMergedGroupKernelParams(customUpdateKernels, modelMerged.getMergedCustomUpdateTransposeWUGroups(), true);
            customUpdateKernels << model.getTimePrecision() << " t";
            if(shouldUseSubBufferAllocations()) {
                customUpdateKernels << ", __global void *d_staticBuffer"; 
            }
            customUpdateKernels << ")";
            {
                CodeStream::Scope b(customUpdateKernels);

                Substitutions kernelSubs(openclLFSRFunctions);
                kernelSubs.addVarSubstitution("t", "t");

                customUpdateKernels << "const unsigned int id = get_global_id(0); " << std::endl;

                customUpdateKernels << "// ------------------------------------------------------------------------" << std::endl;

                customUpdateKernels << "// Custom WU transpose updates" << std::endl;
                genCustomTransposeUpdateWUKernel(customUpdateKernels, kernelSubs, modelMerged, g.first, customWUTransposeUpdateHandler, g.second.second);
            }
        }

        os << "void update" << g.first<< "()";
        {
            CodeStream::Scope b(os);

            // Push any required EGPs
            pushEGPHandler(os);

            // If there are any custom update work-items
            if(g.second.first > 0) {
                CodeStream::Scope b(os);
                genKernelDimensions(os, KernelCustomUpdate, g.second.first, 1);
                const size_t numCustomUpdateGroups = (modelMerged.getMergedCustomUpdateGroups().size() +  modelMerged.getMergedCustomUpdateWUGroups().size());
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelCustomUpdate] << g.first << ".setArg(" << numCustomUpdateGroups << ", t));" << std::endl;
                os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelCustomUpdate] << g.first << ", cl::NullRange, globalWorkSize, localWorkSize";
                if(model.isTimingEnabled()) {
                    os << ", nullptr, &customUpdate" + g.first + "Event";
                }
                os << "));" << std::endl;
            }

            // If there are any custom transpose update work-items
            if(g.second.second > 0) {
                CodeStream::Scope b(os);
                genKernelDimensions(os, KernelCustomTransposeUpdate, g.second.second, 1, 8);
                const size_t numCustomTransposeUpdateGroups = modelMerged.getMergedCustomUpdateTransposeWUGroups().size();
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelCustomTransposeUpdate] << g.first << ".setArg(" << numCustomTransposeUpdateGroups << ", t));" << std::endl;
                os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelCustomTransposeUpdate] << g.first << ", cl::NullRange, globalWorkSize, localWorkSize";
                if(model.isTimingEnabled()) {
                    os << ", nullptr, &customUpdate" + g.first + "TransposeEvent";
                }
                os << "));" << std::endl;
            }

            if(model.isTimingEnabled()) {
                os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
                if(g.second.first > 0) {
                    CodeStream::Scope b(os);
                    genReadEventTiming(os, "customUpdate" + g.first);
                }
                
                if(g.second.second > 0) {
                    CodeStream::Scope b(os);
                    genReadEventTiming(os, "customUpdate" + g.first + "Transpose");
                }   
            }
        }

        os << std::endl;
    }

     // Write out kernel source string literal
    os << "const char* customUpdateSrc = ";
    divideKernelStreamInParts(os, customUpdateKernelsStream, 5000);
    os << ";" << std::endl;
    os << std::endl;

    os << "// Initialize the custom update kernel(s)" << std::endl;
    os << "void buildCustomUpdateProgram()";
    {
        CodeStream::Scope b(os);

        // If there are any custom update groups
        if(!customUpdateGroups.empty()) {
            os << "// Build program" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(customUpdateProgram = cl::Program(clContext, customUpdateSrc, false, &error));" << std::endl;
            genBuildProgramFlagsString(os);
            os << "if(customUpdateProgram.build(buildProgramFlags.c_str()) != CL_SUCCESS)";
            {
                CodeStream::Scope b(os);
                os << "std::cerr << customUpdateProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevice);" << std::endl;
                os << "throw std::runtime_error(\"Custom update program compile error\");" << std::endl;
            }
            os << std::endl;

            os << "// Configure merged struct building kernels" << std::endl;
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedCustomUpdateGroups(), "customUpdateProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedCustomUpdateWUGroups(), "customUpdateProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedCustomUpdateTransposeWUGroups(), "customUpdateProgram");
            os << std::endl;

            // Loop through custom update groups
            for(const auto &g : customUpdateGroups) {
                os << "// Configure custom update kernels" << std::endl;
                if(g.second.first > 0) {
                    os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelCustomUpdate] << g.first <<" = cl::Kernel(customUpdateProgram, \"" << KernelNames[KernelCustomUpdate] << g.first << "\", &error));" << std::endl;

                    size_t start = 0;
                    setMergedGroupKernelParams(os, KernelNames[KernelCustomUpdate] + g.first, modelMerged.getMergedCustomUpdateGroups(), start);
                    setMergedGroupKernelParams(os, KernelNames[KernelCustomUpdate] + g.first, modelMerged.getMergedCustomUpdateWUGroups(), start);
                    if(shouldUseSubBufferAllocations()) {
                        os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelCustomUpdate] << g.first << ".setArg(" << (start + 1) << ", d_staticBuffer));" << std::endl;
                    }
                }

                if(g.second.second) {
                    os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelCustomTransposeUpdate] << g.first <<" = cl::Kernel(customUpdateProgram, \"" << KernelNames[KernelCustomTransposeUpdate] << g.first << "\", &error));" << std::endl;

                    setMergedGroupKernelParams(os, KernelNames[KernelCustomTransposeUpdate] + g.first, modelMerged.getMergedCustomUpdateTransposeWUGroups());
                    if(shouldUseSubBufferAllocations()) {
                        os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelCustomTransposeUpdate] << g.first << ".setArg(" << (modelMerged.getMergedCustomUpdateTransposeWUGroups().size() + 1) << ", d_staticBuffer));" << std::endl;
                    }
                }
                os << std::endl;
            }
        }

    }
}
//--------------------------------------------------------------------------
void Backend::genInit(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces&,
                      HostHandler preambleHandler, NeuronInitGroupMergedHandler localNGHandler, CustomUpdateInitGroupMergedHandler cuHandler,
                      CustomWUUpdateDenseInitGroupMergedHandler cuDenseHandler, SynapseDenseInitGroupMergedHandler sgDenseInitHandler, 
                      SynapseConnectivityInitMergedGroupHandler sgSparseRowConnectHandler, SynapseConnectivityInitMergedGroupHandler sgSparseColConnectHandler, 
                      SynapseConnectivityInitMergedGroupHandler sgKernelInitHandler, SynapseSparseInitGroupMergedHandler sgSparseInitHandler, 
                      CustomWUUpdateSparseInitGroupMergedHandler cuSparseHandler, HostHandler initPushEGPHandler, HostHandler initSparsePushEGPHandler) const
{
    // Generate reset kernel to be run before the neuron kernel
    const ModelSpecInternal &model = modelMerged.getModel();

    os << "//--------------------------------------------------------------------------" << std::endl;
    os << "// OpenCL program and kernels" << std::endl;
    os << "//--------------------------------------------------------------------------" << std::endl;
    os << "cl::Program initializeProgram;" << std::endl;
    os << "cl::Kernel " << KernelNames[KernelInitialize] << ";" << std::endl;
    os << "cl::Kernel " << KernelNames[KernelInitializeSparse] << ";" << std::endl;
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedNeuronInitGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedCustomUpdateInitGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedCustomWUUpdateDenseInitGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedSynapseDenseInitGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedSynapseConnectivityInitGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedSynapseSparseInitGroups());
    genMergedStructPreamble(os, modelMerged, modelMerged.getMergedCustomWUUpdateSparseInitGroups());
    os << std::endl;

    // Generate preamble
    preambleHandler(os);

    // initialization kernel code
    size_t idInitStart = 0;

    //! KernelInitialize BODY START
    Substitutions kernelSubs(openclPhilloxFunctions);

    // Creating the kernel body separately so it can be split into multiple string literals
    std::stringstream initializeKernelsStream;
    CodeStream initializeKernels(initializeKernelsStream);

    // Include definitions
    genKernelPreamble(initializeKernels, modelMerged);  

    // Generate struct definitions
    modelMerged.genMergedNeuronInitGroupStructs(initializeKernels, *this);
    modelMerged.genMergedCustomUpdateInitGroupStructs(initializeKernels, *this);
    modelMerged.genMergedCustomWUUpdateDenseInitGroupStructs(initializeKernels, *this);
    modelMerged.genMergedSynapseDenseInitGroupStructs(initializeKernels, *this);
    modelMerged.genMergedSynapseConnectivityInitGroupStructs(initializeKernels, *this);
    modelMerged.genMergedSynapseSparseInitGroupStructs(initializeKernels, *this);
    modelMerged.genMergedCustomWUUpdateSparseInitGroupStructs(initializeKernels, *this);

    // Generate data structure for accessing merged groups from within initialisation kernel
    // **NOTE** pass in zero constant cache here as it's precious and would be wasted on init kernels which are only launched once
    genMergedKernelDataStructures(
        initializeKernels,
        modelMerged.getMergedNeuronInitGroups(), [this](const NeuronGroupInternal &ng) { return padKernelSize(ng.getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedCustomUpdateInitGroups(), [this](const CustomUpdateInternal &cg) { return padKernelSize(cg.getSize(), KernelInitialize); },
        modelMerged.getMergedCustomWUUpdateDenseInitGroups(), [this](const CustomUpdateWUInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedSynapseDenseInitGroups(), [this](const SynapseGroupInternal &sg) { return padKernelSize(sg.getTrgNeuronGroup()->getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedSynapseConnectivityInitGroups(), [this](const SynapseGroupInternal &sg) { return padKernelSize(getNumConnectivityInitThreads(sg), KernelInitialize); });

    // Generate data structure for accessing merged groups from within sparse initialisation kernel
    genMergedKernelDataStructures(
        initializeKernels,
        modelMerged.getMergedSynapseSparseInitGroups(), [this](const SynapseGroupInternal &sg) { return padKernelSize(sg.getMaxConnections(), KernelInitializeSparse); },
        modelMerged.getMergedCustomWUUpdateSparseInitGroups(), [this](const CustomUpdateWUInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getMaxConnections(), KernelInitializeSparse); });
    initializeKernels << std::endl;

    // Generate kernels used to populate merged structs
    genMergedStructBuildKernels(initializeKernels, modelMerged, modelMerged.getMergedNeuronInitGroups());
    genMergedStructBuildKernels(initializeKernels, modelMerged, modelMerged.getMergedCustomUpdateInitGroups());
    genMergedStructBuildKernels(initializeKernels, modelMerged, modelMerged.getMergedCustomWUUpdateDenseInitGroups());
    genMergedStructBuildKernels(initializeKernels, modelMerged, modelMerged.getMergedSynapseDenseInitGroups());
    genMergedStructBuildKernels(initializeKernels, modelMerged, modelMerged.getMergedSynapseConnectivityInitGroups());
    genMergedStructBuildKernels(initializeKernels, modelMerged, modelMerged.getMergedSynapseSparseInitGroups());
    genMergedStructBuildKernels(initializeKernels, modelMerged, modelMerged.getMergedCustomWUUpdateSparseInitGroups());

    initializeKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelInitialize) << ", 1, 1)))" << std::endl;
    initializeKernels << "__kernel void " << KernelNames[KernelInitialize] << "(";
    bool globalRNGRequired = isGlobalDeviceRNGRequired(modelMerged);
    bool additionalParamsRequired = globalRNGRequired || shouldUseSubBufferAllocations();
    const bool anyCustomUpdateInitGroups = !modelMerged.getMergedCustomUpdateInitGroups().empty();
    const bool anyCustomWUUpdateDenseInitGroups = !modelMerged.getMergedCustomWUUpdateDenseInitGroups().empty();
    const bool anyDenseInitGroups = !modelMerged.getMergedSynapseDenseInitGroups().empty();
    const bool anyConnectivityInitGroups = !modelMerged.getMergedSynapseConnectivityInitGroups().empty();
    genMergedGroupKernelParams(initializeKernels, modelMerged.getMergedNeuronInitGroups(), 
                               anyCustomUpdateInitGroups || anyCustomWUUpdateDenseInitGroups || anyDenseInitGroups || anyConnectivityInitGroups || additionalParamsRequired);
    genMergedGroupKernelParams(initializeKernels, modelMerged.getMergedCustomUpdateInitGroups(), 
                               anyCustomWUUpdateDenseInitGroups || anyDenseInitGroups || anyConnectivityInitGroups || additionalParamsRequired);
    genMergedGroupKernelParams(initializeKernels, modelMerged.getMergedCustomWUUpdateDenseInitGroups(), 
                               anyDenseInitGroups || anyConnectivityInitGroups || additionalParamsRequired);
    genMergedGroupKernelParams(initializeKernels, modelMerged.getMergedSynapseDenseInitGroups(), anyConnectivityInitGroups || additionalParamsRequired);
    genMergedGroupKernelParams(initializeKernels, modelMerged.getMergedSynapseConnectivityInitGroups(), additionalParamsRequired);
    if(globalRNGRequired) {
        initializeKernels << "__global clrngPhilox432HostStream *d_rng";
    }
    if(shouldUseSubBufferAllocations()) {
        if(globalRNGRequired) {
            initializeKernels << ", ";
        }
        initializeKernels << "__global void *d_staticBuffer"; 
    }
    initializeKernels << ")";
    {
        CodeStream::Scope b(initializeKernels);
        initializeKernels << "const unsigned int id = get_global_id(0);" << std::endl;
        genInitializeKernel(initializeKernels, kernelSubs, modelMerged, localNGHandler, cuHandler,
                            cuDenseHandler, sgDenseInitHandler, sgSparseRowConnectHandler, 
                            sgSparseColConnectHandler, sgKernelInitHandler, idInitStart);
    }
    const size_t numStaticInitThreads = idInitStart;

    // Generate sparse initialisation kernel
    size_t idSparseInitStart = 0;
    if(!modelMerged.getMergedSynapseSparseInitGroups().empty()) {
        initializeKernels << "__attribute__((reqd_work_group_size(" << getKernelBlockSize(KernelInitializeSparse) << ", 1, 1)))" << std::endl;
        initializeKernels << "__kernel void " << KernelNames[KernelInitializeSparse] << "(";
        const bool anyCustomWUUpdateSparseInitGroups = !modelMerged.getMergedCustomWUUpdateSparseInitGroups().empty();
        genMergedGroupKernelParams(initializeKernels, modelMerged.getMergedSynapseSparseInitGroups(), anyCustomWUUpdateSparseInitGroups || additionalParamsRequired);
        genMergedGroupKernelParams(initializeKernels, modelMerged.getMergedCustomWUUpdateSparseInitGroups(), additionalParamsRequired);
        if(globalRNGRequired) {
            initializeKernels << "__global clrngPhilox432HostStream *d_rng";
        }
        if(shouldUseSubBufferAllocations()) {
            if(globalRNGRequired) {
                initializeKernels << ", ";
            }

            initializeKernels << "__global void *d_staticBuffer"; 
        }
        initializeKernels << ")";
        {
            CodeStream::Scope b(initializeKernels);
            initializeKernels << "const unsigned int id = get_global_id(0);" << std::endl;
            genInitializeSparseKernel(initializeKernels, kernelSubs, modelMerged, 
                                      sgSparseInitHandler, cuSparseHandler, numStaticInitThreads, idSparseInitStart);
            os << std::endl;
        }
    }

    // Write out kernel source string literal
    os << "const char* initializeSrc = ";
    divideKernelStreamInParts(os, initializeKernelsStream, 5000);
    os << ";" << std::endl;
    os << std::endl;

    // Function for initializing the initialization kernels
    os << "// Initialize the initialization kernel(s)" << std::endl;
    os << "void buildInitializeProgram()";
    {
        CodeStream::Scope b(os);

        // If there are any kernels (some implementations complain)
        if(idInitStart > 0 || idSparseInitStart > 0) {
            os << "// Build program" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(initializeProgram = cl::Program(clContext, initializeSrc, false, &error));" << std::endl;
            genBuildProgramFlagsString(os);
            os << "if(initializeProgram.build(buildProgramFlags.c_str()) != CL_SUCCESS)";
            {
                CodeStream::Scope b(os);
                os << "std::cerr << initializeProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevice);" << std::endl;
                os << "throw std::runtime_error(\"Initialize program compile error\");" << std::endl;
            }
            os << std::endl;

            os << "// Configure merged struct building kernels" << std::endl;
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedNeuronInitGroups(), "initializeProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedCustomUpdateInitGroups(), "initializeProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedCustomWUUpdateDenseInitGroups(), "initializeProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedSynapseDenseInitGroups(), "initializeProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedSynapseConnectivityInitGroups(), "initializeProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedSynapseSparseInitGroups(), "initializeProgram");
            genMergedStructBuild(os, modelMerged, modelMerged.getMergedCustomWUUpdateSparseInitGroups(), "initializeProgram");
            os << std::endl;
        }

        if (idInitStart > 0) {
            os << "// Configure initialization kernel" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelInitialize] << " = cl::Kernel(initializeProgram, \"" << KernelNames[KernelInitialize] << "\", &error));" << std::endl;
            size_t start = 0;
            setMergedGroupKernelParams(os, KernelNames[KernelInitialize], modelMerged.getMergedNeuronInitGroups(), start);
            setMergedGroupKernelParams(os, KernelNames[KernelInitialize], modelMerged.getMergedCustomUpdateInitGroups(), start);
            setMergedGroupKernelParams(os, KernelNames[KernelInitialize], modelMerged.getMergedCustomWUUpdateDenseInitGroups(), start);
            setMergedGroupKernelParams(os, KernelNames[KernelInitialize], modelMerged.getMergedSynapseDenseInitGroups(), start);
            setMergedGroupKernelParams(os, KernelNames[KernelInitialize], modelMerged.getMergedSynapseConnectivityInitGroups(), start);
            if(shouldUseSubBufferAllocations()) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelInitialize] << ".setArg(" << (start + (globalRNGRequired ? 1 : 0)) << ", d_staticBuffer));" << std::endl;
            }
            os << std::endl;
        }

        if(idSparseInitStart > 0) {
            os << "// Configure sparse initialization kernel" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << KernelNames[KernelInitializeSparse] << " = cl::Kernel(initializeProgram, \"" << KernelNames[KernelInitializeSparse] << "\", &error));" << std::endl;
            size_t start = 0;
            setMergedGroupKernelParams(os, KernelNames[KernelInitializeSparse], modelMerged.getMergedSynapseSparseInitGroups(), start);
            setMergedGroupKernelParams(os, KernelNames[KernelInitializeSparse], modelMerged.getMergedCustomWUUpdateSparseInitGroups(), start);
            if(shouldUseSubBufferAllocations()) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelInitializeSparse] << ".setArg(" << (start + (globalRNGRequired ? 1 : 0)) << ", d_staticBuffer));" << std::endl;
            }
            os << std::endl;
        }
    }

    os << std::endl;

    os << "void initialize()";
    {
        CodeStream::Scope b(os);

        // Loop through all synapse groups
        // **TODO** this logic belongs in BackendSIMT
        // **TODO** apply merging to this process - large models could generate thousands of lines of code here
        for(const auto &s : model.getSynapseGroups()) {
            // If this synapse population has BITMASK connectivity and is intialised on device, enqueue a buffer fill operation to zero the whole bitmask
            if(s.second.isSparseConnectivityInitRequired() && s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                const size_t gpSize = ceilDivide((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * getSynapticMatrixRowStride(s.second), 32);
                os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueFillBuffer<uint32_t>(d_gp" << s.first << ", 0, 0, " << gpSize << " * sizeof(uint32_t)));" << std::endl;
            }
            
            // If this synapse population has SPARSE connectivity and column-based on device connectivity, enqueue a buffer fill operation to zero row lengths
            // **NOTE** we could also use this code path for row-based connectivity but, leaving this in the kernel is much better as it gets merged
            if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE && !s.second.getConnectivityInitialiser().getSnippet()->getColBuildCode().empty()
               && !s.second.isWeightSharingSlave())
            {
                os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueFillBuffer<unsigned int>(d_rowLength" << s.first << ", 0, 0, " << s.second.getSrcNeuronGroup()->getNumNeurons() << " * sizeof(unsigned int)));" << std::endl;
            }

            // If this synapse population has SPARSE connectivity and has postsynaptic learning, enqueue a buffer fill operation to zero column lengths
            if((s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getLearnPostCode().empty()) {
                os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueFillBuffer<unsigned int>(d_colLength" << s.first << ", 0, 0, " << s.second.getTrgNeuronGroup()->getNumNeurons() << " * sizeof(unsigned int)));" << std::endl;
            }
        }
        os << std::endl;

        // Push any required EGPs
        initPushEGPHandler(os);

        // If there are any initialisation work-items
        if (idInitStart > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelInitialize, idInitStart, 1);
            if(globalRNGRequired) {
                const size_t numInitGroups = (modelMerged.getMergedNeuronInitGroups().size() + modelMerged.getMergedCustomUpdateInitGroups().size() +
                                              modelMerged.getMergedCustomWUUpdateDenseInitGroups().size() + modelMerged.getMergedSynapseDenseInitGroups().size() +
                                              modelMerged.getMergedSynapseConnectivityInitGroups().size());

                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelInitialize] << ".setArg(" << numInitGroups << ", d_rng));" << std::endl;
            }
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelInitialize] << ", cl::NullRange, globalWorkSize, localWorkSize";
            if(model.isTimingEnabled()) {
                os << ", nullptr, &initEvent";
            }
            os << "));" << std::endl;

            if(model.isTimingEnabled()) {
                os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
                genReadEventTiming(os, "init");
            }
        }
    }

    os << std::endl;

    // Generating code for initializing all OpenCL elements - Using intializeSparse
    os << "// Initialize all OpenCL elements" << std::endl;
    os << "void initializeSparse()";
    {
        CodeStream::Scope b(os);

        // Push any required EGPs
        initSparsePushEGPHandler(os);

        // Copy all uninitialised state variables to device
        os << "copyStateToDevice(true);" << std::endl;
        os << "copyConnectivityToDevice(true);" << std::endl;

        // If there are any sparse initialisation work-items
        if (idSparseInitStart > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelInitializeSparse, idSparseInitStart, 1);
            if(globalRNGRequired) {
                const size_t numInitGroups = (modelMerged.getMergedSynapseSparseInitGroups().size() +  modelMerged.getMergedCustomWUUpdateSparseInitGroups().size());
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelInitializeSparse] << ".setArg(" << numInitGroups << ", d_rng));" << std::endl;
            }
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelInitializeSparse] << ", cl::NullRange, globalWorkSize, localWorkSize";
            if(model.isTimingEnabled()) {
                os << ", nullptr, &initSparseEvent";
            }
            os << "));" << std::endl;

            if(model.isTimingEnabled()) {
                os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
                genReadEventTiming(os, "initSparse");
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    // If any neuron groups require support code, raise error
    if(std::any_of(modelMerged.getModel().getNeuronGroups().cbegin(), modelMerged.getModel().getNeuronGroups().cbegin(),
                   [](const ModelSpec::NeuronGroupValueType &ng){ return !ng.second.getNeuronModel()->getSupportCode().empty(); }))
    {
        throw std::runtime_error("OpenCL backend does not currently support models which use support code.");
    }

     // If any synapse groups require support code, raise error
    if(std::any_of(modelMerged.getModel().getSynapseGroups().cbegin(), modelMerged.getModel().getSynapseGroups().cbegin(),
                   [](const ModelSpec::SynapseGroupValueType &sg)
                   {
                       const auto *wum = sg.second.getWUModel();
                       const auto *psm = sg.second.getPSModel();
                       return (!wum->getSimSupportCode().empty() || !wum->getLearnPostSupportCode().empty()
                               || !wum->getSynapseDynamicsSuppportCode().empty() || !psm->getSupportCode().empty());
                   }))
    {
        throw std::runtime_error("OpenCL backend does not currently support models which use support code.");
    }
    // If any synapse groups require procedural weights or connectivity, raise error
    if(std::any_of(modelMerged.getModel().getSynapseGroups().cbegin(), modelMerged.getModel().getSynapseGroups().cbegin(), 
                   [](const ModelSpec::SynapseGroupValueType &sg) 
                   { 
                       return ((sg.second.getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL)
                               || (sg.second.getMatrixType() & SynapseMatrixWeight::PROCEDURAL));
                   }))
    {
        throw std::runtime_error("OpenCL backend does not currently support procedural weights or connectivity.");
    }


    os << "// Standard C++ includes" << std::endl;
    os << "#include <iostream>" << std::endl;
    os << "#include <random>" << std::endl;
    os << "#include <string>" << std::endl;
    os << "#include <stdexcept>" << std::endl;
    os << std::endl;
    os << "// Standard C includes" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << "#include <cassert>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged&) const
{
#ifdef _WIN32
    os << "#pragma warning(disable: 4297)" << std::endl;
#endif
    os << "// OpenCL includes" << std::endl;
    os << "#include \"opencl/cl2.hpp\"" << std::endl;
    os << "#include <clRNG/lfsr113.h>" << std::endl;
    os << "#include <clRNG/philox432.h>" << std::endl;
    os << std::endl;

    os << std::endl;
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper macro for error-checking OpenCL calls" << std::endl;
    os << "#define CHECK_OPENCL_ERRORS(call) {\\" << std::endl;
    os << "    cl_int error = call;\\" << std::endl;
    os << "    if (error != CL_SUCCESS) {\\" << std::endl;
    os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": opencl error \" + std::to_string(error) + \": \" + clGetErrorString(error));\\" << std::endl;
    os << "    }\\" << std::endl;
    os << "}" << std::endl;
    os << std::endl;
    os << "#define CHECK_OPENCL_ERRORS_POINTER(call) {\\" << std::endl;
    os << "    cl_int error;\\" << std::endl;
    os << "    call;\\" << std::endl;
    os << "    if (error != CL_SUCCESS) {\\" << std::endl;
    os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": opencl error \" + std::to_string(error) + \": \" + clGetErrorString(error));\\" << std::endl;
    os << "    }\\" << std::endl;
    os <<"}" << std::endl;

    // Declaration of OpenCL functions
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// OpenCL functions declaration" << std::endl;
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "const char* clGetErrorString(cl_int error);" << std::endl;
    os << "std::string getAbsoluteCodePath();" << std::endl;
    os << std::endl;

    // Declaration of OpenCL variables
    os << "// OpenCL variables" << std::endl;
    os << "extern cl::Context clContext;" << std::endl;
    os << "extern cl::Device clDevice;" << std::endl;
    os << "extern cl::CommandQueue commandQueue;" << std::endl;
    if(shouldUseSubBufferAllocations()) {
        os << "extern cl::Buffer d_staticBuffer;" << std::endl;
    }
    os << std::endl;

    os << "// OpenCL program initialization functions" << std::endl;
    os << "void buildInitializeProgram();" << std::endl;
    os << "void buildNeuronUpdateProgram();" << std::endl;
    os << "void buildSynapseUpdateProgram();" << std::endl;
    os << "void buildCustomUpdateProgram();" << std::endl;
    
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const
{
#ifdef _WIN32
    os << "#include <windows.h>" << std::endl;
#else
    os << "#include <dlfcn.h>" << std::endl;
#ifdef __APPLE__
    os << "#include <sys/syslimits.h>" << std::endl;
#else
    os << "#include <linux/limits.h>" << std::endl;
#endif
#endif
    os << std::endl;

    // Generating OpenCL variables for the runner
    os << "// OpenCL variables" << std::endl;
    os << "cl::Context clContext;" << std::endl;
    os << "cl::Device clDevice;" << std::endl;
    os << "cl::CommandQueue commandQueue;" << std::endl;
    if(shouldUseSubBufferAllocations()) {
        os << "cl::Buffer d_staticBuffer;" << std::endl;
        os << "size_t dynamicAllocationOffset = " << memAlloc.getDeviceBytes() << ";" << std::endl;
    }
    os << std::endl;

    os << "// Get OpenCL error as string" << std::endl;
    os << "const char* clGetErrorString(cl_int error)";
    {
        CodeStream::Scope b(os);
        os << "switch(error)";
        {
            CodeStream::Scope b(os);

            #define STRINGIFY(ERR) #ERR
            #define GEN_CL_ERROR_CASE(ERR) os << "case " STRINGIFY(ERR) ": return \"" #ERR << "\";" << std::endl
            
            // run-time and JIT compiler errors
            GEN_CL_ERROR_CASE(CL_SUCCESS);
            GEN_CL_ERROR_CASE(CL_DEVICE_NOT_FOUND);
            GEN_CL_ERROR_CASE(CL_DEVICE_NOT_AVAILABLE);
            GEN_CL_ERROR_CASE(CL_COMPILER_NOT_AVAILABLE);
            GEN_CL_ERROR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
            GEN_CL_ERROR_CASE(CL_OUT_OF_RESOURCES);
            GEN_CL_ERROR_CASE(CL_OUT_OF_HOST_MEMORY);
            GEN_CL_ERROR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
            GEN_CL_ERROR_CASE(CL_MEM_COPY_OVERLAP);
            GEN_CL_ERROR_CASE(CL_IMAGE_FORMAT_MISMATCH);
            GEN_CL_ERROR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
            GEN_CL_ERROR_CASE(CL_BUILD_PROGRAM_FAILURE);
            GEN_CL_ERROR_CASE(CL_MAP_FAILURE);
            GEN_CL_ERROR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
            GEN_CL_ERROR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
            GEN_CL_ERROR_CASE(CL_COMPILE_PROGRAM_FAILURE);
            GEN_CL_ERROR_CASE(CL_LINKER_NOT_AVAILABLE);
            GEN_CL_ERROR_CASE(CL_LINK_PROGRAM_FAILURE);
            GEN_CL_ERROR_CASE(CL_DEVICE_PARTITION_FAILED);
            GEN_CL_ERROR_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);

            // compile-time errors
            GEN_CL_ERROR_CASE(CL_INVALID_VALUE);
            GEN_CL_ERROR_CASE(CL_INVALID_DEVICE_TYPE);
            GEN_CL_ERROR_CASE(CL_INVALID_PLATFORM);
            GEN_CL_ERROR_CASE(CL_INVALID_DEVICE);
            GEN_CL_ERROR_CASE(CL_INVALID_CONTEXT);
            GEN_CL_ERROR_CASE(CL_INVALID_QUEUE_PROPERTIES);
            GEN_CL_ERROR_CASE(CL_INVALID_COMMAND_QUEUE);
            GEN_CL_ERROR_CASE(CL_INVALID_HOST_PTR);
            GEN_CL_ERROR_CASE(CL_INVALID_MEM_OBJECT);
            GEN_CL_ERROR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
            GEN_CL_ERROR_CASE(CL_INVALID_IMAGE_SIZE);
            GEN_CL_ERROR_CASE(CL_INVALID_SAMPLER);
            GEN_CL_ERROR_CASE(CL_INVALID_BINARY);
            GEN_CL_ERROR_CASE(CL_INVALID_BUILD_OPTIONS);
            GEN_CL_ERROR_CASE(CL_INVALID_PROGRAM);
            GEN_CL_ERROR_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
            GEN_CL_ERROR_CASE(CL_INVALID_KERNEL_NAME);
            GEN_CL_ERROR_CASE(CL_INVALID_KERNEL_DEFINITION);
            GEN_CL_ERROR_CASE(CL_INVALID_KERNEL);
            GEN_CL_ERROR_CASE(CL_INVALID_ARG_INDEX);
            GEN_CL_ERROR_CASE(CL_INVALID_ARG_VALUE);
            GEN_CL_ERROR_CASE(CL_INVALID_ARG_SIZE);
            GEN_CL_ERROR_CASE(CL_INVALID_KERNEL_ARGS);
            GEN_CL_ERROR_CASE(CL_INVALID_WORK_DIMENSION);
            GEN_CL_ERROR_CASE(CL_INVALID_WORK_GROUP_SIZE);
            GEN_CL_ERROR_CASE(CL_INVALID_WORK_ITEM_SIZE);
            GEN_CL_ERROR_CASE(CL_INVALID_GLOBAL_OFFSET);
            GEN_CL_ERROR_CASE(CL_INVALID_EVENT_WAIT_LIST);
            GEN_CL_ERROR_CASE(CL_INVALID_EVENT);
            GEN_CL_ERROR_CASE(CL_INVALID_OPERATION);
            GEN_CL_ERROR_CASE(CL_INVALID_GL_OBJECT);
            GEN_CL_ERROR_CASE(CL_INVALID_BUFFER_SIZE);
            GEN_CL_ERROR_CASE(CL_INVALID_MIP_LEVEL);
            GEN_CL_ERROR_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
            GEN_CL_ERROR_CASE(CL_INVALID_PROPERTY);
            GEN_CL_ERROR_CASE(CL_INVALID_IMAGE_DESCRIPTOR);
            GEN_CL_ERROR_CASE(CL_INVALID_COMPILER_OPTIONS);
            GEN_CL_ERROR_CASE(CL_INVALID_LINKER_OPTIONS);
            GEN_CL_ERROR_CASE(CL_INVALID_DEVICE_PARTITION_COUNT);
            os << "default: return \"Unknown OpenCL error\";" << std::endl;
            
            #undef GEN_CL_ERROR_CASE
            #undef STRINGIFY
        }
    }
    os << std::endl;

    os << "// Get OpenCL error as string" << std::endl;
    os << "std::string getAbsoluteCodePath()";
    {
        CodeStream::Scope b(os);
#ifdef _WIN32
        // Find the module i.e. dll within which this function appears
        os << "HMODULE hm = NULL;" << std::endl;
        os << "if(GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT," << std::endl;
        os << "                     (LPCSTR)&getAbsoluteCodePath, &hm) == 0)";
        {
            CodeStream::Scope b(os);
            os << "throw std::runtime_error(\"GetModuleHandle failed with error:\" + std::to_string(GetLastError()));" << std::endl;
        }

        // Get the path to this module
        os << "char libraryPathRaw[MAX_PATH];" << std::endl;
        os << "if(GetModuleFileName(hm, libraryPathRaw, MAX_PATH) == 0)";
        {
            CodeStream::Scope b(os);
            os << "throw std::runtime_error(\"GetModuleFileName failed with error:\" + std::to_string(GetLastError()));" << std::endl;
        }

        // Convert library path to std::string
        os << "std::string libraryPath(libraryPathRaw);" << std::endl;

        // Replace backslashes with slashes
        os << "std::replace(libraryPath.begin(), libraryPath.end(), '\\\\', '/');" << std::endl;

        // Remove library filename from end of library path
        os << "libraryPath = libraryPath.substr(0, libraryPath.find_last_of('/'));" << std::endl;

        // Return code directory
        os << "return libraryPath + \"/" << modelMerged.getModel().getName() << "_CODE\";" << std::endl;
#else
        // Suppress unused parameters
        (void)(modelMerged);

        // Get info about dynamic library this function is in
        os << "Dl_info dlInfo;" << std::endl;
        os << "if(dladdr((void*)&getAbsoluteCodePath, &dlInfo) == 0)";
        {
            CodeStream::Scope b(os);
            os << "throw std::runtime_error(\"dladdr failed with error:\" + std::string(dlerror()));" << std::endl;
        }

        os << "char libraryPathRaw[PATH_MAX];" << std::endl;
        os << "if(realpath(dlInfo.dli_fname, libraryPathRaw) == nullptr)";
        {
            CodeStream::Scope b(os);
            os << "throw std::runtime_error(\"realpath failed with error:\" + std::to_string(errno));" << std::endl;
        }
        // Convert library path to std::string
        os << "std::string libraryPath(libraryPathRaw);" << std::endl;

        // Remove library filename from end of library path and return
        os << "return libraryPath.substr(0, libraryPath.find_last_of('/'));" << std::endl;
#endif
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const
{
    // Initializing OpenCL programs
    os << "// Get platforms" << std::endl;
    os << "std::vector<cl::Platform> platforms; " << std::endl;
    os << "cl::Platform::get(&platforms);" << std::endl;
    os << std::endl;

    os << "// Get platform devices" << std::endl;
    os << "std::vector<cl::Device> platformDevices; " << std::endl;
    os << "platforms[" << m_ChosenPlatformIndex << "].getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);" << std::endl;
    os << std::endl;

    os << "// Select device and create context and command queue" << std::endl;
    os << "clDevice = platformDevices[" << m_ChosenDeviceIndex << "];" << std::endl;
    os << "clContext = cl::Context(clDevice);" << std::endl;
    os << "commandQueue = cl::CommandQueue(clContext, clDevice, ";
    os << (modelMerged.getModel().isTimingEnabled() ? "CL_QUEUE_PROFILING_ENABLE" : "0") << ");" << std::endl;
    // Create static buffer, within which all other device allocations are created as sub-buffers
    if(shouldUseSubBufferAllocations()) {
        os << "CHECK_OPENCL_ERRORS_POINTER(d_staticBuffer = cl::Buffer(clContext, CL_MEM_READ_WRITE, size_t{";
        
        // If model has any pointer EGPs, allocate all allocatable memory
        if(modelMerged.anyPointerEGPs()) {
            os << m_ChosenDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        }
        // Otherwise, static buffer only needs to be large enough for all static allocations
        else {
            os << memAlloc.getDeviceBytes();
        }
        os << "}, nullptr, &error));" << std::endl;
    }
    os << std::endl;

    os << "// Build OpenCL programs" << std::endl;
    os << "buildInitializeProgram();" << std::endl;
    os << "buildNeuronUpdateProgram();" << std::endl;
    os << "buildSynapseUpdateProgram();" << std::endl;
    os << "buildCustomUpdateProgram();" << std::endl;
    os << std::endl;

    // If any neuron groups require a simulation RNG
    const ModelSpecInternal &model = modelMerged.getModel();
    if(std::any_of(model.getNeuronGroups().cbegin(), model.getNeuronGroups().cend(),
                   [](const ModelSpec::NeuronGroupValueType &n) { return n.second.isSimRNGRequired(); }))
    {
        os << "// Seed LFSR113 RNG" << std::endl;
        os << "clrngLfsr113StreamCreator *lfsrStreamCreator = clrngLfsr113CopyStreamCreator(nullptr, nullptr);" << std::endl;
        {
            // If no seed is specified, use system randomness to configure base state
            const long minSeedValues[4] = {2, 8, 16, 128};
            CodeStream::Scope b(os);
            os << "clrngLfsr113StreamState lfsrBaseState;" << std::endl;
            if(model.getSeed() == 0) {
                os << "std::random_device seedSource;" << std::endl;
                for(size_t i = 0; i < 4; i++) {
                    os << "do{ lfsrBaseState.g[" << i << "] = seedSource(); } while(lfsrBaseState.g[" << i << "] < " << minSeedValues[i] << ");" << std::endl;
                }
            }
            // Otherwise
            else {
                const long IA = 16807l;
                const long IM = 2147483647l;
                const long IQ = 127773l;
                const long IR = 2836l;

                // Initially 'smear' seed out across 4 words of state
                // using procedure from http ://web.mst.edu/~vojtat/class_5403/lfsr113/lfsr113.cpp
                long idum = std::max(1l, (long)model.getSeed());
                uint32_t g[4];
                for(size_t i = 0; i < 4; i++) {
                    const long k = idum / IQ;
                    idum = IA * (idum - k * IQ) - IR * k;
                    if(idum < 0) {
                        idum += IM;
                    }
                    g[i] = (idum < minSeedValues[i]) ? (idum + minSeedValues[i]) : idum;
                }         
                // Perform single round of LFSR113 to improve seed
                uint32_t b = ((g[0] << 6) ^ g[0]) >> 13;
                g[0] = ((g[0] & 4294967294U) << 18) ^ b;
                b = ((g[1] << 2) ^ g[1]) >> 27;
                g[1] = ((g[1] & 4294967288U) << 2) ^ b;
                b = ((g[2] << 13) ^ g[2]) >> 21;
                g[2] = ((g[2] & 4294967280U) << 7) ^ b;
                b = ((g[3] << 3) ^ g[3]) >> 12;
                g[3] = ((g[3] & 4294967168U) << 13) ^ b;
                
                // Write out state
                for(size_t i = 0; i < 4; i++) {
                    os << "lfsrBaseState.g[" << i << "] = " << g[i] << "u;" << std::endl;
                }
            }

            // Configure stream creator
            os << "clrngLfsr113SetBaseCreatorState(lfsrStreamCreator, &lfsrBaseState);" << std::endl;
        }
    }
    
    // If a global device RNG is required
    if(isGlobalDeviceRNGRequired(modelMerged)) {
        os << "// Seed Philox RNG" << std::endl;
        os << "clrngPhilox432StreamCreator *philoxStreamCreator = clrngPhilox432CopyStreamCreator(nullptr, nullptr);" << std::endl;
        {
            CodeStream::Scope b(os);
            
            // If no seed is specified, get system random device
            if(model.getSeed() == 0) {
                os << "std::random_device seedSource;" << std::endl;
            }

            // Define Philox base state
            os << "clrngPhilox432StreamState philoxBaseState = ";
            {
                CodeStream::Scope b(os);
                if(model.getSeed() == 0) {
                    os << "{seedSource(), seedSource()},    // key" << std::endl;
                }
                // Otherwise use model seed as first key word
                else {
                    os << "{" << model.getSeed() << ", 0},  // key" << std::endl;
                }

                // Zero counter, deck and deck index
                os << "{{0, 0}, {0, 0}},                // ctr" << std::endl;
                os << "{0, 0, 0, 0},                    // deck" << std::endl;
                os << "0                                // deck index" << std::endl;
            }
            os << ";" << std::endl;

            // Configure stream creator
            os << "clrngPhilox432SetBaseCreatorState(philoxStreamCreator, &philoxBaseState);" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    // If timing is enabled, synchronise 
    // **THINK** is it better to wait on events?
    if(modelMerged.getModel().isTimingEnabled() || modelMerged.getModel().zeroCopyInUse()) {
        os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const
{
    const bool deviceType = isDeviceType(type);

    if (loc & VarLocation::HOST) {
        if (deviceType) {
            throw std::runtime_error("Variable '" + name + "' is of device-only type '" + type + "' but is located on the host");
        }
        definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;

        if(!(loc & VarLocation::ZERO_COPY)) {
            definitionsInternal << "EXPORT_VAR cl::Buffer h_" << name << ";" << std::endl;
        }
    }
    if (loc & VarLocation::DEVICE) {
        definitionsInternal << "EXPORT_VAR cl::Buffer d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    if (loc & VarLocation::HOST) {
        os << type << " " << name << ";" << std::endl;

        if(!(loc & VarLocation::ZERO_COPY)) {
            os << "cl::Buffer h_" << name << ";" << std::endl;
        }
    }
    if (loc & VarLocation::DEVICE) {
        os << "cl::Buffer d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count, MemAlloc &memAlloc) const
{
    // If variable is present on device then initialize the device buffer
    if (loc & VarLocation::DEVICE) {
        CodeStream::Scope b(os);

        const size_t sizeBytes = count * getSize(type);
        if(shouldUseSubBufferAllocations()) {
            // Check zero-copy isn't in use
            assert(!(loc & VarLocation::ZERO_COPY));

            // Apply minimum alignement to size
            const size_t alignedSizeBytes = padSize(sizeBytes, m_AllocationAlignementBytes);

            // Create region struct defining location of this variable
            os << "const cl_buffer_region region{size_t{" << memAlloc.getDeviceBytes() << "}, size_t{" << alignedSizeBytes << "}};" << std::endl;

            // Create sub-buffer encapsulating this region within main static buffer
            os << "CHECK_OPENCL_ERRORS_POINTER(d_" << name << " = d_staticBuffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &error));" << std::endl;

            // Add aligned size to mem allocation tracker
            memAlloc += MemAlloc::device(alignedSizeBytes);
        }
        else {
            os << "CHECK_OPENCL_ERRORS_POINTER(d_" << name << " = cl::Buffer(clContext, CL_MEM_READ_WRITE";
            if(loc & VarLocation::ZERO_COPY) {
                os << " | CL_MEM_ALLOC_HOST_PTR";
            }
            os << ", " << count << " * sizeof(" << type << "), nullptr, &error));" << std::endl;
            memAlloc += MemAlloc::device(sizeBytes);
        }
    }

    if(loc & VarLocation::HOST) {
        if(loc & VarLocation::ZERO_COPY) {
            os << "CHECK_OPENCL_ERRORS_POINTER(" << name << " = (" << type << "*)commandQueue.enqueueMapBuffer(d_" << name;
            os << ", CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, " << count << " * sizeof(" << type << "), nullptr, nullptr, &error));" << std::endl;
        }
        else {
            os << "CHECK_OPENCL_ERRORS_POINTER(h_" << name << " = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, " << count << " * sizeof(" << type << "), nullptr, &error));" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << name << " = (" << type << "*)commandQueue.enqueueMapBuffer(h_" << name;
            os << ", CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, " << count << " * sizeof(" << type << "), nullptr, nullptr, &error));" << std::endl;
            memAlloc += MemAlloc::host(count * getSize(type));
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const
{
    if(loc & VarLocation::ZERO_COPY) {
        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueUnmapMemObject(d_" << name << ", " << name << "));" << std::endl;
    }
    else if(loc & VarLocation::HOST) {
        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueUnmapMemObject(h_" << name << ", " << name << "));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamDefinition(CodeStream &definitions, CodeStream &definitionsInternal, 
                                            const std::string &type, const std::string &name, VarLocation loc) const
{
    if (loc & VarLocation::HOST) {
        definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
        definitionsInternal << "EXPORT_VAR cl::Buffer h_" << name << ";" << std::endl;
    }
    if (loc & VarLocation::DEVICE && ::Utils::isTypePointer(type)) {
        definitionsInternal << "EXPORT_VAR cl::Buffer d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    if (loc & VarLocation::HOST) {
        os << type << " " << name << ";" << std::endl;
        os << "cl::Buffer h_" << name << ";" << std::endl;
    }
    if (loc & VarLocation::DEVICE && ::Utils::isTypePointer(type)) {
        os << "cl::Buffer d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name,
                                            VarLocation loc, const std::string &countVarName, const std::string &prefix) const
{
    // Get underlying type
    const std::string underlyingType = ::Utils::getUnderlyingType(type);
    const bool pointerToPointer = ::Utils::isTypePointerToPointer(type);

    const std::string hostPointer = pointerToPointer ? ("*" + prefix + name) : (prefix + name);
    const std::string deviceBuffer = pointerToPointer ? ("*" + prefix + "d_" + name) : (prefix + "d_" + name);
    const std::string hostBuffer = pointerToPointer ? ("*" + prefix + "h_" + name) : (prefix + "h_" + name);

    // If variable is present on device at all
    if(loc & VarLocation::DEVICE) {
        if(shouldUseSubBufferAllocations()) {
            // Check zero-copy isn't in use
            assert(!(loc & VarLocation::ZERO_COPY));

            CodeStream::Scope b(os);
            os << "const unsigned int sizeBytes = " << countVarName << " * sizeof(" << underlyingType << ");" << std::endl;
            os << "const unsigned int alignedSizeBytes = ((sizeBytes + " << m_AllocationAlignementBytes - 1 << ") / " << m_AllocationAlignementBytes << ") * " << m_AllocationAlignementBytes << ";" << std::endl;

            // Create region struct defining location of this variable
            os << "const cl_buffer_region region{size_t{dynamicAllocationOffset}, size_t{alignedSizeBytes}};" << std::endl;

            // Create sub-buffer encapsulating this region within main static buffer
            os << "CHECK_OPENCL_ERRORS_POINTER(" << deviceBuffer << " = d_staticBuffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &error));" << std::endl;

            // Advance dynamic allocation offset
            os << "dynamicAllocationOffset += alignedSizeBytes;" << std::endl;
        }
        else {
            os << "CHECK_OPENCL_ERRORS_POINTER(" << deviceBuffer << " = cl::Buffer(clContext, CL_MEM_READ_WRITE";
            if(loc & VarLocation::ZERO_COPY) {
                os << " | CL_MEM_ALLOC_HOST_PTR";
            }
            os << ", " << countVarName << " * sizeof(" << underlyingType << "), nullptr, &error));" << std::endl;
        }
    }

    if(loc & VarLocation::HOST) {
        if(loc & VarLocation::ZERO_COPY) {
            os << "CHECK_OPENCL_ERRORS_POINTER(" << hostPointer << " = (" << underlyingType << "*)commandQueue.enqueueMapBuffer(" << deviceBuffer;
            os << ", CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, " << countVarName << " * sizeof(" << underlyingType << "), nullptr, nullptr, &error));" << std::endl;
        }
        else {
            os << "CHECK_OPENCL_ERRORS_POINTER(" << hostBuffer << " = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, " << countVarName << " * sizeof(" << underlyingType << "), nullptr, &error));" << std::endl;
            os << "CHECK_OPENCL_ERRORS_POINTER(" << hostPointer << " = (" << underlyingType << "*)commandQueue.enqueueMapBuffer(" << hostBuffer;
            os << ", CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, " << countVarName << " * sizeof(" << underlyingType << "), nullptr, nullptr, &error));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name,
                                      VarLocation loc, const std::string &countVarName, const std::string &prefix) const
{
    assert(!getPreferences().automaticCopy);

    // Get underlying type
    const std::string underlyingType = ::Utils::getUnderlyingType(type);
    const bool pointerToPointer = ::Utils::isTypePointerToPointer(type);

    const std::string hostPointer = pointerToPointer ? ("*" + prefix + name) : (prefix + name);
    const std::string devicePointer = pointerToPointer ? ("*" + prefix + "d_" + name) : (prefix + "d_" + name);

    if (!(loc & VarLocation::ZERO_COPY)) {
        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(" << devicePointer;
        os << ", CL_TRUE";
        os << ", 0";
        os << ", " << countVarName << " * sizeof(" << underlyingType << ")";
        os << ", " << hostPointer << "));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPull(CodeStream &os, const std::string &type, const std::string &name,
                                      VarLocation loc, const std::string &countVarName, const std::string &prefix) const
{
    assert(!getPreferences().automaticCopy);

    // Get underlying type
    const std::string underlyingType = ::Utils::getUnderlyingType(type);
    const bool pointerToPointer = ::Utils::isTypePointerToPointer(type);

    const std::string hostPointer = pointerToPointer ? ("*" + prefix + name) : (prefix + name);
    const std::string devicePointer = pointerToPointer ? ("*" + prefix + "d_" + name) : (prefix + "d_" + name);

    if (!(loc & VarLocation::ZERO_COPY)) {
        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(" << devicePointer;
        os << ", " << "CL_TRUE";
        os << ", " << "0";
        os << ", " << countVarName << " * sizeof(" << underlyingType << ")";
        os << ", " << hostPointer << "));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genMergedExtraGlobalParamPush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx,
                                            const std::string &groupIdx, const std::string &fieldName,
                                            const std::string &egpName) const
{
    CodeStream::Scope b(os);
    const std::string kernelName = "setMerged" + suffix + std::to_string(mergedGroupIdx) + fieldName + "Kernel";
    os << "CHECK_OPENCL_ERRORS(" << kernelName << ".setArg(1, " << groupIdx << "));" << std::endl;
    os << "CHECK_OPENCL_ERRORS(" << kernelName << ".setArg(2, " << egpName << "));" << std::endl;
    os << "const cl::NDRange globalWorkSize(1, 1);" << std::endl;
    os << "const cl::NDRange localWorkSize(1, 1);" << std::endl;
    os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << kernelName << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
}
//--------------------------------------------------------------------------
std::string Backend::getMergedGroupFieldHostType(const std::string &type) const
{
    // If type is a pointer, on the host it is represented by an OpenCL buffer
    if(::Utils::isTypePointerToPointer(type)) {
        return "cl::Buffer*";
    }
    if(::Utils::isTypePointer(type)) {
        return "cl::Buffer";
    }
    // Otherwise, type remains the same
    else {
        return type;
    }
}
//--------------------------------------------------------------------------
void Backend::genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const
{
    if (!(loc & VarLocation::ZERO_COPY)) {
        // Only copy if uninitialisedOnly isn't set
        if (autoInitialized) {
            os << "if(!uninitialisedOnly)" << CodeStream::OB(1101);
        }

        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_" << name;
        os << ", CL_TRUE";
        os << ", 0";
        os << ", " << count << " * sizeof(" << type << ")";
        os << ", " << name << "));" << std::endl;

        if (autoInitialized) {
            os << CodeStream::CB(1101);
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const
{
    if (!(loc & VarLocation::ZERO_COPY)) {
        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_" << name;
        os << ", " << "CL_TRUE";
        os << ", " << "0";
        os << ", " << count << " * sizeof(" << type << ")";
        os << ", " << name << "));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, 
                                     const std::string &name, VarLocation loc, unsigned int batchSize) const
{
    assert(!getPreferences().automaticCopy);

    // If this variable requires queuing and isn't zero-copy
    if (ng.isVarQueueRequired(name) && ng.isDelayRequired() && !(loc & VarLocation::ZERO_COPY)) {
        // If batch size is one, generate 1D memcpy to copy current timestep's data
        if(batchSize == 1) {
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_" << name << ng.getName();
            os << ", CL_TRUE";
            os << ", spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << " * sizeof(" << type << ")";
            os << ", " << ng.getNumNeurons() << " * sizeof(" << type << ")";
            os << ", &" << name << ng.getName() << "[spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << "]));" << std::endl;
        }
        // Otherwise, perform a 2D memcpy to copy current timestep's data from each batch
        else {
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBufferRect(d_" << name << ng.getName();
            os << ", CL_TRUE";
            os << ", {0, spkQuePtr" << ng.getName() << ", 0}, {0, spkQuePtr" << ng.getName() << ", 0}";
            os << ", {" << ng.getNumNeurons() << " * sizeof(" << type << "), 1, " << batchSize << "}";
            os << ", " << ng.getNumNeurons() << " * sizeof(" << type << ")";
            os << ", " << ng.getNumNeurons() * ng.getNumDelaySlots() << " * sizeof(" << type << ")";
            os << ", " << ng.getNumNeurons() << " * sizeof(" << type << ")";
            os << ", " << ng.getNumNeurons() * ng.getNumDelaySlots() << " * sizeof(" << type << ")";
            os << ", " << name << ng.getName() << "));" << std::endl;
        }
    }
    // Otherwise, generate standard push
    else {
        genVariablePush(os, type, name + ng.getName(), loc, false, ng.getNumNeurons() * batchSize);
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePull(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, 
                                     const std::string &name, VarLocation loc, unsigned int batchSize) const
{
    assert(!getPreferences().automaticCopy);

    // If this variable requires queuing and isn't zero-copy
    if (ng.isVarQueueRequired(name) && ng.isDelayRequired() && !(loc & VarLocation::ZERO_COPY)) {
        // If batch size is one, generate 1D memcpy to copy current timestep's data
        if(batchSize == 1) {
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_" << name << ng.getName();
            os << ", CL_TRUE";
            os << ", spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << " * sizeof(" << type << ")";
            os << ", " << ng.getNumNeurons() << " * sizeof(" << type << ")";
            os << ", &" << name << ng.getName() << "[spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << "]));" << std::endl;
        }
        // Otherwise, perform a 2D memcpy to copy current timestep's data from each batch
        else {
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBufferRect(d_" << name << ng.getName();
            os << ", CL_TRUE";
            os << ", {0, spkQuePtr" << ng.getName() << ", 0}, {0, spkQuePtr" << ng.getName() << ", 0}";
            os << ", {" << ng.getNumNeurons() << " * sizeof(" << type << "), 1, " << batchSize << "}";
            os << ", " << ng.getNumNeurons() << " * sizeof(" << type << ")";
            os << ", " << ng.getNumNeurons() * ng.getNumDelaySlots() << " * sizeof(" << type << ")";
            os << ", " << ng.getNumNeurons() << " * sizeof(" << type << ")";
            os << ", " << ng.getNumNeurons() * ng.getNumDelaySlots() << " * sizeof(" << type << ")";
            os << ", " << name << ng.getName() << "));" << std::endl;
        }
    }
    // Otherwise, generate standard push
    else {
        genVariablePull(os, type, name + ng.getName(), loc, ng.getNumNeurons() * batchSize);
    }
}
//--------------------------------------------------------------------------
void Backend::genGlobalDeviceRNG(CodeStream&, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free, MemAlloc&) const
{
    definitionsInternal << "EXPORT_VAR clrngPhilox432Stream* rng;" << std::endl;
    definitionsInternal << "EXPORT_VAR cl::Buffer d_rng;" << std::endl;

    runner << "clrngPhilox432Stream* rng;" << std::endl;
    runner << "cl::Buffer d_rng;" << std::endl;

    free << "clrngPhilox432DestroyStreams(rng);" << std::endl;

    {
        CodeStream::Scope b(allocations);

        allocations << "size_t deviceBytes;" << std::endl;
        allocations << "rng = clrngPhilox432CreateStreams(philoxStreamCreator, 1, &deviceBytes, nullptr);" << std::endl;
        allocations << "assert(deviceBytes == " << getSize("clrngPhilox432Stream") << ");" << std::endl;
        allocations << "CHECK_OPENCL_ERRORS_POINTER(d_rng = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, deviceBytes, rng, &error));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genPopulationRNG(CodeStream&, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                               const std::string &name, size_t count, MemAlloc &memAlloc) const
{
    definitionsInternal << "EXPORT_VAR clrngLfsr113Stream* " << name << ";" << std::endl;
    definitionsInternal << "EXPORT_VAR cl::Buffer d_" << name << ";" << std::endl;

    runner << "clrngLfsr113Stream* " << name << ";" << std::endl;
    runner << "cl::Buffer d_" << name << ";" << std::endl;

    free << "clrngLfsr113DestroyStreams(" << name << ");" << std::endl;

    {
        CodeStream::Scope b(allocations);

        const size_t sizeBytes = count * getSize("clrngLfsr113Stream");
        allocations << "size_t deviceBytes;" << std::endl;
        allocations << name << " = clrngLfsr113CreateStreams(lfsrStreamCreator, " << count << ", &deviceBytes, nullptr);" << std::endl;
        allocations << "assert(deviceBytes == " << sizeBytes << ");" << std::endl;

        if(shouldUseSubBufferAllocations()) {
            // Apply minimum alignement to size
            const size_t alignedSizeBytes = padSize(sizeBytes, m_AllocationAlignementBytes);

            // Create region struct defining location of this variable
            allocations << "const cl_buffer_region region{size_t{" << memAlloc.getDeviceBytes() << "}, size_t{" << alignedSizeBytes << "}};" << std::endl;

            // Create sub-buffer encapsulating this region within main static buffer
            allocations << "CHECK_OPENCL_ERRORS_POINTER(d_" << name << " = d_staticBuffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &error));" << std::endl;

            // Copy RNG to sub-buffer
            // **NOTE** not at all clear how CL_MEM_COPY_HOST_PTR works in the context of sub-buffers
            allocations << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_" << name << ", CL_TRUE, 0, " << sizeBytes << ", " << name << "));" << std::endl;

            // Add aligned size to mem allocation tracker
            memAlloc += MemAlloc::device(alignedSizeBytes);
        }
        else {
            allocations << "CHECK_OPENCL_ERRORS_POINTER(d_" << name << " = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, deviceBytes, " << name << ", &error));" << std::endl;

            memAlloc += MemAlloc::hostDevice(sizeBytes);
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genTimer(CodeStream&, CodeStream &definitionsInternal, CodeStream &runner, CodeStream&, CodeStream&,
                       CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const
{
    // Define OpenCL event in internal defintions (as they use CUDA-specific types)
    definitionsInternal << "EXPORT_VAR cl::Event " << name  << "Event;" << std::endl;

    // Implement  event variables
    runner << "cl::Event " << name << "Event;" << std::endl;

    if(updateInStepTime) {
        CodeGenerator::CodeStream::Scope b(stepTimeFinalise);
        genReadEventTiming(stepTimeFinalise, name);
    }

}
//--------------------------------------------------------------------------
void Backend::genReturnFreeDeviceMemoryBytes(CodeStream &os) const
{
    // If device is AMD, use extension to get free device memory
    if(isChosenDeviceAMD()) {
        // CL_DEVICE_GLOBAL_FREE_MEMORY_AMD = 0x4039
        // **NOTE** http://manpages.ubuntu.com/manpages/bionic/man1/clinfo.1.html and https://github.com/Oblomov/clinfo/blob/master/src/clinfo.c#L1406-L1411 suggest that, with newer drivers, you can also get maximum block size by requesting 128 bits of data here
        // **YUCK** https://www.khronos.org/registry/OpenCL/extensions/amd/cl_amd_device_attribute_query.txt suggests this should be in KBytes but it doesn't seem to be in bytes
        os << "cl_ulong freeMem;" << std::endl;
        os << "CHECK_OPENCL_ERRORS(clDevice.getInfo(0x4039, &freeMem));" << std::endl;
        os << "return freeMem;" << std::endl;
    }
    // Otherwise, return 0 as vanilla OpenCL doesn't have this functionality
    else {
        os << "return 0;" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genMakefilePreamble(std::ostream &os) const
{
    os << "OBJECTS += opencl/clRNG/clRNG.o opencl/clRNG/private.o opencl/clRNG/lfsr113.o opencl/clRNG/philox432.o" << std::endl;
    os << "LINKFLAGS := -shared";
#ifdef __APPLE__
    os << " -framework OpenCL";
#else
    os << " -L$(OPENCL_PATH)/lib64";
#endif
    os << std::endl;
    os << "CCFLAGS := -c -fPIC -MMD -MP";
#ifndef __APPLE__
    os << " -I$(OPENCL_PATH)/include";
#endif
    os << " -Iopencl/clRNG/include" << std::endl;
    os << "CXXFLAGS += -std=c++11 -DCLRNG_SINGLE_PRECISION -DCL_HPP_TARGET_OPENCL_VERSION=120 -DCL_HPP_MINIMUM_OPENCL_VERSION=120 -Wno-ignored-attributes $(CCFLAGS)" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileLinkRule(std::ostream &os) const
{
    os << "\t@$(CXX) $(LINKFLAGS) -o $@ $(OBJECTS)";
#ifndef __APPLE__
    os << " -lOpenCL";
#endif
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileCompileRule(std::ostream &os) const
{
    os << "%.o: %.cc" << std::endl;
    os << "\t@$(CXX) $(CXXFLAGS) -o $@ $<" << std::endl;
    os << std::endl;
    os << "opencl/clRNG/%.o: opencl/clRNG/%.c" << std::endl;
    os << "\t@$(CC) $(CCFLAGS) -o $@ $<" << std::endl;
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildConfigProperties(std::ostream&) const
{
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportProps(std::ostream &os) const
{
    os << "\t<ImportGroup Label=\"ExtensionSettings\">" << std::endl;
    os << "\t</ImportGroup>" << std::endl;
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
    os << "\t\t\t<PreprocessorDefinitions Condition=\"'$(Configuration)'=='Release'\">_CRT_SECURE_NO_WARNINGS;WIN32;WIN64;NDEBUG;_CONSOLE;BUILDING_GENERATED_CODE;CLRNG_SINGLE_PRECISION;CL_HPP_TARGET_OPENCL_VERSION=120;CL_HPP_MINIMUM_OPENCL_VERSION=120;%(PreprocessorDefinitions)</PreprocessorDefinitions>" << std::endl;
    os << "\t\t\t<PreprocessorDefinitions Condition=\"'$(Configuration)'=='Debug'\">_CRT_SECURE_NO_WARNINGS;WIN32;WIN64;_DEBUG;_CONSOLE;BUILDING_GENERATED_CODE;CLRNG_SINGLE_PRECISION;CL_HPP_TARGET_OPENCL_VERSION=120;CL_HPP_MINIMUM_OPENCL_VERSION=120;%(PreprocessorDefinitions)</PreprocessorDefinitions>" << std::endl;
    os << "\t\t\t<AdditionalIncludeDirectories>opencl\\clRNG\\include;$(OPENCL_PATH)\\include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>" << std::endl;
    os << "\t\t</ClCompile>" << std::endl;

    // Add item definition for linking
    os << "\t\t<Link>" << std::endl;
    os << "\t\t\t<GenerateDebugInformation>true</GenerateDebugInformation>" << std::endl;
    os << "\t\t\t<EnableCOMDATFolding Condition=\"'$(Configuration)'=='Release'\">true</EnableCOMDATFolding>" << std::endl;
    os << "\t\t\t<OptimizeReferences Condition=\"'$(Configuration)'=='Release'\">true</OptimizeReferences>" << std::endl;
    os << "\t\t\t<SubSystem>Console</SubSystem>" << std::endl;
    os << "\t\t\t<AdditionalLibraryDirectories>$(OPENCL_PATH)\\lib\\x64;$(OPENCL_PATH)\\lib\\x86_64;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>" << std::endl;
    os << "\t\t\t<AdditionalDependencies>OpenCL.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>" << std::endl;
    os << "\t\t</Link>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const
{
    os << "\t\t<ClCompile Include=\"" << moduleName << ".cc\" />" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportTarget(std::ostream &os) const
{
    os << "\t<ItemGroup Label=\"clRNG\">" << std::endl;
    const std::array<std::string, 4> clrngItems{{ "clRNG.c", "private.c", "lfsr113.c", "philox432.c" }};
    for (const auto& clrngItem : clrngItems) {
        os << "\t\t<ClCompile Include=\"opencl\\clRNG\\" << clrngItem << "\" />" << std::endl;
    }
    os << "\t</ItemGroup>" << std::endl;
}
//--------------------------------------------------------------------------
std::vector<filesystem::path> Backend::getFilesToCopy(const ModelSpecMerged &) const
{
    const auto opencl = filesystem::path("opencl");
    const auto clRNG = opencl / "clRNG";
    const auto clRNGInclude = clRNG / "include" / "clRNG";
    const auto clRNGIncludePrivate = clRNGInclude / "private";
    const auto clRNGIncludePrivateRandom123 = clRNGIncludePrivate / "Random123";
    return {opencl / "cl2.hpp",
            
            clRNG / "lfsr113.c",
            clRNG / "clRNG.c",
            clRNG / "private.h",
            clRNG / "philox432.c",
            clRNG / "modularHost.c.h",
            clRNG / "private.c",

            clRNGInclude / "lfsr113.h",
            clRNGInclude / "clRNG.h",
            clRNGInclude / "clRNG.clh",
            clRNGInclude / "lfsr113.clh",
            clRNGInclude / "philox432.h",
            clRNGInclude / "philox432.clh",

            clRNGIncludePrivate / "philox432.c.h",
            clRNGIncludePrivate / "modular.c.h",
            clRNGIncludePrivate / "lfsr113.c.h",
            clRNGIncludePrivate / "device" / "philox432.c.h",
            clRNGIncludePrivate / "device" / "modular.c.h",
            clRNGIncludePrivate / "device" / "lfsr113.c.h",

            clRNGIncludePrivateRandom123 / "array.h",
            clRNGIncludePrivateRandom123 / "philox.h",
            clRNGIncludePrivateRandom123 / "features" / "clangfeatures.h",
            clRNGIncludePrivateRandom123 / "features" / "compilerfeatures.h",
            clRNGIncludePrivateRandom123 / "features" / "gccfeatures.h",
            clRNGIncludePrivateRandom123 / "features" / "msvcfeatures.h",
            clRNGIncludePrivateRandom123 / "features" / "open64features.h",
            clRNGIncludePrivateRandom123 / "features" / "openclfeatures.h",
            clRNGIncludePrivateRandom123 / "features" / "sse.h"};
}
//--------------------------------------------------------------------------
Backend::MemorySpaces Backend::getMergedGroupMemorySpaces(const ModelSpecMerged &) const
{
    return {};
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikePushPull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize, bool spikeEvent, bool push) const
{
    assert(!getPreferences().automaticCopy);

    if (!(ng.getSpikeLocation() & VarLocation::ZERO_COPY)) {
        // Is delay required
        const bool delayRequired = spikeEvent ?
            ng.isDelayRequired() :
            (ng.isTrueSpikeRequired() && ng.isDelayRequired());

        const char *function = push ? "Write" : "Read";
        const char* spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
        const char* spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

        if (delayRequired) {
            // If there's only a single batch
            if(batchSize == 1) {
                os << "CHECK_OPENCL_ERRORS(commandQueue.enqueue" << function << "Buffer(d_" << spikeCntPrefix << ng.getName();
                os << ", CL_TRUE";
                os << ", spkQuePtr" << ng.getName() << " * sizeof(unsigned int)";
                os << ", sizeof(unsigned int)";
                os << ", &" << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << "]));" << std::endl;

                os << "if(" << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << "] > 0)";
                {
                    CodeStream::Scope b(os);
                    os << "CHECK_OPENCL_ERRORS(commandQueue.enqueue" << function << "Buffer(d_" << spikePrefix << ng.getName();
                    os << ", CL_TRUE";
                    os << ", spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << " * sizeof(unsigned int)";
                    os << ", " << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << "] * sizeof(unsigned int)";
                    os << ", &" << spikePrefix << ng.getName() << "[spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << "]));" << std::endl;
                }
            }
            else {
                // Copy spike count for current timestep  from each batch using 2D memcpy
                // **NOTE** it's not at all clear from the documentation but offset[0] is in bytes 
                // whereas offset[1] and offset[2] are in entries (as is documented in the spec if not the manual for region)
                os << "CHECK_OPENCL_ERRORS(commandQueue.enqueue" << function << "BufferRect(d_" << spikeCntPrefix << ng.getName();
                os << ", CL_TRUE";
                os << ", {spkQuePtr" << ng.getName() << " * sizeof(unsigned int), 0, 0}";
                os << ", {spkQuePtr" << ng.getName() << " * sizeof(unsigned int), 0, 0}";
                os << ", {sizeof(unsigned int), " << batchSize << ", 1}";
                os << ", " << ng.getNumDelaySlots() << " * sizeof(unsigned int)";
                os << ", 0";
                os << ", " << ng.getNumDelaySlots() << " * sizeof(unsigned int)";
                os << ", 0";
                os << ", " << spikeCntPrefix << ng.getName() << "));" << std::endl;
               
                // Loop through batches and launch asynchronous memcpys to copy spikes from each one
                os << "for(unsigned int b = 0; b < " << batchSize << "; b++)";
                {
                    CodeStream::Scope b(os);
                    os << "const unsigned int spikeCount = " << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << " + (b * " << ng.getNumDelaySlots() << ")];" << std::endl;
                    os << "if(spikeCount > 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "const unsigned int spikeOffset = (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ") + (b * " << (ng.getNumNeurons() * ng.getNumDelaySlots()) << ");" << std::endl;
                        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueue" << function << "Buffer(d_" << spikePrefix << ng.getName();
                        os << ", CL_FALSE";
                        os << ", spikeOffset * sizeof(unsigned int)";
                        os << ", spikeCount * sizeof(unsigned int)";
                        os << ", " << spikePrefix << ng.getName() << " + spikeOffset));" << std::endl;
                    }
                }

                 // Wait until queued copies have completed
                os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
            }
        }
        else {
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueue" << function << "Buffer(d_" << spikeCntPrefix << ng.getName();
            os << ", CL_TRUE";
            os << ", 0";
            os << ", " << batchSize << " * sizeof(unsigned int)";
            os << ", " << spikeCntPrefix << ng.getName() << "));" << std::endl;

            // If there's only a single batch, copy spikes
            if(batchSize == 1) {
                os << "if(" << spikeCntPrefix << ng.getName() << "[0] > 0)";
                {
                    CodeStream::Scope b(os);
                    os << "CHECK_OPENCL_ERRORS(commandQueue.enqueue" << function << "Buffer(d_" << spikePrefix << ng.getName();
                    os << ", CL_TRUE";
                    os << ", 0";
                    os << ", " << spikeCntPrefix << ng.getName() << "[0] * sizeof(unsigned int)";
                    os << ", " << spikePrefix << ng.getName() << "));" << std::endl;
                }
            }
            else {
                // Otherwise, loop through batches and launch asynchronous memcpys to copy spikes from each one
                os << "for(unsigned int b = 0; b < " << batchSize << "; b++)";
                {
                    CodeStream::Scope b(os);
                    os << "if(" << spikeCntPrefix << ng.getName() << "[b] > 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueue" << function << "Buffer(d_" << spikePrefix << ng.getName();
                        os << ", CL_FALSE";
                        os << ", b * " << ng.getNumNeurons() << " * sizeof(unsigned int)";
                        os << ", " << spikeCntPrefix << ng.getName() << "[b] * sizeof(unsigned int)";
                        os << ", " << spikePrefix << ng.getName() << " + (b * " << ng.getNumNeurons() << ")));" << std::endl;
                    }
                }

                // Wait until queued copies have completed
                os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genAtomicAddFloat(CodeStream &os, const std::string &memoryType) const
{
    os << "inline void atomic_add_f_" << memoryType << "(volatile __" << memoryType << " float *source, const float operand)";
    {
        CodeStream::Scope b(os);

        // If device is NVIDIA running an NVIDIA platoform, insert PTX code for fire-and-forget floating point atomic add
        // https://github.com/openai/blocksparse/blob/master/src/ew_op_gpu.h#L648-L652
        if(isChosenDeviceNVIDIA() && isChosenPlatformNVIDIA()) {
            const std::string space = (memoryType == "global") ? "global" : "shared";
            os << "asm volatile(\"red." << space << ".add.f32[%0], %1;\" :: \"l\"(source), \"f\"(operand));" << std::endl;
        }
        // Otherwise use atomic_cmpxchg-based solution
        else {
            os << "union { unsigned int intVal; float floatVal; } newVal;" << std::endl;
            os << "union { unsigned int intVal; float floatVal; } prevVal;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "prevVal.floatVal = *source;" << std::endl;
                os << "newVal.floatVal = prevVal.floatVal + operand;" << std::endl;
            }
            os << "while (atomic_cmpxchg((volatile __" << memoryType << " unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);" << std::endl;
        }
    }

    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreadsX, size_t batchSize, size_t numBlockThreadsY) const
{
    // Calculate global and local work size
    const auto maxWorkItemDims = m_ChosenDevice.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    const size_t numWorkGroups = ceilDivide(numThreadsX, getKernelBlockSize(kernel));
    assert(numBlockThreadsY < maxWorkItemDims[0]);

    os << "const cl::NDRange localWorkSize(" << getKernelBlockSize(kernel) << ", " << numBlockThreadsY << ");" << std::endl;
    if(numBlockThreadsY > 1) {
        assert(batchSize < (size_t)maxWorkItemDims[2]);
        os << "const cl::NDRange globalWorkSize(" << (getKernelBlockSize(kernel) * numWorkGroups) << ", " << numBlockThreadsY << ", " << batchSize << ");" << std::endl;
    }
    else {
        assert(batchSize < (size_t)maxWorkItemDims[1]);
        os << "const cl::NDRange globalWorkSize(" << (getKernelBlockSize(kernel) * numWorkGroups) << ", " << batchSize << ");" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genKernelPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    const ModelSpecInternal &model = modelMerged.getModel();
    const std::string precision = model.getPrecision();

    // **YUCK** OpenCL doesn't let you include C99 system header so, instead, 
    // manually define C99 types in terms of OpenCL types (whose sizes are guaranteed)
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// C99 sized types" << std::endl;
    os << "typedef uchar uint8_t;" << std::endl;
    os << "typedef ushort uint16_t;" << std::endl;
    os << "typedef uint uint32_t;" << std::endl;
    os << "typedef ulong uint64_t;" << std::endl;
    os << "typedef char int8_t;" << std::endl;
    os << "typedef short int16_t;" << std::endl;
    os << "typedef int int32_t;" << std::endl;
    os << "typedef long int64_t;" << std::endl;
    os << std::endl;

    // Include clRNG headers in kernel
    os << "#define CLRNG_SINGLE_PRECISION" << std::endl;
    os << "#include <clRNG/lfsr113.clh>" << std::endl;
    os << "#include <clRNG/philox432.clh>" << std::endl;

    os << "typedef " << precision << " scalar;" << std::endl;
    os << "#define DT " << model.scalarExpr(model.getDT()) << std::endl;
    os << "#define SUPPORT_CODE_FUNC" << std::endl;
    genTypeRange(os, model.getTimePrecision(), "TIME");

    // Generate non-uniform generators for each supported RNG type
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Non-uniform generators" << std::endl;
    const std::array<std::string, 2> rngs{{"Lfsr113", "Philox432"}};
    for(const std::string &r : rngs) {
        os << "inline " << precision << " exponentialDist" << r << "(clrng" << r << "Stream *rng)";
        {
            CodeStream::Scope b(os);
            os << "while (true)";
            {
                CodeStream::Scope b(os);
                os << "const " << precision << " u = clrng" << r << "RandomU01(rng);" << std::endl;
                os << "if (u != " << model.scalarExpr(0.0) << ")";
                {
                    CodeStream::Scope b(os);
                    os << "return -log(u);" << std::endl;
                }
            }
        }
        os << std::endl;

        // Box-Muller algorithm based on https://www.johndcook.com/SimpleRNG.cpp
        os << "inline " << precision << " normalDist" << r << "(clrng" << r << "Stream *rng)";
        {
            CodeStream::Scope b(os);
            const std::string pi = (model.getPrecision() == "float") ? "M_PI_F" : "M_PI";
            os << "const " << precision << " u1 = clrng" << r << "RandomU01(rng);" << std::endl;
            os << "const " << precision << " u2 = clrng" << r << "RandomU01(rng);" << std::endl;
            os << "const " << precision << " r = sqrt(" << model.scalarExpr(-2.0) << " * log(u1));" << std::endl;
            os << "const " << precision << " theta = " << model.scalarExpr(2.0) << " * " << pi << " * u2;" << std::endl;
            os << "return r * sin(theta);" << std::endl;
        }
        os << std::endl;

        os << "inline " << precision << " logNormalDist" << r << "(clrng" << r << "Stream *rng, " << precision << " mean," << precision << " stddev)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "return exp(mean + (stddev * normalDist" << r << "(rng)));" << std::endl;
        }
        os << std::endl;

        // Generate gamma-distributed variates using Marsaglia and Tsang's method
        // G. Marsaglia and W. Tsang. A simple method for generating gamma variables. ACM Transactions on Mathematical Software, 26(3):363-372, 2000.
        os << "inline " << precision << " gammaDistInternal" << r << "(clrng" << r << "Stream *rng, " << precision << " c, " << precision << " d)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "" << precision << " x, v, u;" << std::endl;
            os << "while (true)";
            {
                CodeStream::Scope b(os);
                os << "do";
                {
                    CodeStream::Scope b(os);
                    os << "x = normalDist" << r << "(rng);" << std::endl;
                    os << "v = " << model.scalarExpr(1.0) << " + c*x;" << std::endl;
                }
                os << "while (v <= " << model.scalarExpr(0.0) << ");" << std::endl;
                os << std::endl;
                os << "v = v*v*v;" << std::endl;
                os << "do";
                {
                    CodeStream::Scope b(os);
                    os << "u = clrng" << r << "RandomU01(rng);" << std::endl;
                }
                os << "while (u == " << model.scalarExpr(1.0) << ");" << std::endl;
                os << std::endl;
                os << "if (u < " << model.scalarExpr(1.0) << " - " << model.scalarExpr(0.0331) << "*x*x*x*x) break;" << std::endl;
                os << "if (log(u) < " << model.scalarExpr(0.5) << "*x*x + d*(" << model.scalarExpr(1.0) << " - v + log(v))) break;" << std::endl;
            }
            os << std::endl;
            os << "return d*v;" << std::endl;
        }
        os << std::endl;

        os << "inline " << precision << " gammaDist" << r << "(clrng" << r << "Stream *rng, " << precision << " a)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "if (a > 1)" << std::endl;
            {
                CodeStream::Scope b(os);
                os << "const " << precision << " u = clrng" << r << "RandomU01 (rng);" << std::endl;
                os << "const " << precision << " d = (" << model.scalarExpr(1.0) << " + a) - " << model.scalarExpr(1.0) << " / " << model.scalarExpr(3.0) << ";" << std::endl;
                os << "const " << precision << " c = (" << model.scalarExpr(1.0) << " / " << model.scalarExpr(3.0) << ") / sqrt(d);" << std::endl;
                os << "return gammaDistInternal" << r << "(rng, c, d) * pow(u, " << model.scalarExpr(1.0) << " / a);" << std::endl;
            }
            os << "else" << std::endl;
            {
                CodeStream::Scope b(os);
                os << "const " << precision << " d = a - " << model.scalarExpr(1.0) << " / " << model.scalarExpr(3.0) << ";" << std::endl;
                os << "const " << precision << " c = (" << model.scalarExpr(1.0) << " / " << model.scalarExpr(3.0) << ") / sqrt(d);" << std::endl;
                os << "return gammaDistInternal" << r << "(rng, c, d);" << std::endl;
            }
        }
        os << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genBuildProgramFlagsString(CodeStream &os) const
{
    os << "const std::string buildProgramFlags = \"-cl-std=CL1.2 -I \" + getAbsoluteCodePath() + \"/opencl/clRNG/include";
    if(getPreferences().optimizeCode) {
        os << " -cl-fast-relaxed-math";
    }
    os << "\";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::divideKernelStreamInParts(CodeStream &os, const std::stringstream &kernelCode, size_t partLength) const
{
    const std::string kernelStr = kernelCode.str();
    const size_t parts = ceilDivide(kernelStr.length(), partLength);
    for(size_t i = 0; i < parts; i++) {
        os << "R\"(" << kernelStr.substr(i * partLength, partLength) << ")\"" << std::endl;
    }
}
//--------------------------------------------------------------------------
bool Backend::isChosenDeviceAMD() const
{
    // **YUCK** this is a horrible test but vendor IDs are not consistent 
    // and I have seen "AMD" and "Advanced Micro Devices, Inc."
    const std::string device = m_ChosenDevice.getInfo<CL_DEVICE_VENDOR>();
    return ((device.find("AMD") != std::string::npos)
            || (device.find("Advanced Micro Devices") != std::string::npos));
}
//--------------------------------------------------------------------------
bool Backend::isChosenDeviceNVIDIA() const
{
    // **YUCK** this is a horrible test but vendor IDs are not consistant 
    // and I have seen "NVIDIA" and "NVIDIA corporation"
    const std::string device = m_ChosenDevice.getInfo<CL_DEVICE_VENDOR>();
    return (device.find("NVIDIA") != std::string::npos);
}
//--------------------------------------------------------------------------
bool Backend::isChosenPlatformNVIDIA() const
{
    const std::string platform = m_ChosenPlatform.getInfo<CL_PLATFORM_NAME>();
    return (platform.find("NVIDIA") != std::string::npos);
}
//--------------------------------------------------------------------------
bool Backend::shouldUseSubBufferAllocations() const
{
    return isChosenDeviceAMD();
}
} // namespace OpenCL
} // namespace CodeGenerator

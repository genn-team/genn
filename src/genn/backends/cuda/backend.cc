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

// CUDA backend includes
#include "utils.h"

using namespace CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const std::vector<Substitutions::FunctionTemplate> cudaSinglePrecisionFunctions = {
    {"gennrand_uniform", 0, "curand_uniform($(rng))"},
    {"gennrand_normal", 0, "curand_normal($(rng))"},
    {"gennrand_exponential", 0, "exponentialDistFloat($(rng))"},
    {"gennrand_log_normal", 2, "curand_log_normal_float($(rng), $(0), $(1))"},
    {"gennrand_gamma", 1, "gammaDistFloat($(rng), $(0))"},
    {"gennrand_binomial", 2, "binomialDistFloat($(rng), $(0), $(1))"}
};
//--------------------------------------------------------------------------
const std::vector<Substitutions::FunctionTemplate> cudaDoublePrecisionFunctions = {
    {"gennrand_uniform", 0, "curand_uniform_double($(rng))"},
    {"gennrand_normal", 0, "curand_normal_double($(rng))"},
    {"gennrand_exponential", 0, "exponentialDistDouble($(rng))"},
    {"gennrand_log_normal", 2, "curand_log_normal_double($(rng), $(0), $(1))"},
    {"gennrand_gamma", 1, "gammaDistDouble($(rng), $(0))"},
    {"gennrand_binomial", 2, "binomialDistDouble($(rng), $(0), $(1))"} 
};
//--------------------------------------------------------------------------
// Timer
//--------------------------------------------------------------------------
class Timer
{
public:
    Timer(CodeStream &codeStream, const std::string &name, bool timingEnabled, bool synchroniseOnStop = false)
    :   m_CodeStream(codeStream), m_Name(name), m_TimingEnabled(timingEnabled), m_SynchroniseOnStop(synchroniseOnStop)
    {
        // Record start event
        if(m_TimingEnabled) {
            m_CodeStream << "CHECK_CUDA_ERRORS(cudaEventRecord(" << m_Name << "Start));" << std::endl;
        }
    }

    ~Timer()
    {
        // Record stop event
        if(m_TimingEnabled) {
            m_CodeStream << "CHECK_CUDA_ERRORS(cudaEventRecord(" << m_Name << "Stop));" << std::endl;

            // If we should synchronise on stop, insert call
            if(m_SynchroniseOnStop) {
                m_CodeStream << "CHECK_CUDA_ERRORS(cudaEventSynchronize(" << m_Name << "Stop));" << std::endl;

                m_CodeStream << "float tmp;" << std::endl;
                m_CodeStream << "CHECK_CUDA_ERRORS(cudaEventElapsedTime(&tmp, " << m_Name << "Start, " << m_Name << "Stop));" << std::endl;
                m_CodeStream << m_Name << "Time += tmp / 1000.0;" << std::endl;
            }
        }
    }

private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    CodeStream &m_CodeStream;
    const std::string m_Name;
    const bool m_TimingEnabled;
    const bool m_SynchroniseOnStop;
};
//-----------------------------------------------------------------------
template<typename T, typename G>
void genGroupStartID(CodeStream &os, size_t &idStart, size_t &totalConstMem,
                     const T &m, G getPaddedNumThreads)
{
    // Calculate size of array
    const size_t sizeBytes = m.getGroups().size() * sizeof(unsigned int);

    // If there is enough constant memory left for group, declare it in constant memory space
    if(sizeBytes < totalConstMem) {
        os << "__device__ __constant__ ";
        totalConstMem -= sizeBytes;
    }
    // Otherwise, declare it in global memory space
    else {
        os << "__device__ ";
    }

    // Declare array of starting thread indices for each neuron group
    os << "unsigned int d_merged" << T::name << "GroupStartID" << m.getIndex() << "[] = {";
    for(const auto &ng : m.getGroups()) {
        os << idStart << ", ";
        idStart += getPaddedNumThreads(ng.get());
    }
    os << "};" << std::endl;
}
//-----------------------------------------------------------------------
void genGroupStartIDs(CodeStream &, size_t&, size_t&)
{
}
//-----------------------------------------------------------------------
template<typename T, typename G, typename ...Args>
void genGroupStartIDs(CodeStream &os, size_t &idStart, size_t &totalConstMem, 
                      const std::vector<T> &mergedGroups, G getPaddedNumThreads,
                      Args... args)
{
    // Loop through merged groups
    for(const auto &m : mergedGroups) {
        genGroupStartID(os, idStart, totalConstMem, m, getPaddedNumThreads);
    }

    // Generate any remaining groups
    genGroupStartIDs(os, idStart, totalConstMem, args...);
}
//-----------------------------------------------------------------------
template<typename ...Args>
void genMergedKernelDataStructures(CodeStream &os, size_t &totalConstMem,
                                   Args... args)
{
    // Generate group start id arrays
    size_t idStart = 0;
    genGroupStartIDs(os, std::ref(idStart), std::ref(totalConstMem), args...);
}
//-----------------------------------------------------------------------
void genFilteredGroupStartIDs(CodeStream &, size_t&, size_t&)
{
}
//-----------------------------------------------------------------------
template<typename T, typename G, typename F, typename ...Args>
void genFilteredGroupStartIDs(CodeStream &os, size_t &idStart, size_t &totalConstMem,
                      const std::vector<T> &mergedGroups, G getPaddedNumThreads, F filter,
                      Args... args)
{
    // Loop through merged groups
    for(const auto &m : mergedGroups) {
        if(filter(m)) {
            genGroupStartID(os, idStart, totalConstMem, m, getPaddedNumThreads);
        }
    }

    // Generate any remaining groups
    genFilteredGroupStartIDs(os, idStart, totalConstMem, args...);
}
//-----------------------------------------------------------------------
template<typename ...Args>
void genFilteredMergedKernelDataStructures(CodeStream &os, size_t &totalConstMem,
                                           Args... args)
{
    // Generate group start id arrays
    size_t idStart = 0;
    genFilteredGroupStartIDs(os, std::ref(idStart), std::ref(totalConstMem), args...);
}
//-----------------------------------------------------------------------
template<typename T, typename G>
size_t getNumMergedGroupThreads(const std::vector<T> &groups, G getNumThreads)
{
    // Accumulate the accumulation of all groups in merged group
    return std::accumulate(
        groups.cbegin(), groups.cend(), size_t{0},
        [getNumThreads](size_t acc, const T &n)
        {
            return std::accumulate(n.getGroups().cbegin(), n.getGroups().cend(), acc,
                                   [getNumThreads](size_t acc, std::reference_wrapper<const typename T::GroupInternal> g)
                                   {
                                       return acc + getNumThreads(g.get());
                                   });
        });
}
//-----------------------------------------------------------------------
template<typename T>
size_t getGroupStartIDSize(const std::vector<T> &mergedGroups)
{
    // Calculate size of groups
    return std::accumulate(mergedGroups.cbegin(), mergedGroups.cend(),
                           size_t{0}, [](size_t acc, const T &ng)
                           {
                               return acc + (sizeof(unsigned int) * ng.getGroups().size());
                           });
}
//-----------------------------------------------------------------------
const std::vector<Substitutions::FunctionTemplate> &getFunctionTemplates(const std::string &precision)
{
    return (precision == "double") ? cudaDoublePrecisionFunctions : cudaSinglePrecisionFunctions;
}
//-----------------------------------------------------------------------
std::string getNCCLReductionType(VarAccessMode mode) 
{
    // Convert GeNN reduction types to NCCL
    if(mode & VarAccessModeAttribute::MAX) {
        return "ncclMax";
    }
    else if(mode & VarAccessModeAttribute::SUM) {
        return "ncclSum";
    }
    else {
        throw std::runtime_error("Reduction type unsupported by NCCL");
    }
}
//-----------------------------------------------------------------------
std::string getNCCLType(const std::string &type, const std::string &precision)
{
    // Convert GeNN types to NCCL types
    // **YUCK** GeNN really needs a better type system
    if(type == "scalar") {
        return (precision == "float") ? "ncclFloat32" : "ncclFloat64";
    }
    else if(type == "char" || type == "signed char" || type == "int8_t") {
        return "ncclInt8";
    }
    else if(type == "unsigned char" || type == "uint8_t") {
        return "ncclUint8";
    }
    else if(type == "int" || type == "signed int" || type == "signed" || type == "int32_t") {
        return "ncclInt32";
    }
    else if(type == "unsigned" || type == "unsigned int" || type == "uint32_t") {
        return "ncclUint32";
    }
    else if(type == "half") {
        return "ncclFloat16";
    }
    else if(type == "float") {
        return "ncclFloat32";
    }
    else if(type == "double") {
        return "ncclFloat64";
    }
    else {
        throw std::runtime_error("Data type '" + type + "' unsupported by NCCL");
    }
}
//-----------------------------------------------------------------------
template<typename G>
void genNCCLReduction(CodeStream &os, const G &cg, const std::string &precision)
{
    CodeStream::Scope b(os);
    os << "// merged custom update host reduction group " << cg.getIndex() << std::endl;
    os << "for(unsigned int g = 0; g < " << cg.getGroups().size() << "; g++)";
    {
        CodeStream::Scope b(os);

        // Get reference to group
        os << "const auto *group = &merged" << G::name << "Group" << cg.getIndex() << "[g]; " << std::endl;

        // Loop through variables and add pointers if they are reduction targets
        const CustomUpdateModels::Base *cm = cg.getArchetype().getCustomUpdateModel();
        for(const auto &v : cm->getVars()) {
            if(v.access & VarAccessModeAttribute::REDUCE) {
                os << "CHECK_NCCL_ERRORS(ncclAllReduce(group->" << v.name << ", group->" << v.name << ", group->size";
                os << ", " << getNCCLType(v.type, precision) << ", " << getNCCLReductionType(getVarAccessMode(v.access)) << ", ncclCommunicator, 0)); " << std::endl;
            }
        }

        // Loop through variable references and add pointers if they are reduction targets
        for(const auto &v : cm->getVarRefs()) {
            if(v.access & VarAccessModeAttribute::REDUCE) {
                os << "CHECK_NCCL_ERRORS(ncclAllReduce(group->" << v.name << ", group->" << v.name << ", group->size";
                os << ", " << getNCCLType(v.type, precision) << ", " << getNCCLReductionType(v.access) << ", ncclCommunicator, 0));" << std::endl;
            }
        } 
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::Backend
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace CUDA
{
//--------------------------------------------------------------------------
Backend::Backend(const KernelBlockSize &kernelBlockSizes, const Preferences &preferences,
                 const std::string &scalarType, int device)
:   BackendSIMT(kernelBlockSizes, preferences, scalarType), m_ChosenDeviceID(device)
{
    // Set device
    CHECK_CUDA_ERRORS(cudaSetDevice(device));

    // Get device properties
    CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&m_ChosenDevice, device));

    // Get CUDA runtime version
    cudaRuntimeGetVersion(&m_RuntimeVersion);

    // Give a warning if automatic copy is used on pre-Pascal devices
    if(getPreferences().automaticCopy && m_ChosenDevice.major < 6) {
        LOGW << "Using automatic copy on pre-Pascal devices is supported but likely to be very slow - we recommend copying manually on these devices";
    }

#ifdef _WIN32
    // If we're on Windows and NCCL is enabled, give error
    // **NOTE** There are several NCCL Windows ports e.g. https://github.com/MyCaffe/NCCL but we don't have access to any suitable systems to test
    if(getPreferences<Preferences>().enableNCCLReductions) {
        throw std::runtime_error("GeNN doesn't currently support NCCL on Windows");
    }
#endif

    // Add CUDA-specific types, additionally marking them as 'device types' innaccesible to host code
    addDeviceType("curandState", 44);
    addDeviceType("curandStatePhilox4_32_10_t", 64);
    addDeviceType("half", 2);
}
//--------------------------------------------------------------------------
bool Backend::areSharedMemAtomicsSlow() const
{
    // If device is older than Maxwell, we shouldn't use shared memory as atomics are emulated
    // and actually slower than global memory (see https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
    return (getChosenCUDADevice().major < 5);
}
//--------------------------------------------------------------------------
std::string Backend::getThreadID(unsigned int axis) const
{
    switch(axis) {
    case 0:
        return "threadIdx.x"; 
    case 1:
        return "threadIdx.y"; 
    case 2:
        return "threadIdx.z"; 
    default:
        assert(false);
    }
}
//--------------------------------------------------------------------------
std::string Backend::getBlockID(unsigned int axis) const
{
    switch(axis) {
    case 0:
        return "blockIdx.x"; 
    case 1:
        return "blockIdx.y"; 
    case 2:
        return "blockIdx.z"; 
    default:
        assert(false);
    }
}
//--------------------------------------------------------------------------
std::string Backend::getAtomic(const std::string &type, AtomicOperation op, AtomicMemSpace) const
{
    // If operation is an atomic add
    if(op == AtomicOperation::ADD) {
        if(((getChosenCUDADevice().major < 2) && (type == "float"))
           || (((getChosenCUDADevice().major < 6) || (getRuntimeVersion() < 8000)) && (type == "double")))
        {
            return "atomicAddSW";
        }

        return "atomicAdd";
    }
    // Otherwise, it's an atomic or
    else {
        assert(op == AtomicOperation::OR);
        assert(type == "unsigned int" || type == "int");
        return "atomicOr";
    }
}
//--------------------------------------------------------------------------
void Backend::genSharedMemBarrier(CodeStream &os) const
{
    os << "__syncthreads();" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genPopulationRNGInit(CodeStream &os, const std::string &globalRNG, const std::string &seed, const std::string &sequence) const
{
    os << "curand_init(" << seed << ", " << sequence << ", 0, &" << globalRNG << ");" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genPopulationRNGPreamble(CodeStream &, Substitutions &subs, const std::string &globalRNG, const std::string &name) const
{
    subs.addVarSubstitution(name, "&" + globalRNG);
}
//--------------------------------------------------------------------------
void Backend::genPopulationRNGPostamble(CodeStream&, const std::string&) const
{
}
//--------------------------------------------------------------------------
void Backend::genGlobalRNGSkipAhead(CodeStream &os, Substitutions &subs, const std::string &sequence, const std::string &name) const
{
    // Skipahead RNG
    os << "curandStatePhilox4_32_10_t localRNG = d_rng;" << std::endl;
    os << "skipahead_sequence((unsigned long long)" << sequence << ", &localRNG);" << std::endl;

    // Add substitution for RNG
    subs.addVarSubstitution(name, "&localRNG");
}
//--------------------------------------------------------------------------
void Backend::genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged,
                              HostHandler preambleHandler, HostHandler pushEGPHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Generate struct definitions
    modelMerged.genMergedNeuronUpdateGroupStructs(os, *this);
    modelMerged.genMergedNeuronSpikeQueueUpdateStructs(os, *this);
    modelMerged.genMergedNeuronPrevSpikeTimeUpdateStructs(os, *this);

    // Generate arrays of merged structs and functions to push them
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronSpikeQueueUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronUpdateGroups());

    // Generate preamble
    preambleHandler(os);

    // Generate data structure for accessing merged groups
    // **NOTE** constant cache is preferentially given to synapse groups as, typically, more synapse kernels are launched
    // so subtract constant memory requirements of synapse group start ids from total constant memory
    const size_t synapseGroupStartIDSize = (getGroupStartIDSize(modelMerged.getMergedPresynapticUpdateGroups()) +
                                            getGroupStartIDSize(modelMerged.getMergedPostsynapticUpdateGroups()) +
                                            getGroupStartIDSize(modelMerged.getMergedSynapseDynamicsGroups()));
    size_t totalConstMem = (getChosenDeviceSafeConstMemBytes() > synapseGroupStartIDSize) ? (getChosenDeviceSafeConstMemBytes() - synapseGroupStartIDSize) : 0;
    genMergedKernelDataStructures(os, totalConstMem, modelMerged.getMergedNeuronUpdateGroups(),
                                  [this](const NeuronGroupInternal &ng){ return padKernelSize(ng.getNumNeurons(), KernelNeuronUpdate); });
    genMergedKernelDataStructures(os, totalConstMem, modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups(),
                                  [this](const NeuronGroupInternal &ng){ return padKernelSize(ng.getNumNeurons(), KernelNeuronPrevSpikeTimeUpdate); });
    os << std::endl;

    // If any neuron groups require their previous spike times updating
    size_t idNeuronPrevSpikeTimeUpdate = 0;
    if(!modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << "(" << model.getTimePrecision() << " t)";
        {
            CodeStream::Scope b(os);

            Substitutions kernelSubs(getFunctionTemplates(model.getPrecision()));
            os << "const unsigned int id = " << getKernelBlockSize(KernelNeuronPrevSpikeTimeUpdate) << " * blockIdx.x + threadIdx.x;" << std::endl;
            if(model.getBatchSize() > 1) {
                os << "const unsigned int batch = blockIdx.y;" << std::endl;
                kernelSubs.addVarSubstitution("batch", "batch");
            }
            kernelSubs.addVarSubstitution("t", "t");

            genNeuronPrevSpikeTimeUpdateKernel(os, kernelSubs, modelMerged, idNeuronPrevSpikeTimeUpdate);
        }
        os << std::endl;
    }

    // Generate reset kernel to be run before the neuron kernel
    size_t idNeuronSpikeQueueUpdate = 0;
    os << "extern \"C\" __global__ void " << KernelNames[KernelNeuronSpikeQueueUpdate] << "()";
    {
        CodeStream::Scope b(os);

        os << "const unsigned int id = " << getKernelBlockSize(KernelNeuronSpikeQueueUpdate) << " * blockIdx.x + threadIdx.x;" << std::endl;

        genNeuronSpikeQueueUpdateKernel(os, modelMerged, idNeuronSpikeQueueUpdate);
    }
    os << std::endl;

    size_t idStart = 0;
    os << "extern \"C\" __global__ void " << KernelNames[KernelNeuronUpdate] << "(" << model.getTimePrecision() << " t";
    if(model.isRecordingInUse()) {
        os << ", unsigned int recordingTimestep";
    }
    os << ")" << std::endl;
    {
        CodeStream::Scope b(os);

        Substitutions kernelSubs(getFunctionTemplates(model.getPrecision()));
        kernelSubs.addVarSubstitution("t", "t");

        os << "const unsigned int id = " << getKernelBlockSize(KernelNeuronUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;
        if(model.getBatchSize() > 1) {
            os << "const unsigned int batch = blockIdx.y;" << std::endl;
            kernelSubs.addVarSubstitution("batch", "batch");
        }
        else {
            kernelSubs.addVarSubstitution("batch", "0");
        }
        genNeuronUpdateKernel(os, kernelSubs, modelMerged, idStart);
    }

    os << "void updateNeurons(" << model.getTimePrecision() << " t";
    if(model.isRecordingInUse()) {
        os << ", unsigned int recordingTimestep";
    }
    os << ")";
    {
        CodeStream::Scope b(os);

        // Push any required EGPS
        pushEGPHandler(os);

        if(idNeuronPrevSpikeTimeUpdate > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelNeuronPrevSpikeTimeUpdate, idNeuronPrevSpikeTimeUpdate, model.getBatchSize());
            os << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
        if(idNeuronSpikeQueueUpdate > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelNeuronSpikeQueueUpdate, idNeuronSpikeQueueUpdate, 1);
            os << KernelNames[KernelNeuronSpikeQueueUpdate] << "<<<grid, threads>>>();" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
        if(idStart > 0) {
            CodeStream::Scope b(os);

            Timer t(os, "neuronUpdate", model.isTimingEnabled());

            genKernelDimensions(os, KernelNeuronUpdate, idStart, model.getBatchSize());
            os << KernelNames[KernelNeuronUpdate] << "<<<grid, threads>>>(t";
            if(model.isRecordingInUse()) {
                os << ", recordingTimestep";
            }
            os << ");" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genSynapseUpdate(CodeStream &os, const ModelSpecMerged &modelMerged,
                               HostHandler preambleHandler, HostHandler pushEGPHandler) const
{
    // Generate struct definitions
    modelMerged.genMergedSynapseDendriticDelayUpdateStructs(os, *this);
    modelMerged.genMergedPresynapticUpdateGroupStructs(os, *this);
    modelMerged.genMergedPostsynapticUpdateGroupStructs(os, *this);
    modelMerged.genMergedSynapseDynamicsGroupStructs(os, *this);

    // Generate arrays of merged structs and functions to push them
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseDendriticDelayUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedPresynapticUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedPostsynapticUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseDynamicsGroups());

    // Generate preamble
    preambleHandler(os);

    // Generate data structure for accessing merged groups
    size_t totalConstMem = getChosenDeviceSafeConstMemBytes();
    genMergedKernelDataStructures(os, totalConstMem, modelMerged.getMergedPresynapticUpdateGroups(),
                                  [this](const SynapseGroupInternal &sg)
                                  {
                                      return padKernelSize(getNumPresynapticUpdateThreads(sg, getPreferences()), KernelPresynapticUpdate);
                                  });
    genMergedKernelDataStructures(os, totalConstMem, modelMerged.getMergedPostsynapticUpdateGroups(),
                                  [this](const SynapseGroupInternal &sg){ return padKernelSize(getNumPostsynapticUpdateThreads(sg), KernelPostsynapticUpdate); });

    genMergedKernelDataStructures(os, totalConstMem, modelMerged.getMergedSynapseDynamicsGroups(),
                                  [this](const SynapseGroupInternal &sg){ return padKernelSize(getNumSynapseDynamicsThreads(sg), KernelSynapseDynamicsUpdate); });

    // If any synapse groups require dendritic delay, a reset kernel is required to be run before the synapse kernel
    const ModelSpecInternal &model = modelMerged.getModel();
    size_t idSynapseDendricDelayUpdate = 0;
    if(!modelMerged.getMergedSynapseDendriticDelayUpdateGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelSynapseDendriticDelayUpdate] << "()";
        {
            CodeStream::Scope b(os);

            os << "const unsigned int id = " << getKernelBlockSize(KernelSynapseDendriticDelayUpdate) << " * blockIdx.x + threadIdx.x;" << std::endl;
            genSynapseDendriticDelayUpdateKernel(os, modelMerged, idSynapseDendricDelayUpdate);
        }
        os << std::endl;
    }

    // If there are any presynaptic update groups
    size_t idPresynapticStart = 0;
    if(!modelMerged.getMergedPresynapticUpdateGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelPresynapticUpdate] << "(" << model.getTimePrecision() << " t)" << std::endl; // end of synapse kernel header
        {
            CodeStream::Scope b(os);

            Substitutions kernelSubs((model.getPrecision() == "double") ? cudaDoublePrecisionFunctions : cudaSinglePrecisionFunctions);
            kernelSubs.addVarSubstitution("t", "t");

            os << "const unsigned int id = " << getKernelBlockSize(KernelPresynapticUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;
            if(model.getBatchSize() > 1) {
                os << "const unsigned int batch = blockIdx.y;" << std::endl;
                kernelSubs.addVarSubstitution("batch", "batch");
            }
            else {
                kernelSubs.addVarSubstitution("batch", "0");
            }
            genPresynapticUpdateKernel(os, kernelSubs, modelMerged, idPresynapticStart);
        }
    }

    // If any synapse groups require postsynaptic learning
    size_t idPostsynapticStart = 0;
    if(!modelMerged.getMergedPostsynapticUpdateGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelPostsynapticUpdate] << "(" << model.getTimePrecision() << " t)" << std::endl;
        {
            CodeStream::Scope b(os);

            Substitutions kernelSubs((model.getPrecision() == "double") ? cudaDoublePrecisionFunctions : cudaSinglePrecisionFunctions);
            kernelSubs.addVarSubstitution("t", "t");

            os << "const unsigned int id = " << getKernelBlockSize(KernelPostsynapticUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;
            if(model.getBatchSize() > 1) {
                os << "const unsigned int batch = blockIdx.y;" << std::endl;
                kernelSubs.addVarSubstitution("batch", "batch");
            }
            else {
                kernelSubs.addVarSubstitution("batch", "0");
            }
            genPostsynapticUpdateKernel(os, kernelSubs, modelMerged, idPostsynapticStart);
        }
    }

    size_t idSynapseDynamicsStart = 0;
    if(!modelMerged.getMergedSynapseDynamicsGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelSynapseDynamicsUpdate] << "(" << model.getTimePrecision() << " t)" << std::endl; // end of synapse kernel header
        {
            CodeStream::Scope b(os);

            Substitutions kernelSubs(getFunctionTemplates(model.getPrecision()));
            kernelSubs.addVarSubstitution("t", "t");

            os << "const unsigned int id = " << getKernelBlockSize(KernelSynapseDynamicsUpdate) << " * blockIdx.x + threadIdx.x;" << std::endl;
            if(model.getBatchSize() > 1) {
                os << "const unsigned int batch = blockIdx.y;" << std::endl;
                kernelSubs.addVarSubstitution("batch", "batch");
            }
            else {
                kernelSubs.addVarSubstitution("batch", "0");
            }
            genSynapseDynamicsKernel(os, kernelSubs, modelMerged, idSynapseDynamicsStart);
        }
    }

    os << "void updateSynapses(" << model.getTimePrecision() << " t)";
    {
        CodeStream::Scope b(os);

        // Push any required EGPs
        pushEGPHandler(os);

        // Launch pre-synapse reset kernel if required
        if(idSynapseDendricDelayUpdate > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelSynapseDendriticDelayUpdate, idSynapseDendricDelayUpdate, 1);
            os << KernelNames[KernelSynapseDendriticDelayUpdate] << "<<<grid, threads>>>();" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch synapse dynamics kernel if required
        if(idSynapseDynamicsStart > 0) {
            CodeStream::Scope b(os);
            Timer t(os, "synapseDynamics", model.isTimingEnabled());

            genKernelDimensions(os, KernelSynapseDynamicsUpdate, idSynapseDynamicsStart, model.getBatchSize());
            os << KernelNames[KernelSynapseDynamicsUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch presynaptic update kernel
        if(idPresynapticStart > 0) {
            CodeStream::Scope b(os);
            Timer t(os, "presynapticUpdate", model.isTimingEnabled());

            genKernelDimensions(os, KernelPresynapticUpdate, idPresynapticStart, model.getBatchSize());
            os << KernelNames[KernelPresynapticUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch postsynaptic update kernel
        if(idPostsynapticStart > 0) {
            CodeStream::Scope b(os);
            Timer t(os, "postsynapticUpdate", model.isTimingEnabled());

            genKernelDimensions(os, KernelPostsynapticUpdate, idPostsynapticStart, model.getBatchSize());
            os << KernelNames[KernelPostsynapticUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genCustomUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, 
                              HostHandler preambleHandler, HostHandler pushEGPHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Generate struct definitions
    modelMerged.genMergedCustomUpdateStructs(os, *this);
    modelMerged.genMergedCustomUpdateWUStructs(os, *this);
    modelMerged.genMergedCustomUpdateTransposeWUStructs(os, *this);

    // Generate arrays of merged structs and functions to push them
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateWUGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateTransposeWUGroups());
    
    // Generate preamble
    preambleHandler(os);

    // Generate data structure for accessing merged groups
    // **NOTE** constant cache is preferentially given to neuron and synapse groups as, typically, they are launched more often 
    // than custom update kernels so subtract constant memory requirements of synapse group start ids from total constant memory
    const size_t timestepGroupStartIDSize = (getGroupStartIDSize(modelMerged.getMergedPresynapticUpdateGroups()) +
                                             getGroupStartIDSize(modelMerged.getMergedPostsynapticUpdateGroups()) +
                                             getGroupStartIDSize(modelMerged.getMergedSynapseDynamicsGroups()) +
                                             getGroupStartIDSize(modelMerged.getMergedNeuronUpdateGroups()));
    size_t totalConstMem = (getChosenDeviceSafeConstMemBytes() > timestepGroupStartIDSize) ? (getChosenDeviceSafeConstMemBytes() - timestepGroupStartIDSize) : 0;

    // Build set containing union of all custom update groupsnames
    std::set<std::string> customUpdateGroups;
    std::transform(model.getCustomUpdates().cbegin(), model.getCustomUpdates().cend(),
                   std::inserter(customUpdateGroups, customUpdateGroups.end()),
                   [](const ModelSpec::CustomUpdateValueType &v) { return v.second.getUpdateGroupName(); });
    std::transform(model.getCustomWUUpdates().cbegin(), model.getCustomWUUpdates().cend(),
                   std::inserter(customUpdateGroups, customUpdateGroups.end()),
                   [](const ModelSpec::CustomUpdateWUValueType &v) { return v.second.getUpdateGroupName(); });

    // Loop through custom update groups
    for(const auto &g : customUpdateGroups) {
        // Generate kernel
        size_t idCustomUpdateStart = 0;
        if(std::any_of(modelMerged.getMergedCustomUpdateGroups().cbegin(), modelMerged.getMergedCustomUpdateGroups().cend(),
                       [&g](const CustomUpdateGroupMerged &c) { return (c.getArchetype().getUpdateGroupName() == g); })
           || std::any_of(modelMerged.getMergedCustomUpdateWUGroups().cbegin(), modelMerged.getMergedCustomUpdateWUGroups().cend(),
                       [&g](const CustomUpdateWUGroupMerged &c) { return (c.getArchetype().getUpdateGroupName() == g); }))
        {
            genFilteredMergedKernelDataStructures(os, totalConstMem,
                                                  modelMerged.getMergedCustomUpdateGroups(),
                                                  [this](const CustomUpdateInternal &cg){ return padKernelSize(cg.getSize(), KernelCustomUpdate); },
                                                  [g](const CustomUpdateGroupMerged &cg){ return cg.getArchetype().getUpdateGroupName() == g; },

                                                  modelMerged.getMergedCustomUpdateWUGroups(),
                                                  [&model, this](const CustomUpdateWUInternal &cg){ return getPaddedNumCustomUpdateWUThreads(cg, model.getBatchSize()); },
                                                  [g](const CustomUpdateWUGroupMerged &cg){ return cg.getArchetype().getUpdateGroupName() == g; });

            os << "extern \"C\" __global__ void " << KernelNames[KernelCustomUpdate] << g << "(" << model.getTimePrecision() << " t)" << std::endl;
            {
                CodeStream::Scope b(os);

                Substitutions kernelSubs((model.getPrecision() == "double") ? cudaDoublePrecisionFunctions : cudaSinglePrecisionFunctions);
                kernelSubs.addVarSubstitution("t", "t");

                os << "const unsigned int id = " << getKernelBlockSize(KernelCustomUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;

                os << "// ------------------------------------------------------------------------" << std::endl;
                os << "// Custom updates" << std::endl;
                genCustomUpdateKernel(os, kernelSubs, modelMerged, g, idCustomUpdateStart);

                os << "// ------------------------------------------------------------------------" << std::endl;
                os << "// Custom WU updates" << std::endl;
                genCustomUpdateWUKernel(os, kernelSubs, modelMerged, g, idCustomUpdateStart);
            }
        }

        size_t idCustomTransposeUpdateStart = 0;
        if(std::any_of(modelMerged.getMergedCustomUpdateTransposeWUGroups().cbegin(), modelMerged.getMergedCustomUpdateTransposeWUGroups().cend(),
                       [&g](const CustomUpdateTransposeWUGroupMerged &c){ return (c.getArchetype().getUpdateGroupName() == g); }))
        {
            genFilteredMergedKernelDataStructures(os, totalConstMem, modelMerged.getMergedCustomUpdateTransposeWUGroups(),
                                                  [&model, this](const CustomUpdateWUInternal &cg){ return getPaddedNumCustomUpdateTransposeWUThreads(cg, model.getBatchSize()); },
                                                  [g](const CustomUpdateTransposeWUGroupMerged &cg){ return cg.getArchetype().getUpdateGroupName() == g; });

            os << "extern \"C\" __global__ void " << KernelNames[KernelCustomTransposeUpdate] << g << "(" << model.getTimePrecision() << " t)" << std::endl;
            {
                CodeStream::Scope b(os);

                Substitutions kernelSubs((model.getPrecision() == "double") ? cudaDoublePrecisionFunctions : cudaSinglePrecisionFunctions);
                kernelSubs.addVarSubstitution("t", "t");

                os << "const unsigned int id = " << getKernelBlockSize(KernelCustomTransposeUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;

                os << "// ------------------------------------------------------------------------" << std::endl;
                os << "// Custom WU transpose updates" << std::endl;
                genCustomTransposeUpdateWUKernel(os, kernelSubs, modelMerged, g, idCustomTransposeUpdateStart);
            }
        }
        os << "void update" << g << "()";
        {
            CodeStream::Scope b(os);

            // Push any required EGPs
            pushEGPHandler(os);

            // Launch custom update kernel if required
            if(idCustomUpdateStart > 0) {
                CodeStream::Scope b(os);
                genKernelDimensions(os, KernelCustomUpdate, idCustomUpdateStart, 1);
                Timer t(os, "customUpdate" + g, model.isTimingEnabled());
                os << KernelNames[KernelCustomUpdate] << g << "<<<grid, threads>>>(t);" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
            }

            // Launch custom transpose update kernel if required
            if(idCustomTransposeUpdateStart > 0) {
                CodeStream::Scope b(os);
                // **TODO** make block height parameterizable
                genKernelDimensions(os, KernelCustomTransposeUpdate, idCustomTransposeUpdateStart, 1, 8);
                Timer t(os, "customUpdate" + g + "Transpose", model.isTimingEnabled());
                os << KernelNames[KernelCustomTransposeUpdate] << g << "<<<grid, threads>>>(t);" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
            }

            // If NCCL reductions are enabled
            if(getPreferences<Preferences>().enableNCCLReductions) {
                // Loop through custom update host reduction groups and
                // generate reductions for those in this custom update group
                for(const auto &cg : modelMerged.getMergedCustomUpdateHostReductionGroups()) {
                    if(cg.getArchetype().getUpdateGroupName() == g) {
                        genNCCLReduction(os, cg, model.getPrecision());
                    }
                }

                // Loop through custom update host reduction groups and
                // generate reductions for those in this custom update group
                for(const auto &cg : modelMerged.getMergedCustomWUUpdateHostReductionGroups()) {
                    if(cg.getArchetype().getUpdateGroupName() == g) {
                        genNCCLReduction(os, cg, model.getPrecision());
                    }
                }
            }

            // If timing is enabled
            if(model.isTimingEnabled()) {
                // Synchronise last event
                os << "CHECK_CUDA_ERRORS(cudaEventSynchronize(customUpdate" << g;
                if(idCustomTransposeUpdateStart > 0) {
                    os << "Transpose";
                }
                os << "Stop)); " << std::endl;

                if(idCustomUpdateStart > 0) {
                    CodeGenerator::CodeStream::Scope b(os);
                    os << "float tmp;" << std::endl;
                    os << "CHECK_CUDA_ERRORS(cudaEventElapsedTime(&tmp, customUpdate" << g << "Start, customUpdate" << g << "Stop));" << std::endl;
                    os << "customUpdate" << g << "Time += tmp / 1000.0;" << std::endl;
                }
                if(idCustomTransposeUpdateStart > 0) {
                    CodeGenerator::CodeStream::Scope b(os);
                    os << "float tmp;" << std::endl;
                    os << "CHECK_CUDA_ERRORS(cudaEventElapsedTime(&tmp, customUpdate" << g << "TransposeStart, customUpdate" << g << "TransposeStop));" << std::endl;
                    os << "customUpdate" << g << "TransposeTime += tmp / 1000.0;" << std::endl;
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genInit(CodeStream &os, const ModelSpecMerged &modelMerged, 
                      HostHandler preambleHandler, HostHandler initPushEGPHandler, HostHandler initSparsePushEGPHandler) const
{
    os << "#include <iostream>" << std::endl;
    os << "#include <random>" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << std::endl;

    // Generate struct definitions
    modelMerged.genMergedNeuronInitGroupStructs(os, *this);
    modelMerged.genMergedCustomUpdateInitGroupStructs(os, *this);
    modelMerged.genMergedCustomWUUpdateDenseInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseDenseInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseKernelInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseConnectivityInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseSparseInitGroupStructs(os, *this);
    modelMerged.genMergedCustomWUUpdateSparseInitGroupStructs(os, *this);

    // Generate arrays of merged structs and functions to push them
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomWUUpdateDenseInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseDenseInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseKernelInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseConnectivityInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseSparseInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomWUUpdateSparseInitGroups());

    // Generate preamble
    preambleHandler(os);

    // Generate data structure for accessing merged groups from within initialisation kernel
    // **NOTE** pass in zero constant cache here as it's precious and would be wasted on init kernels which are only launched once
    const ModelSpecInternal &model = modelMerged.getModel();
    size_t totalConstMem = 0;
    genMergedKernelDataStructures(
        os, totalConstMem,
        modelMerged.getMergedNeuronInitGroups(), [this](const NeuronGroupInternal &ng){ return padKernelSize(ng.getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedCustomUpdateInitGroups(), [this](const CustomUpdateInternal &cg) { return padKernelSize(cg.getSize(), KernelInitialize); },
        modelMerged.getMergedCustomWUUpdateDenseInitGroups(), [this](const CustomUpdateWUInternal &cg){ return padKernelSize(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedSynapseDenseInitGroups(), [this](const SynapseGroupInternal &sg){ return padKernelSize(sg.getTrgNeuronGroup()->getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedSynapseKernelInitGroups(), [this](const SynapseGroupInternal &sg){ return padKernelSize(sg.getKernelSizeFlattened(), KernelInitialize); },
        modelMerged.getMergedSynapseConnectivityInitGroups(), [this](const SynapseGroupInternal &sg){ return padKernelSize(getNumConnectivityInitThreads(sg), KernelInitialize); });

    // Generate data structure for accessing merged groups from within sparse initialisation kernel
    genMergedKernelDataStructures(
        os, totalConstMem,
        modelMerged.getMergedSynapseSparseInitGroups(), [this](const SynapseGroupInternal &sg){ return padKernelSize(sg.getMaxConnections(), KernelInitializeSparse); },
        modelMerged.getMergedCustomWUUpdateSparseInitGroups(), [this](const CustomUpdateWUInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getMaxConnections(), KernelInitializeSparse); });
    os << std::endl;

    // If device RNG is required, generate kernel to initialise it
    if(isGlobalDeviceRNGRequired(modelMerged)) {
        os << "extern \"C\" __global__ void initializeRNGKernel(unsigned long long deviceRNGSeed)";
        {
            CodeStream::Scope b(os);
            os << "if(threadIdx.x == 0)";
            {
                CodeStream::Scope b(os);
                os << "curand_init(deviceRNGSeed, 0, 0, &d_rng);" << std::endl;
            }
        }
        os << std::endl;
    }

    // init kernel header
    os << "extern \"C\" __global__ void " << KernelNames[KernelInitialize] << "(unsigned long long deviceRNGSeed)";

    // initialization kernel code
    size_t idInitStart = 0;
    {
        Substitutions kernelSubs(getFunctionTemplates(model.getPrecision()));

        // common variables for all cases
        CodeStream::Scope b(os);

        os << "const unsigned int id = " << getKernelBlockSize(KernelInitialize) << " * blockIdx.x + threadIdx.x;" << std::endl;
        genInitializeKernel(os, kernelSubs, modelMerged, idInitStart);
    }
    const size_t numStaticInitThreads = idInitStart;

    // Sparse initialization kernel code
    size_t idSparseInitStart = 0;
    if(!modelMerged.getMergedSynapseSparseInitGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelInitializeSparse] << "()";
        {
            CodeStream::Scope b(os);

            // common variables for all cases
            Substitutions kernelSubs(getFunctionTemplates(model.getPrecision()));

            os << "const unsigned int id = " << getKernelBlockSize(KernelInitializeSparse) << " * blockIdx.x + threadIdx.x;" << std::endl;
            genInitializeSparseKernel(os, kernelSubs, modelMerged, numStaticInitThreads, idSparseInitStart);
        }
    }

    os << "void initialize()";
    {
        CodeStream::Scope b(os);

        os << "unsigned long long deviceRNGSeed = 0;" << std::endl;

        // If any sort of on-device global RNG is required
        const bool simRNGRequired = std::any_of(model.getNeuronGroups().cbegin(), model.getNeuronGroups().cend(),
                                                [](const ModelSpec::NeuronGroupValueType &n) { return n.second.isSimRNGRequired(); });
        const bool globalDeviceRNGRequired = isGlobalDeviceRNGRequired(modelMerged);
        if(simRNGRequired || globalDeviceRNGRequired) {
            // If no seed is specified
            if (model.getSeed() == 0) {
                CodeStream::Scope b(os);

                // Use system randomness to generate one unsigned long long worth of seed words
                os << "std::random_device seedSource;" << std::endl;
                os << "uint32_t *deviceRNGSeedWord = reinterpret_cast<uint32_t*>(&deviceRNGSeed);" << std::endl;
                os << "for(int i = 0; i < " << sizeof(unsigned long long) / sizeof(uint32_t) << "; i++)";
                {
                    CodeStream::Scope b(os);
                    os << "deviceRNGSeedWord[i] = seedSource();" << std::endl;
                }
            }
            // Otherwise, use model seed
            else {
                os << "deviceRNGSeed = " << model.getSeed() << ";" << std::endl;
            }

            // If global RNG is required, launch kernel to initalize it
            if (globalDeviceRNGRequired) {
                os << "initializeRNGKernel<<<1, 1>>>(deviceRNGSeed);" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
            }
        }

        // Loop through all synapse groups
        // **TODO** this logic belongs in BackendSIMT
        // **TODO** apply merging to this process - large models could generate thousands of lines of code here
        for(const auto &s : model.getSynapseGroups()) {
            // If this synapse population has BITMASK connectivity and is intialised on device, insert a call to cudaMemset to zero the whole bitmask
            if(s.second.isSparseConnectivityInitRequired() && s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                const size_t gpSize = ceilDivide((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * getSynapticMatrixRowStride(s.second), 32);
                os << "CHECK_CUDA_ERRORS(cudaMemset(d_gp" << s.first << ", 0, " << gpSize << " * sizeof(uint32_t)));" << std::endl;
            }
            
            // If this synapse population has SPARSE connectivity and column-based on device connectivity, insert a call to cudaMemset to zero row lengths
            // **NOTE** we could also use this code path for row-based connectivity but, leaving this in the kernel is much better as it gets merged
            if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE && !s.second.getConnectivityInitialiser().getSnippet()->getColBuildCode().empty() 
               && !s.second.isWeightSharingSlave()) 
            {
                os << "CHECK_CUDA_ERRORS(cudaMemset(d_rowLength" << s.first << ", 0, " << s.second.getSrcNeuronGroup()->getNumNeurons() << " * sizeof(unsigned int)));" << std::endl;
            }

            // If this synapse population has SPARSE connectivity and has postsynaptic learning, insert a call to cudaMemset to zero column lengths
            if((s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getLearnPostCode().empty()) {
                os << "CHECK_CUDA_ERRORS(cudaMemset(d_colLength" << s.first << ", 0, " << s.second.getTrgNeuronGroup()->getNumNeurons() << " * sizeof(unsigned int)));" << std::endl;
            }
        }

        // Push any required EGPs
        initPushEGPHandler(os);

        // If there are any initialisation threads
        if(idInitStart > 0) {
            CodeStream::Scope b(os);
            {
                Timer t(os, "init", model.isTimingEnabled(), true);

                genKernelDimensions(os, KernelInitialize, idInitStart, 1);
                os << KernelNames[KernelInitialize] << "<<<grid, threads>>>(deviceRNGSeed);" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
            }
        }
    }
    os << std::endl;
    os << "void initializeSparse()";
    {
        CodeStream::Scope b(os);

        // Push any required EGPs
        initSparsePushEGPHandler(os);

        // Copy all uninitialised state variables to device
        if(!getPreferences().automaticCopy) {
            os << "copyStateToDevice(true);" << std::endl;
            os << "copyConnectivityToDevice(true);" << std::endl << std::endl;
        }

        // If there are any sparse initialisation threads
        if(idSparseInitStart > 0) {
            CodeStream::Scope b(os);
            {
                Timer t(os, "initSparse", model.isTimingEnabled(), true);

                genKernelDimensions(os, KernelInitializeSparse, idSparseInitStart, 1);
                os << KernelNames[KernelInitializeSparse] << "<<<grid, threads>>>();" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &) const
{
    os << "// Standard C++ includes" << std::endl;
    os << "#include <random>" << std::endl;
    os << "#include <string>" << std::endl;
    os << "#include <stdexcept>" << std::endl;
    os << std::endl;
    os << "// Standard C includes" << std::endl;
    os << "#include <cassert>" << std::endl;
    os << "#include <cstdint>" << std::endl;

    // If NCCL is enabled, export ncclGetUniqueId function
    if(getPreferences<Preferences>().enableNCCLReductions) {
        os << "extern \"C\" {" << std::endl;
        os << "EXPORT_VAR const unsigned int ncclUniqueIDBytes;" << std::endl;
        os << "EXPORT_FUNC void ncclGenerateUniqueID();" << std::endl;
        os << "EXPORT_FUNC void ncclInitCommunicator(int rank, int numRanks);" << std::endl;
        os << "EXPORT_FUNC unsigned char *ncclGetUniqueID();" << std::endl;
        os << "}" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged &) const
{
    os << "// CUDA includes" << std::endl;
    os << "#include <curand_kernel.h>" << std::endl;
    if(getRuntimeVersion() >= 9000) {
        os <<"#include <cuda_fp16.h>" << std::endl;
    }

    // If NCCL is enabled
    if(getPreferences<Preferences>().enableNCCLReductions) {
        // Include NCCL header
        os << "#include <nccl.h>" << std::endl;
        os << std::endl;
        // Define NCCL ID and communicator
        os << "EXPORT_VAR ncclUniqueId ncclID;" << std::endl;
        os << "EXPORT_VAR ncclComm_t ncclCommunicator;" << std::endl;
        os << std::endl;
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Helper macro for error-checking NCCL calls" << std::endl;
        os << "#define CHECK_NCCL_ERRORS(call) {\\" << std::endl;
        os << "    ncclResult_t error = call;\\" << std::endl;
        os << "    if (error != ncclSuccess) {\\" << std::endl;
        os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": nccl error \" + std::to_string(error) + \": \" + ncclGetErrorString(error));\\" << std::endl;
        os << "    }\\" << std::endl;
        os << "}" << std::endl;
    }

    os << std::endl;
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper macro for error-checking CUDA calls" << std::endl;
    os << "#define CHECK_CUDA_ERRORS(call) {\\" << std::endl;
    os << "    cudaError_t error = call;\\" << std::endl;
    os << "    if (error != cudaSuccess) {\\" << std::endl;
    os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": cuda error \" + std::to_string(error) + \": \" + cudaGetErrorString(error));\\" << std::endl;
    os << "    }\\" << std::endl;
    os << "}" << std::endl;
    os << std::endl;
    os << "#define SUPPORT_CODE_FUNC __device__ __host__ inline" << std::endl;
    os << std::endl;

    // If device is older than SM 6 or we're using a version of CUDA older than 8
    if ((getChosenCUDADevice().major < 6) || (getRuntimeVersion() < 8000)){
        os << "// software version of atomic add for double precision" << std::endl;
        os << "__device__ inline double atomicAddSW(double* address, double val)";
        {
            CodeStream::Scope b(os);
            os << "unsigned long long int* address_as_ull = (unsigned long long int*)address;" << std::endl;
            os << "unsigned long long int old = *address_as_ull, assumed;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "assumed = old;" << std::endl;
                os << "old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));" << std::endl;
            }
            os << "while (assumed != old);" << std::endl;
            os << "return __longlong_as_double(old);" << std::endl;
        }
        os << std::endl;
    }

    // If we're using a CUDA device with SM < 2
    if (getChosenCUDADevice().major < 2) {
        os << "// software version of atomic add for single precision float" << std::endl;
        os << "__device__ inline float atomicAddSW(float* address, float val)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "int* address_as_ull = (int*)address;" << std::endl;
            os << "int old = *address_as_ull, assumed;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "assumed = old;" << std::endl;
                os << "old = atomicCAS(address_as_ull, assumed, __float_as_int(val + __int_as_float(assumed)));" << std::endl;
            }
            os << "while (assumed != old);" << std::endl;
            os << "return __int_as_float(old);" << std::endl;
        }
        os << std::endl;
    }
    os << std::endl;
    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline float exponentialDistFloat(RNG *rng)";
    {
        CodeStream::Scope b(os);
        os << "while (true)";
        {
            CodeStream::Scope b(os);
            os << "const float u = curand_uniform(rng);" << std::endl;
            os << "if (u != 0.0f)";
            {
                CodeStream::Scope b(os);
                os << "return -logf(u);" << std::endl;
            }
        }
    }
    os << std::endl;
    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline double exponentialDistDouble(RNG *rng)";
    {
        CodeStream::Scope b(os);
        os << "while (true)";
        {
            CodeStream::Scope b(os);
            os << "const double u = curand_uniform_double(rng);" << std::endl;
            os << "if (u != 0.0)";
            {
                CodeStream::Scope b(os);
                os << "return -log(u);" << std::endl;
            }
        }
    }
    os << std::endl;

    // Generate gamma-distributed variates using Marsaglia and Tsang's method
    // G. Marsaglia and W. Tsang. A simple method for generating gamma variables. ACM Transactions on Mathematical Software, 26(3):363-372, 2000.
    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline float gammaDistFloatInternal(RNG *rng, float c, float d)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "float x, v, u;" << std::endl;
        os << "while (true)";
        {
            CodeStream::Scope b(os);
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "x = curand_normal(rng);" << std::endl;
                os << "v = 1.0f + c*x;" << std::endl;
            }
            os << "while (v <= 0.0f);" << std::endl;
            os << std::endl;
            os << "v = v*v*v;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "u = curand_uniform(rng);" << std::endl;
            }
            os << "while (u == 1.0f);" << std::endl;
            os << std::endl;
            os << "if (u < 1.0f - 0.0331f*x*x*x*x) break;" << std::endl;
            os << "if (logf(u) < 0.5f*x*x + d*(1.0f - v + logf(v))) break;" << std::endl;
        }
        os << std::endl;
        os << "return d*v;" << std::endl;
    }
    os << std::endl;
    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline float gammaDistFloat(RNG *rng, float a)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "if (a > 1)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "const float u = curand_uniform (rng);" << std::endl;
            os << "const float d = (1.0f + a) - 1.0f / 3.0f;" << std::endl;
            os << "const float c = (1.0f / 3.0f) / sqrtf(d);" << std::endl;
            os << "return gammaDistFloatInternal (rng, c, d) * powf(u, 1.0f / a);" << std::endl;
        }
        os << "else" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "const float d = a - 1.0f / 3.0f;" << std::endl;
            os << "const float c = (1.0f / 3.0f) / sqrtf(d);" << std::endl;
            os << "return gammaDistFloatInternal(rng, c, d);" << std::endl;
        }
    }
    os << std::endl;

    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline float gammaDistDoubleInternal(RNG *rng, double c, double d)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "double x, v, u;" << std::endl;
        os << "while (true)";
        {
            CodeStream::Scope b(os);
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "x = curand_normal_double(rng);" << std::endl;
                os << "v = 1.0 + c*x;" << std::endl;
            }
            os << "while (v <= 0.0);" << std::endl;
            os << std::endl;
            os << "v = v*v*v;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "u = curand_uniform_double(rng);" << std::endl;
            }
            os << "while (u == 1.0);" << std::endl;
            os << std::endl;
            os << "if (u < 1.0 - 0.0331*x*x*x*x) break;" << std::endl;
            os << "if (log(u) < 0.5*x*x + d*(1.0 - v + log(v))) break;" << std::endl;
        }
        os << std::endl;
        os << "return d*v;" << std::endl;
    }
    os << std::endl;

    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline float gammaDistDouble(RNG *rng, double a)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "if (a > 1.0)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "const double u = curand_uniform (rng);" << std::endl;
            os << "const double d = (1.0 + a) - 1.0 / 3.0;" << std::endl;
            os << "const double c = (1.0 / 3.0) / sqrt(d);" << std::endl;
            os << "return gammaDistDoubleInternal (rng, c, d) * pow(u, 1.0 / a);" << std::endl;
        }
        os << "else" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "const float d = a - 1.0 / 3.0;" << std::endl;
            os << "const float c = (1.0 / 3.0) / sqrt(d);" << std::endl;
            os << "return gammaDistDoubleInternal(rng, c, d);" << std::endl;
        }
    }
    os << std::endl;

    // The following code is an almost exact copy of numpy's
    // rk_binomial_inversion function (numpy/random/mtrand/distributions.c)
    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline unsigned int binomialDistFloatInternal(RNG *rng, unsigned int n, float p)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "const float q = 1.0f - p;" << std::endl;
        os << "const float qn = expf(n * logf(q));" << std::endl;
        os << "const float np = n * p;" << std::endl;
        os << "const unsigned int bound = min(n, (unsigned int)(np + (10.0f * sqrtf((np * q) + 1.0f))));" << std::endl;

        os << "unsigned int x = 0;" << std::endl;
        os << "float px = qn;" << std::endl;
        os << "float u = curand_uniform(rng);" << std::endl;
        os << "while(u > px)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "x++;" << std::endl;
            os << "if(x > bound)";
            {
                CodeStream::Scope b(os);
                os << "x = 0;" << std::endl;
                os << "px = qn;" << std::endl;
                os << "u = curand_uniform(rng);" << std::endl;
            }
            os << "else";
            {
                CodeStream::Scope b(os);
                os << "u -= px;" << std::endl;
                os << "px = ((n - x + 1) * p * px) / (x * q);" << std::endl;
            }
        }
        os << "return x;" << std::endl;
    }
    os << std::endl;

    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline unsigned int binomialDistFloat(RNG *rng, unsigned int n, float p)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "if(p <= 0.5f)";
        {
            CodeStream::Scope b(os);
            os << "return binomialDistFloatInternal(rng, n, p);" << std::endl;

        }
        os << "else";
        {
            CodeStream::Scope b(os);
            os << "return (n - binomialDistFloatInternal(rng, n, 1.0f - p));" << std::endl;
        }
    }

    // The following code is an almost exact copy of numpy's
    // rk_binomial_inversion function (numpy/random/mtrand/distributions.c)
    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline unsigned int binomialDistDoubleInternal(RNG *rng, unsigned int n, double p)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "const double q = 1.0 - p;" << std::endl;
        os << "const double qn = exp(n * log(q));" << std::endl;
        os << "const double np = n * p;" << std::endl;
        os << "const unsigned int bound = min(n, (unsigned int)(np + (10.0 * sqrt((np * q) + 1.0))));" << std::endl;

        os << "unsigned int x = 0;" << std::endl;
        os << "double px = qn;" << std::endl;
        os << "double u = curand_uniform_double(rng);" << std::endl;
        os << "while(u > px)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "x++;" << std::endl;
            os << "if(x > bound)";
            {
                CodeStream::Scope b(os);
                os << "x = 0;" << std::endl;
                os << "px = qn;" << std::endl;
                os << "u = curand_uniform_double(rng);" << std::endl;
            }
            os << "else";
            {
                CodeStream::Scope b(os);
                os << "u -= px;" << std::endl;
                os << "px = ((n - x + 1) * p * px) / (x * q);" << std::endl;
            }
        }
        os << "return x;" << std::endl;
    }
    os << std::endl;

    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline unsigned int binomialDistDouble(RNG *rng, unsigned int n, double p)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "if(p <= 0.5)";
        {
            CodeStream::Scope b(os);
            os << "return binomialDistDoubleInternal(rng, n, p);" << std::endl;

        }
        os << "else";
        {
            CodeStream::Scope b(os);
            os << "return (n - binomialDistDoubleInternal(rng, n, 1.0 - p));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genRunnerPreamble(CodeStream &os, const ModelSpecMerged&, const MemAlloc&) const
{
#ifdef _WIN32
    // **YUCK** on Windows, disable "function assumed not to throw an exception but does" warning
    // Setting /Ehs SHOULD solve this but CUDA rules don't give this option and it's not clear it gets through to the compiler anyway
    os << "#pragma warning(disable: 4297)" << std::endl;
#endif

     // If NCCL is enabled
    if(getPreferences<Preferences>().enableNCCLReductions) {
        // Define NCCL ID and communicator
        os << "ncclUniqueId ncclID;" << std::endl;
        os << "ncclComm_t ncclCommunicator;" << std::endl;

        // Define constant to expose NCCL_UNIQUE_ID_BYTES
        os << "const unsigned int ncclUniqueIDBytes = NCCL_UNIQUE_ID_BYTES;" << std::endl;

        // Define wrapper to generate a unique NCCL ID
        os << std::endl;
        os << "void ncclGenerateUniqueID()";
        {
            CodeStream::Scope b(os);
            os << "CHECK_NCCL_ERRORS(ncclGetUniqueId(&ncclID));" << std::endl;
        }
        os << std::endl;
        os << "unsigned char *ncclGetUniqueID()";
        {
            CodeStream::Scope b(os);
            os << "return reinterpret_cast<unsigned char*>(&ncclID);" << std::endl;
        }
        os << std::endl;
        os << "void ncclInitCommunicator(int rank, int numRanks)";
        {
            CodeStream::Scope b(os);
            os << "CHECK_NCCL_ERRORS(ncclCommInitRank(&ncclCommunicator, numRanks, ncclID, rank));" << std::endl;
        }
        os << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc&) const
{
    // If the model requires zero-copy
    if(modelMerged.getModel().zeroCopyInUse()) {
        // If device doesn't support mapping host memory error
        if(!getChosenCUDADevice().canMapHostMemory) {
            throw std::runtime_error("Device does not support mapping CPU host memory!");
        }

        // set appropriate device flags
        os << "CHECK_CUDA_ERRORS(cudaSetDeviceFlags(cudaDeviceMapHost));" << std::endl;
        os << std::endl;
    }

    // If we should select GPU by device ID, do so
    const bool runtimeDeviceSelect = (getPreferences<Preferences>().deviceSelectMethod == DeviceSelect::MANUAL_RUNTIME);
    if(getPreferences<Preferences>().selectGPUByDeviceID) {
        os << "CHECK_CUDA_ERRORS(cudaSetDevice(";
        if(runtimeDeviceSelect) {
            os << "deviceID";
        }
        else {
            os << m_ChosenDeviceID;
        }
        os << "));" << std::endl;
    }
    // Otherwise, write code to get device by PCI bus ID
    // **NOTE** this is required because device IDs are not guaranteed to remain the same and we want the code to be run on the same GPU it was optimise for
    else {
        os << "int deviceID;" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaDeviceGetByPCIBusId(&deviceID, ";
        if(runtimeDeviceSelect) {
            os << "pciBusID";
        }
        else {
            // Get chosen device's PCI bus ID and write into code
            char pciBusID[32];
            CHECK_CUDA_ERRORS(cudaDeviceGetPCIBusId(pciBusID, 32, m_ChosenDeviceID));
            os << "\"" << pciBusID << "\"";
        }
        os << "));" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));" << std::endl;
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genFreeMemPreamble(CodeStream &os, const ModelSpecMerged&) const
{
    // Free NCCL communicator
    if(getPreferences<Preferences>().enableNCCLReductions) {
        os << "CHECK_NCCL_ERRORS(ncclCommDestroy(ncclCommunicator));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    // Synchronise if automatic copying or zero-copy are in use
    // **THINK** Is this only required with automatic copy on older SM, CUDA and on non-Linux?
    if(getPreferences().automaticCopy || modelMerged.getModel().zeroCopyInUse()) {
        os << "CHECK_CUDA_ERRORS(cudaDeviceSynchronize());" << std::endl;
    }

    // If timing is enabled, synchronise last event
    if(modelMerged.getModel().isTimingEnabled()) {
        os << "CHECK_CUDA_ERRORS(cudaEventSynchronize(neuronUpdateStop));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const
{
    const bool deviceType = isDeviceType(type);

    if(getPreferences().automaticCopy && ::Utils::isTypePointer(type)) {
        // Export pointer, either in definitionsInternal if variable has a device type
        // or to definitions if it should be accessable on host
        CodeStream &d = deviceType ? definitionsInternal : definitions;
        d << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
    }
    else {
        if(loc & VarLocation::HOST) {
            if(deviceType) {
                throw std::runtime_error("Variable '" + name + "' is of device-only type '" + type + "' but is located on the host");
            }

            definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
        }
        if(loc & VarLocation::DEVICE) {
            // If the type is a pointer type we need a device pointer
            if(::Utils::isTypePointer(type)) {
                // Write host definition to internal definitions stream if type is device only
                CodeStream &d = deviceType ? definitionsInternal : definitions;
                d << "EXPORT_VAR " << type << " d_" << name << ";" << std::endl;
            }
            // Otherwise we just need a device variable, made volatile for safety
            else {
                definitionsInternal << "EXPORT_VAR __device__ volatile " << type << " d_" << name << ";" << std::endl;
            }
        }
    }


}
//--------------------------------------------------------------------------
void Backend::genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    if(getPreferences().automaticCopy && ::Utils::isTypePointer(type)) {
        os << type << " " << name << ";" << std::endl;
    }
    else {
        if(loc & VarLocation::HOST) {
            os << type << " " << name << ";" << std::endl;
        }
        if(loc & VarLocation::DEVICE) {
            // If the type is a pointer type we need a host and a device pointer
            if(::Utils::isTypePointer(type)) {
                os << type << " d_" << name << ";" << std::endl;
            }
            // Otherwise we just need a device variable, made volatile for safety
            else {
                os << "__device__ volatile " << type << " d_" << name << ";" << std::endl;
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count, MemAlloc &memAlloc) const
{
    if(getPreferences().automaticCopy) {
        os << "CHECK_CUDA_ERRORS(cudaMallocManaged(&" << name << ", " << count << " * sizeof(" << type << ")));" << std::endl;
        memAlloc += MemAlloc::device(count * getSize(type));
    }
    else {
        if(loc & VarLocation::HOST) {
            const char *flags = (loc & VarLocation::ZERO_COPY) ? "cudaHostAllocMapped" : "cudaHostAllocPortable";
            os << "CHECK_CUDA_ERRORS(cudaHostAlloc(&" << name << ", " << count << " * sizeof(" << type << "), " << flags << "));" << std::endl;
            memAlloc += MemAlloc::host(count * getSize(type));
        }

        // If variable is present on device at all
        if(loc & VarLocation::DEVICE) {
            // Insert call to correct helper depending on whether variable should be allocated in zero-copy mode or not
            if(loc & VarLocation::ZERO_COPY) {
                os << "CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void **)&d_" << name << ", (void *)" << name << ", 0));" << std::endl;
                memAlloc += MemAlloc::zeroCopy(count * getSize(type));
            }
            else {
                os << "CHECK_CUDA_ERRORS(cudaMalloc(&d_" << name << ", " << count << " * sizeof(" << type << ")));" << std::endl;
                memAlloc += MemAlloc::device(count * getSize(type));
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const
{
    if(getPreferences().automaticCopy) {
        os << "CHECK_CUDA_ERRORS(cudaFree(" << name << "));" << std::endl;
    }
    else {
        // **NOTE** because we pinned the variable we need to free it with cudaFreeHost rather than use the host code generator
        if(loc & VarLocation::HOST) {
            os << "CHECK_CUDA_ERRORS(cudaFreeHost(" << name << "));" << std::endl;
        }

        // If this variable wasn't allocated in zero-copy mode, free it
        if((loc & VarLocation::DEVICE) && !(loc & VarLocation::ZERO_COPY)) {
            os << "CHECK_CUDA_ERRORS(cudaFree(d_" << name << "));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamDefinition(CodeStream &definitions, CodeStream &, 
                                            const std::string &type, const std::string &name, VarLocation loc) const
{
    if(getPreferences().automaticCopy) {
        definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
    }
    else {
        if(loc & VarLocation::HOST) {
            definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
        }
        if(loc & VarLocation::DEVICE && ::Utils::isTypePointer(type)) {
            definitions << "EXPORT_VAR " << type << " d_" << name << ";" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    if(getPreferences().automaticCopy) {
        os << type << " " << name << ";" << std::endl;
    }
    else {
        if(loc & VarLocation::HOST) {
            os << type << " " << name << ";" << std::endl;
        }
        if(loc & VarLocation::DEVICE && ::Utils::isTypePointer(type)) {
            os << type << " d_" << name << ";" << std::endl;
        }
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
    const std::string hostPointerToPointer = pointerToPointer ? (prefix + name) : ("&" + prefix + name);
    const std::string devicePointerToPointer = pointerToPointer ? (prefix + "d_" + name) : ("&" + prefix + "d_" + name);
    if(getPreferences().automaticCopy) {
        os << "CHECK_CUDA_ERRORS(cudaMallocManaged(" << hostPointerToPointer << ", " << countVarName << " * sizeof(" << underlyingType << ")));" << std::endl;
    }
    else {
        if(loc & VarLocation::HOST) {
            const char *flags = (loc & VarLocation::ZERO_COPY) ? "cudaHostAllocMapped" : "cudaHostAllocPortable";
            os << "CHECK_CUDA_ERRORS(cudaHostAlloc(" << hostPointerToPointer << ", " << countVarName << " * sizeof(" << underlyingType << "), " << flags << "));" << std::endl;
        }

        // If variable is present on device at all
        if(loc & VarLocation::DEVICE) {
            if(loc & VarLocation::ZERO_COPY) {
                os << "CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void**)" << devicePointerToPointer << ", (void*)" << hostPointer << ", 0));" << std::endl;
            }
            else {
                os << "CHECK_CUDA_ERRORS(cudaMalloc(" << devicePointerToPointer << ", " << countVarName << " * sizeof(" << underlyingType << ")));" << std::endl;
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name, 
                                      VarLocation loc, const std::string &countVarName, const std::string &prefix) const
{
    assert(!getPreferences().automaticCopy);

    if(!(loc & VarLocation::ZERO_COPY)) {
        // Get underlying type
        const std::string underlyingType = ::Utils::getUnderlyingType(type);
        const bool pointerToPointer = ::Utils::isTypePointerToPointer(type);

        const std::string hostPointer = pointerToPointer ? ("*" + prefix + name) : (prefix + name);
        const std::string devicePointer = pointerToPointer ? ("*" + prefix + "d_" + name) : (prefix + "d_" + name);

        os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << devicePointer;
        os << ", " << hostPointer;
        os << ", " << countVarName << " * sizeof(" << underlyingType << "), cudaMemcpyHostToDevice));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPull(CodeStream &os, const std::string &type, const std::string &name, 
                                      VarLocation loc, const std::string &countVarName, const std::string &prefix) const
{
    assert(!getPreferences().automaticCopy);

    if(!(loc & VarLocation::ZERO_COPY)) {
        // Get underlying type
        const std::string underlyingType = ::Utils::getUnderlyingType(type);
        const bool pointerToPointer = ::Utils::isTypePointerToPointer(type);

        const std::string hostPointer = pointerToPointer ? ("*" + prefix + name) : (prefix + name);
        const std::string devicePointer = pointerToPointer ? ("*" + prefix + "d_" + name) : (prefix + "d_" + name);

        os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << hostPointer;
        os << ", " << devicePointer;
        os << ", " << countVarName << " * sizeof(" << underlyingType << "), cudaMemcpyDeviceToHost));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genMergedExtraGlobalParamPush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx,
                                            const std::string &groupIdx, const std::string &fieldName,
                                            const std::string &egpName) const
{
    const std::string structName = "Merged" + suffix + "Group" + std::to_string(mergedGroupIdx);
    os << "CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_merged" << suffix << "Group" << mergedGroupIdx;
    os << ", &" << egpName << ", sizeof(" << egpName << ")";
    os << ", (sizeof(" << structName << ") * (" << groupIdx << ")) + offsetof(" << structName << ", " << fieldName << ")));" << std::endl;
}
//--------------------------------------------------------------------------
std::string Backend::getMergedGroupFieldHostType(const std::string &type) const
{
    return type;
}
//--------------------------------------------------------------------------
void Backend::genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const
{
    assert(!getPreferences().automaticCopy);

    if(!(loc & VarLocation::ZERO_COPY)) {
        // Only copy if uninitialisedOnly isn't set
        if(autoInitialized) {
            os << "if(!uninitialisedOnly)" << CodeStream::OB(1101);
        }

        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << name;
        os << ", " << name;
        os << ", " << count << " * sizeof(" << type << "), cudaMemcpyHostToDevice));" << std::endl;

        if(autoInitialized) {
            os << CodeStream::CB(1101);
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const
{
    assert(!getPreferences().automaticCopy);

    if(!(loc & VarLocation::ZERO_COPY)) {
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << name;
        os << ", d_"  << name;
        os << ", " << count << " * sizeof(" << type << "), cudaMemcpyDeviceToHost));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, 
                                     const std::string &name, VarLocation loc, unsigned int batchSize) const
{
    assert(!getPreferences().automaticCopy);

    // If this variable requires queuing and isn't zero-copy
    if(ng.isVarQueueRequired(name) && ng.isDelayRequired() && !(loc & VarLocation::ZERO_COPY)) {
        // If batch size is one, generate 1D memcpy to copy current timestep's data
        if(batchSize == 1) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", " << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", " << ng.getNumNeurons() << " * sizeof(" << type << "), cudaMemcpyHostToDevice));" << std::endl;
        }
        // Otherwise, perform a 2D memcpy to copy current timestep's data from each batch
        else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy2D(d_" << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", " << ng.getNumNeurons() * ng.getNumDelaySlots() << " * sizeof(" << type << ")";
            os << ", " << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", " << ng.getNumNeurons() * ng.getNumDelaySlots() << " * sizeof(" << type << ")";
            os << ", " << ng.getNumNeurons() << " * sizeof(" << type << ")";
            os << ", " << batchSize << ", cudaMemcpyHostToDevice));" << std::endl;
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
    if(ng.isVarQueueRequired(name) && ng.isDelayRequired() && !(loc & VarLocation::ZERO_COPY)) {
        // If batch size is one, generate 1D memcpy to copy current timestep's data
        if(batchSize == 1) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", d_" << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", " << ng.getNumNeurons() << " * sizeof(" << type << "), cudaMemcpyDeviceToHost));" << std::endl;
        }
        else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy2D(" << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", " << ng.getNumNeurons() * ng.getNumDelaySlots() << " * sizeof(" << type << ")";
            os << ", d_" << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", " << ng.getNumNeurons() * ng.getNumDelaySlots() << " * sizeof(" << type << ")";
            os << ", " << ng.getNumNeurons() << " * sizeof(" << type << ")";
            os << ", " << batchSize << ", cudaMemcpyDeviceToHost));" << std::endl;
        }
    }
    // Otherwise, generate standard pull
    else {
        genVariablePull(os, type, name + ng.getName(), loc, ng.getNumNeurons() * batchSize);
    }
}
//--------------------------------------------------------------------------
void Backend::genGlobalDeviceRNG(CodeStream &, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &, CodeStream &, MemAlloc &memAlloc) const
{
    // Define global Phillox RNG
    // **NOTE** this is actually accessed as a global so, unlike other variables, needs device global
    definitionsInternal << "EXPORT_VAR __device__ curandStatePhilox4_32_10_t d_rng;" << std::endl;

    // Implement global Phillox RNG
    runner << "__device__ curandStatePhilox4_32_10_t d_rng;" << std::endl;

    memAlloc += MemAlloc::device(getSize("curandStatePhilox4_32_10_t"));
}
//--------------------------------------------------------------------------
void Backend::genPopulationRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                                   const std::string &name, size_t count, MemAlloc &memAlloc) const
{
    // Create an array or XORWOW RNGs
    genArray(definitions, definitionsInternal, runner, allocations, free, "curandState", name, VarLocation::DEVICE, count, memAlloc);
}
//--------------------------------------------------------------------------
void Backend::genTimer(CodeStream &, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                       CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const
{
    // Define CUDA start and stop events in internal defintions (as they use CUDA-specific types)
    definitionsInternal << "EXPORT_VAR cudaEvent_t " << name << "Start;" << std::endl;
    definitionsInternal << "EXPORT_VAR cudaEvent_t " << name << "Stop;" << std::endl;

    // Implement start and stop event variables
    runner << "cudaEvent_t " << name << "Start;" << std::endl;
    runner << "cudaEvent_t " << name << "Stop;" << std::endl;

    // Create start and stop events in allocations
    allocations << "CHECK_CUDA_ERRORS(cudaEventCreate(&" << name << "Start));" << std::endl;
    allocations << "CHECK_CUDA_ERRORS(cudaEventCreate(&" << name << "Stop));" << std::endl;

    // Destroy start and stop events in allocations
    free << "CHECK_CUDA_ERRORS(cudaEventDestroy(" << name << "Start));" << std::endl;
    free << "CHECK_CUDA_ERRORS(cudaEventDestroy(" << name << "Stop));" << std::endl;

    if(updateInStepTime) {
        CodeGenerator::CodeStream::Scope b(stepTimeFinalise);
        stepTimeFinalise << "float tmp;" << std::endl;
        stepTimeFinalise << "CHECK_CUDA_ERRORS(cudaEventElapsedTime(&tmp, " << name << "Start, " << name << "Stop));" << std::endl;
        stepTimeFinalise << name << "Time += tmp / 1000.0;" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genReturnFreeDeviceMemoryBytes(CodeStream &os) const
{
    os << "size_t free;" << std::endl;
    os << "size_t total;" << std::endl;
    os << "CHECK_CUDA_ERRORS(cudaMemGetInfo(&free, &total));" << std::endl;
    os << "return free;" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefilePreamble(std::ostream &os) const
{
    const std::string architecture = "sm_" + std::to_string(getChosenCUDADevice().major) + std::to_string(getChosenCUDADevice().minor);
    std::string linkFlags = "--shared -arch " + architecture;
    
    // If NCCL reductions are enabled, link NCCL
    if(getPreferences<Preferences>().enableNCCLReductions) {
        linkFlags += " -lnccl";
    }
    // Write variables to preamble
    os << "CUDA_PATH ?=/usr/local/cuda" << std::endl;
    os << "NVCC := $(CUDA_PATH)/bin/nvcc" << std::endl;
    os << "NVCCFLAGS := " << getNVCCFlags() << std::endl;
    os << "LINKFLAGS := " << linkFlags << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileLinkRule(std::ostream &os) const
{
    os << "\t@$(NVCC) $(LINKFLAGS) -o $@ $(OBJECTS)" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileCompileRule(std::ostream &os) const
{
    // Add one rule to generate dependency files from cc files
    os << "%.d: %.cc" << std::endl;
    os << "\t@$(NVCC) -M $(NVCCFLAGS) $< 1> $@" << std::endl;
    os << std::endl;

    // Add another to build object files from cc files
    os << "%.o: %.cc %.d" << std::endl;
    os << "\t@$(NVCC) -dc $(NVCCFLAGS) $<" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildConfigProperties(std::ostream &os) const
{
    // Add property to extract CUDA path
    os << "\t\t<!-- **HACK** determine the installed CUDA version by regexing CUDA path -->" << std::endl;
    os << "\t\t<CudaVersion>$([System.Text.RegularExpressions.Regex]::Match($(CUDA_PATH), \"\\\\v([0-9.]+)$\").Groups[1].Value)</CudaVersion>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportProps(std::ostream &os) const
{
    // Import CUDA props file
    os << "\t<ImportGroup Label=\"ExtensionSettings\">" << std::endl;
    os << "\t\t<Import Project=\"$(VCTargetsPath)\\BuildCustomizations\\CUDA $(CudaVersion).props\" />" << std::endl;
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
    os << "\t\t\t<PreprocessorDefinitions Condition=\"'$(Configuration)'=='Release'\">WIN32;WIN64;NDEBUG;_CONSOLE;BUILDING_GENERATED_CODE;%(PreprocessorDefinitions)</PreprocessorDefinitions>" << std::endl;
    os << "\t\t\t<PreprocessorDefinitions Condition=\"'$(Configuration)'=='Debug'\">WIN32;WIN64;_DEBUG;_CONSOLE;BUILDING_GENERATED_CODE;%(PreprocessorDefinitions)</PreprocessorDefinitions>" << std::endl;
    os << "\t\t</ClCompile>" << std::endl;

    // Add item definition for linking
    os << "\t\t<Link>" << std::endl;
    os << "\t\t\t<GenerateDebugInformation>true</GenerateDebugInformation>" << std::endl;
    os << "\t\t\t<EnableCOMDATFolding Condition=\"'$(Configuration)'=='Release'\">true</EnableCOMDATFolding>" << std::endl;
    os << "\t\t\t<OptimizeReferences Condition=\"'$(Configuration)'=='Release'\">true</OptimizeReferences>" << std::endl;
    os << "\t\t\t<SubSystem>Console</SubSystem>" << std::endl;
    os << "\t\t\t<AdditionalDependencies>cudart_static.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>" << std::endl;
    os << "\t\t</Link>" << std::endl;

    // Add item definition for CUDA compilation
    // **YUCK** the CUDA Visual Studio plugin build system demands that you specify both a virtual an actual architecture 
    // (which NVCC itself doesn't require). While, in general, actual architectures are usable as virtual architectures, 
    // there is no compute_21 so we need to replace that with compute_20
    const std::string architecture = std::to_string(getChosenCUDADevice().major) + std::to_string(getChosenCUDADevice().minor);
    const std::string virtualArchitecture = (architecture == "21") ? "20" : architecture;
    os << "\t\t<CudaCompile>" << std::endl;
    os << "\t\t\t<TargetMachinePlatform>64</TargetMachinePlatform>" << std::endl;
    os << "\t\t\t<GenerateRelocatableDeviceCode>true</GenerateRelocatableDeviceCode>" << std::endl;
    os << "\t\t\t<CodeGeneration>compute_" << virtualArchitecture <<",sm_" << architecture << "</CodeGeneration>" << std::endl;
    os << "\t\t\t<FastMath>" << (getPreferences().optimizeCode ? "true" : "false") << "</FastMath>" << std::endl;
    os << "\t\t\t<GenerateLineInfo>" << (getPreferences<Preferences>().generateLineInfo ? "true" : "false") << "</GenerateLineInfo>" << std::endl;
    os << "\t\t</CudaCompile>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const
{
    os << "\t\t<CudaCompile Include=\"" << moduleName << ".cc\" >" << std::endl;
    // **YUCK** for some reasons you can't call .Contains on %(BaseCommandLineTemplate) directly
    // Solution suggested by https://stackoverflow.com/questions/9512577/using-item-functions-on-metadata-values
    os << "\t\t\t<AdditionalOptions Condition=\" !$([System.String]::new('%(BaseCommandLineTemplate)').Contains('-x cu')) \">-x cu %(AdditionalOptions)</AdditionalOptions>" << std::endl;
    os << "\t\t</CudaCompile>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportTarget(std::ostream &os) const
{
    os << "\t<ImportGroup Label=\"ExtensionTargets\">" << std::endl;
    os << "\t\t<Import Project=\"$(VCTargetsPath)\\BuildCustomizations\\CUDA $(CudaVersion).targets\" />" << std::endl;
    os << "\t</ImportGroup>" << std::endl;
}
//--------------------------------------------------------------------------
std::string Backend::getAllocateMemParams(const ModelSpecMerged &) const
{
    // If device should be selected at runtime
    if(getPreferences<Preferences>().deviceSelectMethod == DeviceSelect::MANUAL_RUNTIME) {
        // If devices should be delected by ID, add an integer parameter
        if(getPreferences<Preferences>().selectGPUByDeviceID) {
            return "int deviceID";
        }
        // Otherwise, add a pci bus ID parameter
        else {
            return "const char *pciBusID";
        }
    }
    // Othewise, no parameters are required
    else {
        return "";
    }
}
//--------------------------------------------------------------------------
Backend::MemorySpaces Backend::getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const
{
    // Get size of update group start ids (constant cache is precious so don't use for init groups
    const size_t groupStartIDSize = (getGroupStartIDSize(modelMerged.getMergedNeuronUpdateGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedPresynapticUpdateGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedPostsynapticUpdateGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedSynapseDynamicsGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedCustomUpdateGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedCustomUpdateWUGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedCustomUpdateTransposeWUGroups()));

    // Return available constant memory and to
    return {{"__device__ __constant__", (groupStartIDSize > getChosenDeviceSafeConstMemBytes()) ? 0 : (getChosenDeviceSafeConstMemBytes() - groupStartIDSize)},
            {"__device__", m_ChosenDevice.totalGlobalMem}};
}
//--------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Backend::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash was name of backend
    Utils::updateHash("CUDA", hash);
    
    // Update hash with chosen device ID and kernel block sizes
    Utils::updateHash(m_ChosenDeviceID, hash);
    Utils::updateHash(getKernelBlockSize(), hash);

    // Update hash with preferences
    getPreferences<Preferences>().updateHash(hash);

    return hash.get_digest();
}
//--------------------------------------------------------------------------
std::string Backend::getNVCCFlags() const
{
    // **NOTE** now we don't include runner.cc when building standalone modules we get loads of warnings about
    // How you hide device compiler warnings is totally non-documented but https://stackoverflow.com/a/17095910/1476754
    // holds the answer! For future reference --display_error_number option can be used to get warning ids to use in --diag-supress
    // HOWEVER, on CUDA 7.5 and 8.0 this causes a fatal error and, as no warnings are shown when --diag-suppress is removed,
    // presumably this is because this warning simply wasn't implemented until CUDA 9
    const std::string architecture = "sm_" + std::to_string(getChosenCUDADevice().major) + std::to_string(getChosenCUDADevice().minor);
    std::string nvccFlags = "-x cu -arch " + architecture;
#ifndef _WIN32
    nvccFlags += " -std=c++11 --compiler-options \"-fPIC -Wno-return-type-c-linkage\"";
#endif
    if(m_RuntimeVersion >= 9020) {
        nvccFlags += " -Xcudafe \"--diag_suppress=extern_entity_treated_as_static\"";
    }

    nvccFlags += " " + getPreferences<Preferences>().userNvccFlags;
    if(getPreferences().optimizeCode) {
        nvccFlags += " -O3 -use_fast_math";
    }
    if(getPreferences().debugCode) {
        nvccFlags += " -O0 -g -G";
    }
    if(getPreferences<Preferences>().showPtxInfo) {
        nvccFlags += " -Xptxas \"-v\"";
    }
    if(getPreferences<Preferences>().generateLineInfo) {
        nvccFlags += " --generate-line-info";
    }
#ifdef MPI_ENABLE
    // If MPI is enabled, add MPI include path
    nvccFlags +=" -I\"$(MPI_PATH)/include\"";
#endif
    return nvccFlags;
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikePush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize, bool spikeEvent) const
{
    assert(!getPreferences().automaticCopy);

    if(!(ng.getSpikeLocation() & VarLocation::ZERO_COPY)) {
        // Is delay required
        const bool delayRequired = spikeEvent ?
            ng.isDelayRequired() :
            (ng.isTrueSpikeRequired() && ng.isDelayRequired());

        const char *spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
        const char *spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

        if (delayRequired) {
            // If there's only a single batch
            if(batchSize == 1) {
                // Copy spike count for current timestep
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", " << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
                
                // Copy this many spikes from current timestep
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikePrefix << ng.getName() << " + (spkQuePtr" << ng.getName() << "*" << ng.getNumNeurons() << ")";
                os << ", " << spikePrefix << ng.getName();
                os << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
                os << ", " << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << "] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
            }
            else {
                // Copy spike count for current timestep  from each batch using 2D memcpy
                os << "CHECK_CUDA_ERRORS(cudaMemcpy2D(d_" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", " << ng.getNumDelaySlots() << " * sizeof(unsigned int)";
                os << ", " << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", " << ng.getNumDelaySlots() << " * sizeof(unsigned int)";
                os << ", sizeof(unsigned int), " << batchSize << ", cudaMemcpyHostToDevice));" << std::endl;

                // Loop through batches and launch asynchronous memcpys to copy spikes from each one
                os << "for(unsigned int b = 0; b < " << batchSize << "; b++)";
                {
                    CodeStream::Scope b(os);
                    os << "const unsigned int spikeOffset = (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ") + (b * " << (ng.getNumNeurons() * ng.getNumDelaySlots()) << ");" << std::endl;
                    os << "CHECK_CUDA_ERRORS(cudaMemcpyAsync(d_" << spikePrefix << ng.getName() << " + spikeOffset";
                    os << ", " << spikePrefix << ng.getName() << " + spikeOffset";
                    os << ", " << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << " + (b * " << ng.getNumDelaySlots() << ")] * sizeof(unsigned int)";
                    os << ", cudaMemcpyHostToDevice)); " << std::endl;
                }

                // Wait until queued copies have completed
                os << "CHECK_CUDA_ERRORS(cudaStreamSynchronize(0));" << std::endl;
            }
        }
        else {
            // Copy the spike count for each batch
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikeCntPrefix << ng.getName();
            os << ", " << spikeCntPrefix << ng.getName();
            os << ", " << batchSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;

            // If there's only a single batch, copy spikes
            if(batchSize == 1) {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikePrefix << ng.getName();
                os << ", " << spikePrefix << ng.getName();
                os << ", " << spikeCntPrefix << ng.getName() << "[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
            }
            // Otherwise, loop through batches and launch asynchronous memcpys to copy spikes from each one
            else {
                os << "for(unsigned int b = 0; b < " << batchSize << "; b++)";
                {
                    CodeStream::Scope b(os);
                    os << "CHECK_CUDA_ERRORS(cudaMemcpyAsync(d_" << spikePrefix << ng.getName() << " + (b * " << ng.getNumNeurons() << ")";
                    os << ", " << spikePrefix << ng.getName() << " + (b * " << ng.getNumNeurons() << ")";
                    os << ", " << spikeCntPrefix << ng.getName() << "[b] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
                }

                // Wait until queued copies have completed
                os << "CHECK_CUDA_ERRORS(cudaStreamSynchronize(0));" << std::endl;
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikePull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize, bool spikeEvent) const
{
    if(!(ng.getSpikeLocation() & VarLocation::ZERO_COPY)) {
        // Is delay required
        const bool delayRequired = spikeEvent ?
            ng.isDelayRequired() :
            (ng.isTrueSpikeRequired() && ng.isDelayRequired());

        const char *spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
        const char *spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

        if (delayRequired) {
            // If there's only a single batch
            if(batchSize == 1) {
                // Copy spike count for current timestep
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", d_" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;

                // Copy this many spikes from current timestep
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikePrefix << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
                os << ", d_" << spikePrefix << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
                os << ", " << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << "] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }
            else {
                // Copy spike count for current timestep  from each batch using 2D memcpy
                os << "CHECK_CUDA_ERRORS(cudaMemcpy2D(" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", " << ng.getNumDelaySlots() << " * sizeof(unsigned int)";
                os << ", d_" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", " << ng.getNumDelaySlots() << " * sizeof(unsigned int)";
                os << ", sizeof(unsigned int), " << batchSize << ", cudaMemcpyDeviceToHost));" << std::endl;

                // Loop through batches and launch asynchronous memcpys to copy spikes from each one
                os << "for(unsigned int b = 0; b < " << batchSize << "; b++)";
                {
                    CodeStream::Scope b(os);
                    os << "const unsigned int spikeOffset = (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ") + (b * " << (ng.getNumNeurons() * ng.getNumDelaySlots()) << ");" << std::endl;
                    os << "CHECK_CUDA_ERRORS(cudaMemcpyAsync(" << spikePrefix << ng.getName() << " + spikeOffset";
                    os << ", d_" << spikePrefix << ng.getName() << " + spikeOffset";
                    os << ", " << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << " + (b * " << ng.getNumDelaySlots() << ")] * sizeof(unsigned int)";
                    os << ", cudaMemcpyDeviceToHost)); " << std::endl;
                }

                // Wait until queued copies have completed
                os << "CHECK_CUDA_ERRORS(cudaStreamSynchronize(0));" << std::endl;
            }
        }
        else {
            // Copy the spike count for each batch
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikeCntPrefix << ng.getName();
            os << ", d_" << spikeCntPrefix << ng.getName();
            os << ", " << batchSize << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;

            // If there's only a single batch, copy spikes
            if(batchSize == 1) {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikePrefix << ng.getName();
                os << ", d_" << spikePrefix << ng.getName();
                os << ", " << spikeCntPrefix << ng.getName() << "[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }
            // Otherwise, loop through batches and launch asynchronous memcpys to copy spikes from each one
            else {
                os << "for(unsigned int b = 0; b < " << batchSize << "; b++)";
                {
                    CodeStream::Scope b(os);
                    os << "CHECK_CUDA_ERRORS(cudaMemcpyAsync(" << spikePrefix << ng.getName() << " + (b * " << ng.getNumNeurons() << ")";
                    os << ", d_" << spikePrefix << ng.getName() << " + (b * " << ng.getNumNeurons() << ")";
                    os << ", " << spikeCntPrefix << ng.getName() << "[b] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
                }

                // Wait until queued copies have completed
                os << "CHECK_CUDA_ERRORS(cudaStreamSynchronize(0));" << std::endl;
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreadsX, size_t batchSize, size_t numBlockThreadsY) const
{
    // Calculate grid size
    const size_t gridSize = ceilDivide(numThreadsX, getKernelBlockSize(kernel));
    assert(gridSize < (size_t)getChosenCUDADevice().maxGridSize[0]);
    assert(numBlockThreadsY < (size_t)getChosenCUDADevice().maxThreadsDim[0]);

    os << "const dim3 threads(" << getKernelBlockSize(kernel) << ", " << numBlockThreadsY << ");" << std::endl;
    if(numBlockThreadsY > 1) {
        assert(batchSize < (size_t)getChosenCUDADevice().maxThreadsDim[2]);
        os << "const dim3 grid(" << gridSize << ", 1, " << batchSize << ");" << std::endl;
    }
    else {
        assert(batchSize < (size_t)getChosenCUDADevice().maxThreadsDim[1]);
        os << "const dim3 grid(" << gridSize << ", " << batchSize << ");" << std::endl;
    }
}
}   // namespace CUDA
}   // namespace CodeGenerator

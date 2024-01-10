#include "backend.h"

// Standard C++ includes
#include <algorithm>
#include <iterator>

// CUDA includes
#include <curand_kernel.h>

// GeNN includes
#include "gennUtils.h"
#include "logging.h"
#include "type.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/standardLibrary.h"

// CUDA backend includes
#include "utils.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const EnvironmentLibrary::Library floatRandomFunctions = {
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Float, {}), "curand_uniform(&$(_rng))"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Float, {}), "curand_normal(&$(_rng))"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Float, {}), "exponentialDistFloat(&$(_rng))"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Float, {Type::Float, Type::Float}), "curand_log_normal_float(&$(_rng), $(0), $(1))"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Float, {Type::Float}), "gammaDistFloat(&$(_rng), $(0))"}},
    {"gennrand_binomial", {Type::ResolvedType::createFunction(Type::Uint32, {Type::Uint32, Type::Float}), "binomialDistFloat(&$(_rng), $(0), $(1))"}},
};

const EnvironmentLibrary::Library doubleRandomFunctions = {
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Double, {}), "curand_uniform_double(&$(_rng))"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Double, {}), "curand_normal_double(&$(_rng))"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Double, {}), "exponentialDistDouble(&$(_rng))"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Double, {Type::Double, Type::Double}), "curand_log_normal_double(&$(_rng), $(0), $(1))"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Double, {Type::Double}), "gammaDistDouble(&$(_rng), $(0))"}},
    {"gennrand_binomial", {Type::ResolvedType::createFunction(Type::Uint32, {Type::Uint32, Type::Double}), "binomialDistDouble(&$(_rng), $(0), $(1))"}},
};

//--------------------------------------------------------------------------
// CUDADeviceType
//--------------------------------------------------------------------------
const Type::ResolvedType CURandState = Type::ResolvedType::createValue<curandState>("curandState", Type::Qualifier{0}, nullptr, true);
const Type::ResolvedType CURandStatePhilox43210 = Type::ResolvedType::createValue<curandStatePhilox4_32_10_t>("curandStatePhilox4_32_10_t", Type::Qualifier{0}, nullptr, true);

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

//--------------------------------------------------------------------------
// Array
//--------------------------------------------------------------------------
class Array : public Runtime::ArrayBase
{
public:
    Array(const Type::ResolvedType &type, size_t count, 
          VarLocation location, bool uninitialized)
    :   ArrayBase(type, count, location, uninitialized), m_DevicePointer(nullptr)
    {
        // Allocate if count is specified
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
        // Set count
        setCount(count);

        // Malloc host pointer
        if(getLocation() & VarLocation::HOST) {
            const unsigned int flags = (getLocation() & VarLocation::ZERO_COPY) ? cudaHostAllocMapped : cudaHostAllocPortable;

            std::byte *hostPointer = nullptr;
            CHECK_CUDA_ERRORS(cudaHostAlloc(&hostPointer, getSizeBytes(), flags));
            setHostPointer(hostPointer);
        }

        // If variable is present on device at all
        if(getLocation() & VarLocation::DEVICE) {
            // Insert call to correct helper depending on whether variable should be allocated in zero-copy mode or not
            if(getLocation() & VarLocation::ZERO_COPY) {
                CHECK_CUDA_ERRORS(cudaHostGetDevicePointer(&m_DevicePointer, getHostPointer(), 0));
            }
            else {
                CHECK_CUDA_ERRORS(cudaMalloc(&m_DevicePointer, getSizeBytes()));
            }
        }
    }

    //! Free array
    virtual void free() final
    {
        // **NOTE** because we pinned the variable we need to free it with cudaFreeHost rather than use free
        if(getLocation() & VarLocation::HOST) {
            CHECK_CUDA_ERRORS(cudaFreeHost(getHostPointer()));
            setHostPointer(nullptr);
        }

        // If this variable wasn't allocated in zero-copy mode, free it
        if((getLocation() & VarLocation::DEVICE) && !(getLocation() & VarLocation::ZERO_COPY)) {
            CHECK_CUDA_ERRORS(cudaFree(getDevicePointer()));
            m_DevicePointer = nullptr;
        }

        // Zero count
        setCount(0);
    }
    //! Copy entire array to device
    virtual void pushToDevice() final
    {
        if(!(getLocation() & VarLocation::DEVICE) || !(getLocation() & VarLocation::HOST)) {
            throw std::runtime_error("Cannot push array that isn't present on host and device");
        }

        if(!(getLocation() & VarLocation::ZERO_COPY)) {
            CHECK_CUDA_ERRORS(cudaMemcpy(getDevicePointer(), getHostPointer(), getSizeBytes(), cudaMemcpyHostToDevice));
        }
    }

    //! Copy entire array from device
    virtual void pullFromDevice() final
    {
        if(!(getLocation() & VarLocation::DEVICE) || !(getLocation() & VarLocation::HOST)) {
            throw std::runtime_error("Cannot pull array that isn't present on host and device");
        }

        if(!(getLocation() & VarLocation::ZERO_COPY)) {
            CHECK_CUDA_ERRORS(cudaMemcpy(getHostPointer(), getDevicePointer(), getSizeBytes(), cudaMemcpyDeviceToHost));
        }
    }

    //! Copy a 1D slice of elements to device 
    /*! \param offset   Offset in elements to start copying from
        \param count    Number of elements to copy*/
    virtual void pushSlice1DToDevice(size_t offset, size_t count) final
    {
        if(!(getLocation() & VarLocation::DEVICE) || !(getLocation() & VarLocation::HOST)) {
            throw std::runtime_error("Cannot push array that isn't present on host and device");
        }

        if(!(getLocation() & VarLocation::ZERO_COPY)) {
            // If end of slice overflows array, give error
            if((offset + count) > getCount()) {
                throw std::runtime_error("Cannot pull slice that overflows array");
            }

            // Convert offset and count to bytes and copy
            const size_t offsetBytes = offset * getType().getValue().size;
            const size_t countBytes = count * getType().getValue().size;
            CHECK_CUDA_ERRORS(cudaMemcpy(getDevicePointer() + offsetBytes, getHostPointer() + offsetBytes, 
                                         countBytes, cudaMemcpyHostToDevice));
        }
    }

    //! Copy a 1D slice of elements from device 
    /*! \param offset   Offset in elements to start copying from
        \param count    Number of elements to copy*/
    virtual void pullSlice1DFromDevice(size_t offset, size_t count) final
    {
        if(!(getLocation() & VarLocation::DEVICE) || !(getLocation() & VarLocation::HOST)) {
            throw std::runtime_error("Cannot pull array that isn't present on host and device");
        }

        if(!(getLocation() & VarLocation::ZERO_COPY)) {
            // If end of slice overflows array, give error
            if((offset + count) > getCount()) {
                throw std::runtime_error("Cannot pull slice that overflows array");
            }

            // Convert offset and count to bytes and copy
            const size_t offsetBytes = offset * getType().getValue().size;
            const size_t countBytes = count * getType().getValue().size;
            CHECK_CUDA_ERRORS(cudaMemcpy(getHostPointer() + offsetBytes, getDevicePointer() + offsetBytes, 
                                         countBytes, cudaMemcpyDeviceToHost));
        }

    }

    //! Memset the host pointer
    virtual void memsetDeviceObject(int value) final
    {
        CHECK_CUDA_ERRORS(cudaMemset(m_DevicePointer, value, getSizeBytes()));
    }

    //! Serialise backend-specific device object to bytes
    virtual void serialiseDeviceObject(std::vector<std::byte> &bytes, bool pointerToPointer) const final
    {
        std::byte vBytes[sizeof(void*)];
        if(pointerToPointer) {
            std::byte* const *devicePointerPointer = &m_DevicePointer;
            std::memcpy(vBytes, &devicePointerPointer, sizeof(void*));
        }
        else {
            std::memcpy(vBytes, &m_DevicePointer, sizeof(void*));
        }
        std::copy(std::begin(vBytes), std::end(vBytes), std::back_inserter(bytes));
    }

    //! Serialise backend-specific host object to bytes
    virtual void serialiseHostObject(std::vector<std::byte>&, bool) const
    {
        throw std::runtime_error("CUDA arrays have no host objects");
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    std::byte *getDevicePointer() const{ return m_DevicePointer; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::byte *m_DevicePointer;
};

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
                      Args&&... args)
{
    // Loop through merged groups
    for(const auto &m : mergedGroups) {
        genGroupStartID(os, idStart, totalConstMem, m, getPaddedNumThreads);
    }

    // Generate any remaining groups
    genGroupStartIDs(os, idStart, totalConstMem, std::forward<Args>(args)...);
}
//-----------------------------------------------------------------------
template<typename ...Args>
void genMergedKernelDataStructures(CodeStream &os, size_t &totalConstMem,
                                   Args&&... args)
{
    // Generate group start id arrays
    size_t idStart = 0;
    genGroupStartIDs(os, std::ref(idStart), std::ref(totalConstMem), std::forward<Args>(args)...);
}
//-----------------------------------------------------------------------
void genFilteredGroupStartIDs(CodeStream &, size_t&, size_t&)
{
}
//-----------------------------------------------------------------------
template<typename T, typename G, typename F, typename ...Args>
void genFilteredGroupStartIDs(CodeStream &os, size_t &idStart, size_t &totalConstMem,
                              const std::vector<T> &mergedGroups, G getPaddedNumThreads, F filter,
                              Args&&... args)
{
    // Loop through merged groups
    for(const auto &m : mergedGroups) {
        if(filter(m)) {
            genGroupStartID(os, idStart, totalConstMem, m, getPaddedNumThreads);
        }
    }

    // Generate any remaining groups
    genFilteredGroupStartIDs(os, idStart, totalConstMem, std::forward<Args>(args)...);
}
//-----------------------------------------------------------------------
template<typename ...Args>
void genFilteredMergedKernelDataStructures(CodeStream &os, size_t &totalConstMem,
                                           Args&&... args)
{
    // Generate group start id arrays
    size_t idStart = 0;
    genFilteredGroupStartIDs(os, std::ref(idStart), std::ref(totalConstMem), std::forward<Args>(args)...);
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
const EnvironmentLibrary::Library &getRNGFunctions(const Type::ResolvedType &precision)
{
    if(precision == Type::Float) {
        return floatRandomFunctions;
    }
    else {
        assert(precision == Type::Double);
        return doubleRandomFunctions;
    }
}
}   // Anonymous namespace


//--------------------------------------------------------------------------
// GeNN::CodeGenerator::CUDA::State
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::CUDA
{
State::State(const Runtime::Runtime &runtime)
{
    // Lookup NCCL symbols
    m_NCCLGenerateUniqueID = (VoidFunction)runtime.getSymbol("ncclGenerateUniqueID", true);
    m_NCCLGetUniqueID = (BytePtrFunction)runtime.getSymbol("ncclGetUniqueID", true);
    m_NCCLInitCommunicator = (NCCLInitCommunicatorFunction)runtime.getSymbol("ncclInitCommunicator", true);
    m_NCCLUniqueIDSize = (size_t*)runtime.getSymbol("ncclUniqueIDSize", true);
}
//--------------------------------------------------------------------------
void State::ncclGenerateUniqueID()
{
    if(m_NCCLGenerateUniqueID == nullptr) {
        throw std::runtime_error("Cannot generate NCCL unique ID - model may not have been built with NCCL support");
    }
    m_NCCLGenerateUniqueID();
}
//--------------------------------------------------------------------------
unsigned char *State::ncclGetUniqueID()
{ 
    if(m_NCCLGetUniqueID == nullptr) {
        throw std::runtime_error("Cannot get NCCL unique ID - model may not have been built with NCCL support");
    }
    return m_NCCLGetUniqueID();
}
//--------------------------------------------------------------------------
size_t State::ncclGetUniqueIDSize() const
{
    if(m_NCCLUniqueIDSize == nullptr) {
        throw std::runtime_error("Cannot get NCCL unique ID size - model may not have been built with NCCL support");
    }
    
    return *m_NCCLUniqueIDSize;
}
//--------------------------------------------------------------------------    
void State::ncclInitCommunicator(int rank, int numRanks)
{
     if(m_NCCLInitCommunicator == nullptr) {
        throw std::runtime_error("Cannot initialise NCCL communicator - model may not have been built with NCCL support");
    }
    m_NCCLInitCommunicator(rank, numRanks);
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::CUDA::Backend
//--------------------------------------------------------------------------
Backend::Backend(const KernelBlockSize &kernelBlockSizes, const Preferences &preferences, 
                 int device, bool zeroCopy)
:   BackendSIMT(kernelBlockSizes, preferences), m_ChosenDeviceID(device)
{
    // Set device
    CHECK_CUDA_ERRORS(cudaSetDevice(device));

    // Get device properties
    CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&m_ChosenDevice, device));

    // Get CUDA runtime version
    cudaRuntimeGetVersion(&m_RuntimeVersion);

#ifdef _WIN32
    // If we're on Windows and NCCL is enabled, give error
    // **NOTE** There are several NCCL Windows ports e.g. https://github.com/MyCaffe/NCCL but we don't have access to any suitable systems to test
    if(getPreferences<Preferences>().enableNCCLReductions) {
        throw std::runtime_error("GeNN doesn't currently support NCCL on Windows");
    }
#endif

    // If the model requires zero-copy
    if(zeroCopy) {
        // If device doesn't support mapping host memory error
        if(!getChosenCUDADevice().canMapHostMemory) {
            throw std::runtime_error("Device does not support mapping CPU host memory!");
        }

        // Set map host device flag
        CHECK_CUDA_ERRORS(cudaSetDeviceFlags(cudaDeviceMapHost));
    }
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
std::string Backend::getAtomic(const Type::ResolvedType &type, AtomicOperation op, AtomicMemSpace) const
{
    // If operation is an atomic add
    if(op == AtomicOperation::ADD) {
        if(((getChosenCUDADevice().major < 2) && (type == Type::Float))
           || (((getChosenCUDADevice().major < 6) || (getRuntimeVersion() < 8000)) && (type == Type::Double)))
        {
            return "atomicAddSW";
        }

        return "atomicAdd";
    }
    // Otherwise, it's an atomic or
    else {
        assert(op == AtomicOperation::OR);
        assert(type == Type::Uint32 || type == Type::Int32);
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
std::string Backend::genPopulationRNGPreamble(CodeStream &, const std::string &globalRNG) const
{
    return "&" + globalRNG;
}
//--------------------------------------------------------------------------
void Backend::genPopulationRNGPostamble(CodeStream&, const std::string&) const
{
}
//--------------------------------------------------------------------------
std::string Backend::genGlobalRNGSkipAhead(CodeStream &os, const std::string &sequence) const
{
    // Skipahead RNG
    os << "curandStatePhilox4_32_10_t localRNG = d_rng;" << std::endl;
    os << "skipahead_sequence((unsigned long long)" << sequence << ", &localRNG);" << std::endl;
    return "localRNG";
}
//--------------------------------------------------------------------------
Type::ResolvedType Backend::getPopulationRNGType() const
{
    return CURandState;
}
//--------------------------------------------------------------------------
void Backend::genNeuronUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                              HostHandler preambleHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Generate stream with neuron update code
    std::ostringstream neuronUpdateStream;
    CodeStream neuronUpdate(neuronUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary neuronUpdateEnv(neuronUpdate, StandardLibrary::getMathsFunctions());

    // If any neuron groups require their previous spike times updating
    size_t idNeuronPrevSpikeTimeUpdate = 0;
    if(!modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups().empty()) {
        neuronUpdateEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << "(" << modelMerged.getModel().getTimePrecision().getName() << " t)";
        {
            CodeStream::Scope b(neuronUpdateEnv.getStream());

            EnvironmentExternal funcEnv(neuronUpdateEnv);
            funcEnv.add(model.getTimePrecision().addConst(), "t", "t");
            funcEnv.add(model.getTimePrecision().addConst(), "dt", 
                        Type::writeNumeric(model.getDT(), model.getTimePrecision()));
       
            funcEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelNeuronPrevSpikeTimeUpdate) << " * blockIdx.x + threadIdx.x;" << std::endl;
            if(model.getBatchSize() > 1) {
                funcEnv.add(Type::Uint32.addConst(), "batch", "batch",
                            {funcEnv.addInitialiser("const unsigned int batch = blockIdx.y;")});
            }
            else {
                funcEnv.add(Type::Uint32.addConst(), "batch", "0");
            }

            genNeuronPrevSpikeTimeUpdateKernel(funcEnv, modelMerged, memorySpaces, idNeuronPrevSpikeTimeUpdate);
        }
        neuronUpdateEnv.getStream() << std::endl;
    }

    // Generate reset kernel to be run before the neuron kernel
    size_t idNeuronSpikeQueueUpdate = 0;
    if(!modelMerged.getMergedNeuronSpikeQueueUpdateGroups().empty()) {
        neuronUpdateEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelNeuronSpikeQueueUpdate] << "()";
        {
            CodeStream::Scope b(neuronUpdateEnv.getStream());

            neuronUpdateEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelNeuronSpikeQueueUpdate) << " * blockIdx.x + threadIdx.x;" << std::endl;

            genNeuronSpikeQueueUpdateKernel(neuronUpdateEnv, modelMerged, memorySpaces, idNeuronSpikeQueueUpdate);
        }
        neuronUpdateEnv.getStream() << std::endl;
    }

    size_t idStart = 0;
    neuronUpdateEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelNeuronUpdate] << "(" << modelMerged.getModel().getTimePrecision().getName() << " t";
    if(model.isRecordingInUse()) {
        neuronUpdateEnv.getStream() << ", unsigned int recordingTimestep";
    }
    neuronUpdateEnv.getStream() << ")" << std::endl;
    {
        CodeStream::Scope b(neuronUpdateEnv.getStream());

        EnvironmentExternal funcEnv(neuronUpdateEnv);
        funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t");
        funcEnv.add(model.getTimePrecision().addConst(), "dt", 
                    Type::writeNumeric(model.getDT(), model.getTimePrecision()));
        funcEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelNeuronUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;
        if(model.getBatchSize() > 1) {
            funcEnv.add(Type::Uint32.addConst(), "batch", "batch",
                        {funcEnv.addInitialiser("const unsigned int batch = blockIdx.y;")});
        }
        else {
            funcEnv.add(Type::Uint32.addConst(), "batch", "0");
        }

        // Add RNG functions to environment and generate kernel
        EnvironmentLibrary rngEnv(funcEnv, getRNGFunctions(model.getPrecision()));
        genNeuronUpdateKernel(rngEnv, modelMerged, memorySpaces, idStart);
    }

    neuronUpdateEnv.getStream() << "void updateNeurons(" << modelMerged.getModel().getTimePrecision().getName() << " t";
    if(model.isRecordingInUse()) {
        neuronUpdateEnv.getStream() << ", unsigned int recordingTimestep";
    }
    neuronUpdateEnv.getStream() << ")";
    {
        CodeStream::Scope b(neuronUpdateEnv.getStream());

        if(idNeuronPrevSpikeTimeUpdate > 0) {
            CodeStream::Scope b(neuronUpdateEnv.getStream());
            genKernelDimensions(neuronUpdateEnv.getStream(), KernelNeuronPrevSpikeTimeUpdate, idNeuronPrevSpikeTimeUpdate, model.getBatchSize());
            neuronUpdateEnv.getStream() << KernelNames[KernelNeuronPrevSpikeTimeUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            neuronUpdateEnv.getStream() << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
        if(idNeuronSpikeQueueUpdate > 0) {
            CodeStream::Scope b(neuronUpdateEnv.getStream());
            genKernelDimensions(neuronUpdateEnv.getStream(), KernelNeuronSpikeQueueUpdate, idNeuronSpikeQueueUpdate, 1);
            neuronUpdateEnv.getStream() << KernelNames[KernelNeuronSpikeQueueUpdate] << "<<<grid, threads>>>();" << std::endl;
            neuronUpdateEnv.getStream() << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
        if(idStart > 0) {
            CodeStream::Scope b(neuronUpdateEnv.getStream());

            Timer t(neuronUpdateEnv.getStream(), "neuronUpdate", model.isTimingEnabled());

            genKernelDimensions(neuronUpdateEnv.getStream(), KernelNeuronUpdate, idStart, model.getBatchSize());
            neuronUpdateEnv.getStream() << KernelNames[KernelNeuronUpdate] << "<<<grid, threads>>>(t";
            if(model.isRecordingInUse()) {
                neuronUpdateEnv.getStream() << ", recordingTimestep";
            }
            neuronUpdateEnv.getStream() << ");" << std::endl;
            neuronUpdateEnv.getStream() << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
    }

    
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
    os << neuronUpdateStream.str();
}
//--------------------------------------------------------------------------
void Backend::genSynapseUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                               HostHandler preambleHandler) const
{
    // Generate stream with synapse update code
    std::ostringstream synapseUpdateStream;
    CodeStream synapseUpdate(synapseUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary synapseUpdateEnv(synapseUpdate, StandardLibrary::getMathsFunctions());

    // If any synapse groups require dendritic delay, a reset kernel is required to be run before the synapse kernel
    const ModelSpecInternal &model = modelMerged.getModel();
    size_t idSynapseDendricDelayUpdate = 0;
    if(!modelMerged.getMergedSynapseDendriticDelayUpdateGroups().empty()) {
        synapseUpdateEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelSynapseDendriticDelayUpdate] << "()";
        {
            CodeStream::Scope b(synapseUpdateEnv.getStream());

            synapseUpdateEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelSynapseDendriticDelayUpdate) << " * blockIdx.x + threadIdx.x;" << std::endl;
            genSynapseDendriticDelayUpdateKernel(synapseUpdateEnv, modelMerged, memorySpaces, idSynapseDendricDelayUpdate);
        }
        synapseUpdateEnv.getStream() << std::endl;
    }

    // If there are any presynaptic update groups
    size_t idPresynapticStart = 0;
    if(!modelMerged.getMergedPresynapticUpdateGroups().empty()) {
        synapseUpdateEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelPresynapticUpdate] << "(" << modelMerged.getModel().getTimePrecision().getName() << " t)" << std::endl; // end of synapse kernel header
        {
            CodeStream::Scope b(synapseUpdateEnv.getStream());

            EnvironmentExternal funcEnv(synapseUpdateEnv);
            funcEnv.add(model.getTimePrecision().addConst(), "t", "t");
            funcEnv.add(model.getTimePrecision().addConst(), "dt", 
                        Type::writeNumeric(model.getDT(), model.getTimePrecision()));
            funcEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelPresynapticUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;
            if(model.getBatchSize() > 1) {
                funcEnv.add(Type::Uint32.addConst(), "batch", "batch",
                            {funcEnv.addInitialiser("const unsigned int batch = blockIdx.y;")});
            }
            else {
                funcEnv.add(Type::Uint32.addConst(), "batch", "0");
            }

            // Add RNG functions to environment and generate kernel
            EnvironmentLibrary rngEnv(funcEnv, getRNGFunctions(model.getPrecision()));
            genPresynapticUpdateKernel(rngEnv, modelMerged, memorySpaces, idPresynapticStart);
        }
    }

    // If any synapse groups require postsynaptic learning
    size_t idPostsynapticStart = 0;
    if(!modelMerged.getMergedPostsynapticUpdateGroups().empty()) {
        synapseUpdateEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelPostsynapticUpdate] << "(" << modelMerged.getModel().getTimePrecision().getName() << " t)" << std::endl;
        {
            CodeStream::Scope b(synapseUpdateEnv.getStream());

            EnvironmentExternal funcEnv(synapseUpdateEnv);
            funcEnv.add(model.getTimePrecision().addConst(), "t", "t");
            funcEnv.add(model.getTimePrecision().addConst(), "dt", 
                        Type::writeNumeric(model.getDT(), model.getTimePrecision()));
            funcEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelPostsynapticUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;
            if(model.getBatchSize() > 1) {
                funcEnv.add(Type::Uint32.addConst(), "batch", "batch",
                            {funcEnv.addInitialiser("const unsigned int batch = blockIdx.y;")});
            }
            else {
                funcEnv.add(Type::Uint32.addConst(), "batch", "0");
            }
            genPostsynapticUpdateKernel(funcEnv, modelMerged, memorySpaces, idPostsynapticStart);
        }
    }
    
    // If any synapse groups require synapse dynamics
    size_t idSynapseDynamicsStart = 0;
    if(!modelMerged.getMergedSynapseDynamicsGroups().empty()) {
        synapseUpdateEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelSynapseDynamicsUpdate] << "(" << modelMerged.getModel().getTimePrecision().getName() << " t)" << std::endl; // end of synapse kernel header
        {
            CodeStream::Scope b(synapseUpdateEnv.getStream());

            EnvironmentExternal funcEnv(synapseUpdateEnv);
            funcEnv.add(model.getTimePrecision().addConst(), "t", "t");
            funcEnv.add(model.getTimePrecision().addConst(), "dt", 
                        Type::writeNumeric(model.getDT(), model.getTimePrecision()));
            funcEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelSynapseDynamicsUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;
            if(model.getBatchSize() > 1) {
                funcEnv.add(Type::Uint32.addConst(), "batch", "batch",
                            {funcEnv.addInitialiser("const unsigned int batch = blockIdx.y;")});
            }
            else {
                funcEnv.add(Type::Uint32.addConst(), "batch", "0");
            }
            genSynapseDynamicsKernel(funcEnv, modelMerged, memorySpaces, idSynapseDynamicsStart);
        }
    }

    synapseUpdateEnv.getStream() << "void updateSynapses(" << modelMerged.getModel().getTimePrecision().getName() << " t)";
    {
        CodeStream::Scope b(synapseUpdateEnv.getStream());

        // Launch pre-synapse reset kernel if required
        if(idSynapseDendricDelayUpdate > 0) {
            CodeStream::Scope b(synapseUpdateEnv.getStream());
            genKernelDimensions(synapseUpdateEnv.getStream(), KernelSynapseDendriticDelayUpdate, idSynapseDendricDelayUpdate, 1);
            synapseUpdateEnv.getStream() << KernelNames[KernelSynapseDendriticDelayUpdate] << "<<<grid, threads>>>();" << std::endl;
            synapseUpdateEnv.getStream() << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch synapse dynamics kernel if required
        if(idSynapseDynamicsStart > 0) {
            CodeStream::Scope b(synapseUpdateEnv.getStream());
            Timer t(synapseUpdateEnv.getStream(), "synapseDynamics", model.isTimingEnabled());

            genKernelDimensions(synapseUpdateEnv.getStream(), KernelSynapseDynamicsUpdate, idSynapseDynamicsStart, model.getBatchSize());
            synapseUpdateEnv.getStream() << KernelNames[KernelSynapseDynamicsUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            synapseUpdateEnv.getStream() << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch presynaptic update kernel
        if(idPresynapticStart > 0) {
            CodeStream::Scope b(synapseUpdateEnv.getStream());
            Timer t(synapseUpdateEnv.getStream(), "presynapticUpdate", model.isTimingEnabled());

            genKernelDimensions(synapseUpdateEnv.getStream(), KernelPresynapticUpdate, idPresynapticStart, model.getBatchSize());
            synapseUpdateEnv.getStream() << KernelNames[KernelPresynapticUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            synapseUpdateEnv.getStream() << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch postsynaptic update kernel
        if(idPostsynapticStart > 0) {
            CodeStream::Scope b(synapseUpdateEnv.getStream());
            Timer t(synapseUpdateEnv.getStream(), "postsynapticUpdate", model.isTimingEnabled());

            genKernelDimensions(synapseUpdateEnv.getStream(), KernelPostsynapticUpdate, idPostsynapticStart, model.getBatchSize());
            synapseUpdateEnv.getStream() << KernelNames[KernelPostsynapticUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            synapseUpdateEnv.getStream() << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
    }

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

    os << synapseUpdateStream.str();

}
//--------------------------------------------------------------------------
void Backend::genCustomUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                              HostHandler preambleHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Generate stream with synapse update code
    std::ostringstream customUpdateStream;
    CodeStream customUpdate(customUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary customUpdateEnv(customUpdate, StandardLibrary::getMathsFunctions());

    // Build set containing union of all custom update group names
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

    // Generate data structure for accessing merged groups
    // **THINK** I don't think there was any need for these to be filtered
    // **NOTE** constant cache is preferentially given to neuron and synapse groups as, typically, they are launched more often 
    // than custom update kernels so subtract constant memory requirements of synapse group start ids from total constant memory
    const size_t timestepGroupStartIDSize = (getGroupStartIDSize(modelMerged.getMergedPresynapticUpdateGroups()) +
                                             getGroupStartIDSize(modelMerged.getMergedPostsynapticUpdateGroups()) +
                                             getGroupStartIDSize(modelMerged.getMergedSynapseDynamicsGroups()) +
                                             getGroupStartIDSize(modelMerged.getMergedNeuronUpdateGroups()));
    size_t totalConstMem = (getChosenDeviceSafeConstMemBytes() > timestepGroupStartIDSize) ? (getChosenDeviceSafeConstMemBytes() - timestepGroupStartIDSize) : 0;

    // Loop through custom update groups
    for(const auto &g : customUpdateGroups) {
        // Generate kernel
        size_t idCustomUpdateStart = 0;
        if(std::any_of(modelMerged.getMergedCustomUpdateGroups().cbegin(), modelMerged.getMergedCustomUpdateGroups().cend(),
                       [&g](const auto &cg) { return (cg.getArchetype().getUpdateGroupName() == g); })
           || std::any_of(modelMerged.getMergedCustomUpdateWUGroups().cbegin(), modelMerged.getMergedCustomUpdateWUGroups().cend(),
                       [&g](const auto &cg) { return (cg.getArchetype().getUpdateGroupName() == g); })
           || std::any_of(modelMerged.getMergedCustomConnectivityUpdateGroups().cbegin(), modelMerged.getMergedCustomConnectivityUpdateGroups().cend(),
                          [&g](const auto &cg) { return (cg.getArchetype().getUpdateGroupName() == g); }))
        {
            genFilteredMergedKernelDataStructures(customUpdateEnv.getStream(), totalConstMem,
                                                  modelMerged.getMergedCustomUpdateGroups(),
                                                  [&model, this](const CustomUpdateInternal &cg){ return getPaddedNumCustomUpdateThreads(cg, model.getBatchSize()); },
                                                  [g](const CustomUpdateGroupMerged &cg){ return cg.getArchetype().getUpdateGroupName() == g; },

                                                  modelMerged.getMergedCustomUpdateWUGroups(),
                                                  [&model, this](const CustomUpdateWUInternal &cg){ return getPaddedNumCustomUpdateWUThreads(cg, model.getBatchSize()); },
                                                  [g](const CustomUpdateWUGroupMerged &cg){ return cg.getArchetype().getUpdateGroupName() == g; },
                                                  
                                                  modelMerged.getMergedCustomConnectivityUpdateGroups(),
                                                  [this](const CustomConnectivityUpdateInternal &cg){ return padKernelSize(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(), KernelCustomUpdate); },
                                                  [g](const CustomConnectivityUpdateGroupMerged &cg){ return cg.getArchetype().getUpdateGroupName() == g; });


            customUpdateEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelCustomUpdate] << g << "(" << modelMerged.getModel().getTimePrecision().getName() << " t)" << std::endl;
            {
                CodeStream::Scope b(customUpdateEnv.getStream());

                EnvironmentExternal funcEnv(customUpdateEnv);
                funcEnv.add(model.getTimePrecision().addConst(), "t", "t");
                funcEnv.add(model.getTimePrecision().addConst(), "dt", 
                            Type::writeNumeric(model.getDT(), model.getTimePrecision()));
                funcEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelCustomUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;

                funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
                funcEnv.getStream() << "// Custom updates" << std::endl;
                genCustomUpdateKernel(funcEnv, modelMerged, memorySpaces, g, idCustomUpdateStart);

                funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
                funcEnv.getStream() << "// Custom WU updates" << std::endl;
                genCustomUpdateWUKernel(funcEnv, modelMerged, memorySpaces, g, idCustomUpdateStart);
                
                funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
                funcEnv.getStream() << "// Custom connectivity updates" << std::endl;

                // Add RNG functions to environment and generate kernel
                EnvironmentLibrary rngEnv(funcEnv, getRNGFunctions(model.getPrecision()));
                genCustomConnectivityUpdateKernel(rngEnv, modelMerged, memorySpaces, g, idCustomUpdateStart);
            }
        }

        size_t idCustomTransposeUpdateStart = 0;
        if(std::any_of(modelMerged.getMergedCustomUpdateTransposeWUGroups().cbegin(), modelMerged.getMergedCustomUpdateTransposeWUGroups().cend(),
                       [&g](const auto &cg){ return (cg.getArchetype().getUpdateGroupName() == g); }))
        {
            genFilteredMergedKernelDataStructures(os, totalConstMem, modelMerged.getMergedCustomUpdateTransposeWUGroups(),
                                                  [&model, this](const CustomUpdateWUInternal &cg){ return getPaddedNumCustomUpdateTransposeWUThreads(cg, model.getBatchSize()); },
                                                  [g](const CustomUpdateTransposeWUGroupMerged &cg){ return cg.getArchetype().getUpdateGroupName() == g; });

            customUpdateEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelCustomTransposeUpdate] << g << "(" << modelMerged.getModel().getTimePrecision().getName() << " t)" << std::endl;
            {
                CodeStream::Scope b(customUpdateEnv.getStream());

                EnvironmentExternal funcEnv(customUpdateEnv);
                funcEnv.add(model.getTimePrecision().addConst(), "t", "t");
                funcEnv.add(model.getTimePrecision().addConst(), "dt", 
                            Type::writeNumeric(model.getDT(), model.getTimePrecision()));
                funcEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelCustomTransposeUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;

                funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
                funcEnv.getStream() << "// Custom WU transpose updates" << std::endl;
                genCustomTransposeUpdateWUKernel(funcEnv, modelMerged, memorySpaces, g, idCustomTransposeUpdateStart);
            }
        }
        customUpdateEnv.getStream() << "void update" << g << "(unsigned long long timestep)";
        {
            CodeStream::Scope b(customUpdateEnv.getStream());

            EnvironmentExternal funcEnv(customUpdateEnv);
            funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t",
                        {funcEnv.addInitialiser("const " + model.getTimePrecision().getName() + " t = timestep * " + Type::writeNumeric(model.getDT(), model.getTimePrecision()) + ";")});

            // Loop through host update groups and generate code for those in this custom update group
            modelMerged.genMergedCustomConnectivityHostUpdateGroups(
                *this, memorySpaces, g, 
                [this, &funcEnv](auto &c)
                {
                    c.generateUpdate(*this, funcEnv);
                });

            // Launch custom update kernel if required
            if(idCustomUpdateStart > 0) {
                CodeStream::Scope b(funcEnv.getStream());
                genKernelDimensions(funcEnv.getStream(), KernelCustomUpdate, idCustomUpdateStart, 1);
                Timer t(funcEnv.getStream(), "customUpdate" + g, model.isTimingEnabled());
                funcEnv.printLine(KernelNames[KernelCustomUpdate] + g + "<<<grid, threads>>>($(t));");
                funcEnv.printLine("CHECK_CUDA_ERRORS(cudaPeekAtLastError());");
            }

            // Launch custom transpose update kernel if required
            if(idCustomTransposeUpdateStart > 0) {
                CodeStream::Scope b(funcEnv.getStream());
                // **TODO** make block height parameterizable
                genKernelDimensions(funcEnv.getStream(), KernelCustomTransposeUpdate, idCustomTransposeUpdateStart, 1, 8);
                Timer t(funcEnv.getStream(), "customUpdate" + g + "Transpose", model.isTimingEnabled());
                funcEnv.printLine(KernelNames[KernelCustomTransposeUpdate]  + g + "<<<grid, threads>>>($(t));");
                funcEnv.printLine("CHECK_CUDA_ERRORS(cudaPeekAtLastError());");
            }

            // If NCCL reductions are enabled
            if(getPreferences<Preferences>().enableNCCLReductions) {
                // Loop through custom update host reduction groups and
                // generate reductions for those in this custom update group
                modelMerged.genMergedCustomUpdateHostReductionGroups(
                    *this, memorySpaces, g, 
                    [this, &funcEnv, &modelMerged](auto &cg)
                    {
                        genNCCLReduction(funcEnv, cg);
                    });

                // Loop through custom WU update host reduction groups and
                // generate reductions for those in this custom update group
                modelMerged.genMergedCustomWUUpdateHostReductionGroups(
                    *this, memorySpaces, g,
                    [this, &funcEnv, &modelMerged](auto &cg)
                    {
                        genNCCLReduction(funcEnv, cg);
                    });
            }

            // If timing is enabled
            if(model.isTimingEnabled()) {
                // Synchronise last event
                funcEnv.getStream() << "CHECK_CUDA_ERRORS(cudaEventSynchronize(customUpdate" << g;
                if(idCustomTransposeUpdateStart > 0) {
                    funcEnv.getStream() << "Transpose";
                }
                funcEnv.getStream() << "Stop)); " << std::endl;

                if(idCustomUpdateStart > 0) {
                    CodeGenerator::CodeStream::Scope b(funcEnv.getStream());
                    funcEnv.getStream() << "float tmp;" << std::endl;
                    funcEnv.getStream() << "CHECK_CUDA_ERRORS(cudaEventElapsedTime(&tmp, customUpdate" << g << "Start, customUpdate" << g << "Stop));" << std::endl;
                    funcEnv.getStream() << "customUpdate" << g << "Time += tmp / 1000.0;" << std::endl;
                }
                if(idCustomTransposeUpdateStart > 0) {
                    CodeGenerator::CodeStream::Scope b(funcEnv.getStream());
                    funcEnv.getStream() << "float tmp;" << std::endl;
                    funcEnv.getStream() << "CHECK_CUDA_ERRORS(cudaEventElapsedTime(&tmp, customUpdate" << g << "TransposeStart, customUpdate" << g << "TransposeStop));" << std::endl;
                    funcEnv.getStream() << "customUpdate" << g << "TransposeTime += tmp / 1000.0;" << std::endl;
                }
            }
        }
    }

    // Generate struct definitions
    modelMerged.genMergedCustomUpdateStructs(os, *this);
    modelMerged.genMergedCustomUpdateWUStructs(os, *this);
    modelMerged.genMergedCustomUpdateTransposeWUStructs(os, *this);
    modelMerged.genMergedCustomConnectivityUpdateStructs(os, *this);
    modelMerged.genMergedCustomConnectivityHostUpdateStructs(os, *this);

    // Generate arrays of merged structs and functions to push them
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateWUGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateTransposeWUGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomConnectivityUpdateGroups());
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

    // Generate stream with synapse update code
    std::ostringstream initStream;
    CodeStream init(initStream);

    // Begin environment with RNG library and standard library
    EnvironmentLibrary rngEnv(init, getRNGFunctions(model.getPrecision()));
    EnvironmentLibrary initEnv(rngEnv, StandardLibrary::getMathsFunctions());

    // If device RNG is required, generate kernel to initialise it
    if(isGlobalDeviceRNGRequired(model)) {
        initEnv.getStream() << "extern \"C\" __global__ void initializeRNGKernel(unsigned long long deviceRNGSeed)";
        {
            CodeStream::Scope b(initEnv.getStream());
            initEnv.getStream() << "if(threadIdx.x == 0)";
            {
                CodeStream::Scope b(initEnv.getStream());
                initEnv.getStream() << "curand_init(deviceRNGSeed, 0, 0, &d_rng);" << std::endl;
            }
        }
        initEnv.getStream() << std::endl;
    }

    // init kernel header
    initEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelInitialize] << "(unsigned long long deviceRNGSeed)";

    // initialization kernel code
    size_t idInitStart = 0;
    {
        // common variables for all cases
        CodeStream::Scope b(initEnv.getStream());
        
        EnvironmentExternal funcEnv(initEnv);
        funcEnv.add(model.getTimePrecision().addConst(), "dt", 
                    Type::writeNumeric(model.getDT(), model.getTimePrecision()));

        funcEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelInitialize) << " * blockIdx.x + threadIdx.x;" << std::endl;
        genInitializeKernel(funcEnv, modelMerged, memorySpaces, idInitStart);
    }
    const size_t numStaticInitThreads = idInitStart;

    // Sparse initialization kernel code
    size_t idSparseInitStart = 0;
    if(!modelMerged.getMergedSynapseSparseInitGroups().empty() 
       || !modelMerged.getMergedCustomWUUpdateSparseInitGroups().empty()
       || !modelMerged.getMergedCustomConnectivityUpdateSparseInitGroups().empty())
    {
        initEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelInitializeSparse] << "()";
        {
            CodeStream::Scope b(initEnv.getStream());

            EnvironmentExternal funcEnv(initEnv);
            funcEnv.add(model.getTimePrecision().addConst(), "dt", 
                        Type::writeNumeric(model.getDT(), model.getTimePrecision()));

            funcEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelInitializeSparse) << " * blockIdx.x + threadIdx.x;" << std::endl;
            genInitializeSparseKernel(funcEnv, modelMerged, numStaticInitThreads, memorySpaces, idSparseInitStart);
        }
    }

    initEnv.getStream() << "void initialize()";
    {
        CodeStream::Scope b(initEnv.getStream());

        initEnv.getStream() << "unsigned long long deviceRNGSeed = 0;" << std::endl;

        // If any sort of on-device global RNG is required
        const bool simRNGRequired = std::any_of(model.getNeuronGroups().cbegin(), model.getNeuronGroups().cend(),
                                                [](const ModelSpec::NeuronGroupValueType &n) { return n.second.isSimRNGRequired(); });
        const bool globalDeviceRNGRequired = isGlobalDeviceRNGRequired(model);
        if(simRNGRequired || globalDeviceRNGRequired) {
            // If no seed is specified
            if (model.getSeed() == 0) {
                CodeStream::Scope b(initEnv.getStream());

                // Use system randomness to generate one unsigned long long worth of seed words
                initEnv.getStream() << "std::random_device seedSource;" << std::endl;
                initEnv.getStream() << "uint32_t *deviceRNGSeedWord = reinterpret_cast<uint32_t*>(&deviceRNGSeed);" << std::endl;
                initEnv.getStream() << "for(int i = 0; i < " << sizeof(unsigned long long) / sizeof(uint32_t) << "; i++)";
                {
                    CodeStream::Scope b(initEnv.getStream());
                    initEnv.getStream() << "deviceRNGSeedWord[i] = seedSource();" << std::endl;
                }
            }
            // Otherwise, use model seed
            else {
                initEnv.getStream() << "deviceRNGSeed = " << model.getSeed() << ";" << std::endl;
            }

            // If global RNG is required, launch kernel to initalize it
            if (globalDeviceRNGRequired) {
                initEnv.getStream() << "initializeRNGKernel<<<1, 1>>>(deviceRNGSeed);" << std::endl;
                initEnv.getStream() << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
            }
        }

        // If there are any initialisation threads
        if(idInitStart > 0) {
            CodeStream::Scope b(initEnv.getStream());
            {
                Timer t(initEnv.getStream(), "init", model.isTimingEnabled(), true);

                genKernelDimensions(initEnv.getStream(), KernelInitialize, idInitStart, 1);
                initEnv.getStream() << KernelNames[KernelInitialize] << "<<<grid, threads>>>(deviceRNGSeed);" << std::endl;
                initEnv.getStream() << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
            }
        }
    }
    initEnv.getStream() << std::endl;
    initEnv.getStream() << "void initializeSparse()";
    {
        CodeStream::Scope b(initEnv.getStream());

        // If there are any sparse initialisation threads
        if(idSparseInitStart > 0) {
            CodeStream::Scope b(initEnv.getStream());
            {
                Timer t(initEnv.getStream(), "initSparse", model.isTimingEnabled(), true);

                genKernelDimensions(initEnv.getStream(), KernelInitializeSparse, idSparseInitStart, 1);
                initEnv.getStream() << KernelNames[KernelInitializeSparse] << "<<<grid, threads>>>();" << std::endl;
                initEnv.getStream() << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
            }
        }
    }

    os << "#include <iostream>" << std::endl;
    os << "#include <random>" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << std::endl;

    // Generate struct definitions
    modelMerged.genMergedNeuronInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseConnectivityInitGroupStructs(os, *this);
    modelMerged.genMergedSynapseSparseInitGroupStructs(os, *this);
    modelMerged.genMergedCustomUpdateInitGroupStructs(os, *this);
    modelMerged.genMergedCustomWUUpdateInitGroupStructs(os, *this);
    modelMerged.genMergedCustomWUUpdateSparseInitGroupStructs(os, *this);
    modelMerged.genMergedCustomConnectivityUpdatePreInitStructs(os, *this);
    modelMerged.genMergedCustomConnectivityUpdatePostInitStructs(os, *this);
    modelMerged.genMergedCustomConnectivityUpdateSparseInitStructs(os, *this);

    // Generate arrays of merged structs and functions to push them
    genMergedStructArrayPush(os, modelMerged.getMergedNeuronInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseConnectivityInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedSynapseSparseInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomWUUpdateInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomWUUpdateSparseInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomConnectivityUpdatePreInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomConnectivityUpdatePostInitGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomConnectivityUpdateSparseInitGroups());

    // Generate preamble
    preambleHandler(os);

    // Generate data structure for accessing merged groups from within initialisation kernel
    // **NOTE** pass in zero constant cache here as it's precious and would be wasted on init kernels which are only launched once
    size_t totalConstMem = 0;
    genMergedKernelDataStructures(
        os, totalConstMem,
        modelMerged.getMergedNeuronInitGroups(), [this](const NeuronGroupInternal &ng){ return padKernelSize(ng.getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedSynapseInitGroups(), [this](const SynapseGroupInternal &sg){ return padKernelSize(getNumInitThreads(sg), KernelInitialize); },
        modelMerged.getMergedCustomUpdateInitGroups(), [this](const CustomUpdateInternal &cg) { return padKernelSize(cg.getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedCustomConnectivityUpdatePreInitGroups(), [this](const CustomConnectivityUpdateInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedCustomConnectivityUpdatePostInitGroups(), [this](const CustomConnectivityUpdateInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedCustomWUUpdateInitGroups(), [this](const CustomUpdateWUInternal &cg){ return padKernelSize(getNumInitThreads(cg), KernelInitialize); },        
        modelMerged.getMergedSynapseConnectivityInitGroups(), [this](const SynapseGroupInternal &sg){ return padKernelSize(getNumConnectivityInitThreads(sg), KernelInitialize); });

    // Generate data structure for accessing merged groups from within sparse initialisation kernel
    genMergedKernelDataStructures(
        os, totalConstMem,
        modelMerged.getMergedSynapseSparseInitGroups(), [this](const SynapseGroupInternal &sg){ return padKernelSize(sg.getMaxConnections(), KernelInitializeSparse); },
        modelMerged.getMergedCustomWUUpdateSparseInitGroups(), [this](const CustomUpdateWUInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getMaxConnections(), KernelInitializeSparse); },
        modelMerged.getMergedCustomConnectivityUpdateSparseInitGroups(), [this](const CustomConnectivityUpdateInternal &cg){ return padKernelSize(cg.getSynapseGroup()->getMaxConnections(), KernelInitializeSparse); });
    os << std::endl;

    os << initStream.str();
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

        os << std::endl;
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Helper macro for error-checking NCCL calls" << std::endl;
        os << "#define CHECK_NCCL_ERRORS(call) {\\" << std::endl;
        os << "    ncclResult_t error = call;\\" << std::endl;
        os << "    if (error != ncclSuccess) {\\" << std::endl;
        os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": nccl error \" + std::to_string(error) + \": \" + ncclGetErrorString(error));\\" << std::endl;
        os << "    }\\" << std::endl;
        os << "}" << std::endl;

        // Define NCCL ID and communicator
        os << "extern ncclUniqueId ncclID;" << std::endl;
        os << "extern ncclComm_t ncclCommunicator;" << std::endl;

        // Export ncclGetUniqueId function
        os << "extern \"C\" {" << std::endl;
        os << "EXPORT_VAR const size_t ncclUniqueIDSize;" << std::endl;
        os << "EXPORT_FUNC void ncclGenerateUniqueID();" << std::endl;
        os << "EXPORT_FUNC void ncclInitCommunicator(int rank, int numRanks);" << std::endl;
        os << "EXPORT_FUNC unsigned char *ncclGetUniqueID();" << std::endl;
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
void Backend::genRunnerPreamble(CodeStream &os, const ModelSpecMerged&) const
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
        os << "const size_t ncclUniqueIDSize = NCCL_UNIQUE_ID_BYTES;" << std::endl;

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
void Backend::genAllocateMemPreamble(CodeStream&, const ModelSpecMerged&) const
{
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
    // Synchronise if zero-copy are in use
    if(modelMerged.getModel().zeroCopyInUse()) {
        os << "CHECK_CUDA_ERRORS(cudaDeviceSynchronize());" << std::endl;
    }

    // If timing is enabled, synchronise last event
    if(modelMerged.getModel().isTimingEnabled()) {
        os << "CHECK_CUDA_ERRORS(cudaEventSynchronize(neuronUpdateStop));" << std::endl;
    }
}
//--------------------------------------------------------------------------
std::unique_ptr<GeNN::Runtime::StateBase> Backend::createState(const Runtime::Runtime &runtime) const
{
    return std::make_unique<State>(runtime);
}
//--------------------------------------------------------------------------
std::unique_ptr<Runtime::ArrayBase> Backend::createArray(const Type::ResolvedType &type, size_t count, 
                                                         VarLocation location, bool uninitialized) const
{
    return std::make_unique<Array>(type, count, location, uninitialized);
}
//--------------------------------------------------------------------------
std::unique_ptr<Runtime::ArrayBase> Backend::createPopulationRNG(size_t count) const
{
    return createArray(CURandState, count, VarLocation::DEVICE, false);
}
//--------------------------------------------------------------------------
void Backend::genLazyVariableDynamicAllocation(CodeStream &os, 
                                               const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                               const std::string &countVarName) const
{
    const auto &underlyingType = type.isPointer() ? *type.getPointer().valueType : type;
    const std::string hostPointer = type.isPointer() ? ("*$(_" + name + ")") : ("$(_" + name + ")");
    const std::string hostPointerToPointer = type.isPointer() ? ("$(_" + name + ")") : ("&$(_" + name + ")");
    const std::string devicePointerToPointer = type.isPointer() ? ("$(_d_" + name + ")") : ("&$(_d_" + name + ")");
   
    if(loc & VarLocation::HOST) {
        const char *flags = (loc & VarLocation::ZERO_COPY) ? "cudaHostAllocMapped" : "cudaHostAllocPortable";
        os << "CHECK_CUDA_ERRORS(cudaHostAlloc(" << hostPointerToPointer << ", " << countVarName << " * sizeof(" << underlyingType.getName() << "), " << flags << "));" << std::endl;
    }

    // If variable is present on device at all
    if(loc & VarLocation::DEVICE) {
        if(loc & VarLocation::ZERO_COPY) {
            os << "CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void**)" << devicePointerToPointer << ", (void*)" << hostPointer << ", 0));" << std::endl;
        }
        else {
            os << "CHECK_CUDA_ERRORS(cudaMalloc(" << devicePointerToPointer << ", " << countVarName << " * sizeof(" << underlyingType.getName() << ")));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genLazyVariableDynamicPush(CodeStream &os, 
                                         const Type::ResolvedType &type, const std::string &name,
                                         VarLocation loc, const std::string &countVarName) const
{
    if(!(loc & VarLocation::ZERO_COPY)) {
        if (type.isPointer()) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(*$(_d_" << name << "), *$(_" << name << "), ";
            os << countVarName << " * sizeof(" << type.getPointer().valueType->getName() << "), cudaMemcpyHostToDevice));" << std::endl;
        }
        else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy($(_d_" << name << "), $(_" << name << "), ";
            os << countVarName << " * sizeof(" << type.getName() << "), cudaMemcpyHostToDevice));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genLazyVariableDynamicPull(CodeStream &os, 
                                         const Type::ResolvedType &type, const std::string &name,
                                         VarLocation loc, const std::string &countVarName) const
{
    if(!(loc & VarLocation::ZERO_COPY)) {
        if (type.isPointer()) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(*$(_" << name << "), *$(_d_" << name << "), ";
            os << countVarName << " * sizeof(" << type.getPointer().valueType->getName() << "), cudaMemcpyDeviceToHost));" << std::endl;
        }
        else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy($(_" << name << "), $(_d_" << name << "), ";
            os << countVarName << " * sizeof(" << type.getName() << "), cudaMemcpyDeviceToHost));" << std::endl;
        }
        
    }
}
//--------------------------------------------------------------------------
void Backend::genMergedDynamicVariablePush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                           const std::string &groupIdx, const std::string &fieldName,
                                           const std::string &egpName) const
{
    const std::string structName = "Merged" + suffix + "Group" + std::to_string(mergedGroupIdx);
    os << "CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_merged" << suffix << "Group" << mergedGroupIdx;
    os << ", &" << egpName << ", sizeof(" << egpName << ")";
    os << ", (sizeof(" << structName << ") * (" << groupIdx << ")) + offsetof(" << structName << ", " << fieldName << ")));" << std::endl;
}
//--------------------------------------------------------------------------
std::string Backend::getMergedGroupFieldHostTypeName(const Type::ResolvedType &type) const
{
    return type.getName();
}
//--------------------------------------------------------------------------
void Backend::genGlobalDeviceRNG(CodeStream &definitions, CodeStream &runner,
                                 CodeStream &, CodeStream &) const
{
    // Define global Phillox RNG
    // **NOTE** this is actually accessed as a global so, unlike other variables, needs device global
    definitions << "extern __device__ curandStatePhilox4_32_10_t d_rng;" << std::endl;

    // Implement global Phillox RNG
    runner << "__device__ curandStatePhilox4_32_10_t d_rng;" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genTimer(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                       CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const
{
    // Define CUDA start and stop events in internal defintions (as they use CUDA-specific types)
    definitions << "extern cudaEvent_t " << name << "Start;" << std::endl;
    definitions << "extern cudaEvent_t " << name << "Stop;" << std::endl;

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
void Backend::genAssert(CodeStream &os, const std::string &condition) const
{
    os << "assert(" << condition << ");" << std::endl;
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
    os << "\t\t\t<MultiProcessorCompilation>true</MultiProcessorCompilation>" << std::endl;
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

    return nvccFlags;
}
//-----------------------------------------------------------------------
std::string Backend::getNCCLReductionType(VarAccessMode mode) const
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
std::string Backend::getNCCLType(const Type::ResolvedType &type) const
{
    assert(type.isNumeric());
    
    // Convert GeNN types to NCCL types
    if(type == Type::Int8) {
        return "ncclInt8";
    }
    else if(type == Type::Uint8) {
        return "ncclUint8";
    }
    else if(type == Type::Int32) {
        return "ncclInt32";
    }
    else if(type == Type::Uint32){
        return "ncclUint32";
    }
    /*else if(type == "half") {
        return "ncclFloat16";
    }*/
    else if(type == Type::Float){
        return "ncclFloat32";
    }
    else if(type == Type::Double) {
        return "ncclFloat64";
    }
    else {
        throw std::runtime_error("Data type '" + type.getName() + "' unsupported by NCCL");
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
}   // namespace GeNN::CodeGenerator::CUDA

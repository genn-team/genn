#include "code_generator/backendCUDAHIP.h"

// Standard C++ includes
#include <algorithm>
#include <iterator>

// GeNN includes
#include "gennUtils.h"
#include "logging.h"
#include "type.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/standardLibrary.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const EnvironmentLibrary::Library backendFunctions = {
    {"clz", {Type::ResolvedType::createFunction(Type::Int32, {Type::Uint32}), "__clz($(0))"}},
    {"atomic_or", {Type::ResolvedType::createFunction(Type::Void, {Type::Uint32.createPointer(), Type::Uint32}), "atomicOr($(0), $(1))"}},
};

const Type::ResolvedType XORWowStateInternal = Type::ResolvedType::createValue("XORWowStateInternal", 24, false, nullptr, true);

//--------------------------------------------------------------------------
// Timer
//--------------------------------------------------------------------------
class Timer
{
public:
    Timer(CodeStream &codeStream, const std::string &name, const std::string &runtimePrefix,
          bool timingEnabled, bool synchroniseOnStop = false)
    :   m_CodeStream(codeStream), m_Name(name), m_RuntimePrefix(runtimePrefix),
        m_TimingEnabled(timingEnabled), m_SynchroniseOnStop(synchroniseOnStop)
    {
        // Record start event
        if(m_TimingEnabled) {
            m_CodeStream << "CHECK_RUNTIME_ERRORS(" << m_RuntimePrefix << "EventRecord(" << m_Name << "Start));" << std::endl;
        }
    }

    ~Timer()
    {
        // Record stop event
        if(m_TimingEnabled) {
            m_CodeStream << "CHECK_RUNTIME_ERRORS(" << m_RuntimePrefix << "EventRecord(" << m_Name << "Stop));" << std::endl;

            // If we should synchronise on stop, insert call
            if(m_SynchroniseOnStop) {
                m_CodeStream << "CHECK_RUNTIME_ERRORS(" << m_RuntimePrefix << "EventSynchronize(" << m_Name << "Stop));" << std::endl;

                m_CodeStream << "float tmp;" << std::endl;
                m_CodeStream << "CHECK_RUNTIME_ERRORS(" << m_RuntimePrefix << "EventElapsedTime(&tmp, " << m_Name << "Start, " << m_Name << "Stop));" << std::endl;
                m_CodeStream << m_Name << "Time += tmp / 1000.0;" << std::endl;
            }
        }
    }

private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    CodeStream &m_CodeStream;
    std::string m_Name;
    std::string m_RuntimePrefix;
    bool m_TimingEnabled;
    bool m_SynchroniseOnStop;
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
template<typename G>
void buildPopulationRNGEnvironment(EnvironmentGroupMergedField<G> &env, const std::string &randPrefix,
                                   const Type::ResolvedType &popRNGInternalType)
{
    // Generate initialiser code to create CURandState from internal RNG state
    std::stringstream init;
    init << randPrefix << "State rngState;" << std::endl;

    // Copy useful components into full object
    init << "rngState.d = $(_rng_internal).d;" << std::endl;
    for(int i = 0; i < 5; i++) {
        init << "rngState.v[" << i << "] = $(_rng_internal).v[" << i << "];" << std::endl;
    }

    // Zero box-muller flag
    init << "rngState.boxmuller_flag = 0;" << std::endl;

    // Generate finaliser code to copy CURandState back into internal RNG state
    std::stringstream finalise;

    // Copy useful components into internal object
    finalise << "$(_rng_internal).d = rngState.d;" << std::endl;
    for(int i = 0; i < 5; i++) {
        finalise << "$(_rng_internal).v[" << i << "] = rngState.v[" << i << "];" << std::endl;
    }

    // Add alias with initialiser and destructor statements
    env.add(popRNGInternalType, "_rng", "rngState",
            {env.addInitialiser(init.str())},
            {env.addFinaliser(finalise.str())});
}
}   // Anonymous namespace


//--------------------------------------------------------------------------
// GeNN::CodeGenerator::BackendCUDAHIP
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
std::string BackendCUDAHIP::getThreadID(unsigned int axis) const
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
std::string BackendCUDAHIP::getBlockID(unsigned int axis) const
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
void BackendCUDAHIP::genWarpReduction(CodeStream& os, const std::string& variable,
                                      VarAccessMode access, const Type::ResolvedType& type) const
{
    const unsigned int lanes = getNumLanes();
    std::string mask;
    if(getNumLanes() == 32) {
        mask = "0xFFFFFFFFull";
    }
    else {
        assert(getNumLanes() == 64);
        mask = "0xFFFFFFFFFFFFFFFFull";
    }
    for (unsigned int i = (lanes / 2); i > 0; i /= 2) {
        os << getReductionOperation(variable, "__shfl_down_sync(" + mask + ", " + variable + ", " + std::to_string(i) + ")",
                                    access, type);
        os <<  ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genSharedMemBarrier(CodeStream &os) const
{
    os << "__syncthreads();" << std::endl;
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genPopulationRNGInit(CodeStream &os, const std::string &globalRNG, const std::string &seed, const std::string &sequence) const
{
    // Initialise full curandState/hiprandState object
    os << getRandPrefix() << "State rngState;" << std::endl;
    os << getRandPrefix() << "_init(" << seed << ", " << sequence << ", 0, &rngState);" << std::endl;

    // Copy useful components into internal object
    os << globalRNG << ".d = rngState.d;" << std::endl;
    for(int i = 0; i < 5; i++) {
        os << globalRNG << ".v[" << i << "] = rngState.v[" << i << "];" << std::endl;
    }
}
//--------------------------------------------------------------------------
std::string BackendCUDAHIP::genGlobalRNGSkipAhead(CodeStream &os, const std::string &sequence) const
{
    // Skipahead RNG
    os << getRandPrefix() <<  "StatePhilox4_32_10_t localRNG = *d_rng;" << std::endl;
    os << "skipahead_sequence((unsigned long long)" << sequence << ", &localRNG);" << std::endl;
    return "localRNG";
}
//--------------------------------------------------------------------------
Type::ResolvedType BackendCUDAHIP::getPopulationRNGType() const
{
    return XORWowStateInternal;
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::buildPopulationRNGEnvironment(EnvironmentGroupMergedField<NeuronUpdateGroupMerged> &env) const
{
    ::buildPopulationRNGEnvironment(env, getRandPrefix(), getPopulationRNGInternalType());
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::buildPopulationRNGEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env) const
{
    ::buildPopulationRNGEnvironment(env, getRandPrefix(), getPopulationRNGInternalType());
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genNeuronUpdate(CodeStream &os, FileStreamCreator, ModelSpecMerged &modelMerged, 
                                     BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Generate stream with neuron update code
    std::ostringstream neuronUpdateStream;
    CodeStream neuronUpdate(neuronUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary backendEnv(neuronUpdate, backendFunctions);
    EnvironmentLibrary neuronUpdateEnv(backendEnv, StandardLibrary::getMathsFunctions());

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
            neuronUpdateEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "PeekAtLastError());" << std::endl;
        }
        if(idNeuronSpikeQueueUpdate > 0) {
            CodeStream::Scope b(neuronUpdateEnv.getStream());
            genKernelDimensions(neuronUpdateEnv.getStream(), KernelNeuronSpikeQueueUpdate, idNeuronSpikeQueueUpdate, 1);
            neuronUpdateEnv.getStream() << KernelNames[KernelNeuronSpikeQueueUpdate] << "<<<grid, threads>>>();" << std::endl;
            neuronUpdateEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "PeekAtLastError());" << std::endl;
        }
        if(idStart > 0) {
            CodeStream::Scope b(neuronUpdateEnv.getStream());

            Timer t(neuronUpdateEnv.getStream(), "neuronUpdate", getRuntimePrefix(),
                    model.isTimingEnabled());

            genKernelDimensions(neuronUpdateEnv.getStream(), KernelNeuronUpdate, idStart, model.getBatchSize());
            neuronUpdateEnv.getStream() << KernelNames[KernelNeuronUpdate] << "<<<grid, threads>>>(t";
            if(model.isRecordingInUse()) {
                neuronUpdateEnv.getStream() << ", recordingTimestep";
            }
            neuronUpdateEnv.getStream() << ");" << std::endl;
            neuronUpdateEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "PeekAtLastError());" << std::endl;
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
void BackendCUDAHIP::genSynapseUpdate(CodeStream &os, FileStreamCreator, ModelSpecMerged &modelMerged,
                                      BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    // Generate stream with synapse update code
    std::ostringstream synapseUpdateStream;
    CodeStream synapseUpdate(synapseUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary backendEnv(synapseUpdate, backendFunctions);
    EnvironmentLibrary synapseUpdateEnv(backendEnv, StandardLibrary::getMathsFunctions());

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
            synapseUpdateEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "PeekAtLastError());" << std::endl;
        }

        // Launch synapse dynamics kernel if required
        if(idSynapseDynamicsStart > 0) {
            CodeStream::Scope b(synapseUpdateEnv.getStream());
            Timer t(synapseUpdateEnv.getStream(), "synapseDynamics", getRuntimePrefix(), model.isTimingEnabled());

            genKernelDimensions(synapseUpdateEnv.getStream(), KernelSynapseDynamicsUpdate, idSynapseDynamicsStart, model.getBatchSize());
            synapseUpdateEnv.getStream() << KernelNames[KernelSynapseDynamicsUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            synapseUpdateEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "PeekAtLastError());" << std::endl;
        }

        // Launch presynaptic update kernel
        if(idPresynapticStart > 0) {
            CodeStream::Scope b(synapseUpdateEnv.getStream());
            Timer t(synapseUpdateEnv.getStream(), "presynapticUpdate", getRuntimePrefix(), model.isTimingEnabled());

            genKernelDimensions(synapseUpdateEnv.getStream(), KernelPresynapticUpdate, idPresynapticStart, model.getBatchSize());
            synapseUpdateEnv.getStream() << KernelNames[KernelPresynapticUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            synapseUpdateEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "PeekAtLastError());" << std::endl;
        }

        // Launch postsynaptic update kernel
        if(idPostsynapticStart > 0) {
            CodeStream::Scope b(synapseUpdateEnv.getStream());
            Timer t(synapseUpdateEnv.getStream(), "postsynapticUpdate", getRuntimePrefix(), model.isTimingEnabled());

            genKernelDimensions(synapseUpdateEnv.getStream(), KernelPostsynapticUpdate, idPostsynapticStart, model.getBatchSize());
            synapseUpdateEnv.getStream() << KernelNames[KernelPostsynapticUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            synapseUpdateEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "PeekAtLastError());" << std::endl;
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
void BackendCUDAHIP::genCustomUpdate(CodeStream &os, FileStreamCreator, ModelSpecMerged &modelMerged,
                                     BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Generate stream with synapse update code
    std::ostringstream customUpdateStream;
    CodeStream customUpdate(customUpdateStream);

    // Begin environment with standard library
    EnvironmentLibrary backendEnv(customUpdate, backendFunctions);
    EnvironmentLibrary customUpdateEnv(backendEnv, StandardLibrary::getMathsFunctions());

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

        size_t idCustomConnectivityRemapUpdateStart = 0;
        if (std::any_of(modelMerged.getMergedCustomConnectivityRemapUpdateGroups().cbegin(), modelMerged.getMergedCustomConnectivityRemapUpdateGroups().cend(),
                        [&g](const auto &cg) { return (cg.getArchetype().getUpdateGroupName() == g); }))
        {
            genFilteredMergedKernelDataStructures(os, totalConstMem, modelMerged.getMergedCustomConnectivityRemapUpdateGroups(),
                                                  [&model, this](const CustomConnectivityUpdateInternal &cg) { return padKernelSize(cg.getSynapseGroup()->getMaxConnections(), KernelCustomUpdate); },
                                                  [g](const CustomConnectivityRemapUpdateGroupMerged &cg) { return cg.getArchetype().getUpdateGroupName() == g; });

            customUpdateEnv.getStream() << "extern \"C\" __global__ void " << KernelNames[KernelCustomConnectivityRemapUpdate] << g << "()" << std::endl;
            {
                CodeStream::Scope b(customUpdateEnv.getStream());

                EnvironmentExternal funcEnv(customUpdateEnv);
                funcEnv.getStream() << "const unsigned int id = " << getKernelBlockSize(KernelCustomUpdate) << " * blockIdx.x + threadIdx.x; " << std::endl;

                funcEnv.getStream() << "// ------------------------------------------------------------------------" << std::endl;
                funcEnv.getStream() << "// Custom connectiviy remap updates" << std::endl;
                genCustomConnectivityRemapUpdateKernel(funcEnv, modelMerged, memorySpaces, g, idCustomConnectivityRemapUpdateStart);
            }
        }

        customUpdateEnv.getStream() << "void update" << g << "(unsigned long long timestep)";
        {
            CodeStream::Scope b(customUpdateEnv.getStream());

            EnvironmentExternal funcEnv(customUpdateEnv);
            funcEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "t", "t",
                        {funcEnv.addInitialiser("const " + model.getTimePrecision().getName() + " t = timestep * " + Type::writeNumeric(model.getDT(), model.getTimePrecision()) + ";")});

            // Loop through host update groups and generate code for those in this custom update group
            if(std::any_of(modelMerged.getMergedCustomConnectivityHostUpdateGroups().cbegin(), 
                           modelMerged.getMergedCustomConnectivityHostUpdateGroups().cend(),
                           [&g](const auto &cg){ return cg.getArchetype().getUpdateGroupName() == g; }))
            {
                HostTimer t(funcEnv.getStream(), "customUpdate" + g + "Host", modelMerged.getModel().isTimingEnabled());
                modelMerged.genMergedCustomConnectivityHostUpdateGroups(
                    *this, memorySpaces, g, 
                    [this, &funcEnv](auto &c)
                    {
                        c.generateUpdate(*this, funcEnv);
                    });
            }

            // Launch custom update kernel if required
            if(idCustomUpdateStart > 0) {
                CodeStream::Scope b(funcEnv.getStream());
                genKernelDimensions(funcEnv.getStream(), KernelCustomUpdate, idCustomUpdateStart, 1);
                Timer t(funcEnv.getStream(), "customUpdate" + g, getRuntimePrefix(), model.isTimingEnabled());
                funcEnv.printLine(KernelNames[KernelCustomUpdate] + g + "<<<grid, threads>>>($(t));");
                funcEnv.printLine("CHECK_RUNTIME_ERRORS(" + getRuntimePrefix() + "PeekAtLastError());");
            }

            // Launch custom transpose update kernel if required
            if(idCustomTransposeUpdateStart > 0) {
                CodeStream::Scope b(funcEnv.getStream());
                // **TODO** make block height parameterizable
                genKernelDimensions(funcEnv.getStream(), KernelCustomTransposeUpdate, idCustomTransposeUpdateStart, 1, 8);
                Timer t(funcEnv.getStream(), "customUpdate" + g + "Transpose", getRuntimePrefix(), model.isTimingEnabled());
                funcEnv.printLine(KernelNames[KernelCustomTransposeUpdate]  + g + "<<<grid, threads>>>($(t));");
                funcEnv.printLine("CHECK_RUNTIME_ERRORS(" + getRuntimePrefix() + "PeekAtLastError());");
            }

            // Launch custom connectivity remap update kernel if required
            if (idCustomConnectivityRemapUpdateStart > 0) {
                CodeStream::Scope b(funcEnv.getStream());
                genKernelDimensions(funcEnv.getStream(), KernelCustomUpdate, idCustomConnectivityRemapUpdateStart, 1);
                Timer t(funcEnv.getStream(), "customUpdate" + g + "Remap", getRuntimePrefix(), model.isTimingEnabled());
                funcEnv.printLine(KernelNames[KernelCustomConnectivityRemapUpdate] + g + "<<<grid, threads>>>();");
                funcEnv.printLine("CHECK_RUNTIME_ERRORS(" + getRuntimePrefix() + "PeekAtLastError());");
            }

            // If NCCL reductions are enabled
            if(getPreferences<PreferencesCUDAHIP>().enableNCCLReductions) {
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
                funcEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "EventSynchronize(customUpdate" << g;
                if (idCustomConnectivityRemapUpdateStart > 0) {
                    funcEnv.getStream() << "Remap";
                }
                else if(idCustomTransposeUpdateStart > 0) {
                    funcEnv.getStream() << "Transpose";
                }
                funcEnv.getStream() << "Stop)); " << std::endl;

                if(idCustomUpdateStart > 0) {
                    CodeGenerator::CodeStream::Scope b(funcEnv.getStream());
                    funcEnv.getStream() << "float tmp;" << std::endl;
                    funcEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "EventElapsedTime(&tmp, customUpdate" << g << "Start, customUpdate" << g << "Stop));" << std::endl;
                    funcEnv.getStream() << "customUpdate" << g << "Time += tmp / 1000.0;" << std::endl;
                }
                if(idCustomTransposeUpdateStart > 0) {
                    CodeGenerator::CodeStream::Scope b(funcEnv.getStream());
                    funcEnv.getStream() << "float tmp;" << std::endl;
                    funcEnv.getStream() << "CHECK_RUNTIME_ERRORS(cudaEventElapsedTime(&tmp, customUpdate" << g << "TransposeStart, customUpdate" << g << "TransposeStop));" << std::endl;
                    funcEnv.getStream() << "customUpdate" << g << "TransposeTime += tmp / 1000.0;" << std::endl;
                }
                if (idCustomConnectivityRemapUpdateStart > 0) {
                    CodeGenerator::CodeStream::Scope b(funcEnv.getStream());
                    funcEnv.getStream() << "float tmp;" << std::endl;
                    funcEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "EventElapsedTime(&tmp, customUpdate" << g << "RemapStart, customUpdate" << g << "RemapStop));" << std::endl;
                    funcEnv.getStream() << "customUpdate" << g << "RemapTime += tmp / 1000.0;" << std::endl;
                }
            }
        }
    }

    // Generate struct definitions
    modelMerged.genMergedCustomUpdateStructs(os, *this);
    modelMerged.genMergedCustomUpdateWUStructs(os, *this);
    modelMerged.genMergedCustomUpdateTransposeWUStructs(os, *this);
    modelMerged.genMergedCustomUpdateHostReductionStructs(os, *this);
    modelMerged.genMergedCustomWUUpdateHostReductionStructs(os, *this);
    modelMerged.genMergedCustomConnectivityUpdateStructs(os, *this);
    modelMerged.genMergedCustomConnectivityRemapUpdateStructs(os, *this);
    modelMerged.genMergedCustomConnectivityHostUpdateStructs(os, *this);

    // Generate arrays of merged structs and functions to push them
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateWUGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomUpdateTransposeWUGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomConnectivityUpdateGroups());
    genMergedStructArrayPush(os, modelMerged.getMergedCustomConnectivityRemapUpdateGroups());
    modelMerged.genMergedCustomConnectivityHostUpdateStructArrayPush(os, *this);
    modelMerged.genMergedCustomUpdateHostReductionHostStructArrayPush(os, *this);
    modelMerged.genMergedCustomWUUpdateHostReductionHostStructArrayPush(os, *this);

    // Generate preamble
    preambleHandler(os);

    os << customUpdateStream.str();
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genInit(CodeStream &os, FileStreamCreator, ModelSpecMerged &modelMerged, 
                             BackendBase::MemorySpaces &memorySpaces, HostHandler preambleHandler) const
{
    const ModelSpecInternal &model = modelMerged.getModel();

    // Generate stream with synapse update code
    std::ostringstream initStream;
    CodeStream init(initStream);

    // Begin environment with RNG library and standard library
    EnvironmentLibrary rngEnv(init, getRNGFunctions(model.getPrecision()));
    EnvironmentLibrary backendEnv(rngEnv, backendFunctions);
    EnvironmentLibrary initEnv(backendEnv, StandardLibrary::getMathsFunctions());

    // If device RNG is required, generate kernel to initialise it
    if(isGlobalDeviceRNGRequired(model)) {
        initEnv.getStream() << "extern \"C\" __global__ void initializeRNGKernel(unsigned long long deviceRNGSeed)";
        {
            CodeStream::Scope b(initEnv.getStream());
            initEnv.getStream() << "if(threadIdx.x == 0)";
            {
                CodeStream::Scope b(initEnv.getStream());
                initEnv.getStream() << getRandPrefix() << "_init(deviceRNGSeed, 0, 0, d_rng);" << std::endl;
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
                initEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "PeekAtLastError());" << std::endl;
            }
        }

        // If there are any initialisation threads
        if(idInitStart > 0) {
            CodeStream::Scope b(initEnv.getStream());
            {
                Timer t(initEnv.getStream(), "init", getRuntimePrefix(), model.isTimingEnabled(), true);

                genKernelDimensions(initEnv.getStream(), KernelInitialize, idInitStart, 1);
                initEnv.getStream() << KernelNames[KernelInitialize] << "<<<grid, threads>>>(deviceRNGSeed);" << std::endl;
                initEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "PeekAtLastError());" << std::endl;
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
                Timer t(initEnv.getStream(), "initSparse", getRuntimePrefix(), model.isTimingEnabled(), true);

                genKernelDimensions(initEnv.getStream(), KernelInitializeSparse, idSparseInitStart, 1);
                initEnv.getStream() << KernelNames[KernelInitializeSparse] << "<<<grid, threads>>>();" << std::endl;
                initEnv.getStream() << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "PeekAtLastError());" << std::endl;
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
        modelMerged.getMergedCustomWUUpdateInitGroups(), [this](const CustomUpdateWUInternal &cg){ return padKernelSize(getNumInitThreads(cg), KernelInitialize); },        
        modelMerged.getMergedCustomConnectivityUpdatePreInitGroups(), [this](const CustomConnectivityUpdateInternal& cg) { return padKernelSize(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(), KernelInitialize); },
        modelMerged.getMergedCustomConnectivityUpdatePostInitGroups(), [this](const CustomConnectivityUpdateInternal& cg) { return padKernelSize(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(), KernelInitialize); },
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
void BackendCUDAHIP::genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    os << "// Standard C++ includes" << std::endl;
    os << "#include <chrono>" << std::endl;
    os << "#include <random>" << std::endl;
    os << "#include <string>" << std::endl;
    os << "#include <stdexcept>" << std::endl;
    os << std::endl;
    os << "// Standard C includes" << std::endl;
    //os << "#include <cassert>" << std::endl;
    os << "#include <cstdint>" << std::endl;

    genDefinitionsPreambleInternal(os, modelMerged);

    // **HACK** HIP defines a horrible assert macro which is
    // a) Incorrectly parenthesized https://github.com/ROCm/hipother/pull/1
    // b) Just calls abort_, not killing kernels with correct exit code
    // c) undefs the assert in <cassert> which actually works
    os << "#include <cassert>" << std::endl;
    os << std::endl;

    os << "struct XORWowStateInternal" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "unsigned int d;" << std::endl;
        os << "unsigned int v[5];" << std::endl;
    }
    os << ";" << std::endl;

    os << std::endl;
    os << "template<typename RNG>" << std::endl;
    os << "__device__ inline float exponentialDistFloat(RNG *rng)";
    {
        CodeStream::Scope b(os);
        os << "while (true)";
        {
            CodeStream::Scope b(os);
            os << "const float u = " << getRandPrefix() << "_uniform(rng);" << std::endl;
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
            os << "const double u = " << getRandPrefix() << "_uniform_double(rng);" << std::endl;
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
                os << "x = " << getRandPrefix() << "_normal(rng);" << std::endl;
                os << "v = 1.0f + c*x;" << std::endl;
            }
            os << "while (v <= 0.0f);" << std::endl;
            os << std::endl;
            os << "v = v*v*v;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "u = " << getRandPrefix() << "_uniform(rng);" << std::endl;
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
            os << "const float u = " << getRandPrefix() << "_uniform (rng);" << std::endl;
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
                os << "x = " << getRandPrefix() << "_normal_double(rng);" << std::endl;
                os << "v = 1.0 + c*x;" << std::endl;
            }
            os << "while (v <= 0.0);" << std::endl;
            os << std::endl;
            os << "v = v*v*v;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "u = " << getRandPrefix() << "_uniform_double(rng);" << std::endl;
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
            os << "const double u = " << getRandPrefix() << "_uniform (rng);" << std::endl;
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
        os << "float u = " << getRandPrefix() << "_uniform(rng);" << std::endl;
        os << "while(u > px)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "x++;" << std::endl;
            os << "if(x > bound)";
            {
                CodeStream::Scope b(os);
                os << "x = 0;" << std::endl;
                os << "px = qn;" << std::endl;
                os << "u = " << getRandPrefix() << "_uniform(rng);" << std::endl;
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
        os << "double u = " << getRandPrefix() << "_uniform_double(rng);" << std::endl;
        os << "while(u > px)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "x++;" << std::endl;
            os << "if(x > bound)";
            {
                CodeStream::Scope b(os);
                os << "x = 0;" << std::endl;
                os << "px = qn;" << std::endl;
                os << "u = " << getRandPrefix() << "_uniform_double(rng);" << std::endl;
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
void BackendCUDAHIP::genRunnerPreamble(CodeStream &os, const ModelSpecMerged&) const
{
#ifdef _WIN32
    // **YUCK** on Windows, disable "function assumed not to throw an exception but does" warning
    // Setting /Ehs SHOULD solve this but CUDA rules don't give this option and it's not clear it gets through to the compiler anyway
    os << "#pragma warning(disable: 4297)" << std::endl;
#endif

     // If NCCL is enabled
    if(getPreferences<PreferencesCUDAHIP>().enableNCCLReductions) {
        // Define NCCL ID and communicator
        os << getCCLPrefix() << "UniqueId ncclID;" << std::endl;
        os << getCCLPrefix() << "Comm_t ncclCommunicator;" << std::endl;

        // Define constant to expose NCCL_UNIQUE_ID_BYTES
        os << "const size_t ncclUniqueIDSize = NCCL_UNIQUE_ID_BYTES;" << std::endl;

        // Define wrapper to generate a unique NCCL ID
        os << std::endl;
        os << "void ncclGenerateUniqueID()";
        {
            CodeStream::Scope b(os);
            os << "CHECK_CCL_ERRORS(" << getCCLPrefix() << "GetUniqueId(&ncclID));" << std::endl;
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
            os << "CHECK_CCL_ERRORS(" << getCCLPrefix() << "CommInitRank(&ncclCommunicator, numRanks, ncclID, rank));" << std::endl;
        }
        os << std::endl;
    }
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genFreeMemPreamble(CodeStream &os, const ModelSpecMerged&) const
{
    // Free NCCL communicator
    if(getPreferences<PreferencesCUDAHIP>().enableNCCLReductions) {
        os << "CHECK_CCL_ERRORS(" << getCCLPrefix() << "CommDestroy(ncclCommunicator));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    // Synchronise if zero-copy are in use
    if(modelMerged.getModel().zeroCopyInUse()) {
        os << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "DeviceSynchronize());" << std::endl;
    }

    // If timing is enabled, synchronise last event
    if(modelMerged.getModel().isTimingEnabled()) {
        os << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "EventSynchronize(neuronUpdateStop));" << std::endl;
    }
}
//--------------------------------------------------------------------------
std::unique_ptr<Runtime::ArrayBase> BackendCUDAHIP::createPopulationRNG(size_t count) const
{
    return createArray(getPopulationRNGType(), count, VarLocation::DEVICE, false);
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genLazyVariableDynamicPush(CodeStream &os, 
                                                const Type::ResolvedType &type, const std::string &name,
                                                VarLocation loc, const std::string &countVarName) const
{
    if(!(loc & VarLocationAttribute::ZERO_COPY)) {
        if (type.isPointer()) {
            os << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "Memcpy(*$(_d_" << name << "), *$(_" << name << "), ";
            os << countVarName << " * sizeof(" << type.getPointer().valueType->getName() << "), " << getRuntimePrefix() << "MemcpyHostToDevice));" << std::endl;
        }
        else {
            os << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "Memcpy($(_d_" << name << "), $(_" << name << "), ";
            os << countVarName << " * sizeof(" << type.getName() << "), " << getRuntimePrefix() << "MemcpyHostToDevice));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genLazyVariableDynamicPull(CodeStream &os, 
                                                const Type::ResolvedType &type, const std::string &name,
                                                VarLocation loc, const std::string &countVarName) const
{
    if(!(loc & VarLocationAttribute::ZERO_COPY)) {
        if (type.isPointer()) {
            os << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "Memcpy(*$(_" << name << "), *$(_d_" << name << "), ";
            os << countVarName << " * sizeof(" << type.getPointer().valueType->getName() << "), " << getRuntimePrefix() << "MemcpyDeviceToHost));" << std::endl;
        }
        else {
            os << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "Memcpy($(_" << name << "), $(_d_" << name << "), ";
            os << countVarName << " * sizeof(" << type.getName() << "), " << getRuntimePrefix() << "MemcpyDeviceToHost));" << std::endl;
        }
        
    }
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genMergedDynamicVariablePush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                                  const std::string &groupIdx, const std::string &fieldName,
                                                  const std::string &egpName) const
{
    const std::string structName = "Merged" + suffix + "Group" + std::to_string(mergedGroupIdx);
    os << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "MemcpyToSymbolAsync(d_merged" << suffix << "Group" << mergedGroupIdx;
    os << ", &" << egpName << ", sizeof(" << egpName << ")";
    os << ", (sizeof(" << structName << ") * (" << groupIdx << ")) + offsetof(" << structName << ", " << fieldName << "), ";
    os << getRuntimePrefix() << "MemcpyHostToDevice, 0));" << std::endl;
}
//--------------------------------------------------------------------------
std::string BackendCUDAHIP::getRestrictKeyword() const
{
    return " __restrict__";
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genGlobalDeviceRNG(CodeStream &definitions, CodeStream &runner,
                                        CodeStream &, CodeStream &) const
{
    // Define global Phillox RNG
    // **YUCK** when using HIP with ROCm backend, these objects are proper classes so these need to be pointers
    // **NOTE** this is actually accessed as a global so, unlike other variables, needs device global
    definitions << "extern __device__ " << getRandPrefix() << "StatePhilox4_32_10_t *d_rng;" << std::endl;

    // Implement global Phillox RNG
    runner << "__device__ " << getRandPrefix() << "StatePhilox4_32_10_t *d_rng;" << std::endl;
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genTimer(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                              CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const
{
    // Define CUDA start and stop events in internal defintions (as they use CUDA-specific types)
    definitions << "extern " << getRuntimePrefix() << "Event_t " << name << "Start;" << std::endl;
    definitions << "extern " << getRuntimePrefix() << "Event_t " << name << "Stop;" << std::endl;

    // Implement start and stop event variables
    runner << getRuntimePrefix() << "Event_t " << name << "Start;" << std::endl;
    runner << getRuntimePrefix() << "Event_t " << name << "Stop;" << std::endl;

    // Create start and stop events in allocations
    allocations << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "EventCreate(&" << name << "Start));" << std::endl;
    allocations << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "EventCreate(&" << name << "Stop));" << std::endl;

    // Destroy start and stop events in allocations
    free << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "EventDestroy(" << name << "Start));" << std::endl;
    free << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "EventDestroy(" << name << "Stop));" << std::endl;

    if(updateInStepTime) {
        CodeGenerator::CodeStream::Scope b(stepTimeFinalise);
        stepTimeFinalise << "float tmp;" << std::endl;
        stepTimeFinalise << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "EventElapsedTime(&tmp, " << name << "Start, " << name << "Stop));" << std::endl;
        stepTimeFinalise << name << "Time += tmp / 1000.0;" << std::endl;
    }
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genReturnFreeDeviceMemoryBytes(CodeStream &os) const
{
    os << "size_t free;" << std::endl;
    os << "size_t total;" << std::endl;
    os << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "MemGetInfo(&free, &total));" << std::endl;
    os << "return free;" << std::endl;
}
//--------------------------------------------------------------------------
void BackendCUDAHIP::genAssert(CodeStream &os, const std::string &condition) const
{
    os << "assert(" << condition << ");" << std::endl;
}
//--------------------------------------------------------------------------
BackendCUDAHIP::MemorySpaces BackendCUDAHIP::getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const
{
    // Get size of update group start ids (constant cache is precious so don't use for init groups
    const size_t groupStartIDSize = (getGroupStartIDSize(modelMerged.getMergedNeuronUpdateGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedPresynapticUpdateGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedPostsynapticUpdateGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedSynapseDynamicsGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedCustomUpdateGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedCustomUpdateWUGroups()) +
                                     getGroupStartIDSize(modelMerged.getMergedCustomUpdateTransposeWUGroups()));

    // Return available constant memory and to
    return {{"__device__ __constant__", (groupStartIDSize > getChosenDeviceSafeConstMemBytes()) ? 0 : (getChosenDeviceSafeConstMemBytes() - groupStartIDSize)},
            {"__device__", getDeviceMemoryBytes()}};
}
//-----------------------------------------------------------------------
std::string BackendCUDAHIP::getNCCLReductionType(VarAccessMode mode) const
{
    // Convert GeNN reduction types to NCCL
    if(mode & VarAccessModeAttribute::MAX) {
        return getCCLPrefix() + "Max";
    }
    else if(mode & VarAccessModeAttribute::SUM) {
        return getCCLPrefix() + "Sum";
    }
    else {
        throw std::runtime_error("Reduction type unsupported by NCCL");
    }
}
//-----------------------------------------------------------------------
std::string BackendCUDAHIP::getNCCLType(const Type::ResolvedType &type) const
{
    assert(type.isNumeric());
    
    // Convert GeNN types to NCCL types
    if(type == Type::Int8) {
        return getCCLPrefix() + "Int8";
    }
    else if(type == Type::Uint8) {
        return getCCLPrefix() + "Uint8";
    }
    else if(type == Type::Int32) {
        return getCCLPrefix() + "Int32";
    }
    else if(type == Type::Uint32){
        return getCCLPrefix() + "Uint32";
    }
    /*else if(type == "half") {
        return "ncclFloat16";
    }*/
    else if(type == Type::Float){
        return getCCLPrefix() + "Float32";
    }
    else if(type == Type::Double) {
        return getCCLPrefix() + "Float64";
    }
    else {
        throw std::runtime_error("Data type '" + type.getName() + "' unsupported by NCCL");
    }
}
}   // namespace GeNN::CodeGenerator
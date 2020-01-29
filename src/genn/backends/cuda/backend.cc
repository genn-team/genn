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

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const std::vector<CodeGenerator::FunctionTemplate> cudaFunctions = {
    {"gennrand_uniform", 0, "curand_uniform_double($(rng))", "curand_uniform($(rng))"},
    {"gennrand_normal", 0, "curand_normal_double($(rng))", "curand_normal($(rng))"},
    {"gennrand_exponential", 0, "exponentialDistDouble($(rng))", "exponentialDistFloat($(rng))"},
    {"gennrand_log_normal", 2, "curand_log_normal_double($(rng), $(0), $(1))", "curand_log_normal_float($(rng), $(0), $(1))"},
    {"gennrand_gamma", 1, "gammaDistDouble($(rng), $(0))", "gammaDistFloat($(rng), $(0))"}
};

//--------------------------------------------------------------------------
// Timer
//--------------------------------------------------------------------------
class Timer
{
public:
    Timer(CodeGenerator::CodeStream &codeStream, const std::string &name, bool timingEnabled, bool synchroniseOnStop = false)
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
    CodeGenerator::CodeStream &m_CodeStream;
    const std::string m_Name;
    const bool m_TimingEnabled;
    const bool m_SynchroniseOnStop;
};


//-----------------------------------------------------------------------
void genGroupStartIDs(CodeGenerator::CodeStream &, size_t &, size_t)
{
}
//-----------------------------------------------------------------------
template<typename T, typename G, typename ...Args>
void genGroupStartIDs(CodeGenerator::CodeStream &os, size_t &idStart, size_t blockSize,
                      const std::vector<T> &mergedGroups, const std::string &groupStartPrefix, G getNumThreads,
                      Args... args)
{
    // Loop through merged groups
    for(const auto &m : mergedGroups) {
        // Declare array of starting thread indices for each neuron group
        os << "__device__ __constant__ unsigned int d_merged" << groupStartPrefix << "GroupStartID" << m.getIndex() << "[] = {";
        for(const auto &ng : m.getGroups()) {
            os << idStart << ", ";
            idStart += CodeGenerator::padSize(getNumThreads(ng.get()), blockSize);
        }
        os << "};" << std::endl;
    }

    // Generate any remaining groups
    genGroupStartIDs(os, idStart, blockSize, args...);
}
//-----------------------------------------------------------------------
template<typename ...Args>
void genMergedKernelDataStructures(CodeGenerator::CodeStream &os, size_t blockSize,
                                   Args... args)
{
    // Generate group start id arrays
    size_t idStart = 0;
    genGroupStartIDs(os, std::ref(idStart), blockSize, args...);
}
//-----------------------------------------------------------------------
template<typename T, typename G>
size_t getNumMergedGroupThreads(const std::vector<T> &groups, G getNumThreads)
{
    // Accumulate the accumulation of all groups in merged group
    return std::accumulate(
        groups.cbegin(), groups.cend(), 0,
        [getNumThreads](size_t acc, const T &n)
        {
            return std::accumulate(n.getGroups().cbegin(), n.getGroups().cend(), acc,
                                   [getNumThreads](size_t acc, std::reference_wrapper<const typename T::GroupInternal> g)
                                   {
                                       return acc + getNumThreads(g.get());
                                   });
        });
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::Backend
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace CUDA
{
const char *Backend::KernelNames[KernelMax] = {
    "updateNeuronsKernel",
    "updatePresynapticKernel",
    "updatePostsynapticKernel",
    "updateSynapseDynamicsKernel",
    "initializeKernel",
    "initializeSparseKernel",
    "preNeuronResetKernel",
    "preSynapseResetKernel"};
//--------------------------------------------------------------------------
std::vector<PresynapticUpdateStrategy::Base*> Backend::s_PresynapticUpdateStrategies = {
    new PresynapticUpdateStrategy::PreSpan,
    new PresynapticUpdateStrategy::PostSpan,
    new PresynapticUpdateStrategy::PreSpanProcedural,
    new PresynapticUpdateStrategy::PostSpanBitmask,
};
//--------------------------------------------------------------------------
Backend::Backend(const KernelBlockSize &kernelBlockSizes, const Preferences &preferences,
                 const std::string &scalarType, int device)
:   BackendBase(scalarType), m_KernelBlockSizes(kernelBlockSizes), m_Preferences(preferences), m_ChosenDeviceID(device)
{
    // Set device
    CHECK_CUDA_ERRORS(cudaSetDevice(device));

    // Get device properties
    CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&m_ChosenDevice, device));

    // Get CUDA runtime version
    cudaRuntimeGetVersion(&m_RuntimeVersion);

    // Give a warning if automatic copy is used on pre-Pascal devices
    if(m_Preferences.automaticCopy && m_ChosenDevice.major < 6) {
        LOGW << "Using automatic copy on pre-Pascal devices is supported but likely to be very slow - we recommend copying manually on these devices";
    }

    // Add CUDA-specific types, additionally marking them as 'device types' innaccesible to host code
    addDeviceType("curandState", 44);
    addDeviceType("curandStatePhilox4_32_10_t", 64);
    addDeviceType("half", 2);
}
//--------------------------------------------------------------------------
void Backend::genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged,
                              NeuronGroupSimHandler simHandler, NeuronGroupMergedHandler wuVarUpdateHandler,
                              HostHandler pushEGPHandler) const
{
    // Generate data structure for accessing merged groups
    const ModelSpecInternal &model = modelMerged.getModel();
    genMergedKernelDataStructures(
        os, m_KernelBlockSizes[KernelNeuronUpdate],
        modelMerged.getMergedNeuronUpdateGroups(), "NeuronUpdate",
        [](const NeuronGroupInternal &ng){ return ng.getNumNeurons(); });
    os << std::endl;

    // Generate reset kernel to be run before the neuron kernel
    size_t idPreNeuronReset = 0;
    os << "extern \"C\" __global__ void " << KernelNames[KernelPreNeuronReset] << "()";
    {
        CodeStream::Scope b(os);

        os << "const unsigned int id = " << m_KernelBlockSizes[KernelPreNeuronReset] << " * blockIdx.x + threadIdx.x;" << std::endl;

        // Loop through local neuron groups
        for(const auto &n : modelMerged.getMergedNeuronSpikeQueueUpdateGroups()) {
            os << "// merged" << n.getIndex() << std::endl;
            if(idPreNeuronReset == 0) {
                os << "if(id < " << n.getGroups().size() << ")";
            }
            else {
                os << "if(id >= " << idPreNeuronReset << " && id < " << idPreNeuronReset + n.getGroups().size() << ")";
            }
            {
                CodeStream::Scope b(os);

                // Use this to get reference to merged group structure
                os << "const auto &group = d_mergedNeuronSpikeQueueUpdateGroup" << n.getIndex() << "[id - " << idPreNeuronReset << "]; " << std::endl;

                if(n.getArchetype().isDelayRequired()) { // with delay
                    os << "*group.spkQuePtr  = (*group.spkQuePtr + 1) % " << n.getArchetype().getNumDelaySlots() << ";" << std::endl;
                }
                genMergedGroupSpikeCountReset(os, n);
            }
            idPreNeuronReset += n.getGroups().size();
        }
    }
    os << std::endl;

    size_t idStart = 0;
    os << "extern \"C\" __global__ void " << KernelNames[KernelNeuronUpdate] << "("  << model.getTimePrecision() << " t)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "const unsigned int id = " << m_KernelBlockSizes[KernelNeuronUpdate] << " * blockIdx.x + threadIdx.x; " << std::endl;

        Substitutions kernelSubs(cudaFunctions, model.getPrecision());
        kernelSubs.addVarSubstitution("t", "t");

        // If any neuron groups emit spike events
        if(std::any_of(modelMerged.getMergedNeuronUpdateGroups().cbegin(), modelMerged.getMergedNeuronUpdateGroups().cend(),
            [](const NeuronGroupMerged &n){ return n.getArchetype().isSpikeEventRequired(); }))
        {
            os << "__shared__ volatile unsigned int shSpkEvnt[" << m_KernelBlockSizes[KernelNeuronUpdate] << "];" << std::endl;
            os << "__shared__ volatile unsigned int shPosSpkEvnt;" << std::endl;
            os << "__shared__ volatile unsigned int shSpkEvntCount;" << std::endl;
            os << std::endl;
            os << "if (threadIdx.x == 1);";
            {
                CodeStream::Scope b(os);
                os << "shSpkEvntCount = 0;" << std::endl;
            }
            os << std::endl;
        }

        // If any neuron groups emit true spikes
        if(std::any_of(modelMerged.getMergedNeuronUpdateGroups().cbegin(), modelMerged.getMergedNeuronUpdateGroups().cend(),
            [](const NeuronGroupMerged &n){ return !n.getArchetype().getNeuronModel()->getThresholdConditionCode().empty(); }))
        {
            os << "__shared__ volatile unsigned int shSpk[" << m_KernelBlockSizes[KernelNeuronUpdate] << "];" << std::endl;
            os << "__shared__ volatile unsigned int shPosSpk;" << std::endl;
            os << "__shared__ volatile unsigned int shSpkCount;" << std::endl;
            os << "if (threadIdx.x == 0);";
            {
                CodeStream::Scope b(os);
                os << "shSpkCount = 0;" << std::endl;
            }
            os << std::endl;
        }
            
        os << "__syncthreads();" << std::endl;

        // Parallelise over neuron groups
        genParallelGroup<NeuronGroupMerged>(os, kernelSubs, modelMerged.getMergedNeuronUpdateGroups(), "NeuronUpdate", idStart,
            [this](const NeuronGroupInternal &ng){ return padSize(ng.getNumNeurons(), getKernelBlockSize(KernelNeuronUpdate)); },
            [&model, simHandler, wuVarUpdateHandler, this](CodeStream &os, const NeuronGroupMerged &ng, Substitutions &popSubs)
            {
                // If axonal delays are required
                if (ng.getArchetype().isDelayRequired()) {
                    // We should READ from delay slot before spkQuePtr
                    os << "const unsigned int readDelayOffset = " << ng.getPrevQueueOffset() << ";" << std::endl;

                    // And we should WRITE to delay slot pointed to be spkQuePtr
                    os << "const unsigned int writeDelayOffset = " << ng.getCurrentQueueOffset() << ";" << std::endl;
                }
                os << std::endl;

                // If this neuron group requires a simulation RNG, substitute in this neuron group's RNG
                if(ng.getArchetype().isSimRNGRequired()) {
                    popSubs.addVarSubstitution("rng", "&group.rng[" + popSubs["id"] + "]");
                }

                // Call handler to generate generic neuron code
                os << "if(" << popSubs["id"] << " < group.numNeurons)";
                {
                    CodeStream::Scope b(os);
                    simHandler(os, ng, popSubs,
                        // Emit true spikes
                        [this](CodeStream &os, const NeuronGroupMerged &, Substitutions &subs)
                        {
                            genEmitSpike(os, subs, "");
                        },
                        // Emit spike-like events
                        [this](CodeStream &os, const NeuronGroupMerged &, Substitutions &subs)
                        {
                            genEmitSpike(os, subs, "Evnt");
                        });
                }

                os << "__syncthreads();" << std::endl;

                if (ng.getArchetype().isSpikeEventRequired()) {
                    os << "if (threadIdx.x == 1)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (shSpkEvntCount > 0)";
                        {
                            CodeStream::Scope b(os);
                            os << "shPosSpkEvnt = atomicAdd((unsigned int*)&group.spkCntEvnt";
                            if (ng.getArchetype().isDelayRequired()) {
                                os << "[*group.spkQuePtr], shSpkEvntCount);" << std::endl;
                            }
                            else {
                                os << "[0], shSpkEvntCount);" << std::endl;
                            }
                        }
                    } // end if (threadIdx.x == 0)
                    os << "__syncthreads();" << std::endl;
                }

                if (!ng.getArchetype().getNeuronModel()->getThresholdConditionCode().empty()) {
                    os << "if (threadIdx.x == 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (shSpkCount > 0)";
                        {
                            CodeStream::Scope b(os);
                            os << "shPosSpk = atomicAdd((unsigned int*)&group.spkCnt";
                            if (ng.getArchetype().isDelayRequired() && ng.getArchetype().isTrueSpikeRequired()) {
                                os << "[*group.spkQuePtr], shSpkCount);" << std::endl;
                            }
                            else {
                                os << "[0], shSpkCount);" << std::endl;
                            }
                        }
                    } // end if (threadIdx.x == 1)
                    os << "__syncthreads();" << std::endl;
                }

                const std::string queueOffset = ng.getArchetype().isDelayRequired() ? "writeDelayOffset + " : "";
                if (ng.getArchetype().isSpikeEventRequired()) {
                    os << "if (threadIdx.x < shSpkEvntCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "group.spkEvnt[" << queueOffset << "shPosSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << std::endl;
                    }
                }

                if (!ng.getArchetype().getNeuronModel()->getThresholdConditionCode().empty()) {
                    const std::string queueOffsetTrueSpk = ng.getArchetype().isTrueSpikeRequired() ? queueOffset : "";

                    os << "if (threadIdx.x < shSpkCount)";
                    {
                        CodeStream::Scope b(os);

                        os << "const unsigned int n = shSpk[threadIdx.x];" << std::endl;

                        // Create new substition stack and explicitly replace id with 'n' and perform WU var update
                        Substitutions wuSubs(&popSubs);
                        wuSubs.addVarSubstitution("id", "n", true);
                        wuVarUpdateHandler(os, ng, wuSubs);

                        os << "group.spk[" << queueOffsetTrueSpk << "shPosSpk + threadIdx.x] = n;" << std::endl;
                        if (ng.getArchetype().isSpikeTimeRequired()) {
                            os << "group.sT[" << queueOffset << "n] = t;" << std::endl;
                        }
                    }
                }
            }
        );
    }

    os << "void updateNeurons(" << model.getTimePrecision() << ")";
    {
        CodeStream::Scope b(os);

        // Push any required EGPS
        pushEGPHandler(os);

        if(idPreNeuronReset > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelPreNeuronReset, idPreNeuronReset);
            os << KernelNames[KernelPreNeuronReset] << "<<<grid, threads>>>();" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
        if(idStart > 0) {
            CodeStream::Scope b(os);

            Timer t(os, "neuronUpdate", model.isTimingEnabled());

            genKernelDimensions(os, KernelNeuronUpdate, idStart);
            os << KernelNames[KernelNeuronUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genSynapseUpdate(CodeStream &os, const ModelSpecMerged &modelMerged,
                               SynapseGroupMergedHandler wumThreshHandler, SynapseGroupMergedHandler wumSimHandler,
                               SynapseGroupMergedHandler wumEventHandler, SynapseGroupMergedHandler wumProceduralConnectHandler,
                               SynapseGroupMergedHandler postLearnHandler, SynapseGroupMergedHandler synapseDynamicsHandler,
                               HostHandler pushEGPHandler) const
{
    // Generate data structure for accessing merged groups
    if(!modelMerged.getMergedPresynapticUpdateGroups().empty()) {
        genMergedKernelDataStructures(
            os, m_KernelBlockSizes[KernelPresynapticUpdate],
            modelMerged.getMergedPresynapticUpdateGroups(), "PresynapticUpdate",
            [this](const SynapseGroupInternal &sg)
            {
                return getNumPresynapticUpdateThreads(sg, m_ChosenDevice, m_Preferences);
            });
    }

    if(!modelMerged.getMergedPostsynapticUpdateGroups().empty()) {
        genMergedKernelDataStructures(
            os, m_KernelBlockSizes[KernelPostsynapticUpdate],
            modelMerged.getMergedPostsynapticUpdateGroups(), "PostsynapticUpdate",
            [this](const SynapseGroupInternal &sg)
            {
                return getNumPostsynapticUpdateThreads(sg);
            });
    }

    if(!modelMerged.getMergedSynapseDynamicsGroups().empty()) {
        genMergedKernelDataStructures(
            os, m_KernelBlockSizes[KernelSynapseDynamicsUpdate],
            modelMerged.getMergedSynapseDynamicsGroups(), "SynapseDynamics",
            [this](const SynapseGroupInternal &sg)
            {
                return getNumSynapseDynamicsThreads(sg);
            });
    }

    // If any synapse groups require dendritic delay, a reset kernel is required to be run before the synapse kernel
    const ModelSpecInternal &model = modelMerged.getModel();
    size_t idPreSynapseReset = 0;
    if(!modelMerged.getMergedSynapseDendriticDelayUpdateGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelPreSynapseReset] << "()";
        {
            CodeStream::Scope b(os);

            os << "const unsigned int id = " << m_KernelBlockSizes[KernelPreSynapseReset] << " * blockIdx.x + threadIdx.x;" << std::endl;

            // Loop through merged synapse groups
            for(const auto &n : modelMerged.getMergedSynapseDendriticDelayUpdateGroups()) {
                os << "// merged" << n.getIndex() << std::endl;
                if(idPreSynapseReset == 0) {
                    os << "if(id < " << n.getGroups().size() << ")";
                }
                else {
                    os << "if(id >= " << idPreSynapseReset << " && id < " << idPreSynapseReset + n.getGroups().size() << ")";
                }
                {
                    CodeStream::Scope b(os);

                    // Use this to get reference to merged group structure
                    os << "const auto &group = d_mergedSynapseDendriticDelayUpdateGroup" << n.getIndex() << "[id - " << idPreSynapseReset << "]; " << std::endl;

                    os << "*group.denDelayPtr = (*group.denDelayPtr + 1) % " << n.getArchetype().getMaxDendriticDelayTimesteps() << ";" << std::endl;
                }
                idPreSynapseReset += n.getGroups().size();
            }
        }
        os << std::endl;
    }

    // If there are any presynaptic update groups
    size_t idPresynapticStart = 0;
    if(!modelMerged.getMergedPresynapticUpdateGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelPresynapticUpdate] << "(" << model.getTimePrecision() << " t)" << std::endl; // end of synapse kernel header
        {
            CodeStream::Scope b(os);

            Substitutions kernelSubs(cudaFunctions, model.getPrecision());
            kernelSubs.addVarSubstitution("t", "t");

            os << "const unsigned int id = " << m_KernelBlockSizes[KernelPresynapticUpdate] << " * blockIdx.x + threadIdx.x; " << std::endl;

            // We need shLg if any synapse groups accumulate into shared memory
            // Determine the maximum shared memory outputs 
            size_t maxSharedMemPerThread = 0;
            for (const auto &s : modelMerged.getMergedPresynapticUpdateGroups()) {
                maxSharedMemPerThread = std::max(maxSharedMemPerThread,
                                                 getPresynapticUpdateStrategy(s.getArchetype())->getSharedMemoryPerThread(s, *this));
            }

            // If any shared memory is required, declare array
            if(maxSharedMemPerThread > 0) {
                os << "__shared__ " << model.getPrecision() << " shLg[" << maxSharedMemPerThread * m_KernelBlockSizes[KernelPresynapticUpdate] << "];" << std::endl;
            }

            // If any of these synapse groups also have sparse connectivity, allocate shared memory for row length
            if(std::any_of(modelMerged.getMergedPresynapticUpdateGroups().cbegin(), modelMerged.getMergedPresynapticUpdateGroups().cend(),
                           [&model](const SynapseGroupMerged &s)
                           {
                               return (s.getArchetype().getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC
                                       && (s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE));
                           }))
            {
                os << "__shared__ unsigned int shRowLength[" << m_KernelBlockSizes[KernelPresynapticUpdate] << "];" << std::endl;
            }

            if(std::any_of(modelMerged.getMergedPresynapticUpdateGroups().cbegin(), modelMerged.getMergedPresynapticUpdateGroups().cend(),
                           [&model](const SynapseGroupMerged &s)
                           {
                               return (s.getArchetype().isTrueSpikeRequired() || !s.getArchetype().getWUModel()->getLearnPostCode().empty());
                           }))
            {
                os << "__shared__ unsigned int shSpk[" << m_KernelBlockSizes[KernelPresynapticUpdate] << "];" << std::endl;
            }

            if(std::any_of(modelMerged.getMergedPresynapticUpdateGroups().cbegin(), modelMerged.getMergedPresynapticUpdateGroups().cend(),
                           [](const SynapseGroupMerged &s){ return (s.getArchetype().isSpikeEventRequired()); }))
            {
                os << "__shared__ unsigned int shSpkEvnt[" << m_KernelBlockSizes[KernelPresynapticUpdate] << "];" << std::endl;
            }

            // Parallelise over presynaptic update groups
            genParallelGroup<SynapseGroupMerged>(os, kernelSubs, modelMerged.getMergedPresynapticUpdateGroups(), "PresynapticUpdate", idPresynapticStart,
                [this](const SynapseGroupInternal &sg){ return padSize(getNumPresynapticUpdateThreads(sg, m_ChosenDevice, m_Preferences), m_KernelBlockSizes[KernelPresynapticUpdate]); },
                [&idPresynapticStart, wumThreshHandler, wumSimHandler, wumEventHandler, wumProceduralConnectHandler, &modelMerged, this]
                (CodeStream &os, const SynapseGroupMerged &sg, Substitutions &popSubs)
                {
                    // Get presynaptic update strategy to use for this synapse group
                    const auto *presynapticUpdateStrategy = getPresynapticUpdateStrategy(sg.getArchetype());
                    LOGD_BACKEND << "Using '" << typeid(*presynapticUpdateStrategy).name() << "' presynaptic update strategy for merged synapse group '" << sg.getIndex() << "'";

                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelaySlot = " << sg.getPresynapticAxonalDelaySlot() << ";" << std::endl;
                        os << "const unsigned int preReadDelayOffset = preReadDelaySlot * group.numSrcNeurons;" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelayOffset = " << sg.getPostsynapticBackPropDelaySlot() << " * group.numTrgNeurons;" << std::endl;
                    }

                    // Generate preamble
                    presynapticUpdateStrategy->genPreamble(os, modelMerged, sg, popSubs, *this, idPresynapticStart);
                  
                    // If spike events should be processed
                    if (sg.getArchetype().isSpikeEventRequired()) {
                        CodeStream::Scope b(os);
                        presynapticUpdateStrategy->genUpdate(os, modelMerged, sg, popSubs, *this, false, idPresynapticStart,
                                                             wumThreshHandler, wumEventHandler, wumProceduralConnectHandler);
                    }

                    // If true spikes should be processed
                    if (sg.getArchetype().isTrueSpikeRequired()) {
                        CodeStream::Scope b(os);
                        presynapticUpdateStrategy->genUpdate(os, modelMerged, sg, popSubs, *this, true, idPresynapticStart,
                                                             wumThreshHandler, wumSimHandler, wumProceduralConnectHandler);
                    }

                    os << std::endl;

                    // Generate pre-amble
                    presynapticUpdateStrategy->genPostamble(os, modelMerged, sg, popSubs, *this, idPresynapticStart);
                }
            );
        }
    }

    // If any synapse groups require postsynaptic learning
    size_t idPostsynapticStart = 0;
    if(!modelMerged.getMergedPostsynapticUpdateGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelPostsynapticUpdate] << "(" << model.getTimePrecision() << " t)" << std::endl;
        {
            CodeStream::Scope b(os);

            Substitutions kernelSubs(cudaFunctions, model.getPrecision());
            kernelSubs.addVarSubstitution("t", "t");

            os << "const unsigned int id = " << m_KernelBlockSizes[KernelPostsynapticUpdate] << " * blockIdx.x + threadIdx.x; " << std::endl;

            os << "__shared__ unsigned int shSpk[" << m_KernelBlockSizes[KernelPostsynapticUpdate] << "];" << std::endl;
            if(std::any_of(model.getSynapseGroups().cbegin(), model.getSynapseGroups().cend(),
                [&model](const ModelSpec::SynapseGroupValueType &s)
                {
                    return ((s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getLearnPostCode().empty());
                }))
            {
                os << "__shared__ unsigned int shColLength[" << m_KernelBlockSizes[KernelPostsynapticUpdate] << "];" << std::endl;
            }

            // Parallelise over postsynaptic update groups
            genParallelGroup<SynapseGroupMerged>(os, kernelSubs, modelMerged.getMergedPostsynapticUpdateGroups(), "PostsynapticUpdate", idPostsynapticStart,
                [this](const SynapseGroupInternal &sg) { return padSize(getNumPostsynapticUpdateThreads(sg), m_KernelBlockSizes[KernelPostsynapticUpdate]); },
                [postLearnHandler, &model, this](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &popSubs)
                {
                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelayOffset = " << sg.getPresynapticAxonalDelaySlot() << " * group.numSrcNeurons;" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelaySlot = " << sg.getPostsynapticBackPropDelaySlot() << ";" << std::endl;
                        os << "const unsigned int postReadDelayOffset = postReadDelaySlot * group.numTrgNeurons;" << std::endl;
                    }

                    if (sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int numSpikes = group.trgSpkCnt[postReadDelaySlot];" << std::endl;
                    }
                    else {
                        os << "const unsigned int numSpikes = group.trgSpkCnt[0];" << std::endl;
                    }

                    os << "const unsigned int numSpikeBlocks = (numSpikes + " << m_KernelBlockSizes[KernelPostsynapticUpdate]-1 << ") / " << m_KernelBlockSizes[KernelPostsynapticUpdate] << ";" << std::endl;
                    os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
                    {
                        CodeStream::Scope b(os);
                        os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << m_KernelBlockSizes[KernelPostsynapticUpdate] << ") + 1 : " << m_KernelBlockSizes[KernelPostsynapticUpdate] << ";" << std::endl;

                        os << "if (threadIdx.x < numSpikesInBlock)";
                        {
                            CodeStream::Scope b(os);
                            const std::string offsetTrueSpkPost = (sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) ? "postReadDelayOffset + " : "";
                            os << "const unsigned int spk = group.trgSpk[" << offsetTrueSpkPost << "(r * " << m_KernelBlockSizes[KernelPostsynapticUpdate] << ") + threadIdx.x];" << std::endl;
                            os << "shSpk[threadIdx.x] = spk;" << std::endl;

                            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                os << "shColLength[threadIdx.x] = group.colLength[spk];" << std::endl;
                            }
                        }

                        os << "__syncthreads();" << std::endl;
                        os << "// only work on existing neurons" << std::endl;
                        os << "if (" << popSubs["id"] << " < group.colStride)";
                        {
                            CodeStream::Scope b(os);
                            os << "// loop through all incoming spikes for learning" << std::endl;
                            os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
                            {
                                CodeStream::Scope b(os);

                                Substitutions synSubs(&popSubs);
                                if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    os << "if (" << synSubs["id"] << " < shColLength[j])" << CodeStream::OB(1540);
                                    os << "const unsigned int synAddress = group.remap[(shSpk[j] * group.colStride) + " << popSubs["id"] << "];" << std::endl;

                                    // **OPTIMIZE** we can do a fast constant divide optimization here
                                    os << "const unsigned int ipre = synAddress / group.rowStride;" << std::endl;
                                    synSubs.addVarSubstitution("id_pre", "ipre");
                                }
                                else {
                                    os << "const unsigned int synAddress = (" << synSubs["id"] << " * group.numTrgNeurons) + shSpk[j];" << std::endl;
                                    synSubs.addVarSubstitution("id_pre", synSubs["id"]);
                                }

                                synSubs.addVarSubstitution("id_post", "shSpk[j]");
                                synSubs.addVarSubstitution("id_syn", "synAddress");

                                postLearnHandler(os, sg, synSubs);

                                if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    os << CodeStream::CB(1540);
                                }
                            }
                        }
                    }
                }
            );
        }
    }

    size_t idSynapseDynamicsStart = 0;
    if(!modelMerged.getMergedSynapseDynamicsGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelSynapseDynamicsUpdate] << "(" << model.getTimePrecision() << " t)" << std::endl; // end of synapse kernel header
        {
            CodeStream::Scope b(os);
            os << "const unsigned int id = " << m_KernelBlockSizes[KernelSynapseDynamicsUpdate] << " * blockIdx.x + threadIdx.x;" << std::endl;

            Substitutions kernelSubs(cudaFunctions, model.getPrecision());
            kernelSubs.addVarSubstitution("t", "t");

            // Parallelise over synapse groups whose weight update models have code for synapse dynamics
            genParallelGroup<SynapseGroupMerged>(os, kernelSubs, modelMerged.getMergedSynapseDynamicsGroups(), "SynapseDynamics", idSynapseDynamicsStart,
                [this](const SynapseGroupInternal &sg) { return padSize(getNumSynapseDynamicsThreads(sg), m_KernelBlockSizes[KernelSynapseDynamicsUpdate]); },
                [synapseDynamicsHandler, &model, this](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &popSubs)
                {
                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelayOffset = " << sg.getPresynapticAxonalDelaySlot() << " * group.numSrcNeurons;" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelayOffset = " << sg.getPostsynapticBackPropDelaySlot() << " * group.numTrgNeurons;" << std::endl;
                    }

                    Substitutions synSubs(&popSubs);

                    if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        os << "if (" << popSubs["id"] << " < group.synRemap[0])";
                    }
                    else {
                        os << "if (" << popSubs["id"] << " < (group.numSrcNeurons * group.numTrgNeurons))";
                    }
                    {
                        CodeStream::Scope b(os);

                        if (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            // Determine synapse and presynaptic indices for this thread
                            os << "const unsigned int s = group.synRemap[1 + " << popSubs["id"] << "];" << std::endl;

                            synSubs.addVarSubstitution("id_pre", "(s / group.rowStride)");
                            synSubs.addVarSubstitution("id_post", "group.ind[s]");
                            synSubs.addVarSubstitution("id_syn", "s");
                        }
                        else {
                            // **OPTIMIZE** we can do a fast constant divide optimization here and use the result to calculate the remainder
                            synSubs.addVarSubstitution("id_pre", "(" + popSubs["id"] + " / group.rowStride)");
                            synSubs.addVarSubstitution("id_post", "(" + popSubs["id"] + " % group.rowStride)");
                            synSubs.addVarSubstitution("id_syn", popSubs["id"]);
                        }

                        // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                        if(sg.getArchetype().isDendriticDelayRequired()) {
                            synSubs.addFuncSubstitution("addToInSynDelay", 2, getFloatAtomicAdd(model.getPrecision()) + "(&group.denDelay[" + sg.getDendriticDelayOffset("$(1)") + synSubs["id_post"] + "], $(0))");
                        }
                        // Otherwise
                        else {
                            synSubs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&group.inSyn[" + synSubs["id_post"] + "], $(0))");
                        }

                        synapseDynamicsHandler(os, sg, synSubs);
                    }
                });
        }
    }

    os << "void updateSynapses(" << model.getTimePrecision() << " t)";
    {
        CodeStream::Scope b(os);

        // Push any required EGPs
        pushEGPHandler(os);

        // Launch pre-synapse reset kernel if required
        if(idPreSynapseReset > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelPreSynapseReset, idPreSynapseReset);
            os << KernelNames[KernelPreSynapseReset] << "<<<grid, threads>>>();" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch synapse dynamics kernel if required
        if(idSynapseDynamicsStart > 0) {
            CodeStream::Scope b(os);
            Timer t(os, "synapseDynamics", model.isTimingEnabled());

            genKernelDimensions(os, KernelSynapseDynamicsUpdate, idSynapseDynamicsStart);
            os << KernelNames[KernelSynapseDynamicsUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch presynaptic update kernel
        if(idPresynapticStart > 0) {
            CodeStream::Scope b(os);
            Timer t(os, "presynapticUpdate", model.isTimingEnabled());

            genKernelDimensions(os, KernelPresynapticUpdate, idPresynapticStart);
            os << KernelNames[KernelPresynapticUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch postsynaptic update kernel
        if(idPostsynapticStart > 0) {
            CodeStream::Scope b(os);
            Timer t(os, "postsynapticUpdate", model.isTimingEnabled());

            genKernelDimensions(os, KernelPostsynapticUpdate, idPostsynapticStart);
            os << KernelNames[KernelPostsynapticUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genInit(CodeStream &os, const ModelSpecMerged &modelMerged,
                      NeuronGroupMergedHandler localNGHandler, SynapseGroupMergedHandler sgDenseInitHandler, 
                      SynapseGroupMergedHandler sgSparseConnectHandler, SynapseGroupMergedHandler sgSparseInitHandler,
                      HostHandler initPushEGPHandler, HostHandler initSparsePushEGPHandler) const
{
    os << "#include <iostream>" << std::endl;
    os << "#include <random>" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << std::endl;

    // Generate data structure for accessing merged groups from within initialisation kernel
    const ModelSpecInternal &model = modelMerged.getModel();
    genMergedKernelDataStructures(os, m_KernelBlockSizes[KernelInitialize],
        modelMerged.getMergedNeuronInitGroups(), "NeuronInit",
        [](const NeuronGroupInternal &ng){ return ng.getNumNeurons(); },
        modelMerged.getMergedSynapseDenseInitGroups(), "SynapseDenseInit",
        [](const SynapseGroupInternal &sg){ return sg.getTrgNeuronGroup()->getNumNeurons(); },
        modelMerged.getMergedSynapseConnectivityInitGroups(), "SynapseConnectivityInit",
        [](const SynapseGroupInternal &sg){ return sg.getSrcNeuronGroup()->getNumNeurons(); });

    // If sparse initialisation is required, generate data structure for accessing merged groups from within sparse initialisation kernel
    if(!modelMerged.getMergedSynapseSparseInitGroups().empty()) {
        genMergedKernelDataStructures(os, m_KernelBlockSizes[KernelInitializeSparse],
            modelMerged.getMergedSynapseSparseInitGroups(), "SynapseSparseInit",
            [](const SynapseGroupInternal &sg){ return sg.getMaxConnections(); });
    }
    os << std::endl;

    // If device RNG is required, generate kernel to initialise it
    if(isGlobalRNGRequired(modelMerged)) {
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
        Substitutions kernelSubs(cudaFunctions, model.getPrecision());

        // common variables for all cases
        CodeStream::Scope b(os);

        os << "const unsigned int id = " << m_KernelBlockSizes[KernelInitialize] << " * blockIdx.x + threadIdx.x;" << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Local neuron groups" << std::endl;
        genParallelGroup<NeuronGroupMerged>(os, kernelSubs, modelMerged.getMergedNeuronInitGroups(), "NeuronInit", idInitStart,
            [this](const NeuronGroupInternal &ng){ return padSize(ng.getNumNeurons(), m_KernelBlockSizes[KernelInitialize]); },
            [this, &model, localNGHandler](CodeStream &os, const NeuronGroupMerged &ng, Substitutions &popSubs)
            {
                os << "// only do this for existing neurons" << std::endl;
                os << "if(" << popSubs["id"] << " < group.numNeurons)";
                {
                    CodeStream::Scope b(os);
                    // If this neuron is going to require a simulation RNG, initialise one using GLOBAL thread id for sequence
                    if(ng.getArchetype().isSimRNGRequired()) {
                        os << "curand_init(deviceRNGSeed, id, 0, &group.rng[" << popSubs["id"] << "]);" << std::endl;
                    }

                    // If this neuron requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    // **NOTE** not LOCAL id
                    if(ng.getArchetype().isInitRNGRequired()) {
                        os << "curandStatePhilox4_32_10_t initRNG = d_rng;" << std::endl;
                        os << "skipahead_sequence((unsigned long long)id, &initRNG);" << std::endl;

                        // Add substitution for RNG
                        popSubs.addVarSubstitution("rng", "&initRNG");
                    }

                    localNGHandler(os, ng, popSubs);
                }
            });
        os << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with dense connectivity" << std::endl;
        genParallelGroup<SynapseGroupMerged>(os, kernelSubs, modelMerged.getMergedSynapseDenseInitGroups(), "SynapseDenseInit", idInitStart,
            [this](const SynapseGroupInternal &sg){ return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), m_KernelBlockSizes[KernelInitialize]); },
            [sgDenseInitHandler](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &popSubs)
            {
                os << "// only do this for existing postsynaptic neurons" << std::endl;
                os << "if(" << popSubs["id"] << " < group.numTrgNeurons)";
                {
                    CodeStream::Scope b(os);
                    // If this post synapse requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    // **NOTE** not LOCAL id
                    if(sg.getArchetype().isWUInitRNGRequired()) {
                        os << "curandStatePhilox4_32_10_t initRNG = d_rng;" << std::endl;
                        os << "skipahead_sequence((unsigned long long)id, &initRNG);" << std::endl;

                        // Add substitution for RNG
                        popSubs.addVarSubstitution("rng", "&initRNG");
                    }

                    popSubs.addVarSubstitution("id_post", popSubs["id"]);
                    sgDenseInitHandler(os, sg, popSubs);
                }
            });
        os << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with sparse connectivity" << std::endl;
        genParallelGroup<SynapseGroupMerged>(os, kernelSubs, modelMerged.getMergedSynapseConnectivityInitGroups(), "SynapseConnectivityInit", idInitStart,
            [this](const SynapseGroupInternal &sg){ return padSize(sg.getSrcNeuronGroup()->getNumNeurons(), m_KernelBlockSizes[KernelInitialize]); },
            [this, sgSparseConnectHandler](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &popSubs)
            {
                os << "// only do this for existing presynaptic neurons" << std::endl;
                os << "if(" << popSubs["id"] << " < group.numSrcNeurons)";
                {
                    CodeStream::Scope b(os);
                    popSubs.addVarSubstitution("id_pre", popSubs["id"]);
                    popSubs.addVarSubstitution("id_post_begin", "0");
                    popSubs.addVarSubstitution("id_thread", "0");
                    popSubs.addVarSubstitution("num_threads", "1");
                    popSubs.addVarSubstitution("num_post", "group.numTrgNeurons");
                    
                    // If the synapse group has bitmask connectivity
                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // Calculate the maximum number of synapses in any groups
                        size_t maxSynapses = 0;
                        for(const auto &g : sg.getGroups()) {
                            const size_t numSynapses = (size_t)g.get().getSrcNeuronGroup()->getNumNeurons() * (size_t)getSynapticMatrixRowStride(g.get());
                            maxSynapses = std::max(maxSynapses, numSynapses);
                        }

                        // Calculate indices of bits at start and end of row
                        os << "// Calculate indices" << std::endl;
                        if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                            os << "const uint64_t rowStartGID = " << popSubs["id"] << " * (uint64_t)group.rowStride;" << std::endl;
                        }
                        else {
                            os << "const unsigned int rowStartGID = " << popSubs["id"] << " * group.rowStride;" << std::endl;
                        }

                        // Build function template to set correct bit in bitmask
                        popSubs.addFuncSubstitution("addSynapse", 1,
                                                    "atomicOr(&group.gp[(rowStartGID + $(0)) / 32], 0x80000000 >> ((rowStartGID + $(0)) & 31))");
                    }
                    // Otherwise, if synapse group has ragged connectivity
                    else if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        // Zero row length
                        const std::string rowLength = "group.rowLength[" + popSubs["id"] + "]";
                        os << rowLength << " = 0;" << std::endl;

                        // Build function template to increment row length and insert synapse into ind array
                        popSubs.addFuncSubstitution("addSynapse", 1,
                                                    "group.ind[(" + popSubs["id"] + " * group.rowStride) + (" + rowLength + "++)] = $(0)");
                    }
                    else {
                        assert(false);
                    }

                    // If this connectivity requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    // **NOTE** not LOCAL id
                    if(::Utils::isRNGRequired(sg.getArchetype().getConnectivityInitialiser().getSnippet()->getRowBuildCode())) {
                        os << "curandStatePhilox4_32_10_t connectivityRNG = d_rng;" << std::endl;
                        os << "skipahead_sequence((unsigned long long)id, &connectivityRNG);" << std::endl;

                        // Add substitution for RNG
                        popSubs.addVarSubstitution("rng", "&connectivityRNG");
                    }

                    sgSparseConnectHandler(os, sg, popSubs);
                }
            });
        os << std::endl;
    }
    const size_t numStaticInitThreads = idInitStart;

    // Sparse initialization kernel code
    size_t idSparseInitStart = 0;
    if(!modelMerged.getMergedSynapseSparseInitGroups().empty()) {
        os << "extern \"C\" __global__ void " << KernelNames[KernelInitializeSparse] << "()";
        {
            CodeStream::Scope b(os);

            // common variables for all cases
            Substitutions kernelSubs(cudaFunctions, model.getPrecision());

            os << "const unsigned int id = " << m_KernelBlockSizes[KernelInitializeSparse] << " * blockIdx.x + threadIdx.x;" << std::endl;

            // Shared memory array so row lengths don't have to be read by EVERY postsynaptic thread
            // **TODO** check actually required
            os << "__shared__ unsigned int shRowLength[" << m_KernelBlockSizes[KernelInitializeSparse] << "];" << std::endl;
            if(std::any_of(model.getSynapseGroups().cbegin(), model.getSynapseGroups().cend(),
                           [](const ModelSpec::SynapseGroupValueType &s) { return (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getSynapseDynamicsCode().empty(); }))
            {
                os << "__shared__ unsigned int shRowStart[" << m_KernelBlockSizes[KernelInitializeSparse] + 1 << "];" << std::endl;
            }

            // Initialise weight update variables for synapse groups with sparse connectivity
            genParallelGroup<SynapseGroupMerged>(os, kernelSubs, modelMerged.getMergedSynapseSparseInitGroups(), "SynapseSparseInit", idSparseInitStart,
                [this](const SynapseGroupInternal &sg){ return padSize(sg.getMaxConnections(), m_KernelBlockSizes[KernelInitializeSparse]); },
                [this, &model, sgSparseInitHandler, numStaticInitThreads](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &popSubs)
                {
                    // If this post synapse requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    // **NOTE** not LOCAL id
                    if(sg.getArchetype().isWUInitRNGRequired()) {
                        os << "curandStatePhilox4_32_10_t initRNG = d_rng;" << std::endl;
                        os << "skipahead_sequence((unsigned long long)" << numStaticInitThreads << " + id, &initRNG);" << std::endl;

                        // Add substitution for RNG
                        popSubs.addVarSubstitution("rng", "&initRNG");
                    }

                    // Calculate how many blocks rows need to be processed in (in order to store row lengths in shared memory)
                    const size_t blockSize = m_KernelBlockSizes[KernelInitializeSparse];
                    os << "const unsigned int numBlocks = (group.numSrcNeurons + " << blockSize << " - 1) / " << blockSize << ";" << std::endl;

                    os << "unsigned int idx = " << popSubs["id"] << ";" << std::endl;

                    // Loop through blocks
                    os << "for(unsigned int r = 0; r < numBlocks; r++)";
                    {
                        CodeStream::Scope b(os);

                        // Calculate number of rows to process in this block
                        os << "const unsigned numRowsInBlock = (r == (numBlocks - 1))";
                        os << " ? ((group.numSrcNeurons - 1) % " << m_KernelBlockSizes[KernelInitializeSparse] << ") + 1";
                        os << " : " << m_KernelBlockSizes[KernelInitializeSparse] << ";" << std::endl;

                        // Use threads to copy block of sparse structure into shared memory
                        os << "__syncthreads();" << std::endl;
                        os << "if (threadIdx.x < numRowsInBlock)";
                        {
                            CodeStream::Scope b(os);
                            os << "shRowLength[threadIdx.x] = group.rowLength[(r * " << m_KernelBlockSizes[KernelInitializeSparse] << ") + threadIdx.x];" << std::endl;
                        }

                        // If this synapse group has synapse dynamics
                        if(!sg.getArchetype().getWUModel()->getSynapseDynamicsCode().empty()) {
                            os << "__syncthreads();" << std::endl;

                            // Use first thread to generate cumulative sum
                            os << "if (threadIdx.x == 0)";
                            {
                                CodeStream::Scope b(os);

                                // Get index of last row in resultant synapse dynamics structure
                                // **NOTE** if there IS a previous block, it will always have had initSparseBlkSz rows in it
                                os << "unsigned int rowStart = (r == 0) ? 0 : shRowStart[" << m_KernelBlockSizes[KernelInitializeSparse] << "];" << std::endl;
                                os << "shRowStart[0] = rowStart;" << std::endl;

                                // Loop through rows in block
                                os << "for(unsigned int i = 0; i < numRowsInBlock; i++)";
                                {
                                    CodeStream::Scope b(os);

                                    // Add this row's length to cumulative sum and write this to this row's end
                                    os << "rowStart += shRowLength[i];" << std::endl;
                                    os << "shRowStart[i + 1] = rowStart;" << std::endl;
                                }

                                // If this is the first thread block of the first block in the group AND the last block of rows,
                                // write the total cumulative sum to the first entry of the remap structure
                                os << "if(" << popSubs["id"] << " == 0 && (r == (numBlocks - 1)))";
                                {
                                    CodeStream::Scope b(os);
                                    os << "group.synRemap[0] = shRowStart[numRowsInBlock];" << std::endl;
                                }

                            }
                        }

                        os << "__syncthreads();" << std::endl;

                        // Loop through rows
                        os << "for(unsigned int i = 0; i < numRowsInBlock; i++)";
                        {
                            CodeStream::Scope b(os);

                            // If there is a synapse for this thread to initialise
                            os << "if(" << popSubs["id"] << " < shRowLength[i])";
                            {
                                CodeStream::Scope b(os);

                                // Generate sparse initialisation code
                                if(sg.getArchetype().isWUVarInitRequired()) {
                                    popSubs.addVarSubstitution("id_pre", "((r * " + std::to_string(m_KernelBlockSizes[KernelInitializeSparse]) + ") + i)");
                                    popSubs.addVarSubstitution("id_post", "group.ind[idx]");
                                    sgSparseInitHandler(os, sg, popSubs);
                                }

                                // If postsynaptic learning is required
                                if(!sg.getArchetype().getWUModel()->getLearnPostCode().empty()) {
                                    CodeStream::Scope b(os);

                                    // Extract index of synapse's postsynaptic target
                                    os << "const unsigned int postIndex = group.ind[idx];" << std::endl;

                                    // Atomically increment length of column of connectivity associated with this target
                                    // **NOTE** this returns previous length i.e. where to insert new entry
                                    os << "const unsigned int colLocation = atomicAdd(&group.colLength[postIndex], 1);" << std::endl;

                                    // From this calculate index into column-major matrix
                                    os << "const unsigned int colMajorIndex = (postIndex * group.colStride) + colLocation;" << std::endl;

                                    // Add remapping entry at this location poining back to row-major index
                                    os << "group.remap[colMajorIndex] = idx;" << std::endl;
                                }

                                // If synapse dynamics are required, copy idx into syn remap structure
                                if(!sg.getArchetype().getWUModel()->getSynapseDynamicsCode().empty()) {
                                    CodeStream::Scope b(os);
                                    os << "group.synRemap[shRowStart[i] + " + popSubs["id"] + " + 1] = idx;" << std::endl;
                                }
                            }

                            // If matrix is ragged, advance index to next row by adding stride
                            os << "idx += group.rowStride;" << std::endl;
                        }
                    }
                });
        }
        os << std::endl;
    }

    os << "void initialize()";
    {
        CodeStream::Scope b(os);

        os << "unsigned long long deviceRNGSeed = 0;" << std::endl;

        // If on-device global RNG is required
        if(isGlobalRNGRequired(modelMerged)) {
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

            // Launch kernel to initalize RNG
            os << "initializeRNGKernel<<<1, 1>>>(deviceRNGSeed);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        for(const auto &s : model.getSynapseGroups()) {
            // If this synapse population has BITMASK connectivity and is intialised on device, insert a call to cudaMemset to zero the whole bitmask
            if(s.second.isSparseConnectivityInitRequired() && s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                const size_t gpSize = ceilDivide((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * getSynapticMatrixRowStride(s.second), 32);
                os << "CHECK_CUDA_ERRORS(cudaMemset(d_gp" << s.first << ", 0, " << gpSize << " * sizeof(uint32_t)));" << std::endl;
            }
            // Otherwise, if this synapse population has RAGGED connectivity and has postsynaptic learning, insert a call to cudaMemset to zero column lengths
            else if((s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getLearnPostCode().empty()) {
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

                genKernelDimensions(os, KernelInitialize, idInitStart);
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
        if(!m_Preferences.automaticCopy) {
            os << "copyStateToDevice(true);" << std::endl;
            os << "copyConnectivityToDevice(true);" << std::endl << std::endl;
        }

        // If there are any sparse initialisation threads
        if(idSparseInitStart > 0) {
            CodeStream::Scope b(os);
            {
                Timer t(os, "initSparse", model.isTimingEnabled(), true);

                genKernelDimensions(os, KernelInitializeSparse, idSparseInitStart);
                os << KernelNames[KernelInitializeSparse] << "<<<grid, threads>>>();" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
            }
        }
    }
}
//--------------------------------------------------------------------------
size_t Backend::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    return getPresynapticUpdateStrategy(sg)->getSynapticMatrixRowStride(sg);
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &) const
{
    os << "// Standard C++ includes" << std::endl;
    os << "#include <string>" << std::endl;
    os << "#include <stdexcept>" << std::endl;
    os << std::endl;
    os << "// Standard C includes" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << std::endl;
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper macro for error-checking CUDA calls" << std::endl;
    os << "#define CHECK_CUDA_ERRORS(call) {\\" << std::endl;
    os << "    cudaError_t error = call;\\" << std::endl;
    os << "    if (error != cudaSuccess) {\\" << std::endl;
    os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": cuda error \" + std::to_string(error) + \": \" + cudaGetErrorString(error));\\" << std::endl;
    os << "    }\\" << std::endl;
    os << "}" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged &) const
{
    os << "// CUDA includes" << std::endl;
    os << "#include <curand_kernel.h>" << std::endl;
    if(getRuntimeVersion() >= 9000) {
        os <<"#include <cuda_fp16.h>" << std::endl;
    }
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
}
//--------------------------------------------------------------------------
void Backend::genRunnerPreamble(CodeStream &os, const ModelSpecMerged &) const
{
#ifdef _WIN32
    // **YUCK** on Windows, disable "function assumed not to throw an exception but does" warning
    // Setting /Ehs SHOULD solve this but CUDA rules don't give this option and it's not clear it gets through to the compiler anyway
    os << "#pragma warning(disable: 4297)" << std::endl;
#endif
    os << "template<class T>" << std::endl;
    os << "T *getSymbolAddress(T &devSymbol)";
    {
        CodeStream::Scope b(os);
        os << "void *devPtr;" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, devSymbol));" << std::endl;
        os << "return reinterpret_cast<T*>(devPtr);" << std::endl;
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    // Get chosen device's PCI bus ID
    char pciBusID[32];
    CHECK_CUDA_ERRORS(cudaDeviceGetPCIBusId(pciBusID, 32, m_ChosenDeviceID));

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
    if(m_Preferences.selectGPUByDeviceID) {
        os << "CHECK_CUDA_ERRORS(cudaSetDevice(" << m_ChosenDeviceID << "));" << std::endl;
    }
    // Otherwise, write code to get device by PCI bus ID
    // **NOTE** this is required because device IDs are not guaranteed to remain the same and we want the code to be run on the same GPU it was optimise for
    else {
        os << "int deviceID;" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaDeviceGetByPCIBusId(&deviceID, \"" << pciBusID << "\"));" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));" << std::endl;
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    // Synchronise if automatic copying or zero-copy are in use
    // **THINK** Is this only required with automatic copy on older SM, CUDA and on non-Linux?
    if(m_Preferences.automaticCopy || modelMerged.getModel().zeroCopyInUse()) {
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

    if(m_Preferences.automaticCopy && ::Utils::isTypePointer(type)) {
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
    if(m_Preferences.automaticCopy && ::Utils::isTypePointer(type)) {
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
MemAlloc Backend::genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const
{
    auto allocation = MemAlloc::zero();

    if(m_Preferences.automaticCopy) {
        os << "CHECK_CUDA_ERRORS(cudaMallocManaged(&" << name << ", " << count << " * sizeof(" << type << ")));" << std::endl;
        allocation += MemAlloc::device(count * getSize(type));
    }
    else {
        if(loc & VarLocation::HOST) {
            const char *flags = (loc & VarLocation::ZERO_COPY) ? "cudaHostAllocMapped" : "cudaHostAllocPortable";
            os << "CHECK_CUDA_ERRORS(cudaHostAlloc(&" << name << ", " << count << " * sizeof(" << type << "), " << flags << "));" << std::endl;
            allocation += MemAlloc::host(count * getSize(type));
        }

        // If variable is present on device at all
        if(loc & VarLocation::DEVICE) {
            // Insert call to correct helper depending on whether variable should be allocated in zero-copy mode or not
            if(loc & VarLocation::ZERO_COPY) {
                os << "CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void **)&d_" << name << ", (void *)" << name << ", 0));" << std::endl;
                allocation += MemAlloc::zeroCopy(count * getSize(type));
            }
            else {
                os << "CHECK_CUDA_ERRORS(cudaMalloc(&d_" << name << ", " << count << " * sizeof(" << type << ")));" << std::endl;
                allocation += MemAlloc::device(count * getSize(type));
            }
        }
    }

    return allocation;
}
//--------------------------------------------------------------------------
void Backend::genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const
{
    if(m_Preferences.automaticCopy) {
        os << "CHECK_CUDA_ERRORS(cudaFree(" << name << "));" << std::endl;
    }
    else {
        // **NOTE** because we pinned the variable we need to free it with cudaFreeHost rather than use the host code generator
        if(loc & VarLocation::HOST) {
            os << "CHECK_CUDA_ERRORS(cudaFreeHost(" << name << "));" << std::endl;
        }

        // If this variable wasn't allocated in zero-copy mode, free it
        if(loc & VarLocation::DEVICE) {
            os << "CHECK_CUDA_ERRORS(cudaFree(d_" << name << "));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamDefinition(CodeStream &definitions, const std::string &type, const std::string &name, VarLocation loc) const
{
    if(m_Preferences.automaticCopy) {
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
    if(m_Preferences.automaticCopy) {
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
void Backend::genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    // Get underlying type
    // **NOTE** could use std::remove_pointer but it seems unnecessarily elaborate
    const std::string underlyingType = ::Utils::getUnderlyingType(type);

    if(m_Preferences.automaticCopy) {
        os << "CHECK_CUDA_ERRORS(cudaMallocManaged(&" << name << ", count * sizeof(" << underlyingType << ")));" << std::endl;
    }
    else {
        if(loc & VarLocation::HOST) {
            const char *flags = (loc & VarLocation::ZERO_COPY) ? "cudaHostAllocMapped" : "cudaHostAllocPortable";
            os << "CHECK_CUDA_ERRORS(cudaHostAlloc(&" << name << ", count * sizeof(" << underlyingType << "), " << flags << "));" << std::endl;
        }

        // If variable is present on device at all
        if(loc & VarLocation::DEVICE) {
            if(loc & VarLocation::ZERO_COPY) {
                os << "CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void**)&d_" << name << ", (void*)" << name << ", 0));" << std::endl;
            }
            else {
                os << "CHECK_CUDA_ERRORS(cudaMalloc(&d_" << name << ", count * sizeof(" << underlyingType << ")));" << std::endl;
            }
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    assert(!m_Preferences.automaticCopy);

    if(!(loc & VarLocation::ZERO_COPY)) {
        // Get underlying type
        // **NOTE** could use std::remove_pointer but it seems unnecessarily elaborate
        const std::string underlyingType = ::Utils::getUnderlyingType(type);

        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << name;
        os << ", " << name;
        os << ", count * sizeof(" << underlyingType << "), cudaMemcpyHostToDevice));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    assert(!m_Preferences.automaticCopy);

    if(!(loc & VarLocation::ZERO_COPY)) {
        // Get underlying type
        // **NOTE** could use std::remove_pointer but it seems unnecessarily elaborate
        const std::string underlyingType = ::Utils::getUnderlyingType(type);

        os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << name;
        os << ", d_"  << name;
        os << ", count * sizeof(" << underlyingType << "), cudaMemcpyDeviceToHost));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genMergedGroupImplementation(CodeStream &os, const std::string &suffix, size_t idx, size_t numGroups) const
{
    const std::string prefix = m_Preferences.useConstantCacheForMergedStructs ? "__device__ __constant__" : "__device__";
    os << prefix << " Merged" << suffix << "Group" << idx << " d_merged" << suffix << "Group" << idx << "[" << numGroups << "];" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMergedGroupPush(CodeStream &os, const std::string &suffix, size_t idx, size_t numGroups) const
{
    os << "CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_merged" << suffix << "Group" << idx << ", group, ";
    os << numGroups << " * sizeof(Merged" << suffix << "Group" << idx << ")));" << std::endl;
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
void Backend::genPopVariableInit(CodeStream &os, const Substitutions &kernelSubs, Handler handler) const
{
    Substitutions varSubs(&kernelSubs);

    // If this is first thread in group
    os << "if(" << varSubs["id"] << " == 0)";
    {
        CodeStream::Scope b(os);
        handler(os, varSubs);
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableInit(CodeStream &os, const std::string &, const std::string &countVarName,
                              const Substitutions &kernelSubs, Handler handler) const
{
    // Variable should already be provided via parallelism
    assert(kernelSubs.hasVarSubstitution(countVarName));

    Substitutions varSubs(&kernelSubs);
    handler(os, varSubs);
}
//--------------------------------------------------------------------------
void Backend::genSynapseVariableRowInit(CodeStream &os, const SynapseGroupMerged &, 
                                        const Substitutions &kernelSubs, Handler handler) const
{
    // Pre and postsynaptic ID should already be provided via parallelism
    assert(kernelSubs.hasVarSubstitution("id_pre"));
    assert(kernelSubs.hasVarSubstitution("id_post"));

    Substitutions varSubs(&kernelSubs);
    varSubs.addVarSubstitution("id_syn", "(" + kernelSubs["id_pre"] + " * group.rowStride) + " + kernelSubs["id"]);
    handler(os, varSubs);
}
//--------------------------------------------------------------------------
void Backend::genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const
{
    assert(!m_Preferences.automaticCopy);

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
    assert(!m_Preferences.automaticCopy);

    if(!(loc & VarLocation::ZERO_COPY)) {
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << name;
        os << ", d_"  << name;
        os << ", " << count << " * sizeof(" << type << "), cudaMemcpyDeviceToHost));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, const std::string &name, VarLocation loc) const
{
    assert(!m_Preferences.automaticCopy);

    // If this variable requires queuing and isn't zero-copy
    if(ng.isVarQueueRequired(name) && ng.isDelayRequired() && !(loc & VarLocation::ZERO_COPY)) {
        // Generate memcpy to copy only current timestep's data
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
        os << ", " << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
        os << ", " << ng.getNumNeurons() << " * sizeof(" << type << "), cudaMemcpyHostToDevice));" << std::endl;
    }
    // Otherwise, generate standard s
    else {
        genVariablePush(os, type, name + ng.getName(), loc, false, ng.getNumNeurons());
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePull(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, const std::string &name, VarLocation loc) const
{
    assert(!m_Preferences.automaticCopy);

    // If this variable requires queuing and isn't zero-copy
    if(ng.isVarQueueRequired(name) && ng.isDelayRequired() && !(loc & VarLocation::ZERO_COPY)) {
        // Generate memcpy to copy only current timestep's data
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
        os << ", d_" << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
        os << ", " << ng.getNumNeurons() << " * sizeof(" << type << "), cudaMemcpyDeviceToHost));" << std::endl;
    }
    // Otherwise, generate standard pull
    else {
        genVariablePull(os, type, name + ng.getName(), loc, ng.getNumNeurons());
    }
}
//--------------------------------------------------------------------------
MemAlloc Backend::genGlobalRNG(CodeStream &, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &, CodeStream &) const
{
    // Define global Phillox RNG
    // **NOTE** this is actually accessed as a global so, unlike other variables, needs device global
    definitionsInternal << "EXPORT_VAR __device__ curandStatePhilox4_32_10_t d_rng;" << std::endl;

    // Implement global Phillox RNG
    runner << "__device__ curandStatePhilox4_32_10_t d_rng;" << std::endl;

    return MemAlloc::device(getSize("curandStatePhilox4_32_10_t"));
}
//--------------------------------------------------------------------------
MemAlloc Backend::genPopulationRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                                   const std::string &name, size_t count) const
{
    // Create an array or XORWOW RNGs
    return genArray(definitions, definitionsInternal, runner, allocations, free, "curandState", name, VarLocation::DEVICE, count);
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
    os << "\t\t\t<FastMath>" << (m_Preferences.optimizeCode ? "true" : "false") << "</FastMath>" << std::endl;
    os << "\t\t\t<GenerateLineInfo>" << (m_Preferences.generateLineInfo ? "true" : "false") << "</GenerateLineInfo>" << std::endl;
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
bool Backend::isGlobalRNGRequired(const ModelSpecMerged &modelMerged) const
{
    // If any neuron groups require  RNG for initialisation, return true
    // **NOTE** this takes postsynaptic model initialisation into account
    const ModelSpecInternal &model = modelMerged.getModel();
    if(std::any_of(model.getNeuronGroups().cbegin(), model.getNeuronGroups().cend(),
        [](const ModelSpec::NeuronGroupValueType &n)
        {
            return n.second.isInitRNGRequired();
        }))
    {
        return true;
    }

    // If any synapse groups require an RNG for weight update model initialisation or procedural connectivity, return true
    if(std::any_of(model.getSynapseGroups().cbegin(), model.getSynapseGroups().cend(),
        [](const ModelSpec::SynapseGroupValueType &s)
        {
            return (s.second.isWUInitRNGRequired() || s.second.isProceduralConnectivityRNGRequired());
        }))
    {
        return true;
    }

    return false;
}
//--------------------------------------------------------------------------
std::string Backend::getNVCCFlags() const
{
    const std::string architecture = "sm_" + std::to_string(getChosenCUDADevice().major) + std::to_string(getChosenCUDADevice().minor);
    std::string nvccFlags = "-x cu -arch " + architecture;
#ifndef _WIN32
    nvccFlags += " -std=c++11 --compiler-options '-fPIC -Wno-return-type-c-linkage'";
#endif

    nvccFlags += " " + m_Preferences.userNvccFlags;
    if(m_Preferences.optimizeCode) {
        nvccFlags += " -O3 -use_fast_math";
    }
    if(m_Preferences.debugCode) {
        nvccFlags += " -O0 -g -G";
    }
    if(m_Preferences.showPtxInfo) {
        nvccFlags += " -Xptxas \"-v\"";
    }
    if(m_Preferences.generateLineInfo) {
        nvccFlags += " --generate-line-info";
    }
#ifdef MPI_ENABLE
    // If MPI is enabled, add MPI include path
    nvccFlags +=" -I\"$(MPI_PATH)/include\"";
#endif
    return nvccFlags;
}
//--------------------------------------------------------------------------
std::string Backend::getFloatAtomicAdd(const std::string &ftype) const
{
    int version;
    cudaRuntimeGetVersion(&version);
    if (((getChosenCUDADevice().major < 2) && (ftype == "float"))
        || (((getChosenCUDADevice().major < 6) || (version < 8000)) && (ftype == "double"))) {
        return "atomicAddSW";
    }
    else {
        return "atomicAdd";
    }
}
//--------------------------------------------------------------------------
size_t Backend::getNumInitialisationRNGStreams(const ModelSpecMerged &modelMerged) const
{
    // Calculate total number of threads used for neuron initialisation group
    size_t numInitThreads = getNumMergedGroupThreads(modelMerged.getMergedNeuronInitGroups(),
                                                     [this](const NeuronGroupInternal &ng)
                                                     {
                                                         return padSize(ng.getNumNeurons(), getKernelBlockSize(Kernel::KernelInitialize));
                                                     });


    // Add on total number of threads used for dense synapse initialisation
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedSynapseDenseInitGroups(),
                                               [this](const SynapseGroupInternal &sg)
                                               {
                                                   return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), getKernelBlockSize(Kernel::KernelInitialize));
                                               });

    // Add on total number of threads used for synapse connectivity initialisation
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedSynapseConnectivityInitGroups(),
                                               [this](const SynapseGroupInternal &sg)
                                               {
                                                   return padSize(sg.getSrcNeuronGroup()->getNumNeurons(), getKernelBlockSize(Kernel::KernelInitialize));
                                               });

    // Finally, add on total number of threads used for sparse synapse initialisation
    numInitThreads += getNumMergedGroupThreads(modelMerged.getMergedSynapseDenseInitGroups(),
                                               [this](const SynapseGroupInternal &sg)
                                               {
                                                   return padSize(sg.getMaxConnections(), getKernelBlockSize(Kernel::KernelInitializeSparse));
                                               });

    return numInitThreads;
}
//--------------------------------------------------------------------------
size_t Backend::getNumPresynapticUpdateThreads(const SynapseGroupInternal &sg, const cudaDeviceProp &deviceProps,
                                               const Preferences &preferences)
{
     return getPresynapticUpdateStrategy(sg, deviceProps, preferences)->getNumThreads(sg);
}
//--------------------------------------------------------------------------
size_t Backend::getNumPostsynapticUpdateThreads(const SynapseGroupInternal &sg)
{
    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxSourceConnections();
    }
    else {
        return sg.getSrcNeuronGroup()->getNumNeurons();
    }
}
//--------------------------------------------------------------------------
size_t Backend::getNumSynapseDynamicsThreads(const SynapseGroupInternal &sg)
{
    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * (size_t)sg.getMaxConnections();
    }
    else {
        return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * (size_t)sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//--------------------------------------------------------------------------
void Backend::addPresynapticUpdateStrategy(PresynapticUpdateStrategy::Base *strategy)
{
    s_PresynapticUpdateStrategies.push_back(strategy);
}
//--------------------------------------------------------------------------
void Backend::genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const
{
    os << "const unsigned int spk" << suffix << "Idx = atomicAdd((unsigned int *) &shSpk" << suffix << "Count, 1);" << std::endl;
    os << "shSpk" << suffix << "[spk" << suffix << "Idx] = " << subs["id"] << ";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikePush(CodeStream &os, const NeuronGroupInternal &ng, bool spikeEvent) const
{
    assert(!m_Preferences.automaticCopy);

    if(!(ng.getSpikeLocation() & VarLocation::ZERO_COPY)) {
        // Is delay required
        const bool delayRequired = spikeEvent ?
            ng.isDelayRequired() :
            (ng.isTrueSpikeRequired() && ng.isDelayRequired());

        const char *spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
        const char *spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

        if (delayRequired) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
            os << ", " << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
            os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikePrefix << ng.getName() << " + (spkQuePtr" << ng.getName() << "*" << ng.getNumNeurons() << ")";
            os << ", " << spikePrefix << ng.getName();
            os << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", " << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << "] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
        }
        else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikeCntPrefix << ng.getName();
            os << ", " << spikeCntPrefix << ng.getName();
            os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikePrefix << ng.getName();
            os << ", " << spikePrefix << ng.getName();
            os << ", " << spikeCntPrefix << ng.getName() << "[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikePull(CodeStream &os, const NeuronGroupInternal &ng, bool spikeEvent) const
{
    if(!(ng.getSpikeLocation() & VarLocation::ZERO_COPY)) {
        // Is delay required
        const bool delayRequired = spikeEvent ?
            ng.isDelayRequired() :
            (ng.isTrueSpikeRequired() && ng.isDelayRequired());

        const char *spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
        const char *spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

        if (delayRequired) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
            os << ", d_" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
            os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;

            os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikePrefix << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", d_" << spikePrefix << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
            os << ", " << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << "] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
        }
        else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikeCntPrefix << ng.getName();
            os << ", d_" << spikeCntPrefix << ng.getName();
            os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikePrefix << ng.getName();
            os << ", d_" << spikePrefix << ng.getName();
            os << ", " << spikeCntPrefix << ng.getName() << "[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreads) const
{
    // Calculate grid size
    const size_t gridSize = ceilDivide(numThreads, m_KernelBlockSizes[kernel]);
    os << "const dim3 threads(" << m_KernelBlockSizes[kernel] << ", 1);" << std::endl;

    if (gridSize < (size_t)getChosenCUDADevice().maxGridSize[0]) {
        os << "const dim3 grid(" << gridSize << ", 1);" << std::endl;
    }
    else {
        // **TODO** this needs to be implemented in genParallelGroup
        assert(false);
        const size_t squareGridSize = (size_t)std::ceil(std::sqrt(gridSize));
        os << "const dim3 grid(" << squareGridSize << ", "<< squareGridSize <<");" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::addDeviceType(const std::string &type, size_t size)
{
    addType(type, size);
    m_DeviceTypes.emplace(type);
}
//--------------------------------------------------------------------------
bool Backend::isDeviceType(const std::string &type) const
{
    // Get underlying type
    const std::string underlyingType = ::Utils::isTypePointer(type) ? ::Utils::getUnderlyingType(type) : type;

    // Return true if it is in device types set
    return (m_DeviceTypes.find(underlyingType) != m_DeviceTypes.cend());
}
//--------------------------------------------------------------------------
const PresynapticUpdateStrategy::Base *Backend::getPresynapticUpdateStrategy(const SynapseGroupInternal &sg,
                                                                             const cudaDeviceProp &deviceProps,
                                                                             const Preferences &preferences)
{
    // Loop through presynaptic update strategies until we find one that is compatible with this synapse group
    // **NOTE** this is done backwards so that user-registered strategies get first priority
    for(auto s = s_PresynapticUpdateStrategies.rbegin(); s != s_PresynapticUpdateStrategies.rend(); ++s) {
        if((*s)->isCompatible(sg, deviceProps, preferences)) {
            return *s;
        }
    }

    throw std::runtime_error("Unable to find a suitable presynaptic update strategy for synapse group '" + sg.getName() + "'");
    return nullptr;
}
}   // namespace CUDA
}   // namespace CodeGenerator

#include "backend.h"

// Standard C++ includes
#include <algorithm>
#include <iterator>

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "gennUtils.h"
#include "modelSpecMerged.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/substitutions.h"
#include "code_generator/codeGenUtils.h"

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


void gennExtraGlobalParamPass(CodeGenerator::CodeStream &os, const std::map<std::string, std::string>::value_type &p)
{
    if(Utils::isTypePointer(p.second)) {
        os << "d_" << p.first << ", ";
    }
    else {
        os << p.first << ", ";
    }
}
//-----------------------------------------------------------------------
template<typename T>
void genMergedKernelDataStructures(CodeGenerator::CodeStream &os, const std::vector<T> &mergedGroups,
                                   const std::string &prefix, size_t blockSize,
                                   std::function<bool(const T&)> filter,
                                   std::function<size_t(const T&, const typename T::GroupInternal&)> getNumThreads)
{
    // Declare array of indices (into mergedNeuronGroupXXX arrays)
    os << "__device__ __constant__ uint16_t dd_" << prefix << "GroupBlockIndices[] = {";
    for(const auto &m : mergedGroups) {
        if(filter(m)) {
            // Loop through neuron groups within merged neuron group
            size_t n = 0;
            for(const auto &ng : m.getGroups()) {
                // Write index to this neuron group for each block used to simulate it
                const size_t numBlocks = CodeGenerator::ceilDivide(getNumThreads(m, ng.get()), blockSize);
                std::fill_n(std::ostream_iterator<std::string>(os), numBlocks,
                            std::to_string(n++) + ", ");
            }
        }
    };
    os << "};" << std::endl;

    // Loop through merged groups
    size_t id = 0;
    for(const auto &m : mergedGroups) {
        if(filter(m)) {
            // Declare array of starting thread indices for each neuron group
            os << "__device__ __constant__ unsigned int dd_" << prefix << "GroupStartID" << m.getIndex() << "[] = {";
            for(const auto &ng : m.getGroups()) {
                os << id << ", ";
                id += CodeGenerator::padSize(getNumThreads(m, ng.get()), blockSize);
            }
            os << "};" << std::endl;
        }
    }
}
//-----------------------------------------------------------------------
bool isSparseInitRequired(const SynapseGroupInternal &sg)
{
    return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE)
            && (sg.isWUVarInitRequired() || !sg.getWUModel()->getLearnPostCode().empty() || !sg.getWUModel()->getSynapseDynamicsCode().empty()));
}
//-----------------------------------------------------------------------
void updateExtraGlobalParams(const std::string &varSuffix, const std::string &codeSuffix, const Snippet::Base::EGPVec &extraGlobalParameters,
                             std::map<std::string, std::string> &kernelParameters, const std::vector<std::string> &codeStrings)
{
    // Loop through list of global parameters
    for(const auto &p : extraGlobalParameters) {
        // If this parameter is used in any codestrings, add it to list of kernel parameters
        if(std::any_of(codeStrings.cbegin(), codeStrings.cend(),
            [p, codeSuffix](const std::string &c){ return c.find("$(" + p.name + codeSuffix + ")") != std::string::npos; }))
        {
            kernelParameters.emplace(p.name + varSuffix, p.type);
        }
    }
}
//--------------------------------------------------------------------------
void updateSynapseGroupExtraGlobalParams(const SynapseGroupInternal &sg, std::map<std::string, std::string> &kernelParameters,
                                         const std::vector<std::string> &codeStrings)
{
    // Synapse kernel
    // --------------
    // Add any of the pre or postsynaptic neuron group's extra global
    // parameters referenced in code strings to the map of kernel parameters
    updateExtraGlobalParams(sg.getSrcNeuronGroup()->getName(), "_pre", sg.getSrcNeuronGroup()->getNeuronModel()->getExtraGlobalParams(),
                             kernelParameters, codeStrings);
    updateExtraGlobalParams(sg.getTrgNeuronGroup()->getName(), "_post", sg.getTrgNeuronGroup()->getNeuronModel()->getExtraGlobalParams(),
                             kernelParameters, codeStrings);

    // Finally add any weight update model extra global parameters referenced in code strings to the map of kernel paramters
    updateExtraGlobalParams(sg.getName(), "", sg.getWUModel()->getExtraGlobalParams(), kernelParameters, codeStrings);
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
                 int localHostID, const std::string &scalarType, int device)
:   BackendBase(localHostID, scalarType), m_KernelBlockSizes(kernelBlockSizes), m_Preferences(preferences), m_ChosenDeviceID(device)
{
    // Set device
    CHECK_CUDA_ERRORS(cudaSetDevice(device));

    // Get device properties
    CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&m_ChosenDevice, device));

    // Get CUDA runtime version
    cudaRuntimeGetVersion(&m_RuntimeVersion);

    // Add CUDA-specific types, additionally marking them as 'device types' innaccesible to host code
    addDeviceType("curandState", 44);
    addDeviceType("curandStatePhilox4_32_10_t", 64);
    addDeviceType("half", 2);
}
//--------------------------------------------------------------------------
void Backend::genNeuronUpdate(CodeStream &os, const ModelSpecMerged &model, 
                              NeuronGroupSimHandler simHandler, NeuronGroupMergedHandler wuVarUpdateHandler) const
{
    // Generate data structure for accessing merged groups
    genMergedKernelDataStructures<NeuronGroupMerged>(
        os, model.getMergedLocalNeuronGroups(), "neuron", m_KernelBlockSizes[KernelNeuronUpdate],
        [](const NeuronGroupMerged&){ return true; },
        [](const NeuronGroupMerged&, const NeuronGroupInternal &ng){ return ng.getNumNeurons(); });

    // Generate reset kernel to be run before the neuron kernel
    size_t idPreNeuronReset = 0;
    os << "extern \"C\" __global__ void " << KernelNames[KernelPreNeuronReset] << "()";
    {
        CodeStream::Scope b(os);

        os << "const unsigned int id = " << m_KernelBlockSizes[KernelPreNeuronReset] << " * blockIdx.x + threadIdx.x;" << std::endl;

        // Loop through remote neuron groups
        for(const auto &n : model.getModel().getRemoteNeuronGroups()) {
            if(n.second.hasOutputToHost(getLocalHostID()) && n.second.isDelayRequired()) {
                if(idPreNeuronReset > 0) {
                    os << "else ";
                }
                os << "if(id == " << (idPreNeuronReset++) << ")";
                {
                    CodeStream::Scope b(os);
                    os << "dd_spkQuePtr" << n.first << " = (dd_spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
                }
            }
        }

        // Loop through local neuron groups
        for(const auto &n : model.getModel().getLocalNeuronGroups()) {
            if(idPreNeuronReset > 0) {
                os << "else ";
            }
            os << "if(id == " << (idPreNeuronReset++) << ")";
            {
                CodeStream::Scope b(os);

                if (n.second.isDelayRequired()) { // with delay
                    os << "dd_spkQuePtr" << n.first << " = (dd_spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;

                    if (n.second.isSpikeEventRequired()) {
                        os << "dd_glbSpkCntEvnt" << n.first << "[dd_spkQuePtr" << n.first << "] = 0;" << std::endl;
                    }
                    if (n.second.isTrueSpikeRequired()) {
                        os << "dd_glbSpkCnt" << n.first << "[dd_spkQuePtr" << n.first << "] = 0;" << std::endl;
                    }
                    else {
                        os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
                    }
                }
                else { // no delay
                    if (n.second.isSpikeEventRequired()) {
                        os << "dd_glbSpkCntEvnt" << n.first << "[0] = 0;" << std::endl;
                    }
                    os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
                }
            }
        }
    }

    size_t idStart = 0;
    os << "extern \"C\" __global__ void " << KernelNames[KernelNeuronUpdate] << "("  << model.getTimePrecision() << " t)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "const unsigned int id = " << m_KernelBlockSizes[KernelNeuronUpdate] << " * blockIdx.x + threadIdx.x; " << std::endl;
        os << "const unsigned int blk = blockIdx.x;" << std::endl;

        Substitutions kernelSubs(cudaFunctions, model.getPrecision());
        kernelSubs.addVarSubstitution("t", "t");

        // If any neuron groups emit spike events
        if(std::any_of(model.getMergedLocalNeuronGroups().cbegin(), model.getMergedLocalNeuronGroups().cend(),
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
        if(std::any_of(model.getMergedLocalNeuronGroups().cbegin(), model.getMergedLocalNeuronGroups().cend(),
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
        genParallelGroup<NeuronGroupMerged>(os, kernelSubs, model.getMergedLocalNeuronGroups(), idStart,
            [this](const NeuronGroupMerged &, const NeuronGroupInternal &ng)
            {
                return padSize(ng.getNumNeurons(), getKernelBlockSize(KernelNeuronUpdate));
            },
            [&model, simHandler, wuVarUpdateHandler, this](CodeStream &os, const NeuronGroupMerged &ng, Substitutions &popSubs)
            {
                // Get the index of the neuron group within the merged group
                os << "const unsigned int neuronGroupIndex = dd_neuronGroupBlockIndices[blk];" << std::endl;

                // Use this to get reference to MergedNeuronGroup structure
                // Get reference to neuron group that this block should be simulating
                os << "const MergedNeuronGroup" << ng.getIndex() << " &neuronGroup = dd_mergedNeuronGroup" << ng.getIndex() << "[neuronGroupIndex]; " << std::endl;

                // Use this and starting thread of merged group to calculate local id within neuron group
                os << "const unsigned int lid = id - (dd_neuronGroupStartID" << ng.getIndex() << "[neuronGroupIndex]);" << std::endl;
                popSubs.addVarSubstitution("id", "lid");

                // **TODO** calculate 
                // If axonal delays are required
                if (ng.getArchetype().isDelayRequired()) {
                    assert(false);
                    // We should READ from delay slot before spkQuePtr
                    os << "const unsigned int readDelayOffset = " << ng.getArchetype().getPrevQueueOffset("dd_") << ";" << std::endl;

                    // And we should WRITE to delay slot pointed to be spkQuePtr
                    os << "const unsigned int writeDelayOffset = " << ng.getArchetype().getCurrentQueueOffset("dd_") << ";" << std::endl;
                }
                os << std::endl;

                // If this neuron group requires a simulation RNG, substitute in this neuron group's RNG
                if(ng.getArchetype().isSimRNGRequired()) {
                    popSubs.addVarSubstitution("rng", "&neuronGroup.rng[" + popSubs["id"] + "]");
                }

                // Call handler to generate generic neuron code
                os << "if(" << popSubs["id"] << " < neuronGroup.numNeurons)";
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
                            os << "shPosSpkEvnt = atomicAdd((unsigned int*)&neurongroup.spkCntEvnt";
                            if (ng.getArchetype().isDelayRequired()) {
                                os << "[*neuronGroup.spkQuePtr], shSpkEvntCount);" << std::endl;
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
                            os << "shPosSpk = atomicAdd((unsigned int*)&neuronGroup.spkCnt";
                            if (ng.getArchetype().isDelayRequired() && ng.getArchetype().isTrueSpikeRequired()) {
                                os << "[*neuronGroup.spkQuePtr], shSpkCount);" << std::endl;
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
                    assert(false);
                    os << "if (threadIdx.x < shSpkEvntCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "neuronGroup.spkEvnt[" << queueOffset << "shPosSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << std::endl;
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

                        os << "neuronGroup.spk[" << queueOffsetTrueSpk << "shPosSpk + threadIdx.x] = n;" << std::endl;
                        if (ng.getArchetype().isSpikeTimeRequired()) {
                            assert(false);
                            //os << "dd_sT" << ng.getArchetype().getName() << "[" << queueOffset << "n] = t;" << std::endl;
                        }
                    }
                }
            }
        );
    }

    os << "void updateNeurons(" << model.getTimePrecision() << ")";
    {
        CodeStream::Scope b(os);
        if(idPreNeuronReset > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelPreNeuronReset, idPreNeuronReset);
            os << KernelNames[KernelPreNeuronReset] << "<<<grid, threads>>>();" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
        if(idStart > 0) {
            CodeStream::Scope b(os);
            Timer t(os, "neuronUpdate", model.getModel().isTimingEnabled());

            genKernelDimensions(os, KernelNeuronUpdate, idStart);
            os << KernelNames[KernelNeuronUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genSynapseUpdate(CodeStream &os, const ModelSpecMerged &model,
                               SynapseGroupMergedHandler wumThreshHandler, SynapseGroupMergedHandler wumSimHandler,
                               SynapseGroupMergedHandler wumEventHandler, SynapseGroupMergedHandler wumProceduralConnectHandler,
                               SynapseGroupMergedHandler postLearnHandler, SynapseGroupMergedHandler synapseDynamicsHandler) const
{
    // If any synapse groups require dendritic delay, a reset kernel is required to be run before the synapse kernel
    size_t idPreSynapseReset = 0;
    if(std::any_of(model.getMergedLocalSynapseGroups().cbegin(), model.getMergedLocalSynapseGroups().cend(),
                   [](const SynapseGroupMerged &s){ return s.getArchetype().isDendriticDelayRequired(); }))
    {
        // pre synapse reset kernel header
        os << "extern \"C\" __global__ void " << KernelNames[KernelPreSynapseReset] << "()";
        {
            CodeStream::Scope b(os);

            os << "const unsigned int id = " << m_KernelBlockSizes[KernelPreSynapseReset] << " * blockIdx.x + threadIdx.x;" << std::endl;

            // Loop through neuron groups
            for(const auto &n : model.getModel().getLocalNeuronGroups()) {
                // Loop through incoming synaptic populations
                for(const auto &m : n.second.getMergedInSyn()) {
                    const auto *sg = m.first;

                     // If this kernel requires dendritic delay
                    if(sg->isDendriticDelayRequired()) {
                        if(idPreSynapseReset > 0) {
                            os << "else ";
                        }
                        os << "if(id == " << (idPreSynapseReset++) << ")";
                        {
                            CodeStream::Scope b(os);

                            os << "dd_denDelayPtr" << sg->getPSModelTargetName() << " = (dd_denDelayPtr" << sg->getPSModelTargetName() << " + 1) % " << sg->getMaxDendriticDelayTimesteps() << ";" << std::endl;
                        }
                    }
                }
            }
        }
    }

    // If any synapse groups require spike-driven presynaptic updates
    size_t idPresynapticStart = 0;
    if(std::any_of(model.getMergedLocalSynapseGroups().cbegin(), model.getMergedLocalSynapseGroups().cend(),
                   [](const SynapseGroupMerged &s){ return (s.getArchetype().isSpikeEventRequired() || s.getArchetype().isTrueSpikeRequired()); }))
    {
        // Generate data structure for accessing merged groups
        genMergedKernelDataStructures<SynapseGroupMerged>(
            os, model.getMergedLocalSynapseGroups(), "presynaptic", m_KernelBlockSizes[KernelPresynapticUpdate],
            [](const SynapseGroupMerged &sg){ return (sg.getArchetype().isSpikeEventRequired() || sg.getArchetype().isTrueSpikeRequired()); },
            [this](const SynapseGroupMerged &sgMerge, const SynapseGroupInternal &sg)
            {
                return getNumPresynapticUpdateThreads(sgMerge, sg, m_ChosenDevice, m_Preferences);
            });

        os << "extern \"C\" __global__ void " << KernelNames[KernelPresynapticUpdate] << "(" << model.getTimePrecision() << " t)" << std::endl; // end of synapse kernel header
        {
            CodeStream::Scope b(os);

            Substitutions kernelSubs(cudaFunctions, model.getPrecision());
            kernelSubs.addVarSubstitution("t", "t");

            os << "const unsigned int id = " << m_KernelBlockSizes[KernelPresynapticUpdate] << " * blockIdx.x + threadIdx.x; " << std::endl;
            os << "const unsigned int blk = blockIdx.x;" << std::endl;

            // We need shLg if any synapse groups accumulate into shared memory
            // Determine the maximum shared memory outputs 
            size_t maxSharedMemPerThread = 0;
            for (const auto &s : model.getMergedLocalSynapseGroups()) {
                if (s.getArchetype().isTrueSpikeRequired() || !s.getArchetype().getWUModel()->getLearnPostCode().empty()) {
                    maxSharedMemPerThread = std::max(maxSharedMemPerThread,
                                                     getPresynapticUpdateStrategy(s)->getSharedMemoryPerThread(s, *this));
                }
            }

            // If any shared memory is required, declare array
            if(maxSharedMemPerThread > 0) {
                os << "__shared__ " << model.getPrecision() << " shLg[" << maxSharedMemPerThread * m_KernelBlockSizes[KernelPresynapticUpdate] << "];" << std::endl;
            }

            // If any of these synapse groups also have sparse connectivity, allocate shared memory for row length
            if(std::any_of(model.getMergedLocalSynapseGroups().cbegin(), model.getMergedLocalSynapseGroups().cend(),
                           [&model](const SynapseGroupMerged &s)
                           {
                               return (s.getArchetype().getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC
                                       && (s.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE));
                           }))
            {
                os << "__shared__ unsigned int shRowLength[" << m_KernelBlockSizes[KernelPresynapticUpdate] << "];" << std::endl;
            }

            if(std::any_of(model.getMergedLocalSynapseGroups().cbegin(), model.getMergedLocalSynapseGroups().cend(),
                           [&model](const SynapseGroupMerged &s)
                           {
                               return (s.getArchetype().isTrueSpikeRequired() || !s.getArchetype().getWUModel()->getLearnPostCode().empty());
                           }))
            {
                os << "__shared__ unsigned int shSpk[" << m_KernelBlockSizes[KernelPresynapticUpdate] << "];" << std::endl;
            }

            if(std::any_of(model.getMergedLocalSynapseGroups().cbegin(), model.getMergedLocalSynapseGroups().cend(),
                           [](const SynapseGroupMerged &s){ return (s.getArchetype().isSpikeEventRequired()); }))
            {
                os << "__shared__ unsigned int shSpkEvnt[" << m_KernelBlockSizes[KernelPresynapticUpdate] << "];" << std::endl;
            }

            // Parallelise over synapse groups
            genParallelGroup<SynapseGroupMerged>(os, kernelSubs, model.getMergedLocalSynapseGroups(), idPresynapticStart,
                [this](const SynapseGroupMerged &sgMerge, const SynapseGroupInternal &sg)
                {
                    return padSize(getNumPresynapticUpdateThreads(sgMerge, sg, m_ChosenDevice, m_Preferences), m_KernelBlockSizes[KernelPresynapticUpdate]);
                },
                [](const SynapseGroupMerged &sg){ return (sg.getArchetype().isSpikeEventRequired() || sg.getArchetype().isTrueSpikeRequired()); },
                [&idPresynapticStart, wumThreshHandler, wumSimHandler, wumEventHandler, wumProceduralConnectHandler, &model, this]
                (CodeStream &os, const SynapseGroupMerged &sg, Substitutions &popSubs)
                {
                    // Get the index of the neuron group within the merged group
                    os << "const unsigned int synapticGroupIndex = dd_presynapticGroupBlockIndices[blk];" << std::endl;

                    // Use this to get reference to MergedSynapseGroup structure
                    // Get reference to neuron group that this block should be simulating
                    os << "const MergedSynapseGroup" << sg.getIndex() << " &synapseGroup = dd_mergedSynapseGroup" << sg.getIndex() << "[synapticGroupIndex]; " << std::endl;

                    // Use this and starting thread of merged group to calculate local id within neuron group
                    os << "const unsigned int lid = id - (dd_presynapticGroupStartID" << sg.getIndex() << "[synapticGroupIndex]);" << std::endl;
                    popSubs.addVarSubstitution("id", "lid");

                    // Get presynaptic update strategy to use for this synapse group
                    const auto *presynapticUpdateStrategy = getPresynapticUpdateStrategy(sg);
                    LOGD << "Using '" << typeid(*presynapticUpdateStrategy).name() << "' presynaptic update strategy for merged synapse group '" << sg.getIndex() << "'";

                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
                        assert(false);
                        //os << "const unsigned int preReadDelaySlot = " << sg.getArchetype().getPresynapticAxonalDelaySlot("dd_") << ";" << std::endl;
                        os << "const unsigned int preReadDelayOffset = preReadDelaySlot * synapseGroup.numSrcNeurons;" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
                        assert(false);
                        //os << "const unsigned int postReadDelayOffset = " << sg.getArchetype().getPostsynapticBackPropDelaySlot("dd_") << " * " << sg.getArchetype().getTrgNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }

                    // Generate preamble
                    presynapticUpdateStrategy->genPreamble(os, model, sg, popSubs, *this, idPresynapticStart);
                  
                    // If spike events should be processed
                    if (sg.getArchetype().isSpikeEventRequired()) {
                        CodeStream::Scope b(os);
                        presynapticUpdateStrategy->genUpdate(os, model, sg, popSubs, *this, false, idPresynapticStart,
                                                             wumThreshHandler, wumEventHandler, wumProceduralConnectHandler);
                    }

                    // If true spikes should be processed
                    if (sg.getArchetype().isTrueSpikeRequired()) {
                        CodeStream::Scope b(os);
                        presynapticUpdateStrategy->genUpdate(os, model, sg, popSubs, *this, true, idPresynapticStart,
                                                             wumThreshHandler, wumSimHandler, wumProceduralConnectHandler);
                    }

                    os << std::endl;

                    // Generate pre-amble
                    presynapticUpdateStrategy->genPostamble(os, model, sg, popSubs, *this, idPresynapticStart);
                }
            );
        }
    }

    // If any synapse groups require postsynaptic learning
    /*size_t idPostsynapticStart = 0;
    if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
        [](const ModelSpec::SynapseGroupValueType &s){ return !s.second.getWUModel()->getLearnPostCode().empty(); }))
    {
        os << "extern \"C\" __global__ void " << KernelNames[KernelPostsynapticUpdate] << "(";
        for (const auto &p : postsynapticKernelParameters) {
            os << p.second << " " << p.first << ", ";
        }
        os << model.getTimePrecision() << " t)" << std::endl; // end of synapse kernel header
        {
            CodeStream::Scope b(os);

            Substitutions kernelSubs(cudaFunctions, model.getPrecision());
            kernelSubs.addVarSubstitution("t", "t");

            os << "const unsigned int id = " << m_KernelBlockSizes[KernelPostsynapticUpdate] << " * blockIdx.x + threadIdx.x; " << std::endl;
            os << "__shared__ unsigned int shSpk[" << m_KernelBlockSizes[KernelPostsynapticUpdate] << "];" << std::endl;
            if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
                [&model](const ModelSpec::SynapseGroupValueType &s)
                {
                    return ((s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getLearnPostCode().empty());
                }))
            {
                os << "__shared__ unsigned int shColLength[" << m_KernelBlockSizes[KernelPostsynapticUpdate] << "];" << std::endl;
            }

            // Parallelise over synapse groups whose weight update models have code for postsynaptic learning
            genParallelGroup<SynapseGroupInternal>(os, kernelSubs, model.getLocalSynapseGroups(), idPostsynapticStart,
                [this](const SynapseGroupInternal &sg){ return padSize(getNumPostsynapticUpdateThreads(sg), m_KernelBlockSizes[KernelPostsynapticUpdate]); },
                [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getLearnPostCode().empty(); },
                [postLearnHandler, &model, this](CodeStream &os, const SynapseGroupInternal &sg, const Substitutions &popSubs)
                {
                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(sg.getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelayOffset = " << sg.getPresynapticAxonalDelaySlot("dd_") << " * " << sg.getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(sg.getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelaySlot = " << sg.getPostsynapticBackPropDelaySlot("dd_") << ";" << std::endl;
                        os << "const unsigned int postReadDelayOffset = postReadDelaySlot * " << sg.getTrgNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }

                    if (sg.getTrgNeuronGroup()->isDelayRequired() && sg.getTrgNeuronGroup()->isTrueSpikeRequired()) {
                        os << "const unsigned int numSpikes = dd_glbSpkCnt" << sg.getTrgNeuronGroup()->getName() << "[postReadDelaySlot];" << std::endl;
                    }
                    else {
                        os << "const unsigned int numSpikes = dd_glbSpkCnt" << sg.getTrgNeuronGroup()->getName() << "[0];" << std::endl;
                    }

                    os << "const unsigned int numSpikeBlocks = (numSpikes + " << m_KernelBlockSizes[KernelPostsynapticUpdate]-1 << ") / " << m_KernelBlockSizes[KernelPostsynapticUpdate] << ";" << std::endl;
                    os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
                    {
                        CodeStream::Scope b(os);
                        os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << m_KernelBlockSizes[KernelPostsynapticUpdate] << ") + 1 : " << m_KernelBlockSizes[KernelPostsynapticUpdate] << ";" << std::endl;

                        os << "if (threadIdx.x < numSpikesInBlock)";
                        {
                            CodeStream::Scope b(os);
                            const std::string offsetTrueSpkPost = (sg.getTrgNeuronGroup()->isTrueSpikeRequired() && sg.getTrgNeuronGroup()->isDelayRequired()) ? "postReadDelayOffset + " : "";
                            os << "const unsigned int spk = dd_glbSpk" << sg.getTrgNeuronGroup()->getName() << "[" << offsetTrueSpkPost << "(r * " << m_KernelBlockSizes[KernelPostsynapticUpdate] << ") + threadIdx.x];" << std::endl;
                            os << "shSpk[threadIdx.x] = spk;" << std::endl;

                            if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                os << "shColLength[threadIdx.x] = dd_colLength" << sg.getName() << "[spk];" << std::endl;
                            }
                        }

                        os << "__syncthreads();" << std::endl;
                        os << "// only work on existing neurons" << std::endl;
                        os << "if (" << popSubs["id"] << " < " << sg.getMaxSourceConnections() << ")";
                        {
                            CodeStream::Scope b(os);
                            os << "// loop through all incoming spikes for learning" << std::endl;
                            os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
                            {
                                CodeStream::Scope b(os);

                                Substitutions synSubs(&popSubs);
                                if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    os << "if (" << synSubs["id"] << " < shColLength[j])" << CodeStream::OB(1540);
                                    os << "const unsigned int synAddress = dd_remap" << sg.getName() << "[(shSpk[j] * " << sg.getMaxSourceConnections() << ") + " << popSubs["id"] << "];" << std::endl;
                                    os << "const unsigned int ipre = synAddress / " << getSynapticMatrixRowStride(sg) << ";" << std::endl;
                                    synSubs.addVarSubstitution("id_pre", "ipre");
                                }
                                else {
                                    os << "const unsigned int synAddress = (" << synSubs["id"] << " * " << std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()) << ") + shSpk[j];" << std::endl;
                                    synSubs.addVarSubstitution("id_pre", synSubs["id"]);
                                }

                                synSubs.addVarSubstitution("id_post", "shSpk[j]");
                                synSubs.addVarSubstitution("id_syn", "synAddress");

                                postLearnHandler(os, sg, synSubs);

                                if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
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
    if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
        [](const ModelSpec::SynapseGroupValueType &s){ return !s.second.getWUModel()->getSynapseDynamicsCode().empty(); }))
    {
        os << "extern \"C\" __global__ void " << KernelNames[KernelSynapseDynamicsUpdate] << "(";
        for (const auto &p : synapseDynamicsKernelParameters) {
            os << p.second << " " << p.first << ", ";
        }
        os << model.getTimePrecision() << " t)" << std::endl; // end of synapse kernel header
        {
            CodeStream::Scope b(os);
            os << "const unsigned int id = " << m_KernelBlockSizes[KernelSynapseDynamicsUpdate] << " * blockIdx.x + threadIdx.x;" << std::endl;

            Substitutions kernelSubs(cudaFunctions, model.getPrecision());
            kernelSubs.addVarSubstitution("t", "t");

            // Parallelise over synapse groups whose weight update models have code for synapse dynamics
            genParallelGroup<SynapseGroupInternal>(os, kernelSubs, model.getLocalSynapseGroups(), idSynapseDynamicsStart,
                [this](const SynapseGroupInternal &sg){ return padSize(getNumSynapseDynamicsThreads(sg), m_KernelBlockSizes[KernelSynapseDynamicsUpdate]); },
                [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getSynapseDynamicsCode().empty(); },
                [synapseDynamicsHandler, &model, this](CodeStream &os, const SynapseGroupInternal &sg, const Substitutions &popSubs)
                {
                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(sg.getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelayOffset = " << sg.getPresynapticAxonalDelaySlot("dd_") << " * " << sg.getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(sg.getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelayOffset = " << sg.getPostsynapticBackPropDelaySlot("dd_") << " * " << sg.getTrgNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }

                    Substitutions synSubs(&popSubs);

                    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        os << "if (" << popSubs["id"] << " < dd_synRemap" << sg.getName() << "[0])";
                    }
                    else {
                        os << "if (" << popSubs["id"] << " < " << sg.getSrcNeuronGroup()->getNumNeurons() * sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    }
                    {
                        CodeStream::Scope b(os);

                        if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                            // Determine synapse and presynaptic indices for this thread
                            os << "const unsigned int s = dd_synRemap" << sg.getName() << "[1 + " << popSubs["id"] << "];" << std::endl;

                            synSubs.addVarSubstitution("id_pre", "s / " + std::to_string(getSynapticMatrixRowStride(sg)));
                            synSubs.addVarSubstitution("id_post", "dd_ind" + sg.getName() + "[s]");
                            synSubs.addVarSubstitution("id_syn", "s");
                        }
                        else {
                            synSubs.addVarSubstitution("id_pre", popSubs["id"] + " / " + std::to_string(getSynapticMatrixRowStride(sg)));
                            synSubs.addVarSubstitution("id_post", popSubs["id"] + " % " + std::to_string(getSynapticMatrixRowStride(sg)));
                            synSubs.addVarSubstitution("id_syn", popSubs["id"]);
                        }

                        // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                        if(sg.isDendriticDelayRequired()) {
                            synSubs.addFuncSubstitution("addToInSynDelay", 2, getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + synSubs["id_post"] + "], $(0))");
                        }
                        // Otherwise
                        else {
                            synSubs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[" + synSubs["id_post"] + "], $(0))");
                        }

                        synapseDynamicsHandler(os, sg, synSubs);
                    }
                });
        }
    }*/

    os << "void updateSynapses(" << model.getTimePrecision() << " t)";
    {
        CodeStream::Scope b(os);

        // Launch pre-synapse reset kernel if required
        if(idPreSynapseReset > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelPreSynapseReset, idPreSynapseReset);
            os << KernelNames[KernelPreSynapseReset] << "<<<grid, threads>>>();" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch synapse dynamics kernel if required
        /*if(idSynapseDynamicsStart > 0) {
            CodeStream::Scope b(os);
            Timer t(os, "synapseDynamics", model.isTimingEnabled());

            genKernelDimensions(os, KernelSynapseDynamicsUpdate, idSynapseDynamicsStart);
            os << KernelNames[KernelSynapseDynamicsUpdate] << "<<<grid, threads>>>(";
            for(const auto &p : synapseDynamicsKernelParameters) {
                gennExtraGlobalParamPass(os, p);
            }
            os << "t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }*/

        // Launch presynaptic update kernel
        if(idPresynapticStart > 0) {
            CodeStream::Scope b(os);
            Timer t(os, "presynapticUpdate", model.isTimingEnabled());

            genKernelDimensions(os, KernelPresynapticUpdate, idPresynapticStart);
            os << KernelNames[KernelPresynapticUpdate] << "<<<grid, threads>>>(t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }

        // Launch postsynaptic update kernel
        /*if(idPostsynapticStart > 0) {
            CodeStream::Scope b(os);
            Timer t(os, "postsynapticUpdate", model.isTimingEnabled());

            genKernelDimensions(os, KernelPostsynapticUpdate, idPostsynapticStart);
            os << KernelNames[KernelPostsynapticUpdate] << "<<<grid, threads>>>(";
            for(const auto &p : postsynapticKernelParameters) {
                gennExtraGlobalParamPass(os, p);
            }
            os << "t);" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
        }*/
    }
}
//--------------------------------------------------------------------------
void Backend::genInit(CodeStream &os, const ModelSpecInternal &model,
                      NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                      SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                      SynapseGroupHandler sgSparseInitHandler) const
{
    os << "#include <iostream>" << std::endl;
    os << "#include <random>" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << std::endl;

    // If device RNG is required, generate kernel to initialise it
    if(isGlobalRNGRequired(model)) {
        os << "extern \"C\" __global__ void initializeRNGKernel(unsigned long long deviceRNGSeed)";
        {
            CodeStream::Scope b(os);
            os << "if(threadIdx.x == 0)";
            {
                CodeStream::Scope b(os);
                os << "curand_init(deviceRNGSeed, 0, 0, &dd_rng[0]);" << std::endl;
            }
        }
        os << std::endl;
    }

    // Build map of extra global parameters for init kernel
    std::map<std::string, std::string> initKernelParameters;
    for(const auto &s : model.getLocalSynapseGroups()) {
        const auto *initSparseConnectivitySnippet = s.second.getConnectivityInitialiser().getSnippet();
        updateExtraGlobalParams(s.first, "", initSparseConnectivitySnippet->getExtraGlobalParams(), initKernelParameters,
                                {initSparseConnectivitySnippet->getRowBuildCode()});
    }
    
    // init kernel header
    os << "extern \"C\" __global__ void " << KernelNames[KernelInitialize] << "(";
    for(const auto &p : initKernelParameters) {
        os << p.second << " " << p.first << ", ";
    }
    os << "unsigned long long deviceRNGSeed)";

    // initialization kernel code
    size_t idInitStart = 0;
    {
        Substitutions kernelSubs(cudaFunctions, model.getPrecision());

        // common variables for all cases
        CodeStream::Scope b(os);

        os << "const unsigned int id = " << m_KernelBlockSizes[KernelInitialize] << " * blockIdx.x + threadIdx.x;" << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Remote neuron groups" << std::endl;
        genParallelGroup<NeuronGroupInternal>(os, kernelSubs, model.getRemoteNeuronGroups(), idInitStart,
            [this](const NeuronGroupInternal &ng){ return padSize(ng.getNumNeurons(), m_KernelBlockSizes[KernelInitialize]); },
            [this](const NeuronGroupInternal &ng){ return ng.hasOutputToHost(getLocalHostID()); },
            [this, remoteNGHandler](CodeStream &os, const NeuronGroupInternal &ng, Substitutions &popSubs)
            {
                os << "// only do this for existing neurons" << std::endl;
                os << "if(" << popSubs["id"] << " < " << ng.getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);

                    remoteNGHandler(os, ng, popSubs);
                }
            });
        os << std::endl;
   
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Local neuron groups" << std::endl;
        genParallelGroup<NeuronGroupInternal>(os, kernelSubs, model.getLocalNeuronGroups(), idInitStart,
            [this](const NeuronGroupInternal &ng){ return padSize(ng.getNumNeurons(), m_KernelBlockSizes[KernelInitialize]); },
            [this](const NeuronGroupInternal &){ return true; },
            [this, &model, localNGHandler](CodeStream &os, const NeuronGroupInternal &ng, Substitutions &popSubs)
            {
                os << "// only do this for existing neurons" << std::endl;
                os << "if(" << popSubs["id"] << " < " << ng.getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);
                    // If this neuron is going to require a simulation RNG, initialise one using GLOBAL thread id for sequence
                    if(ng.isSimRNGRequired()) {
                        os << "curand_init(deviceRNGSeed, id, 0, &dd_rng" << ng.getName() << "[" << popSubs["id"] << "]);" << std::endl;
                    }

                    // If this neuron requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    // **NOTE** not LOCAL id
                    if(ng.isInitRNGRequired()) {
                        os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
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
        genParallelGroup<SynapseGroupInternal>(os, kernelSubs, model.getLocalSynapseGroups(), idInitStart,
            [this](const SynapseGroupInternal &sg){ return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), m_KernelBlockSizes[KernelInitialize]); },
            [](const SynapseGroupInternal &sg){ return (sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) && sg.isWUVarInitRequired(); },
            [sgDenseInitHandler](CodeStream &os, const SynapseGroupInternal &sg, Substitutions &popSubs)
            {
                os << "// only do this for existing postsynaptic neurons" << std::endl;
                os << "if(" << popSubs["id"] << " < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);
                    // If this post synapse requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    // **NOTE** not LOCAL id
                    if(sg.isWUInitRNGRequired()) {
                        os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
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
        genParallelGroup<SynapseGroupInternal>(os, kernelSubs, model.getLocalSynapseGroups(), idInitStart,
            [this](const SynapseGroupInternal &sg){ return padSize(sg.getSrcNeuronGroup()->getNumNeurons(), m_KernelBlockSizes[KernelInitialize]); },
            [](const SynapseGroupInternal &sg){ return sg.isSparseConnectivityInitRequired(); },
            [this, sgSparseConnectHandler](CodeStream &os, const SynapseGroupInternal &sg, Substitutions &popSubs)
            {
                const size_t numSrcNeurons = sg.getSrcNeuronGroup()->getNumNeurons();
                const size_t numTrgNeurons = sg.getTrgNeuronGroup()->getNumNeurons();
                // **HACK**
                //const size_t synapticMatrixRowStride = getSynapticMatrixRowStride(sg);
                const size_t synapticMatrixRowStride = sg.getMaxConnections();

                os << "// only do this for existing presynaptic neurons" << std::endl;
                os << "if(" << popSubs["id"] << " < " << numSrcNeurons << ")";
                {
                    CodeStream::Scope b(os);
                    popSubs.addVarSubstitution("id_pre", popSubs["id"]);
                    popSubs.addVarSubstitution("id_post_begin", "0");
                    popSubs.addVarSubstitution("id_thread", "0");
                    popSubs.addVarSubstitution("num_threads", "1");
                    popSubs.addVarSubstitution("num_post", std::to_string(numTrgNeurons));
                    
                    // If the synapse group has bitmask connectivity
                    if(sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // Calculate indices of bits at start and end of row
                        os << "// Calculate indices" << std::endl;
                        const size_t maxSynapses = numSrcNeurons * synapticMatrixRowStride;
                        if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                            os << "const uint64_t rowStartGID = " << popSubs["id"] << " * " << synapticMatrixRowStride << "ull;" << std::endl;
                        }
                        else {
                            os << "const unsigned int rowStartGID = " << popSubs["id"] << " * " << synapticMatrixRowStride << ";" << std::endl;
                        }

                        // Build function template to set correct bit in bitmask
                        popSubs.addFuncSubstitution("addSynapse", 1,
                                                    "atomicOr(&dd_gp" + sg.getName() + "[(rowStartGID + $(0)) / 32], 0x80000000 >> ((rowStartGID + $(0)) & 31))");
                    }
                    // Otherwise, if synapse group has ragged connectivity
                    else if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        const std::string rowLength = "dd_rowLength" + sg.getName() + "[" + popSubs["id"] + "]";
                        const std::string ind = "dd_ind" + sg.getName();

                        // Zero row length
                        os << rowLength << " = 0;" << std::endl;

                        // Build function template to increment row length and insert synapse into ind array
                        popSubs.addFuncSubstitution("addSynapse", 1,
                                                    ind + "[(" + popSubs["id"] + " * " + std::to_string(synapticMatrixRowStride) + ") + (" + rowLength + "++)] = $(0)");
                    }
                    else {
                        assert(false);
                    }

                    // If this connectivity requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    // **NOTE** not LOCAL id
                    if(::Utils::isRNGRequired(sg.getConnectivityInitialiser().getSnippet()->getRowBuildCode())) {
                        os << "curandStatePhilox4_32_10_t connectivityRNG = dd_rng[0];" << std::endl;
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
    if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
        [](const ModelSpec::SynapseGroupValueType &s) { return isSparseInitRequired(s.second); }))
    {
        os << "extern \"C\" __global__ void " << KernelNames[KernelInitializeSparse] << "()";
        {
            CodeStream::Scope b(os);

            // common variables for all cases
            Substitutions kernelSubs(cudaFunctions, model.getPrecision());

            os << "const unsigned int id = " << m_KernelBlockSizes[KernelInitializeSparse] << " * blockIdx.x + threadIdx.x;" << std::endl;

            // Shared memory array so row lengths don't have to be read by EVERY postsynaptic thread
            // **TODO** check actually required
            os << "__shared__ unsigned int shRowLength[" << m_KernelBlockSizes[KernelInitializeSparse] << "];" << std::endl;
            if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
                           [](const ModelSpec::SynapseGroupValueType &s) { return (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getSynapseDynamicsCode().empty(); }))
            {
                os << "__shared__ unsigned int shRowStart[" << m_KernelBlockSizes[KernelInitializeSparse] + 1 << "];" << std::endl;
            }

            // Initialise weight update variables for synapse groups with sparse connectivity
            genParallelGroup<SynapseGroupInternal>(os, kernelSubs, model.getLocalSynapseGroups(), idSparseInitStart,
                [this](const SynapseGroupInternal &sg){ return padSize(sg.getMaxConnections(), m_KernelBlockSizes[KernelInitializeSparse]); },
                [](const SynapseGroupInternal &sg){ return isSparseInitRequired(sg); },
                [this, &model, sgSparseInitHandler, numStaticInitThreads](CodeStream &os, const SynapseGroupInternal &sg, Substitutions &popSubs)
                {
                    // If this post synapse requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    // **NOTE** not LOCAL id
                    if(sg.isWUInitRNGRequired()) {
                        os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                        os << "skipahead_sequence((unsigned long long)" << numStaticInitThreads << " + id, &initRNG);" << std::endl;

                        // Add substitution for RNG
                        popSubs.addVarSubstitution("rng", "&initRNG");
                    }

                    os << "unsigned int idx = " << popSubs["id"] << ";" << std::endl;

                    // Calculate how many blocks rows need to be processed in (in order to store row lengths in shared memory)
                    const unsigned int numSrcNeurons = sg.getSrcNeuronGroup()->getNumNeurons();
                    const size_t numBlocks = ceilDivide(numSrcNeurons, m_KernelBlockSizes[KernelInitializeSparse]);

                    // Loop through blocks
                    os << "for(unsigned int r = 0; r < " << numBlocks << "; r++)";
                    {
                        CodeStream::Scope b(os);

                        // Calculate number of rows to process in this block
                        os << "const unsigned numRowsInBlock = (r == " << numBlocks - 1 << ")";
                        os << " ? " << ((numSrcNeurons - 1) % m_KernelBlockSizes[KernelInitializeSparse]) + 1;
                        os << " : " << m_KernelBlockSizes[KernelInitializeSparse] << ";" << std::endl;

                        // Use threads to copy block of sparse structure into shared memory
                        os << "__syncthreads();" << std::endl;
                        os << "if (threadIdx.x < numRowsInBlock)";
                        {
                            CodeStream::Scope b(os);
                            os << "shRowLength[threadIdx.x] = dd_rowLength" << sg.getName() << "[(r * " << m_KernelBlockSizes[KernelInitializeSparse] << ") + threadIdx.x];" << std::endl;
                        }

                        // If this synapse group has synapse dynamics
                        if(!sg.getWUModel()->getSynapseDynamicsCode().empty()) {
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
                                os << "if(" << popSubs["id"] << " == 0 && (r == " << numBlocks - 1 << "))";
                                {
                                    CodeStream::Scope b(os);
                                    os << "dd_synRemap" << sg.getName() << "[0] = shRowStart[numRowsInBlock];" << std::endl;
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
                                if(sg.isWUVarInitRequired()) {
                                    popSubs.addVarSubstitution("id_pre", "((r * " + std::to_string(m_KernelBlockSizes[KernelInitializeSparse]) + ") + i)");
                                    popSubs.addVarSubstitution("id_post", "dd_ind" + sg.getName() + "[idx]");
                                    sgSparseInitHandler(os, sg, popSubs);
                                }

                                // If postsynaptic learning is required
                                if(!sg.getWUModel()->getLearnPostCode().empty()) {
                                    CodeStream::Scope b(os);

                                    // Extract index of synapse's postsynaptic target
                                    os << "const unsigned int postIndex = dd_ind" << sg.getName() << "[idx];" << std::endl;

                                    // Atomically increment length of column of connectivity associated with this target
                                    // **NOTE** this returns previous length i.e. where to insert new entry
                                    os << "const unsigned int colLocation = atomicAdd(&dd_colLength" << sg.getName() << "[postIndex], 1);" << std::endl;

                                    // From this calculate index into column-major matrix
                                    os << "const unsigned int colMajorIndex = (postIndex * " << sg.getMaxSourceConnections() << ") + colLocation;" << std::endl;

                                    // Add remapping entry at this location poining back to row-major index
                                    os << "dd_remap" << sg.getName() << "[colMajorIndex] = idx;" << std::endl;
                                }

                                // If synapse dynamics are required, copy idx into syn remap structure
                                if(!sg.getWUModel()->getSynapseDynamicsCode().empty()) {
                                    CodeStream::Scope b(os);
                                    os << "dd_synRemap" << sg.getName() << "[shRowStart[i] + " + popSubs["id"] + " + 1] = idx;" << std::endl;
                                }
                            }

                            // If matrix is ragged, advance index to next row by adding stride
                            // **HACK**
                            os << "idx += " << sg.getMaxConnections()/*getSynapticMatrixRowStride(sg)*/ << ";" << std::endl;
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
        if(isGlobalRNGRequired(model)) {
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

        for(const auto &s : model.getLocalSynapseGroups()) {
            // If this synapse population has BITMASK connectivity and is intialised on device, insert a call to cudaMemset to zero the whole bitmask
            if(s.second.isSparseConnectivityInitRequired() && s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                // **HACK**
                const size_t gpSize = ceilDivide((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getMaxConnections()/*getSynapticMatrixRowStride(s.second)*/, 32);
                os << "CHECK_CUDA_ERRORS(cudaMemset(d_gp" << s.first << ", 0, " << gpSize << " * sizeof(uint32_t)));" << std::endl;
            }
            // Otherwise, if this synapse population has RAGGED connectivity and has postsynaptic learning, insert a call to cudaMemset to zero column lengths
            else if((s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getLearnPostCode().empty()) {
                os << "CHECK_CUDA_ERRORS(cudaMemset(d_colLength" << s.first << ", 0, " << s.second.getTrgNeuronGroup()->getNumNeurons() << " * sizeof(unsigned int)));" << std::endl;
            }
        }

        // If there are any initialisation threads
        if(idInitStart > 0) {
            CodeStream::Scope b(os);
            {
                Timer t(os, "init", model.isTimingEnabled(), true);

                genKernelDimensions(os, KernelInitialize, idInitStart);
                os << KernelNames[KernelInitialize] << "<<<grid, threads>>>(";
                for(const auto &p : initKernelParameters) {
                    gennExtraGlobalParamPass(os, p);
                }
                os << "deviceRNGSeed);" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaPeekAtLastError());" << std::endl;
            }
        }
    }
    os << std::endl;
    os << "void initializeSparse()";
    {
        CodeStream::Scope b(os);

        // Copy all uninitialised state variables to device
        os << "copyStateToDevice(true);" << std::endl;
        os << "copyConnectivityToDevice(true);" << std::endl << std::endl;

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
size_t Backend::getSynapticMatrixRowStride(const SynapseGroupMerged &sgMerged, const SynapseGroupInternal &sg) const
{
    return getPresynapticUpdateStrategy(sgMerged)->getSynapticMatrixRowStride(sg);
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsPreamble(CodeStream &os, const ModelSpecInternal &) const
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
void Backend::genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecInternal &) const
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
void Backend::genRunnerPreamble(CodeStream &os, const ModelSpecInternal &model) const
{
#ifdef _WIN32
    // **YUCK** on Windows, disable "function assumed not to throw an exception but does" warning
    // Setting /Ehs SHOULD solve this but CUDA rules don't give this option and it's not clear it gets through to the compiler anyway
    os << "#pragma warning(disable: 4297)" << std::endl;
#endif

    // **TODO** move these into a header file shipped with GeNN and copied into generated code along with non-uniform RNGs
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper function for allocating memory blocks on the GPU device" << std::endl;
    os << std::endl;
    os << "template<class T>" << std::endl;
    os << "void deviceMemAllocate(T* hostPtr, const T &devSymbol, size_t size)";
    {
        CodeStream::Scope b(os);
        os << "void *devptr;" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaMalloc(hostPtr, size));" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devptr, devSymbol));" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostPtr, sizeof(void*), cudaMemcpyHostToDevice));" << std::endl;
    }
    os << std::endl;

    // If the model requires zero-copy
    if(model.zeroCopyInUse()) {
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Helper function for getting the device pointer corresponding to a zero-copied host pointer and assigning it to a symbol" << std::endl;
        os << std::endl;
        os << "template<class T>" << std::endl;
        os << "void deviceZeroCopy(T hostPtr, const T *devPtr, const T &devSymbol)";
        {
            CodeStream::Scope b(os);
            os << "CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void **)devPtr, (void*)hostPtr, 0));" << std::endl;
            os << "void *devSymbolPtr;" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devSymbolPtr, devSymbol));" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(devSymbolPtr, devPtr, sizeof(void*), cudaMemcpyHostToDevice));" << std::endl;
        }
        os << std::endl;
    }

    os << "template<class T>" << std::endl;
    os << "T *getSymbolAddress(const T &devSymbol)";
    {
        CodeStream::Scope b(os);
        os << "void *devPtr;" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, devSymbol));" << std::endl;
        os << "return reinterpret_cast<T*>(devPtr);" << std::endl;
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream &os, const ModelSpecInternal &model) const
{
    // Get chosen device's PCI bus ID
    char pciBusID[32];
    CHECK_CUDA_ERRORS(cudaDeviceGetPCIBusId(pciBusID, 32, m_ChosenDeviceID));

    // If the model requires zero-copy
    if(model.zeroCopyInUse()) {
        // If device doesn't support mapping host memory error
        if(!getChosenCUDADevice().canMapHostMemory) {
            throw std::runtime_error("Device does not support mapping CPU host memory!");
        }

        // set appropriate device flags
        os << "CHECK_CUDA_ERRORS(cudaSetDeviceFlags(cudaDeviceMapHost));" << std::endl;
        os << std::endl;
    }
    
    // Write code to get device by PCI bus ID
    // **NOTE** this is required because device IDs are not guaranteed to remain the same and we want the code to be run on the same GPU it was optimise for
    os << "int deviceID;" << std::endl;
    os << "CHECK_CUDA_ERRORS(cudaDeviceGetByPCIBusId(&deviceID, \"" << pciBusID << "\"));" << std::endl;
    os << "CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));" << std::endl;
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecInternal &model) const
{
    // Synchronise if zero-copy is in use
    if(model.zeroCopyInUse()) {
        os << "CHECK_CUDA_ERRORS(cudaDeviceSynchronize());" << std::endl;
    }

    // If timing is enabled, synchronise last event
    if(model.isTimingEnabled()) {
        os << "CHECK_CUDA_ERRORS(cudaEventSynchronize(neuronUpdateStop));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const
{
    const bool deviceType = isDeviceType(type);

    if(loc & VarLocation::HOST) {
        if(deviceType) {
            throw std::runtime_error("Variable '" + name + "' is of device-only type '" + type + "' but is located on the host");
        }

        definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
    }
    if(loc & VarLocation::DEVICE) {
        // If the type is a pointer type we need a host and a device pointer
        if(::Utils::isTypePointer(type)) {
            // Write host definition to internal definitions stream if type is device only
            CodeStream &d = deviceType ? definitionsInternal : definitions;
            d << "EXPORT_VAR " << type << " d_" << name << ";" << std::endl;

            definitionsInternal << "EXPORT_VAR __device__ " << type << " dd_" << name << ";" << std::endl;
        }
        // Otherwise we just need a device variable, made volatile for safety
        else {
            definitionsInternal << "EXPORT_VAR __device__ volatile " << type << " dd_" << name << ";" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    if(loc & VarLocation::HOST) {
        os << type << " " << name << ";" << std::endl;
    }
    if(loc & VarLocation::DEVICE) {
        // If the type is a pointer type we need a host and a device pointer
        if(::Utils::isTypePointer(type)) {
            os << type << " d_" << name << ";" << std::endl;
            os << "__device__ " << type << " dd_" << name << ";" << std::endl;
        }
        // Otherwise we just need a device variable, made volatile for safety
        else {
            os << "__device__ volatile " << type << " dd_" << name << ";" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
MemAlloc Backend::genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const
{
    auto allocation = MemAlloc::zero();

    if(loc & VarLocation::HOST) {
        const char *flags = (loc & VarLocation::ZERO_COPY) ? "cudaHostAllocMapped" : "cudaHostAllocPortable";
        os << "CHECK_CUDA_ERRORS(cudaHostAlloc(&" << name << ", " << count << " * sizeof(" << type << "), " << flags << "));" << std::endl;
        allocation += MemAlloc::host(count * getSize(type));
    }

    // If variable is present on device at all
    if(loc & VarLocation::DEVICE) {
        // Insert call to correct helper depending on whether variable should be allocated in zero-copy mode or not
        if(loc & VarLocation::ZERO_COPY) {
            os << "deviceZeroCopy(" << name << ", &d_" << name << ", dd_" << name << ");" << std::endl;
            allocation += MemAlloc::zeroCopy(count * getSize(type));
        }
        else {
            os << "deviceMemAllocate(&d_" << name << ", dd_" << name << ", " << count << " * sizeof(" << type << "));" << std::endl;
            allocation += MemAlloc::device(count * getSize(type));
        }
    }

    return allocation;
}
//--------------------------------------------------------------------------
void Backend::genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const
{
    // **NOTE** because we pinned the variable we need to free it with cudaFreeHost rather than use the host code generator
    if(loc & VarLocation::HOST) {
        os << "CHECK_CUDA_ERRORS(cudaFreeHost(" << name << "));" << std::endl;
    }

    // If this variable wasn't allocated in zero-copy mode, free it
    if(loc & VarLocation::DEVICE) {
        os << "CHECK_CUDA_ERRORS(cudaFree(d_" << name << "));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamDefinition(CodeStream &definitions, const std::string &type, const std::string &name, VarLocation loc) const
{
    if(loc & VarLocation::HOST) {
        definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
    }
    if(loc & VarLocation::DEVICE && ::Utils::isTypePointer(type)) {
        definitions << "EXPORT_VAR " << type << " d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    if(loc & VarLocation::HOST) {
        os << type << " " << name << ";" << std::endl;
    }
    if(loc & VarLocation::DEVICE && ::Utils::isTypePointer(type)) {
        os << type << " d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
    // Get underlying type
    // **NOTE** could use std::remove_pointer but it seems unnecessarily elaborate
    const std::string underlyingType = ::Utils::getUnderlyingType(type);

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
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const
{
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
void Backend::genPopVariableInit(CodeStream &os, VarLocation, const Substitutions &kernelSubs, Handler handler) const
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
void Backend::genVariableInit(CodeStream &os, VarLocation, size_t, const std::string &countVarName,
                              const Substitutions &kernelSubs, Handler handler) const
{
    // Variable should already be provided via parallelism
    assert(kernelSubs.hasVarSubstitution(countVarName));

    Substitutions varSubs(&kernelSubs);
    handler(os, varSubs);
}
//--------------------------------------------------------------------------
void Backend::genSynapseVariableRowInit(CodeStream &os, VarLocation, const SynapseGroupInternal &sg,
                                        const Substitutions &kernelSubs, Handler handler) const
{
    // Pre and postsynaptic ID should already be provided via parallelism
    assert(kernelSubs.hasVarSubstitution("id_pre"));
    assert(kernelSubs.hasVarSubstitution("id_post"));

    // **HACK**
    Substitutions varSubs(&kernelSubs);
    varSubs.addVarSubstitution("id_syn", "(" + kernelSubs["id_pre"] + " * " + std::to_string(sg.getMaxConnections()/*getSynapticMatrixRowStride(sg)*/) + ") + " + kernelSubs["id"]);

    handler(os, varSubs);
}
//--------------------------------------------------------------------------
void Backend::genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const
{
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
    if(!(loc & VarLocation::ZERO_COPY)) {
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << name;
        os << ", d_"  << name;
        os << ", " << count << " * sizeof(" << type << "), cudaMemcpyDeviceToHost));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, const std::string &name, VarLocation loc) const
{
    // If this variable requires queuing and isn't zero-copy
    if(ng.isVarQueueRequired(name) && ng.isDelayRequired() && !(loc & VarLocation::ZERO_COPY)) {
        // Generate memcpy to copy only current timestep's data
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
        os << ", " << name << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
        os << ", " << ng.getNumNeurons() << " * sizeof(" << type << "), cudaMemcpyHostToDevice));" << std::endl;
    }
    // Otherwise, generate standard push
    else {
        genVariablePush(os, type, name + ng.getName(), loc, false, ng.getNumNeurons());
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePull(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, const std::string &name, VarLocation loc) const
{
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
MemAlloc Backend::genGlobalRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free) const
{
    // Create a single Philox4_32_10 RNG
    genVariableDefinition(definitions, definitionsInternal, "curandStatePhilox4_32_10_t*", "rng", VarLocation::DEVICE);
    genVariableImplementation(runner, "curandStatePhilox4_32_10_t*", "rng", VarLocation::DEVICE);
    genVariableFree(free, "rng", VarLocation::DEVICE);
    return genVariableAllocation(allocations, "curandStatePhilox4_32_10_t", "rng", VarLocation::DEVICE, 1);
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
bool Backend::isGlobalRNGRequired(const ModelSpecInternal &model) const
{
    // If any neuron groups require  RNG for initialisation, return true
    // **NOTE** this takes postsynaptic model initialisation into account
    if(std::any_of(model.getLocalNeuronGroups().cbegin(), model.getLocalNeuronGroups().cend(),
        [](const ModelSpec::NeuronGroupValueType &n)
        {
            return n.second.isInitRNGRequired();
        }))
    {
        return true;
    }

    // If any synapse groups require an RNG for weight update model initialisation or procedural connectivity, return true
    if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
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
    nvccFlags += " -std=c++11 --compiler-options '-fPIC'";
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
size_t Backend::getNumInitialisationRNGStreams(const ModelSpecMerged &model) const
{
    // Start by counting remote neuron groups
    size_t numInitThreads = std::accumulate(
        model.getModel().getRemoteNeuronGroups().cbegin(), model.getModel().getRemoteNeuronGroups().cend(), size_t{0},
        [this](size_t acc, const ModelSpec::NeuronGroupValueType &n)
        {
            if (n.second.hasOutputToHost(getLocalHostID())) {
                return acc + padSize(n.second.getNumNeurons(), getKernelBlockSize(Kernel::KernelInitialize));
            }
            else {
                return acc;
            }
        });

    // Then local neuron groups
    numInitThreads = std::accumulate(
        model.getModel().getLocalNeuronGroups().cbegin(), model.getModel().getLocalNeuronGroups().cend(), numInitThreads,
        [this](size_t acc, const ModelSpec::NeuronGroupValueType &n)
        {
            return acc + padSize(n.second.getNumNeurons(), getKernelBlockSize(Kernel::KernelInitialize));
        });


    // Then synapse neuron groups
    numInitThreads = std::accumulate(
        model.getModel().getLocalSynapseGroups().cbegin(), model.getModel().getLocalSynapseGroups().cend(), numInitThreads,
        [this](size_t acc, const ModelSpec::SynapseGroupValueType &s)
        {
            const size_t initBlockSize = getKernelBlockSize(Kernel::KernelInitialize);
            const size_t initSparseBlockSize = getKernelBlockSize(Kernel::KernelInitializeSparse);

            // Add number of threads required for variable initialisation of dense matrices
            if ((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && s.second.isWUVarInitRequired()) {
                acc += padSize(s.second.getTrgNeuronGroup()->getNumNeurons(), initBlockSize);
            }

            // Any for sparse connectivity initialisation
            if (s.second.isSparseConnectivityInitRequired()) {
                acc += padSize(s.second.getSrcNeuronGroup()->getNumNeurons(), initBlockSize);
            }

            // And, finally, any require for variable initialisation of sparse matrices
            if (isSparseInitRequired(s.second)) {
                acc += padSize(s.second.getMaxConnections(), initSparseBlockSize);
            }

            return acc;

        });

    return numInitThreads;
}
//--------------------------------------------------------------------------
size_t Backend::getNumPresynapticUpdateThreads(const SynapseGroupMerged &sgMerged, const SynapseGroupInternal &sg,
                                               const cudaDeviceProp &deviceProps, const Preferences &preferences)
{
     return getPresynapticUpdateStrategy(sgMerged, deviceProps, preferences)->getNumThreads(sg);
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
const PresynapticUpdateStrategy::Base *Backend::getPresynapticUpdateStrategy(const SynapseGroupMerged &sg,
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

    throw std::runtime_error("Unable to find a suitable presynaptic update strategy for merged synapse group '" + std::to_string(sg.getIndex()) + "'");
    return nullptr;
}
}   // namespace CUDA
}   // namespace CodeGenerator

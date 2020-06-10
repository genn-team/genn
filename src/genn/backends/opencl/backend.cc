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
//! TO BE IMPLEMENTED - Use OpenCL functions - clRNG
const std::vector<Substitutions::FunctionTemplate> openclFunctions = {
    {"gennrand_uniform", 0, "uniform_double($(rng))", "uniform_clrngLfsr113($(rng))"},
    {"gennrand_normal", 0, "normal_double($(rng))", "normal($(rng))"},
    {"gennrand_exponential", 0, "exponentialDistDouble($(rng))", "exponentialDistFloat($(rng))"},
    {"gennrand_log_normal", 2, "log_normal_double($(rng), $(0), $(1))", "log_normal_float($(rng), $(0), $(1))"},
    {"gennrand_gamma", 1, "gammaDistDouble($(rng), $(0))", "gammaDistFloat($(rng), $(0))"}
};

//--------------------------------------------------------------------------
// Timer
//--------------------------------------------------------------------------
class Timer {
public:
    //! TO BE REVIEWED
    Timer(CodeStream& codeStream, const std::string& name, bool timingEnabled, bool synchroniseOnStop = false)
        : m_CodeStream(codeStream), m_Name(name), m_TimingEnabled(timingEnabled), m_SynchroniseOnStop(synchroniseOnStop)
    {

    }
private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    CodeStream& m_CodeStream;
    const std::string m_Name;
    const bool m_TimingEnabled;
    const bool m_SynchroniseOnStop;
};

//-----------------------------------------------------------------------
bool isSparseInitRequired(const SynapseGroupInternal& sg)
{
    return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE)
        && (sg.isWUVarInitRequired() || !sg.getWUModel()->getLearnPostCode().empty() || !sg.getWUModel()->getSynapseDynamicsCode().empty()));
}
//--------------------------------------------------------------------------
template<typename T>
void genMergedGroupKernelParams(CodeStream &os, const std::vector<T> &groups, const std::string &name, bool includeFinalComma = false)
{
    // Loop through groups and add pointer
    // **NOTE** __constant is more of a hint in OpenCL - hopefully will work out ok..
    for(size_t i = 0; i < groups.size(); i++) {
        os << "__constant Merged" << name << "Group" << i << " *d_merged" << name << "Group" << i;
        if(includeFinalComma || i != (groups.size() - 1)) {
            os << ", ";
        }
    }
}
//--------------------------------------------------------------------------
template<typename T>
void setMergedGroupKernelParams(CodeStream &os, const std::string &kernelName, const std::vector<T> &groups, const std::string &name)
{
    // Loop through groups and set as kernel arguments
    for(size_t i = 0; i < groups.size(); i++) {
        os << "CHECK_OPENCL_ERRORS(" << kernelName << ".setArg(" << i << ", d_merged" << name << "Group" << i << "));" << std::endl;
    }
}
//-----------------------------------------------------------------------
void genGroupStartIDs(CodeStream &, size_t &, size_t)
{
}
//-----------------------------------------------------------------------
template<typename T, typename G, typename ...Args>
void genGroupStartIDs(CodeStream &os, size_t &idStart, size_t workgroupSize,
                      const std::vector<T> &mergedGroups, const std::string &groupStartPrefix, G getNumThreads,
                      Args... args)
{
    // Loop through merged groups
    for(const auto &m : mergedGroups) {
        // Declare array of starting thread indices for each neuron group
        os << "__constant unsigned int d_merged" << groupStartPrefix << "GroupStartID" << m.getIndex() << "[] = {";
        for(const auto &ng : m.getGroups()) {
            os << idStart << ", ";
            idStart += padSize(getNumThreads(ng.get()), workgroupSize);
        }
        os << "};" << std::endl;
    }

    // Generate any remaining groups
    genGroupStartIDs(os, idStart, workgroupSize, args...);
}
//-----------------------------------------------------------------------
template<typename ...Args>
void genMergedKernelDataStructures(CodeStream &os, size_t workgroupSize, Args... args)
{
    // Generate group start id arrays
    size_t idStart = 0;
    genGroupStartIDs(os, std::ref(idStart), workgroupSize, args...);
}
}

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::Backend
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace OpenCL
{
const char* Backend::KernelNames[KernelMax] = {
    "updateNeuronsKernel",
    "updatePresynapticKernel",
    "updatePostsynapticKernel",
    "updateSynapseDynamicsKernel",
    "initializeKernel",
    "initializeSparseKernel",
    "preNeuronResetKernel",
    "preSynapseResetKernel" };
//--------------------------------------------------------------------------
const char* Backend::ProgramNames[ProgramMax] = {
    "initProgram",
    "updateNeuronsProgram",
    "updateSynapsesProgram" };
//--------------------------------------------------------------------------
std::vector<PresynapticUpdateStrategy::Base*> Backend::s_PresynapticUpdateStrategies = {
    new PresynapticUpdateStrategy::PreSpan,
    new PresynapticUpdateStrategy::PostSpan,
};
//--------------------------------------------------------------------------
Backend::Backend(const KernelWorkGroupSize& kernelWorkGroupSizes, const Preferences& preferences,
                 const std::string& scalarType, unsigned int platformIndex, unsigned int deviceIndex)
:   BackendBase(scalarType), m_KernelWorkGroupSizes(kernelWorkGroupSizes), m_Preferences(preferences), 
    m_ChosenPlatformIndex(platformIndex), m_ChosenDeviceIndex(deviceIndex)
{
    assert(!m_Preferences.automaticCopy);

    // Get platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(m_ChosenPlatformIndex < platforms.size());

    // Show platform name
    LOGI << "Using OpenCL platform:" << platforms[m_ChosenPlatformIndex].getInfo<CL_PLATFORM_NAME>();

    // Get platform devices
    std::vector<cl::Device> platformDevices;
    platforms[m_ChosenPlatformIndex].getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
    assert(m_ChosenDeviceIndex < platformDevices.size());

    // Select device
    m_ChosenDevice = platformDevices[m_ChosenDeviceIndex];
    
    // Show device name
    LOGI << "Using OpenCL device:" << m_ChosenDevice.getInfo<CL_DEVICE_NAME>();
}
//--------------------------------------------------------------------------
void Backend::genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged,
                              NeuronGroupSimHandler simHandler, NeuronUpdateGroupMergedHandler wuVarUpdateHandler,
                              HostHandler pushEGPHandler) const
{
    // Generate reset kernel to be run before the neuron kernel
    const ModelSpecInternal &model = modelMerged.getModel();

    //! KernelPreNeuronReset START
    size_t idPreNeuronReset = 0;

    // Creating the kernel body separately so it can be split into multiple string literals
    std::stringstream neuronUpdateKernelsStream;
    CodeStream neuronUpdateKernels(neuronUpdateKernelsStream);

    // Include definitions
    neuronUpdateKernels << "typedef " << model.getPrecision() << " scalar;" << std::endl;
    neuronUpdateKernels << "#define DT " << std::to_string(model.getDT());
    if(model.getTimePrecision() == "float") {
        neuronUpdateKernels << "f";
    }
    neuronUpdateKernels << std::endl << std::endl;

    // Generate support code
    modelMerged.genNeuronUpdateGroupSupportCode(neuronUpdateKernels);
    neuronUpdateKernels << std::endl << std::endl;
    
    // Generate merged data structures
    genMergedKernelDataStructures(neuronUpdateKernels, m_KernelWorkGroupSizes[KernelNeuronUpdate],
                                  modelMerged.getMergedNeuronUpdateGroups(), "NeuronUpdate",
                                  [](const NeuronGroupInternal &ng) { return ng.getNumNeurons(); });
    neuronUpdateKernels << std::endl;

    neuronUpdateKernels << "__kernel void " << KernelNames[KernelPreNeuronReset] << "(";
    genMergedGroupKernelParams(neuronUpdateKernels, modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), "NeuronSpikeQueueUpdate");
    neuronUpdateKernels << ")";
    {
        CodeStream::Scope b(neuronUpdateKernels);

        neuronUpdateKernels << "const unsigned int id = get_global_id(0);" << std::endl;

        // Loop through local neuron groups
        for(const auto &n : modelMerged.getMergedNeuronSpikeQueueUpdateGroups()) {
            neuronUpdateKernels << "// merged" << n.getIndex() << std::endl;
            if(idPreNeuronReset == 0) {
                neuronUpdateKernels << "if(id < " << n.getGroups().size() << ")";
            }
            else {
                neuronUpdateKernels << "if(id >= " << idPreNeuronReset << " && id < " << idPreNeuronReset + n.getGroups().size() << ")";
            }
            {
                CodeStream::Scope b(neuronUpdateKernels);

                // Use this to get reference to merged group structure
                neuronUpdateKernels << "const MergedNeuronSpikeQueueUpdateGroup" << n.getIndex() << "*group = &d_mergedNeuronSpikeQueueUpdateGroup" << n.getIndex() << "[id - " << idPreNeuronReset << "]; " << std::endl;

                if(n.getArchetype().isDelayRequired()) { // with delay
                    neuronUpdateKernels << "*group->spkQuePtr  = (*group->spkQuePtr + 1) % " << n.getArchetype().getNumDelaySlots() << ";" << std::endl;
                }
                n.genMergedGroupSpikeCountReset(neuronUpdateKernels);
            }
            idPreNeuronReset += n.getGroups().size();
        }
    }
    //! KernelPreNeuronReset END
    size_t idStart = 0;

    //! KernelNeuronUpdate BODY START
    neuronUpdateKernels << "__kernel void " << KernelNames[KernelNeuronUpdate] << "(";
    genMergedGroupKernelParams(neuronUpdateKernels, modelMerged.getMergedNeuronUpdateGroups(), "NeuronUpdate", true);
    neuronUpdateKernels << model.getTimePrecision() << " t)";
    {
        CodeStream::Scope b(neuronUpdateKernels);
        neuronUpdateKernels << "const size_t localId = get_local_id(0);" << std::endl;
        neuronUpdateKernels << "const unsigned int id = get_global_id(0);" << std::endl;

        Substitutions kernelSubs(openclFunctions, model.getPrecision());
        kernelSubs.addVarSubstitution("t", "t");

        // If any neuron groups emit spike events
        if(std::any_of(modelMerged.getMergedNeuronUpdateGroups().cbegin(), modelMerged.getMergedNeuronUpdateGroups().cend(),
                       [](const NeuronUpdateGroupMerged &n) { return n.getArchetype().isSpikeEventRequired(); }))
        {
            neuronUpdateKernels << "volatile __local unsigned int shSpkEvnt[" << m_KernelWorkGroupSizes[KernelNeuronUpdate] << "];" << std::endl;
            neuronUpdateKernels << "volatile __local unsigned int shPosSpkEvnt;" << std::endl;
            neuronUpdateKernels << "volatile __local unsigned int shSpkEvntCount;" << std::endl;
            neuronUpdateKernels << std::endl;
            neuronUpdateKernels << "if (localId == 1)";
            {
                CodeStream::Scope b(neuronUpdateKernels);
                neuronUpdateKernels << "shSpkEvntCount = 0;" << std::endl;
            }
            neuronUpdateKernels << std::endl;
        }

        // If any neuron groups emit true spikes
        if(std::any_of(modelMerged.getMergedNeuronUpdateGroups().cbegin(), modelMerged.getMergedNeuronUpdateGroups().cend(),
                       [](const NeuronUpdateGroupMerged &n) { return !n.getArchetype().getNeuronModel()->getThresholdConditionCode().empty(); }))
        {
            neuronUpdateKernels << "volatile __local unsigned int shSpk[" << m_KernelWorkGroupSizes[KernelNeuronUpdate] << "];" << std::endl;
            neuronUpdateKernels << "volatile __local unsigned int shPosSpk;" << std::endl;
            neuronUpdateKernels << "volatile __local unsigned int shSpkCount;" << std::endl;
            neuronUpdateKernels << "if (localId == 0)";
            {
                CodeStream::Scope b(neuronUpdateKernels);
                neuronUpdateKernels << "shSpkCount = 0;" << std::endl;
            }
            neuronUpdateKernels << std::endl;
        }

        neuronUpdateKernels << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        // Parallelise over neuron groups
        genParallelGroup<NeuronUpdateGroupMerged>(neuronUpdateKernels, kernelSubs, modelMerged.getMergedNeuronUpdateGroups(), "NeuronUpdate", idStart,
            [this](const NeuronGroupInternal &ng) { return padSize(ng.getNumNeurons(), getKernelBlockSize(KernelNeuronUpdate)); },
            [&model, simHandler, wuVarUpdateHandler, this](CodeStream &os, const NeuronUpdateGroupMerged &ng, Substitutions &popSubs)
            {
                // If axonal delays are required
                if(ng.getArchetype().isDelayRequired()) {
                    // We should READ from delay slot before spkQuePtr
                    os << "const unsigned int readDelayOffset = " << ng.getPrevQueueOffset() << ";" << std::endl;

                    // And we should WRITE to delay slot pointed to be spkQuePtr
                    os << "const unsigned int writeDelayOffset = " << ng.getCurrentQueueOffset() << ";" << std::endl;
                }
                os << std::endl;


                // If this neuron group requires a simulation RNG, substitute in this neuron group's RNG
                //! TO BE IMPLEMENTED - Not using rng at this point - 2020/03/08
                if (ng.getArchetype().isSimRNGRequired()) {
                    popSubs.addVarSubstitution("rng", "&group->rng[" + popSubs["id"] + "]");
                }

                // Call handler to generate generic neuron code
                os << "if(" << popSubs["id"] << " < group->numNeurons)";
                {
                    CodeStream::Scope b(os);
                    simHandler(os, ng, popSubs,
                        // Emit true spikes
                        [this](CodeStream& neuronUpdateKernelsBody, const NeuronUpdateGroupMerged &, Substitutions& subs)
                        {
                            genEmitSpike(neuronUpdateKernelsBody, subs, "");
                        },
                        // Emit spike-like events
                        [this](CodeStream& neuronUpdateKernelsBody, const NeuronUpdateGroupMerged &, Substitutions& subs)
                        {
                            genEmitSpike(neuronUpdateKernelsBody, subs, "Evnt");
                        });
                }

                os << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

                if (ng.getArchetype().isSpikeEventRequired()) {
                    os << "if (localId == 1)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (shSpkEvntCount > 0)";
                        {
                            CodeStream::Scope b(os);
                            os << "shPosSpkEvnt = atomic_add(&group->spkCntEvnt";
                            if (ng.getArchetype().isDelayRequired()) {
                                os << "[*group->spkQuePtr], shSpkEvntCount);" << std::endl;
                            }
                            else {
                                os << "[0], shSpkEvntCount);" << std::endl;
                            }
                        }
                    } // end if (localId == 0)
                    os << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
                }

                if (!ng.getArchetype().getNeuronModel()->getThresholdConditionCode().empty()) {
                    os << "if (localId == 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (shSpkCount > 0)";
                        {
                            CodeStream::Scope b(os);
                            os << "shPosSpk = atomic_add(&group->spkCnt";
                            if (ng.getArchetype().isDelayRequired() && ng.getArchetype().isTrueSpikeRequired()) {
                                os << "[*group->spkQuePtr], shSpkCount);" << std::endl;
                            }
                            else {
                                os << "[0], shSpkCount);" << std::endl;
                            }
                        }
                    } // end if (localId == 1)
                    os << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
                }

                const std::string queueOffset = ng.getArchetype().isDelayRequired() ? "writeDelayOffset + " : "";
                if (ng.getArchetype().isSpikeEventRequired()) {
                    os << "if (localId < shSpkEvntCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "group->spkEvnt[" << queueOffset << "shPosSpkEvnt + localId] = shSpkEvnt[localId];" << std::endl;
                    }
                }

                if (!ng.getArchetype().getNeuronModel()->getThresholdConditionCode().empty()) {
                    const std::string queueOffsetTrueSpk = ng.getArchetype().isTrueSpikeRequired() ? queueOffset : "";

                    os << "if (localId < shSpkCount)";
                    {
                        CodeStream::Scope b(os);

                        os << "const unsigned int n = shSpk[localId];" << std::endl;

                        // Create new substition stack and explicitly replace id with 'n' and perform WU var update
                        Substitutions wuSubs(&popSubs);
                        wuSubs.addVarSubstitution("id", "n", true);
                        wuVarUpdateHandler(os, ng, wuSubs);

                        os << "group->spk[" << queueOffsetTrueSpk << "shPosSpk + localId] = n;" << std::endl;
                        if (ng.getArchetype().isSpikeTimeRequired()) {
                            os << "group->spk[" << queueOffset << "n] = t;" << std::endl;
                        }
                    }
                }
            }
        );
    }
    //! KernelNeuronUpdate BODY END
    //! KernelNeuronUpdate END

    // Write out kernel source string literal
    os << "const char* " << ProgramNames[ProgramNeuronsUpdate] << "Src = ";
    divideKernelStreamInParts(os, neuronUpdateKernelsStream, 5000);
    os << ";" << std::endl;

    // Function for initializing the KernelNeuronUpdate kernels
    os << "// Initialize the neuronUpdate kernels" << std::endl;
    os << "void " << ProgramNames[ProgramNeuronsUpdate] << "Kernels()";
    {
        CodeStream::Scope b(os);

        // KernelPreNeuronReset initialization
        os << KernelNames[KernelPreNeuronReset] << " = cl::Kernel(" << ProgramNames[ProgramNeuronsUpdate] << ", \"" << KernelNames[KernelPreNeuronReset] << "\");" << std::endl;
        setMergedGroupKernelParams(os, KernelNames[KernelPreNeuronReset], modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), "NeuronSpikeQueueUpdate");
        
        os << std::endl;

        // KernelNeuronUpdate initialization
        os << KernelNames[KernelNeuronUpdate] << " = cl::Kernel(" << ProgramNames[ProgramNeuronsUpdate] << ", \"" << KernelNames[KernelNeuronUpdate] << "\");" << std::endl;
        setMergedGroupKernelParams(os, KernelNames[KernelNeuronUpdate], modelMerged.getMergedNeuronUpdateGroups(), "NeuronUpdate");
    }

    os << std::endl;

    os << "void updateNeurons(" << model.getTimePrecision() << " t)";
    {
        CodeStream::Scope b(os);
        if (idPreNeuronReset > 0) {
            CodeStream::Scope b(os);
            genKernelDimensions(os, KernelPreNeuronReset, idPreNeuronReset);
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelPreNeuronReset] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
            os << std::endl;
        }
        if (idStart > 0) {
            CodeStream::Scope b(os);
            os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelNeuronUpdate] << ".setArg(" << modelMerged.getMergedNeuronUpdateGroups().size() << ", t));" << std::endl;
            os << std::endl;
            genKernelDimensions(os, KernelNeuronUpdate, idStart);
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelNeuronUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genSynapseUpdate(CodeStream &os, const ModelSpecMerged &modelMerged,
                               PresynapticUpdateGroupMergedHandler wumThreshHandler, PresynapticUpdateGroupMergedHandler wumSimHandler,
                               PresynapticUpdateGroupMergedHandler wumEventHandler, PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler,
                               PostsynapticUpdateGroupMergedHandler postLearnHandler, SynapseDynamicsGroupMergedHandler synapseDynamicsHandler,
                               HostHandler pushEGPHandler) const
{
    // If any synapse groups require dendritic delay, a reset kernel is required to be run before the synapse kernel
    /*size_t idPreSynapseReset = 0;
    std::stringstream preSynapseResetKernelBodyStream;
    std::map<std::string, std::string> preSynapseResetKernelParams;

    bool hasPreSynapseResetKernel = std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
        [](const ModelSpec::SynapseGroupValueType& s) { return s.second.isDendriticDelayRequired(); });

    if (hasPreSynapseResetKernel)
    {
        CodeStream preSynapseResetKernelBody(preSynapseResetKernelBodyStream);

        preSynapseResetKernelBody << "const size_t localId = get_local_id(0);" << std::endl;
        preSynapseResetKernelBody << "const unsigned int id = get_global_id(0);" << std::endl;

        // Loop through neuron groups
        for (const auto& n : model.getLocalNeuronGroups()) {
            // Loop through incoming synaptic populations
            for (const auto& m : n.second.getMergedInSyn()) {
                const auto* sg = m.first;

                // If this kernel requires dendritic delay
                if (sg->isDendriticDelayRequired()) {
                    if (idPreSynapseReset > 0) {
                        preSynapseResetKernelBody << "else ";
                    }
                    preSynapseResetKernelBody << "if(id == " << (idPreSynapseReset++) << ")";
                    {
                        CodeStream::Scope b(preSynapseResetKernelBody);

                        preSynapseResetKernelBody << "denDelayPtr" << sg->getPSModelTargetName() << " = (denDelayPtr" << sg->getPSModelTargetName() << " + 1) % " << sg->getMaxDendriticDelayTimesteps() << ";" << std::endl;

                        preSynapseResetKernelParams.insert({ "denDelayPtr" + sg->getPSModelTargetName(), "volatile unsigned int" });
                    }
                }
            }
        }
    }

    // Add extra global parameters references by weight updates models to maps of kernel parameters
    std::map<std::string, std::string> presynapticUpdateKernelParams;
    std::map<std::string, std::string> postsynapticUpdateKernelParams;
    std::map<std::string, std::string> synapseDynamicsUpdateKernelParams;
    for (const auto& s : model.getLocalSynapseGroups()) {
        const auto* wum = s.second.getWUModel();
        updateSynapseGroupExtraGlobalParams(s.second, presynapticUpdateKernelParams,
            { wum->getSimCode(), wum->getEventCode(), wum->getEventThresholdConditionCode() });
        updateSynapseGroupExtraGlobalParams(s.second, postsynapticUpdateKernelParams, { wum->getLearnPostCode() });
        updateSynapseGroupExtraGlobalParams(s.second, synapseDynamicsUpdateKernelParams, { wum->getSynapseDynamicsCode() });
    }

    // If any synapse groups require spike-driven presynaptic updates
    size_t idPresynapticStart = 0;
    std::stringstream presynapticUpdateKernelBodyStream;

    bool hasPresynapticUpdateKernel = std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
        [](const ModelSpec::SynapseGroupValueType& s) { return (s.second.isSpikeEventRequired() || s.second.isTrueSpikeRequired()); });

    //! KernelPresynapticUpdate BODY START
    if (hasPresynapticUpdateKernel)
    {
        CodeStream presynapticUpdateKernelBody(presynapticUpdateKernelBodyStream);
        Substitutions kernelSubs(openclFunctions, model.getPrecision());
        kernelSubs.addVarSubstitution("t", "t");

        presynapticUpdateKernelBody << "const size_t localId = get_local_id(0);" << std::endl;
        presynapticUpdateKernelBody << "const unsigned int id = get_global_id(0);" << std::endl;

        // We need shLg if any synapse groups accumulate into shared memory
        if (std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [this](const ModelSpec::SynapseGroupValueType& s)
            {
                return this->getPresynapticUpdateStrategy(s.second)->shouldAccumulateInSharedMemory(s.second, *this);
            }))
        {
            presynapticUpdateKernelBody << "__local " << model.getPrecision() << " shLg[" << m_KernelWorkGroupSizes[KernelPresynapticUpdate] << "];" << std::endl;
        }

            // If any of these synapse groups also have sparse connectivity, allocate shared memory for row length
            if (std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
                [&model](const ModelSpec::SynapseGroupValueType& s)
                {
                    return (s.second.getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC
                        && (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE));
                }))
            {
                presynapticUpdateKernelBody << "__local unsigned int shRowLength[" << m_KernelWorkGroupSizes[KernelPresynapticUpdate] << "];" << std::endl;
            }

                if (std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
                    [&model](const ModelSpec::SynapseGroupValueType& s)
                    {
                        return (s.second.isTrueSpikeRequired() || !s.second.getWUModel()->getLearnPostCode().empty());
                    }))
                {
                    presynapticUpdateKernelBody << "__local unsigned int shSpk[" << m_KernelWorkGroupSizes[KernelPresynapticUpdate] << "];" << std::endl;
                }

                    if (std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
                        [](const ModelSpec::SynapseGroupValueType& s) { return (s.second.isSpikeEventRequired()); }))
                    {
                        presynapticUpdateKernelBody << "__local unsigned int shSpkEvnt[" << m_KernelWorkGroupSizes[KernelPresynapticUpdate] << "];" << std::endl;
                    }

                    // Parallelise over synapse groups
                    genParallelGroup<SynapseGroupInternal>(presynapticUpdateKernelBody, kernelSubs, model.getLocalSynapseGroups(), idPresynapticStart, presynapticUpdateKernelParams,
                        [this](const SynapseGroupInternal& sg) { return Utils::padSize(getNumPresynapticUpdateThreads(sg), m_KernelWorkGroupSizes[KernelPresynapticUpdate]); },
                        [](const SynapseGroupInternal& sg) { return (sg.isSpikeEventRequired() || sg.isTrueSpikeRequired()); },
                        [wumThreshHandler, wumSimHandler, wumEventHandler, &model, this, &presynapticUpdateKernelParams](CodeStream& presynapticUpdateKernelBody, const SynapseGroupInternal& sg, const Substitutions& popSubs)
                        {
                            // Get presynaptic update strategy to use for this synapse group
                            const auto* presynapticUpdateStrategy = getPresynapticUpdateStrategy(sg);
                            LOGD << "Using '" << typeid(*presynapticUpdateStrategy).name() << "' presynaptic update strategy for synapse group '" << sg.getName() << "'";

                            // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                            if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                                presynapticUpdateKernelBody << "const unsigned int preReadDelaySlot = " << sg.getPresynapticAxonalDelaySlot("") << ";" << std::endl;
                                presynapticUpdateKernelBody << "const unsigned int preReadDelayOffset = preReadDelaySlot * " << sg.getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;

                                presynapticUpdateKernelParams.insert({ "spkQuePtr" + sg.getSrcNeuronGroup()->getName(), "volatile unsigned int" });
                            }

                            // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                            if (sg.getTrgNeuronGroup()->isDelayRequired()) {
                                presynapticUpdateKernelBody << "const unsigned int postReadDelayOffset = " << sg.getPostsynapticBackPropDelaySlot("") << " * " << sg.getTrgNeuronGroup()->getNumNeurons() << ";" << std::endl;

                                presynapticUpdateKernelParams.insert({ "spkQuePtr" + sg.getTrgNeuronGroup()->getName(), "volatile unsigned int" });
                            }

                            // If we are going to accumulate postsynaptic input into a register, zero register value
                            if (presynapticUpdateStrategy->shouldAccumulateInRegister(sg, *this)) {
                                presynapticUpdateKernelBody << "// only do this for existing neurons" << std::endl;
                                presynapticUpdateKernelBody << model.getPrecision() << " linSyn = 0;" << std::endl;
                            }
                            // Otherwise, if we are going to accumulate into shared memory, zero entry in array for each target neuron
                            // **NOTE** is ok as number of target neurons <= synapseBlkSz
                            else if (presynapticUpdateStrategy->shouldAccumulateInSharedMemory(sg, *this)) {
                                presynapticUpdateKernelBody << "if(localId < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                                {
                                    CodeStream::Scope b(presynapticUpdateKernelBody);
                                    presynapticUpdateKernelBody << "shLg[localId] = 0;" << std::endl;
                                }
                                presynapticUpdateKernelBody << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
                            }

                            // If spike events should be processed
                            if (sg.isSpikeEventRequired()) {
                                CodeStream::Scope b(presynapticUpdateKernelBody);
                                presynapticUpdateStrategy->genCode(presynapticUpdateKernelBody, model, sg, popSubs, *this, false,
                                    wumThreshHandler, wumEventHandler, presynapticUpdateKernelParams);
                            }

                            // If true spikes should be processed
                            if (sg.isTrueSpikeRequired()) {
                                CodeStream::Scope b(presynapticUpdateKernelBody);
                                presynapticUpdateStrategy->genCode(presynapticUpdateKernelBody, model, sg, popSubs, *this, true,
                                    wumThreshHandler, wumSimHandler, presynapticUpdateKernelParams);
                            }

                            presynapticUpdateKernelBody << std::endl;

                            // If we have been accumulating into a register, write value back to global memory
                            if (presynapticUpdateStrategy->shouldAccumulateInRegister(sg, *this)) {
                                presynapticUpdateKernelBody << "// only do this for existing neurons" << std::endl;
                                presynapticUpdateKernelBody << "if (" << popSubs["id"] << " < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                                {
                                    CodeStream::Scope b(presynapticUpdateKernelBody);
                                    const std::string inSyn = "d_inSyn" + sg.getPSModelTargetName() + "[" + popSubs["id"] + "]";
                                    presynapticUpdateKernelParams.insert({ "d_inSyn" + sg.getPSModelTargetName(), "__global float*" });
                                    if (sg.isPSModelMerged()) {
                                        presynapticUpdateKernelBody << getFloatAtomicAdd(model.getPrecision()) << "(&" << inSyn << ", linSyn);" << std::endl;
                                    }
                                    else {
                                        presynapticUpdateKernelBody << inSyn << " += linSyn;" << std::endl;
                                    }
                                }
                            }
                            // Otherwise, if we have been accumulating into shared memory, write value back to global memory
                            // **NOTE** is ok as number of target neurons <= synapseBlkSz
                            else if (presynapticUpdateStrategy->shouldAccumulateInSharedMemory(sg, *this)) {
                                presynapticUpdateKernelBody << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
                                presynapticUpdateKernelBody << "if (localId < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                                {
                                    CodeStream::Scope b(presynapticUpdateKernelBody);
                                    const std::string inSyn = "d_inSyn" + sg.getPSModelTargetName() + "[localId]";
                                    presynapticUpdateKernelParams.insert({ "d_inSyn" + sg.getPSModelTargetName(), "__global float*" });

                                    if (sg.isPSModelMerged()) {
                                        presynapticUpdateKernelBody << getFloatAtomicAdd(model.getPrecision()) << "(&" << inSyn << ", shLg[localId]);" << std::endl;
                                    }
                                    else {
                                        presynapticUpdateKernelBody << inSyn << " += shLg[localId];" << std::endl;
                                    }
                                }
                            }
                        }
                    );
    }
    //! KernelPresynapticUpdate BODY END

    // If any synapse groups require postsynaptic learning
    size_t idPostsynapticStart = 0;
    std::stringstream postsynapticUpdateKernelBodyStream;

    bool hasPostsynapticUpdateKernel = std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
        [](const ModelSpec::SynapseGroupValueType& s) { return !s.second.getWUModel()->getLearnPostCode().empty(); });

    //! KernelPostsynapticUpdate BODY START
    if (hasPostsynapticUpdateKernel)
    {
        CodeStream postsynapticUpdateKernelBody(postsynapticUpdateKernelBodyStream);

        Substitutions kernelSubs(openclFunctions, model.getPrecision());
        kernelSubs.addVarSubstitution("t", "t");

        postsynapticUpdateKernelBody << "const size_t localId = get_local_id(0);" << std::endl;
        postsynapticUpdateKernelBody << "const unsigned int id = get_global_id(0);" << std::endl;
        postsynapticUpdateKernelBody << "__local unsigned int shSpk[" << m_KernelWorkGroupSizes[KernelPostsynapticUpdate] << "];" << std::endl;
        if (std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [&model](const ModelSpec::SynapseGroupValueType& s)
            {
                return ((s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getLearnPostCode().empty());
            }))
        {
            postsynapticUpdateKernelBody << "__local unsigned int shColLength[" << m_KernelWorkGroupSizes[KernelPostsynapticUpdate] << "];" << std::endl;
        }

            // Parallelise over synapse groups whose weight update models have code for postsynaptic learning
            genParallelGroup<SynapseGroupInternal>(postsynapticUpdateKernelBody, kernelSubs, model.getLocalSynapseGroups(), idPostsynapticStart, postsynapticUpdateKernelParams,
                [this](const SynapseGroupInternal& sg) { return Utils::padSize(getNumPostsynapticUpdateThreads(sg), m_KernelWorkGroupSizes[KernelPostsynapticUpdate]); },
                [](const SynapseGroupInternal& sg) { return !sg.getWUModel()->getLearnPostCode().empty(); },
                [postLearnHandler, &model, this, &postsynapticUpdateKernelParams](CodeStream& postsynapticUpdateKernelBody, const SynapseGroupInternal& sg, const Substitutions& popSubs)
                {
                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                        postsynapticUpdateKernelBody << "const unsigned int preReadDelayOffset = " << sg.getPresynapticAxonalDelaySlot("") << " * " << sg.getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;
                        postsynapticUpdateKernelParams.insert({ "spkQuePtr" + sg.getSrcNeuronGroup()->getName(), "volatile unsigned int" });
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if (sg.getTrgNeuronGroup()->isDelayRequired()) {
                        postsynapticUpdateKernelBody << "const unsigned int postReadDelaySlot = " << sg.getPostsynapticBackPropDelaySlot("") << ";" << std::endl;
                        postsynapticUpdateKernelBody << "const unsigned int postReadDelayOffset = postReadDelaySlot * " << sg.getTrgNeuronGroup()->getNumNeurons() << ";" << std::endl;
                        postsynapticUpdateKernelParams.insert({ "spkQuePtr" + sg.getTrgNeuronGroup()->getName(), "volatile unsigned int" });
                    }

                    if (sg.getTrgNeuronGroup()->isDelayRequired() && sg.getTrgNeuronGroup()->isTrueSpikeRequired()) {
                        postsynapticUpdateKernelBody << "const unsigned int numSpikes = d_glbSpkCnt" << sg.getTrgNeuronGroup()->getName() << "[postReadDelaySlot];" << std::endl;
                    }
                    else {
                        postsynapticUpdateKernelBody << "const unsigned int numSpikes = d_glbSpkCnt" << sg.getTrgNeuronGroup()->getName() << "[0];" << std::endl;
                    }
                    postsynapticUpdateKernelParams.insert({ "d_glbSpkCnt" + sg.getTrgNeuronGroup()->getName(), "__global unsigned int*" });

                    postsynapticUpdateKernelBody << "const unsigned int numSpikeBlocks = (numSpikes + " << m_KernelWorkGroupSizes[KernelPostsynapticUpdate] - 1 << ") / " << m_KernelWorkGroupSizes[KernelPostsynapticUpdate] << ";" << std::endl;
                    postsynapticUpdateKernelBody << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
                    {
                        CodeStream::Scope b(postsynapticUpdateKernelBody);
                        postsynapticUpdateKernelBody << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << m_KernelWorkGroupSizes[KernelPostsynapticUpdate] << ") + 1 : " << m_KernelWorkGroupSizes[KernelPostsynapticUpdate] << ";" << std::endl;

                        postsynapticUpdateKernelBody << "if (localId < numSpikesInBlock)";
                        {
                            CodeStream::Scope b(postsynapticUpdateKernelBody);
                            const std::string offsetTrueSpkPost = (sg.getTrgNeuronGroup()->isTrueSpikeRequired() && sg.getTrgNeuronGroup()->isDelayRequired()) ? "postReadDelayOffset + " : "";
                            postsynapticUpdateKernelBody << "const unsigned int spk = d_glbSpk" << sg.getTrgNeuronGroup()->getName() << "[" << offsetTrueSpkPost << "(r * " << m_KernelWorkGroupSizes[KernelPostsynapticUpdate] << ") + localId];" << std::endl;
                            postsynapticUpdateKernelBody << "shSpk[localId] = spk;" << std::endl;

                            postsynapticUpdateKernelParams.insert({ "d_glbSpk" + sg.getTrgNeuronGroup()->getName(), "__global unsigned int*" });

                            if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                postsynapticUpdateKernelBody << "shColLength[localId] = d_colLength" << sg.getName() << "[spk];" << std::endl;
                                postsynapticUpdateKernelParams.insert({ "d_colLength" + sg.getName(), "__global unsigned int*" });
                            }
                        }

                        postsynapticUpdateKernelBody << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
                        postsynapticUpdateKernelBody << "// only work on existing neurons" << std::endl;
                        postsynapticUpdateKernelBody << "if (" << popSubs["id"] << " < " << sg.getMaxSourceConnections() << ")";
                        {
                            CodeStream::Scope b(postsynapticUpdateKernelBody);
                            postsynapticUpdateKernelBody << "// loop through all incoming spikes for learning" << std::endl;
                            postsynapticUpdateKernelBody << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
                            {
                                CodeStream::Scope b(postsynapticUpdateKernelBody);

                                Substitutions synSubs(&popSubs);
                                if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    postsynapticUpdateKernelBody << "if (" << popSubs["id"] << " < shColLength[j])" << CodeStream::OB(1540);
                                    postsynapticUpdateKernelBody << "const unsigned int synAddress = d_remap" + sg.getName() + "[(shSpk[j] * " << std::to_string(sg.getMaxSourceConnections()) << ") + " << popSubs["id"] << "];" << std::endl;
                                    
                                    postsynapticUpdateKernelParams.insert({ "d_remap" + sg.getName(), "__global unsigned int*" });

                                    postsynapticUpdateKernelBody << "const unsigned int ipre = synAddress / " + std::to_string(sg.getMaxConnections()) + ";" << std::endl;
                                    synSubs.addVarSubstitution("id_pre", "ipre");
                                }
                                else {
                                    postsynapticUpdateKernelBody << "const unsigned int synAddress = (" << popSubs["id"] << " * " << std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()) << ") + shSpk[j];" << std::endl;
                                    synSubs.addVarSubstitution("id_pre", synSubs["id"]);
                                }

                                synSubs.addVarSubstitution("id_post", "shSpk[j]");
                                synSubs.addVarSubstitution("id_syn", "synAddress");

                                postLearnHandler(postsynapticUpdateKernelBody, sg, synSubs);

                                if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                                    postsynapticUpdateKernelBody << CodeStream::CB(1540);
                                }
                            }
                        }
                    }
                }
            );
    }
    //! KernelPostsynapticUpdate BODY END

    size_t idSynapseDynamicsStart = 0;
    std::stringstream synapseDynamicsUpdateKernelBodyStream;

    bool hasSynapseDynamicsUpdateKernel = std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
        [](const ModelSpec::SynapseGroupValueType& s) { return !s.second.getWUModel()->getSynapseDynamicsCode().empty(); });

    //! KernelSynapseDynamicsUpdate BODY START
    if (hasSynapseDynamicsUpdateKernel)
    {
        CodeStream synapseDynamicsUpdateKernelBody(synapseDynamicsUpdateKernelBodyStream);

        synapseDynamicsUpdateKernelBody << "const size_t localId = get_local_id(0);" << std::endl;
        synapseDynamicsUpdateKernelBody << "const unsigned int id = get_global_id(0);" << std::endl;

        Substitutions kernelSubs(openclFunctions, model.getPrecision());
        kernelSubs.addVarSubstitution("t", "t");

        // Parallelise over synapse groups whose weight update models have code for synapse dynamics
        genParallelGroup<SynapseGroupInternal>(synapseDynamicsUpdateKernelBody, kernelSubs, model.getLocalSynapseGroups(), idSynapseDynamicsStart, synapseDynamicsUpdateKernelParams,
            [this](const SynapseGroupInternal& sg) { return Utils::padSize(getNumSynapseDynamicsThreads(sg), m_KernelWorkGroupSizes[KernelSynapseDynamicsUpdate]); },
            [](const SynapseGroupInternal& sg) { return !sg.getWUModel()->getSynapseDynamicsCode().empty(); },
            [synapseDynamicsHandler, &model, this, &synapseDynamicsUpdateKernelParams](CodeStream& synapseDynamicsUpdateKernelBody, const SynapseGroupInternal& sg, const Substitutions& popSubs)
            {
                // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                    synapseDynamicsUpdateKernelBody << "const unsigned int preReadDelayOffset = " << sg.getPresynapticAxonalDelaySlot("") << " * " << sg.getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    synapseDynamicsUpdateKernelParams.insert({ "spkQuePtr" + sg.getSrcNeuronGroup()->getName(), "volatile unsigned int" });
                }

                // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                if (sg.getTrgNeuronGroup()->isDelayRequired()) {
                    synapseDynamicsUpdateKernelBody << "const unsigned int postReadDelayOffset = " << sg.getPostsynapticBackPropDelaySlot("") << " * " << sg.getTrgNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    synapseDynamicsUpdateKernelParams.insert({ "spkQuePtr" + sg.getTrgNeuronGroup()->getName(), "volatile unsigned int" });
                }

                Substitutions synSubs(&popSubs);

                if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    synapseDynamicsUpdateKernelBody << "if (" << popSubs["id"] << " < d_synRemap" << sg.getName() << "[0])";
                    synapseDynamicsUpdateKernelParams.insert({ "d_synRemap" + sg.getName(), "__global unsigned int*" });
                }
                else {
                    synapseDynamicsUpdateKernelBody << "if (" << popSubs["id"] << " < " << sg.getSrcNeuronGroup()->getNumNeurons() * sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                }

                {
                    CodeStream::Scope b(synapseDynamicsUpdateKernelBody);

                    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        // Determine synapse and presynaptic indices for this thread
                        synapseDynamicsUpdateKernelBody << "const unsigned int s = d_synRemap" << sg.getName() << "[1 + " << popSubs["id"] << "];" << std::endl;
                        // Parameter d_synRemap for kernel already inserted

                        synSubs.addVarSubstitution("id_pre", "s / " + std::to_string(sg.getMaxConnections()));
                        synSubs.addVarSubstitution("id_post", "d_ind" + sg.getName() + "[s]");
                        synSubs.addVarSubstitution("id_syn", "s");

                        synapseDynamicsUpdateKernelParams.insert({ "d_ind" + sg.getName(), "__global unsigned int*" });
                    }
                    else {
                        synSubs.addVarSubstitution("id_pre", popSubs["id"] + " / " + std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()));
                        synSubs.addVarSubstitution("id_post", popSubs["id"] + " % " + std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()));
                        synSubs.addVarSubstitution("id_syn", popSubs["id"]);
                    }

                    // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                    if (sg.isDendriticDelayRequired()) {
                        synSubs.addFuncSubstitution("addToInSynDelay", 2, getFloatAtomicAdd(model.getPrecision()) + "(&d_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("", "$(1)") + synSubs["id_post"] + "], $(0))");

                        synapseDynamicsUpdateKernelParams.insert({ "d_denDelay" + sg.getPSModelTargetName(), "__global unsigned int*" });
                        synapseDynamicsUpdateKernelParams.insert({ "denDelayPtr" + sg.getPSModelTargetName(), "volatile unsigned int" });
                    }
                    // Otherwise
                    else {
                        synSubs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&d_inSyn" + sg.getPSModelTargetName() + "[" + synSubs["id_post"] + "], $(0))");
                        synapseDynamicsUpdateKernelParams.insert({ "d_inSyn" + sg.getPSModelTargetName(), "__global float*" });
                    }

                    synapseDynamicsHandler(synapseDynamicsUpdateKernelBody, sg, synSubs);
                }
            });
    }
    //! KernelSynapseDynamicsUpdate BODY END

    // Actual kernels definitions
    os << "extern \"C\" const char* " << ProgramNames[ProgramSynapsesUpdate] << "Src = R\"(typedef float scalar;" << std::endl;
    os << "typedef unsigned char uint8_t;" << std::endl;
    os << std::endl;

    os << "#define fmodf fmod" << std::endl;
    // Definitions for bitmask
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// bit tool macros" << std::endl;
    os << "#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x" << std::endl;
    os << "#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1" << std::endl;
    os << "#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0" << std::endl;
    os << std::endl;

    ::genSupportCode(os, model);

    // Float atomic add function
    if (hasPresynapticUpdateKernel || hasSynapseDynamicsUpdateKernel) {
        std::vector<std::string> memoryTypes = { "global" };

        // Check if presynaptic update strategy needs local/shared memory
        if (std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [this](const ModelSpec::SynapseGroupValueType& s)
            { return this->getPresynapticUpdateStrategy(s.second)->shouldAccumulateInSharedMemory(s.second, *this); })) {
            memoryTypes.push_back("local");
        }

        for (const auto& memoryType : memoryTypes) {
            os << "void atomic_add_f_" << memoryType << "(volatile __" << memoryType << " float *source, const float operand)";
            {
                CodeStream::Scope b(os);
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

            os << std::endl;
        }
    }

    //! KernelPreSynapseReset
    if (hasPreSynapseResetKernel) {
        os << "__kernel void " << KernelNames[KernelPreSynapseReset] << "(";
        int argCnt = 0;
        for (const auto& p : preSynapseResetKernelParams) {
            if (argCnt == preSynapseResetKernelParams.size() - 1) {
                os << p.second << " " << p.first;
            }
            else {
                os << p.second << " " << p.first << ", ";
            }
            argCnt++;
        }
        os << ")";
        {
            CodeStream::Scope b(os);
            divideKernelStreamInParts(os, preSynapseResetKernelBodyStream, 5000);
        }

        os << std::endl;
    }

    // Collecting common parameters for KernelPresynapticUpdate, KernelPostsynapticUpdate and KernelSynapseDynamicsUpdate
    for (const auto& sg : model.getLocalSynapseGroups()) {
        // Weight update model
        auto* wum = sg.second.getWUModel();
        std::vector<Models::Base::Var> allVars;
        const auto wuVars = wum->getVars();
        const auto wuPreVars = wum->getPreVars();
        const auto wuPostVars = wum->getPostVars();
        if (sg.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
            allVars.insert(allVars.end(), wuVars.begin(), wuVars.end());
        }
        allVars.insert(allVars.end(), wuPreVars.begin(), wuPreVars.end());
        allVars.insert(allVars.end(), wuPostVars.begin(), wuPostVars.end());
        for (const auto& v : allVars) {
            if (hasPresynapticUpdateKernel) {
                presynapticUpdateKernelParams[getVarPrefix() + v.name + sg.second.getName()] = "__global " + v.type + "*";
            }
            if (hasPostsynapticUpdateKernel) {
                postsynapticUpdateKernelParams[getVarPrefix() + v.name + sg.second.getName()] = "__global " + v.type + "*";
            }
            if (hasSynapseDynamicsUpdateKernel) {
                synapseDynamicsUpdateKernelParams[getVarPrefix() + v.name + sg.second.getName()] = "__global " + v.type + "*";
            }
        }
    }

    //! KernelPresynapticUpdate
    if (hasPresynapticUpdateKernel) {
        os << "__kernel void " << KernelNames[KernelPresynapticUpdate] << "(";
        for (const auto& p : presynapticUpdateKernelParams) {
            os << p.second << " " << p.first << ", ";
        }
        os << model.getTimePrecision() << " t)";
        {
            CodeStream::Scope b(os);
            divideKernelStreamInParts(os, presynapticUpdateKernelBodyStream, 5000);
        }

        os << std::endl;
    }

    //! KernelPostsynapticUpdate
    if (hasPostsynapticUpdateKernel) {
        os << "__kernel void " << KernelNames[KernelPostsynapticUpdate] << "(";
        for (const auto& p : postsynapticUpdateKernelParams) {
            os << p.second << " " << p.first << ", ";
        }
        os << model.getTimePrecision() << " t)";
        {
            CodeStream::Scope b(os);
            divideKernelStreamInParts(os, postsynapticUpdateKernelBodyStream, 5000);
        }

        os << std::endl;
    }

    //! KernelSynapseDynamicsUpdate
    if (hasSynapseDynamicsUpdateKernel) {
        os << "__kernel void " << KernelNames[KernelSynapseDynamicsUpdate] << "(";
        for (const auto& p : synapseDynamicsUpdateKernelParams) {
            os << p.second << " " << p.first << ", ";
        }
        os << model.getTimePrecision() << " t)";
        {
            CodeStream::Scope b(os);
            divideKernelStreamInParts(os, synapseDynamicsUpdateKernelBodyStream, 5000);
        }

        os << std::endl;
    }

    os << ")\";" << std::endl;

    os << std::endl;
    
    // Function for initializing the synapse update kernels
    os << "// Initialize the synapse update kernel(s)" << std::endl;
    os << "void " << ProgramNames[ProgramSynapsesUpdate] << "Kernels()";
    {
        CodeStream::Scope b(os);

        // KernelPreSynapseReset initialization
        if (hasPreSynapseResetKernel && idPreSynapseReset > 0) {
            os << KernelNames[KernelPreSynapseReset] << " = cl::Kernel(" << ProgramNames[ProgramSynapsesUpdate] << ", \"" << KernelNames[KernelPreSynapseReset] << "\");" << std::endl;
            int argCnt = 0;
            for (const auto& arg : preSynapseResetKernelParams) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelPreSynapseReset] << ".setArg(" << argCnt << ", " << arg.first << "));" << std::endl;
                argCnt++;
            }
        }

        // KernelSynapseDynamicsUpdate initialization
        if (hasSynapseDynamicsUpdateKernel && idSynapseDynamicsStart > 0) {
            os << KernelNames[KernelSynapseDynamicsUpdate] << " = cl::Kernel(" << ProgramNames[ProgramSynapsesUpdate] << ", \"" << KernelNames[KernelSynapseDynamicsUpdate] << "\");" << std::endl;
            int argCnt = 0;
            for (const auto& arg : synapseDynamicsUpdateKernelParams) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelSynapseDynamicsUpdate] << ".setArg(" << argCnt << ", " << arg.first << "));" << std::endl;
                argCnt++;
            }
        }

        // KernelPresynapticUpdate initialization
        if (hasPresynapticUpdateKernel && idPresynapticStart > 0) {
            os << KernelNames[KernelPresynapticUpdate] << " = cl::Kernel(" << ProgramNames[ProgramSynapsesUpdate] << ", \"" << KernelNames[KernelPresynapticUpdate] << "\");" << std::endl;
            int argCnt = 0;
            for (const auto& arg : presynapticUpdateKernelParams) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelPresynapticUpdate] << ".setArg(" << argCnt << ", " << arg.first << "));" << std::endl;
                argCnt++;
            }
        }

        // KernelPostsynapticUpdate initialization
        if (hasPostsynapticUpdateKernel && idPostsynapticStart > 0) {
            os << KernelNames[KernelPostsynapticUpdate] << " = cl::Kernel(" << ProgramNames[ProgramSynapsesUpdate] << ", \"" << KernelNames[KernelPostsynapticUpdate] << "\");" << std::endl;
            int argCnt = 0;
            for (const auto& arg : postsynapticUpdateKernelParams) {
                os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelPostsynapticUpdate] << ".setArg(" << argCnt << ", " << arg.first << "));" << std::endl;
                argCnt++;
            }
        }
    }

    os << std::endl;*/

    os << "void updateSynapses(" << modelMerged.getModel().getTimePrecision() << " t)";
    {
        CodeStream::Scope b(os);

        // Launch pre-synapse reset kernel if required
        /*if (idPreSynapseReset > 0) {
            CodeStream::Scope b(os);
            genKernelHostArgs(os, KernelPreSynapseReset, preSynapseResetKernelParams);
            genKernelDimensions(os, KernelPreSynapseReset, idPreSynapseReset);
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelPreSynapseReset] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
            os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
        }

        // Launch synapse dynamics kernel if required
        if (idSynapseDynamicsStart > 0) {
            CodeStream::Scope b(os);
            genKernelHostArgs(os, KernelSynapseDynamicsUpdate, synapseDynamicsUpdateKernelParams);
            os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelSynapseDynamicsUpdate] << ".setArg(" << synapseDynamicsUpdateKernelParams.size() << ", t));" << std::endl;
            os << std::endl;
            genKernelDimensions(os, KernelSynapseDynamicsUpdate, idSynapseDynamicsStart);
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelSynapseDynamicsUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
            os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
        }

        // Launch presynaptic update kernel
        if (idPresynapticStart > 0) {
            CodeStream::Scope b(os);
            genKernelHostArgs(os, KernelPresynapticUpdate, presynapticUpdateKernelParams);
            os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelPresynapticUpdate] << ".setArg(" << presynapticUpdateKernelParams.size() << ", t));" << std::endl;
            os << std::endl;
            genKernelDimensions(os, KernelPresynapticUpdate, idPresynapticStart);
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelPresynapticUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
            os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
        }

        // Launch postsynaptic update kernel
        if (idPostsynapticStart > 0) {
            CodeStream::Scope b(os);
            genKernelHostArgs(os, KernelPostsynapticUpdate, postsynapticUpdateKernelParams);
            os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelPostsynapticUpdate] << ".setArg(" << postsynapticUpdateKernelParams.size() << ", t));" << std::endl;
            os << std::endl;
            genKernelDimensions(os, KernelPostsynapticUpdate, idPostsynapticStart);
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelPostsynapticUpdate] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
            os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
        }*/
    }
}
//--------------------------------------------------------------------------
void Backend::genInit(CodeStream &os, const ModelSpecMerged &modelMerged,
                      NeuronInitGroupMergedHandler localNGHandler, SynapseDenseInitGroupMergedHandler sgDenseInitHandler,
                      SynapseConnectivityInitMergedGroupHandler sgSparseConnectHandler, SynapseSparseInitGroupMergedHandler sgSparseInitHandler,
                      HostHandler initPushEGPHandler, HostHandler initSparsePushEGPHandler) const
{
    os << std::endl;
    //! TO BE IMPLEMENTED - Generating minimal kernel
    //! TO BE IMPLEMENTED - initializeRNGKernel - if needed
    // Generate reset kernel to be run before the neuron kernel
    const ModelSpecInternal &model = modelMerged.getModel();

    // initialization kernel code
    size_t idInitStart = 0;

    //! KernelInitialize BODY START
    Substitutions kernelSubs(openclFunctions, model.getPrecision());

    // Creating the kernel body separately so it can be split into multiple string literals
    std::stringstream initializeKernelsStream;
    CodeStream initializeKernels(initializeKernelsStream);

    // Include definitions
    initializeKernels << "typedef " << model.getPrecision() << " scalar;" << std::endl;
    genTypeRange(initializeKernels, model.getTimePrecision(), "TIME");

    // Generate data structure for accessing merged groups from within initialisation kernel
    // **NOTE** pass in zero constant cache here as it's precious and would be wasted on init kernels which are only launched once
    genMergedKernelDataStructures(initializeKernels, m_KernelWorkGroupSizes[KernelInitialize],
                                  modelMerged.getMergedNeuronInitGroups(), "NeuronInit",
                                  [](const NeuronGroupInternal &ng) { return ng.getNumNeurons(); },
                                  modelMerged.getMergedSynapseDenseInitGroups(), "SynapseDenseInit",
                                  [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getNumNeurons(); },
                                  modelMerged.getMergedSynapseConnectivityInitGroups(), "SynapseConnectivityInit",
                                  [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getNumNeurons(); });

    // Generate data structure for accessing merged groups from within sparse initialisation kernel
    genMergedKernelDataStructures(initializeKernels, m_KernelWorkGroupSizes[KernelInitializeSparse],
                                  modelMerged.getMergedSynapseSparseInitGroups(), "SynapseSparseInit",
                                  [](const SynapseGroupInternal &sg) { return sg.getMaxConnections(); });
    initializeKernels << std::endl;


    initializeKernels << "__kernel void " << KernelNames[KernelInitialize] << "(";
    genMergedGroupKernelParams(initializeKernels, modelMerged.getMergedNeuronInitGroups(), "NeuronInit", true);
    genMergedGroupKernelParams(initializeKernels, modelMerged.getMergedSynapseDenseInitGroups(), "SynapseDenseInit", true);
    genMergedGroupKernelParams(initializeKernels, modelMerged.getMergedSynapseConnectivityInitGroups(), "SynapseConnectivityInit", true);
    initializeKernels << "unsigned int deviceRNGSeed";  // **TODO** check type
    initializeKernels << ")";
    {
        CodeStream::Scope b(initializeKernels);

        initializeKernels << "const unsigned int localId = get_local_id(0);" << std::endl;
        initializeKernels << "const unsigned int id = get_global_id(0);" << std::endl;

        initializeKernels << "// ------------------------------------------------------------------------" << std::endl;
        initializeKernels << "// Local neuron groups" << std::endl;
        // Parallelise over neuron groups
        genParallelGroup<NeuronInitGroupMerged>(initializeKernels, kernelSubs, modelMerged.getMergedNeuronInitGroups(), "NeuronInit", idInitStart,
            [this](const NeuronGroupInternal &ng) { return padSize(ng.getNumNeurons(), getKernelBlockSize(KernelInitialize)); },
            [localNGHandler](CodeStream &os, const NeuronInitGroupMerged &ng, Substitutions &popSubs)
            {
                os << "// only do this for existing neurons" << std::endl;
                os << "if(" << popSubs["id"] << " < group->numNeurons)";
                {
                    CodeStream::Scope b(os);

                    //! TO BE IMPLEMENTED - isSimRNGRequired - isInitRNGRequired

                    localNGHandler(os, ng, popSubs);
                }
            });
        initializeKernels << std::endl;

        /*initializeKernels << "// ------------------------------------------------------------------------" << std::endl;
        initializeKernels << "// Synapse groups with dense connectivity" << std::endl;
        genParallelGroup<SynapseGroupInternal>(initializeKernels, kernelSubs, model.getLocalSynapseGroups(), idInitStart, initializeKernelParams,
            [this](const SynapseGroupInternal& sg) { return Utils::padSize(sg.getTrgNeuronGroup()->getNumNeurons(), m_KernelWorkGroupSizes[KernelInitialize]); },
            [](const SynapseGroupInternal& sg) { return (sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && sg.isWUVarInitRequired(); },
            [sgDenseInitHandler](CodeStream& os, const SynapseGroupInternal& sg, Substitutions& popSubs)
            {
                os << "// only do this for existing postsynaptic neurons" << std::endl;
                os << "if(" << popSubs["id"] << " < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);

                    //! TO BE IMPLEMENTED - isWUInitRNGRequired

                    popSubs.addVarSubstitution("id_post", popSubs["id"]);
                    sgDenseInitHandler(os, sg, popSubs);
                }
            });
        initializeKernels << std::endl;

        initializeKernels << "// ------------------------------------------------------------------------" << std::endl;
        initializeKernels << "// Synapse groups with sparse connectivity" << std::endl;
        genParallelGroup<SynapseGroupInternal>(initializeKernels, kernelSubs, model.getLocalSynapseGroups(), idInitStart, initializeKernelParams,
            [this](const SynapseGroupInternal& sg) { return Utils::padSize(sg.getSrcNeuronGroup()->getNumNeurons(), m_KernelWorkGroupSizes[KernelInitialize]); },
            [](const SynapseGroupInternal& sg) { return sg.isSparseConnectivityInitRequired(); },
            [sgSparseConnectHandler, &initializeKernelParams](CodeStream& initializeKernels, const SynapseGroupInternal& sg, Substitutions& popSubs)
            {
                const size_t numSrcNeurons = sg.getSrcNeuronGroup()->getNumNeurons();
                const size_t numTrgNeurons = sg.getTrgNeuronGroup()->getNumNeurons();

                initializeKernels << "// only do this for existing presynaptic neurons" << std::endl;
                initializeKernels << "if(" << popSubs["id"] << " < " << numSrcNeurons << ")";
                {
                    CodeStream::Scope b(initializeKernels);

                    //! TO BE IMPLEMENTED - ::Utils::isRNGRequired

                    // If the synapse group has bitmask connectivity
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // Calculate indices of bits at start and end of row
                        initializeKernels << "// Calculate indices" << std::endl;
                        const size_t maxSynapses = numSrcNeurons * numTrgNeurons;
                        if ((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                            initializeKernels << "const ulong rowStartGID = " << popSubs["id"] << " * " << numTrgNeurons << "ull;" << std::endl;
                        }
                        else {
                            initializeKernels << "const unsigned int rowStartGID = " << popSubs["id"] << " * " << numTrgNeurons << ";" << std::endl;
                        }

                        // Build function template to set correct bit in bitmask
                        popSubs.addFuncSubstitution("addSynapse", 1,
                            "atomic_or(&d_gp" + sg.getName() + "[(rowStartGID + $(0)) / 32], 0x80000000 >> ((rowStartGID + $(0)) & 31))");
                        initializeKernelParams.insert({ "d_gp" + sg.getName(), "__global unsigned int*" });
                    }
                    // Otherwise, if synapse group has ragged connectivity
                    else if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                        const std::string rowLength = "d_rowLength" + sg.getName() + "[" + popSubs["id"] + "]";
                        const std::string ind = "d_ind" + sg.getName();

                        initializeKernelParams.insert({ "d_rowLength" + sg.getName(), "__global unsigned int*" });
                        initializeKernelParams.insert({ "d_ind" + sg.getName(), "__global unsigned int*" });

                        // Zero row length
                        initializeKernels << rowLength << " = 0;" << std::endl;

                        // Build function template to increment row length and insert synapse into ind array
                        popSubs.addFuncSubstitution("addSynapse", 1,
                            ind + "[(" + popSubs["id"] + " * " + std::to_string(sg.getMaxConnections()) + ") + (" + rowLength + "++)] = $(0)");
                    }
                    else {
                        assert(false);
                    }

                    popSubs.addVarSubstitution("id_pre", popSubs["id"]);
                    sgSparseConnectHandler(initializeKernels, sg, popSubs);
                }
            });*/
    }
    /*os << std::endl;
    const size_t numStaticInitThreads = idInitStart;

    //! KernelInitialize BODY END

    //! KernelInitializeSparse BODY START
    std::stringstream initializeSparseKernelBodyStream;
    // Parameters to pass onto KernelInitializeSparse
    std::map<std::string, std::string> initializeSparseKernelParams;
    size_t idSparseInitStart = 0;

    bool hasInitializeSparseKernel = std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
        [](const ModelSpec::SynapseGroupValueType& s) { return isSparseInitRequired(s.second); });

    // Sparse initialization kernel code
    if (hasInitializeSparseKernel)
    {
        CodeStream initializeSparseKernelBody(initializeSparseKernelBodyStream);

        // Common variables for all cases
        Substitutions kernelSubs(openclFunctions, model.getPrecision());

        initializeSparseKernelBody << "const size_t localId = get_local_id(0);" << std::endl;
        initializeSparseKernelBody << "const unsigned int id = get_global_id(0);" << std::endl;

        // Shared memory array so row lengths don't have to be read by EVERY postsynaptic thread
        // **TODO** check actually required
        initializeSparseKernelBody << "__local unsigned int shRowLength[" << m_KernelWorkGroupSizes[KernelInitializeSparse] << "];" << std::endl;
        if (std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [](const ModelSpec::SynapseGroupValueType& s) { return (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getSynapseDynamicsCode().empty(); }))
        {
            initializeSparseKernelBody << "__local unsigned int shRowStart[" << m_KernelWorkGroupSizes[KernelInitializeSparse] + 1 << "];" << std::endl;
        }

        // Initialise weight update variables for synapse groups with dense connectivity
        genParallelGroup<SynapseGroupInternal>(initializeSparseKernelBody, kernelSubs, model.getLocalSynapseGroups(), idSparseInitStart, initializeSparseKernelParams,
            [this](const SynapseGroupInternal& sg) { return Utils::padSize(sg.getMaxConnections(), m_KernelWorkGroupSizes[KernelInitializeSparse]); },
            [](const SynapseGroupInternal& sg) { return isSparseInitRequired(sg); },
            [this, &model, sgSparseInitHandler, numStaticInitThreads, &initializeSparseKernelParams](CodeStream& initializeSparseKernelBody, const SynapseGroupInternal& sg, Substitutions& popSubs)
            {

                //! TO BE IMPLEMENTED - isWUInitRNGRequired

                initializeSparseKernelBody << "unsigned int idx = " << popSubs["id"] << ";" << std::endl;

                // Calculate how many blocks rows need to be processed in (in order to store row lengths in shared memory)
                const unsigned int numSrcNeurons = sg.getSrcNeuronGroup()->getNumNeurons();
                const size_t numBlocks = Utils::ceilDivide(numSrcNeurons, m_KernelWorkGroupSizes[KernelInitializeSparse]);

                // Loop through blocks
                initializeSparseKernelBody << "for(unsigned int r = 0; r < " << numBlocks << "; r++)";
                {
                    CodeStream::Scope b(initializeSparseKernelBody);

                    // Calculate number of rows to process in this block
                    initializeSparseKernelBody << "const unsigned numRowsInBlock = (r == " << numBlocks - 1 << ")";
                    initializeSparseKernelBody << " ? " << ((numSrcNeurons - 1) % m_KernelWorkGroupSizes[KernelInitializeSparse]) + 1;
                    initializeSparseKernelBody << " : " << m_KernelWorkGroupSizes[KernelInitializeSparse] << ";" << std::endl;

                    // Use threads to copy block of sparse structure into shared memory
                    initializeSparseKernelBody << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
                    initializeSparseKernelBody << "if (localId < numRowsInBlock)";
                    {
                        CodeStream::Scope b(initializeSparseKernelBody);
                        initializeSparseKernelBody << "shRowLength[localId] = d_rowLength" << sg.getName() << "[(r * " << m_KernelWorkGroupSizes[KernelInitializeSparse] << ") + localId];" << std::endl;
                        initializeSparseKernelParams.insert({ "d_rowLength" + sg.getName(), "__global unsigned int*" });
                    }

                    // If this synapse group has synapse dynamics
                    if (!sg.getWUModel()->getSynapseDynamicsCode().empty()) {
                        initializeSparseKernelBody << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

                        // Use first thread to generate cumulative sum
                        initializeSparseKernelBody << "if (localId == 0)";
                        {
                            CodeStream::Scope b(initializeSparseKernelBody);

                            // Get index of last row in resultant synapse dynamics structure
                            // **NOTE** if there IS a previous block, it will always have had initSparseBlkSz rows in it
                            initializeSparseKernelBody << "unsigned int rowStart = (r == 0) ? 0 : shRowStart[" << m_KernelWorkGroupSizes[KernelInitializeSparse] << "];" << std::endl;
                            initializeSparseKernelBody << "shRowStart[0] = rowStart;" << std::endl;

                            // Loop through rows in block
                            initializeSparseKernelBody << "for(unsigned int i = 0; i < numRowsInBlock; i++)";
                            {
                                CodeStream::Scope b(initializeSparseKernelBody);

                                // Add this row's length to cumulative sum and write this to this row's end
                                initializeSparseKernelBody << "rowStart += shRowLength[i];" << std::endl;
                                initializeSparseKernelBody << "shRowStart[i + 1] = rowStart;" << std::endl;
                            }

                            // If this is the first thread block of the first block in the group AND the last block of rows,
                            // write the total cumulative sum to the first entry of the remap structure
                            initializeSparseKernelBody << "if(" << popSubs["id"] << " == 0 && (r == " << numBlocks - 1 << "))";
                            {
                                CodeStream::Scope b(initializeSparseKernelBody);
                                initializeSparseKernelBody << "d_synRemap" << sg.getName() << "[0] = shRowStart[numRowsInBlock];" << std::endl;
                                initializeSparseKernelParams.insert({ "d_synRemap" + sg.getName(), "__global unsigned int*" });
                            }

                        }
                    }

                    initializeSparseKernelBody << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

                    // Loop through rows
                    initializeSparseKernelBody << "for(unsigned int i = 0; i < numRowsInBlock; i++)";
                    {
                        CodeStream::Scope b(initializeSparseKernelBody);

                        // If there is a synapse for this thread to initialise
                        initializeSparseKernelBody << "if(" << popSubs["id"] << " < shRowLength[i])";
                        {
                            CodeStream::Scope b(initializeSparseKernelBody);

                            // Generate sparse initialisation code
                            if (sg.isWUVarInitRequired()) {
                                popSubs.addVarSubstitution("id_pre", "((r * " + std::to_string(m_KernelWorkGroupSizes[KernelInitializeSparse]) + ") + i)");
                                popSubs.addVarSubstitution("id_post", "d_ind" + sg.getName() + "[idx]");

                                initializeSparseKernelParams.insert({ "d_ind" + sg.getName(), "__global unsigned int*" });

                                sgSparseInitHandler(initializeSparseKernelBody, sg, popSubs);
                            }

                            // If postsynaptic learning is required
                            if (!sg.getWUModel()->getLearnPostCode().empty()) {
                                CodeStream::Scope b(initializeSparseKernelBody);

                                // Extract index of synapse's postsynaptic target
                                initializeSparseKernelBody << "const unsigned int postIndex = d_ind" << sg.getName() << "[idx];" << std::endl;
                                initializeSparseKernelParams.insert({ "d_ind" + sg.getName(), "__global unsigned int*" });

                                // Atomically increment length of column of connectivity associated with this target
                                // **NOTE** this returns previous length i.e. where to insert new entry
                                initializeSparseKernelBody << "const unsigned int colLocation = atomic_add(&d_colLength" << sg.getName() << "[postIndex], 1);" << std::endl;
                                initializeSparseKernelParams.insert({ "d_colLength" + sg.getName(), "__global unsigned int*" });

                                // From this calculate index into column-major matrix
                                initializeSparseKernelBody << "const unsigned int colMajorIndex = (postIndex * " << sg.getMaxSourceConnections() << ") + colLocation;" << std::endl;

                                // Add remapping entry at this location poining back to row-major index
                                initializeSparseKernelBody << "d_remap" << sg.getName() << "[colMajorIndex] = idx;" << std::endl;
                                initializeSparseKernelParams.insert({ "d_remap" + sg.getName(), "__global unsigned int*" });
                            }

                            // If synapse dynamics are required, copy idx into syn remap structure
                            if (!sg.getWUModel()->getSynapseDynamicsCode().empty()) {
                                CodeStream::Scope b(initializeSparseKernelBody);
                                initializeSparseKernelBody << "d_synRemap" << sg.getName() << "[shRowStart[i] + " + popSubs["id"] + " + 1] = idx;" << std::endl;
                                initializeSparseKernelParams.insert({ "d_synRemap" + sg.getName(), "__global unsigned int*" });
                            }
                        }

                        // If matrix is ragged, advance index to next row by adding stride
                        initializeSparseKernelBody << "idx += " << sg.getMaxConnections() << ";" << std::endl;
                    }
                }
            });
        initializeSparseKernelBody << std::endl;
    }*/
    //! KernelInitializeSparse BODY END

    // Write out kernel source string literal
    os << "const char* " << ProgramNames[ProgramInitialize] << "Src = ";
    divideKernelStreamInParts(os, initializeKernelsStream, 5000);
    os << ";" << std::endl;

    // Function for initializing the initialization kernels
    os << "// Initialize the initialization kernel(s)" << std::endl;
    os << "void " << ProgramNames[ProgramInitialize] << "Kernels()";
    {
        CodeStream::Scope b(os);

        if (idInitStart > 0) {
            // KernelInitialize initialization
            os << KernelNames[KernelInitialize] << " = cl::Kernel(" << ProgramNames[ProgramInitialize] << ", \"" << KernelNames[KernelInitialize] << "\");" << std::endl;
            setMergedGroupKernelParams(os, KernelNames[KernelInitialize], modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), "NeuronSpikeQueueUpdate");
            setMergedGroupKernelParams(os, KernelNames[KernelInitialize], modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), "NeuronSpikeQueueUpdate");
            setMergedGroupKernelParams(os, KernelNames[KernelInitialize], modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), "NeuronSpikeQueueUpdate");
        }

        /*if (hasInitializeSparseKernel && idSparseInitStart > 0)
        {
            os << std::endl;
            // KernelInitializeSparse initialization
            os << KernelNames[KernelInitializeSparse] << " = cl::Kernel(" << ProgramNames[ProgramInitialize] << ", \"" << KernelNames[KernelInitializeSparse] << "\");" << std::endl;
            {
                int argCnt = 0;
                for (const auto& arg : initializeSparseKernelParams) {
                    os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelInitializeSparse] << ".setArg(" << argCnt << ", " << arg.first << "));" << std::endl;
                    argCnt++;
                }
            }
        }*/
    }

    os << std::endl;

    os << "void initialize()";
    {
        CodeStream::Scope b(os);

        // If there are any initialisation work-items
        if (idInitStart > 0) {
            CodeStream::Scope b(os);
            //! TO BE IMPLEMENTED - Using hard coded deviceRNGSeed for now
            os << "unsigned int deviceRNGSeed = 0;" << std::endl;
            /*for (const auto& s : model.getLocalSynapseGroups()) {
                // If this synapse population has BITMASK connectivity and is intialised on device, insert a call to cudaMemset to zero the whole bitmask
                if (s.second.isSparseConnectivityInitRequired() && s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    const size_t gpSize = ((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * (size_t)s.second.getTrgNeuronGroup()->getNumNeurons()) / 32 + 1;
                    os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueFillBuffer(d_gp" << s.first << ", 0, 0, " << gpSize << " * sizeof(uint32_t)));" << std::endl;
                }
                // Otherwise, if this synapse population has RAGGED connectivity and has postsynaptic learning, insert a call to cudaMemset to zero column lengths
                else if ((s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !s.second.getWUModel()->getLearnPostCode().empty()) {
                    os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueFillBuffer(d_colLength" << s.first << ", 0, 0, " << s.second.getTrgNeuronGroup()->getNumNeurons() << " * sizeof(unsigned int)));" << std::endl;
                }
            }*/
            os << std::endl;
            //os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelInitialize] << ".setArg(" << initializeKernelParams.size() /*last arg*/ << ", deviceRNGSeed));" << std::endl;
            os << std::endl;
            genKernelDimensions(os, KernelInitialize, idInitStart);
            const size_t numInitGroups = (modelMerged.getMergedNeuronInitGroups().size() + modelMerged.getMergedSynapseDenseInitGroups().size() + 
                                          modelMerged.getMergedSynapseConnectivityInitGroups().size());
            os << "CHECK_OPENCL_ERRORS(" << KernelNames[KernelInitialize] << ".setArg(" << numInitGroups << ", deviceRNGSeed));" << std::endl;
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelInitialize] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
            os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
        }
    }

    os << std::endl;

    // Generating code for initializing all OpenCL elements - Using intializeSparse
    os << "// Initialize all OpenCL elements" << std::endl;
    os << "void initializeSparse()";
    {
        CodeStream::Scope b(os);
        // Copy all uninitialised state variables to device
        os << "copyStateToDevice(true);" << std::endl;
        os << "copyConnectivityToDevice(true);" << std::endl;

        // If there are any sparse initialisation work-items
        /*if (idSparseInitStart > 0) {
            CodeStream::Scope b(os);
            {
                genKernelHostArgs(os, KernelInitializeSparse, initializeSparseKernelParams);
                genKernelDimensions(os, KernelInitializeSparse, idSparseInitStart);
                os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << KernelNames[KernelInitializeSparse] << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
                os << "CHECK_OPENCL_ERRORS(commandQueue.finish());" << std::endl;
            }
        }*/
    }
}
//--------------------------------------------------------------------------
size_t Backend::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    return getPresynapticUpdateStrategy(sg)->getSynapticMatrixRowStride(sg);
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsPreamble(CodeStream& os, const ModelSpecMerged&) const
{
    os << "// Standard C++ includes" << std::endl;
    os << "#include <string>" << std::endl;
    os << "#include <stdexcept>" << std::endl;
    os << std::endl;
    os << "// Standard C includes" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << "#include <cassert>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsInternalPreamble(CodeStream& os, const ModelSpecMerged &) const
{
#ifdef _WIN32
    os << "#pragma warning(disable: 4297)" << std::endl;
#endif
    os << "// OpenCL includes" << std::endl;
    os << "#define CL_USE_DEPRECATED_OPENCL_1_2_APIS" << std::endl;
    os << "#include <CL/cl.hpp>" << std::endl;
    os << "#include \"clRNG/lfsr113.h\"" << std::endl;
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

    // Declaration of OpenCL functions
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// OpenCL functions declaration" << std::endl;
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "const char* clGetErrorString(cl_int error);" << std::endl;

    os << std::endl;

    // Declaration of OpenCL variables
    os << "// OpenCL variables" << std::endl;
    os << "EXPORT_VAR cl::Context clContext;" << std::endl;
    os << "EXPORT_VAR cl::Device clDevice;" << std::endl;
    os << "EXPORT_VAR cl::CommandQueue commandQueue;" << std::endl;
    os << std::endl;

    // **TODO** move into kernels
    os << "// OpenCL programs" << std::endl;
    for (const auto& programName : ProgramNames) {
        os << "EXPORT_VAR cl::Program " << programName << ";" << std::endl;
    }
    os << std::endl;
    os << "// OpenCL kernels" << std::endl;
    for (const auto& kernelName : KernelNames) {
        os << "EXPORT_VAR cl::Kernel " << kernelName << ";" << std::endl;
    }
    os << "// OpenCL kernels initialization functions and kernels sources" << std::endl;
    for (const auto& programName : ProgramNames) {
        os << "EXPORT_FUNC void " << programName << "Kernels();" << std::endl;
        os << "EXPORT_VAR const char* " << programName << "Src;" << std::endl;
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genRunnerPreamble(CodeStream& os, const ModelSpecMerged &) const
{
    // Generating OpenCL variables for the runner
    os << "// OpenCL variables" << std::endl;
    os << "cl::Context clContext;" << std::endl;
    os << "cl::Device clDevice;" << std::endl;
    os << "cl::CommandQueue commandQueue;" << std::endl;
    os << std::endl;
    os << "// OpenCL programs" << std::endl;
    for (const auto& programName : ProgramNames) {
        os << "cl::Program " << programName << ";" << std::endl;
    }
    os << std::endl;
    os << "// OpenCL kernels" << std::endl;
    for(const auto &kernelName : KernelNames) {
        os << "cl::Kernel " << kernelName << ";" << std::endl;
    }

    os << std::endl;
    os << "// Get OpenCL error as string" << std::endl;
    os << "const char* clGetErrorString(cl_int error)";
    {
        CodeStream::Scope b(os);
        std::map<cl_int, std::string> allClErrors = {
            // run-time and JIT compiler errors
            { 0, "CL_SUCCESS" },                            { -1, "CL_DEVICE_NOT_FOUND" },              { -2, "CL_DEVICE_NOT_AVAILABLE" },
            { -3, "CL_COMPILER_NOT_AVAILABLE" },            { -4, "CL_MEM_OBJECT_ALLOCATION_FAILURE" }, { -5, "CL_OUT_OF_RESOURCES" },
            { -6, "CL_OUT_OF_HOST_MEMORY" },                { -7, "CL_PROFILING_INFO_NOT_AVAILABLE" },  { -8, "CL_MEM_COPY_OVERLAP" },
            { -9, "CL_IMAGE_FORMAT_MISMATCH" },             { -10, "CL_IMAGE_FORMAT_NOT_SUPPORTED" },   { -11, "CL_BUILD_PROGRAM_FAILURE" },
            { -12, "CL_MAP_FAILURE" },                      { -13, "CL_MISALIGNED_SUB_BUFFER_OFFSET" }, { -14, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST" },
            { -15, "CL_COMPILE_PROGRAM_FAILURE" },          { -16, "CL_LINKER_NOT_AVAILABLE" },         { -17, "CL_LINK_PROGRAM_FAILURE" },
            { -18, "CL_DEVICE_PARTITION_FAILED" },          { -19, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE" },
            // compile-time errors
            { -30, "CL_INVALID_VALUE" },                    { -31, "CL_INVALID_DEVICE_TYPE" },          { -32, "CL_INVALID_PLATFORM" },
            { -33, "CL_INVALID_DEVICE" },                   { -34, "CL_INVALID_CONTEXT" },              { -35, "CL_INVALID_QUEUE_PROPERTIES" },
            { -36, "CL_INVALID_COMMAND_QUEUE" },            { -37, "CL_INVALID_HOST_PTR" },             { -38, "CL_INVALID_MEM_OBJECT" },
            { -39, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR" },  { -40, "CL_INVALID_IMAGE_SIZE" },           { -41, "CL_INVALID_SAMPLER" },
            { -42, "CL_INVALID_BINARY" },                   { -43, "CL_INVALID_BUILD_OPTIONS" },        { -44, "CL_INVALID_PROGRAM" },
            { -45, "CL_INVALID_PROGRAM_EXECUTABLE" },       { -46, "CL_INVALID_KERNEL_NAME" },          { -47, "CL_INVALID_KERNEL_DEFINITION" },
            { -48, "CL_INVALID_KERNEL" },                   { -49, "CL_INVALID_ARG_INDEX" },            { -50, "CL_INVALID_ARG_VALUE" },
            { -51, "CL_INVALID_ARG_SIZE" },                 { -52, "CL_INVALID_KERNEL_ARGS" },          { -53, "CL_INVALID_WORK_DIMENSION" },
            { -54, "CL_INVALID_WORK_GROUP_SIZE" },          { -55, "CL_INVALID_WORK_ITEM_SIZE" },       { -56, "CL_INVALID_GLOBAL_OFFSET" },
            { -57, "CL_INVALID_EVENT_WAIT_LIST" },          { -58, "CL_INVALID_EVENT" },                { -59, "CL_INVALID_OPERATION" },
            { -60, "CL_INVALID_GL_OBJECT" },                { -61, "CL_INVALID_BUFFER_SIZE" },          { -62, "CL_INVALID_MIP_LEVEL" },
            { -63, "CL_INVALID_GLOBAL_WORK_SIZE" },         { -64, "CL_INVALID_PROPERTY" },             { -65, "CL_INVALID_IMAGE_DESCRIPTOR" },
            { -66, "CL_INVALID_COMPILER_OPTIONS" },         { -67, "CL_INVALID_LINKER_OPTIONS" },       { -68, "CL_INVALID_DEVICE_PARTITION_COUNT" }
        };

        os << "switch(error)";
        {
            CodeStream::Scope b(os);
            for (const auto& e : allClErrors) {
                os << "case " << e.first << ": return \"" << e.second << "\";" << std::endl;
            }
            os << "default: return \"Unknown OpenCL error\";" << std::endl;
        }
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream& os, const ModelSpecMerged&) const
{
    // Initializing OpenCL programs
    os << "// Get platforms" << std::endl;
    os << "std::vector<cl::Platform> platforms; " << std::endl;
    os << "cl::Platform::get(&platforms);" << std::endl;
    os << "// Get platform devices" << std::endl;
    os << "std::vector<cl::Device> platformDevices; " << std::endl;
    os << "platforms[" << m_ChosenPlatformIndex << "].getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);" << std::endl;
    os << "// Select device and create context and command queue" << std::endl;
    os << "clDevice = platformDevices[" << m_ChosenDeviceIndex << "];" << std::endl;
    os << "clContext = cl::Context(clDevice);" << std::endl;
    os << "commandQueue = cl::CommandQueue(clContext, clDevice);" << std::endl;
    os << std::endl;
    os << "// Create programs for kernels" << std::endl;
    for(const auto &programName : ProgramNames) {
        os << "// Reading the kernel source for execution" << std::endl;
        os << programName << " = cl::Program(clContext, " << programName << "Src, true);" << std::endl;
        os << programName << ".build(\"-cl-std=CL1.2 -I clRNG/include\");" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPostamble(CodeStream& os, const ModelSpecMerged&) const
{
    // Initializing OpenCL kernels - after buffer initialization
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// OpenCL kernels initialization" << std::endl;
    os << "// ------------------------------------------------------------------------" << std::endl;
    for (const auto& programName : ProgramNames) {
        os << programName << "Kernels();" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genStepTimeFinalisePreamble(CodeStream& os, const ModelSpecMerged&) const
{
    printf("TO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genStepTimeFinalisePreamble\n");
}
//--------------------------------------------------------------------------
void Backend::genVariableDefinition(CodeStream& definitions, CodeStream& definitionsInternal, const std::string& type, const std::string& name, VarLocation loc) const
{
    const bool deviceType = isDeviceType(type);

    if (loc & VarLocation::HOST) {
        if (deviceType) {
            throw std::runtime_error("Variable '" + name + "' is of device-only type '" + type + "' but is located on the host");
        }
        definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
    }
    if (loc & VarLocation::DEVICE) {
        definitionsInternal << "EXPORT_VAR cl::Buffer d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genVariableImplementation(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc) const
{
    if (loc & VarLocation::HOST) {
        os << type << " " << name << ";" << std::endl;
    }
    if (loc & VarLocation::DEVICE) {
        os << "cl::Buffer d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
MemAlloc Backend::genVariableAllocation(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc, size_t count) const
{
    auto allocation = MemAlloc::zero();

    if (loc & VarLocation::HOST) {
        os << name << " = new " << type << "[" << count << "];" << std::endl;
        allocation += MemAlloc::host(count * getSize(type));
    }

    // If variable is present on device then initialize the device buffer
    if (loc & VarLocation::DEVICE) {
        os << "d_" << name << " = cl::Buffer(clContext, CL_MEM_READ_WRITE, " << count << " * sizeof(" << type << "), ";
        if(loc & VarLocation::HOST) {
            os << name << ");" << std::endl;
        }
        else {
            os << "nullptr);" << std::endl;
        }
        
        allocation += MemAlloc::device(count * getSize(type));
    }

    return allocation;
}
//--------------------------------------------------------------------------
void Backend::genVariableFree(CodeStream& os, const std::string& name, VarLocation loc) const
{
    if (loc & VarLocation::HOST) {
        os << "delete[] " << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamDefinition(CodeStream& definitions, const std::string& type, const std::string& name, VarLocation loc) const
{
    if (loc & VarLocation::HOST) {
        definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
    }
    if (loc & VarLocation::DEVICE && ::Utils::isTypePointer(type)) {
        definitions << "EXPORT_VAR " << type << " d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamImplementation(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc) const
{
    if (loc & VarLocation::HOST) {
        os << type << " " << name << ";" << std::endl;
    }
    if (loc & VarLocation::DEVICE && ::Utils::isTypePointer(type)) {
        os << type << " d_" << name << ";" << std::endl;
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
    const std::string devicePointer = pointerToPointer ? ("*" + prefix + "d_" + name) : (prefix + "d_" + name);

    if(loc & VarLocation::HOST) {
        os << hostPointer << " = new " << underlyingType << "[" << countVarName << "];" << std::endl;
    }

    // If variable is present on device at all
    if(loc & VarLocation::DEVICE) {
        os << devicePointer << " = cl::Buffer(clContext, CL_MEM_READ_WRITE, " << countVarName << " * sizeof(" << underlyingType << "), ";
    }

    /*if (loc & VarLocation::HOST) {
        os << name << " = (" << underlyingType << "*)calloc(count, sizeof(" << underlyingType << "));" << std::endl;
    }

    // If variable is present on device at all
    if (loc & VarLocation::DEVICE) {
        os << getVarPrefix() << name << " = (" << underlyingType << "*)calloc(count, sizeof(" << underlyingType << "));" << std::endl;
    }*/
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name,
                                      VarLocation loc, const std::string &countVarName, const std::string &prefix) const
{
    if (!(loc & VarLocation::ZERO_COPY)) {
        throw Utils::ToBeImplemented("genExtraGlobalParamPush");
        //! TO BE REVIEWED - No need to push
    }
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPull(CodeStream &os, const std::string &type, const std::string &name,
                                      VarLocation loc, const std::string &countVarName, const std::string &prefix) const
{
    if (!(loc & VarLocation::ZERO_COPY)) {
        throw Utils::ToBeImplemented("genExtraGlobalParamPull");
    }
}
//--------------------------------------------------------------------------
void Backend::genMergedGroupImplementation(CodeStream &os, const std::string &, const std::string &suffix,
                                           size_t idx, size_t numGroups) const
{
    os << "cl::Buffer d_merged" << suffix << "Group" << idx << ";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMergedGroupPush(CodeStream &os, const std::string &suffix, size_t idx, size_t numGroups) const
{
    os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_merged" << suffix << "Group" << idx;
    os << ", " << "CL_TRUE";
    os << ", " << "0";
    os << ", " << numGroups << " * sizeof(Merged" << suffix << "Group" << idx << ")";
    os << ", group));" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMergedExtraGlobalParamPush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx,
                                            const std::string &groupIdx, const std::string &fieldName,
                                            const std::string &egpName) const
{
    const std::string structName = "Merged" + suffix + "Group" + std::to_string(mergedGroupIdx);
    os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(dd_merged" << suffix << "Group" << mergedGroupIdx;
    os << ", " << "CL_FALSE";
    os << ", " << "(sizeof(" << structName << ") * (" << groupIdx << ")) + offsetof(" << structName << ", " << fieldName << ")";
    os << ", " << "sizeof(" << egpName << ")";
    os << ", &egpName));" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genPopVariableInit(CodeStream& os, const Substitutions& kernelSubs, Handler handler) const
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
void Backend::genSynapseVariableRowInit(CodeStream &os, const SynapseGroupMergedBase &,
                                        const Substitutions &kernelSubs, Handler handler) const
{
    // Pre and postsynaptic ID should already be provided via parallelism
    assert(kernelSubs.hasVarSubstitution("id_pre"));
    assert(kernelSubs.hasVarSubstitution("id_post"));

    Substitutions varSubs(&kernelSubs);
    varSubs.addVarSubstitution("id_syn", "(" + kernelSubs["id_pre"] + " * group->rowStride) + " + kernelSubs["id"]);
    handler(os, varSubs);
}
//--------------------------------------------------------------------------
void Backend::genVariablePush(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc, bool autoInitialized, size_t count) const
{
    if (!(loc & VarLocation::ZERO_COPY)) {
        // Only copy if uninitialisedOnly isn't set
        if (autoInitialized) {
            os << "if(!uninitialisedOnly)" << CodeStream::OB(1101);
        }

        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_" << name;
        os << ", " << "CL_TRUE";
        os << ", " << "0";
        os << ", " << count << " * sizeof(" << type << ")";
        os << ", " << name << "));" << std::endl;

        if (autoInitialized) {
            os << CodeStream::CB(1101);
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genVariablePull(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc, size_t count) const
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
void Backend::genCurrentVariablePush(CodeStream& os, const NeuronGroupInternal& ng, const std::string& type, const std::string& name, VarLocation loc) const
{
    // If this variable requires queuing and isn't zero-copy
    if (ng.isVarQueueRequired(name) && ng.isDelayRequired() && !(loc & VarLocation::ZERO_COPY)) {
        // Generate memcpy to copy only current timestep's data
        //! TO BE IMPLEMENTED - Current push not applicable for OpenCL
        /*
        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(" << getVarPrefix() << name << ng.getName();
        os << "[spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << "]";
        os << ", " << "CL_TRUE";
        os << ", " << "0";
        os << ", " << ng.getNumNeurons() << " * sizeof(" << type << ")";
        os << ", &" << name << ng.getName() << "[spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << "]));" << std::endl;
        */
        genVariablePush(os, type, name + ng.getName(), loc, false, ng.getNumNeurons());
    }
    // Otherwise, generate standard push
    else {
        genVariablePush(os, type, name + ng.getName(), loc, false, ng.getNumNeurons());
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePull(CodeStream& os, const NeuronGroupInternal& ng, const std::string& type, const std::string& name, VarLocation loc) const
{
    // If this variable requires queuing and isn't zero-copy
    if (ng.isVarQueueRequired(name) && ng.isDelayRequired() && !(loc & VarLocation::ZERO_COPY)) {
        // Generate memcpy to copy only current timestep's data
        //! TO BE IMPLEMENTED - Current pull not applicable for OpenCL
        /*
        os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(" << getVarPrefix() << name << ng.getName();
        os << "[spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << "]";
        os << ", " << "CL_TRUE";
        os << ", " << "0";
        os << ", " << ng.getNumNeurons() << " * sizeof(" << type << ")";
        os << ", &" << name << ng.getName() << "[spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << "]));" << std::endl;
        */
        genVariablePull(os, type, name + ng.getName(), loc, ng.getNumNeurons());
    }
    // Otherwise, generate standard push
    else {
        genVariablePull(os, type, name + ng.getName(), loc, ng.getNumNeurons());
    }
}
//--------------------------------------------------------------------------
MemAlloc Backend::genGlobalDeviceRNG(CodeStream &, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &, CodeStream &) const
{
    throw Utils::ToBeImplemented("genGlobalDeviceRNG");
    return MemAlloc::zero();
}
//--------------------------------------------------------------------------
MemAlloc Backend::genPopulationRNG(CodeStream& definitions, CodeStream& definitionsInternal, CodeStream& runner, CodeStream& allocations, CodeStream& free,
    const std::string& name, size_t count) const
{
    genVariableDefinition(definitionsInternal, definitionsInternal, "clrngLfsr113Stream*", name, VarLocation::HOST_DEVICE);
    genVariableImplementation(runner, "clrngLfsr113Stream*", name, VarLocation::HOST_DEVICE);
    genVariableFree(free, name, VarLocation::HOST_DEVICE);

    // genVariableAllocation
    auto allocation = MemAlloc::zero();

    allocations << name << " = clrngLfsr113CreateStreams(NULL, " << count << ", &" << name << "Count, NULL);" << std::endl;
    allocations << "d_" << name << " = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, " << count << ", " << name << ");" << std::endl;

    return allocation;
}
//--------------------------------------------------------------------------
void Backend::genTimer(CodeStream&, CodeStream& definitionsInternal, CodeStream& runner, CodeStream& allocations, CodeStream& free,
    CodeStream& stepTimeFinalise, const std::string& name, bool updateInStepTime) const
{
    throw Utils::ToBeImplemented("genTimer");
}
//--------------------------------------------------------------------------
void Backend::genReturnFreeDeviceMemoryBytes(CodeStream &os) const
{
    // **NOTE** OpenCL does not have this functionality
    os << "return 0;" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefilePreamble(std::ostream& os) const
{
    os << "LIBS := " << "-lOpenCL" << std::endl;
    os << "INCL := " << "-I$(OPENCL_PATH)/include" << std::endl;
    os << "LINKFLAGS := " << "-shared" << std::endl;
    os << "CXXFLAGS := " << "-c -fPIC -std=c++11 -MMD -MP" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileLinkRule(std::ostream& os) const
{
    os << "\t@$(CXX) $(LINKFLAGS) -o $@ $(OBJECTS) $(LIBS)" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileCompileRule(std::ostream& os) const
{
    os << "%.o: %.cc" << std::endl;
    os << "\t@$(CXX) $(CXXFLAGS) $(INCL) -o $@ $<" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildConfigProperties(std::ostream& os) const
{
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportProps(std::ostream& os) const
{
    os << "\t<ImportGroup Label=\"ExtensionSettings\">" << std::endl;
    os << "\t</ImportGroup>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildItemDefinitions(std::ostream& os) const
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
    os << "\t\t\t<AdditionalIncludeDirectories>..\\clRNG\\include;$(OPENCL_PATH)\\include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>" << std::endl;
    os << "\t\t</ClCompile>" << std::endl;

    // Add item definition for linking
    os << "\t\t<Link>" << std::endl;
    os << "\t\t\t<GenerateDebugInformation>true</GenerateDebugInformation>" << std::endl;
    os << "\t\t\t<EnableCOMDATFolding Condition=\"'$(Configuration)'=='Release'\">true</EnableCOMDATFolding>" << std::endl;
    os << "\t\t\t<OptimizeReferences Condition=\"'$(Configuration)'=='Release'\">true</OptimizeReferences>" << std::endl;
    os << "\t\t\t<SubSystem>Console</SubSystem>" << std::endl;
    os << "\t\t\t<AdditionalLibraryDirectories>..\\clRNG\\lib;$(OPENCL_PATH)\\lib\\x64;$(OPENCL_PATH)\\lib\\x86_64;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>" << std::endl;
    os << "\t\t\t<AdditionalDependencies>clRNG.lib;OpenCL.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>" << std::endl;
    os << "\t\t</Link>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildCompileModule(const std::string& moduleName, std::ostream& os) const
{
    os << "\t\t<ClCompile Include=\"" << moduleName << ".cc\" />" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportTarget(std::ostream&) const
{
}
//--------------------------------------------------------------------------
std::string Backend::getFloatAtomicAdd(const std::string& ftype, const char* memoryType) const
{
    if (ftype == "float" || ftype == "double") {
        return "atomic_add_f_" + std::string(memoryType);
    }
    else {
        return "atomic_add";
    }
}
//--------------------------------------------------------------------------
size_t Backend::getNumPresynapticUpdateThreads(const SynapseGroupInternal &sg)
{
     return getPresynapticUpdateStrategy(sg)->getNumThreads(sg);
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
        return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * sg.getMaxConnections();
    }
    else {
        return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//--------------------------------------------------------------------------
void Backend::addPresynapticUpdateStrategy(PresynapticUpdateStrategy::Base *strategy)
{
    s_PresynapticUpdateStrategies.push_back(strategy);
}
//--------------------------------------------------------------------------
bool Backend::isGlobalHostRNGRequired(const ModelSpecMerged &modelMerged) const
{
    // Host RNG is required if any synapse groups require a host initialization RNG
    const ModelSpecInternal &model = modelMerged.getModel();
    return std::any_of(model.getSynapseGroups().cbegin(), model.getSynapseGroups().cend(),
                       [](const ModelSpec::SynapseGroupValueType &s)
    {
        return (s.second.isHostInitRNGRequired());
    });
}
//--------------------------------------------------------------------------
bool Backend::isGlobalDeviceRNGRequired(const ModelSpecMerged &modelMerged) const
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
Backend::MemorySpaces Backend::getMergedGroupMemorySpaces(const ModelSpecMerged &) const
{
    return {{"", getDeviceMemoryBytes() }};
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikePush(CodeStream& os, const NeuronGroupInternal& ng, bool spikeEvent) const
{
    if (!(ng.getSpikeLocation() & VarLocation::ZERO_COPY)) {
        // Is delay required
        const bool delayRequired = spikeEvent ?
            ng.isDelayRequired() :
            (ng.isTrueSpikeRequired() && ng.isDelayRequired());

        const char* spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
        const char* spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

        if (delayRequired) {
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_" << spikeCntPrefix << ng.getName();
            os << ", " << "CL_TRUE";
            os << ", " << "0";
            os << ", sizeof(unsigned int)";
            os << ", " << spikeCntPrefix << ng.getName() << "));" << std::endl;

            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_" << spikePrefix << ng.getName();
            os << ", " << "CL_TRUE";
            os << ", " << "0";
            os << ", " << ng.getNumNeurons() << " * sizeof(unsigned int)";
            os << ", " << spikePrefix << ng.getName() << "));" << std::endl;
        }
        else {
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_" << spikeCntPrefix << ng.getName();
            os << ", " << "CL_TRUE";
            os << ", " << "0";
            os << ", sizeof(unsigned int)";
            os << ", " << spikeCntPrefix << ng.getName() << "));" << std::endl;

            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueWriteBuffer(d_" << spikePrefix << ng.getName();
            os << ", " << "CL_TRUE";
            os << ", " << "0";
            os << ", " << spikeCntPrefix << ng.getName() << "[0] * sizeof(unsigned int)";
            os << ", " << spikePrefix << ng.getName() << "));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikePull(CodeStream& os, const NeuronGroupInternal& ng, bool spikeEvent) const
{
    if (!(ng.getSpikeLocation() & VarLocation::ZERO_COPY)) {
        // Is delay required
        const bool delayRequired = spikeEvent ?
            ng.isDelayRequired() :
            (ng.isTrueSpikeRequired() && ng.isDelayRequired());

        const char* spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
        const char* spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

        if (delayRequired) {
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_" << spikeCntPrefix << ng.getName();
            os << ", " << "CL_TRUE";
            os << ", " << "0";
            os << ", sizeof(unsigned int)";
            os << ", " << spikeCntPrefix << ng.getName() << "));" << std::endl;

            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_" << spikePrefix << ng.getName();
            os << ", " << "CL_TRUE";
            os << ", " << "0";
            os << ", " << ng.getNumNeurons() << " * sizeof(unsigned int)";
            os << ", " << spikePrefix << ng.getName() << "));" << std::endl;
        }
        else {
            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_" << spikeCntPrefix << ng.getName();
            os << ", " << "CL_TRUE";
            os << ", " << "0";
            os << ", sizeof(unsigned int)";
            os << ", " << spikeCntPrefix << ng.getName() << "));" << std::endl;

            os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueReadBuffer(d_" << spikePrefix << ng.getName();
            os << ", " << "CL_TRUE";
            os << ", " << "0";
            os << ", " << spikeCntPrefix << ng.getName() << "[0] * sizeof(unsigned int)";
            os << ", " << spikePrefix << ng.getName() << "));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genEmitSpike(CodeStream& os, const Substitutions& subs, const std::string& suffix) const
{
    os << "const unsigned int spk" << suffix << "Idx = atomic_add(&shSpk" << suffix << "Count, 1);" << std::endl;
    os << "shSpk" << suffix << "[spk" << suffix << "Idx] = " << subs["id"] << ";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genKernelDimensions(CodeStream& os, Kernel kernel, size_t numThreads) const
{
    // Calculate global and local work size
    const size_t numOfWorkGroups = ceilDivide(numThreads, m_KernelWorkGroupSizes[kernel]);
    os << "const cl::NDRange globalWorkSize(" << (m_KernelWorkGroupSizes[kernel] * numOfWorkGroups) << ", 1);" << std::endl;
    os << "const cl::NDRange localWorkSize(" << m_KernelWorkGroupSizes[kernel] << ", 1);" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::addDeviceType(const std::string& type, size_t size)
{
    addType(type, size);
    m_DeviceTypes.emplace(type);
}
//--------------------------------------------------------------------------
bool Backend::isDeviceType(const std::string& type) const
{
    // Get underlying type
    const std::string underlyingType = ::Utils::isTypePointer(type) ? ::Utils::getUnderlyingType(type) : type;

    // Return true if it is in device types set
    return (m_DeviceTypes.find(underlyingType) != m_DeviceTypes.cend());
}
//--------------------------------------------------------------------------
void Backend::divideKernelStreamInParts(CodeStream& os, const std::stringstream& kernelCode, size_t partLength) const
{
    const std::string kernelStr = kernelCode.str();
    const size_t parts = ceilDivide(kernelStr.length(), partLength);
    for(size_t i = 0; i < parts; i++) {
        os << "R\"(" << kernelStr.substr(i * partLength, partLength) << ")\"" << std::endl;
    }
}
//--------------------------------------------------------------------------
const PresynapticUpdateStrategy::Base* Backend::getPresynapticUpdateStrategy(const SynapseGroupInternal& sg)
{
    // Loop through presynaptic update strategies until we find one that is compatible with this synapse group
    // **NOTE** this is done backwards so that user-registered strategies get first priority
    for (auto s = s_PresynapticUpdateStrategies.rbegin(); s != s_PresynapticUpdateStrategies.rend(); ++s) {
        if ((*s)->isCompatible(sg)) {
            return *s;
        }
    }

    throw std::runtime_error("Unable to find a suitable presynaptic update strategy for synapse group '" + sg.getName() + "'");
    return nullptr;
}
} // namespace OpenCL
} // namespace CodeGenerator

#include "runtime/runtime.h"

// Standard C++ includes
#include <fstream>
#include <unordered_set>

// PLOG includes
#include <plog/Log.h>

// Filesystem includes
#include "path.h"

// GeNN includes
#include "varAccess.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/modelSpecMerged.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
size_t getNumSynapseVarElements(VarAccessDim varDims, const BackendBase &backend, const SynapseGroupInternal &sg)
{
    if(varDims & VarAccessDim::ELEMENT) {
        if(sg.getMatrixType() & SynapseMatrixWeight::KERNEL) {
            return sg.getKernelSizeFlattened();
        }
        else {
            return sg.getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(sg);
        }
    }
    else {
        return 1;
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// GeNN::Runtime::ArrayBase
//--------------------------------------------------------------------------
namespace GeNN::Runtime
{
void ArrayBase::memsetHostPointer(int value)
{
    std::memset(m_HostPointer, value, getSizeBytes());
}
//--------------------------------------------------------------------------
void ArrayBase::serialiseHostPointer(std::vector<std::byte> &bytes, bool pointerToPointer) const
{
    std::byte vBytes[sizeof(void*)];
    if(pointerToPointer) {
        std::byte* const *hostPointerPointer = &m_HostPointer;
        std::memcpy(vBytes, &hostPointerPointer, sizeof(void*));
    }
    else {
        std::memcpy(vBytes, &m_HostPointer, sizeof(void*));
    }
    std::copy(std::begin(vBytes), std::end(vBytes), std::back_inserter(bytes));
}

//--------------------------------------------------------------------------
// GeNN::Runtime::Runtime
//--------------------------------------------------------------------------
Runtime::Runtime(const filesystem::path &modelPath, const CodeGenerator::ModelSpecMerged &modelMerged, 
                 const CodeGenerator::BackendBase &backend)
:   m_Timestep(0), m_ModelMerged(modelMerged), m_Backend(backend), m_AllocateMem(nullptr), m_FreeMem(nullptr),
    m_Initialize(nullptr), m_InitializeSparse(nullptr), m_InitializeHost(nullptr), m_StepTime(nullptr)
{

    // Load library
#ifdef _WIN32
    const std::string runnerName = "runner_" + modelMerged.getModel().getName();
    const std::string runnerNameSuffix = backend.getPreferences().debugCode ?  "_Debug.dll" :  "_Release.dll";
    const std::string libraryName = (modelPath / (runnerName + runnerNameSuffix)).str();
    m_Library = LoadLibrary(libraryName.c_str());
#else
    const std::string libraryName = (modelPath / (modelMerged.getModel().getName() + "_CODE") / "librunner.so").str();
    m_Library = dlopen(libraryName.c_str(), RTLD_NOW);
#endif

    // If library was loaded successfully, look up basic functions in library
    if(m_Library != nullptr) {
        m_AllocateMem = (VoidFunction)getSymbol("allocateMem");
        m_FreeMem = (VoidFunction)getSymbol("freeMem");

        m_Initialize = (VoidFunction)getSymbol("initialize");
        m_InitializeSparse = (VoidFunction)getSymbol("initializeSparse");
        m_InitializeHost = (VoidFunction)getSymbol("initializeHost");

        m_StepTime = (StepTimeFunction)getSymbol("stepTime");

        /*m_NCCLGenerateUniqueID = (VoidFunction)getSymbol("ncclGenerateUniqueID", true);
        m_NCCLGetUniqueID = (UCharPtrFunction)getSymbol("ncclGetUniqueID", true);
        m_NCCLInitCommunicator = (NCCLInitCommunicatorFunction)getSymbol("ncclInitCommunicator", true);
        m_NCCLUniqueIDBytes = (unsigned int*)getSymbol("ncclUniqueIDBytes", true);*/

        // Build set of custom update group names
        std::unordered_set<std::string> customUpdateGroupNames;
        std::transform(getModel().getCustomUpdates().cbegin(), getModel().getCustomUpdates().cend(),
                       std::inserter(customUpdateGroupNames, customUpdateGroupNames.end()),
                       [](const auto &v) { return v.second.getUpdateGroupName(); });
        std::transform(getModel().getCustomWUUpdates().cbegin(), getModel().getCustomWUUpdates().cend(),
                       std::inserter(customUpdateGroupNames, customUpdateGroupNames.end()),
                       [](const auto &v) { return v.second.getUpdateGroupName(); });
        std::transform(getModel().getCustomConnectivityUpdates().cbegin(), getModel().getCustomConnectivityUpdates().cend(),
                       std::inserter(customUpdateGroupNames, customUpdateGroupNames.end()),
                       [](const auto &v) { return v.second.getUpdateGroupName(); });

        // Get function pointers to custom update functions for each group
        std::transform(customUpdateGroupNames.cbegin(), customUpdateGroupNames.cend(), 
                       std::inserter(m_CustomUpdateFunctions, m_CustomUpdateFunctions.end()),
                       [this](const auto &n)
                       { 
                           return std::make_pair(n, (CustomUpdateFunction)getSymbol("update" + n)); 
                       });

        // Create  state
        m_State = backend.createState(*this);
    }
    else {
#ifdef _WIN32
        throw std::runtime_error("Unable to load library - error:" + std::to_string(GetLastError()));
#else
        throw std::runtime_error("Unable to load library - error:" + std::string(dlerror()));
#endif
    }
}
//----------------------------------------------------------------------------
Runtime::~Runtime()
{
    if(m_Library) {
        m_FreeMem();

#ifdef _WIN32
        FreeLibrary(m_Library);
#else
        dlclose(m_Library);
#endif
        m_Library = nullptr;
    }
}
//----------------------------------------------------------------------------
void Runtime::allocate(std::optional<size_t> numRecordingTimesteps)
{
    // Call allocate function in generated code
    m_AllocateMem();

    // Store number of recording timesteps
    m_NumRecordingTimesteps = numRecordingTimesteps;

    // Loop through neuron groups
    const size_t batchSize = getModel().getBatchSize();
    for(const auto &n : getModel().getNeuronGroups()) {
        LOGD_RUNTIME << "Allocating memory for neuron group '" << n.first << "'";
        const size_t neuronStride = m_Backend.get().getNeuronStride(n.second);
        const size_t nonDelayedNeuronVarSize = batchSize * neuronStride;
        const size_t delayedNeuronVarSize = nonDelayedNeuronVarSize * n.second.getNumDelaySlots();

        // If spike or spike-like event recording is enabled
        if(n.second.isSpikeRecordingEnabled() || n.second.isSpikeEventRecordingEnabled()) {
            if(!numRecordingTimesteps) {
                throw std::runtime_error("Cannot use recording system without specifying number of recording timesteps");
            }

            if(n.second.isSpikeRecordingEnabled()) {
                const size_t numRecordingWords = (ceilDivide(n.second.getNumNeurons(), 32) * batchSize) * numRecordingTimesteps.value();
                createArray(&n.second, "recordSpk", Type::Uint32, numRecordingWords,
                            n.second.isRecordingZeroCopyEnabled() ? VarLocation::HOST_DEVICE_ZERO_COPY : VarLocation::HOST_DEVICE);
            }
        }

        // If neuron group has axonal or back-propagation delays, add delay queue pointer
        if (n.second.isDelayRequired()) {
            createArray(&n.second, "spkQuePtr", Type::Uint32, 1, VarLocation::DEVICE);
            m_DelayQueuePointer.try_emplace(&n.second, 0);
        }

        // If neuron group needs per-neuron RNGs
        // **NOTE** if SIMT backend vectorises, this is excessive
        if(n.second.isSimRNGRequired()) {
            auto rng = m_Backend.get().createPopulationRNG(nonDelayedNeuronVarSize);
            if(rng) {
                const auto r = m_NeuronGroupArrays[&n.second].try_emplace("rng", std::move(rng));
                if(!r.second) {
                    throw std::runtime_error("Unable to allocate array with " 
                                             "duplicate name 'rng'");
                }
            }
        }

        // If neuron group needs to record its spike times
        if (n.second.isSpikeTimeRequired()) {
            createArray(&n.second, "sT", getModel().getTimePrecision(), 
                        n.second.isSpikeQueueRequired() ? delayedNeuronVarSize : nonDelayedNeuronVarSize, 
                        n.second.getSpikeTimeLocation());
        }

        // If neuron group needs to record its previous spike times
        if (n.second.isPrevSpikeTimeRequired()) {
            createArray(&n.second, "prevST", getModel().getTimePrecision(),
                        n.second.isSpikeQueueRequired() ? delayedNeuronVarSize : nonDelayedNeuronVarSize, 
                        n.second.getPrevSpikeTimeLocation());
        }

        // Create destinations for any dynamic parameters
        createDynamicParamDestinations<NeuronGroupInternal>(n.second, n.second.getModel()->getParams(),
                                                            &NeuronGroup::isParamDynamic);

        // Create arrays for neuron state variables
        createNeuronVarArrays<NeuronVarAdapter>(&n.second, neuronStride, batchSize, true);
        
        // Create arrays for neuron extra global parameters
        createEGPArrays<NeuronEGPAdapter>(&n.second);

        // Create arrays for current source variables and extra global parameters
        for (const auto *cs : n.second.getCurrentSources()) {
            LOGD_RUNTIME << "\tChild current source '" << cs->getName() << "'";
            createNeuronVarArrays<CurrentSourceVarAdapter>(cs, neuronStride, batchSize, true);
            createEGPArrays<CurrentSourceEGPAdapter>(cs);
            createDynamicParamDestinations<CurrentSourceInternal>(*cs, cs->getModel()->getParams(),
                                                                  &CurrentSourceInternal::isParamDynamic, 2);
        }

        // Loop through fused postsynaptic model from incoming populations
        for(const auto *sg : n.second.getFusedPSMInSyn()) {
            LOGD_RUNTIME << "\tFused PSM incoming synapse group '" << sg->getName() << "'";
            createArray(sg, "outPost", getModel().getPrecision(), 
                        nonDelayedNeuronVarSize,
                        sg->getOutputLocation(), false, 2);
            
            if (sg->isDendriticOutputDelayRequired()) {
                createArray(sg, "denDelay", getModel().getPrecision(), 
                            (size_t)sg->getMaxDendriticDelayTimesteps() * nonDelayedNeuronVarSize,
                            sg->getDendriticDelayLocation(), false, 2);
                createArray(sg, "denDelayPtr", Type::Uint32, 1, VarLocation::DEVICE, false, 2);
            }

            // Create arrays for postsynaptic model state variables
            createNeuronVarArrays<SynapsePSMVarAdapter>(sg, neuronStride, batchSize, true);
        }

        // Create arrays for fused pre-output variables
        for(const auto *sg : n.second.getFusedPreOutputOutSyn()) {
            LOGD_RUNTIME << "\tFused pre-output outgoing synapse group '" << sg->getName() << "'";
            createArray(sg, "outPre", getModel().getPrecision(), 
                        nonDelayedNeuronVarSize,
                        sg->getOutputLocation(), false, 2);
        }
        
        // Create arrays for variables from fused incoming synaptic populations
        for(const auto *sg: n.second.getFusedWUPreOutSyn()) {
            LOGD_RUNTIME << "\tFused WU pre incoming synapse group '" << sg->getName() << "'";
            createNeuronVarArrays<SynapseWUPreVarAdapter>(sg, neuronStride, batchSize, true, 2);
        }
        
        // Create arrays for variables from fused outgoing synaptic populations
        for(const auto *sg: n.second.getFusedWUPostInSyn()) {
            LOGD_RUNTIME << "\tFused WU post outgoing synapse group '" << sg->getName() << "'";
            createNeuronVarArrays<SynapseWUPostVarAdapter>(sg, neuronStride, batchSize, true, 2);
        }

        // Create arrays for spikes
        for(const auto *sg: n.second.getFusedSpike()) {
            LOGD_RUNTIME << "\tFused spike '" << sg->getName() << "'";

            // Prefix array names depending on whether neuron group is source or target of merged synapse group
            const std::string prefix = (&n.second == sg->getSrcNeuronGroup()) ? "src" : "trg";
            createArray(sg, prefix + "SpkCnt", Type::Uint32,
                        n.second.isSpikeQueueRequired() ? batchSize * n.second.getNumDelaySlots() : batchSize,
                        n.second.getSpikeLocation(), false, 2);
            createArray(sg, prefix + "Spk", Type::Uint32, 
                        n.second.isSpikeQueueRequired() ? delayedNeuronVarSize : nonDelayedNeuronVarSize,
                        n.second.getSpikeLocation(), false, 2);
        }

        // Create arrays for spike events
        for(const auto *sg: n.second.getFusedSpikeEvent()) {
            LOGD_RUNTIME << "\tFused spike event '" << sg->getName() << "'";

            // Prefix array names depending on whether neuron group is source or target of merged synapse group
            const std::string prefix = (&n.second == sg->getSrcNeuronGroup()) ? "src" : "trg";
            createArray(sg, prefix + "SpkCntEvent", Type::Uint32, 
                        n.second.isSpikeEventQueueRequired() ? batchSize * n.second.getNumDelaySlots() : batchSize,
                        n.second.getSpikeEventLocation(), false, 2);
            createArray(sg, prefix + "SpkEvent", Type::Uint32, 
                        n.second.isSpikeEventQueueRequired() ? delayedNeuronVarSize : nonDelayedNeuronVarSize,
                        n.second.getSpikeEventLocation(), false, 2);

            // If neuron group needs to record its spike-like-event times
            if (n.second.isSpikeEventTimeRequired()) {
                createArray(sg, prefix + "SET", getModel().getTimePrecision(),
                            n.second.isSpikeEventQueueRequired() ? delayedNeuronVarSize : nonDelayedNeuronVarSize,
                            n.second.getSpikeEventTimeLocation(), false, 2);
            }

            // If neuron group needs to record its previous spike-like-event times
            if (n.second.isPrevSpikeEventTimeRequired()) {
                createArray(sg, prefix + "PrevSET", getModel().getTimePrecision(),
                            n.second.isSpikeEventQueueRequired() ? delayedNeuronVarSize : nonDelayedNeuronVarSize,
                            n.second.getPrevSpikeEventTimeLocation(), false, 2);
            }

            if(n.second.isSpikeEventRecordingEnabled()) {
                const size_t numRecordingWords = (ceilDivide(n.second.getNumNeurons(), 32) * batchSize) * numRecordingTimesteps.value();
                createArray(sg, prefix + "RecordSpkEvent", Type::Uint32, numRecordingWords, 
                            n.second.isRecordingZeroCopyEnabled() ? VarLocation::HOST_DEVICE_ZERO_COPY : VarLocation::HOST_DEVICE, false, 2);
            }
        }
    }

    // Loop through synapse groups
    for(const auto &s : getModel().getSynapseGroups()) {
        // If synapse group has individual or kernel weights
        LOGD_RUNTIME << "Allocating memory for synapse group '" << s.first << "'";
        const bool individualWeights = (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);
        const bool kernelWeights = (s.second.getMatrixType() & SynapseMatrixWeight::KERNEL);
        if (individualWeights || kernelWeights) {
            createVarArrays<SynapseWUVarAdapter>(
                &s.second, batchSize, true, 
                [&s, this](const std::string&, VarAccessDim varDims)
                {
                    return getNumSynapseVarElements(varDims, m_Backend.get(), s.second);
                });
        }

        // Create destinations for any dynamic parameters
        createDynamicParamDestinations<SynapseGroupInternal>(s.second, s.second.getWUInitialiser().getSnippet()->getParams(),
                                                            &SynapseGroupInternal::isWUParamDynamic);
        createDynamicParamDestinations<SynapseGroupInternal>(s.second, s.second.getPSInitialiser().getSnippet()->getParams(),
                                                            &SynapseGroupInternal::isPSParamDynamic);

        // If connectivity is bitmask
        const size_t numPre = s.second.getSrcNeuronGroup()->getNumNeurons();
        const size_t rowStride = m_Backend.get().getSynapticMatrixRowStride(s.second);
        const auto &connectInit = s.second.getSparseConnectivityInitialiser();
        const bool uninitialized = (Utils::areTokensEmpty(connectInit.getRowBuildCodeTokens()) 
                                    && Utils::areTokensEmpty(connectInit.getColBuildCodeTokens()));

        if(s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            const size_t gpSize = ceilDivide((size_t)numPre * rowStride, 32);
            createArray(&s.second, "gp", Type::Uint32, gpSize,
                        s.second.getSparseConnectivityLocation(), uninitialized);
            
            // If this isn't uninitialised i.e. it will be 
            // initialised using initialization kernel, zero bitmask
            if(!uninitialized) {
                if(m_Backend.get().isArrayDeviceObjectRequired()) {
                    getArray(s.second, "gp")->memsetDeviceObject(0);
                }
                else {
                    getArray(s.second, "gp")->memsetHostPointer(0);
                }
            }
        }
        // Otherwise, if connectivity is sparse
        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            // Row lengths
            createArray(&s.second, "rowLength", Type::Uint32, numPre,
                        s.second.getSparseConnectivityLocation(), uninitialized);

            // Target indices
            createArray(&s.second, "ind", s.second.getSparseIndType(), numPre * rowStride,
                        s.second.getSparseConnectivityLocation(), uninitialized);
            
            // If this isn't uninitialised i.e. it will be 
            // initialised using initialization kernel, zero row length
            if(!uninitialized) {
                LOGD_RUNTIME << "\tZeroing 'rowLength'";
                if(m_Backend.get().isArrayDeviceObjectRequired()) {
                    getArray(s.second, "rowLength")->memsetDeviceObject(0);
                }
                else {
                    getArray(s.second, "rowLength")->memsetHostPointer(0);
                }
            }

            // **TODO** remap is not always required
            if(m_Backend.get().isPostsynapticRemapRequired() 
               && (s.second.isPostSpikeRequired() || s.second.isPostSpikeEventRequired())) 
            {
                // Create column lengths array
                const size_t numPost = s.second.getTrgNeuronGroup()->getNumNeurons();
                const size_t colStride = s.second.getMaxSourceConnections();
                createArray(&s.second, "colLength", Type::Uint32, numPost, VarLocation::DEVICE);
                
                // Create remap array
                createArray(&s.second, "remap", Type::Uint32, numPost * colStride, VarLocation::DEVICE);

                // Zero column length array
                LOGD_RUNTIME << "\tZeroing 'colLength'";
                if(m_Backend.get().isArrayDeviceObjectRequired()) {
                    getArray(s.second, "colLength")->memsetDeviceObject(0);
                }
                else {
                    getArray(s.second, "colLength")->memsetHostPointer(0);
                }
            }
        }

        // Loop through sparse connectivity initialiser EGPs
        // **THINK** should any of these have locations? if they're not initialised in host code not much scope to do so
        const auto &sparseConnectInit = s.second.getSparseConnectivityInitialiser();
        for(const auto &egp : sparseConnectInit.getSnippet()->getExtraGlobalParams()) {
            const auto resolvedEGPType = egp.type.resolve(getModel().getTypeContext());
            createArray(&s.second, egp.name + "SparseConnect", resolvedEGPType, 0, VarLocation::HOST_DEVICE);
        }

        // Loop through toeplitz connectivity initialiser EGPs        
        const auto &toeplitzConnectInit = s.second.getToeplitzConnectivityInitialiser();
        for(const auto &egp : toeplitzConnectInit.getSnippet()->getExtraGlobalParams()) {
            const auto resolvedEGPType = egp.type.resolve(getModel().getTypeContext());
            createArray(&s.second, egp.name + "ToeplitzConnect", resolvedEGPType, 0, VarLocation::HOST_DEVICE);
        }

        // Create arrays for extra-global parameters
        // **NOTE** postsynaptic models with EGPs can't be fused so no need to worry about that
        createEGPArrays<SynapseWUEGPAdapter>(&s.second);
        createEGPArrays<SynapsePSMEGPAdapter>(&s.second);
    }

    // Allocate custom update variables
    for(const auto &c : getModel().getCustomUpdates()) {
        LOGD_RUNTIME << "Allocating memory for custom update '" << c.first << "'";
        createNeuronVarArrays<CustomUpdateVarAdapter>(&c.second, c.second.getNumNeurons(), batchSize, 
                                                      c.second.getDims() & VarAccessDim::BATCH);
        // Create arrays for custom update extra global parameters
        createEGPArrays<CustomUpdateEGPAdapter>(&c.second);

        createDynamicParamDestinations<CustomUpdateInternal>(c.second, c.second.getModel()->getParams(),
                                                            &CustomUpdateInternal::isParamDynamic);
    }

    // Allocate custom update WU variables
    for(const auto &c : getModel().getCustomWUUpdates()) {
        LOGD_RUNTIME << "Allocating memory for custom WU update '" << c.first << "'";
        createVarArrays<CustomUpdateVarAdapter>(
                &c.second, batchSize, (c.second.getDims() & VarAccessDim::BATCH), 
                [&c, this](const std::string&, VarAccessDim varDims)
                {
                    return getNumSynapseVarElements(varDims, m_Backend.get(), 
                                                    *c.second.getSynapseGroup());
                });
        
        // Create arrays for custom update extra global parameters
        createEGPArrays<CustomUpdateEGPAdapter>(&c.second);

        createDynamicParamDestinations<CustomUpdateWUInternal>(c.second, c.second.getModel()->getParams(),
                                                               &CustomUpdateWUInternal::isParamDynamic);
    }

    // Loop through custom connectivity update variables
    for(const auto &c : getModel().getCustomConnectivityUpdates()) {
        // Allocate presynaptic variables
        LOGD_RUNTIME << "Allocating memory for custom connectivity update '" << c.first << "'";
        const auto* sg = c.second.getSynapseGroup();
        createNeuronVarArrays<CustomConnectivityUpdatePreVarAdapter>(
            &c.second, sg->getSrcNeuronGroup()->getNumNeurons(),
            batchSize, false);
        
        // Allocate postsynaptic variables
        createNeuronVarArrays<CustomConnectivityUpdatePostVarAdapter>(
            &c.second, sg->getTrgNeuronGroup()->getNumNeurons(),
            batchSize, false);

        // Allocate variables
        createVarArrays<CustomConnectivityUpdateVarAdapter>(
                &c.second, batchSize, false, 
                [sg, this](const std::string&, VarAccessDim varDims)
                {
                    return getNumSynapseVarElements(varDims, m_Backend.get(), *sg);
                });
        
        // Create arrays for custom connectivity update extra global parameters
        createEGPArrays<CustomConnectivityUpdateEGPAdapter>(&c.second);

        createDynamicParamDestinations<CustomConnectivityUpdateInternal>(
            c.second, c.second.getModel()->getParams(),
            &CustomConnectivityUpdateInternal::isParamDynamic);

        // If custom connectivity update group needs per-row RNGs
        if(Utils::isRNGRequired(c.second.getRowUpdateCodeTokens())) {
            auto rng = m_Backend.get().createPopulationRNG(sg->getSrcNeuronGroup()->getNumNeurons());
            if(rng) {
                const auto r = m_CustomConnectivityUpdateArrays[&c.second].try_emplace("rowRNG", std::move(rng));
                if(!r.second) {
                    throw std::runtime_error("Unable to allocate array with " 
                                             "duplicate name 'rowRNG'");
                }
            }
        }
    }
    
    // Push merged synapse host connectivity initialisation groups 
    for(const auto &m : m_ModelMerged.get().getMergedSynapseConnectivityHostInitGroups()) {
       pushMergedGroup(m);
    }

    // Perform host initialisation
    m_InitializeHost();

    // Push merged neuron initialisation groups
    for(const auto &m : m_ModelMerged.get().getMergedNeuronInitGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged synapse init groups
    for(const auto &m : m_ModelMerged.get().getMergedSynapseInitGroups()) {
         addMergedArrays(m);
         pushMergedGroup(m);
    }

    // Push merged synapse connectivity initialisation groups
    for(const auto &m : m_ModelMerged.get().getMergedSynapseConnectivityInitGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged sparse synapse init groups
    for(const auto &m : m_ModelMerged.get().getMergedSynapseSparseInitGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged custom update initialisation groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomUpdateInitGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged custom WU update initialisation groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomWUUpdateInitGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged custom sparse WU update initialisation groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomWUUpdateSparseInitGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged custom connectivity update presynaptic initialisation groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomConnectivityUpdatePreInitGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged custom connectivity update postsynaptic initialisation groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomConnectivityUpdatePostInitGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged custom connectivity update synaptic initialisation groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomConnectivityUpdateSparseInitGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged neuron update groups
    for(const auto &m : m_ModelMerged.get().getMergedNeuronUpdateGroups()) {        
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged presynaptic update groups
    for(const auto &m : m_ModelMerged.get().getMergedPresynapticUpdateGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push merged postsynaptic update groups
    for(const auto &m : m_ModelMerged.get().getMergedPostsynapticUpdateGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push synapse dynamics groups
    for(const auto &m : m_ModelMerged.get().getMergedSynapseDynamicsGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push neuron groups whose previous spike times need resetting
    for(const auto &m : m_ModelMerged.get().getMergedNeuronPrevSpikeTimeUpdateGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push neuron groups whose spike queues need resetting
    for(const auto &m : m_ModelMerged.get().getMergedNeuronSpikeQueueUpdateGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push synapse groups whose dendritic delay pointers need updating
    for(const auto &m : m_ModelMerged.get().getMergedSynapseDendriticDelayUpdateGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }
    
    // Push custom variable update groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomUpdateGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push custom WU variable update groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomUpdateWUGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push custom WU transpose variable update groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomUpdateTransposeWUGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push custom update host reduction groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomUpdateHostReductionGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push custom weight update host reduction groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomWUUpdateHostReductionGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push custom connectivity update groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomConnectivityUpdateGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push custom connectivity remap update groups
    for (const auto &m : m_ModelMerged.get().getMergedCustomConnectivityRemapUpdateGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Push custom connectivity host update groups
    for(const auto &m : m_ModelMerged.get().getMergedCustomConnectivityHostUpdateGroups()) {
        addMergedArrays(m);
        pushMergedGroup(m);
    }

    // Loop through merged custom connectivity remap update groups and add 
    // Column length arrays associated with each synapse group to map
    for (const auto &m : m_ModelMerged.get().getMergedCustomConnectivityRemapUpdateGroups()) {
        auto &colLengthArrays = m_CustomUpdateColLengthArrays[m.getArchetype().getUpdateGroupName()];
        std::transform(m.getGroups().cbegin(), m.getGroups().cend(), std::back_inserter(colLengthArrays),
                       [this](auto r){ return getArray(*r.get().getSynapseGroup(), "colLength"); });
    }
}
//----------------------------------------------------------------------------
void Runtime::initialize()
{
    m_Initialize();
}
//----------------------------------------------------------------------------
void Runtime::initializeSparse()
{
    // Push uninitialized arrays to device
    LOGD_RUNTIME << "Pushing uninitialized current source variables";
    pushUninitialized(m_CurrentSourceArrays);
    LOGD_RUNTIME << "Pushing uninitialized neuron group variables";
    pushUninitialized(m_NeuronGroupArrays);
    LOGD_RUNTIME << "Pushing uninitialized synapse group variables";
    pushUninitialized(m_SynapseGroupArrays);
    LOGD_RUNTIME << "Pushing uninitialized custom update variables";
    pushUninitialized(m_CustomUpdateBaseArrays);
    LOGD_RUNTIME << "Pushing uninitialized custom connectivity update variables";
    pushUninitialized(m_CustomConnectivityUpdateArrays);

    m_InitializeSparse();
}
//----------------------------------------------------------------------------
void Runtime::stepTime()
{
   m_StepTime(m_Timestep, m_NumRecordingTimesteps.value_or(0));
    
   // Loop through delay queue pointers and update
   for(auto &d : m_DelayQueuePointer) {
       d.second = (d.second + 1) % d.first->getNumDelaySlots();
   }

    // Advance time
    m_Timestep++;
}
//----------------------------------------------------------------------------
void Runtime::customUpdate(const std::string &name)
{
    // If there are column length arrays that must be zeroed 
    // before making connectivity update in this group
    auto colLengthArrays = m_CustomUpdateColLengthArrays.find(name);
    if(colLengthArrays != m_CustomUpdateColLengthArrays.cend()) {
        // Loop through arrays and zero
        for(auto *a : colLengthArrays->second) {
            if(m_Backend.get().isArrayDeviceObjectRequired()) {
                a->memsetDeviceObject(0);
            }
            else {
                a->memsetHostPointer(0);
            }
        }
    }

    // Run custom update
    m_CustomUpdateFunctions.at(name)(getTimestep());
}
//----------------------------------------------------------------------------
double Runtime::getTime() const
{ 
    return m_Timestep * getModel().getDT();
}
//----------------------------------------------------------------------------
void Runtime::pullRecordingBuffersFromDevice() const
{
    if(!m_NumRecordingTimesteps) {
        throw std::runtime_error("Recording buffer not allocated - cannot pull from device");
    }

    // Loop through neuron groups
    for(const auto &n : getModel().getNeuronGroups()) {
        // If spike recording is enabled, pull array from device
        if(n.second.isSpikeRecordingEnabled()) {
            getArray(n.second, "recordSpk")->pullFromDevice();
        }

        // If spike event recording is enabled, pull array from device
        if(n.second.isSpikeEventRecordingEnabled()) {
            for(const auto *sg : n.second.getFusedSpikeEvent()) {
                const std::string prefix = (&n.second == sg->getSrcNeuronGroup()) ? "src" : "trg";
                getArray(*sg, prefix + "RecordSpkEvent")->pullFromDevice();
            }
        }
    }
}
//----------------------------------------------------------------------------
ArrayBase *Runtime::getFusedEventArray(const CodeGenerator::NeuronGroupMergedBase &ng, size_t i, 
                                       const SynapseGroupInternal &sg, const std::string &name) const
{
    // Get the corresponding merged group in the parent neuron group
    const auto &n = ng.getGroups().at(i);

    // If the neuron group is the SOURCE of synapse group, we should use it's SOURCE prefixed array
    const std::string prefix = (&n.get() == sg.getSrcNeuronGroup()) ? "src" : "trg";
    return getArray(sg, prefix + name);
}
//--------------------------------------------------------------------------
ArrayBase *Runtime::getFusedSrcSpikeArray(const SynapseGroupInternal &g, const std::string &name) const
{
    // Get the synapse group SOURCE spike generation has been fused with
    const auto &f = static_cast<const SynapseGroupInternal&>(g.getFusedSpikeTarget(g.getSrcNeuronGroup()));

    // If the fused target shares a SOURCE neuron with original synapse group, we should use it's SOURCE prefixed array
    const std::string prefix = (g.getSrcNeuronGroup() == f.getSrcNeuronGroup()) ? "src" : "trg";
    return getArray(f, prefix + name); 
}
//--------------------------------------------------------------------------
ArrayBase *Runtime::getFusedTrgSpikeArray(const SynapseGroupInternal &g, const std::string &name) const
{
    // Get the synapse group TARGET spike generation has been fused with
    const auto &f = static_cast<const SynapseGroupInternal&>(g.getFusedSpikeTarget(g.getTrgNeuronGroup()));

    // If the fused target's TARGET neuron matches original synapse group's source, we should use it's SOURDCE prefixed array
    // **YUCK** it's important to check in this order to match SynapseGroup::getFusedSpikeTarget otherwise things go wrong with recurrent connections
    const std::string prefix = (g.getTrgNeuronGroup() == f.getSrcNeuronGroup()) ? "src" : "trg";
    return getArray(f, prefix + name); 
}
//--------------------------------------------------------------------------
ArrayBase *Runtime::getFusedSrcSpikeEventArray(const SynapseGroupInternal &g, const std::string &name) const
{
    // Get the synapse group SOURCE spike generation has been fused with
    const auto &f = static_cast<const SynapseGroupInternal&>(g.getFusedSpikeEventTarget(g.getSrcNeuronGroup()));

    // If the fused target shares a SOURCE neuron with original synapse group, we should use it's SOURCE prefixed array
    const std::string prefix = (g.getSrcNeuronGroup() == f.getSrcNeuronGroup()) ? "src" : "trg";
    return getArray(f, prefix + name); 
}
//--------------------------------------------------------------------------
ArrayBase *Runtime::getFusedTrgSpikeEventArray(const SynapseGroupInternal &g, const std::string &name) const
{
    // Get the synapse group TARGET spike generation has been fused with
    const auto &f = static_cast<const SynapseGroupInternal&>(g.getFusedSpikeEventTarget(g.getTrgNeuronGroup()));

    // If the fused target's TARGET neuron matches original synapse group's source, we should use it's SOURDCE prefixed array
    // **YUCK** it's important to check in this order to match SynapseGroup::getFusedSpikeEventTarget otherwise things go wrong with recurrent connections
    const std::string prefix = (g.getTrgNeuronGroup() == f.getSrcNeuronGroup()) ? "src" : "trg";
    return getArray(f, prefix + name); 
}
//----------------------------------------------------------------------------
void *Runtime::getSymbol(const std::string &symbolName, bool allowMissing) const
{
#ifdef _WIN32
    void *symbol = GetProcAddress(m_Library, symbolName.c_str());
#else
    void *symbol = dlsym(m_Library, symbolName.c_str());
#endif

    // Return symbol if it's found
    if(symbol) {
        return symbol;
    }
    // Otherwise
    else {
        // If this isn't allowed, throw error
        if(!allowMissing) {
            throw std::runtime_error("Cannot find symbol '" + symbolName + "'");
        }
        // Otherwise, return default
        else {
            return nullptr;
        }
    }
}
//----------------------------------------------------------------------------
const ModelSpecInternal &Runtime::getModel() const
{
    return m_ModelMerged.get().getModel();
}
//----------------------------------------------------------------------------
void Runtime::createArray(ArrayMap &groupArrays, const std::string &varName, const Type::ResolvedType &type, 
                          size_t count, VarLocation location, bool uninitialized, unsigned int logIndent)
{
    const auto r = groupArrays.try_emplace(varName, m_Backend.get().createArray(type, count, location, uninitialized));
    if(r.second) {
        LOGD_RUNTIME << std::string(logIndent, '\t') << "Array '" << varName << "' = " << count << " * " << type.getName() << "(" << r.first->second.get() << ")";
    }
    else {
        throw std::runtime_error("Unable to allocate array with " 
                                 "duplicate name '" + varName + "'");
    }
}
//----------------------------------------------------------------------------
void Runtime::createDynamicParamDestinations(std::unordered_map<std::string, std::pair<Type::ResolvedType, MergedDynamicFieldDestinations>> &destinations, 
                                             const std::string &paramName, const Type::ResolvedType &type, unsigned int logIndent)
{
    LOGD_RUNTIME << std::string(logIndent, '\t') << "Dynamic param '" << paramName << "' (" << type.getName() << ")";
    const auto r = destinations.try_emplace(paramName, std::make_pair(type, MergedDynamicFieldDestinations()));
    if(!r.second) {
        throw std::runtime_error("Unable to add dynamic parameter with " 
                                 "duplicate name '" + paramName + "'");
    }
}
//----------------------------------------------------------------------------
Runtime::BatchEventArray Runtime::getRecordedEvents(unsigned int numNeurons, ArrayBase *array) const
{
    if(!m_NumRecordingTimesteps) {
        throw std::runtime_error("Recording buffer not allocated - cannot get recorded events");
    }

    // Calculate number of words per-timestep
    const unsigned int timestepWords = ceilDivide(numNeurons, 32);

    if(m_Timestep < *m_NumRecordingTimesteps) {
        throw std::runtime_error("Event recording data can only be accessed once buffer is full");
    }
    
    // Calculate start time
    const double dt = getModel().getDT();
    const double startTime = (m_Timestep - *m_NumRecordingTimesteps) * dt;

    // Loop through timesteps
    const uint32_t *spkRecordWords = reinterpret_cast<const uint32_t*>(array->getHostPointer());
    BatchEventArray events(getModel().getBatchSize());
    for(size_t t = 0; t < m_NumRecordingTimesteps.value(); t++) {
        // Loop through batched
        const double time = startTime + (t * dt);
        for(size_t b = 0; b < getModel().getBatchSize(); b++) {
            // Loop through words representing timestep
            auto &batchEvents = events[b];
            for(unsigned int w = 0; w < timestepWords; w++) {
                // Get word
                uint32_t spikeWord = *spkRecordWords++;
            
                // Calculate neuron id of highest bit of this word
                unsigned int neuronID = (w * 32) + 31;
            
                // While bits remain
                while(spikeWord != 0) {
                    // Calculate leading zeros
                    const int numLZ = Utils::clz(spikeWord);
                
                    // If all bits have now been processed, zero spike word
                    // Otherwise shift past the spike we have found
                    spikeWord = (numLZ == 31) ? 0 : (spikeWord << (numLZ + 1));
                
                    // Subtract number of leading zeros from neuron ID
                    neuronID -= numLZ;
                
                    // Add time and ID to vectors
                    batchEvents.first.push_back(time);
                    batchEvents.second.push_back(neuronID);
                
                    // New neuron id of the highest bit of this word
                    neuronID--;
                }
            }
        }
    }

    // Return vectors
    return events;
}
//----------------------------------------------------------------------------
void Runtime::writeRecordedEvents(unsigned int numNeurons, ArrayBase *array, const std::string &path) const
{
    // Get events
    const auto events = getRecordedEvents(numNeurons, array);

    // Open file and write header
    std::ofstream file(path);
    file << "Time [ms], Neuron ID";
    if(getModel().getBatchSize() > 1) {
        file << ", Batch";
    }
    file << std::endl;

    // Loop through batches;
    for(size_t b = 0; b < getModel().getBatchSize(); b++) {
        // Loop through events
        const auto &batchEvents = events[b];
        auto t = batchEvents.first.cbegin();
        auto i = batchEvents.second.cbegin();
        for(;t < batchEvents.first.cend(); t++, i++) {
            // Write to file
            file << *t << ", " << *i;
            if(getModel().getBatchSize() > 1) {
                file << ", " << b;
            }
            file << std::endl;
        }
    }
}
//----------------------------------------------------------------------------
void Runtime::setDynamicParamValue(const std::pair<Type::ResolvedType, MergedDynamicFieldDestinations> &mergedDestinations, 
                                   const Type::NumericValue &value)
{
    // Serailise new value
    std::vector<std::byte> valueStorage;
    Type::serialiseNumeric(value, mergedDestinations.first, valueStorage);

    // Build FFI arguments
    ffi_type *argumentTypes[2]{&ffi_type_uint, mergedDestinations.first.getFFIType()};

    // Prepare an FFI Call InterFace for calls to push merged
    // **TODO** cache - these are the same for all calls with same datatype
    ffi_cif cif;
    ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2,
                                     &ffi_type_void, argumentTypes);
    if (status != FFI_OK) {
        throw std::runtime_error("ffi_prep_cif failed: " + std::to_string(status));
    }
    
    // Loop through merged destinations of this array
    for(const auto &d : mergedDestinations.second.getDestinationFields()) {
        // Get push function
        // **TODO** cache in structure instead of mergedGroup and fieldName
        void *pushFunction = getSymbol("pushMerged" + d.first + std::to_string(d.second.mergedGroupIndex) 
                                       + d.second.fieldName + "ToDevice");

        // Call function
        unsigned int groupIndex = d.second.groupIndex;
        void *argumentPointers[2]{&groupIndex, valueStorage.data()};
        ffi_call(&cif, FFI_FN(pushFunction), nullptr, argumentPointers);
    }
}
//----------------------------------------------------------------------------
void Runtime::allocateExtraGlobalParam(ArrayMap &groupArrays, const std::string &varName,
                                       size_t count)
{
    // Find array
    auto *array = groupArrays.at(varName).get();

    // Allocate array
    array->allocate(count);

    // Serialise host pointer
    std::vector<std::byte> serialisedHostPointer;
    array->serialiseHostPointer(serialisedHostPointer, false);

    // If backend requires it, serialise device object
    std::vector<std::byte> serialisedDeviceObject;
    if(m_Backend.get().isArrayDeviceObjectRequired()) {
        array->serialiseDeviceObject(serialisedDeviceObject, false);
    }
    
    // If backend requires it, serialise host object
    std::vector<std::byte> serialisedHostObject;
    if(m_Backend.get().isArrayHostObjectRequired()) {
        array->serialiseHostObject(serialisedHostObject, false);
    }

    // Build FFI arguments
    // **TODO** allow backend to override type
    ffi_type *argumentTypes[2]{&ffi_type_uint, &ffi_type_pointer};

    // Prepare an FFI Call InterFace for calls to push merged
    // **TODO** cache - these are the same for all EGP calls
    ffi_cif cif;
    ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2,
                                     &ffi_type_void, argumentTypes);
    if (status != FFI_OK) {
        throw std::runtime_error("ffi_prep_cif failed: " + std::to_string(status));
    }
    
    // Loop through merged destinations of this array
    const auto &mergedDestinations = m_MergedDynamicArrays.at(array);
    for(const auto &d : mergedDestinations.getDestinationFields()) {
        // Get push function
        // **TODO** cache in structure instead of mergedGroup and fieldName
        void *pushFunction = getSymbol("pushMerged" + d.first + std::to_string(d.second.mergedGroupIndex) 
                                       + d.second.fieldName + "ToDevice");

        // Call function
        unsigned int groupIndex = d.second.groupIndex;
        void *argumentPointers[2]{&groupIndex, nullptr};
        if(d.second.fieldType & GroupMergedFieldType::HOST) {
            assert(!serialisedHostPointer.empty());
            argumentPointers[1] = serialisedHostPointer.data();
        }
        else if(d.second.fieldType & GroupMergedFieldType::HOST_OBJECT) {
            assert(!serialisedHostObject.empty());
            argumentPointers[1] = serialisedHostObject.data();
        }
        // Serialise device object if backend requires it
        else {
            if(m_Backend.get().isArrayDeviceObjectRequired()) {
                assert(!serialisedDeviceObject.empty());
                argumentPointers[1] = serialisedDeviceObject.data();
            }
            // Otherwise, host pointer
            else {
                assert(!serialisedHostPointer.empty());
                argumentPointers[1] = serialisedHostPointer.data();
            }
        }
        ffi_call(&cif, FFI_FN(pushFunction), nullptr, argumentPointers);
    }
}
}   // namespace GeNN::Runtime

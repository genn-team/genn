#include "runtime/runtime.h"

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
    const bool pre = (varDims & VarAccessDim::PRE_NEURON);
    const bool post = (varDims & VarAccessDim::POST_NEURON);
    const unsigned int numPre = sg.getSrcNeuronGroup()->getNumNeurons();
    const unsigned int numPost = sg.getTrgNeuronGroup()->getNumNeurons();
    const unsigned int rowStride = backend.getSynapticMatrixRowStride(sg);
    if(pre && post) {
        if(sg.getMatrixType() & SynapseMatrixWeight::KERNEL) {
            return sg.getKernelSizeFlattened();
        }
        else {
            return numPre * rowStride;
        }
    }
    else if(pre) {
        return numPre;
    }
    else if(post) {
        return numPost;
    }
    else {
        return 1;
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// GeNN::Runtime::Runtime
//--------------------------------------------------------------------------
namespace GeNN::Runtime
{
Runtime::Runtime(const filesystem::path &modelPath, const CodeGenerator::ModelSpecMerged &modelMerged, 
                 const CodeGenerator::BackendBase &backend)
:   m_Timestep(0), m_ModelMerged(modelMerged), m_Backend(backend)
{
    // Load library
#ifdef _WIN32
    const std::string runnerName = "runner_" + modelMerged.getModel().getName();
#ifdef _DEBUG
    const std::string libraryName = (modelPath / (runnerName + "_Debug.dll")).str();
#else
    const std::string libraryName = pathToModel + "\\" + runnerName + "_Release.dll";
#endif
     m_Library = LoadLibrary(libraryName.c_str());
#else
    const std::string libraryName = (modelPath / (modelMerged.getModel().getName() + "_CODE") / "librunner.so").str();
    m_Library = dlopen(libraryName.c_str(), RTLD_NOW);
#endif

    // If no seed is specified
    const auto seed = m_ModelMerged.get().getModel().getSeed();
    if(seed == 0) {
        // Use system randomness to generate seed sequence
        uint32_t seedData[std::mt19937::state_size];
        std::random_device seedSource;
        for(int i = 0; i < std::mt19937::state_size; i++) {
            seedData[i] = seedSource();
        }

        // Seed host RNG
        m_HostRNG.seed(std::seed_seq{std::begin(seedData), std::end(seedData)});
    }
    // Otherwise, seed host RNG from model seed
    // **NOTE** this is a terrible idea see http://www.pcg-random.org/posts/cpp-seeding-surprises.html
    else {
        m_HostRNG.seed(std::seed_seq{seed});
    }
}
//----------------------------------------------------------------------------
Runtime::~Runtime()
{
    if(m_Library) {
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
    // Store number of recording timesteps
    m_NumRecordingTimesteps = numRecordingTimesteps;

    // Loop through neuron groups
    const auto &model = m_ModelMerged.get().getModel();
    const unsigned int batchSize = model.getBatchSize();
    for(const auto &n : model.getNeuronGroups()) {
        // True spike variables
        const size_t numNeuronDelaySlots = batchSize * (size_t)n.second.getNumNeurons() * (size_t)n.second.getNumDelaySlots();
        
        // If spikes are required, allocate arrays for counts and spikes
        if(n.second.isTrueSpikeRequired()) {
            allocateArray(n.first, "spkCnt", Type::Uint32, batchSize * n.second.getNumDelaySlots(), 
                          n.second.getSpikeLocation());
            allocateArray(n.first, "spk", Type::Uint32, numNeuronDelaySlots, 
                          n.second.getSpikeLocation());
        }

        // If spike-like events are required, allocate arrays for counts and spikes
        if(n.second.isSpikeEventRequired()) {
            allocateArray(n.first, "spkEventCnt", Type::Uint32, batchSize * n.second.getNumDelaySlots(), 
                          n.second.getSpikeEventLocation());
            allocateArray(n.first, "spkEvent", Type::Uint32, numNeuronDelaySlots, 
                          n.second.getSpikeEventLocation());
        }
        
        // If spike or spike-like event recording is enabled
        if(n.second.isSpikeRecordingEnabled() || n.second.isSpikeEventRecordingEnabled()) {
            if(!numRecordingTimesteps) {
                throw std::runtime_error("Cannot use recording system without specifying number of recording timesteps");
            }

            // Calculate number of words required and allocate arrays
            const unsigned int numRecordingWords = (ceilDivide(n.second.getNumNeurons(), 32) * batchSize) * numRecordingTimesteps.value();
            if(n.second.isSpikeRecordingEnabled()) {
                allocateArray(n.first, "recordSpk", Type::Uint32,  numRecordingWords, 
                              VarLocation::HOST_DEVICE);

            }
            else if(n.second.isSpikeEventRecordingEnabled()) {
                allocateArray(n.first, "recordSpkEvent", Type::Uint32,  numRecordingWords, 
                              VarLocation::HOST_DEVICE);
            }
           
        }

        // If neuron group has axonal or back-propagation delays, add delay queue pointer
        if (n.second.isDelayRequired()) {
            m_DelayQueuePointer.try_emplace(n.first, 0);
            // **TODO** also make device version
        }
        
        // If neuron group needs to record its spike times
        if (n.second.isSpikeTimeRequired()) {
            allocateArray(n.first, "sT", model.getTimePrecision(), numNeuronDelaySlots, 
                          n.second.getSpikeTimeLocation());
        }

        // If neuron group needs to record its previous spike times
        if (n.second.isPrevSpikeTimeRequired()) {
            allocateArray(n.first, "prevST", model.getTimePrecision(), numNeuronDelaySlots, 
                          n.second.getPrevSpikeTimeLocation());
        }

        // If neuron group needs to record its spike-like-event times
        if (n.second.isSpikeEventTimeRequired()) {
            allocateArray(n.first, "seT", model.getTimePrecision(), numNeuronDelaySlots, 
                          n.second.getSpikeEventTimeLocation());
        }

        // If neuron group needs to record its previous spike-like-event times
        if (n.second.isPrevSpikeEventTimeRequired()) {
            allocateArray(n.first, "prevSET", model.getTimePrecision(), numNeuronDelaySlots, 
                          n.second.getPrevSpikeEventTimeLocation());
        }

        // If neuron group needs per-neuron RNGs
        /*if(n.second.isSimRNGRequired()) {
            backend.genPopulationRNG(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                     "rng" + n.first, batchSize * n.second.getNumNeurons(), mem);
        }*/

        // Allocate neuron state variables
        allocateNeuronVars<NeuronVarAdapter>(n.second, n.second.getNumNeurons(), 
                                             batchSize, n.second.getNumDelaySlots(), true);
        
        // Allocate current source variables
        for (const auto *cs : n.second.getCurrentSources()) {
            allocateNeuronVars<CurrentSourceVarAdapter>(*cs, n.second.getNumNeurons(), batchSize, 1, true);
        }

        // Allocate postsynaptic model variables from incoming populations
        for(const auto *sg : n.second.getFusedPSMInSyn()) {
            allocateArray(sg->getFusedPSVarSuffix(), "outPost", model.getPrecision(), 
                          sg->getTrgNeuronGroup()->getNumNeurons() * batchSize,
                          sg->getInSynLocation());
            
            if (sg->isDendriticDelayRequired()) {
                allocateArray(sg->getFusedPSVarSuffix(), "denDelay", model.getPrecision(), 
                              (size_t)sg->getMaxDendriticDelayTimesteps() * (size_t)sg->getTrgNeuronGroup()->getNumNeurons() * batchSize,
                              sg->getDendriticDelayLocation());
                //genHostDeviceScalar(backend, definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                //                    Type::Uint32, "denDelayPtr" + sg->getFusedPSVarSuffix(), "0", mem);
            }

            allocateNeuronVars<SynapsePSMVarAdapter>(*sg, sg->getTrgNeuronGroup()->getNumNeurons(), batchSize, 1, true);
        }

        // Allocate fused pre-output variables
        for(const auto *sg : n.second.getFusedPreOutputOutSyn()) {
            allocateArray(sg->getFusedPreOutputSuffix(), "outPre", model.getPrecision(), 
                          sg->getSrcNeuronGroup()->getNumNeurons() * batchSize,
                          sg->getInSynLocation());
        }
        
        // Allocate fused postsynaptic weight update variables from incoming synaptic populations
        for(const auto *sg: n.second.getFusedWUPreOutSyn()) {
            const unsigned int preDelaySlots = (sg->getDelaySteps() == NO_DELAY) ? 1 : sg->getSrcNeuronGroup()->getNumDelaySlots();
            allocateNeuronVars<SynapseWUPreVarAdapter>(*sg, sg->getSrcNeuronGroup()->getNumNeurons(), batchSize, preDelaySlots, true);
        }
        
        // Loop through merged postsynaptic weight updates of incoming synaptic populations
        for(const auto *sg: n.second.getFusedWUPostInSyn()) { 
            const unsigned int postDelaySlots = (sg->getBackPropDelaySteps() == NO_DELAY) ? 1 : sg->getTrgNeuronGroup()->getNumDelaySlots();
            allocateNeuronVars<SynapseWUPostVarAdapter>(*sg, sg->getTrgNeuronGroup()->getNumNeurons(), batchSize, postDelaySlots, true);
        }  
    }

    // Loop through synapse groups
    for(const auto &s : model.getSynapseGroups()) {
        // If synapse group has individual or kernel weights
        const bool individualWeights = (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);
        const bool kernelWeights = (s.second.getMatrixType() & SynapseMatrixWeight::KERNEL);
        if (individualWeights || kernelWeights) {
            for(const auto &var : wu->getVars()) {
                const auto resolvedType = var.type.resolve(modelMerged.getModel().getTypeContext());
                const auto varDims = getVarAccessDim(var.access);
                const size_t numVarCopies = ((varDims & VarAccessDim::BATCH) && batched) ? batchSize : 1;
                const size_t numVarElements = getNumSynapseVarElements(varDims, backend, s.second);

                allocateArray(s.first, var.name, resolvedType, numVarCopies * numVarElements,
                              s.second.getWUVarLocation(var.name));
            }
        }

        // If connectivity is bitmask
        const size_t numPre = s.second.getSrcNeuronGroup()->getNumNeurons();
        const size_t rowStride = backend.getSynapticMatrixRowStride(s.second);
        if(s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            const size_t gpSize = ceilDivide((size_t)numPre * rowStride, 32);
            allocateArray(s.first, "gp", Type::Uint32, gpSize,
                          s.second.getSparseConnectivityLocation());
        }
        // Otherwise, if connectivity is sparse
        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            // Row lengths
            allocateArray(s.first, "rowLength", Type::Uint32, numPre,
                          s.second.getSparseConnectivityLocation());

            // Target indices
            allocateArray(s.first, "ind", s.second.getSparseIndType(), numPre * rowStride,
                          s.second.getSparseConnectivityLocation());
        
            // **TODO** remap is not always required
            if(backend.isPostsynapticRemapRequired() && !s.second.getWUModel()->getLearnPostCode().empty()) {
                // Allocate column lengths
                const size_t numPost = s.second.getTrgNeuronGroup()->getNumNeurons();
                const size_t colStride = s.second.getMaxSourceConnections();
                allocateArray(s.first, "colLength", Type::Uint32, numPost, VarLocation::DEVICE);
                
                // Allocate remap
                allocateArray(s.first, "remap", Type::Uint32, numPost * colStride, VarLocation::DEVICE);
            }
        }
    }

    // Allocate custom update variables
    for(const auto &c : model.getCustomUpdates()) {
        allocateNeuronVars<CustomUpdateVarAdapter>(c.second, c.second.getSize(), batchSize, 1, 
                                                   c.second.getDims() & VarAccessDim::BATCH);
    }

    // **TODO** custom update WU

    // **TODO** custom connectivity update

    

}
//----------------------------------------------------------------------------
void Runtime::initialise()
{
}
//----------------------------------------------------------------------------
void Runtime::initialiseSparse()
{
}
//----------------------------------------------------------------------------
void Runtime::stepTime()
{
    // Update synaptic state
    m_UpdateSynapses(getTime());
    
    // Generate code to advance host-side spike queues    
    /*for(const auto &n : model.getNeuronGroups()) {
        if (n.second.isDelayRequired()) {
            runner << "spkQuePtr" << n.first << " = (spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
        }
    }*/

    // Update neuronal state
    m_UpdateNeurons(getTime(), m_NumRecordingTimesteps ? m_Timestep % m_NumRecordingTimesteps : 0);

    // Generate code to advance host side dendritic delay buffers
    /*for(const auto &n : model.getNeuronGroups()) {
        // Loop through incoming synaptic populations
        for(const auto *sg : n.second.getFusedPSMInSyn()) {
            if(sg->isDendriticDelayRequired()) {
                runner << "denDelayPtr" << sg->getFusedPSVarSuffix() << " = (denDelayPtr" << sg->getFusedPSVarSuffix() << " + 1) % " << sg->getMaxDendriticDelayTimesteps() << ";" << std::endl;
            }
        }
    }*/

    // Advance time
    m_Timestep++;
}
//----------------------------------------------------------------------------
void Runtime::allocateExtraGlobalParam(const std::string &groupName, const std::string &varName,
                                       size_t count)
{
    // Allocate array
    // **TODO** dynamic flag determines whether this is allowed
    m_Arrays.at(groupName).at(varName)->allocate(count);

    // **TODO** call push functions
    // **TODO** signature of push function will vary depending on backend and type
}
//----------------------------------------------------------------------------
void Runtime::allocateArray(const std::string &groupName, const std::string &varName,
                            const Type::ResolvedType &type, size_t count, 
                            VarLocation location, MemAlloc &memAlloc)
{
    const auto r = m_Arrays[groupName].try_emplace(
        varName, m_Backend.get().allocateArray(type, count, location, memAlloc));
    if(!r.second) {
        throw std::runtime_error("Unable to allocate array with duplicate name '" 
                                 + varName + "' for group '" + groupName + "'");
    }
}
}   // namespace GeNN::Runtime
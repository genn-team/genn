#include "runtime/runtime.h"

// PLOG includes
#include <plog/Log.h>

// Filesystem includes
#include "path.h"

// FFI includes
#include <ffi.h>

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
    const size_t numPre = sg.getSrcNeuronGroup()->getNumNeurons();
    const size_t numPost = sg.getTrgNeuronGroup()->getNumNeurons();
    const size_t rowStride = backend.getSynapticMatrixRowStride(sg);
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
template<typename G>
void pushMergedGroup(CodeGenerator::GroupMerged<G> &g) 
{
    // Loop through groups
    const auto sortedFields = g.getSortedFields(m_Backend.get());

    // Start vector of argument types with unsigned int group index and them append FFI types of each argument
    // **TODO** allow backend to override type
    std::vector<ffi_type*> argumentTypes{&ffi_type_uint};
    argumentTypes.reserve(sortedFields.size() + 1);
    std::transform(sortedFields.cbegin(), sortedFields.cend(), std::back_inserter(argumentTypes),
                    [](const auto &f){ return std::get<0>(f).getFFIType() });
        
    // Prepare an FFI Call InterFace for calls to push merged
    ffi_cif cif;
    ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, argumentTypes.size(),
                                        &ffi_type_void, argumentTypes.data());
    if (status != FFI_OK) {
        throw std::runtime_error("ffi_prep_cif failed: " + std::to_string(status));
    }

    // Get push function
    void *pushFunction = getSymbol("pushMerged" + g.name + "Group" + std::to_string(g.getIndex()) + "ToDevice");

    // Loop through groups in merged group
    std::vector<void*> arguments(sortedFields.size() + 1);
    for(unsigned int groupIndex = 0; groupIndex < g.getGroups().size(); groupIndex++) {
        arguments[0] = &groupIndex;

        for(const auto &f : sortedFields) {
            const auto &fieldType = std::get<0>(f);

            // If field type is pointer
            //  
            const std::string fieldInitVal = std::get<2>(f)(g.getGroups().at(groupIndex), groupIndex);
        }

        //generateStructFieldArguments(runnerMergedStructAlloc, groupIndex, sortedFields);
        //runnerMergedStructAlloc << ");" << std::endl;
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
    const std::string libraryName = (modelPath / (runnerName + "_Release.dll")).str();
#endif
     m_Library = LoadLibrary(libraryName.c_str());
#else
    const std::string libraryName = (modelPath / (modelMerged.getModel().getName() + "_CODE") / "librunner.so").str();
    m_Library = dlopen(libraryName.c_str(), RTLD_NOW);
#endif
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
            createArray(&n.second, "spkCnt", Type::Uint32, batchSize * n.second.getNumDelaySlots(), 
                        n.second.getSpikeLocation());
            createArray(&n.second, "spk", Type::Uint32, numNeuronDelaySlots, 
                        n.second.getSpikeLocation());
        }

        // If spike-like events are required, allocate arrays for counts and spikes
        if(n.second.isSpikeEventRequired()) {
            createArray(&n.second, "spkEventCnt", Type::Uint32, batchSize * n.second.getNumDelaySlots(), 
                        n.second.getSpikeEventLocation());
            createArray(&n.second, "spkEvent", Type::Uint32, numNeuronDelaySlots, 
                        n.second.getSpikeEventLocation());
        }
        
        // If spike or spike-like event recording is enabled
        if(n.second.isSpikeRecordingEnabled() || n.second.isSpikeEventRecordingEnabled()) {
            if(!numRecordingTimesteps) {
                throw std::runtime_error("Cannot use recording system without specifying number of recording timesteps");
            }

            // Calculate number of words required and allocate arrays
            const size_t numRecordingWords = (ceilDivide(n.second.getNumNeurons(), 32) * batchSize) * numRecordingTimesteps.value();
            if(n.second.isSpikeRecordingEnabled()) {
                createArray(&n.second, "recordSpk", Type::Uint32,  numRecordingWords, 
                            VarLocation::HOST_DEVICE);

            }
            else if(n.second.isSpikeEventRecordingEnabled()) {
                createArray(&n.second, "recordSpkEvent", Type::Uint32,  numRecordingWords, 
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
            createArray(&n.second, "sT", model.getTimePrecision(), numNeuronDelaySlots, 
                        n.second.getSpikeTimeLocation());
        }

        // If neuron group needs to record its previous spike times
        if (n.second.isPrevSpikeTimeRequired()) {
            createArray(&n.second, "prevST", model.getTimePrecision(), numNeuronDelaySlots, 
                        n.second.getPrevSpikeTimeLocation());
        }

        // If neuron group needs to record its spike-like-event times
        if (n.second.isSpikeEventTimeRequired()) {
            createArray(&n.second, "seT", model.getTimePrecision(), numNeuronDelaySlots, 
                        n.second.getSpikeEventTimeLocation());
        }

        // If neuron group needs to record its previous spike-like-event times
        if (n.second.isPrevSpikeEventTimeRequired()) {
            createArray(&n.second, "prevSET", model.getTimePrecision(), numNeuronDelaySlots, 
                        n.second.getPrevSpikeEventTimeLocation());
        }

        // If neuron group needs per-neuron RNGs
        /*if(n.second.isSimRNGRequired()) {
            backend.genPopulationRNG(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                     "rng" + n.first, batchSize * n.second.getNumNeurons(), mem);
        }*/

        // Allocate neuron state variables
        allocateNeuronVars<NeuronVarAdapter>(&n.second, n.second.getNumNeurons(), 
                                             batchSize, n.second.getNumDelaySlots(), true);
        
        // Allocate current source variables
        for (const auto *cs : n.second.getCurrentSources()) {
            allocateNeuronVars<CurrentSourceVarAdapter>(cs, n.second.getNumNeurons(), batchSize, 1, true);
        }

        // Allocate postsynaptic model variables from incoming populations
        for(const auto *sg : n.second.getFusedPSMInSyn()) {
            createArray(sg, "outPost", model.getPrecision(), 
                        sg->getTrgNeuronGroup()->getNumNeurons() * batchSize,
                        sg->getInSynLocation());
            
            if (sg->isDendriticDelayRequired()) {
                createArray(sg, "denDelay", model.getPrecision(), 
                            (size_t)sg->getMaxDendriticDelayTimesteps() * (size_t)sg->getTrgNeuronGroup()->getNumNeurons() * batchSize,
                            sg->getDendriticDelayLocation());
                //genHostDeviceScalar(backend, definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                //                    Type::Uint32, "denDelayPtr" + sg->getFusedPSVarSuffix(), "0", mem);
            }

            allocateNeuronVars<SynapsePSMVarAdapter>(sg, sg->getTrgNeuronGroup()->getNumNeurons(), batchSize, 1, true);
        }

        // Allocate fused pre-output variables
        for(const auto *sg : n.second.getFusedPreOutputOutSyn()) {
            createArray(sg, "outPre", model.getPrecision(), 
                        sg->getSrcNeuronGroup()->getNumNeurons() * batchSize,
                        sg->getInSynLocation());
        }
        
        // Allocate fused postsynaptic weight update variables from incoming synaptic populations
        for(const auto *sg: n.second.getFusedWUPreOutSyn()) {
            const unsigned int preDelaySlots = (sg->getDelaySteps() == NO_DELAY) ? 1 : sg->getSrcNeuronGroup()->getNumDelaySlots();
            allocateNeuronVars<SynapseWUPreVarAdapter>(sg, sg->getSrcNeuronGroup()->getNumNeurons(), batchSize, preDelaySlots, true);
        }
        
        // Loop through merged postsynaptic weight updates of incoming synaptic populations
        for(const auto *sg: n.second.getFusedWUPostInSyn()) { 
            const unsigned int postDelaySlots = (sg->getBackPropDelaySteps() == NO_DELAY) ? 1 : sg->getTrgNeuronGroup()->getNumDelaySlots();
            allocateNeuronVars<SynapseWUPostVarAdapter>(sg, sg->getTrgNeuronGroup()->getNumNeurons(), batchSize, postDelaySlots, true);
        }  
    }

    // Loop through synapse groups
    for(const auto &s : model.getSynapseGroups()) {
        // If synapse group has individual or kernel weights
        const bool individualWeights = (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);
        const bool kernelWeights = (s.second.getMatrixType() & SynapseMatrixWeight::KERNEL);
        if (individualWeights || kernelWeights) {
            for(const auto &var : s.second.getWUModel()->getVars()) {
                const auto resolvedType = var.type.resolve(model.getTypeContext());
                const auto varDims = getVarAccessDim(var.access);
                const size_t numVarCopies = (varDims & VarAccessDim::BATCH) ? batchSize : 1;
                const size_t numVarElements = getNumSynapseVarElements(varDims, m_Backend.get(), s.second);
                createArray(&s.second, var.name, resolvedType, numVarCopies * numVarElements,
                            s.second.getWUVarLocation(var.name));
            }
        }

        // If connectivity is bitmask
        const size_t numPre = s.second.getSrcNeuronGroup()->getNumNeurons();
        const size_t rowStride = m_Backend.get().getSynapticMatrixRowStride(s.second);
        if(s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            const size_t gpSize = ceilDivide((size_t)numPre * rowStride, 32);
            createArray(&s.second, "gp", Type::Uint32, gpSize,
                        s.second.getSparseConnectivityLocation());
        }
        // Otherwise, if connectivity is sparse
        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            // Row lengths
            createArray(&s.second, "rowLength", Type::Uint32, numPre,
                        s.second.getSparseConnectivityLocation());

            // Target indices
            createArray(&s.second, "ind", s.second.getSparseIndType(), numPre * rowStride,
                        s.second.getSparseConnectivityLocation());
        
            // **TODO** remap is not always required
            if(m_Backend.get().isPostsynapticRemapRequired() && !s.second.getWUModel()->getLearnPostCode().empty()) {
                // Allocate column lengths
                const size_t numPost = s.second.getTrgNeuronGroup()->getNumNeurons();
                const size_t colStride = s.second.getMaxSourceConnections();
                createArray(&s.second, "colLength", Type::Uint32, numPost, VarLocation::DEVICE);
                
                // Allocate remap
                createArray(&s.second, "remap", Type::Uint32, numPost * colStride, VarLocation::DEVICE);
            }
        }
    }

    // Allocate custom update variables
    for(const auto &c : model.getCustomUpdates()) {
        allocateNeuronVars<CustomUpdateVarAdapter>(&c.second, c.second.getSize(), batchSize, 1, 
                                                   c.second.getDims() & VarAccessDim::BATCH);
    }

    // **TODO** custom update WU

    // **TODO** custom connectivity update

    

}
//----------------------------------------------------------------------------
void Runtime::initialise()
{
    //m_Initialise()
}
//----------------------------------------------------------------------------
void Runtime::initialiseSparse()
{
    //initEnv.getStream() << "copyStateToDevice(true);" << std::endl;
    //initEnv.getStream() << "copyConnectivityToDevice(true);" << std::endl << std::endl;

    //m_InitialiseSparse();
}
//----------------------------------------------------------------------------
void Runtime::stepTime()
{
   //m_StepTime(getTime());
    
    // Generate code to advance host-side spike queues    
    /*for(const auto &n : model.getNeuronGroups()) {
        if (n.second.isDelayRequired()) {
            runner << "spkQuePtr" << n.first << " = (spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
        }
    }*/
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
double Runtime::getTime() const
{ 
    return m_Timestep * m_ModelMerged.get().getModel().getDT();
}
//----------------------------------------------------------------------------
void *Runtime::getSymbol(const std::string &symbolName) const
{
#ifdef _WIN32
    void *symbol = GetProcAddress(m_Library, symbolName.c_str());
#else
    void *symbol = dlsym(m_Library, symbolName.c_str());
#endif

    // If this symbol's missing
    if(symbol == nullptr) {
        throw std::runtime_error("Cannot find symbol '" + symbolName + "'");
    }
    // Otherwise, return symbolPopulationFuncs
    else {
        return symbol;
    }
}
//----------------------------------------------------------------------------
void Runtime::createArray(ArrayMap &groupArrays, const std::string &varName, const Type::ResolvedType &type, 
                          size_t count, VarLocation location)
{
    const auto r = groupArrays.try_emplace(varName, m_Backend.get().createArray(type, count, location));
    if(!r.second) {
        throw std::runtime_error("Unable to allocate array with " 
                                 "duplicate name '" + varName + "'");
    }
}
//----------------------------------------------------------------------------
void Runtime::allocateExtraGlobalParam(ArrayMap &groupArrays, const std::string &varName,
                                       size_t count)
{
    // Allocate array
    // **TODO** dynamic flag determines whether this is allowed
    groupArrays.at(varName)->allocate(count);

    // **TODO** call push functions
    // **TODO** signature of push function will vary depending on backend and type
}
}   // namespace GeNN::Runtime
#include "backend.h"

// Standard C++ includes
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>

// Standard C includes
#include <cassert>
#include <cmath>
#include <cstdlib>

// GeNN includes
#include "gennUtils.h"
#include "logging.h"
#include "type.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/environment.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/standardLibrary.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
// Library of ISPC random functions
const EnvironmentLibrary::Library ispcRandomFunctions = {
    {"gennrand", {Type::ResolvedType::createFunction(Type::Uint32, {}), "rand()"}},
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Float, {}), "randUniform()"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Float, {}), "randNormal()"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Float, {}), "randExponential()"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Float, {Type::Float, Type::Float}), "randLogNormal($(0), $(1))"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Float, {Type::Float}), "randGamma($(0))"}},
};

// Format for ISPC neuron kernel functions
const char* ispcNeuronKernelTemplate = 
R"(export void %s(
    uniform float dt,
    uniform %s* uniform group,
    uniform unsigned int numNeurons,
    uniform unsigned int batchSize)
{
    // Process all batches
    for (uniform unsigned int batchID = 0; batchID < batchSize; batchID++) {
        // Process neurons in SIMD-parallel using a foreach loop
        foreach (n = 0 ... numNeurons) {
            // Calculate offset for current batch
            unsigned int offset = n + (batchID * numNeurons);
            
%s
        }
    }
}
)";

// Format for ISPC SpikeSourceArray neuron kernel function
const char* ispcSpikeSourceArrayKernelTemplate = 
R"(export void %s(
    uniform float t,
    uniform %s* uniform group,
    uniform unsigned int numNeurons,
    uniform unsigned int batchSize)
{
    // Process neurons in SIMD-parallel fashion
    foreach (i = 0 ... numNeurons) {
        // Calculate offset for batching
        uniform unsigned int batchID = 0; // Use 0 if not batched
        unsigned int offset = i + (batchID * numNeurons);
        
%s
    }
}
)";

// Format for ISPC custom neuron sim code function
const char *ispcCustomNeuronSimCodeTemplate =
R"(// Custom neuron simulation code function for %1$s model
void %1$s_simCode(uniform CustomNeuronGroup* uniform group, varying unsigned int i, unsigned int offset, 
                 uniform unsigned int batchID, uniform float dt, varying int* spiked)
{
    // Access state variables by index
%2$s

    // Access parameters by index
%3$s

    // Model simulation code
%4$s
}
)";

// Format for ISPC custom neuron threshold condition function
const char *ispcCustomNeuronThresholdTemplate =
R"(// Custom neuron threshold condition function for %1$s model
bool %1$s_thresholdCondition(uniform CustomNeuronGroup* uniform group, unsigned int offset)
{
    // Access state variables by index
%2$s

    // Access parameters by index
%3$s

    // Threshold condition code
    return %4$s;
}
)";

// Format for ISPC custom neuron reset code function
const char *ispcCustomNeuronResetTemplate =
R"(// Custom neuron reset code function for %1$s model
void %1$s_resetCode(uniform CustomNeuronGroup* uniform group, unsigned int offset)
{
    // Access state variables by index
%2$s

    // Access parameters by index
%3$s

    // Reset code
%4$s
}
)";

//--------------------------------------------------------------------------
// Timer
//--------------------------------------------------------------------------
class Timer
{
public:
    Timer(CodeStream &codeStream, const std::string &name, bool timingEnabled)
    :   m_CodeStream(codeStream), m_Name(name), m_TimingEnabled(timingEnabled)
    {
        // Record start event
        if(m_TimingEnabled) {
            m_CodeStream << "const auto " << m_Name << "Start = std::chrono::high_resolution_clock::now();" << std::endl;
        }
    }

    ~Timer()
    {
        // Record stop event
        if(m_TimingEnabled) {
            m_CodeStream << m_Name << "Time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - " << m_Name << "Start).count();" << std::endl;
        }
    }

private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    CodeStream &m_CodeStream;
    const std::string m_Name;
    const bool m_TimingEnabled;
};

// Helper function to determine if a variable should be uniform or varying based on access
bool shouldBeUniform(VarAccessMode access, const ISPC::Preferences &preferences)
{
    // If we want to maximize the use of uniforms for performance
    if (preferences.maximizeUniforms) {
        // Only variables that are read-write or reduce need to be varying
        return !((access & VarAccessModeAttribute::READ_WRITE) || 
                 (access & VarAccessModeAttribute::REDUCE));
    }
    else {
        // Another approach here is only variables that are explicitly read-only can be uniform? I think.
        return ((access & VarAccessModeAttribute::READ_ONLY) && 
                !(access & VarAccessModeAttribute::READ_WRITE) && 
                !(access & VarAccessModeAttribute::REDUCE));
    }
}

// Helper function to set ISPC-specific variable attributes based on access modes
std::string getISPCVariablePrefix(VarAccessMode access, const ISPC::Preferences &preferences)
{
    return shouldBeUniform(access, preferences) ? "uniform " : "varying ";
}

// Helper function to identify neuron model type
std::string GeNN::CodeGenerator::ISPC::Backend::getNeuronModelType(const NeuronGroupInternal &ng) const
{
    const auto *model = ng.getModel();
    
    // Check the class name or other identifying features to determine model type
    const std::string className = typeid(*model).name();
    
    // Check for built-in models
    if(className.find("LIF") != std::string::npos) {
        return "LIF";
    }
    else if(className.find("Poisson") != std::string::npos) {
        return "Poisson";
    }
    else if(className.find("IzhikevichV") != std::string::npos) {
        // Identify IzhikevichV as a custom model. This is a temporary hack to get the cde generator working.
        return "Custom";
    }
    else if(className.find("Izhikevich") != std::string::npos) {
        if(className.find("Variable") != std::string::npos) {
            return "IzhikevichVariable";
        }
        return "Izhikevich";
    }
    else if(className.find("RulkovMap") != std::string::npos) {
        return "RulkovMap";
    }
    else if(className.find("TraubMiles") != std::string::npos) {
        return "TraubMiles";
    }
    else if(className.find("SpikeSourceArray") != std::string::npos) {
        return "SpikeSourceArray";
    }
    
    // If no specific match, let's assume it's a custom model
    return "Custom";
}

// Helper function to get the struct type for a neuron model
std::string getNeuronStructType(const std::string &modelType)
{
    if (modelType == "LIF") {
        return "MergedLIFNeuronGroup";
    }
    else if (modelType == "Poisson") {
        return "MergedPoissonNeuronGroup";
    }
    else if (modelType == "SpikeSourceArray") {
        return "MergedSpikeSourceArrayGroup";
    }
    else if (modelType == "RulkovMap") {
        return "MergedRulkovMapGroup";
    }
    else if (modelType == "Izhikevich") {
        return "MergedIzhikevichGroup";
    }
    else if (modelType == "IzhikevichVariable") {
        return "MergedIzhikevichVariableGroup";
    }
    else if (modelType == "TraubMiles") {
        return "MergedTraubMilesGroup";
    }
    
    // Default to LIF
    return "MergedLIFNeuronGroup";
}

// Helper function to get the kernel function name for a neuron model
std::string GeNN::CodeGenerator::ISPC::Backend::getNeuronKernelName(const std::string &modelType) const
{
    // Map model types to the kernel function names
    if(modelType == "LIF") {
        return "updateLIFNeurons";
    }
    else if(modelType == "Poisson") {
        return "updatePoissonNeurons";
    }
    else if(modelType == "SpikeSourceArray") {
        return "updateSpikeSourceArrayNeurons";
    }
    else if(modelType == "RulkovMap") {
        return "updateRulkovMapNeurons";
    }
    else if(modelType == "Izhikevich") {
        return "updateIzhikevichNeurons";
    }
    else if(modelType == "IzhikevichVariable") {
        return "updateIzhikevichVariableNeurons";
    }
    else if(modelType == "TraubMiles") {
        return "updateTraubMilesNeurons";
    }
    
    // For custom models:
    return "updateCustomNeuron";
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Array
//--------------------------------------------------------------------------
class Array : public Runtime::ArrayBase
{
public:
    Array(const Type::ResolvedType &type, size_t count, 
          VarLocation location, bool uninitialized)
    :   ArrayBase(type, count, location, uninitialized)
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
        // Based on reading info, ISPC requires aligned memory for optimal SIMD performance
        // aligned to 64 bytes for AVX-512
        setCount(count);

        // Calculate size including padding for alignment
        const size_t typeSizeBytes = getType().getValue().size;
        size_t sizeBytes = getSizeBytes();
        
        #ifdef _WIN32
        // On Windows, use _aligned_malloc
        setHostPointer(reinterpret_cast<std::byte*>(_aligned_malloc(sizeBytes, 64)));
        if (!getHostPointer()) {
            throw std::bad_alloc();
        }
        #else
        // On Unix systems, use posix_memalign
        void* ptr = nullptr;
        if(posix_memalign(&ptr, 64, sizeBytes) != 0) {
            throw std::bad_alloc();
        }
        setHostPointer(reinterpret_cast<std::byte*>(ptr));
        #endif
    }

    //! Free array
    virtual void free() final
    {
        #ifdef _WIN32
        // On Windows, use _aligned_free
        _aligned_free(getHostPointer());
        #else
        // On Unix systems, use free
        ::free(getHostPointer());
        #endif
        
        setHostPointer(nullptr);
        setCount(0);
    }

    //! Copy entire array to device
    virtual void pushToDevice() final
    {
        // ISPC runs on the CPU, so we don't need to do anything
    }

    //! Copy entire array from device
    virtual void pullFromDevice() final
    {
        // ISPC runs on the CPU, so we don't need to do anything
    }

    //! Copy a 1D slice of elements to device 
    virtual void pushSlice1DToDevice(size_t, size_t) final
    {
        /// ISPC runs on the CPU, so we don't need to do anything
    }

    //! Copy a 1D slice of elements from device 
    virtual void pullSlice1DFromDevice(size_t, size_t) final
    {
        // ISPC runs on the CPU, so we don't need to do anything
    }
    
    //! Memset the host pointer
    virtual void memsetDeviceObject(int) final
    {
        throw std::runtime_error("ISPC arrays have no device objects");
    }

    //! Serialise backend-specific device object to bytes
    virtual void serialiseDeviceObject(std::vector<std::byte>&, bool) const final
    {
        throw std::runtime_error("ISPC arrays have no device objects");
    }

    //! Serialise backend-specific host object to bytes
    virtual void serialiseHostObject(std::vector<std::byte>&, bool) const final
    {
        throw std::runtime_error("ISPC arrays have no host objects");
    }
};

//-----------------------------------------------------------------------
// Helper functions for ISPC code generation
//-----------------------------------------------------------------------

// Get appropriate ISPC type for GeNN type
std::string getISPCType(const Type::ResolvedType &type, bool uniform = false)
{
    const std::string prefix = uniform ? "uniform " : "";
    
    if(type == Type::Int8) return prefix + "int8";
    else if(type == Type::Uint8) return prefix + "unsigned int8";
    else if(type == Type::Int16) return prefix + "int16";
    else if(type == Type::Uint16) return prefix + "unsigned int16";
    else if(type == Type::Int32) return prefix + "int32";
    else if(type == Type::Uint32) return prefix + "unsigned int32";
    else if(type == Type::Int64) return prefix + "int64";
    else if(type == Type::Uint64) return prefix + "unsigned int64";
    else if(type == Type::Float) return prefix + "float";
    else if(type == Type::Double) return prefix + "double";
    else if(type == Type::Bool) return prefix + "bool";
    else {
        throw std::runtime_error("Unsupported type for ISPC: " + type.getName());
    }
}

// Format for ISPC kernels to run over all neurons in a population
const char *ispcNeuronKernelTemplate =
R"(export void %1$s(uniform float dt, uniform %2$s* uniform group, uniform unsigned int numNeurons, 
                   uniform unsigned int batchSize)
{
    // Process all batches
    for (uniform unsigned int batchID = 0; batchID < batchSize; batchID++) {
        // Process neurons in SIMD-parallel using a foreach loop
        foreach (n = 0 ... numNeurons) {
            // Calculate offset for current batch
            unsigned int offset = n + (batchID * numNeurons);
            
%3$s
        }
    }
}
)";

// Format for the body of each neuron update
const char *ispcNeuronUpdateTemplate = 
R"(// Neuron update kernel
%1$s

%2$s  // Variable declarations

%3$s  // Update code
)";

} // anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator::ISPC::State
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::ISPC
{
State::State(const Runtime::Runtime &) 
{
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Backend
//--------------------------------------------------------------------------
void Backend::genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                               HostHandler preambleHandler) const
{
    // Get batch size of the model
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    
    // Include necessary headers for ISPC kernels
    os << "#include \"neuronKernels.ispc.h\"" << std::endl;
    os << "#include \"ispc_utils.h\"" << std::endl;
    os << std::endl;
    
    // To generate any additional includes or definitions:
    if(preambleHandler) {
        preambleHandler(os);
    }
    
    // Get merged neuron update groups
    const auto &mergedNeuronUpdateGroups = modelMerged.getMergedNeuronUpdateGroups();
    
    // Get merged spike queue update groups if any
    const auto &mergedNeuronSpikeQueueUpdateGroups = modelMerged.getMergedNeuronSpikeQueueUpdateGroups();
    
    // Get merged previous spike time update groups if any
    const auto &mergedNeuronPrevSpikeTimeUpdateGroups = modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups();

    // Generate custom neuron model code for all neuron groups
    for(const auto &g : mergedNeuronUpdateGroups) {
        const auto &archetype = g.getArchetype();
        const std::string modelType = getNeuronModelType(archetype);
        
        // Generate custom code for non-builtin models
        if(modelType == "Custom") {
            // Generate model-specific code with a unique name based on the group name
            const std::string customModelName = "custom_" + archetype.getName();
            
            // Generate ISPC code for the custom model
            genCustomNeuronModelCode(os, archetype, customModelName);
            os << std::endl;
        }
    }
    
    // Generate C++ wrapper to call ISPC kernels for spike queue update
    if(!mergedNeuronSpikeQueueUpdateGroups.empty()) {
        os << "// Spike queue update function" << std::endl;
        os << "void updateNeuronSpikeQueues()" << std::endl;
        {
            CodeStream::Scope b(os);
            
            // Generate code to update each spike queue
            for(const auto &g : mergedNeuronSpikeQueueUpdateGroups) {
                // Update spike queue
                const std::string memorySpaceName = g.getMemorySpace();
                
                // Check that a memory space has been assigned
                assert(!memorySpaceName.empty());
                
                // Generate code to update spike queue pointer
                CodeStream::Scope ng(os);
                os << "updateNeuronSpikeQueue(&d_mergedSpikeQueueUpdateGroup" << g.getIndex() << ", " << batchSize << ");" << std::endl;
                
                // If there are spike events, also update spike event queue pointers
                const auto &mergedSpikeEventGroups = g.getMergedSpikeEventGroups();
                if(!mergedSpikeEventGroups.empty()) {
                    CodeStream::Scope sg(os);
                    os << "updateNeuronSpikeEventQueue(&d_mergedSpikeEventQueueUpdateGroup" << g.getIndex() << ", " << batchSize << ");" << std::endl;
                }
            }
        }
    }
    
    // Generate C++ wrapper to call ISPC kernels for previous spike time updates
    if(!mergedNeuronPrevSpikeTimeUpdateGroups.empty()) {
        os << "// Previous spike time update function" << std::endl;
        os << "void updatePrevSpikeTimes(float t, float dt)" << std::endl;
        {
            CodeStream::Scope b(os);
            
            // Generate code to update previous spike times for each group
            for(const auto &g : mergedNeuronPrevSpikeTimeUpdateGroups) {
                if(g.getArchetype().isPrevSpikeTimeRequired()) {
                    CodeStream::Scope ng(os);
                    os << "// Update previous spike time for merged group " << g.getIndex() << std::endl;
                    os << "d_mergedPrevSpikeTimeUpdateGroup" << g.getIndex() << ".t = t;" << std::endl;
                    os << "d_mergedPrevSpikeTimeUpdateGroup" << g.getIndex() << ".dt = dt;" << std::endl;
                    os << "updatePrevSpikeTime(&d_mergedPrevSpikeTimeUpdateGroup" << g.getIndex() << ", " << batchSize << ");" << std::endl;
                }
                if(g.getArchetype().isPrevSpikeEventTimeRequired()) {
                    CodeStream::Scope ng(os);
                    os << "// Update previous spike event time for merged group " << g.getIndex() << std::endl;
                    os << "d_mergedPrevSpikeEventTimeUpdateGroup" << g.getIndex() << ".t = t;" << std::endl;
                    os << "d_mergedPrevSpikeEventTimeUpdateGroup" << g.getIndex() << ".dt = dt;" << std::endl;
                    os << "updatePrevSpikeEventTime(&d_mergedPrevSpikeEventTimeUpdateGroup" << g.getIndex() << ", " << batchSize << ");" << std::endl;
                }
            }
        }
    }
    
    // Store neuron model types to determine appropriate ISPC kernels to call
    std::vector<std::string> neuronModelTypes;
    
    // Extract neuron model types for each merged group
    for(const auto &g : mergedNeuronUpdateGroups) {
        const auto &archetype = g.getArchetype();
        neuronModelTypes.push_back(getNeuronModelType(archetype));
    }
    
    // Generate C++ wrapper to call ISPC kernels
    os << "void updateNeurons(float dt, float t)";
    {
        CodeStream::Scope b(os);
        
        // Add timer if desired
        if (getPreferences<Preferences>().debugCode) {
            Timer t(os, "neuronUpdate", true);
        }

        // Update spike queues if needed before neuron update
        if(!mergedNeuronSpikeQueueUpdateGroups.empty()) {
            os << "updateNeuronSpikeQueues();" << std::endl;
        }
        
        // Generate code to update each neuron group
        for(size_t i = 0; i < mergedNeuronUpdateGroups.size(); i++) {
            const auto &g = mergedNeuronUpdateGroups[i];
            const std::string modelType = neuronModelTypes[i];
            const std::string kernelName = getNeuronKernelName(modelType);
            
            // Create memory space name
            const std::string memorySpaceName = g.getMemorySpace();
            
            // Check that a memory space has been assigned
            assert(!memorySpaceName.empty());
            
            // Add call to ISPC kernel for this group based on model type
            os << "// Update neurons for merged group " << i << " (" << modelType << ")" << std::endl;
            if(modelType == "SpikeSourceArray") {
                // SpikeSourceArray takes time (t) instead of dt
                os << kernelName << "(t, &d_mergedNeuronGroup" << i << ", " << 
                    g.getGroups().front().get().getNumNeurons() << ", " << batchSize << ");" << std::endl;
            }
            else if(modelType == "Custom") {
                // For custom neuron models, register function pointers before calling updateCustomNeuron
                const auto &archetype = g.getArchetype();
                const std::string customModelName = "custom_" + archetype.getName();
                
                // Setup function pointers for this custom neuron model
                os << "// Setup function pointers for custom neuron model" << std::endl;
                
                // Sim code function pointer
                if(!archetype.getModel()->getSimCode().empty()) {
                    os << "d_mergedNeuronGroup" << i << ".simCodeFunc = " << customModelName << "_simCode;" << std::endl;
                }
                else {
                    os << "d_mergedNeuronGroup" << i << ".simCodeFunc = NULL;" << std::endl;
                }
                
                // Threshold condition function pointer
                if(!archetype.getModel()->getThresholdConditionCode().empty()) {
                    os << "d_mergedNeuronGroup" << i << ".thresholdConditionFunc = " << customModelName << "_thresholdCondition;" << std::endl;
                }
                else {
                    os << "d_mergedNeuronGroup" << i << ".thresholdConditionFunc = NULL;" << std::endl;
                }
                
                // Reset code function pointer
                if(!archetype.getModel()->getResetCode().empty()) {
                    os << "d_mergedNeuronGroup" << i << ".resetCodeFunc = " << customModelName << "_resetCode;" << std::endl;
                }
                else {
                    os << "d_mergedNeuronGroup" << i << ".resetCodeFunc = NULL;" << std::endl;
                }
                
                // Call the general custom neuron update kernel
                os << "updateCustomNeuron(&d_mergedNeuronGroup" << i << ", NULL, dt, " <<
                    g.getGroups().front().get().getNumNeurons() << ", " << batchSize << ");" << std::endl;
            }
            else {
                // Standard neuron update kernel
                os << kernelName << "(dt, &d_mergedNeuronGroup" << i << ", " << 
                    g.getGroups().front().get().getNumNeurons() << ", " << batchSize << ");" << std::endl;
            }
        }
        
        // Update previous spike times if needed after neuron update
        if(!mergedNeuronPrevSpikeTimeUpdateGroups.empty()) {
            os << "updatePrevSpikeTimes(t, dt);" << std::endl;
        }
    }
    
    // Generate recording related functions
    std::vector<NeuronGroupInternal*> recordableNeuronGroups;
    for(const auto &g : modelMerged.getModel().getNeuronGroups()) {
        // If this group has any recordable variables
        if(!g.second.getRecordableVariables().empty()) {
            recordableNeuronGroups.push_back(&g.second);
        }
    }
    
    if(!recordableNeuronGroups.empty()) {
        os << "// Function to record neuron state variables" << std::endl;
        os << "void recordNeuronState(unsigned int timestep)";
        {
            CodeStream::Scope b(os);
            
            // For each neuron group with recordable variables
            for(const auto *ng : recordableNeuronGroups) {
                const std::string ngName = ng->getName();
                
                // Get all recordable variables for this neuron group
                os << "// Recording variables for neuron group '" << ngName << "'" << std::endl;
                
                const auto &vars = ng->getRecordableVariables();
                for(size_t v = 0; v < vars.size(); v++) {
                    const auto &var = vars[v];
                    os << "if(recordingEnabled_" << ngName << "_" << var.name << ") {" << std::endl;
                    os << "    recordNeuronVariable(&d_" << ngName << ", " << v << ", " <<
                       "recordingBuffer_" << ngName << "_" << var.name << ", " <<
                       ng->getNumNeurons() << ", timestep, " << batchSize << ");" << std::endl;
                    os << "}" << std::endl;
                }
                
                // Record spike counts if needed
                if(ng->isSpikeRecordingEnabled()) {
                    os << "if(recordingEnabled_" << ngName << "_spikes) {" << std::endl;
                    os << "    recordNeuronSpikeCount(&d_" << ngName << "_spikesGroup, " <<
                       "recordingBuffer_" << ngName << "_spikes, " <<
                       ng->getNumNeurons() << ", timestep, " << batchSize << ");" << std::endl;
                    os << "}" << std::endl;
                }
            }
        }
    }
}

//Further milestone stuff beyond this point

void Backend::genSynapseUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                               HostHandler preambleHandler) const
{
    // Placeholder for synapse update implementation
    os << "// Synapse update to be implemented" << std::endl;
}

void Backend::genCustomUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                              HostHandler preambleHandler) const
{
    // Placeholder for custom update implementation
    os << "// Custom update to be implemented" << std::endl;
}

void Backend::genInit(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                      HostHandler preambleHandler) const
{
    // Get model batch size
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    
    // Initialize the batch-specific RNG if batching is enabled
    if(batchSize > 1) {
        os << "// Initialize batch-specific RNG" << std::endl;
        os << "initBatchRNG(" << modelMerged.getModel().getSeed() << ", " << batchSize << ");" << std::endl;
    }
    
    // Rest of initialization implementation
    os << "// Initialization to be implemented" << std::endl;
}

size_t Backend::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    // Row stride? It is just the number of columns in the synaptic matrix
    // For sparse matrices this is the maximum connections
    if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxConnections();
    }
    // For dense matrices this is simply the number of neurons in the target population
    else {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
}

void Backend::genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    // Include ISPC headers
    os << "#include <cmath>" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << "#include <random>" << std::endl;
    os << "#include \"updateNeurons_ispc.h\"" << std::endl;
    os << std::endl;
    
    // Add recording-related variable declarations
    os << "// Number of timesteps to record" << std::endl;
    os << "unsigned int numRecordingTimesteps = 10000;  // Default value, can be changed" << std::endl;
    os << std::endl;
    
    // Declare recording buffers and flags for each recordable variable
    for(const auto &g : modelMerged.getModel().getNeuronGroups()) {
        // If this group has any recordable variables
        const auto &vars = g.second.getRecordableVariables();
        if(!vars.empty()) {
            const std::string ngName = g.first;
            
            for(const auto &var : vars) {
                os << "// Recording buffer and flag for " << ngName << "." << var.name << std::endl;
                os << "float *recordingBuffer_" << ngName << "_" << var.name << " = NULL;" << std::endl;
                os << "bool recordingEnabled_" << ngName << "_" << var.name << " = false;" << std::endl;
            }
            
            // Declare spike recording buffer if needed
            if(g.second.isSpikeRecordingEnabled()) {
                os << "// Spike recording buffer and flag for " << ngName << std::endl;
                os << "unsigned int *recordingBuffer_" << ngName << "_spikes = NULL;" << std::endl;
                os << "bool recordingEnabled_" << ngName << "_spikes = false;" << std::endl;
            }
        }
    }
    os << std::endl;
}

void Backend::genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    const auto &preferences = getPreferences<Preferences>();

    // If timing is enabled
    if(preferences.debugCode) {
        // Include chrono header
        os << "#include <chrono>" << std::endl;
        
        // Define time variables
        os << "double neuronUpdateTime = 0.0;" << std::endl;
        os << "double synapseUpdateTime = 0.0;" << std::endl;
        os << "double initTime = 0.0;" << std::endl;
    }
}

void Backend::genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    // Allocate RNG state for batches if needed
    if(modelMerged.getModel().getBatchSize() > 1) {
        os << "// Initialize batch-specific RNGs" << std::endl;
        os << "initBatchRNG(" << modelMerged.getModel().getSeed() << ", " << modelMerged.getModel().getBatchSize() << ");" << std::endl;
    }
    else {
        os << "// Initialize global RNG" << std::endl;
        os << "initRNG(" << modelMerged.getModel().getSeed() << ");" << std::endl;
    }
    
    // Allocate recording buffers for recordable variables if any
    for(const auto &g : modelMerged.getModel().getNeuronGroups()) {
        // If this group has any recordable variables
        const auto &vars = g.second.getRecordableVariables();
        if(!vars.empty()) {
            const std::string ngName = g.first;
            const unsigned int numNeurons = g.second.getNumNeurons();
            
            for(const auto &var : vars) {
                os << "// Allocate recording buffer for " << ngName << "." << var.name << std::endl;
                os << "recordingBuffer_" << ngName << "_" << var.name << " = (float*)_aligned_malloc(sizeof(float) * " << 
                   numNeurons << " * numRecordingTimesteps, 64);" << std::endl;
                os << "recordingEnabled_" << ngName << "_" << var.name << " = false;" << std::endl;
            }
            
            // Allocate spike recording buffer if needed
            if(g.second.isSpikeRecordingEnabled()) {
                os << "// Allocate spike recording buffer for " << ngName << std::endl;
                os << "recordingBuffer_" << ngName << "_spikes = (unsigned int*)_aligned_malloc(sizeof(unsigned int) * " << 
                   numNeurons << " * numRecordingTimesteps, 64);" << std::endl;
                os << "recordingEnabled_" << ngName << "_spikes = false;" << std::endl;
            }
        }
    }
}

void Backend::genFreeMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    // Free recording buffers for recordable variables if any
    for(const auto &g : modelMerged.getModel().getNeuronGroups()) {
        // If this group has any recordable variables
        const auto &vars = g.second.getRecordableVariables();
        if(!vars.empty()) {
            const std::string ngName = g.first;
            
            for(const auto &var : vars) {
                os << "// Free recording buffer for " << ngName << "." << var.name << std::endl;
                os << "if(recordingBuffer_" << ngName << "_" << var.name << ") {" << std::endl;
                os << "    _aligned_free(recordingBuffer_" << ngName << "_" << var.name << ");" << std::endl;
                os << "    recordingBuffer_" << ngName << "_" << var.name << " = NULL;" << std::endl;
                os << "}" << std::endl;
            }
            
            // Free spike recording buffer if needed
            if(g.second.isSpikeRecordingEnabled()) {
                os << "// Free spike recording buffer for " << ngName << std::endl;
                os << "if(recordingBuffer_" << ngName << "_spikes) {" << std::endl;
                os << "    _aligned_free(recordingBuffer_" << ngName << "_spikes);" << std::endl;
                os << "    recordingBuffer_" << ngName << "_spikes = NULL;" << std::endl;
                os << "}" << std::endl;
            }
        }
    }
}

void Backend::genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    // Add recording logic to step time finalize
    bool hasRecordable = false;
    for(const auto &g : modelMerged.getModel().getNeuronGroups()) {
        if(!g.second.getRecordableVariables().empty() || g.second.isSpikeRecordingEnabled()) {
            hasRecordable = true;
            break;
        }
    }
    
    if(hasRecordable) {
        os << "// Record neuron state variables for this timestep if recording is enabled" << std::endl;
        os << "if(recordingTimestep < numRecordingTimesteps) {" << std::endl;
        os << "    recordNeuronState(recordingTimestep);" << std::endl;
        os << "    recordingTimestep++;" << std::endl;
        os << "}" << std::endl;
    }
}

//! Create backend-specific runtime state object
std::unique_ptr<GeNN::Runtime::StateBase> Backend::createState(const Runtime::Runtime &runtime) const
{
    return std::make_unique<State>(runtime);
}

std::unique_ptr<Runtime::ArrayBase> Backend::createArray(const Type::ResolvedType &type, size_t count, 
                                                         VarLocation location, bool uninitialized) const
{
    return std::make_unique<Array>(type, count, location, uninitialized);
}

std::unique_ptr<Runtime::ArrayBase> Backend::createPopulationRNG(size_t count) const
{
    // To be implemented
    return std::unique_ptr<Runtime::ArrayBase>();
}

void Backend::genLazyVariableDynamicAllocation(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                            const std::string &countVarName) const
{
    // Generate code to allocate memory aligned for SIMD access
    os << name << " = (" << type.getValue().name << "*)_aligned_malloc(sizeof(" << type.getValue().name << ") * " << countVarName << ", 64);" << std::endl;
    os << "if(!" << name << ") throw std::bad_alloc();" << std::endl;
}

void Backend::genLazyVariableDynamicPush(CodeStream &os, 
                                      const Type::ResolvedType &type, const std::string &name,
                                      VarLocation loc, const std::string &countVarName) const
{
    // No need to push to device for ISPC
}

void Backend::genLazyVariableDynamicPull(CodeStream &os, 
                                      const Type::ResolvedType &type, const std::string &name,
                                      VarLocation loc, const std::string &countVarName) const
{
    // No need to pull from device for ISPC
}

void Backend::genMergedDynamicVariablePush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                        const std::string &groupIdx, const std::string &fieldName,
                                        const std::string &egpName) const
{
    // No need to push to device for ISPC
}

std::string Backend::getMergedGroupFieldHostTypeName(const Type::ResolvedType &type) const
{
    return type.getValue().name;
}

void Backend::genPopVariableInit(EnvironmentExternalBase &env, HandlerEnv handler) const
{
    // Add little block scope using standard pattern
    {
        handler(env);
    }
}

void Backend::genVariableInit(EnvironmentExternalBase &env, const std::string &count, const std::string &indexVarName, HandlerEnv handler) const
{
    // Generate simple for loop through variables
    env.print("for(unsigned int " + indexVarName + " = 0; " + indexVarName + " < " + count + "; " + indexVarName + "++)");
    {
        CodeStream::Scope b(env.getStream());
        handler(env);
    }
}

void Backend::genSparseSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const
{
    // To be implemented
    handler(env);
}

void Backend::genDenseSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const
{
    handler(env);
}

void Backend::genKernelSynapseVariableInit(EnvironmentExternalBase &env, SynapseInitGroupMerged &sg, HandlerEnv handler) const
{
    // To be implemented
    handler(env);
}

void Backend::genKernelCustomUpdateVariableInit(EnvironmentExternalBase &env, CustomWUUpdateInitGroupMerged &cu, HandlerEnv handler) const
{
    // To be implemented
    handler(env);
}

std::string Backend::getAtomicOperation(const std::string &lhsPointer, const std::string &rhsValue,
                                      const Type::ResolvedType &type, AtomicOperation op) const
{
    // Basic atomic operations for ISPC
    std::string operation;
    if(op == AtomicOperation::ADD) {
        operation = " += ";
    }
    else {
        assert(op == AtomicOperation::OR);
        operation = " |= ";
    }
    return "*(" + lhsPointer + ")" + operation + "(" + rhsValue + ")";
}

void Backend::genGlobalDeviceRNG(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free) const
{
    // Define random number generator
    definitions << "// Random number generator" << std::endl;
    definitions << "std::mt19937 rng;" << std::endl;
    
    // Seed RNG in allocate function
    allocations << "// Seed RNG" << std::endl;
    allocations << "rng.seed(42);" << std::endl;
}

void Backend::genTimer(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free, 
                    CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const
{
    // Add timer variable to definitions
    definitions << "double " << name << "Time = 0.0;" << std::endl;
    
    // Reset timer in allocations
    allocations << name << "Time = 0.0;" << std::endl;
    
    // Add timing output if requested
    if(updateInStepTime) {
        stepTimeFinalise << "printf(\"" << name << " time: %f s\\n\", " << name << "Time);" << std::endl;
    }
}

void Backend::genReturnFreeDeviceMemoryBytes(CodeStream &os) const
{
    // ISPC uses host memory
    os << "return 0;" << std::endl;
}

void Backend::genAssert(CodeStream &os, const std::string &condition) const
{
    os << "assert(" << condition << ");" << std::endl;
}

void Backend::genMakefilePreamble(std::ostream &os) const
{
    const auto &preferences = getPreferences<Preferences>();
    
    // Add ISPC compiler settings to makefile
    os << "ISPC              :=ispc" << std::endl;
    os << "ISPC_FLAGS        :=";
    
    // Target instruction set
    os << "--target=" << preferences.targetISA;
    
    // Optimization level 
    if(preferences.optimizeForSize) {
        os << " -O1";  // Optimize for size
    }
    else if(preferences.optimizeCode) {
        os << " -O3";  // Maximum optimization
    }
    else {
        os << " -O2";  // Standard optimization
    }
    
    // Debug symbols if requested
    if(preferences.debugCode) {
        os << " -g";
    }
    
    os << std::endl;
}

void Backend::genMakefileLinkRule(std::ostream &os) const
{
    // Link with ISPC objects
    os << "\t@$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS) $(ISPC_OBJECTS) $(LDFLAGS)" << std::endl;
}

void Backend::genMakefileCompileRule(std::ostream &os) const
{
    // Add ISPC compile rule for ISPC files
    os << "%.o: %.ispc" << std::endl;
    os << "\t@$(ISPC) $(ISPC_FLAGS) -o $@ $<" << std::endl;
}

void Backend::genMSBuildConfigProperties(std::ostream &os) const
{
    // Add MSBuild properties for ISPC
    os << "<ISPCPath Condition=\"'$(ISPCPath)'==''\">ispc</ISPCPath>" << std::endl;
    
    const auto &preferences = getPreferences<Preferences>();
    
    // Add ISPC flags based on preferences
    os << "<ISPCFlags>";
    
    // Target instruction set
    os << "--target=" << preferences.targetISA;
    
    // Optimization level 
    if(preferences.optimizeForSize) {
        os << " -O1";  // Optimize for size
    }
    else if(preferences.optimizeCode) {
        os << " -O3";  // Maximum optimization
    }
    else {
        os << " -O2";  // Standard optimization
    }
    
    // Debug symbols if requested
    if(preferences.debugCode) {
        os << " -g";
    }
    
    os << "</ISPCFlags>" << std::endl;
}

void Backend::genMSBuildImportProps(std::ostream &os) const
{
    // Not needed for ISPC
}

void Backend::genMSBuildItemDefinitions(std::ostream &os) const
{
    // Define ISPC compilation steps for MSBuild
    os << "<ISPC>" << std::endl;
    os << "  <Command>$(ISPCPath) %(ISPCFlags) -o \"$(IntDir)%(Filename).obj\" \"%(FullPath)\"</Command>" << std::endl;
    os << "  <Outputs>$(IntDir)%(Filename).obj</Outputs>" << std::endl;
    os << "</ISPC>" << std::endl;
}

void Backend::genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const
{
    // Add ISPC compile step for module
    os << "<ISPC Include=\"" << moduleName << ".ispc\" />" << std::endl;
}

void Backend::genMSBuildImportTarget(std::ostream &os) const
{
    // Not needed for ISPC
}

std::vector<filesystem::path> Backend::getFilesToCopy(const ModelSpecMerged&) const
{
    // Copy ISPC helper files if needed
    return {};
}

bool Backend::isGlobalHostRNGRequired(const ModelSpecInternal &model) const
{
    // Check if any neuron or synapse groups require an RNG
    return std::any_of(model.getNeuronGroups().cbegin(), model.getNeuronGroups().cend(),
                     [](const ModelSpecInternal::NeuronGroupValueType &n) { return n.second.isSimRNGRequired(); }) ||
           std::any_of(model.getSynapseGroups().cbegin(), model.getSynapseGroups().cend(),
                     [](const ModelSpecInternal::SynapseGroupValueType &s) { return s.second.isSimRNGRequired(); });
}

bool Backend::isGlobalDeviceRNGRequired(const ModelSpecInternal &) const
{
    // ISPC uses host RNG
    return false;
}

BackendBase::MemorySpaces Backend::getMergedGroupMemorySpaces(const ModelSpecMerged &) const
{
    // ISPC doesn't use special memory spaces
    return {};
}

boost::uuids::detail::sha1::digest_type Backend::getHashDigest() const
{
    const auto &preferences = getPreferences<Preferences>();
    
    boost::uuids::detail::sha1 hash;
    
    // Update hash with preferences
    preferences.updateHash(hash);
    
    // Update hash with backend name
    Utils::updateHash("ISPC", hash);
    
    return hash.get_digest();
}

void Backend::genPresynapticUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                   double dt, bool trueSpike) const
{
    // To be implemented
}

void Backend::genPostsynapticUpdate(EnvironmentExternalBase &env, PostsynapticUpdateGroupMerged &sg, 
                                    double dt, bool trueSpike) const
{
    // To be implemented
}

void Backend::genPrevEventTimeUpdate(EnvironmentExternalBase &env, NeuronPrevSpikeTimeUpdateGroupMerged &ng,
                                    bool trueSpike) const
{
    // Get the suffix based on whether we're working with true spikes or spike events
    const std::string suffix = trueSpike ? "" : "_event";
    const std::string prevTime = trueSpike ? "prevSpikeTime" : "prevSpikeEventTime";
    
    // If the current group requires delays
    if((trueSpike && ng.getArchetype().isSpikeDelayRequired()) ||
       (!trueSpike && ng.getArchetype().isSpikeEventDelayRequired()))
    {
        // Generate if condition to only process spikes within range
        env.print("if($(id) < $(_spk_cnt" + suffix + ")[lastTimestepDelaySlot])");
        {
            CodeStream::Scope b(env.getStream());
            env.printLine("$(_prev_" + (trueSpike ? "st" : "set") + ")[lastTimestepDelayOffset + $(_spk" + suffix + ")[lastTimestepDelayOffset + $(id)]] = $(t) - $(dt);");
        }
    }
    else
    {
        // Generate if condition to only process spikes within range
        env.print("if($(id) < $(_spk_cnt" + suffix + ")[$(batch)])");
        {
            CodeStream::Scope b(env.getStream());
            if(ng.getArchetype().getModel().getBatchSize() > 1) {
                env.printLine("$(_prev_" + (trueSpike ? "st" : "set") + ")[$(_batch_offset) + $(_spk" + suffix + ")[$(_batch_offset) + $(id)]] = $(t) - $(dt);");
            }
            else {
                env.printLine("$(_prev_" + (trueSpike ? "st" : "set") + ")[" + "$(_spk" + suffix + ")[$(id)]] = $(t) - $(dt);");
            }
        }
    }
}

void Backend::genEmitEvent(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng, bool trueSpike) const
{
    // To be implemented
}

template<typename G, typename R>
void Backend::genWriteBackReductions(EnvironmentExternalBase &env, G &cg, const std::string &idxName, R getVarRefIndexFn) const
{
    const auto *cm = cg.getArchetype().getModel();
    for(const auto &v : cm->getVars()) {
        // If variable is a reduction target, copy value from register straight back into global memory
        if(v.access & VarAccessModeAttribute::REDUCE) {
            const std::string idx = env.getName(idxName);
            const VarAccessDim varAccessDim = getVarAccessDim(v.access, cg.getArchetype().getDims());
            env.getStream() << "group->" << v.name << "[" << cg.getVarIndex(1, varAccessDim, idx) << "] = " << env[v.name] << ";" << std::endl;
        }
    }

    // Loop through all variable references
    for(const auto &modelVarRef : cm->getVarRefs()) {
        const auto &varRef = cg.getArchetype().getVarReferences().at(modelVarRef.name);

        // If variable reference is a reduction target, copy value from register straight back into global memory
        if(modelVarRef.access & VarAccessModeAttribute::REDUCE) {
            const std::string idx = env.getName(idxName);
            env.getStream() << "group->" << modelVarRef.name << "[" << getVarRefIndexFn(varRef, idx) << "] = " << env[modelVarRef.name] << ";" << std::endl;
        }
    }
}

// Generate custom neuron model code for ISPC. Test Implementation.
void Backend::genCustomNeuronModelCode(CodeStream &os, const NeuronGroupInternal &ng, 
                                     const std::string &modelName) const
{
    // Get the neuron model
    const auto *model = ng.getModel();
    
    // Get model variables and parameters
    const auto &vars = model->getVars();
    const auto &params = model->getParams();
    
    // Create variable access code
    std::stringstream varAccessStream;
    for (size_t i = 0; i < vars.size(); i++) {
        varAccessStream << "    varying " << getISPCType(vars[i].type.resolve(ng.getTypeContext())) << " " << vars[i].name 
                       << " = *((" << getISPCType(vars[i].type.resolve(ng.getTypeContext())) << "*)group->stateVarPointers[" << i << "] + offset);" << std::endl;
    }
    std::string varAccessCode = varAccessStream.str();
    
    // Create parameter access code
    std::stringstream paramAccessStream;
    for (size_t i = 0; i < params.size(); i++) {
        paramAccessStream << "    varying " << getISPCType(params[i].type.resolve(ng.getTypeContext())) << " " << params[i].name 
                         << " = *((" << getISPCType(params[i].type.resolve(ng.getTypeContext())) << "*)group->paramValues[" << i << "]);" << std::endl;
    }
    std::string paramAccessCode = paramAccessStream.str();
    
    // Generate sim code function if present
    const std::string &simCode = model->getSimCode();
    if (!simCode.empty()) {
        // Adapt simulation code to use ISPC syntax
        std::string ispcSimCode = "    " + Utils::replaceCPPStdNamespace(simCode);
        ispcSimCode = Utils::findAndReplace(ispcSimCode, "\n", "\n    "); // Indent each line
        
        // Output simulation code function
        os << Utils::formatString(ispcCustomNeuronSimCodeTemplate, modelName, 
                                varAccessCode, paramAccessCode, ispcSimCode);
    }
    
    // Generate threshold condition function if present
    const std::string &thresholdConditionCode = model->getThresholdConditionCode();
    if (!thresholdConditionCode.empty()) {
        // Output threshold condition function
        os << Utils::formatString(ispcCustomNeuronThresholdTemplate, modelName, 
                                varAccessCode, paramAccessCode, thresholdConditionCode);
    }
    
    // Generate reset code function if present
    const std::string &resetCode = model->getResetCode();
    if (!resetCode.empty()) {
        // Adapt reset code to use ISPC syntax
        std::string ispcResetCode = "    " + Utils::replaceCPPStdNamespace(resetCode);
        ispcResetCode = Utils::findAndReplace(ispcResetCode, "\n", "\n    "); // Indent each line
        
        // Output reset code function
        os << Utils::formatString(ispcCustomNeuronResetTemplate, modelName, 
                                varAccessCode, paramAccessCode, ispcResetCode);
    }
}

}   // namespace GeNN::CodeGenerator::ISPC

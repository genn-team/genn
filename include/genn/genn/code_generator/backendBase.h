#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// Filesystem includes
#include "path.h"

// PLOG includes
#include <plog/Severity.h>

// GeNN includes
#include "codeStream.h"
#include "gennExport.h"
#include "variableMode.h"

// Forward declarations
class CustomUpdateInternal;
class CustomUpdateWUInternal;
class NeuronGroupInternal;
class SynapseGroupInternal;

namespace CodeGenerator
{
    class ModelSpecMerged;
    class NeuronUpdateGroupMerged;
    class Substitutions;
    class SynapseGroupMergedBase;
    class PresynapticUpdateGroupMerged;
    class PostsynapticUpdateGroupMerged;
    class SynapseDynamicsGroupMerged;
    class CustomUpdateGroupMerged;
    class CustomUpdateWUGroupMerged;
    class CustomUpdateTransposeWUGroupMerged;
    class NeuronInitGroupMerged;
    class CustomUpdateInitGroupMerged;
    class CustomWUUpdateDenseInitGroupMerged;
    class CustomWUUpdateSparseInitGroupMerged;
    class SynapseConnectivityInitGroupMerged;
    class SynapseDenseInitGroupMerged;
    class SynapseSparseInitGroupMerged;
    
}

//--------------------------------------------------------------------------
// CodeGenerator::PreferencesBase
//--------------------------------------------------------------------------
namespace CodeGenerator
{
//! Base class for backend preferences - can be accessed via a global in 'classic' C++ code generator
struct PreferencesBase
{
    //! Generate speed-optimized code, potentially at the expense of floating-point accuracy
    bool optimizeCode = false;

    //! Generate code with debug symbols
    bool debugCode = false;

    //! New optimizations made to kernels for simulating synapse groups with BITMASK connectivity
    //! Improve performance but break backward compatibility due to word-padding each row
    bool enableBitmaskOptimisations = false;

    //! If backend/device supports it, copy data automatically when required rather than requiring push and pull
    bool automaticCopy = false;

    //! Should GeNN generate empty state push and pull functions
    bool generateEmptyStatePushPull = true;

    //! Should GeNN generate pull functions for extra global parameters? These are very rarely used
    bool generateExtraGlobalParamPull = true;

    //! C++ compiler options to be used for building all host side code (used for unix based platforms)
    std::string userCxxFlagsGNU = "";

    //! NVCC compiler options they may want to use for all GPU code (used for unix based platforms)
    std::string userNvccFlagsGNU = "";

    //! Logging level to use for code generation
    plog::Severity logLevel = plog::info;
};

//--------------------------------------------------------------------------
// CodeGenerator::MemAlloc
//--------------------------------------------------------------------------
class MemAlloc
{
public:
    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    size_t getHostBytes() const{ return m_HostBytes; }
    size_t getDeviceBytes() const{ return m_DeviceBytes; }
    size_t getZeroCopyBytes() const{ return m_ZeroCopyBytes; }
    size_t getHostMBytes() const{ return m_HostBytes / (1024 * 1024); }
    size_t getDeviceMBytes() const{ return m_DeviceBytes / (1024 * 1024); }
    size_t getZeroCopyMBytes() const{ return m_ZeroCopyBytes / (1024 * 1024); }

    //--------------------------------------------------------------------------
    // Operators
    //--------------------------------------------------------------------------
    MemAlloc &operator+=(const MemAlloc& rhs){
        m_HostBytes += rhs.m_HostBytes;
        m_DeviceBytes += rhs.m_DeviceBytes;
        m_ZeroCopyBytes += rhs.m_ZeroCopyBytes;
        return *this;
    }

    //--------------------------------------------------------------------------
    // Static API
    //--------------------------------------------------------------------------
    static MemAlloc zero(){ return MemAlloc(0, 0, 0); }
    static MemAlloc host(size_t hostBytes){ return MemAlloc(hostBytes, 0, 0); }
    static MemAlloc hostDevice(size_t bytes) { return MemAlloc(bytes, bytes, 0); }
    static MemAlloc device(size_t deviceBytes){ return MemAlloc(0, deviceBytes, 0); }
    static MemAlloc zeroCopy(size_t zeroCopyBytes){ return MemAlloc(0, 0, zeroCopyBytes); }

private:
    MemAlloc(size_t hostBytes, size_t deviceBytes, size_t zeroCopyBytes)
    :   m_HostBytes(hostBytes), m_DeviceBytes(deviceBytes), m_ZeroCopyBytes(zeroCopyBytes)
    {
    }

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    size_t m_HostBytes;
    size_t m_DeviceBytes;
    size_t m_ZeroCopyBytes;
};

//--------------------------------------------------------------------------
// CodeGenerator::BackendBase
//--------------------------------------------------------------------------
class GENN_EXPORT BackendBase
{
public:
    //--------------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------------
    typedef std::function<void(CodeStream &)> HostHandler;

    typedef std::function<void(CodeStream &, Substitutions&)> Handler;
    
    template<typename T>
    using GroupHandler = std::function <void(CodeStream &, const T &, Substitutions&)> ;

    //! Standard callback type which provides a CodeStream to write platform-independent code for the specified SynapseGroup to.
    typedef GroupHandler<NeuronUpdateGroupMerged> NeuronUpdateGroupMergedHandler;
    typedef GroupHandler<PresynapticUpdateGroupMerged> PresynapticUpdateGroupMergedHandler;
    typedef GroupHandler<PostsynapticUpdateGroupMerged> PostsynapticUpdateGroupMergedHandler;
    typedef GroupHandler<SynapseDynamicsGroupMerged> SynapseDynamicsGroupMergedHandler;
    typedef GroupHandler<CustomUpdateGroupMerged> CustomUpdateGroupMergedHandler;
    typedef GroupHandler<CustomUpdateWUGroupMerged> CustomUpdateWUGroupMergedHandler;
    typedef GroupHandler<CustomUpdateTransposeWUGroupMerged> CustomUpdateTransposeWUGroupMergedHandler;
    typedef GroupHandler<NeuronInitGroupMerged> NeuronInitGroupMergedHandler;
    typedef GroupHandler<CustomUpdateInitGroupMerged> CustomUpdateInitGroupMergedHandler;
    typedef GroupHandler<CustomWUUpdateDenseInitGroupMerged> CustomWUUpdateDenseInitGroupMergedHandler;
    typedef GroupHandler<CustomWUUpdateSparseInitGroupMerged> CustomWUUpdateSparseInitGroupMergedHandler;
    typedef GroupHandler<SynapseConnectivityInitGroupMerged> SynapseConnectivityInitMergedGroupHandler;
    typedef GroupHandler<SynapseDenseInitGroupMerged> SynapseDenseInitGroupMergedHandler;
    typedef GroupHandler<SynapseSparseInitGroupMerged> SynapseSparseInitGroupMergedHandler;
    
    //! Callback function type for generation neuron group simulation code
    /*! Provides additional callbacks to insert code to emit spikes */
    typedef std::function <void(CodeStream &, const NeuronUpdateGroupMerged &, Substitutions&,
                                NeuronUpdateGroupMergedHandler, NeuronUpdateGroupMergedHandler)> NeuronGroupSimHandler;
    
    //! Vector of prefixes required to allocate in memory space and size of memory space
    typedef std::vector<std::pair<std::string, size_t>> MemorySpaces;

    BackendBase(const std::string &scalarType, const PreferencesBase &preferences);
    virtual ~BackendBase(){}

    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    //! Generate platform-specific function to update the state of all neurons
    /*! \param os                       CodeStream to write function to
        \param modelMerged              merged model to generate code for
        \param memorySpaces             for supported backends, data structure containing the remaining space in different named memory spaces
        \param preambleHandler          callback to write functions for pushing extra-global parameters
        \param simHandler               callback to write platform-independent code to update an individual NeuronGroup
        \param wuVarUpdateHandler       callback to write platform-independent code to update pre and postsynaptic weight update model variables when neuron spikes
        \param pushEGPHandler           callback to write required extra-global parameter pushing code to start of synapseUpdate function*/
    virtual void genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces, 
                                 HostHandler preambleHandler, NeuronGroupSimHandler simHandler, NeuronUpdateGroupMergedHandler wuVarUpdateHandler,
                                 HostHandler pushEGPHandler) const = 0;

    //! Generate platform-specific function to update the state of all synapses
    /*! \param os                           CodeStream to write function to
        \param modelMerged                  merged model to generate code for
        \param memorySpaces                 for supported backends, data structure containing the remaining space in different named memory spaces
        \param preambleHandler              callback to write functions for pushing extra-global parameters
        \param wumThreshHandler             callback to write platform-independent code to update an individual NeuronGroup
        \param wumSimHandler                callback to write platform-independent code to process presynaptic spikes.
                                            "id_pre", "id_post" and "id_syn" variables; and either "addToInSynDelay" or "addToInSyn" function will be provided
                                            to callback via Substitutions.
        \param wumEventHandler              callback to write platform-independent code to process presynaptic spike-like events.
                                            "id_pre", "id_post" and "id_syn" variables; and either "addToInSynDelay" or "addToInSyn" function will be provided
                                            to callback via Substitutions.
        \param wumProceduralConnectHandler  callback to write platform-indepent code to procedurally generate connectivity
                                            "id_pre" variable and "addSynapse" function will be provided to callback via Substitutions.
                                            callback needs to implement loop over synapses in row, providing "synAddress" variable if INDIVIDUALG
        \param postLearnHandler             callback to write platform-independent code to process postsynaptic spikes.
                                            "id_pre", "id_post" and "id_syn" variables will be provided to callback via Substitutions.
        \param synapseDynamicsHandler       callback to write platform-independent code to update time-driven synapse dynamics.
                                            "id_pre", "id_post" and "id_syn" variables; and either "addToInSynDelay" or "addToInSyn" function will be provided
                                            to callback via Substitutions.
        \param pushEGPHandler               callback to write required extra-global parameter pushing code to start of synapseUpdate function*/
    virtual void genSynapseUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces,
                                  HostHandler preambleHandler, PresynapticUpdateGroupMergedHandler wumThreshHandler, PresynapticUpdateGroupMergedHandler wumSimHandler,
                                  PresynapticUpdateGroupMergedHandler wumEventHandler, PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler,
                                  PostsynapticUpdateGroupMergedHandler postLearnHandler, SynapseDynamicsGroupMergedHandler synapseDynamicsHandler,
                                  HostHandler pushEGPHandler) const = 0;

    //! Generate platform-specific functions to perform custom updates
    /*! \param os                           CodeStream to write function to
        \param modelMerged                  merged model to generate code for
        \param memorySpaces                 for supported backends, data structure containing the remaining space in different named memory spaces
        \param preambleHandler              callback to write functions for pushing extra-global parameters
      
        \param pushEGPHandler               callback to write required extra-global parameter pushing code to start of synapseUpdate function*/
    virtual void genCustomUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces, HostHandler preambleHandler, 
                                 CustomUpdateGroupMergedHandler customUpdateHandler, CustomUpdateWUGroupMergedHandler customWUUpdateHandler, 
                                 CustomUpdateTransposeWUGroupMergedHandler customWUTransposeUpdateHandler, HostHandler pushEGPHandler) const = 0;

    //! Generate platform-specific function to initialise model
    /*! \param os                           CodeStream to write function to
        \param modelMerged                  merged model to generate code for
        \param memorySpaces                 for supported backends, data structure containing the remaining space in different named memory spaces
        \param preambleHandler              callback to write functions for pushing extra-global parameters
        \param localNGHandler               callback to write platform-independent code to initialize a merged neuron group
        \param sgDenseInitHandler           callback to write platform-independent code to initialize the synapse variables of
                                            a merged synapse group with dense connectivity
        \param sgSparseRowConnectHandler    callback to write platform-independent code to initialize sparse connectivity
                                            for a merged synapse group with a row-wise connectivity initialisation snippet
        \param sgSparseColConnectHandler    callback to write platform-independent code to initialize sparse connectivity
                                            for a merged synapse group with a column-wise connectivity initialisation snippet
        \param sgKernelInitHandler          callback to write platform-independent code to initialize the synapse variables  of
                                            merged synapse group variables which require a kernel
        \param sgSparseInitHandler          callback to write platform-independent code to initialize the synapse variables of
                                            a merged synapse group with sparse connectivity
        \param initPushEGPHandler           callback to write required extra-global parameter pushing code to start of initialize function
        \param initSparsePushEGPHandler     callback to write required extra-global parameter pushing code to start of initialize function*/
    virtual void genInit(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces,
                         HostHandler preambleHandler, NeuronInitGroupMergedHandler localNGHandler, CustomUpdateInitGroupMergedHandler cuHandler,
                         CustomWUUpdateDenseInitGroupMergedHandler cuDenseHandler, SynapseDenseInitGroupMergedHandler sgDenseInitHandler, 
                         SynapseConnectivityInitMergedGroupHandler sgSparseRowConnectHandler,  SynapseConnectivityInitMergedGroupHandler sgSparseColConnectHandler, 
                         SynapseConnectivityInitMergedGroupHandler sgKernelInitHandler, SynapseSparseInitGroupMergedHandler sgSparseInitHandler, 
                         CustomWUUpdateSparseInitGroupMergedHandler cuSparseHandler, HostHandler initPushEGPHandler, HostHandler initSparsePushEGPHandler) const = 0;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const = 0;
    
    //! Definitions is the usercode-facing header file for the generated code. This function generates a 'preamble' to this header file.
    /*! This will be included from a standard C++ compiler so shouldn't include any platform-specific types or headers*/
    virtual void genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    //! Definitions internal is the internal header file for the generated code. This function generates a 'preamble' to this header file.
    /*! This will only be included by the platform-specific compiler used to build this backend so can include platform-specific types or headers*/
    virtual void genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const = 0;

    //! Allocate memory is the first function in GeNN generated code called by usercode and it should only ever be called once.
    //! Therefore it's a good place for any global initialisation. This function generates a 'preamble' to this function.
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const = 0;

    //! After all timestep logic is complete
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    virtual void genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count, MemAlloc &memAlloc) const = 0;
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const = 0;

    virtual void genExtraGlobalParamDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name, 
                                               VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const = 0;
    virtual void genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name, 
                                         VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const = 0;
    virtual void genExtraGlobalParamPull(CodeStream &os, const std::string &type, const std::string &name, 
                                         VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const = 0;

    //! Generate code for pushing an updated EGP value into the merged group structure on 'device'
    virtual void genMergedExtraGlobalParamPush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                               const std::string &groupIdx, const std::string &fieldName,
                                               const std::string &egpName) const = 0;

    //! When generating function calls to push to merged groups, backend without equivalent of Unified Virtual Addressing e.g. OpenCL 1.2 may use different types on host
    virtual std::string getMergedGroupFieldHostType(const std::string &type) const = 0;

    //! When generating merged structures what type to use for simulation RNGs
    virtual std::string getMergedGroupSimRNGType() const = 0;

    virtual void genPopVariableInit(CodeStream &os, const Substitutions &kernelSubs, Handler handler) const = 0;
    virtual void genVariableInit(CodeStream &os, const std::string &count, const std::string &indexVarName,
                                 const Substitutions &kernelSubs, Handler handler) const = 0;
    virtual void genSparseSynapseVariableRowInit(CodeStream &os, const Substitutions &kernelSubs, Handler handler) const = 0;
    virtual void genDenseSynapseVariableRowInit(CodeStream &os, const Substitutions &kernelSubs, Handler handler) const = 0;

    //! Generate code for pushing a variable to the 'device'
    virtual void genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const = 0;

    //! Generate code for pulling a variable from the 'device'
    virtual void genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const = 0;

    //! Generate code for pushing a variable's value in the current timestep to the 'device'
    virtual void genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, 
                                        const std::string &name, VarLocation loc, unsigned int batchSize) const = 0;

    //! Generate code for pulling a variable's value in the current timestep from the 'device'
    virtual void genCurrentVariablePull(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, 
                                        const std::string &name, VarLocation loc, unsigned int batchSize) const = 0;

    //! Generate code for pushing true spikes emitted by a neuron group in the current timestep to the 'device'
    virtual void genCurrentTrueSpikePush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const = 0;

    //! Generate code for pulling true spikes emitted by a neuron group in the current timestep from the 'device'
    virtual void genCurrentTrueSpikePull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const = 0;

    //! Generate code for pushing spike-like events emitted by a neuron group in the current timestep to the 'device'
    virtual void genCurrentSpikeLikeEventPush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const = 0;

    //! Generate code for pulling spike-like events emitted by a neuron group in the current timestep from the 'device'
    virtual void genCurrentSpikeLikeEventPull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const = 0;

    //! Generate a single RNG instance
    /*! On single-threaded platforms this can be a standard RNG like M.T. but, on parallel platforms, it is likely to be a counter-based RNG */
    virtual void genGlobalDeviceRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner,
                                    CodeStream &allocations, CodeStream &free, MemAlloc &memAlloc) const = 0;

    //! Generate an RNG with a state per population member
    virtual void genPopulationRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations,
                                  CodeStream &free, const std::string &name, size_t count, MemAlloc &memAlloc) const = 0;

    virtual void genTimer(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                          CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const = 0;

    //! Generate code to return amount of free 'device' memory in bytes
    virtual void genReturnFreeDeviceMemoryBytes(CodeStream &os) const = 0;

    //! This function can be used to generate a preamble for the GNU makefile used to build
    virtual void genMakefilePreamble(std::ostream &os) const = 0;

    //! The GNU make build system will populate a variable called ``$(OBJECTS)`` with a list of objects to link.
    //! This function should generate a GNU make rule to build these objects into a shared library.
    virtual void genMakefileLinkRule(std::ostream &os) const = 0;

    //! The GNU make build system uses 'pattern rules' (https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html) to build backend modules into objects.
    //! This function should generate a GNU make pattern rule capable of building each module (i.e. compiling .cc file $< into .o file $@).
    virtual void genMakefileCompileRule(std::ostream &os) const = 0;

    //! In MSBuild, 'properties' are used to configure global project settings e.g. whether the MSBuild project builds a static or dynamic library
    //! This function can be used to add additional XML properties to this section.
    /*! see https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-properties for more information. */
    virtual void genMSBuildConfigProperties(std::ostream &os) const = 0;
    virtual void genMSBuildImportProps(std::ostream &os) const = 0;

    //! In MSBuild, the 'item definitions' are used to override the default properties of 'items' such as ``<ClCompile>`` or ``<Link>``.
    //! This function should generate XML to correctly configure the 'items' required to build the generated code, taking into account ``$(Configuration)`` etc.
    /*! see https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-items#item-definitions for more information. */
    virtual void genMSBuildItemDefinitions(std::ostream &os) const = 0;
    virtual void genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const = 0;
    virtual void genMSBuildImportTarget(std::ostream &os) const = 0;

    //! Get backend-specific allocate memory parameters
    virtual std::string getAllocateMemParams(const ModelSpecMerged &) const { return ""; }

    //! Get list of files to copy into generated code
    /*! Paths should be relative to share/genn/backends/ */
    virtual std::vector<filesystem::path> getFilesToCopy(const ModelSpecMerged&) const{ return {}; }

    //! When backends require separate 'device' and 'host' versions of variables, they are identified with a prefix.
    //! This function returns the device prefix so it can be used in otherwise platform-independent code.
    virtual std::string getDeviceVarPrefix() const{ return ""; }

    //! When backends require separate 'device' and 'host' versions of variables, they are identified with a prefix.
    //! This function returns the host prefix so it can be used in otherwise platform-independent code.
    virtual std::string getHostVarPrefix() const { return ""; }

    //! Different backends may have different or no pointer prefix (e.g. __global for OpenCL)
    virtual std::string getPointerPrefix() const { return ""; }

    //! Should 'scalar' variables be implemented on device or can host variables be used directly?
    virtual bool isDeviceScalarRequired() const = 0;

    //! Different backends use different RNGs for different things. Does this one require a global host RNG for the specified model?
    virtual bool isGlobalHostRNGRequired(const ModelSpecMerged &modelMerged) const = 0;

    //! Different backends use different RNGs for different things. Does this one require a global device RNG for the specified model?
    virtual bool isGlobalDeviceRNGRequired(const ModelSpecMerged &modelMerged) const = 0;

    //! Different backends use different RNGs for different things. Does this one require population RNGs?
    virtual bool isPopulationRNGRequired() const = 0;

    //! Different backends seed RNGs in different ways. Does this one initialise population RNGS on device?
    virtual bool isPopulationRNGInitialisedOnDevice() const = 0;

    //! Different backends may implement synapse dynamics differently. Does this one require a synapse remapping data structure for synapse group?
    virtual bool isSynRemapRequired(const SynapseGroupInternal &sg) const = 0;

    //! Different backends may implement synaptic plasticity differently. Does this one require a postsynaptic remapping data structure?
    virtual bool isPostsynapticRemapRequired() const = 0;

    //! How many bytes of memory does 'device' have
    virtual size_t getDeviceMemoryBytes() const = 0;

    //! Some backends will have additional small, fast, memory spaces for read-only data which might
    //! Be well-suited to storing merged group structs. This method returns the prefix required to
    //! Place arrays in these and their size in preferential order
    virtual MemorySpaces getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const = 0;

    virtual bool supportsNamespace() const = 0;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    //! Helper function to generate matching push and pull functions for a variable
    void genVariablePushPull(CodeStream &push, CodeStream &pull,
                             const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const
    {
        genVariablePush(push, type, name, loc, autoInitialized, count);
        genVariablePull(pull, type, name, loc, count);
    }

    //! Helper function to generate matching push and pull functions for the current state of a variable
    void genCurrentVariablePushPull(CodeStream &push, CodeStream &pull, const NeuronGroupInternal &ng, const std::string &type, 
                                    const std::string &name, VarLocation loc, unsigned int batchSize) const
    {
        genCurrentVariablePush(push, ng, type, name, loc, batchSize);
        genCurrentVariablePull(pull, ng, type, name, loc, batchSize);
    }

    //! Helper function to generate matching definition, declaration, allocation and free code for an array
    void genArray(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                  const std::string &type, const std::string &name, VarLocation loc, size_t count, MemAlloc &memAlloc) const
    {
        genVariableDefinition(definitions, definitionsInternal, type + "*", name, loc);
        genVariableImplementation(runner, type + "*", name, loc);
        genVariableFree(free, name, loc);
        genVariableAllocation(allocations, type, name, loc, count, memAlloc);
    }

    //! Get the size of the type
    size_t getSize(const std::string &type) const;

    //! Get the prefix for accessing the address of 'scalar' variables
    std::string getScalarAddressPrefix() const
    {
        return isDeviceScalarRequired() ? getDeviceVarPrefix() : ("&" + getDeviceVarPrefix());
    }

    bool areSixtyFourBitSynapseIndicesRequired(const SynapseGroupMergedBase &sg) const;

    const PreferencesBase &getPreferences() const { return m_Preferences; }

    template<typename T>
    const T &getPreferences() const { return static_cast<const T &>(m_Preferences); }

protected:
    //--------------------------------------------------------------------------
    // Protected API
    //--------------------------------------------------------------------------
    void addType(const std::string &type, size_t size)
    {
        m_TypeBytes.emplace(type, size);
    }

    void setPointerBytes(size_t pointerBytes) 
    {
        m_PointerBytes = pointerBytes;
    }

    void genNeuronIndexCalculation(CodeStream &os, const NeuronUpdateGroupMerged &ng, unsigned int batchSize) const;

    void genSynapseIndexCalculation(CodeStream &os, const SynapseGroupMergedBase &sg, unsigned int batchSize) const;

    void genCustomUpdateIndexCalculation(CodeStream &os, const CustomUpdateGroupMerged &cu) const;

private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    //! How large is a device pointer? E.g. on some AMD devices this != sizeof(char*)
    size_t m_PointerBytes;

    //! Size of supported types in bytes - used for estimating memory usage
    std::unordered_map<std::string, size_t> m_TypeBytes;

    //! Preferences
    const PreferencesBase &m_Preferences;
};
}   // namespace CodeGenerator

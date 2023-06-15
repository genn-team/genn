#pragma once

// Standard C++ includes
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Filesystem includes
#include "path.h"

// PLOG includes
#include <plog/Severity.h>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "type.h"
#include "varAccess.h"
#include "variableMode.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"

// Forward declarations
namespace GeNN
{
class CustomUpdateInternal;
class CustomUpdateWUInternal;
class NeuronGroupInternal;
class SynapseGroupInternal;

namespace CodeGenerator
{
template<typename G>
class EnvironmentGroupMergedField;
class EnvironmentExternalBase;
class ModelSpecMerged;
class NeuronUpdateGroupMerged;
class Substitutions;
class SynapseGroupMergedBase;
class PresynapticUpdateGroupMerged;
class PostsynapticUpdateGroupMerged;
class SynapseDynamicsGroupMerged;
class CustomConnectivityUpdateGroupMerged;
class CustomUpdateGroupMerged;
class CustomUpdateWUGroupMerged;
class CustomUpdateTransposeWUGroupMerged;
class NeuronInitGroupMerged;
class CustomUpdateInitGroupMerged;
class CustomWUUpdateInitGroupMerged;
class CustomWUUpdateSparseInitGroupMerged;
class SynapseConnectivityInitGroupMerged;
class SynapseInitGroupMerged;
class SynapseSparseInitGroupMerged;
}
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PreferencesBase
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
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

    void updateHash(boost::uuids::detail::sha1 &hash) const
    {
        // **NOTE** optimizeCode, debugCode and various compiler flags only affect makefiles/msbuild 

        //! Update hash with preferences
        Utils::updateHash(enableBitmaskOptimisations, hash);
        Utils::updateHash(automaticCopy, hash);
        Utils::updateHash(generateEmptyStatePushPull, hash);
        Utils::updateHash(generateExtraGlobalParamPull, hash);
    }
};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::MemAlloc
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

    //--------------------------------------------------------------------------
    // Friend operators
    //--------------------------------------------------------------------------
    friend std::ostream& operator << (std::ostream &out, const MemAlloc &m);
    friend std::istream& operator >> (std::istream &in,  MemAlloc &m);
};

inline std::ostream & operator << (std::ostream &out, const MemAlloc &m)
{
    out << m.m_HostBytes << " " << m.m_DeviceBytes << " " << m.m_ZeroCopyBytes;
    return out;
}
 
inline std::istream & operator >> (std::istream &in,  MemAlloc &m)
{
    in >> m.m_HostBytes;
    in >> m.m_DeviceBytes;
    in >> m.m_ZeroCopyBytes;
    return in;
}

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

    typedef std::function<void(EnvironmentExternalBase&)> HandlerEnv;
    
    template<typename T>
    using GroupHandler = std::function <void(CodeStream &, const T &, Substitutions&)> ;

    template<typename T>
    using GroupHandlerEnv = std::function <void(EnvironmentExternalBase&, const T &)> ;
    
    //! Vector of prefixes required to allocate in memory space and size of memory space
    typedef std::vector<std::pair<std::string, size_t>> MemorySpaces;

    BackendBase(const PreferencesBase &preferences);
    virtual ~BackendBase(){}

    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    //! Generate platform-specific function to update the state of all neurons
    /*! \param os                       CodeStream to write function to
        \param modelMerged              merged model to generate code for
        \param preambleHandler          callback to write functions for pushing extra-global parameters*/
    virtual void genNeuronUpdate(CodeStream &os, ModelSpecMerged &modelMerged, HostHandler preambleHandler) const = 0;

    //! Generate platform-specific function to update the state of all synapses
    /*! \param os                           CodeStream to write function to
        \param modelMerged                  merged model to generate code for
        \param preambleHandler              callback to write functions for pushing extra-global parameters*/
    virtual void genSynapseUpdate(CodeStream &os, ModelSpecMerged &modelMerged, HostHandler preambleHandler) const = 0;

    //! Generate platform-specific functions to perform custom updates
    /*! \param os                           CodeStream to write function to
        \param modelMerged                  merged model to generate code for
        \param preambleHandler              callback to write functions for pushing extra-global parameters*/
    virtual void genCustomUpdate(CodeStream &os, ModelSpecMerged &modelMerged, HostHandler preambleHandler) const = 0;

    //! Generate platform-specific function to initialise model
    /*! \param os                           CodeStream to write function to
        \param modelMerged                  merged model to generate code for
        \param preambleHandler              callback to write functions for pushing extra-global parameters*/
    virtual void genInit(CodeStream &os, ModelSpecMerged &modelMerged, HostHandler preambleHandler) const = 0;

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

    //! Free memory is called by usercode to free all memory allocatd by GeNN and should only ever be called once.
    //! This function generates a 'preamble' to this function, for example to free backend-specific objects
    virtual void genFreeMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    //! After all timestep logic is complete
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    //! Generate code to define a variable in the appropriate header file
    virtual void genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, 
                                       const Type::ResolvedType &type, const std::string &name, VarLocation loc) const = 0;
    
    //! Generate code to instantiate a variable in the provided stream
    virtual void genVariableInstantiation(CodeStream &os, 
                                          const Type::ResolvedType &type, const std::string &name, VarLocation loc) const = 0;

    //! Generate code to allocate variable with a size known at compile-time
    virtual void genVariableAllocation(CodeStream &os, 
                                       const Type::ResolvedType &type, const std::string &name, 
                                       VarLocation loc, size_t count, MemAlloc &memAlloc) const = 0;
    
    //! Generate code to allocate variable with a size known at runtime
    virtual void genVariableDynamicAllocation(CodeStream &os, 
                                              const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                              const std::string &countVarName = "count", const std::string &prefix = "") const = 0;

    //! Generate code to free a variable
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const = 0;

    //! Generate code for pushing a variable with a size known at compile-time to the 'device'
    virtual void genVariablePush(CodeStream &os, 
                                 const Type::ResolvedType &type, const std::string &name, 
                                 VarLocation loc, bool autoInitialized, size_t count) const = 0;
    
    //! Generate code for pulling a variable with a size known at compile-time from the 'device'
    virtual void genVariablePull(CodeStream &os, 
                                 const Type::ResolvedType &type, const std::string &name, 
                                 VarLocation loc, size_t count) const = 0;

    //! Generate code for pushing a variable's value in the current timestep to the 'device'
    virtual void genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, 
                                        const Type::ResolvedType &type, const std::string &name, 
                                        VarLocation loc, unsigned int batchSize) const = 0;

    //! Generate code for pulling a variable's value in the current timestep from the 'device'
    virtual void genCurrentVariablePull(CodeStream &os, const NeuronGroupInternal &ng, 
                                        const Type::ResolvedType &type, const std::string &name, 
                                        VarLocation loc, unsigned int batchSize) const = 0;

    //! Generate code for pushing a variable with a size known at tuntime to the 'device'
    virtual void genVariableDynamicPush(CodeStream &os, 
                                        const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                        const std::string &countVarName = "count", const std::string &prefix = "") const = 0;

    //! Generate code for pulling a variable with a size known at runtime from the 'device'
    virtual void genVariableDynamicPull(CodeStream &os, 
                                        const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                        const std::string &countVarName = "count", const std::string &prefix = "") const = 0;

    //! Generate code for pushing a new pointer to a dynamic variable into the merged group structure on 'device'
    virtual void genMergedDynamicVariablePush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                              const std::string &groupIdx, const std::string &fieldName,
                                              const std::string &egpName) const = 0;

    //! When generating function calls to push to merged groups, backend without equivalent of Unified Virtual Addressing e.g. OpenCL 1.2 may use different types on host
    virtual std::string getMergedGroupFieldHostTypeName(const Type::ResolvedType &type) const = 0;

    //! When generating merged structures what type to use for simulation RNGs
    virtual std::optional<Type::ResolvedType> getMergedGroupSimRNGType() const = 0;

    virtual void genPopVariableInit(EnvironmentExternalBase &env, HandlerEnv handler) const = 0;
    virtual void genVariableInit(EnvironmentExternalBase &env, const std::string &count, const std::string &indexVarName, HandlerEnv handler) const = 0;
    virtual void genSparseSynapseVariableRowInit(EnvironmentExternalBase &env, Handler handler) const = 0;
    virtual void genDenseSynapseVariableRowInit(EnvironmentExternalBase &env, Handler handler) const = 0;
    virtual void genKernelSynapseVariableInit(EnvironmentExternalBase &env, const SynapseInitGroupMerged &sg, HandlerEnv handler) const = 0;
    virtual void genKernelCustomUpdateVariableInit(EnvironmentExternalBase &env, const CustomWUUpdateInitGroupMerged &cu, HandlerEnv handler) const = 0;

    //! Generate a single RNG instance
    /*! On single-threaded platforms this can be a standard RNG like M.T. but, on parallel platforms, it is likely to be a counter-based RNG */
    virtual void genGlobalDeviceRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner,
                                    CodeStream &allocations, CodeStream &free, MemAlloc &memAlloc) const = 0;

    //! Generate an RNG with a state per population member
    virtual void genPopulationRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, 
                                  CodeStream &allocations, CodeStream &free, 
                                  const std::string &name, size_t count, MemAlloc &memAlloc) const = 0;

    virtual void genTimer(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                          CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const = 0;

    //! Generate code to return amount of free 'device' memory in bytes
    virtual void genReturnFreeDeviceMemoryBytes(CodeStream &os) const = 0;

    //! On backends which support it, generate a runtime assert
    virtual void genAssert(CodeStream &os, const std::string &condition) const = 0;

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

    //! Different backends may implement synaptic plasticity differently. Does this one require a postsynaptic remapping data structure?
    virtual bool isPostsynapticRemapRequired() const = 0;

    //! Backends which support batch-parallelism might require an additional host reduction phase after reduction kernels
    virtual bool isHostReductionRequired() const = 0;

    //! How many bytes of memory does 'device' have
    virtual size_t getDeviceMemoryBytes() const = 0;

    //! Some backends will have additional small, fast, memory spaces for read-only data which might
    //! Be well-suited to storing merged group structs. This method returns the prefix required to
    //! Place arrays in these and their size in preferential order
    virtual MemorySpaces getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const = 0;

    //! Does this backend support namespaces i.e. can C++ implementation of support functions be used
    virtual bool supportsNamespace() const = 0;

    //! Get hash digest of this backends identification and the preferences it has been configured with
    virtual boost::uuids::detail::sha1::digest_type getHashDigest() const = 0;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    //! Helper function to generate matching push and pull functions for a variable
    void genVariablePushPull(CodeStream &push, CodeStream &pull,
                             const Type::ResolvedType &type, const std::string &name, 
                             VarLocation loc, bool autoInitialized, size_t count) const
    {
        genVariablePush(push, type, name, loc, autoInitialized, count);
        genVariablePull(pull, type, name, loc, count);
    }

    //! Helper function to generate matching push and pull functions for the current state of a variable
    void genCurrentVariablePushPull(CodeStream &push, CodeStream &pull, const NeuronGroupInternal &ng, 
                                    const Type::ResolvedType &type, const std::string &name, 
                                    VarLocation loc, unsigned int batchSize) const
    {
        genCurrentVariablePush(push, ng, type, name, loc, batchSize);
        genCurrentVariablePull(pull, ng, type, name, loc, batchSize);
    }


    //! Helper function to generate matching definition, declaration, allocation and free code for a statically-sized array
    void genArray(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                  const Type::ResolvedType &type, const std::string &name, 
                  VarLocation loc, size_t count, MemAlloc &memAlloc) const
    {
        genVariableDefinition(definitions, definitionsInternal, type, name, loc);
        genVariableInstantiation(runner, type, name, loc);
        genVariableFree(free, name, loc);
        genVariableAllocation(allocations, type, name, loc, count, memAlloc);
    }

    //! Get the prefix for accessing the address of 'scalar' variables
    std::string getScalarAddressPrefix() const
    {
        return isDeviceScalarRequired() ? getDeviceVarPrefix() : ("&" + getDeviceVarPrefix());
    }

    bool areSixtyFourBitSynapseIndicesRequired(const SynapseGroupMergedBase &sg) const;

    //! Get backend-specific pointer size in bytes
    size_t getPointerBytes() const{ return m_PointerBytes; }

    const PreferencesBase &getPreferences() const { return m_Preferences; }

    template<typename T>
    const T &getPreferences() const { return static_cast<const T &>(m_Preferences); }

protected:
    //--------------------------------------------------------------------------
    // ReductionTarget
    //--------------------------------------------------------------------------
    //! Simple struct to hold reduction targets
    struct ReductionTarget
    {
        std::string name;
        Type::ResolvedType type;
        VarAccessMode access;
        std::string index;
    };

    //--------------------------------------------------------------------------
    // Protected API
    //--------------------------------------------------------------------------
    void setPointerBytes(size_t pointerBytes) 
    {
        m_PointerBytes = pointerBytes;
    }

    template<typename G>
    void genNeuronIndexCalculation(EnvironmentGroupMergedField<G> &env, unsigned int batchSize) const
    {
        env.add(Type::Uint32.addConst(), "num_neurons",
                Type::Uint32, "numNeurons",
                [](const auto &ng, size_t) { return std::to_string(ng.getNumNeurons()); });
        env.add(Type::Uint32.createPointer(), "_spk_cnt", "spkCnt",
                [this](const auto &g, size_t) { return getDeviceVarPrefix() + "glbSpkCnt" + g.getName(); });
        env.add(Type::Uint32.createPointer(), "_spk", "spk",
                [this](const auto &g, size_t) { return getDeviceVarPrefix() + "glbSpk" + g.getName(); });
        env.add(Type::Uint32.createPointer(), "_spk_cnt_evnt", "spkCntEvnt",
                [this](const auto &g, size_t) { return getDeviceVarPrefix() + "glbSpkCntEvnt" + g.getName(); });
        env.add(Type::Uint32.createPointer(), "_spk_evnt", "spkEvnt",
                [this](const auto &g, size_t) { return getDeviceVarPrefix() + "glbSpkEvnt" + g.getName(); });
        env.add(Type::Uint32.createPointer(), "_spk_que_ptr", "spkQuePtr",
                [this](const auto &g, size_t) { return getScalarAddressPrefix() + "spkQuePtr" + g.getName(); });

        env.add(env.getGroup().getTimeType().createPointer(), "_spk_time", "sT",
                [this](const auto &g, size_t) { return getDeviceVarPrefix() + "sT" + g.getName(); });
        env.add(env.getGroup().getTimeType().createPointer(), "_spk_evnt_time", "seT",
                [this](const auto &g, size_t) { return getDeviceVarPrefix() + "seT" + g.getName(); });
        env.add(env.getGroup().getTimeType().createPointer(), "_prev_spk_time", "prevST", 
                [this](const auto &g, size_t) { return getDeviceVarPrefix() + "prevST" + g.getName(); });
        env.add(env.getGroup().getTimeType().createPointer(), "_prev_spk_evnt_time", "prevSET",
                [this](const auto &g, size_t) { return getDeviceVarPrefix() + "prevSET" + g.getName(); });


        // If batching is enabled, calculate batch offset
        if(batchSize > 1) {
            env.add(Type::Uint32.addConst(), "_batchOffset", "batchOffset",
                    {env.addInitialiser("const unsigned int batchOffset = " + env["num_neurons"] + " * batch;")},
                    {"num_neurons"});
        }
            
        // If axonal delays are required
        if(env.getGroup().getArchetype().isDelayRequired()) {
            // We should READ from delay slot before spkQuePtr
            const unsigned int numDelaySlots = env.getGroup().getArchetype().getNumDelaySlots();
            const std::string numDelaySlotsStr = std::to_string(numDelaySlots);
            env.add(Type::Uint32.addConst(), "_read_delay_slot", "readDelaySlot",
                    {env.addInitialiser("const unsigned int readDelaySlot = (*" + env["_spk_que_ptr"] + " + " + std::to_string(numDelaySlots - 1) + ") % " + numDelaySlotsStr+ ";")},
                    {"_spk_que_ptr"});
            env.add(Type::Uint32.addConst(), "_read_delay_offset", "readDelayOffset",
                    {env.addInitialiser("const unsigned int readDelayOffset = readDelaySlot * " + env["num_neurons"] + ";")},
                    {"num_neurons", "_read_delay_slot"});

            // And we should WRITE to delay slot pointed to be spkQuePtr
            env.add(Type::Uint32.addConst(), "_write_delay_slot", "writeDelaySlot",
                    {env.addInitialiser("const unsigned int writeDelaySlot = *" + env["_spk_que_ptr"] + ";")},
                    {"_spk_que_ptr"});
            env.add(Type::Uint32.addConst(), "_write_delay_offset", "writeDelayOffset",
                    {env.addInitialiser("const unsigned int writeDelayOffset = writeDelaySlot * " + env["num_neurons"] + ";")},
                    {"num_neurons", "_write_delay_slot"});

            // If batching is also enabled
            if(batchSize > 1) {
                // Calculate batched delay slots
                env.add(Type::Uint32.addConst(), "_read_batch_delay_slot", "readBatchDelaySlot",
                        {env.addInitialiser("const unsigned int readBatchDelaySlot = (batch * " + numDelaySlotsStr + ") + readDelaySlot;")},
                        {"_read_delay_slot"});
                env.add(Type::Uint32.addConst(), "_write_batch_delay_slot", "writeBatchDelaySlot",
                        {env.addInitialiser("const unsigned int writeBatchDelaySlot = (batch * " + numDelaySlotsStr + ") + writeDelaySlot;")},
                        {"_write_delay_slot"});

                // Calculate current batch offset
                env.add(Type::Uint32.addConst(), "_batch_delay_offset", "batchDelayOffset",
                        {env.addInitialiser("const unsigned int batchDelayOffset = batchOffset * " + numDelaySlotsStr + ";")},
                        {"_batch_offset"});

                // Calculate further offsets to include delay and batch
                env.add(Type::Uint32.addConst(), "_read_batch_delay_offset", "readBatchDelayOffset",
                        {env.addInitialiser("const unsigned int readBatchDelayOffset = readDelayOffset + batchDelayOffset;")},
                        {"_read_delay_offset", "_batchDelayOffset"});
                env.add(Type::Uint32.addConst(), "_write_batch_delay_offset", "writeBatchDelayOffset",
                        {env.addInitialiser("const unsigned int writeBatchDelayOffset = writeDelayOffset + batchDelayOffset;")},
                        {"_write_delay_offset", "_batchDelayOffset"});
            }
        }
    }

    template<typename G>
    void genSynapseIndexCalculation(EnvironmentGroupMergedField<G> &env, unsigned int batchSize) const
    {
        // Synapse group fields 
        groupEnv.add(Type::Uint32.addConst(), "num_pre",
                     Type::Uint32, "numSrcNeurons", 
                     [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
        groupEnv.add(Type::Uint32.addConst(), "num_post",
                     Type::Uint32, "numTrgNeurons", 
                     [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
        groupEnv.add(Type::Uint32, "_row_stride", "rowStride", 
                     [](const SynapseGroupInternal &sg, size_t) { return std::to_string(getSynapticMatrixRowStride(sg)); });
        groupEnv.add(Type::Uint32, "_col_stride", "colStride", 
                     [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getMaxSourceConnections()); });
                        
        // Postsynaptic model fields         
        groupEnv.add(modelMerged.getModel().getPrecision().createPointer(), "_out_post", "outPost",
                     [this](const auto &g, size_t) { return getDeviceVarPrefix() + "outPost" + g.getFusedPSVarSuffix(); });
        groupEnv.add(modelMerged.getModel().getPrecision().createPointer(), "_den_delay", "denDelay",
                     [this](const auto &g, size_t) { return getDeviceVarPrefix() + "denDelay" + g.getFusedPSVarSuffix(); });
        groupEnv.add(Type::Uint32.createPointer(), "_den_delay_ptr", "denDelayPtr",
                     [this](const auto &g, size_t) { return getDeviceVarPrefix() + "denDelayPtr" + g.getFusedPSVarSuffix(); });
                       
        // Presynaptic output fields
        groupEnv.add(modelMerged.getModel().getPrecision().createPointer(), "_out_pre", "outPre",
                     [this](const auto &g, size_t) { return getDeviceVarPrefix() + "outPre" + g.getFusedPreOutputSuffix(); });
                        

        // Source neuron fields
        groupEnv.add(Type::Uint32.createPointer(), "_src_spk_que_ptr", "srcSpkQuePtr",
                     [this](const auto &g, size_t) { return getScalarAddressPrefix() + "spkQuePtr" + g.getSrcNeuronGroup()->getName(); });
        groupEnv.add(Type::Uint32.createPointer(), "_src_spk_cnt", "srcSpkCnt",
                     [this](const auto &g, size_t) { return getDeviceVarPrefix() + "glbSpkCnt" + g.getSrcNeuronGroup()->getName(); });
        groupEnv.add(Type::Uint32.createPointer(), "_src_spk", "srcSpk",
                     [this](const auto &g, size_t) { return getDeviceVarPrefix() + "glbSpk" + g.getSrcNeuronGroup()->getName(); });
        groupEnv.add(Type::Uint32.createPointer(), "_src_spk_evnt_cnt", "srcSpkCntEvnt",
                     [this](const auto &g, size_t) { return getDeviceVarPrefix() + "glbSpkCntEvnt" + g.getSrcNeuronGroup()->getName(); });
        groupEnv.add(Type::Uint32.createPointer(), "_src_spk_evnt", "srcSpkEvnt",
                     [this](const auto &g, size_t) { return getDeviceVarPrefix() + "glbSpkEvnt" + g.getSrcNeuronGroup()->getName(); });

        // Target neuron fields
        groupEnv.add(Type::Uint32.createPointer(), "_trg_spk_que_ptr", "trgSpkQuePtr",
                     [this](const auto &g, size_t) { return getScalarAddressPrefix() + "spkQuePtr" + g.getTrgNeuronGroup()->getName(); });
        groupEnv.add(Type::Uint32.createPointer(), "_trg_spk_cnt", "trgSpkCnt",
                     [this](const auto &g, size_t) { return getDeviceVarPrefix() + "glbSpkCnt" + g.getTrgNeuronGroup()->getName(); });
        groupEnv.add(Type::Uint32.createPointer(), "_trg_spk", "trgSpk",
                     [this](const auto &g, size_t) { return getDeviceVarPrefix() + "glbSpk" + g.getTrgNeuronGroup()->getName(); });

        // If batching is enabled
        if(batchSize > 1) {
            // Calculate batch offsets into pre and postsynaptic populations
            env.add(Type::Uint32.addConst(), "_pre_batch_offset", "preBatchOffset",
                    {env.addInitialiser("const unsigned int preBatchOffset = " + env["num_pre"] + " * batch;")},
                    {"num_pre"});
            env.add(Type::Uint32.addConst(), "_post_batch_offset", "postBatchOffset",
                    {env.addInitialiser("const unsigned int preBatchOffset = " + env["num_post"] + " * batch;")},
                    {"num_post"});
        
            // Calculate batch offsets into synapse arrays, using 64-bit arithmetic if necessary
            if(areSixtyFourBitSynapseIndicesRequired(env.getGroup())) {
                assert(false);
                //os << "const uint64_t synBatchOffset = (uint64_t)preBatchOffset * (uint64_t)group->rowStride;" << std::endl;
            }
            else {
                env.add(Type::Uint32.addConst(), "_syn_batch_offset", "synBatchOffset",
                        {env.addInitialiser("const unsigned int synBatchOffset = " + env["_pre_batch_offset"] + " * " + env["_row_stride"] + ";")},
                        {"_pre_batch_offset", "_row_stride"});
            }
        
            // If synapse group has kernel
            const auto &kernelSize = env.getGroup().getArchetype().getKernelSize();
            if(!kernelSize.empty()) {
                // Loop through kernel dimensions and multiply together
                // **TODO** extract list of kernel size variables referenced
                std::ostringstream kernBatchOffsetInit;
                kernBatchOffsetInit << "const unsigned int kernBatchOffset = ";
                for(size_t i = 0; i < kernelSize.size(); i++) {
                    kernBatchOffsetInit << env.getGroup().getKernelSize(i) << " * ";
                }
            
                // And finally by batch
                kernBatchOffsetInit << "batch;" << std::endl;

                env.add(Type::Uint32.addConst(), "_kern_batch_offset", "kernBatchOffset",
                        {env.addInitialiser(kernBatchOffsetInit.str())});
            }
        }

        // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
        if(env.getGroup().getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
            const unsigned int numDelaySteps = env.getGroup().getArchetype().getDelaySteps();
            const unsigned int numSrcDelaySlots = env.getGroup().getArchetype().getSrcNeuronGroup()->getNumDelaySlots();

            std::ostringstream preDelaySlotInit;
            preDelaySlotInit << "const unsigned int preDelaySlot = ";
            if(numDelaySteps == 0) {
                preDelaySlotInit << "*" << env["_src_spk_que_ptr"] << ";" << std::endl;
            }
            else {
                preDelaySlotInit << "(*" << env["_src_spk_que_ptr"] << " + " << (numSrcDelaySlots - numDelaySteps) << ") % " << numSrcDelaySlots <<  ";" << std::endl;
            }
            env.add(Type::Uint32, "_pre_delay_slot", "preDelaySlot", 
                    {env.addInitialiser(preDelaySlotInit.str())}, {"_src_spk_que_ptr"});

            env.add(Type::Uint32, "_pre_delay_offset", "preDelayOffset",
                    {env.addInitialiser("const unsigned int preDelayOffset = preDelaySlot * " + env["num_pre"] + ";")},
                    {"num_pre", "_pre_delay_slot"});

            if(batchSize > 1) {
                env.add(Type::Uint32, "_pre_batch_delay_slot", "preBatchDelaySlot",
                        {env.addInitialiser("const unsigned int preBatchDelaySlot = preDelaySlot + (batch * " + std::to_string(numSrcDelaySlots) + ");")},
                        {"_pre_delay_slot"});

                os <<  << std::endl;

                env.add(Type::Uint32, "_pre_batch_delay_offset", "preBatchDelayOffset",
                        {env.addInitialiser("const unsigned int preBatchDelayOffset = preDelayOffset + (preBatchOffset * " + std::to_string(numSrcDelaySlots) + ");")},
                        {"_pre_delay_offset", "_pre_batch_offset"});
            }

            if(env.getGroup().getArchetype().getWUModel()->isPrevPreSpikeTimeRequired() 
               || env.getGroup().getArchetype().getWUModel()->isPrevPreSpikeEventTimeRequired()) 
            {
                os << "const unsigned int prePrevSpikeTimeDelayOffset = " << "((*group->srcSpkQuePtr + " << (numSrcDelaySlots - numDelaySteps - 1) << ") % " << numSrcDelaySlots << ")" << " * group->numSrcNeurons;" << std::endl;

                if(batchSize > 1) {
                    os << "const unsigned int prePrevSpikeTimeBatchDelayOffset = prePrevSpikeTimeDelayOffset + (preBatchOffset * " << numSrcDelaySlots << ");" << std::endl;
                }
            }
        }

        // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
        if(env.getGroup().getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
            const unsigned int numBackPropDelaySteps = env.getGroup().getArchetype().getBackPropDelaySteps();
            const unsigned int numTrgDelaySlots = env.getGroup().getArchetype().getTrgNeuronGroup()->getNumDelaySlots();

            os << "const unsigned int postDelaySlot = ";
            if(numBackPropDelaySteps == 0) {
                os << "*group->trgSpkQuePtr;" << std::endl;
            }
            else {
                os << "(*group->trgSpkQuePtr + " << (numTrgDelaySlots - numBackPropDelaySteps) << ") % " << numTrgDelaySlots << ";" << std::endl;
            }
            os << "const unsigned int postDelayOffset = postDelaySlot * group->numTrgNeurons;" << std::endl;

            if(batchSize > 1) {
                os << "const unsigned int postBatchDelaySlot = postDelaySlot + (batch * " << numTrgDelaySlots << ");" << std::endl;
                os << "const unsigned int postBatchDelayOffset = postDelayOffset + (postBatchOffset * " << numTrgDelaySlots << ");" << std::endl;
            }

            if(env.getGroup().getArchetype().getWUModel()->isPrevPostSpikeTimeRequired()) {
                os << "const unsigned int postPrevSpikeTimeDelayOffset = " << "((*group->trgSpkQuePtr + " << (numTrgDelaySlots - numBackPropDelaySteps - 1) << ") % " << numTrgDelaySlots << ")" << " * group->numTrgNeurons;" << std::endl;
            
                if(batchSize > 1) {
                    os << "const unsigned int postPrevSpikeTimeBatchDelayOffset = postPrevSpikeTimeDelayOffset + (postBatchOffset * " << numTrgDelaySlots << ");" << std::endl;
                }

            }
        }
    }
    void genCustomUpdateIndexCalculation(CodeStream &os, const CustomUpdateGroupMerged &cu) const;
    
    void genCustomConnectivityUpdateIndexCalculation(CodeStream &os, const CustomConnectivityUpdateGroupMerged &cu) const;
    
    //! Get the initial value to start reduction operations from
    std::string getReductionInitialValue(VarAccessMode access, const Type::ResolvedType &type) const;

    //! Generate a reduction operation to reduce value into reduction
    std::string getReductionOperation(const std::string &reduction, const std::string &value,
                                      VarAccessMode access, const Type::ResolvedType &type) const;


    //! Helper function to generate initialisation code for any reduction operations carried out be custom update group.
    //! Returns vector of ReductionTarget structs, providing all information to write back reduction results to memory
    std::vector<ReductionTarget> genInitReductionTargets(CodeStream &os, const CustomUpdateGroupMerged &cg, const std::string &idx = "") const;

    //! Helper function to generate initialisation code for any reduction operations carried out be custom weight update group.
    //! //! Returns vector of ReductionTarget structs, providing all information to write back reduction results to memory
    std::vector<ReductionTarget> genInitReductionTargets(CodeStream &os, const CustomUpdateWUGroupMerged &cg, const std::string &idx = "") const;

private:
    //--------------------------------------------------------------------------
    // Private API
    //--------------------------------------------------------------------------
    template<typename G, typename R>
    std::vector<ReductionTarget> genInitReductionTargets(CodeStream &os, const G &cg, const std::string &idx, R getVarRefIndexFn) const
    {
        // Loop through variables
        std::vector<ReductionTarget> reductionTargets;
        const auto *cm = cg.getArchetype().getCustomUpdateModel();
        for (const auto &v : cm->getVars()) {
            // If variable is a reduction target, define variable initialised to correct initial value for reduction
            if (v.access & VarAccessModeAttribute::REDUCE) {
                const auto resolvedType = v.type.resolve(cg.getTypeContext());
                os << resolvedType.getName() << " lr" << v.name << " = " << getReductionInitialValue(getVarAccessMode(v.access), resolvedType) << ";" << std::endl;
                reductionTargets.push_back({v.name, resolvedType, getVarAccessMode(v.access),
                                            cg.getVarIndex(getVarAccessDuplication(v.access), idx)});
            }
        }

        // Loop through all variable references
        for(const auto &modelVarRef : cm->getVarRefs()) {
            const auto &varRef = cg.getArchetype().getVarReferences().at(modelVarRef.name);

            // If variable reference is a reduction target, define variable initialised to correct initial value for reduction
            if (modelVarRef.access & VarAccessModeAttribute::REDUCE) {
                const auto resolvedType = modelVarRef.type.resolve(cg.getTypeContext());
                os << resolvedType.getName() << " lr" << modelVarRef.name << " = " << getReductionInitialValue(modelVarRef.access, resolvedType) << ";" << std::endl;
                reductionTargets.push_back({modelVarRef.name, resolvedType, modelVarRef.access,
                                            getVarRefIndexFn(varRef, idx)});
            }
        }
        return reductionTargets;
    }


    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
     //! How large is a device pointer? E.g. on some AMD devices this != sizeof(char*)
    size_t m_PointerBytes;

    //! Preferences
    const PreferencesBase &m_Preferences;
};
}   // namespace GeNN::CodeGenerator

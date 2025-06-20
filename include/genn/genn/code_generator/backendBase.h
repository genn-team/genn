#pragma once

// Standard C++ includes
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

// Filesystem includes
#include "path.h"

// PLOG includes
#include <plog/Severity.h>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "synapseMatrixType.h"
#include "type.h"
#include "varAccess.h"
#include "varLocation.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"

// Forward declarations
namespace GeNN
{
class CustomConnectivityUpdateInternal;
class CustomUpdateInternal;
class CustomUpdateWUInternal;
class ModelSpecInternal;
class NeuronGroupInternal;
class SynapseGroupInternal;

namespace CodeGenerator
{
template<typename G, typename F = G>
class EnvironmentGroupMergedField;
class EnvironmentExternalBase;
class ModelSpecMerged;
template<typename G>
class GroupMerged;
class NeuronUpdateGroupMerged;
class NeuronPrevSpikeTimeUpdateGroupMerged;
class NeuronSpikeQueueUpdateGroupMerged;
class PresynapticUpdateGroupMerged;
class PostsynapticUpdateGroupMerged;
class SynapseDynamicsGroupMerged;
class SynapseDendriticDelayUpdateGroupMerged;
class CustomConnectivityUpdateGroupMerged;
class CustomConnectivityRemapUpdateGroupMerged;
class CustomUpdateGroupMerged;
class CustomUpdateWUGroupMerged;
class CustomUpdateTransposeWUGroupMerged;
class CustomUpdateHostReductionGroupMerged;
class CustomWUUpdateHostReductionGroupMerged;
class NeuronInitGroupMerged;
class CustomUpdateInitGroupMerged;
class CustomWUUpdateInitGroupMerged;
class CustomWUUpdateSparseInitGroupMerged;
class SynapseConnectivityInitGroupMerged;
class CustomConnectivityUpdatePreInitGroupMerged;
class CustomConnectivityUpdatePostInitGroupMerged;
class CustomConnectivityUpdateSparseInitGroupMerged;
class SynapseInitGroupMerged;
class SynapseSparseInitGroupMerged;
}

namespace Runtime
{
class ArrayBase;
class Runtime;
class StateBase;
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

    //! C++ compiler options to be used for building all host side code (used for unix based platforms)
    std::string userCxxFlagsGNU = "";

    //! NVCC compiler options they may want to use for all GPU code (used for unix based platforms)
    std::string userNvccFlagsGNU = "";

    //! Logging level to use for model description
    plog::Severity gennLogLevel = plog::info;

    //! Logging level to use for code generation
    plog::Severity codeGeneratorLogLevel = plog::info;

    //! Logging level to use for transpiler
    plog::Severity transpilerLogLevel = plog::info;

    //! Logging level to use for runtime
    plog::Severity runtimeLogLevel = plog::info;

    //! Logging level to use for backend
    plog::Severity backendLogLevel = plog::info;

    void updateHash(boost::uuids::detail::sha1&) const
    {
        // **NOTE** optimizeCode, debugCode and various compiler flags only affect makefiles/msbuild 
    }
};

//--------------------------------------------------------------------------
// CodeGenerator::BackendBase
//--------------------------------------------------------------------------
class GENN_EXPORT BackendBase
{
public:
    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    //! What atomic operation is required
    enum class AtomicOperation
    {
        ADD,
        OR,
    };

    //--------------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------------
    typedef std::function<void(CodeStream &)> HostHandler;

    typedef std::function<void(EnvironmentExternalBase&)> HandlerEnv;
    
    template<typename T>
    using GroupHandlerEnv = std::function <void(EnvironmentExternalBase&, T &)> ;
    
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
    virtual void genNeuronUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                 HostHandler preambleHandler) const = 0;

    //! Generate platform-specific function to update the state of all synapses
    /*! \param os                           CodeStream to write function to
        \param modelMerged                  merged model to generate code for
        \param preambleHandler              callback to write functions for pushing extra-global parameters*/
    virtual void genSynapseUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                  HostHandler preambleHandler) const = 0;

    //! Generate platform-specific functions to perform custom updates
    /*! \param os                           CodeStream to write function to
        \param modelMerged                  merged model to generate code for
        \param preambleHandler              callback to write functions for pushing extra-global parameters*/
    virtual void genCustomUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                 HostHandler preambleHandler) const = 0;

    //! Generate platform-specific function to initialise model
    /*! \param os                           CodeStream to write function to
        \param modelMerged                  merged model to generate code for
        \param preambleHandler              callback to write functions for pushing extra-global parameters*/
    virtual void genInit(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                         HostHandler preambleHandler) const = 0;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const = 0;
    
    //! Definitions is the usercode-facing header file for the generated code. This function generates a 'preamble' to this header file.
    /*! This will be included from a standard C++ compiler so shouldn't include any platform-specific types or headers*/
    virtual void genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    //! Allocate memory is the first function in GeNN generated code called by usercode and it should only ever be called once.
    //! Therefore it's a good place for any global initialisation. This function generates a 'preamble' to this function.
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    //! Free memory is called by usercode to free all memory allocatd by GeNN and should only ever be called once.
    //! This function generates a 'preamble' to this function, for example to free backend-specific objects
    virtual void genFreeMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    //! After all timestep logic is complete
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    //! Create backend-specific runtime state object
    /*! \param runtime  runtime object */
    virtual std::unique_ptr<GeNN::Runtime::StateBase> createState(const Runtime::Runtime &runtime) const = 0;

    //! Create backend-specific array object
    /*! \param type         data type of array
        \param count        number of elements in array, if non-zero will allocate
        \param location     location of array e.g. device-only*/
    virtual std::unique_ptr<GeNN::Runtime::ArrayBase> createArray(const Type::ResolvedType &type, size_t count, 
                                                                  VarLocation location, bool uninitializedn) const = 0;

    //! Create array of backend-specific population RNGs (if they are initialised on host this will occur here)
    /*! \param count        number of RNGs required*/
    virtual std::unique_ptr<GeNN::Runtime::ArrayBase> createPopulationRNG(size_t count) const = 0;

    //! Generate code to allocate variable with a size known at runtime
    virtual void genLazyVariableDynamicAllocation(CodeStream &os, 
                                                  const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                                  const std::string &countVarName) const = 0;

    //! Generate code for pushing a variable with a size known at runtime to the 'device'
    virtual void genLazyVariableDynamicPush(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name,
                                            VarLocation loc, const std::string &countVarName) const = 0;

    //! Generate code for pulling a variable with a size known at runtime from the 'device'
    virtual void genLazyVariableDynamicPull(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name,
                                            VarLocation loc, const std::string &countVarName) const = 0;

    //! Generate code for pushing a new pointer to a dynamic variable into the merged group structure on 'device'
    virtual void genMergedDynamicVariablePush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                              const std::string &groupIdx, const std::string &fieldName,
                                              const std::string &egpName) const = 0;

    virtual void genPopVariableInit(EnvironmentExternalBase &env, HandlerEnv handler) const = 0;
    virtual void genVariableInit(EnvironmentExternalBase &env, const std::string &count, const std::string &indexVarName, HandlerEnv handler) const = 0;
    virtual void genSparseSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const = 0;
    virtual void genDenseSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const = 0;
    virtual void genKernelSynapseVariableInit(EnvironmentExternalBase &env, SynapseInitGroupMerged &sg, HandlerEnv handler) const = 0;
    virtual void genKernelCustomUpdateVariableInit(EnvironmentExternalBase &env, CustomWUUpdateInitGroupMerged &cu, HandlerEnv handler) const = 0;

    //! Get suitable atomic *lhsPointer += rhsValue or *lhsPointer |= rhsValue style operation
    virtual std::string getAtomicOperation(const std::string &lhsPointer, const std::string &rhsValue,
                                           const Type::ResolvedType &type, AtomicOperation op = AtomicOperation::ADD) const = 0;

    //! GeNN knows that pointers used in some places in thew code e.g. in merged groups are
    //! "restricted" i.e. not aliased. What keyword should be used to indicate this?
    virtual std::string getRestrictKeyword() const = 0;

    //! Generate a single RNG instance
    /*! On single-threaded platforms this can be a standard RNG like M.T. but, on parallel platforms, it is likely to be a counter-based RNG */
    virtual void genGlobalDeviceRNG(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free) const = 0;

    virtual void genTimer(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free,
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

    //! As well as host pointers, are device objects required?
    virtual bool isArrayDeviceObjectRequired() const = 0;

    //! As well as host pointers, are additional host objects required e.g. for buffers in OpenCL?
    virtual bool isArrayHostObjectRequired() const = 0;

    //! Different backends use different RNGs for different things. Does this one require a global host RNG for the specified model?
    virtual bool isGlobalHostRNGRequired(const ModelSpecInternal &model) const = 0;

    //! Different backends use different RNGs for different things. Does this one require a global device RNG for the specified model?
    virtual bool isGlobalDeviceRNGRequired(const ModelSpecInternal &model) const = 0;

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

    //! Get hash digest of this backends identification and the preferences it has been configured with
    virtual boost::uuids::detail::sha1::digest_type getHashDigest() const = 0;

    //! Generate code to push a profiler range marker
    virtual void genPushProfilerRange(CodeStream&, const std::string&) const {}
    
    //! Generate code to pop the current profiler range marker
    virtual void genPopProfilerRange(CodeStream&) const {}

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    //! Get the type to use for synaptic indices within a merged synapse group
    Type::ResolvedType getSynapseIndexType(const GroupMerged<SynapseGroupInternal> &sg) const;

    //! Get the type to use for synaptic indices within a merged custom weight update group
    Type::ResolvedType getSynapseIndexType(const GroupMerged<CustomUpdateWUInternal> &sg) const;

    //! Get the type to use for synaptic indices within a merged custom connectivity update group
    Type::ResolvedType getSynapseIndexType(const GroupMerged<CustomConnectivityUpdateInternal> &cg) const;

    void buildSizeEnvironment(EnvironmentGroupMergedField<CustomUpdateGroupMerged> &env) const;
    void buildSizeEnvironment(EnvironmentGroupMergedField<CustomUpdateHostReductionGroupMerged> &env) const;
    void buildSizeEnvironment(EnvironmentGroupMergedField<CustomUpdateWUGroupMerged> &env) const;
    void buildSizeEnvironment(EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> &env) const;
    void buildSizeEnvironment(EnvironmentGroupMergedField<CustomWUUpdateHostReductionGroupMerged> &env) const;

    void buildStandardEnvironment(EnvironmentGroupMergedField<NeuronUpdateGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<NeuronPrevSpikeTimeUpdateGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<NeuronSpikeQueueUpdateGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<PostsynapticUpdateGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<SynapseDynamicsGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<SynapseDendriticDelayUpdateGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomUpdateGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomUpdateWUGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomConnectivityRemapUpdateGroupMerged> &env) const;

    void buildStandardEnvironment(EnvironmentGroupMergedField<NeuronInitGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<SynapseInitGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomUpdateInitGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomWUUpdateInitGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomWUUpdateSparseInitGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdatePreInitGroupMerged> &env) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdatePostInitGroupMerged> &env) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<SynapseSparseInitGroupMerged> &env, unsigned int batchSize) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdateSparseInitGroupMerged> &env) const;
    void buildStandardEnvironment(EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> &env, unsigned int batchSize) const;

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

    //! Get the initial value to start reduction operations from
    std::string getReductionInitialValue(VarAccessMode access, const Type::ResolvedType &type) const;

    //! Generate a reduction operation to reduce value into reduction
    std::string getReductionOperation(const std::string &reduction, const std::string &value,
                                      VarAccessMode access, const Type::ResolvedType &type) const;


    //! Helper function to generate initialisation code for any reduction operations carried out be custom update group.
    //! Returns vector of ReductionTarget structs, providing all information to write back reduction results to memory
    std::vector<ReductionTarget> genInitReductionTargets(CodeStream &os, const CustomUpdateGroupMerged &cg, 
                                                         unsigned int batchSize, const std::string &idx = "") const;

    //! Helper function to generate initialisation code for any reduction operations carried out be custom weight update group.
    //! //! Returns vector of ReductionTarget structs, providing all information to write back reduction results to memory
    std::vector<ReductionTarget> genInitReductionTargets(CodeStream &os, const CustomUpdateWUGroupMerged &cg, 
                                                         unsigned int batchSize, const std::string &idx = "") const;

private:
    //--------------------------------------------------------------------------
    // Private API
    //--------------------------------------------------------------------------
    template<typename A, typename G, typename R>
    std::vector<ReductionTarget> genInitReductionTargets(CodeStream &os, const G &cg, unsigned int batchSize, 
                                                         const std::string &idx, R getVarRefIndexFn) const
    {
        // Loop through variables
        std::vector<ReductionTarget> reductionTargets;
        const auto *cm = cg.getArchetype().getModel();
        for (const auto &v : cm->getVars()) {
            // If variable is a reduction target, define variable initialised to correct initial value for reduction
            if (v.access & VarAccessModeAttribute::REDUCE) {
                const auto resolvedType = v.type.resolve(cg.getTypeContext());
                os << resolvedType.getName() << " _lr" << v.name << " = " << getReductionInitialValue(getVarAccessMode(v.access), resolvedType) << ";" << std::endl;
                const VarAccessDim varAccessDim = getVarAccessDim(v.access, cg.getArchetype().getDims());
                reductionTargets.push_back({v.name, resolvedType, getVarAccessMode(v.access),
                                            cg.getVarIndex(batchSize, varAccessDim, idx)});
            }
        }

        // Loop through all variable references
        for(const auto &modelVarRef : cm->getVarRefs()) {
            const auto &varRef = cg.getArchetype().getVarReferences().at(modelVarRef.name);

            // If variable reference is a reduction target, define variable initialised to correct initial value for reduction
            if (modelVarRef.access & VarAccessModeAttribute::REDUCE) {
                const auto resolvedType = modelVarRef.type.resolve(cg.getTypeContext());
                os << resolvedType.getName() << " _lr" << modelVarRef.name << " = " << getReductionInitialValue(modelVarRef.access, resolvedType) << ";" << std::endl;
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

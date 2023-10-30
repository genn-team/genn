#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>

// GeNN includes
#include "backendExport.h"
#include "varAccess.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/environment.h"

// Forward declarations
namespace GeNN::CodeGenerator
{
    class CustomUpdateWUGroupMergedBase;
}
namespace filesystem
{
    class path;
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::SingleThreadedCPU::Preferences
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::SingleThreadedCPU
{
struct Preferences : public PreferencesBase
{
};

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Array
//--------------------------------------------------------------------------
class BACKEND_EXPORT Array : public Runtime::ArrayBase
{
public:
    Array(const Type::ResolvedType &type, size_t count, 
          VarLocation location, bool uninitialized);
    virtual ~Array();
    
    //------------------------------------------------------------------------
    // ArrayBase virtuals
    //------------------------------------------------------------------------
    //! Allocate array
    virtual void allocate(size_t count) final;

    //! Free array
    virtual void free() final;

    //! Copy array to device
    virtual void pushToDevice() final
    {
    }

    //! Copy array from device
    virtual void pullFromDevice() final
    {
    }
    
    //! Memset the host pointer
    virtual void memsetDeviceObject(int) final
    {
        throw std::runtime_error("Single-threaded CPU arrays have no device objects");
    }

    //! Serialise backend-specific device object to bytes
    virtual void serialiseDeviceObject(std::vector<std::byte>&, bool) const final
    {
        throw std::runtime_error("Single-threaded CPU arrays have no device objects");
    }

    //! Serialise backend-specific host object to bytes
    virtual void serialiseHostObject(std::vector<std::byte>&, bool) const
    {
        throw std::runtime_error("Single-threaded CPU arrays have no host objects");
    }
};

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Backend
//--------------------------------------------------------------------------
class BACKEND_EXPORT Backend : public BackendBase
{
public:
    Backend(const Preferences &preferences)
    :   BackendBase(preferences)
    {
    }

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendBase virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                 HostHandler preambleHandler) const final;

    virtual void genSynapseUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                  HostHandler preambleHandler) const final;

    virtual void genCustomUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                 HostHandler preambleHandler) const final;

    virtual void genInit(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                         HostHandler preambleHandler) const final;

    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const final;

    virtual void genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genFreeMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;

    //! Create backend-specific array object
    /*! \param type         data type of array
        \param count        number of elements in array, if non-zero will allocate
        \param location     location of array e.g. device-only*/
    virtual std::unique_ptr<Runtime::ArrayBase> createArray(const Type::ResolvedType &type, size_t count, 
                                                            VarLocation location, bool uninitialized) const final;

    //! Create array of backend-specific population RNGs (if they are initialised on host this will occur here)
    /*! \param count        number of RNGs required*/
    virtual std::unique_ptr<Runtime::ArrayBase> createPopulationRNG(size_t) const final{ return std::unique_ptr<Runtime::ArrayBase>(); }

    //! Generate code to allocate variable with a size known at runtime
    virtual void genLazyVariableDynamicAllocation(CodeStream &os, 
                                                  const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                                  const std::string &countVarName) const final;

    //! Generate code for pushing a variable with a size known at runtime to the 'device'
    virtual void genLazyVariableDynamicPush(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name,
                                            VarLocation loc, const std::string &countVarName) const final;

    //! Generate code for pulling a variable with a size known at runtime from the 'device'
    virtual void genLazyVariableDynamicPull(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name,
                                            VarLocation loc, const std::string &countVarName) const final;

    //! Generate code for pushing a new pointer to a dynamic variable into the merged group structure on 'device'
    virtual void genMergedDynamicVariablePush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                              const std::string &groupIdx, const std::string &fieldName,
                                              const std::string &egpName) const final;

    //! When generating function calls to push to merged groups, backend without equivalent of Unified Virtual Addressing e.g. OpenCL 1.2 may use different types on host
    virtual std::string getMergedGroupFieldHostTypeName(const Type::ResolvedType &type) const final;

    virtual void genPopVariableInit(EnvironmentExternalBase &env, HandlerEnv handler) const final;
    virtual void genVariableInit(EnvironmentExternalBase &env, const std::string &count, const std::string &indexVarName, HandlerEnv handler) const final;
    virtual void genSparseSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const final;
    virtual void genDenseSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const final;
    virtual void genKernelSynapseVariableInit(EnvironmentExternalBase &env, SynapseInitGroupMerged &sg, HandlerEnv handler) const final;
    virtual void genKernelCustomUpdateVariableInit(EnvironmentExternalBase &env, CustomWUUpdateInitGroupMerged &cu, HandlerEnv handler) const final;

    virtual void genGlobalDeviceRNG(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free) const final;
    virtual void genTimer(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free, CodeStream &stepTimeFinalise, 
                          const std::string &name, bool updateInStepTime) const final;

    //! Generate code to return amount of free 'device' memory in bytes
    virtual void genReturnFreeDeviceMemoryBytes(CodeStream &os) const final;

     //! On backends which support it, generate a runtime assert
    virtual void genAssert(CodeStream &os, const std::string &condition) const final;

    virtual void genMakefilePreamble(std::ostream &os) const final;
    virtual void genMakefileLinkRule(std::ostream &os) const final;
    virtual void genMakefileCompileRule(std::ostream &os) const final;

    virtual void genMSBuildConfigProperties(std::ostream &os) const final;
    virtual void genMSBuildImportProps(std::ostream &os) const final;
    virtual void genMSBuildItemDefinitions(std::ostream &os) const final;
    virtual void genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const final;
    virtual void genMSBuildImportTarget(std::ostream &os) const final;

    //! As well as host pointers, are device objects required?
    virtual bool isArrayDeviceObjectRequired() const final{ return false; }

    //! As well as host pointers, are additional host objects required e.g. for buffers in OpenCL?
    virtual bool isArrayHostObjectRequired() const final{ return false; }

    virtual bool isGlobalHostRNGRequired(const ModelSpecInternal &model) const final;
    virtual bool isGlobalDeviceRNGRequired(const ModelSpecInternal &model) const final;

    //! Different backends seed RNGs in different ways. Does this one initialise population RNGS on device?
    virtual bool isPopulationRNGInitialisedOnDevice() const final { return false; }

    virtual bool isPostsynapticRemapRequired() const final{ return true; }

    //! Backends which support batch-parallelism might require an additional host reduction phase after reduction kernels
    virtual bool isHostReductionRequired() const final { return false; }

    //! How many bytes of memory does 'device' have
    virtual size_t getDeviceMemoryBytes() const final{ return 0; }

    //! Some backends will have additional small, fast, memory spaces for read-only data which might
    //! Be well-suited to storing merged group structs. This method returns the prefix required to
    //! Place arrays in these and their size in preferential order
    virtual MemorySpaces getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const final;

    //! Get hash digest of this backends identification and the preferences it has been configured with
    virtual boost::uuids::detail::sha1::digest_type getHashDigest() const final;

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    void genPresynapticUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                              double dt, bool trueSpike) const;

    void genEmitSpike(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng, bool trueSpike, bool recordingEnabled) const;

    //! Helper to generate code to copy reduced custom update group variables back to memory
    /*! Because reduction operations are unnecessary in unbatched single-threaded CPU models so there's no need to actually reduce */
    void genWriteBackReductions(EnvironmentExternalBase &env, CustomUpdateGroupMerged &cg, const std::string &idxName) const;

    //! Helper to generate code to copy reduced custom weight update group variables back to memory
    /*! Because reduction operations are unnecessary in unbatched single-threaded CPU models so there's no need to actually reduce */
    void genWriteBackReductions(EnvironmentExternalBase &env, CustomUpdateWUGroupMergedBase &cg, const std::string &idxName) const;

    template<typename G, typename R>
    void genWriteBackReductions(EnvironmentExternalBase &env, G &cg, const std::string &idxName, R getVarRefIndexFn) const
    {
        const auto *cm = cg.getArchetype().getCustomUpdateModel();
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
};
}   // namespace GeNN::SingleThreadedCPU::CodeGenerator

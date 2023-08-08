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
    virtual void genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const final;
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const final;
    virtual void genFreeMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;

    //! Generate code to define a variable in the appropriate header file
    virtual void genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, 
                                       const Type::ResolvedType &type, const std::string &name, VarLocation loc) const final;
    
    //! Generate code to instantiate a variable in the provided stream
    virtual void genVariableInstantiation(CodeStream &os, 
                                          const Type::ResolvedType &type, const std::string &name, VarLocation loc) const final;

    //! Generate code to allocate variable with a size known at compile-time
    virtual void genVariableAllocation(CodeStream &os, 
                                       const Type::ResolvedType &type, const std::string &name, 
                                       VarLocation loc, size_t count, MemAlloc &memAlloc) const final;
    
    //! Generate code to allocate variable with a size known at runtime
    virtual void genVariableDynamicAllocation(CodeStream &os, 
                                              const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                              const std::string &countVarName = "count", const std::string &prefix = "") const final;

    //! Generate code to allocate variable with a size known at runtime
    virtual void genLazyVariableDynamicAllocation(CodeStream &os, 
                                                  const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                                  const std::string &countVarName) const final;

    //! Generate code to free a variable
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const final;

    //! Generate code for pushing a variable with a size known at compile-time to the 'device'
    virtual void genVariablePush(CodeStream &os, 
                                 const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                 bool autoInitialized, size_t count) const final;
    
    //! Generate code for pulling a variable with a size known at compile-time from the 'device'
    virtual void genVariablePull(CodeStream &os, 
                                 const Type::ResolvedType &type, const std::string &name, 
                                 VarLocation loc, size_t count) const final;

    //! Generate code for pushing a variable's value in the current timestep to the 'device'
    virtual void genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, 
                                        const Type::ResolvedType &type, const std::string &name, 
                                        VarLocation loc, unsigned int batchSize) const final;

    //! Generate code for pulling a variable's value in the current timestep from the 'device'
    virtual void genCurrentVariablePull(CodeStream &os, const NeuronGroupInternal &ng, 
                                        const Type::ResolvedType &type, const std::string &name,
                                        VarLocation loc, unsigned int batchSize) const final;

    //! Generate code for pushing a variable with a size known at runtime to the 'device'
    virtual void genVariableDynamicPush(CodeStream &os, 
                                        const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                        const std::string &countVarName = "count", const std::string &prefix = "") const final;

    //! Generate code for pushing a variable with a size known at runtime to the 'device'
    virtual void genLazyVariableDynamicPush(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                            const std::string &countVarName) const final;

    //! Generate code for pulling a variable with a size known at runtime from the 'device'
    virtual void genVariableDynamicPull(CodeStream &os, 
                                        const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                        const std::string &countVarName = "count", const std::string &prefix = "") const final;

    //! Generate code for pulling a variable with a size known at runtime from the 'device'
    virtual void genLazyVariableDynamicPull(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                            const std::string &countVarName) const final;

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

    virtual void genGlobalDeviceRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, 
                                    CodeStream &allocations, CodeStream &free, MemAlloc &memAlloc) const final;
    virtual void genPopulationRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, 
                                  CodeStream &allocations, CodeStream &free,
                                  const std::string &name, size_t count, MemAlloc &memAlloc) const final;
    virtual void genTimer(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                          CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const final;

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

    virtual std::string getDeviceVarPrefix() const final{ return ""; }

    //! Should 'scalar' variables be implemented on device or can host variables be used directly?
    virtual bool isDeviceScalarRequired() const final { return false; }

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

    virtual bool supportsNamespace() const final { return true; };
    
    //! Get hash digest of this backends identification and the preferences it has been configured with
    virtual boost::uuids::detail::sha1::digest_type getHashDigest() const final;

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    void genPresynapticUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const ModelSpecMerged &modelMerged, bool trueSpike) const;

    void genEmitSpike(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng, bool trueSpike, bool recordingEnabled) const;

    template<typename T>
    void genMergedStructArrayPush(CodeStream &os, const std::vector<T> &groups) const
    {
        // Loop through groups
        for(const auto &g : groups) {
            // Check there's no memory space assigned as single-threaded CPU backend doesn't support them
            assert(g.getMemorySpace().empty());

            // Implement merged group
            os << "static Merged" << T::name << "Group" << g.getIndex() << " merged" << T::name << "Group" << g.getIndex() << "[" << g.getGroups().size() << "];" << std::endl;

            // Write function to update
            os << "void pushMerged" << T::name << "Group" << g.getIndex() << "ToDevice(unsigned int idx, ";
            g.generateStructFieldArgumentDefinitions(os, *this);
            os << ")";
            {
                CodeStream::Scope b(os);

                // Loop through sorted fields and set array entry
                const auto sortedFields = g.getSortedFields(*this);
                for(const auto &f : sortedFields) {
                    os << "merged" << T::name << "Group" << g.getIndex() << "[idx]." << std::get<1>(f) << " = " << std::get<1>(f) << ";" << std::endl;
                }
            }
        }
    }

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
                env.getStream() << "group->" << v.name << "[" << cg.getVarIndex(getVarAccessDuplication(v.access), idx) << "] = " << env[v.name] << ";" << std::endl;
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

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
// GeNN::CodeGenerator::ISPC::Preferences
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::ISPC
{
struct Preferences : public PreferencesBase
{
    //! Which SIMD instruction set to target e.g. sse4, avx, avx2, avx512skx-i32x8
    std::string targetISA = "avx2";

    //! Update hash with preferences
    virtual void updateHash(boost::uuids::detail::sha1 &hash) const final;

};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::State
//--------------------------------------------------------------------------
class State : public GeNN::Runtime::StateBase
{
public:
    State(const GeNN::Runtime::Runtime &runtime);
};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Array
//--------------------------------------------------------------------------
class Array : public GeNN::Runtime::ArrayBase
{
public:
    Array(const Type::ResolvedType &type, size_t count, 
          VarLocation location, bool uninitialized, size_t alignment);
    
    virtual ~Array();
    
    //------------------------------------------------------------------------
    // ArrayBase virtuals
    //------------------------------------------------------------------------
    virtual void allocate(size_t count) final;
    virtual void free() final;
    virtual void pushToDevice() final;
    virtual void pullFromDevice() final;
    virtual void pushSlice1DToDevice(size_t sliceIndex, size_t sliceSize) final;
    virtual void pullSlice1DFromDevice(size_t sliceIndex, size_t sliceSize) final;
    virtual void memsetDeviceObject(int value) final;
    virtual void serialiseDeviceObject(std::vector<std::byte> &result, bool compress) const final;
    virtual void serialiseHostObject(std::vector<std::byte> &result, bool compress) const final;

private:
    size_t m_Alignment;
};


//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Backend
//--------------------------------------------------------------------------
class BACKEND_EXPORT Backend : public BackendBase
{
public:
    Backend();

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

    virtual std::unique_ptr<GeNN::Runtime::StateBase> createState(const Runtime::Runtime &runtime) const final;

    virtual std::unique_ptr<Runtime::ArrayBase> createArray(const Type::ResolvedType &type, size_t count, 
                                                            VarLocation location, bool uninitialized) const final;

    virtual std::unique_ptr<Runtime::ArrayBase> createPopulationRNG(size_t) const final;

    virtual void genLazyVariableDynamicAllocation(CodeStream &os, 
                                                  const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                                  const std::string &countVarName) const final;

    virtual void genLazyVariableDynamicPush(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name,
                                            VarLocation loc, const std::string &countVarName) const final;

    virtual void genLazyVariableDynamicPull(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name,
                                            VarLocation loc, const std::string &countVarName) const final;

    virtual void genMergedDynamicVariablePush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                              const std::string &groupIdx, const std::string &fieldName,
                                              const std::string &egpName) const final;

    virtual std::string getMergedGroupFieldHostTypeName(const Type::ResolvedType &type) const final;

    virtual void genPopVariableInit(EnvironmentExternalBase &env, HandlerEnv handler) const final;
    virtual void genVariableInit(EnvironmentExternalBase &env, const std::string &count, const std::string &indexVarName, HandlerEnv handler) const final;
    virtual void genSparseSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const final;
    virtual void genDenseSynapseVariableRowInit(EnvironmentExternalBase &env, HandlerEnv handler) const final;
    virtual void genKernelSynapseVariableInit(EnvironmentExternalBase &env, SynapseInitGroupMerged &sg, HandlerEnv handler) const final;
    virtual void genKernelCustomUpdateVariableInit(EnvironmentExternalBase &env, CustomWUUpdateInitGroupMerged &cu, HandlerEnv handler) const final;

    virtual std::string getAtomicOperation(const std::string &lhsPointer, const std::string &rhsValue,
                                           const Type::ResolvedType &type, AtomicOperation op = AtomicOperation::ADD) const final;

    virtual void genGlobalDeviceRNG(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free) const final;
    virtual void genTimer(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free, CodeStream &stepTimeFinalise, 
                          const std::string &name, bool updateInStepTime) const final;
    
    virtual void genReturnFreeDeviceMemoryBytes(CodeStream &os) const final;
    
    virtual void genAssert(CodeStream &os, const std::string &condition) const final;

    virtual void genMakefilePreamble(std::ostream &os) const final;
    virtual void genMakefileLinkRule(std::ostream &os) const final;
    virtual void genMakefileCompileRule(std::ostream &os) const final;

    virtual void genMSBuildConfigProperties(std::ostream &os) const final;
    virtual void genMSBuildImportProps(std::ostream &os) const final;
    virtual void genMSBuildItemDefinitions(std::ostream &os) const final;
    virtual void genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const final;
    virtual void genMSBuildImportTarget(std::ostream &os) const final;
    
    virtual bool isArrayDeviceObjectRequired() const final;
    
    virtual bool isArrayHostObjectRequired() const final;

    virtual bool isGlobalHostRNGRequired(const ModelSpecInternal &model) const final;
    virtual bool isGlobalDeviceRNGRequired(const ModelSpecInternal &model) const final;
    
    virtual bool isPopulationRNGInitialisedOnDevice() const final;

    virtual bool isPostsynapticRemapRequired() const final;
    
    virtual bool isHostReductionRequired() const final;
    
    virtual size_t getDeviceMemoryBytes() const final;
    
    virtual MemorySpaces getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const final;
    
    virtual boost::uuids::detail::sha1::digest_type getHashDigest() const final;

};
}

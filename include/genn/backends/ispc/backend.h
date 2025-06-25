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
    //! Which SIMD instruction set to target
    std::string targetISA = "avx2";
    
    //! Maximize use of uniform variables for performance
    bool maximizeUniforms = true;

    //! Debug flag - if true, more debug information is generated
    bool debugCode = false;
    
    //! Optimize flag - if true, generates optimized code
    bool optimizeCode = true;
    
    //! If true, optimize for size rather than speed
    bool optimizeForSize = false;
    
    //! Update hash with preferences
    EXPORT_GENN virtual void updateHash(boost::uuids::detail::sha1 &hash) const override;

    //! Get import suffix
    EXPORT_GENN virtual const char *getImportSuffix() const override;
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
          VarLocation location, bool uninitialized);
    
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
};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Backend
//--------------------------------------------------------------------------
class Backend : public GeNN::CodeGenerator::BackendBase
{
public:
    Backend()
    {
        setPreferencesBase(std::make_shared<Preferences>());
    }
    
    //--------------------------------------------------------------------------
    // GeNN::CodeGenerator::BackendBase virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                HostHandler preambleHandler) const override;
                                
    virtual void genSynapseUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                 HostHandler preambleHandler) const override;
                                 
    virtual void genCustomUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                               HostHandler preambleHandler) const override;
                               
    virtual void genInit(CodeStream &os, const ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces,
                       HostHandler preambleHandler) const override;
                       
    virtual void genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const std::string&) const override;
    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    
    virtual void genNeuronPrevSpikeTimeUpdateKernel(EnvironmentExternalBase &env, NeuronPrevSpikeTimeUpdateGroupMerged &ng,
                                                 BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const override;
                                                  
    virtual void genNeuronSpikeQueueUpdateKernel(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged, 
                                               BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const override;
                                               
    virtual void genNeuronUpdateKernel(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged,
                                     BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const override;
                                     
    virtual void genInitializeKernel(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged,
                                   BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const override;
                                   
    virtual void genInitializeSparseKernel(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged,
                                         BackendBase::MemorySpaces &memorySpaces, size_t &idStart) const override;
    
    virtual void genAssert(CodeStream &os, const std::string &condition) const override;
    virtual void genMakefilePreamble(std::ostream &os) const override;
    virtual void genMakefileLinkRule(std::ostream &os) const override;
    virtual void genMakefileCompileRule(std::ostream &os) const override;
    virtual void genMSBuildConfigProperties(std::ostream &os) const override;
    virtual void genMSBuildImportProps(std::ostream &os) const override;
    virtual void genMSBuildItemDefinitions(std::ostream &os) const override;
    virtual void genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const override;
    virtual void genMSBuildImportTarget(std::ostream &os) const override;
    
    virtual std::vector<filesystem::path> getFilesToCopy(const ModelSpecMerged &model) const override;
    virtual boost::uuids::detail::sha1::digest_type getHashDigest() const override;
    virtual void genReturnFreeDeviceMemoryBytes(CodeStream &os) const override;
    
    virtual bool isGlobalHostRNGRequired(const ModelSpecInternal &model) const override;
    virtual bool isGlobalDeviceRNGRequired(const ModelSpecInternal &model) const override;
    virtual BackendBase::MemorySpaces getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const override;
    virtual Type::ResolvedType getPopulationRNGType() const override { return Type::Void; }
    
    //--------------------------------------------------------------------------
    // Public methods
    //--------------------------------------------------------------------------
    //! Get appropriate ISPC type for GeNN type
    std::string getISPCType(const Type::ResolvedType &type, bool uniform = false) const;
    
    //! Generate custom neuron model code for ISPC
    void genCustomNeuronModelCode(CodeStream &os, const NeuronGroupInternal &ng, 
                                const std::string &modelName) const;
    
protected:
    //--------------------------------------------------------------------------
    // Protected methods
    //--------------------------------------------------------------------------
    //! Get the string name for the given neuron model
    std::string getNeuronModelType(const NeuronGroupInternal &ng) const;
    
    //! Get the kernel function name for the given neuron model type
    std::string getNeuronKernelName(const std::string &modelType) const;
    
    // Backend-specific implementation to generate for conditions in presynaptic update
    virtual void genPresynapticUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                    double dt, bool trueSpike) const override;
                                    
    // Backend-specific implementation to generate for each synapse post-update
    virtual void genPostsynapticUpdate(EnvironmentExternalBase &env, PostsynapticUpdateGroupMerged &sg, 
                                     double dt, bool trueSpike) const override;
                                     
    // Backend-specific implementation to generate prev. spike time update triggered by events
    virtual void genPrevEventTimeUpdate(EnvironmentExternalBase &env, NeuronPrevSpikeTimeUpdateGroupMerged &ng,
                                      bool trueSpike) const override;
};

} // namespace GeNN::CodeGenerator::ISPC 

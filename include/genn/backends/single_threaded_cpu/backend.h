#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>

// GeNN includes
#include "backendExport.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

// Forward declarations
namespace filesystem
{
    class path;
}

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Preferences
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace SingleThreadedCPU
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
    Backend(const std::string &scalarType, const Preferences &preferences)
    :   BackendBase(scalarType), m_Preferences(preferences)
    {
    }

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendBase virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged,
                                 NeuronGroupSimHandler simHandler, NeuronGroupMergedHandler wuVarUpdateHandler,
                                 HostHandler pushEGPHandler) const override;

    virtual void genSynapseUpdate(CodeStream &os, const ModelSpecMerged &modelMerged,
                                  SynapseGroupMergedHandler wumThreshHandler, SynapseGroupMergedHandler wumSimHandler,
                                  SynapseGroupMergedHandler wumEventHandler, SynapseGroupMergedHandler wumProceduralConnectHandler,
                                  SynapseGroupMergedHandler postLearnHandler, SynapseGroupMergedHandler synapseDynamicsHandler,
                                  HostHandler pushEGPHandler) const override;

    virtual void genInit(CodeStream &os, const ModelSpecMerged &modelMerged,
                         NeuronGroupMergedHandler localNGHandler, SynapseGroupMergedHandler sgDenseInitHandler, 
                         SynapseGroupMergedHandler sgSparseConnectHandler, SynapseGroupMergedHandler sgSparseInitHandler,
                         HostHandler initPushEGPHandler, HostHandler initSparsePushEGPHandler) const override;

    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const override;

    virtual void genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;

    virtual void genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual MemAlloc genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const override;
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const override;

    virtual void genExtraGlobalParamDefinition(CodeStream &definitions, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name, 
                                               VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;
    virtual void genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name, 
                                         VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;
    virtual void genExtraGlobalParamPull(CodeStream &os, const std::string &type, const std::string &name, 
                                         VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;

    //! Generate code for declaring merged group data to the 'device'
    virtual void genMergedGroupImplementation(CodeStream &os, const std::string &suffix, size_t idx, size_t numGroups) const override;
    
    //! Generate code for pushing merged group data to the 'device'
    virtual void genMergedGroupPush(CodeStream &os, const std::string &suffix, size_t idx, size_t numGroups) const override;

    ///! Generate code for pushing an updated EGP value into the merged group structure on 'device'
    virtual void genMergedExtraGlobalParamPush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                               const std::string &groupIdx, const std::string &fieldName,
                                               const std::string &egpName) const override;

    virtual void genPopVariableInit(CodeStream &os,const Substitutions &kernelSubs, Handler handler) const override;
    virtual void genVariableInit(CodeStream &os, const std::string &count, const std::string &indexVarName,
                                 const Substitutions &kernelSubs, Handler handler) const override;
    virtual void genSynapseVariableRowInit(CodeStream &os, const SynapseGroupMerged &sg,
                                           const Substitutions &kernelSubs, Handler handler) const override;

    virtual void genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const override;
    virtual void genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const override;
    virtual void genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genCurrentVariablePull(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, const std::string &name, VarLocation loc) const override;

    virtual void genCurrentTrueSpikePush(CodeStream &os, const NeuronGroupInternal &ng) const override;
    virtual void genCurrentTrueSpikePull(CodeStream &os, const NeuronGroupInternal &ng) const override;
    virtual void genCurrentSpikeLikeEventPush(CodeStream &os, const NeuronGroupInternal &ng) const override;
    virtual void genCurrentSpikeLikeEventPull(CodeStream &os, const NeuronGroupInternal &ng) const override;

    virtual MemAlloc genGlobalRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free) const override;
    virtual MemAlloc genPopulationRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                                      const std::string &name, size_t count) const override;
    virtual void genTimer(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                          CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const override;

    //! Generate code to return amount of free 'device' memory in bytes
    virtual void genReturnFreeDeviceMemoryBytes(CodeStream &os) const override;

    virtual void genMakefilePreamble(std::ostream &os) const override;
    virtual void genMakefileLinkRule(std::ostream &os) const override;
    virtual void genMakefileCompileRule(std::ostream &os) const override;

    virtual void genMSBuildConfigProperties(std::ostream &os) const override;
    virtual void genMSBuildImportProps(std::ostream &os) const override;
    virtual void genMSBuildItemDefinitions(std::ostream &os) const override;
    virtual void genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const override;
    virtual void genMSBuildImportTarget(std::ostream &os) const override;

    virtual std::string getArrayPrefix() const override{ return ""; }
    virtual std::string getScalarPrefix() const override{ return ""; }

    virtual bool isGlobalDeviceRNGRequired(const ModelSpecMerged &modelMerged) const override;
    virtual bool isPopulationRNGRequired() const override { return false; }
    virtual bool isSynRemapRequired() const override{ return false; }
    virtual bool isPostsynapticRemapRequired() const override{ return true; }

    //! Is automatic copy mode enabled in the preferences?
    virtual bool isAutomaticCopyEnabled() const override { return m_Preferences.automaticCopy; }

    //! Should GeNN generate empty state push and pull functions?
    virtual bool shouldGenerateEmptyStatePushPull() const override { return m_Preferences.generateEmptyStatePushPull; }

    //! Should GeNN generate pull functions for extra global parameters? These are very rarely used
    virtual bool shouldGenerateExtraGlobalParamPull() const override { return m_Preferences.generateExtraGlobalParamPull; }

    //! How many bytes of memory does 'device' have
    virtual size_t getDeviceMemoryBytes() const override{ return 0; }

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    void genPresynapticUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const SynapseGroupMerged &sg, const Substitutions &popSubs,
                              bool trueSpike, SynapseGroupMergedHandler wumThreshHandler, SynapseGroupMergedHandler wumSimHandler) const;

    void genEmitSpike(CodeStream &os, const NeuronGroupMerged &ng, const Substitutions &subs, bool trueSpike) const;

  
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const Preferences m_Preferences;
};
}   // namespace SingleThreadedCPU
}   // namespace CodeGenerator

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
    :   BackendBase(scalarType, preferences)
    {
    }

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendBase virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces,
                                 HostHandler preambleHandler, NeuronGroupSimHandler simHandler, NeuronUpdateGroupMergedHandler wuVarUpdateHandler,
                                 HostHandler pushEGPHandler) const override;

    virtual void genSynapseUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces,
                                  HostHandler preambleHandler, PresynapticUpdateGroupMergedHandler wumThreshHandler, PresynapticUpdateGroupMergedHandler wumSimHandler,
                                  PresynapticUpdateGroupMergedHandler wumEventHandler, PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler,
                                  PostsynapticUpdateGroupMergedHandler postLearnHandler, SynapseDynamicsGroupMergedHandler synapseDynamicsHandler,
                                  HostHandler pushEGPHandler) const override;

    virtual void genCustomUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces, HostHandler preambleHandler,
                                 CustomUpdateGroupMergedHandler customUpdateHandler, CustomUpdateWUGroupMergedHandler customWUUpdateHandler,
                                 CustomUpdateTransposeWUGroupMergedHandler customWUTransposeUpdateHandler, HostHandler pushEGPHandler) const override;

    virtual void genInit(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces,
                         HostHandler preambleHandler, NeuronInitGroupMergedHandler localNGHandler, CustomUpdateInitGroupMergedHandler cuHandler,
                         CustomWUUpdateDenseInitGroupMergedHandler cuDenseHandler, SynapseDenseInitGroupMergedHandler sgDenseInitHandler, 
                         SynapseConnectivityInitMergedGroupHandler sgSparseRowConnectHandler,  SynapseConnectivityInitMergedGroupHandler sgSparseColConnectHandler,
                         SynapseConnectivityInitMergedGroupHandler sgKernelInitHandler, SynapseSparseInitGroupMergedHandler sgSparseInitHandler, 
                         CustomWUUpdateSparseInitGroupMergedHandler cuSparseHandler, HostHandler initPushEGPHandler, HostHandler initSparsePushEGPHandler) const override;

    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const override;

    virtual void genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const override;
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const override;
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;

    virtual void genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count, MemAlloc &memAlloc) const override;
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const override;

    virtual void genExtraGlobalParamDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name, 
                                               VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;
    virtual void genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name, 
                                         VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;
    virtual void genExtraGlobalParamPull(CodeStream &os, const std::string &type, const std::string &name, 
                                         VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;

    ///! Generate code for pushing an updated EGP value into the merged group structure on 'device'
    virtual void genMergedExtraGlobalParamPush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                               const std::string &groupIdx, const std::string &fieldName,
                                               const std::string &egpName) const override;

    //! When generating function calls to push to merged groups, backend without equivalent of Unified Virtual Addressing e.g. OpenCL 1.2 may use different types on host
    virtual std::string getMergedGroupFieldHostType(const std::string &type) const override;

    //! When generating merged structures what type to use for simulation RNGs
    virtual std::string getMergedGroupSimRNGType() const override;

    virtual void genPopVariableInit(CodeStream &os,const Substitutions &kernelSubs, Handler handler) const override;
    virtual void genVariableInit(CodeStream &os, const std::string &count, const std::string &indexVarName,
                                 const Substitutions &kernelSubs, Handler handler) const override;
    virtual void genSparseSynapseVariableRowInit(CodeStream &os, const Substitutions &kernelSubs, Handler handler) const override;
    virtual void genDenseSynapseVariableRowInit(CodeStream &os, const Substitutions &kernelSubs, Handler handler) const override;

    virtual void genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const override;
    virtual void genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const override;
    virtual void genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, 
                                        const std::string &name, VarLocation loc, unsigned int batchSize) const override;
    virtual void genCurrentVariablePull(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, 
                                        const std::string &name, VarLocation loc, unsigned int batchSize) const override;

    virtual void genCurrentTrueSpikePush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override;
    virtual void genCurrentTrueSpikePull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override;
    virtual void genCurrentSpikeLikeEventPush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override;
    virtual void genCurrentSpikeLikeEventPull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override;

    virtual void genGlobalDeviceRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, 
                                    CodeStream &allocations, CodeStream &free, MemAlloc &memAlloc) const override;
    virtual void genPopulationRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, 
                                  CodeStream &allocations, CodeStream &free,
                                  const std::string &name, size_t count, MemAlloc &memAlloc) const override;
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

    virtual std::string getDeviceVarPrefix() const override{ return ""; }

    //! Should 'scalar' variables be implemented on device or can host variables be used directly?
    virtual bool isDeviceScalarRequired() const override { return false; }

    virtual bool isGlobalHostRNGRequired(const ModelSpecMerged &modelMerged) const override;
    virtual bool isGlobalDeviceRNGRequired(const ModelSpecMerged &modelMerged) const override;
    virtual bool isPopulationRNGRequired() const override { return false; }

    //! Different backends seed RNGs in different ways. Does this one initialise population RNGS on device?
    virtual bool isPopulationRNGInitialisedOnDevice() const override { return false; }

    virtual bool isSynRemapRequired(const SynapseGroupInternal&) const override{ return false; }
    virtual bool isPostsynapticRemapRequired() const override{ return true; }

    //! How many bytes of memory does 'device' have
    virtual size_t getDeviceMemoryBytes() const override{ return 0; }

    //! Some backends will have additional small, fast, memory spaces for read-only data which might
    //! Be well-suited to storing merged group structs. This method returns the prefix required to
    //! Place arrays in these and their size in preferential order
    virtual MemorySpaces getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const override;

    virtual bool supportsNamespace() const override { return true; };

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    void genPresynapticUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg, const Substitutions &popSubs,
                              bool trueSpike, PresynapticUpdateGroupMergedHandler wumThreshHandler, PresynapticUpdateGroupMergedHandler wumSimHandler) const;

    void genEmitSpike(CodeStream &os, const NeuronUpdateGroupMerged &ng, const Substitutions &subs, bool trueSpike, bool recordingEnabled) const;

    template<typename T>
    void genMergedStructArrayPush(CodeStream &os, const std::vector<T> &groups) const
    {
        // Loop through groups
        for(const auto &g : groups) {
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
};
}   // namespace SingleThreadedCPU
}   // namespace CodeGenerator

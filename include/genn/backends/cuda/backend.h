#pragma once

// Standard C++ includes
#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <numeric>
#include <string>

// Standard C includes
#include <cassert>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>

// GeNN includes
#include "backendExport.h"

// GeNN code generator includes
#include "code_generator/backendSIMT.h"
#include "code_generator/codeStream.h"
#include "code_generator/substitutions.h"

// Forward declarations
namespace filesystem
{
    class path;
}

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::DeviceSelectMethod
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace CUDA
{
//! Methods for selecting CUDA device
enum class DeviceSelect
{
    OPTIMAL,        //!< Pick optimal device based on how well kernels can be simultaneously simulated and occupancy
    MOST_MEMORY,    //!< Pick device with most global memory
    MANUAL,         //!< Use device specified by user
    MANUAL_RUNTIME, //!< Use device specified by user at runtime with allocateMem parameter. Optimisation will be performed on device specified with manualDeviceID.
};

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::BlockSizeSelect
//--------------------------------------------------------------------------
//! Methods for selecting CUDA kernel block size
enum class BlockSizeSelect
{
    OCCUPANCY,  //!< Pick optimal blocksize for each kernel based on occupancy
    MANUAL,     //!< Use block sizes specified by user
};

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::Preferences
//--------------------------------------------------------------------------
//! Preferences for CUDA backend
struct Preferences : public PreferencesBase
{
    Preferences()
    {
        std::fill(manualBlockSizes.begin(), manualBlockSizes.end(), 32);
    }

    //! Should PTX assembler information be displayed for each CUDA kernel during compilation?
    bool showPtxInfo = false;

    //! Should line info be included in resultant executable for debugging/profiling purposes?
    bool generateLineInfo = false;

    //! GeNN normally identifies devices by PCI bus ID to ensure that the model is run on the same device
    //! it was optimized for. However if, for example, you are running on a cluser with NVML this is not desired behaviour.
    bool selectGPUByDeviceID = false;

    //! How to select GPU device
    DeviceSelect deviceSelectMethod = DeviceSelect::OPTIMAL;

    //! If device select method is set to DeviceSelect::MANUAL, id of device to use
    unsigned int manualDeviceID = 0;

    //! How to select CUDA blocksize
    BlockSizeSelect blockSizeSelectMethod = BlockSizeSelect::OCCUPANCY;

    //! If block size select method is set to BlockSizeSelect::MANUAL, block size to use for each kernel
    KernelBlockSize manualBlockSizes;

    //! How much constant cache is already used and therefore can't be used by GeNN?
    /*! Each of the four modules which includes CUDA headers(neuronUpdate, synapseUpdate, init and runner)
        Takes 72 bytes of constant memory for a lookup table used by cuRAND. If your application requires
        additional constant cache, increase this */
    size_t constantCacheOverhead = 72 * 4;

    //! NVCC compiler options for all GPU code
    std::string userNvccFlags = "";
};

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::Backend
//--------------------------------------------------------------------------
class BACKEND_EXPORT Backend : public BackendSIMT
{
public:
    Backend(const KernelBlockSize &kernelBlockSizes, const Preferences &preferences,
            const std::string &scalarType, int device);

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendSIMT virtuals
    //--------------------------------------------------------------------------
    //! On some older devices, shared memory atomics are actually slower than global memory atomics so should be avoided
    virtual bool areSharedMemAtomicsSlow() const override;

    //! Get the prefix to use for shared memory variables
    virtual std::string getSharedPrefix() const override{ return "__shared__ "; }

    //! Get the ID of the current thread within the threadblock
    virtual std::string getThreadID(unsigned int axis = 0) const override;

    //! Get the ID of the current thread block
    virtual std::string getBlockID(unsigned int axis = 0) const override;

    //! Get the name of the count-leading-zeros function
    virtual std::string getCLZ() const override { return "__clz"; }

    //! Get name of atomic operation
    virtual std::string getAtomic(const std::string &type, AtomicOperation op = AtomicOperation::ADD,
                                  AtomicMemSpace memSpace = AtomicMemSpace::GLOBAL) const override;

    //! Generate a shared memory barrier
    virtual void genSharedMemBarrier(CodeStream &os) const override;

    //! For SIMT backends which initialize RNGs on device, initialize population RNG with specified seed and sequence
    virtual void genPopulationRNGInit(CodeStream &os, const std::string &globalRNG, const std::string &seed, const std::string &sequence) const override;

    //! Generate a preamble to add substitution name for population RNG
    virtual void genPopulationRNGPreamble(CodeStream &os, Substitutions &subs, const std::string &globalRNG, const std::string &name = "rng") const override;

    //! If required, generate a postamble for population RNG
    /*! For example, in OpenCL, this is used to write local RNG state back to global memory*/
    virtual void genPopulationRNGPostamble(CodeStream &os, const std::string &globalRNG) const override;

    //! Generate code to skip ahead local copy of global RNG
    virtual void genGlobalRNGSkipAhead(CodeStream &os, Substitutions &subs, const std::string &sequence, const std::string &name = "rng") const override;

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendBase virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces,
                                 HostHandler preambleHandler, NeuronGroupSimHandler simHandler, NeuronUpdateGroupMergedHandler wuVarUpdateHandler,
                                 HostHandler pushEGPHandler) const override;

    virtual void genSynapseUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces,
                                  HostHandler preambleHandler, PresynapticUpdateGroupMergedHandler wumThreshHandler, PresynapticUpdateGroupMergedHandler wumSimHandler,
                                  PresynapticUpdateGroupMergedHandler wumEventHandler, PostsynapticUpdateGroupMergedHandler postLearnHandler, SynapseDynamicsGroupMergedHandler synapseDynamicsHandler, 
                                  PresynapticUpdateGroupMergedHandler sgSparseRowConnectHandler,  PostsynapticUpdateGroupMergedHandler sgSparseColConnectHandler,
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

    virtual void genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;

    virtual void genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual MemAlloc genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const override;
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const override;

    virtual void genExtraGlobalParamDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name, 
                                               VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;
    virtual void genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name, 
                                         VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;
    virtual void genExtraGlobalParamPull(CodeStream &os, const std::string &type, const std::string &name, 
                                         VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;

    //! Generate code for pushing an updated EGP value into the merged group structure on 'device'
    virtual void genMergedExtraGlobalParamPush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                               const std::string &groupIdx, const std::string &fieldName,
                                               const std::string &egpName) const override;

    //! When generating function calls to push to merged groups, backend without equivalent of Unified Virtual Addressing e.g. OpenCL 1.2 may use different types on host
    virtual std::string getMergedGroupFieldHostType(const std::string &type) const override;

    //! When generating merged structures what type to use for simulation RNGs
    virtual std::string getMergedGroupSimRNGType() const override { return "curandState"; }

    virtual void genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const override;
    virtual void genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const override;

    virtual void genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, 
                                        const std::string &name, VarLocation loc, unsigned int batchSize) const override;
    virtual void genCurrentVariablePull(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type, 
                                        const std::string &name, VarLocation loc, unsigned int batchSize) const override;

    virtual void genCurrentTrueSpikePush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override
    {
        genCurrentSpikePush(os, ng, batchSize, false);
    }
    virtual void genCurrentTrueSpikePull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override
    {
        genCurrentSpikePull(os, ng, batchSize, false);
    }
    virtual void genCurrentSpikeLikeEventPush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override
    {
        genCurrentSpikePush(os, ng, batchSize, true);
    }
    virtual void genCurrentSpikeLikeEventPull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override
    {
        genCurrentSpikePull(os, ng, batchSize, true);
    }
    
    virtual MemAlloc genGlobalDeviceRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free) const override;
    virtual MemAlloc genPopulationRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner,
                                      CodeStream &allocations, CodeStream &free, const std::string &name, size_t count) const override;
    virtual void genTimer(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner,
                          CodeStream &allocations, CodeStream &free, CodeStream &stepTimeFinalise,
                          const std::string &name, bool updateInStepTime) const override;

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

    //! Get backend-specific allocate memory parameters
    virtual std::string getAllocateMemParams(const ModelSpecMerged &) const override;

    //! Different backends seed RNGs in different ways. Does this one initialise population RNGS on device?
    virtual bool isPopulationRNGInitialisedOnDevice() const override { return true; }

    //! How many bytes of memory does 'device' have
    virtual size_t getDeviceMemoryBytes() const override{ return m_ChosenDevice.totalGlobalMem; }

    //! Some backends will have additional small, fast, memory spaces for read-only data which might
    //! Be well-suited to storing merged group structs. This method returns the prefix required to
    //! Place arrays in these and their size in preferential order
    virtual MemorySpaces getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const override;

    virtual bool supportsNamespace() const override { return true; };

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    const cudaDeviceProp &getChosenCUDADevice() const{ return m_ChosenDevice; }
    int getChosenDeviceID() const{ return m_ChosenDeviceID; }
    int getRuntimeVersion() const{ return m_RuntimeVersion; }
    std::string getNVCCFlags() const;

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    template<typename T>
    void genMergedStructArrayPush(CodeStream &os, const std::vector<T> &groups, MemorySpaces &memorySpaces) const
    {
        // Loop through groups
        for(const auto &g : groups) {
            // Get size of group in bytes
            const size_t groupBytes = g.getStructArraySize(*this);

            // Loop through memory spaces
            bool memorySpaceFound = false;
            for(auto &m : memorySpaces) {
                // If there is space in this memory space for group
                if(m.second > groupBytes) {
                    // Implement merged group array in this memory space
                    os << m.first << " Merged" << T::name << "Group" << g.getIndex() << " d_merged" << T::name << "Group" << g.getIndex() << "[" << g.getGroups().size() << "];" << std::endl;

                    // Set flag
                    memorySpaceFound = true;

                    // Subtract
                    m.second -= groupBytes;

                    // Stop searching
                    break;
                }
            }

            assert(memorySpaceFound);

            // Write function to update
            os << "void pushMerged" << T::name << "Group" << g.getIndex() << "ToDevice(unsigned int idx, ";
            g.generateStructFieldArgumentDefinitions(os, *this);
            os << ")";
            {
                CodeStream::Scope b(os);

                // Loop through sorted fields and build struct on the stack
                os << "Merged" << T::name << "Group" << g.getIndex() << " group = {";
                const auto sortedFields = g.getSortedFields(*this);
                for(const auto &f : sortedFields) {
                    os << std::get<1>(f) << ", ";
                }
                os << "};" << std::endl;

                // Push to device
                os << "CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_merged" << T::name << "Group" << g.getIndex() << ", &group, ";
                os << "sizeof(Merged" << T::name << "Group" << g.getIndex() << "), idx * sizeof(Merged" << T::name << "Group" << g.getIndex() << ")));" << std::endl;
            }
        }
    }


    //! Get the safe amount of constant cache we can use
    size_t getChosenDeviceSafeConstMemBytes() const
    {
        return m_ChosenDevice.totalConstMem - getPreferences<Preferences>().constantCacheOverhead;
    }

    void genCurrentSpikePush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize, bool spikeEvent) const;
    void genCurrentSpikePull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize, bool spikeEvent) const;

    void genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreadsX, size_t batchSize, size_t numBlockThreadsY = 1) const;

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const int m_ChosenDeviceID;
    cudaDeviceProp m_ChosenDevice;
    int m_RuntimeVersion;
};
}   // CUDA
}   // CodeGenerator

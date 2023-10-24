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

// Forward declarations
namespace filesystem
{
    class path;
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::CUDA::DeviceSelectMethod
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::CUDA
{
//! Methods for selecting CUDA device
enum class DeviceSelect
{
    OPTIMAL,        //!< Pick optimal device based on how well kernels can be simultaneously simulated and occupancy
    MOST_MEMORY,    //!< Pick device with most global memory
    MANUAL,         //!< Use device specified by user
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

    //! Generate corresponding NCCL batch reductions
    bool enableNCCLReductions = false;
    
    //! How to select GPU device
    DeviceSelect deviceSelectMethod = DeviceSelect::OPTIMAL;

    //! If device select method is set to DeviceSelect::MANUAL, id of device to use
    unsigned int manualDeviceID = 0;

    //! How to select CUDA blocksize
    BlockSizeSelect blockSizeSelectMethod = BlockSizeSelect::OCCUPANCY;

    //! If block size select method is set to BlockSizeSelect::MANUAL, block size to use for each kernel
    KernelBlockSize manualBlockSizes;

    //! How much constant cache is already used and therefore can't be used by GeNN?
    /*! Each of the four modules which includes CUDA headers(neuronUpdate, synapseUpdate, custom update, init and runner)
        Takes 72 bytes of constant memory for a lookup table used by cuRAND. If your application requires
        additional constant cache, increase this */
    size_t constantCacheOverhead = 72 * 5;

    //! NVCC compiler options for all GPU code
    std::string userNvccFlags = "";

    void updateHash(boost::uuids::detail::sha1 &hash) const
    {
        // Superclass 
        PreferencesBase::updateHash(hash);

        // **NOTE** showPtxInfo, generateLineInfo and userNvccFlags only affect makefiles/msbuild 
        // **NOTE** block size optimization is also not relevant, the chosen block size is hashed in the backend
        // **NOTE** while device selection is also not relevant as the chosen device is hashed in the backend, DeviceSelect::MANUAL_OVERRIDE is used in the backend

        //! Update hash with preferences
        Utils::updateHash(deviceSelectMethod, hash);
        Utils::updateHash(constantCacheOverhead, hash);
        Utils::updateHash(enableNCCLReductions, hash);
    }
};


//--------------------------------------------------------------------------
// CodeGenerator::CUDA::Pointer
//--------------------------------------------------------------------------
class BACKEND_EXPORT Array : public ArrayBase
{
public:
    Array(const Type::ResolvedType &type, size_t count, 
          VarLocation location);
    virtual ~Array();
    
    //------------------------------------------------------------------------
    // ArrayBase virtuals
    //------------------------------------------------------------------------
    //! Allocate array
    virtual void allocate(size_t count) final;

    //! Free array
    virtual void free() final;

    //! Copy array to device
    virtual void pushToDevice() final;

    //! Copy array from device
    virtual void pullFromDevice() final;

    //! Serialise backend-specific device object to bytes
    virtual void serialiseDeviceObject(std::vector<std::byte> &bytes, bool pointerToPointer) const final;

    //! Serialise backend-specific host object to bytes
    virtual void serialiseHostObject(std::vector<std::byte>&, bool) const
    {
        throw std::runtime_error("CUDA arrays have no host objects");
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    std::byte *getDevicePointer() const{ return m_DevicePointer; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::byte *m_DevicePointer;
};

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::Backend
//--------------------------------------------------------------------------
class BACKEND_EXPORT Backend : public BackendSIMT
{
public:
    Backend(const KernelBlockSize &kernelBlockSizes, const Preferences &preferences, 
            int device, bool zeroCopy);

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendSIMT virtuals
    //--------------------------------------------------------------------------
    //! On some older devices, shared memory atomics are actually slower than global memory atomics so should be avoided
    virtual bool areSharedMemAtomicsSlow() const final;

    //! Get the prefix to use for shared memory variables
    virtual std::string getSharedPrefix() const final{ return "__shared__ "; }

    //! Get the ID of the current thread within the threadblock
    virtual std::string getThreadID(unsigned int axis = 0) const final;

    //! Get the ID of the current thread block
    virtual std::string getBlockID(unsigned int axis = 0) const final;

    //! Get the name of the count-leading-zeros function
    virtual std::string getCLZ() const final { return "__clz"; }

    //! Get name of atomic operation
    virtual std::string getAtomic(const Type::ResolvedType &type,
                                  AtomicOperation op = AtomicOperation::ADD, 
                                  AtomicMemSpace memSpace = AtomicMemSpace::GLOBAL) const final;

    //! Generate a shared memory barrier
    virtual void genSharedMemBarrier(CodeStream &os) const final;

    //! For SIMT backends which initialize RNGs on device, initialize population RNG with specified seed and sequence
    virtual void genPopulationRNGInit(CodeStream &os, const std::string &globalRNG, const std::string &seed, const std::string &sequence) const final;

    //! Generate a preamble to add substitution name for population RNG
    virtual std::string genPopulationRNGPreamble(CodeStream &os, const std::string &globalRNG) const final;

    //! If required, generate a postamble for population RNG
    /*! For example, in OpenCL, this is used to write local RNG state back to global memory*/
    virtual void genPopulationRNGPostamble(CodeStream &os, const std::string &globalRNG) const final;

    //! Generate code to skip ahead local copy of global RNG
    virtual std::string genGlobalRNGSkipAhead(CodeStream &os, const std::string &sequence) const final;

    //! Get type of population RNG
    virtual Type::ResolvedType getPopulationRNGType() const final;

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

    virtual void genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genFreeMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;

    //! Create backend-specific array object
    /*! \param type         data type of array
        \param count        number of elements in array, if non-zero will allocate
        \param location     location of array e.g. device-only*/
    virtual std::unique_ptr<ArrayBase> createArray(const Type::ResolvedType &type, size_t count, 
                                                   VarLocation location) const final;

    //! Create array of backend-specific population RNGs (if they are initialised on host this will occur here)
    /*! \param count        number of RNGs required*/
    virtual std::unique_ptr<ArrayBase> createPopulationRNG(size_t count) const final;

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

    //! Generate a single RNG instance
    /*! On single-threaded platforms this can be a standard RNG like M.T. but, on parallel platforms, it is likely to be a counter-based RNG */
    virtual void genGlobalDeviceRNG(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free) const final;

    virtual void genTimer(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free, 
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

    //! As well as host pointers, are device objects required?
    virtual bool isArrayDeviceObjectRequired() const final{ return true; }

    //! As well as host pointers, are additional host objects required e.g. for buffers in OpenCL?
    virtual bool isArrayHostObjectRequired() const final{ return false; }

    //! Different backends seed RNGs in different ways. Does this one initialise population RNGS on device?
    virtual bool isPopulationRNGInitialisedOnDevice() const final { return true; }

    //! Backends which support batch-parallelism might require an additional host reduction phase after reduction kernels
    virtual bool isHostReductionRequired() const final { return getPreferences<Preferences>().enableNCCLReductions; }

    //! How many bytes of memory does 'device' have
    virtual size_t getDeviceMemoryBytes() const final{ return m_ChosenDevice.totalGlobalMem; }

    //! Some backends will have additional small, fast, memory spaces for read-only data which might
    //! Be well-suited to storing merged group structs. This method returns the prefix required to
    //! Place arrays in these and their size in preferential order
    virtual MemorySpaces getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const final;

    //! Get hash digest of this backends identification and the preferences it has been configured with
    virtual boost::uuids::detail::sha1::digest_type getHashDigest() const final;

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
    std::string getNCCLReductionType(VarAccessMode mode) const;
    std::string getNCCLType(const Type::ResolvedType &type) const;
    
    void genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreadsX, size_t batchSize, size_t numBlockThreadsY = 1) const;

    template<typename T>
    void genMergedStructArrayPush(CodeStream &os, const std::vector<T> &groups) const
    {
        // Loop through groups
        for(const auto &g : groups) {
            // Check that a memory space has been assigned
            assert(!g.getMemorySpace().empty());

            // Implement merged group array in previously assigned memory space
            os << g.getMemorySpace() << " Merged" << T::name << "Group" << g.getIndex() << " d_merged" << T::name << "Group" << g.getIndex() << "[" << g.getGroups().size() << "];" << std::endl;

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

    template<typename G>
    void genNCCLReduction(EnvironmentExternalBase &env, G &cg) const
    {
        CodeStream::Scope b(env.getStream());
        env.getStream() << "// merged custom update host reduction group " << cg.getIndex() << std::endl;
        env.getStream() << "for(unsigned int g = 0; g < " << cg.getGroups().size() << "; g++)";
        {
            CodeStream::Scope b(env.getStream());

            // Get reference to group
            env.getStream() << "const auto *group = &merged" << G::name << "Group" << cg.getIndex() << "[g]; " << std::endl;
            EnvironmentGroupMergedField<G> groupEnv(env, cg);

            // Loop through variables
            const auto *cm = cg.getArchetype().getCustomUpdateModel();
            for(const auto &v : cm->getVars()) {
                // If variable is reduction target
                if(v.access & VarAccessModeAttribute::REDUCE) {
                    // Add pointer field
                    const auto resolvedType = v.type.resolve(cg.getTypeContext());
                    groupEnv.addField(resolvedType.createPointer(), "_" + v.name, v.name,
                                      [v](const auto &runtime, const auto &g, size_t) 
                                      { 
                                          return runtime.getArray(g, v.name);
                                      });
                    
                    // Add NCCL reduction
                    groupEnv.print("CHECK_NCCL_ERRORS(ncclAllReduce($(_" + v.name + "), $(_" + v.name + "), $(_size)");
                    groupEnv.printLine(", " + getNCCLType(resolvedType) + ", " + getNCCLReductionType(getVarAccessMode(v.access)) + ", ncclCommunicator, 0));");
                }
            }

            // Loop through variable references
            for(const auto &v : cm->getVarRefs()) {
                // If variable reference ios reduction target
                if(v.access & VarAccessModeAttribute::REDUCE) {
                    // Add pointer field
                    const auto resolvedType = v.type.resolve(cg.getTypeContext());
                    groupEnv.addField(resolvedType.createPointer(), "_" + v.name, v.name,
                                      [v](const auto &runtime, const auto &g, size_t) 
                                      { 
                                          const auto varRef = g.getVarReferences().at(v.name);
                                          return varRef.getTargetArray(runtime);
                                      });

                    // Add NCCL reduction
                    groupEnv.print("CHECK_NCCL_ERRORS(ncclAllReduce($(_" + v.name + "), $(_" + v.name + "), $(_size)");
                    groupEnv.printLine(", " + getNCCLType(v.type.resolve(cg.getTypeContext())) + ", " + getNCCLReductionType(v.access) + ", ncclCommunicator, 0));");
                }
            } 
        }
    }

    //! Get the safe amount of constant cache we can use
    size_t getChosenDeviceSafeConstMemBytes() const
    {
        return m_ChosenDevice.totalConstMem - getPreferences<Preferences>().constantCacheOverhead;
    }

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const int m_ChosenDeviceID;
    cudaDeviceProp m_ChosenDevice;
    int m_RuntimeVersion;
};
}   // GeNN::CUDA::CodeGenerator

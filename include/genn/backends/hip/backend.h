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

// **YUCK** disable the myriad of warning produced by HIP NVIDIA backend
#if defined(__HIP_PLATFORM_NVIDIA__) && defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    #pragma GCC diagnostic ignored "-Wmissing-field-initializers"
    #pragma GCC diagnostic ignored "-Wsign-compare"
    #pragma GCC diagnostic ignored "-Wreturn-local-addr" 
#endif

// HIP includes
#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_NVIDIA__) && defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

/*#if __has_include(<hip/nccl.h>)
    #include <nccl.h>

    #define NCCL_AVAILABLE
#endif*/

// GeNN includes
#include "backendExport.h"

// GeNN code generator includes
#include "code_generator/backendCUDAHIP.h"
#include "code_generator/codeStream.h"

// Forward declarations
namespace filesystem
{
    class path;
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::HIP::Preferences
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::HIP
{
//! Preferences for HIP backend
struct Preferences : public PreferencesCUDAHIP
{
    Preferences()
    {
        std::fill(manualBlockSizes.begin(), manualBlockSizes.end(), 32);
    }

    //! Should PTX assembler information be displayed for each CUDA kernel during compilation?
    //bool showPtxInfo = false;

    //! Should line info be included in resultant executable for debugging/profiling purposes?
    //bool generateLineInfo = false;

    //! How to select GPU device
    //DeviceSelect deviceSelectMethod = DeviceSelect::MANUAL;

    //! If device select method is set to DeviceSelect::MANUAL, id of device to use
    unsigned int manualDeviceID = 0;

    //! How to select CUDA blocksize
    //BlockSizeSelect blockSizeSelectMethod = BlockSizeSelect::OCCUPANCY;

    //! If block size select method is set to BlockSizeSelect::MANUAL, block size to use for each kernel
    KernelBlockSize manualBlockSizes;

    //! How much constant cache is already used and therefore can't be used by GeNN?
    /*! Each of the four modules which includes CUDA headers(neuronUpdate, synapseUpdate, custom update, init and runner)
        Takes 72 bytes of constant memory for a lookup table used by cuRAND. If your application requires
        additional constant cache, increase this */
    size_t constantCacheOverhead = 72 * 5;

    //! HIPCC compiler options for all GPU code
    std::string userHipccFlags = "";

    void updateHash(boost::uuids::detail::sha1 &hash) const
    {
        // Superclass
        PreferencesCUDAHIP::updateHash(hash);

        // **NOTE** showPtxInfo, generateLineInfo and userNvccFlags only affect makefiles/msbuild 
        // **NOTE** block size optimization is also not relevant, the chosen block size is hashed in the backend
        // **NOTE** while device selection is also not relevant as the chosen device is hashed in the backend, DeviceSelect::MANUAL_OVERRIDE is used in the backend

        //! Update hash with preferences
        //Utils::updateHash(deviceSelectMethod, hash);
        Utils::updateHash(constantCacheOverhead, hash);
    }
};

//--------------------------------------------------------------------------
// CodeGenerator::HIP::Backend
//--------------------------------------------------------------------------
class BACKEND_EXPORT State : public Runtime::StateBase
{
public:
    State(const Runtime::Runtime &base);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! To be called on one rank to generate ID before creating communicator
    void ncclGenerateUniqueID();

    //! Get pointer to unique ID
    unsigned char *ncclGetUniqueID();

    //! Get size of unique ID in bytes
    size_t ncclGetUniqueIDSize() const;

    //! Initialise communicator
    void ncclInitCommunicator(int rank, int numRanks);

private:
    //----------------------------------------------------------------------------
    // Type defines
    //----------------------------------------------------------------------------
    typedef void (*VoidFunction)(void);
    typedef unsigned char* (*BytePtrFunction)(void);
    typedef void (*NCCLInitCommunicatorFunction)(int, int);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    VoidFunction m_NCCLGenerateUniqueID;
    BytePtrFunction m_NCCLGetUniqueID;
    NCCLInitCommunicatorFunction m_NCCLInitCommunicator;
    const size_t *m_NCCLUniqueIDSize;
};

//--------------------------------------------------------------------------
// CodeGenerator::HIP::Backend
//--------------------------------------------------------------------------
class BACKEND_EXPORT Backend : public BackendCUDAHIP
{
public:
    Backend(const KernelBlockSize &kernelBlockSizes, const Preferences &preferences, 
            int device, bool zeroCopy);

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendSIMT virtuals
    //--------------------------------------------------------------------------
    //! On some older devices, shared memory atomics are actually slower than global memory atomics so should be avoided
    virtual bool areSharedMemAtomicsSlow() const final;

    //! How many 'lanes' does underlying hardware have?
    /*! This is typically used for warp-shuffle algorithms */
    virtual unsigned int getNumLanes() const final;

    //! Get name of atomic operation
    virtual std::string getAtomic(const Type::ResolvedType &type,
                                  AtomicOperation op = AtomicOperation::ADD, 
                                  AtomicMemSpace memSpace = AtomicMemSpace::GLOBAL) const final;

    //! For SIMT backends which initialize RNGs on device, initialize population RNG with specified seed and sequence
    virtual void genPopulationRNGInit(CodeStream &os, const std::string &globalRNG, const std::string &seed, const std::string &sequence) const final;

    //! Generate a preamble to add substitution name for population RNG
    virtual void buildPopulationRNGEnvironment(EnvironmentGroupMergedField<NeuronUpdateGroupMerged> &env) const final;

    //! Add $(_rng) to environment based on $(_rng_internal) field with any initialisers and destructors required
    virtual void buildPopulationRNGEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env) const final;

    //! Get type of population RNG
    virtual Type::ResolvedType getPopulationRNGType() const final;

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendBase virtuals
    //--------------------------------------------------------------------------
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;

    //! Create backend-specific runtime state object
    /*! \param runtime  runtime object */
    virtual std::unique_ptr<GeNN::Runtime::StateBase> createState(const Runtime::Runtime &runtime) const final;

    //! Create backend-specific array object
    /*! \param type         data type of array
        \param count        number of elements in array, if non-zero will allocate
        \param location     location of array e.g. device-only*/
    virtual std::unique_ptr<Runtime::ArrayBase> createArray(const Type::ResolvedType &type, size_t count, 
                                                            VarLocation location, bool uninitialized) const final;

    //! Generate code to allocate variable with a size known at runtime
    virtual void genLazyVariableDynamicAllocation(CodeStream &os, 
                                                  const Type::ResolvedType &type, const std::string &name, VarLocation loc, 
                                                  const std::string &countVarName) const final;
    
    virtual bool shouldUseNMakeBuildSystem() const final{ return true; }

    virtual void genMakefilePreamble(std::ostream &os) const final;
    virtual void genMakefileLinkRule(std::ostream &os) const final;
    virtual void genMakefileCompileRule(std::ostream &os) const final;
    
    virtual void genNMakefilePreamble(std::ostream &os) const final;
    virtual void genNMakefileLinkRule(std::ostream &os) const final;
    virtual void genNMakefileCompileRule(std::ostream &os) const final;

    virtual void genMSBuildConfigProperties(std::ostream &os) const final;
    virtual void genMSBuildImportProps(std::ostream &os) const final;
    virtual void genMSBuildItemDefinitions(std::ostream &os) const final;
    virtual void genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const final;
    virtual void genMSBuildImportTarget(std::ostream &os) const final;

    //! How many bytes of memory does 'device' have
    virtual size_t getDeviceMemoryBytes() const final{ return m_ChosenDevice.totalGlobalMem; }

    //! Get hash digest of this backends identification and the preferences it has been configured with
    virtual boost::uuids::detail::sha1::digest_type getHashDigest() const final;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    const hipDeviceProp_t &getChosenHIPDevice() const{ return m_ChosenDevice; }
    int getChosenDeviceID() const{ return m_ChosenDeviceID; }
    int getRuntimeVersion() const{ return m_RuntimeVersion; }
    std::string getHIPCCFlags() const;

protected:
    //--------------------------------------------------------------------------
    // BackendCUDAHIP virtuals
    //--------------------------------------------------------------------------
    //! Get the safe amount of constant cache we can use
    virtual size_t getChosenDeviceSafeConstMemBytes() const final
    {
        return m_ChosenDevice.totalConstMem - getPreferences<Preferences>().constantCacheOverhead;
    }

    //! Get internal type population RNG gets loaded into
    virtual Type::ResolvedType getPopulationRNGInternalType() const final;
    
    //! Get library of RNG functions to use
    virtual const EnvironmentLibrary::Library &getRNGFunctions(const Type::ResolvedType &precision) const final;

    //! Generate HIP/CUDA specific bits of definitions preamble
    virtual void genDefinitionsPreambleInternal(CodeStream &os, const ModelSpecMerged &modelMerged) const final;

    virtual void genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreadsX, size_t batchSize, size_t numBlockThreadsY = 1) const final;

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const int m_ChosenDeviceID;
    hipDeviceProp_t m_ChosenDevice;
    int m_RuntimeVersion;
};
}   // GeNN::CUDA::CodeGenerator

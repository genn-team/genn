#include "backend.h"

// Standard C++ includes
#include <algorithm>
#include <iterator>

// hipRAND includes
#include <hiprand/hiprand_kernel.h>

// GeNN includes
#include "gennUtils.h"
#include "logging.h"
#include "type.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/standardLibrary.h"

// CUDA backend includes
#include "utils.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const EnvironmentLibrary::Library floatRandomFunctions = {
    {"gennrand", {Type::ResolvedType::createFunction(Type::Uint32, {}), "hiprand(&$(_rng))"}},
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Float, {}), "hiprand_uniform(&$(_rng))"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Float, {}), "hiprand_normal(&$(_rng))"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Float, {}), "exponentialDistFloat(&$(_rng))"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Float, {Type::Float, Type::Float}), "hiprand_log_normal_float(&$(_rng), $(0), $(1))"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Float, {Type::Float}), "gammaDistFloat(&$(_rng), $(0))"}},
    {"gennrand_binomial", {Type::ResolvedType::createFunction(Type::Uint32, {Type::Uint32, Type::Float}), "binomialDistFloat(&$(_rng), $(0), $(1))"}},
};

const EnvironmentLibrary::Library doubleRandomFunctions = {
    {"gennrand", {Type::ResolvedType::createFunction(Type::Uint32, {}), "hiprand(&$(_rng))"}},
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Double, {}), "hiprand_uniform_double(&$(_rng))"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Double, {}), "hiprand_normal_double(&$(_rng))"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Double, {}), "exponentialDistDouble(&$(_rng))"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Double, {Type::Double, Type::Double}), "hiprand_log_normal_double(&$(_rng), $(0), $(1))"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Double, {Type::Double}), "gammaDistDouble(&$(_rng), $(0))"}},
    {"gennrand_binomial", {Type::ResolvedType::createFunction(Type::Uint32, {Type::Uint32, Type::Double}), "binomialDistDouble(&$(_rng), $(0), $(1))"}},
};

//--------------------------------------------------------------------------
// CUDADeviceType
//--------------------------------------------------------------------------
const Type::ResolvedType HIPRandState = Type::ResolvedType::createValue<hiprandState>("hiprandState", false, nullptr, true);
const Type::ResolvedType HIPRandStatePhilox43210 = Type::ResolvedType::createValue<hiprandStatePhilox4_32_10_t>("hiprandStatePhilox4_32_10_t", false, nullptr, true);

//--------------------------------------------------------------------------
// Array
//--------------------------------------------------------------------------
class Array : public Runtime::ArrayBase
{
public:
    Array(const Type::ResolvedType &type, size_t count, 
          VarLocation location, bool uninitialized)
    :   ArrayBase(type, count, location, uninitialized), m_DevicePointer(nullptr)
    {
        // Allocate if count is specified
        if(count > 0) {
            allocate(count);
        }
    }

    virtual ~Array()
    {
        if(getCount() > 0) {
            free();
        }
    }
    
    //------------------------------------------------------------------------
    // ArrayBase virtuals
    //------------------------------------------------------------------------
    //! Allocate array
    virtual void allocate(size_t count) final
    {
        // Set count
        setCount(count);

        // Malloc host pointer
        if(getLocation() & VarLocationAttribute::HOST) {
            const unsigned int flags = (getLocation() & VarLocationAttribute::ZERO_COPY) ? hipHostMallocMapped : hipHostMallocPortable;

            std::byte *hostPointer = nullptr;
            CHECK_HIP_ERRORS(hipHostMalloc(&hostPointer, getSizeBytes(), flags));
            setHostPointer(hostPointer);
        }

        // If variable is present on device at all
        if(getLocation() & VarLocationAttribute::DEVICE) {
            // Insert call to correct helper depending on whether variable should be allocated in zero-copy mode or not
            if(getLocation() & VarLocationAttribute::ZERO_COPY) {
                CHECK_HIP_ERRORS(hipHostGetDevicePointer(reinterpret_cast<void**>(&m_DevicePointer), getHostPointer(), 0));
            }
            else {
                CHECK_HIP_ERRORS(hipMalloc(&m_DevicePointer, getSizeBytes()));
            }
        }
    }

    //! Free array
    virtual void free() final
    {
        // **NOTE** because we pinned the variable we need to free it with cudaFreeHost rather than use free
        if(getLocation() & VarLocationAttribute::HOST) {
            CHECK_HIP_ERRORS(hipHostFree(getHostPointer()));
            setHostPointer(nullptr);
        }

        // If this variable wasn't allocated in zero-copy mode, free it
        if((getLocation() & VarLocationAttribute::DEVICE) && !(getLocation() & VarLocationAttribute::ZERO_COPY)) {
            CHECK_HIP_ERRORS(hipFree(getDevicePointer()));
            m_DevicePointer = nullptr;
        }

        // Zero count
        setCount(0);
    }
    //! Copy entire array to device
    virtual void pushToDevice() final
    {
        if(!(getLocation() & VarLocationAttribute::DEVICE) || !(getLocation() & VarLocationAttribute::HOST)) {
            throw std::runtime_error("Cannot push array that isn't present on host and device");
        }

        if(!(getLocation() & VarLocationAttribute::ZERO_COPY)) {
            CHECK_HIP_ERRORS(hipMemcpy(getDevicePointer(), getHostPointer(), getSizeBytes(), hipMemcpyHostToDevice));
        }
    }

    //! Copy entire array from device
    virtual void pullFromDevice() final
    {
        if(!(getLocation() & VarLocationAttribute::DEVICE) || !(getLocation() & VarLocationAttribute::HOST)) {
            throw std::runtime_error("Cannot pull array that isn't present on host and device");
        }

        if(!(getLocation() & VarLocationAttribute::ZERO_COPY)) {
            CHECK_HIP_ERRORS(hipMemcpy(getHostPointer(), getDevicePointer(), getSizeBytes(), hipMemcpyDeviceToHost));
        }
    }

    //! Copy a 1D slice of elements to device 
    /*! \param offset   Offset in elements to start copying from
        \param count    Number of elements to copy*/
    virtual void pushSlice1DToDevice(size_t offset, size_t count) final
    {
        if(!(getLocation() & VarLocationAttribute::DEVICE) || !(getLocation() & VarLocationAttribute::HOST)) {
            throw std::runtime_error("Cannot push array that isn't present on host and device");
        }

        if(!(getLocation() & VarLocationAttribute::ZERO_COPY)) {
            // If end of slice overflows array, give error
            if((offset + count) > getCount()) {
                throw std::runtime_error("Cannot pull slice that overflows array");
            }

            // Convert offset and count to bytes and copy
            const size_t offsetBytes = offset * getType().getValue().size;
            const size_t countBytes = count * getType().getValue().size;
            CHECK_HIP_ERRORS(hipMemcpy(getDevicePointer() + offsetBytes, getHostPointer() + offsetBytes, 
                                       countBytes, hipMemcpyHostToDevice));
        }
    }

    //! Copy a 1D slice of elements from device 
    /*! \param offset   Offset in elements to start copying from
        \param count    Number of elements to copy*/
    virtual void pullSlice1DFromDevice(size_t offset, size_t count) final
    {
        if(!(getLocation() & VarLocationAttribute::DEVICE) || !(getLocation() & VarLocationAttribute::HOST)) {
            throw std::runtime_error("Cannot pull array that isn't present on host and device");
        }

        if(!(getLocation() & VarLocationAttribute::ZERO_COPY)) {
            // If end of slice overflows array, give error
            if((offset + count) > getCount()) {
                throw std::runtime_error("Cannot pull slice that overflows array");
            }

            // Convert offset and count to bytes and copy
            const size_t offsetBytes = offset * getType().getValue().size;
            const size_t countBytes = count * getType().getValue().size;
            CHECK_HIP_ERRORS(hipMemcpy(getHostPointer() + offsetBytes, getDevicePointer() + offsetBytes, 
                                       countBytes, hipMemcpyDeviceToHost));
        }

    }

    //! Memset the host pointer
    virtual void memsetDeviceObject(int value) final
    {
        CHECK_HIP_ERRORS(hipMemset(m_DevicePointer, value, getSizeBytes()));
    }

    //! Serialise backend-specific device object to bytes
    virtual void serialiseDeviceObject(std::vector<std::byte> &bytes, bool pointerToPointer) const final
    {
        std::byte vBytes[sizeof(void*)];
        if(pointerToPointer) {
            std::byte* const *devicePointerPointer = &m_DevicePointer;
            std::memcpy(vBytes, &devicePointerPointer, sizeof(void*));
        }
        else {
            std::memcpy(vBytes, &m_DevicePointer, sizeof(void*));
        }
        std::copy(std::begin(vBytes), std::end(vBytes), std::back_inserter(bytes));
    }

    //! Serialise backend-specific host object to bytes
    virtual void serialiseHostObject(std::vector<std::byte>&, bool) const
    {
        throw std::runtime_error("HIP arrays have no host objects");
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
}   // Anonymous namespace


//--------------------------------------------------------------------------
// GeNN::CodeGenerator::HIP::State
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::HIP
{
State::State(const Runtime::Runtime &runtime)
{
    // Lookup NCCL symbols
    m_NCCLGenerateUniqueID = (VoidFunction)runtime.getSymbol("ncclGenerateUniqueID", true);
    m_NCCLGetUniqueID = (BytePtrFunction)runtime.getSymbol("ncclGetUniqueID", true);
    m_NCCLInitCommunicator = (NCCLInitCommunicatorFunction)runtime.getSymbol("ncclInitCommunicator", true);
    m_NCCLUniqueIDSize = (size_t*)runtime.getSymbol("ncclUniqueIDSize", true);
}
//--------------------------------------------------------------------------
void State::ncclGenerateUniqueID()
{
    if(m_NCCLGenerateUniqueID == nullptr) {
        throw std::runtime_error("Cannot generate NCCL unique ID - model may not have been built with NCCL support");
    }
    m_NCCLGenerateUniqueID();
}
//--------------------------------------------------------------------------
unsigned char *State::ncclGetUniqueID()
{ 
    if(m_NCCLGetUniqueID == nullptr) {
        throw std::runtime_error("Cannot get NCCL unique ID - model may not have been built with NCCL support");
    }
    return m_NCCLGetUniqueID();
}
//--------------------------------------------------------------------------
size_t State::ncclGetUniqueIDSize() const
{
    if(m_NCCLUniqueIDSize == nullptr) {
        throw std::runtime_error("Cannot get NCCL unique ID size - model may not have been built with NCCL support");
    }
    
    return *m_NCCLUniqueIDSize;
}
//--------------------------------------------------------------------------    
void State::ncclInitCommunicator(int rank, int numRanks)
{
     if(m_NCCLInitCommunicator == nullptr) {
        throw std::runtime_error("Cannot initialise NCCL communicator - model may not have been built with NCCL support");
    }
    m_NCCLInitCommunicator(rank, numRanks);
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::HIP::Backend
//--------------------------------------------------------------------------
Backend::Backend(const KernelBlockSize &kernelBlockSizes, const Preferences &preferences, 
                 int device, bool zeroCopy)
:   BackendCUDAHIP(kernelBlockSizes, preferences, "hip", "hiprand", "hccl"), 
    m_ChosenDeviceID(device)
{
    // Set device
    CHECK_HIP_ERRORS(hipSetDevice(device));

    // Get device properties
    CHECK_HIP_ERRORS(hipGetDeviceProperties(&m_ChosenDevice, device));

    // Get HIP runtime version
    CHECK_HIP_ERRORS(hipRuntimeGetVersion(&m_RuntimeVersion));

#ifdef _WIN32
    // If we're on Windows and NCCL is enabled, give error
    // **NOTE** There are several NCCL Windows ports e.g. https://github.com/MyCaffe/NCCL but we don't have access to any suitable systems to test
    if(getPreferences<Preferences>().enableNCCLReductions) {
        throw std::runtime_error("GeNN doesn't currently support NCCL on Windows");
    }
#endif

    // If the model requires zero-copy
    if(zeroCopy) {
        // If device doesn't support mapping host memory error
        if(!getChosenHIPDevice().canMapHostMemory) {
            throw std::runtime_error("Device does not support mapping CPU host memory!");
        }

        // Set map host device flag
        CHECK_HIP_ERRORS(hipSetDeviceFlags(hipDeviceMapHost));
    }
}
//--------------------------------------------------------------------------
bool Backend::areSharedMemAtomicsSlow() const
{
    return false;
}
//--------------------------------------------------------------------------
unsigned int Backend::getNumLanes() const
{
    return getChosenHIPDevice().warpSize;
}
//--------------------------------------------------------------------------
std::string Backend::getAtomic(const Type::ResolvedType &type, AtomicOperation op, AtomicMemSpace) const
{
    // If operation is an atomic add
    if(op == AtomicOperation::ADD) {
        return "atomicAdd";
    }
    // Otherwise, it's an atomic or
    else {
        assert(op == AtomicOperation::OR);
        assert(type == Type::Uint32 || type == Type::Int32);
        return "atomicOr";
    }
}
//--------------------------------------------------------------------------
std::unique_ptr<GeNN::Runtime::StateBase> Backend::createState(const Runtime::Runtime &runtime) const
{
    return std::make_unique<State>(runtime);
}
//--------------------------------------------------------------------------
std::unique_ptr<Runtime::ArrayBase> Backend::createArray(const Type::ResolvedType &type, size_t count, 
                                                         VarLocation location, bool uninitialized) const
{
    return std::make_unique<Array>(type, count, location, uninitialized);
}
//--------------------------------------------------------------------------
void Backend::genLazyVariableDynamicAllocation(CodeStream &os, const Type::ResolvedType &type, const std::string &name,
                                               VarLocation loc, const std::string &countVarName) const
{
    const auto &underlyingType = type.isPointer() ? *type.getPointer().valueType : type;
    const std::string hostPointer = type.isPointer() ? ("*$(_" + name + ")") : ("$(_" + name + ")");
    const std::string hostPointerToPointer = type.isPointer() ? ("$(_" + name + ")") : ("&$(_" + name + ")");
    const std::string devicePointerToPointer = type.isPointer() ? ("$(_d_" + name + ")") : ("&$(_d_" + name + ")");

    if(loc & VarLocationAttribute::HOST) {
        const char *flags = (loc & VarLocationAttribute::ZERO_COPY) ? "HostMallocMapped" : "HostMallocPortable";
        os << "CHECK_RUNTIME_ERRORS(hipHostMalloc(" << hostPointerToPointer << ", " << countVarName << " * sizeof(" << underlyingType.getName() << "), hip" << flags << "));" << std::endl;
    }

    // If variable is present on device at all
    if(loc & VarLocationAttribute::DEVICE) {
        if(loc & VarLocationAttribute::ZERO_COPY) {
            os << "CHECK_RUNTIME_ERRORS(hipHostGetDevicePointer((void**)" << devicePointerToPointer << ", (void*)" << hostPointer << ", 0));" << std::endl;
        }
        else {
            os << "CHECK_RUNTIME_ERRORS(hipMalloc(" << devicePointerToPointer << ", " << countVarName << " * sizeof(" << underlyingType.getName() << ")));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void Backend::genMakefilePreamble(std::ostream &os) const
{
    const std::string architecture = "sm_" + std::to_string(getChosenHIPDevice().major) + std::to_string(getChosenHIPDevice().minor);
    std::string linkFlags = "--shared -arch " + architecture;

    // If NCCL reductions are enabled, link NCCL
    if(getPreferences<Preferences>().enableNCCLReductions) {
        linkFlags += " -lnccl";
    }
    // Write variables to preamble
    os << "HIP_PATH ?=/opt/rocm" << std::endl;
    os << "HIPCC := $(HIP_PATH)/bin/hipcc" << std::endl;
    os << "HIPCCFLAGS := " << getHIPCCFlags() << std::endl;
    os << "LINKFLAGS := " << linkFlags << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileLinkRule(std::ostream &os) const
{
    os << "\t@$(HIPCC) $(LINKFLAGS) -o $@ $(OBJECTS)" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileCompileRule(std::ostream &os) const
{
    // Add one rule to generate dependency files from cc files
    os << "%.d: %.cc" << std::endl;
    os << "\t@$(HIPCC) -M $(HIPCCFLAGS) $< 1> $@" << std::endl;
    os << std::endl;

    // Add another to build object files from cc files
    os << "%.o: %.cc %.d" << std::endl;
    os << "\t@$(HIPCC) -dc $(HIPCCFLAGS) $<" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genNMakefilePreamble(std::ostream &os) const
{
    const std::string architecture = "sm_" + std::to_string(getChosenHIPDevice().major) + std::to_string(getChosenHIPDevice().minor);
    std::string linkFlags = "--shared -arch " + architecture;
    
    // Write variables to preamble
    os << "HIPCC = \"$(HIP_PATH)/bin/hipcc.exe\"" << std::endl;
    os << "HIPCCFLAGS = " << getNVCCFlags() << std::endl;
    os << "LINKFLAGS = " << linkFlags << std::endl;

    // Prefer explicit CUDA_LIBRARY_PATH; otherwise fall back to typical CUDA_PATH layouts on Windows.
    // Final fallback leaves LIBCUDA empty so the toolchain can use LIB environment paths.
    os << "!IF DEFINED(CUDA_LIBRARY_PATH)" << std::endl;
    os << "LIBCUDA=/LIBPATH:\"$(CUDA_LIBRARY_PATH)\"" << std::endl;

    // Fall back to CUDA_PATH default \"lib\\x64\" (common on Windows)
    os << "!ELSEIF EXIST(\"$(CUDA_PATH)\\lib\\x64\\cudart.lib\")" << std::endl;
    os << "LIBCUDA=/LIBPATH:\"$(CUDA_PATH)\\lib\\x64\"" << std::endl;

    // Older CUDA installs may only have \"lib\" (no x64 subdir)
    os << "!ELSEIF EXIST(\"$(CUDA_PATH)\\lib\\cudart.lib\")" << std::endl;
    os << "LIBCUDA=/LIBPATH:\"$(CUDA_PATH)\\lib\"" << std::endl;

    // No explicit CUDA library path found – rely on LIB from toolchain/environment
    os << "!ELSE" << std::endl;
    os << "LIBCUDA=" << std::endl;
    os << "!ENDIF" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genNMakefileLinkRule(std::ostream &os) const
{
    // Use Visual C++ linker to link objects with device object code
    // **YUCK** there should be some way to do this with $(CXX) /LINK
    os << "runner.dll: $(OBJECTS) runner_dlink.obj" << std::endl;
    os << "\t@link.exe /OUT:runner.dll $(LIBCUDA) cudart.lib cuda.lib cudadevrt.lib /DLL $(OBJECTS) runner_dlink.obj\n";
    os << std::endl;

    // Use HIPCC to link the device code
    os << "runner_dlink.obj: $(OBJECTS)" << std::endl;
    os << "\t@$(HIPCC) $(LINKFLAGS) -dlink $(OBJECTS) -o runner_dlink.obj" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genNMakefileCompileRule(std::ostream &os) const
{
    // Add rule to build object files from cc files
    os << ".cc.obj:" << std::endl;
    os << "\t@$(HIPCC) -dc $(HIPCCFLAGS) $< -o $@" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildConfigProperties(std::ostream&) const
{
    throw std::runtime_error("The HIP backend does not currently support the MSBuild build system");
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportProps(std::ostream&) const
{
    throw std::runtime_error("The HIP backend does not currently support the MSBuild build system");
}
//--------------------------------------------------------------------------
void Backend::genMSBuildItemDefinitions(std::ostream&) const
{
    throw std::runtime_error("The HIP backend does not currently support the MSBuild build system");
}
//--------------------------------------------------------------------------
void Backend::genMSBuildCompileModule(const std::string&, std::ostream&) const
{
    throw std::runtime_error("The HIP backend does not currently support the MSBuild build system");
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportTarget(std::ostream&) const
{
    throw std::runtime_error("The HIP backend does not currently support the MSBuild build system");
}
//--------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Backend::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash was name of backend
    Utils::updateHash("HIP", hash);

    // Update hash with chosen device ID and kernel block sizes
    Utils::updateHash(m_ChosenDeviceID, hash);
    Utils::updateHash(getKernelBlockSize(), hash);

    // Update hash with preferences
    getPreferences<Preferences>().updateHash(hash);

    return hash.get_digest();
}
//--------------------------------------------------------------------------
std::string Backend::getHIPCCFlags() const
{
#ifdef __HIP_PLATFORM_NVIDIA__
    // **NOTE** now we don't include runner.cc when building standalone modules we get loads of warnings about
    // How you hide device compiler warnings is totally non-documented but https://stackoverflow.com/a/17095910/1476754
    // holds the answer! For future reference --display_error_number option can be used to get warning ids to use in --diag-supress
    // HOWEVER, on CUDA 7.5 and 8.0 this causes a fatal error and, as no warnings are shown when --diag-suppress is removed,
    // presumably this is because this warning simply wasn't implemented until CUDA 9
    const std::string architecture = "sm_" + std::to_string(getChosenHIPDevice().major) + std::to_string(getChosenHIPDevice().minor);
    std::string nvccFlags = "-fPIC -arch " + architecture + " -I\"$(HIP_PATH)/include\"";
#ifndef _WIN32
    nvccFlags += " -std=c++11";
#endif
    nvccFlags += " -Xcudafe \"--diag_suppress=extern_entity_treated_as_static\"";
    
    //nvccFlags += " " + getPreferences<Preferences>().userNvccFlags;
    if(getPreferences().optimizeCode) {
        nvccFlags += " -O3 -use_fast_math";
    }
    if(getPreferences().debugCode) {
        nvccFlags += " -O0 -g -G";
    }
    //if(getPreferences<Preferences>().showPtxInfo) {
    //    nvccFlags += " -Xptxas \"-v\"";
    //}
    //if(getPreferences<Preferences>().generateLineInfo) {
    //    nvccFlags += " --generate-line-info";
    //}

    return nvccFlags;
#else
    assert(false);
    return "";
#endif
}
//--------------------------------------------------------------------------
Type::ResolvedType Backend::getPopulationRNGInternalType() const
{
    return HIPRandState;
}
//--------------------------------------------------------------------------
const EnvironmentLibrary::Library &Backend::getRNGFunctions(const Type::ResolvedType &precision) const
{
    if(precision == Type::Float) {
        return floatRandomFunctions;
    }
    else {
        assert(precision == Type::Double);
        return doubleRandomFunctions;
    }
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsPreambleInternal(CodeStream &os, const ModelSpecMerged &) const
{
    os << "// HIP includes" << std::endl;
    // **YUCK** disable the myriad of warning produced by HIP NVIDIA backend
#if defined(__HIP_PLATFORM_NVIDIA__) && defined(__GNUC__)
    os << "#pragma GCC diagnostic push" << std::endl;
    os << "#pragma GCC diagnostic ignored \"-Wdeprecated-declarations\"" << std::endl;
    os << "#pragma GCC diagnostic ignored \"-Wmissing-field-initializers\"" << std::endl;
    os << "#pragma GCC diagnostic ignored \"-Wsign-compare\"" << std::endl;
    os << "#pragma GCC diagnostic ignored \"-Wreturn-local-addr\"" << std::endl;
#endif
    os <<"#include <hip/hip_runtime.h>" << std::endl;
    os <<"#include <hip/hip_fp16.h>" << std::endl;
    os << "#include <hiprand/hiprand_kernel.h>" << std::endl;
#if defined(__HIP_PLATFORM_NVIDIA__) && defined(__GNUC__)
    os << "#pragma GCC diagnostic pop" << std::endl;
#endif
    // If NCCL is enabled
    if(getPreferences<Preferences>().enableNCCLReductions) {
        // Include RCCL header
        os << "#include <rccl.h>" << std::endl;
        os << std::endl;

        os << std::endl;
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Helper macro for error-checking RCCL calls" << std::endl;
        os << "#define CHECK_CCL_ERRORS(call) {\\" << std::endl;
        os << "    rcclResult_t error = call;\\" << std::endl;
        os << "    if (error != rcclSuccess) {\\" << std::endl;
        os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": rccl error \" + std::to_string(error) + \": \" + rcclGetErrorString(error));\\" << std::endl;
        os << "    }\\" << std::endl;
        os << "}" << std::endl;

        // Define NCCL ID and communicator
        os << "extern rcclUniqueId ncclID;" << std::endl;
        os << "extern rcclComm_t ncclCommunicator;" << std::endl;

        // Export ncclGetUniqueId function
        os << "extern \"C\" {" << std::endl;
        os << "EXPORT_VAR const size_t ncclUniqueIDSize;" << std::endl;
        os << "EXPORT_FUNC void ncclGenerateUniqueID();" << std::endl;
        os << "EXPORT_FUNC void ncclInitCommunicator(int rank, int numRanks);" << std::endl;
        os << "EXPORT_FUNC unsigned char *ncclGetUniqueID();" << std::endl;
        os << "}" << std::endl;
    }

    os << std::endl;
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper macro for error-checking CUDA calls" << std::endl;
    os << "#define CHECK_RUNTIME_ERRORS(call) {\\" << std::endl;
    os << "    hipError_t error = call;\\" << std::endl;
    os << "    if (error != hipSuccess) {\\" << std::endl;
    os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": HIP error \" + std::to_string(error) + \": \" + hipGetErrorString(error));\\" << std::endl;
    os << "    }\\" << std::endl;
    os << "}" << std::endl;
    os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreadsX, size_t batchSize, size_t numBlockThreadsY) const
{
    // Calculate grid size
    const size_t gridSize = ceilDivide(numThreadsX, getKernelBlockSize(kernel));
    assert(gridSize < (size_t)getChosenHIPDevice().maxGridSize[0]);
    assert(numBlockThreadsY < (size_t)getChosenHIPDevice().maxThreadsDim[0]);

    os << "const dim3 threads(" << getKernelBlockSize(kernel) << ", " << numBlockThreadsY << ");" << std::endl;
    if(numBlockThreadsY > 1) {
        assert(batchSize < (size_t)getChosenHIPDevice().maxThreadsDim[2]);
        os << "const dim3 grid(" << gridSize << ", 1, " << batchSize << ");" << std::endl;
    }
    else {
        assert(batchSize < (size_t)getChosenHIPDevice().maxThreadsDim[1]);
        os << "const dim3 grid(" << gridSize << ", " << batchSize << ");" << std::endl;
    }
}
}   // namespace GeNN::CodeGenerator::CUDA

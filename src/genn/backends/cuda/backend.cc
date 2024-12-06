#include "backend.h"

// Standard C++ includes
#include <algorithm>
#include <iterator>

// CUDA includes
#include <curand_kernel.h>

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
    {"gennrand", {Type::ResolvedType::createFunction(Type::Uint32, {}), "curand(&$(_rng))"}},
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Float, {}), "curand_uniform(&$(_rng))"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Float, {}), "curand_normal(&$(_rng))"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Float, {}), "exponentialDistFloat(&$(_rng))"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Float, {Type::Float, Type::Float}), "curand_log_normal_float(&$(_rng), $(0), $(1))"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Float, {Type::Float}), "gammaDistFloat(&$(_rng), $(0))"}},
    {"gennrand_binomial", {Type::ResolvedType::createFunction(Type::Uint32, {Type::Uint32, Type::Float}), "binomialDistFloat(&$(_rng), $(0), $(1))"}},
};

const EnvironmentLibrary::Library doubleRandomFunctions = {
    {"gennrand", {Type::ResolvedType::createFunction(Type::Uint32, {}), "curand(&$(_rng))"}},
    {"gennrand_uniform", {Type::ResolvedType::createFunction(Type::Double, {}), "curand_uniform_double(&$(_rng))"}},
    {"gennrand_normal", {Type::ResolvedType::createFunction(Type::Double, {}), "curand_normal_double(&$(_rng))"}},
    {"gennrand_exponential", {Type::ResolvedType::createFunction(Type::Double, {}), "exponentialDistDouble(&$(_rng))"}},
    {"gennrand_log_normal", {Type::ResolvedType::createFunction(Type::Double, {Type::Double, Type::Double}), "curand_log_normal_double(&$(_rng), $(0), $(1))"}},
    {"gennrand_gamma", {Type::ResolvedType::createFunction(Type::Double, {Type::Double}), "gammaDistDouble(&$(_rng), $(0))"}},
    {"gennrand_binomial", {Type::ResolvedType::createFunction(Type::Uint32, {Type::Uint32, Type::Double}), "binomialDistDouble(&$(_rng), $(0), $(1))"}},
};

//--------------------------------------------------------------------------
// CUDADeviceType
//--------------------------------------------------------------------------
const Type::ResolvedType CURandState = Type::ResolvedType::createValue<curandState>("curandState", false, nullptr, true);
const Type::ResolvedType CURandStatePhilox43210 = Type::ResolvedType::createValue<curandStatePhilox4_32_10_t>("curandStatePhilox4_32_10_t", false, nullptr, true);

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
            const unsigned int flags = (getLocation() & VarLocationAttribute::ZERO_COPY) ? cudaHostAllocMapped : cudaHostAllocPortable;

            std::byte *hostPointer = nullptr;
            CHECK_CUDA_ERRORS(cudaHostAlloc(&hostPointer, getSizeBytes(), flags));
            setHostPointer(hostPointer);
        }

        // If variable is present on device at all
        if(getLocation() & VarLocationAttribute::DEVICE) {
            // Insert call to correct helper depending on whether variable should be allocated in zero-copy mode or not
            if(getLocation() & VarLocationAttribute::ZERO_COPY) {
                CHECK_CUDA_ERRORS(cudaHostGetDevicePointer(&m_DevicePointer, getHostPointer(), 0));
            }
            else {
                CHECK_CUDA_ERRORS(cudaMalloc(&m_DevicePointer, getSizeBytes()));
            }
        }
    }

    //! Free array
    virtual void free() final
    {
        // **NOTE** because we pinned the variable we need to free it with cudaFreeHost rather than use free
        if(getLocation() & VarLocationAttribute::HOST) {
            CHECK_CUDA_ERRORS(cudaFreeHost(getHostPointer()));
            setHostPointer(nullptr);
        }

        // If this variable wasn't allocated in zero-copy mode, free it
        if((getLocation() & VarLocationAttribute::DEVICE) && !(getLocation() & VarLocationAttribute::ZERO_COPY)) {
            CHECK_CUDA_ERRORS(cudaFree(getDevicePointer()));
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
            CHECK_CUDA_ERRORS(cudaMemcpy(getDevicePointer(), getHostPointer(), getSizeBytes(), cudaMemcpyHostToDevice));
        }
    }

    //! Copy entire array from device
    virtual void pullFromDevice() final
    {
        if(!(getLocation() & VarLocationAttribute::DEVICE) || !(getLocation() & VarLocationAttribute::HOST)) {
            throw std::runtime_error("Cannot pull array that isn't present on host and device");
        }

        if(!(getLocation() & VarLocationAttribute::ZERO_COPY)) {
            CHECK_CUDA_ERRORS(cudaMemcpy(getHostPointer(), getDevicePointer(), getSizeBytes(), cudaMemcpyDeviceToHost));
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
            CHECK_CUDA_ERRORS(cudaMemcpy(getDevicePointer() + offsetBytes, getHostPointer() + offsetBytes, 
                                         countBytes, cudaMemcpyHostToDevice));
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
            CHECK_CUDA_ERRORS(cudaMemcpy(getHostPointer() + offsetBytes, getDevicePointer() + offsetBytes, 
                                         countBytes, cudaMemcpyDeviceToHost));
        }

    }

    //! Memset the host pointer
    virtual void memsetDeviceObject(int value) final
    {
        CHECK_CUDA_ERRORS(cudaMemset(m_DevicePointer, value, getSizeBytes()));
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
}   // Anonymous namespace


//--------------------------------------------------------------------------
// GeNN::CodeGenerator::CUDA::State
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::CUDA
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
// GeNN::CodeGenerator::CUDA::Backend
//--------------------------------------------------------------------------
Backend::Backend(const KernelBlockSize &kernelBlockSizes, const Preferences &preferences, 
                 int device, bool zeroCopy)
:   BackendCUDAHIP(kernelBlockSizes, preferences, "cuda", "nccl"), m_ChosenDeviceID(device)
{
    // Set device
    CHECK_CUDA_ERRORS(cudaSetDevice(device));

    // Get device properties
    CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&m_ChosenDevice, device));

    // Get CUDA runtime version
    cudaRuntimeGetVersion(&m_RuntimeVersion);

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
        if(!getChosenCUDADevice().canMapHostMemory) {
            throw std::runtime_error("Device does not support mapping CPU host memory!");
        }

        // Set map host device flag
        CHECK_CUDA_ERRORS(cudaSetDeviceFlags(cudaDeviceMapHost));
    }
}
//--------------------------------------------------------------------------
bool Backend::areSharedMemAtomicsSlow() const
{
    // If device is older than Maxwell, we shouldn't use shared memory as atomics are emulated
    // and actually slower than global memory (see https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
    return (getChosenCUDADevice().major < 5);
}
//--------------------------------------------------------------------------
unsigned int Backend::getNumLanes() const
{
    return 32;
}
//--------------------------------------------------------------------------
std::string Backend::getAtomic(const Type::ResolvedType &type, AtomicOperation op, AtomicMemSpace) const
{
    // If operation is an atomic add
    if(op == AtomicOperation::ADD) {
        if(((getChosenCUDADevice().major < 2) && (type == Type::Float))
           || (((getChosenCUDADevice().major < 6) || (getRuntimeVersion() < 8000)) && (type == Type::Double)))
        {
            return "atomicAddSW";
        }

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
Type::ResolvedType Backend::getPopulationRNGType() const
{
    return CURandState;
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
std::unique_ptr<Runtime::ArrayBase> Backend::createPopulationRNG(size_t count) const
{
    return createArray(CURandState, count, VarLocation::DEVICE, false);
}
//--------------------------------------------------------------------------
void Backend::genMakefilePreamble(std::ostream &os) const
{
    const std::string architecture = "sm_" + std::to_string(getChosenCUDADevice().major) + std::to_string(getChosenCUDADevice().minor);
    std::string linkFlags = "--shared -arch " + architecture;
    
    // If NCCL reductions are enabled, link NCCL
    if(getPreferences<Preferences>().enableNCCLReductions) {
        linkFlags += " -lnccl";
    }
    // Write variables to preamble
    os << "CUDA_PATH ?=/usr/local/cuda" << std::endl;
    os << "NVCC := $(CUDA_PATH)/bin/nvcc" << std::endl;
    os << "NVCCFLAGS := " << getNVCCFlags() << std::endl;
    os << "LINKFLAGS := " << linkFlags << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileLinkRule(std::ostream &os) const
{
    os << "\t@$(NVCC) $(LINKFLAGS) -o $@ $(OBJECTS)" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMakefileCompileRule(std::ostream &os) const
{
    // Add one rule to generate dependency files from cc files
    os << "%.d: %.cc" << std::endl;
    os << "\t@$(NVCC) -M $(NVCCFLAGS) $< 1> $@" << std::endl;
    os << std::endl;

    // Add another to build object files from cc files
    os << "%.o: %.cc %.d" << std::endl;
    os << "\t@$(NVCC) -dc $(NVCCFLAGS) $<" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildConfigProperties(std::ostream &os) const
{
    // Add property to extract CUDA path
    os << "\t\t<!-- **HACK** determine the installed CUDA version by regexing CUDA path -->" << std::endl;
    os << "\t\t<CudaVersion>$([System.Text.RegularExpressions.Regex]::Match($(CUDA_PATH), \"\\\\v([0-9.]+)$\").Groups[1].Value)</CudaVersion>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportProps(std::ostream &os) const
{
    // Import CUDA props file
    os << "\t<ImportGroup Label=\"ExtensionSettings\">" << std::endl;
    os << "\t\t<Import Project=\"$(CUDA_PATH)\\extras\\visual_studio_integration\\MSBuildExtensions\\CUDA $(CudaVersion).props\" />" << std::endl;
    os << "\t</ImportGroup>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildItemDefinitions(std::ostream &os) const
{
    // Add item definition for host compilation
    os << "\t\t<ClCompile>" << std::endl;
    os << "\t\t\t<WarningLevel>Level3</WarningLevel>" << std::endl;
    os << "\t\t\t<Optimization Condition=\"'$(Configuration)'=='Release'\">MaxSpeed</Optimization>" << std::endl;
    os << "\t\t\t<Optimization Condition=\"'$(Configuration)'=='Debug'\">Disabled</Optimization>" << std::endl;
    os << "\t\t\t<FunctionLevelLinking Condition=\"'$(Configuration)'=='Release'\">true</FunctionLevelLinking>" << std::endl;
    os << "\t\t\t<IntrinsicFunctions Condition=\"'$(Configuration)'=='Release'\">true</IntrinsicFunctions>" << std::endl;
    os << "\t\t\t<PreprocessorDefinitions Condition=\"'$(Configuration)'=='Release'\">WIN32;WIN64;NDEBUG;_CONSOLE;BUILDING_GENERATED_CODE;%(PreprocessorDefinitions)</PreprocessorDefinitions>" << std::endl;
    os << "\t\t\t<PreprocessorDefinitions Condition=\"'$(Configuration)'=='Debug'\">WIN32;WIN64;_DEBUG;_CONSOLE;BUILDING_GENERATED_CODE;%(PreprocessorDefinitions)</PreprocessorDefinitions>" << std::endl;
    os << "\t\t\t<MultiProcessorCompilation>true</MultiProcessorCompilation>" << std::endl;
    os << "\t\t</ClCompile>" << std::endl;

    // Add item definition for linking
    os << "\t\t<Link>" << std::endl;
    os << "\t\t\t<GenerateDebugInformation>true</GenerateDebugInformation>" << std::endl;
    os << "\t\t\t<EnableCOMDATFolding Condition=\"'$(Configuration)'=='Release'\">true</EnableCOMDATFolding>" << std::endl;
    os << "\t\t\t<OptimizeReferences Condition=\"'$(Configuration)'=='Release'\">true</OptimizeReferences>" << std::endl;
    os << "\t\t\t<SubSystem>Console</SubSystem>" << std::endl;
    os << "\t\t\t<AdditionalDependencies>cudart_static.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>" << std::endl;
    os << "\t\t</Link>" << std::endl;

    // Add item definition for CUDA compilation
    // **YUCK** the CUDA Visual Studio plugin build system demands that you specify both a virtual an actual architecture 
    // (which NVCC itself doesn't require). While, in general, actual architectures are usable as virtual architectures, 
    // there is no compute_21 so we need to replace that with compute_20
    const std::string architecture = std::to_string(getChosenCUDADevice().major) + std::to_string(getChosenCUDADevice().minor);
    const std::string virtualArchitecture = (architecture == "21") ? "20" : architecture;
    os << "\t\t<CudaCompile>" << std::endl;
    os << "\t\t\t<TargetMachinePlatform>64</TargetMachinePlatform>" << std::endl;
    os << "\t\t\t<GenerateRelocatableDeviceCode>true</GenerateRelocatableDeviceCode>" << std::endl;
    os << "\t\t\t<CodeGeneration>compute_" << virtualArchitecture <<",sm_" << architecture << "</CodeGeneration>" << std::endl;
    os << "\t\t\t<FastMath>" << (getPreferences().optimizeCode ? "true" : "false") << "</FastMath>" << std::endl;
    os << "\t\t\t<GenerateLineInfo>" << (getPreferences<Preferences>().generateLineInfo ? "true" : "false") << "</GenerateLineInfo>" << std::endl;
    os << "\t\t</CudaCompile>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const
{
    os << "\t\t<CudaCompile Include=\"" << moduleName << ".cc\" >" << std::endl;
    // **YUCK** for some reasons you can't call .Contains on %(BaseCommandLineTemplate) directly
    // Solution suggested by https://stackoverflow.com/questions/9512577/using-item-functions-on-metadata-values
    os << "\t\t\t<AdditionalOptions Condition=\" !$([System.String]::new('%(BaseCommandLineTemplate)').Contains('-x cu')) \">-x cu %(AdditionalOptions)</AdditionalOptions>" << std::endl;
    os << "\t\t</CudaCompile>" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportTarget(std::ostream &os) const
{
    os << "\t<ImportGroup Label=\"ExtensionTargets\">" << std::endl;
    os << "\t\t<Import Project=\"$(CUDA_PATH)\\extras\\visual_studio_integration\\MSBuildExtensions\\CUDA $(CudaVersion).targets\" />" << std::endl;
    os << "\t</ImportGroup>" << std::endl;
}
//--------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Backend::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash was name of backend
    Utils::updateHash("CUDA", hash);
    
    // Update hash with chosen device ID and kernel block sizes
    Utils::updateHash(m_ChosenDeviceID, hash);
    Utils::updateHash(getKernelBlockSize(), hash);

    // Update hash with preferences
    getPreferences<Preferences>().updateHash(hash);

    return hash.get_digest();
}
//--------------------------------------------------------------------------
std::string Backend::getNVCCFlags() const
{
    // **NOTE** now we don't include runner.cc when building standalone modules we get loads of warnings about
    // How you hide device compiler warnings is totally non-documented but https://stackoverflow.com/a/17095910/1476754
    // holds the answer! For future reference --display_error_number option can be used to get warning ids to use in --diag-supress
    // HOWEVER, on CUDA 7.5 and 8.0 this causes a fatal error and, as no warnings are shown when --diag-suppress is removed,
    // presumably this is because this warning simply wasn't implemented until CUDA 9
    const std::string architecture = "sm_" + std::to_string(getChosenCUDADevice().major) + std::to_string(getChosenCUDADevice().minor);
    std::string nvccFlags = "-x cu -arch " + architecture;
#ifndef _WIN32
    nvccFlags += " -std=c++11 --compiler-options \"-fPIC -Wno-return-type-c-linkage\"";
#endif
    if(m_RuntimeVersion >= 9020) {
        nvccFlags += " -Xcudafe \"--diag_suppress=extern_entity_treated_as_static\"";
    }

    nvccFlags += " " + getPreferences<Preferences>().userNvccFlags;
    if(getPreferences().optimizeCode) {
        nvccFlags += " -O3 -use_fast_math";
    }
    if(getPreferences().debugCode) {
        nvccFlags += " -O0 -g -G";
    }
    if(getPreferences<Preferences>().showPtxInfo) {
        nvccFlags += " -Xptxas \"-v\"";
    }
    if(getPreferences<Preferences>().generateLineInfo) {
        nvccFlags += " --generate-line-info";
    }

    return nvccFlags;
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
    os << "// CUDA includes" << std::endl;
    os << "#include <curand_kernel.h>" << std::endl;
    if(getRuntimeVersion() >= 9000) {
        os <<"#include <cuda_fp16.h>" << std::endl;
    }

    // If NCCL is enabled
    if(getPreferences<Preferences>().enableNCCLReductions) {
        // Include NCCL header
        os << "#include <nccl.h>" << std::endl;
        os << std::endl;

        os << std::endl;
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Helper macro for error-checking NCCL calls" << std::endl;
        os << "#define CHECK_CCL_ERRORS(call) {\\" << std::endl;
        os << "    ncclResult_t error = call;\\" << std::endl;
        os << "    if (error != ncclSuccess) {\\" << std::endl;
        os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": nccl error \" + std::to_string(error) + \": \" + ncclGetErrorString(error));\\" << std::endl;
        os << "    }\\" << std::endl;
        os << "}" << std::endl;

        // Define NCCL ID and communicator
        os << "extern ncclUniqueId ncclID;" << std::endl;
        os << "extern ncclComm_t ncclCommunicator;" << std::endl;

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
    os << "    cudaError_t error = call;\\" << std::endl;
    os << "    if (error != cudaSuccess) {\\" << std::endl;
    os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": cuda error \" + std::to_string(error) + \": \" + cudaGetErrorString(error));\\" << std::endl;
    os << "    }\\" << std::endl;
    os << "}" << std::endl;
    os << std::endl;


    // If device is older than SM 6 or we're using a version of CUDA older than 8
    if ((getChosenCUDADevice().major < 6) || (getRuntimeVersion() < 8000)){
        os << "// software version of atomic add for double precision" << std::endl;
        os << "__device__ inline double atomicAddSW(double* address, double val)";
        {
            CodeStream::Scope b(os);
            os << "unsigned long long int* address_as_ull = (unsigned long long int*)address;" << std::endl;
            os << "unsigned long long int old = *address_as_ull, assumed;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "assumed = old;" << std::endl;
                os << "old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));" << std::endl;
            }
            os << "while (assumed != old);" << std::endl;
            os << "return __longlong_as_double(old);" << std::endl;
        }
        os << std::endl;
    }

    // If we're using a CUDA device with SM < 2
    if (getChosenCUDADevice().major < 2) {
        os << "// software version of atomic add for single precision float" << std::endl;
        os << "__device__ inline float atomicAddSW(float* address, float val)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "int* address_as_ull = (int*)address;" << std::endl;
            os << "int old = *address_as_ull, assumed;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "assumed = old;" << std::endl;
                os << "old = atomicCAS(address_as_ull, assumed, __float_as_int(val + __int_as_float(assumed)));" << std::endl;
            }
            os << "while (assumed != old);" << std::endl;
            os << "return __int_as_float(old);" << std::endl;
        }
        os << std::endl;
    }
}
//--------------------------------------------------------------------------
void Backend::genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreadsX, size_t batchSize, size_t numBlockThreadsY) const
{
    // Calculate grid size
    const size_t gridSize = ceilDivide(numThreadsX, getKernelBlockSize(kernel));
    assert(gridSize < (size_t)getChosenCUDADevice().maxGridSize[0]);
    assert(numBlockThreadsY < (size_t)getChosenCUDADevice().maxThreadsDim[0]);

    os << "const dim3 threads(" << getKernelBlockSize(kernel) << ", " << numBlockThreadsY << ");" << std::endl;
    if(numBlockThreadsY > 1) {
        assert(batchSize < (size_t)getChosenCUDADevice().maxThreadsDim[2]);
        os << "const dim3 grid(" << gridSize << ", 1, " << batchSize << ");" << std::endl;
    }
    else {
        assert(batchSize < (size_t)getChosenCUDADevice().maxThreadsDim[1]);
        os << "const dim3 grid(" << gridSize << ", " << batchSize << ");" << std::endl;
    }
}
}   // namespace GeNN::CodeGenerator::CUDA

/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *__doc_CodeGenerator_CUDA_Array = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Array_Array = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Array_allocate = R"doc(Allocate array)doc";

static const char *__doc_CodeGenerator_CUDA_Array_free = R"doc(Free array)doc";

static const char *__doc_CodeGenerator_CUDA_Array_getDevicePointer =
R"doc(Get device pointer
Returns the pointer to the device memory)doc";

static const char *__doc_CodeGenerator_CUDA_Array_m_DevicePointer = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Array_memsetDeviceObject = R"doc(Memset the host pointer)doc";

static const char *__doc_CodeGenerator_CUDA_Array_pullFromDevice = R"doc(Copy entire array from device)doc";

static const char *__doc_CodeGenerator_CUDA_Array_pullSlice1DFromDevice =
R"doc(Copy a 1D slice of elements from device


$Parameter ``offset``:

   Offset in elements to start copying from


$Parameter ``count``:

    Number of elements to copy)doc";

static const char *__doc_CodeGenerator_CUDA_Array_pushSlice1DToDevice =
R"doc(Copy a 1D slice of elements to device


$Parameter ``offset``:

   Offset in elements to start copying from


$Parameter ``count``:

    Number of elements to copy)doc";

static const char *__doc_CodeGenerator_CUDA_Array_pushToDevice = R"doc(Copy entire array to device)doc";

static const char *__doc_CodeGenerator_CUDA_Array_serialiseDeviceObject = R"doc(Serialise backend-specific device object to bytes)doc";

static const char *__doc_CodeGenerator_CUDA_Array_serialiseHostObject = R"doc(Serialise backend-specific host object to bytes)doc";

static const char *__doc_CodeGenerator_CUDA_Backend = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_Backend = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_areSharedMemAtomicsSlow = R"doc(On some older devices, shared memory atomics are actually slower than global memory atomics so should be avoided)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_createArray =
R"doc(Create backend-specific array object


$Parameter ``type``:

         data type of array


$Parameter ``count``:

        number of elements in array, if non-zero will allocate


$Parameter ``location``:

     location of array e.g. device-only)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_createState =
R"doc(Create backend-specific runtime state object


$Parameter ``runtime``:

  runtime object)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genDefinitionsPreambleInternal = R"doc(Generate HIP/CUDA specific bits of definitions preamble)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genKernelDimensions = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genLazyVariableDynamicAllocation = R"doc(Generate code to allocate variable with a size known at runtime)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genMSBuildCompileModule = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genMSBuildConfigProperties = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genMSBuildImportProps = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genMSBuildImportTarget = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genMSBuildItemDefinitions = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genMakefileCompileRule = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genMakefileLinkRule = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genMakefilePreamble = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genNMakefileCompileRule = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genNMakefileLinkRule = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genNMakefilePreamble = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genPopProfilerRange = R"doc(Generate code to pop the current profiler range marker)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_genPushProfilerRange = R"doc(Generate code to push a profiler range marker)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getAtomic = R"doc(Get name of atomic operation)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getChosenCUDADevice = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getChosenDeviceID = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getChosenDeviceSafeConstMemBytes = R"doc(Get the safe amount of constant cache we can use)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getDeviceMemoryBytes = R"doc(How many bytes of memory does 'device' have)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getHashDigest = R"doc(Get hash digest of this backends identification and the preferences it has been configured with)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getNVCCFlags = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getNumLanes =
R"doc(How many 'lanes' does underlying hardware have?
This is typically used for warp-shuffle algorithms)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getPopulationRNGInternalType = R"doc(Get internal type population RNG gets loaded into)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getRNGFunctions = R"doc(Get library of RNG functions to use)doc";

static const char *__doc_CodeGenerator_CUDA_Backend_getRuntimeVersion = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_m_ChosenDevice = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_m_ChosenDeviceID = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_m_RuntimeVersion = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Backend_shouldUseNMakeBuildSystem = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_BlockSizeSelect = R"doc(Methods for selecting CUDA kernel block size)doc";

static const char *__doc_CodeGenerator_CUDA_BlockSizeSelect_MANUAL = R"doc(Use block sizes specified by user)doc";

static const char *__doc_CodeGenerator_CUDA_BlockSizeSelect_OCCUPANCY = R"doc(Pick optimal blocksize for each kernel based on occupancy)doc";

static const char *__doc_CodeGenerator_CUDA_DeviceSelect = R"doc(Methods for selecting CUDA device)doc";

static const char *__doc_CodeGenerator_CUDA_DeviceSelect_MANUAL = R"doc(Use device specified by user)doc";

static const char *__doc_CodeGenerator_CUDA_DeviceSelect_MOST_MEMORY = R"doc(Pick device with most global memory)doc";

static const char *__doc_CodeGenerator_CUDA_DeviceSelect_OPTIMAL = R"doc(Pick optimal device based on how well kernels can be simultaneously simulated and occupancy)doc";

static const char *__doc_CodeGenerator_CUDA_Optimiser_createBackend = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Preferences = R"doc(Preferences for CUDA backend)doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_Preferences = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_blockSizeSelectMethod = R"doc(How to select CUDA blocksize)doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_constantCacheOverhead =
R"doc(How much constant cache is already used and therefore can't be used by GeNN?
Each of the four modules which includes CUDA headers(neuronUpdate, synapseUpdate, custom update, init and runner)
Takes 72 bytes of constant memory for a lookup table used by cuRAND. If your application requires
additional constant cache, increase this)doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_deviceSelectMethod = R"doc(How to select GPU device)doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_enableNVTX = R"doc(Should NVTX markers be inserted to make profiling easier?)doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_generateLineInfo = R"doc(Should line info be included in resultant executable for debugging/profiling purposes?)doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_manualBlockSizes = R"doc(If block size select method is set to BlockSizeSelect::MANUAL, block size to use for each kernel)doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_manualDeviceID = R"doc(If device select method is set to DeviceSelect::MANUAL, id of device to use)doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_showPtxInfo = R"doc(Should PTX assembler information be displayed for each CUDA kernel during compilation?)doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_updateHash = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_Preferences_userNvccFlags = R"doc(NVCC compiler options for all GPU code)doc";

static const char *__doc_CodeGenerator_CUDA_State = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_State_State = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_State_m_NCCLGenerateUniqueID = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_State_m_NCCLGetUniqueID = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_State_m_NCCLInitCommunicator = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_State_m_NCCLUniqueIDSize = R"doc()doc";

static const char *__doc_CodeGenerator_CUDA_State_ncclGenerateUniqueID = R"doc(To be called on one rank to generate ID before creating communicator)doc";

static const char *__doc_CodeGenerator_CUDA_State_ncclGetUniqueID = R"doc(Get pointer to unique ID)doc";

static const char *__doc_CodeGenerator_CUDA_State_ncclGetUniqueIDSize = R"doc(Get size of unique ID in bytes)doc";

static const char *__doc_CodeGenerator_CUDA_State_ncclInitCommunicator = R"doc(Initialise communicator)doc";

static const char *__doc_ModelSpecInternal = R"doc()doc";

static const char *__doc_filesystem_path =
R"doc(Simple class for manipulating paths on Linux/Windows/Mac OS

This class is just a temporary workaround to avoid the heavy boost
dependency until boost::filesystem is integrated into the standard template
library at some point in the future.)doc";

static const char *__doc_plog_IAppender = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif


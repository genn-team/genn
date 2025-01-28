#include "optimiser.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <tuple>

// Standard C includes
#include <cstdlib>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>

// PLOG includes
#include <plog/Log.h>

// Filesystem includes
#include "path.h"

// GeNN includes
#include "logging.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/generateModules.h"
#include "code_generator/generateRunner.h"
#include "code_generator/modelSpecMerged.h"

// CUDA backend includes
#include "utils.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace CUDA;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
//! Map of kernel IDs to flag indicating whether all blocks can be run concurrently and occupancy value
typedef std::map<unsigned int, std::pair<bool, size_t>> KernelOptimisationOutput;

//! Pointer to ModelSpecMerged member function for getting archetype hash digest
typedef boost::uuids::detail::sha1::digest_type (ModelSpecMerged::*GetArchetypeHashDigestFn)(void) const;

//! Module structure containing name, function to get hash and vector of kernels
struct Module
{
    std::string name;
    GetArchetypeHashDigestFn getArchetypeHashDigest;
    std::vector<Kernel> kernels;
};

//! Table of module names to functions to get their archetype hash digests and the kernel IDs they might contain
const std::vector<Module> modules = {
    {"customUpdate",    &ModelSpecMerged::getCustomUpdateArchetypeHashDigest,   {KernelCustomUpdate, KernelCustomTransposeUpdate}},
    {"init",            &ModelSpecMerged::getInitArchetypeHashDigest,           {KernelInitialize, KernelInitializeSparse}},
    {"neuronUpdate",    &ModelSpecMerged::getNeuronUpdateArchetypeHashDigest,   {KernelNeuronSpikeQueueUpdate, KernelNeuronPrevSpikeTimeUpdate, KernelNeuronUpdate}},
    {"synapseUpdate",   &ModelSpecMerged::getSynapseUpdateArchetypeHashDigest,  {KernelSynapseDendriticDelayUpdate, KernelPresynapticUpdate, KernelPostsynapticUpdate, KernelSynapseDynamicsUpdate}}
};

bool getKernelResourceUsage(CUmodule module, const std::string &kernelName, int &sharedMemBytes, int &numRegisters)
{
    // If function is found
    CUfunction kern;
    CUresult res = cuModuleGetFunction(&kern, module, kernelName.c_str());
    if(res == CUDA_SUCCESS) {
        LOGD_BACKEND << "\tKernel '" << kernelName << "' found";

        // Read function's shared memory size and register count and add blank entry to map of kernels to optimise
        CHECK_CU_ERRORS(cuFuncGetAttribute(&sharedMemBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kern));
        CHECK_CU_ERRORS(cuFuncGetAttribute(&numRegisters, CU_FUNC_ATTRIBUTE_NUM_REGS, kern));

        LOGD_BACKEND << "\t\tShared memory bytes:" << sharedMemBytes;
        LOGD_BACKEND << "\t\tNum registers:" << numRegisters;
        return true;
    }
    else {
        return false;
    }
}
//--------------------------------------------------------------------------
void getDeviceArchitectureProperties(const cudaDeviceProp &deviceProps, size_t &warpAllocGran, size_t &regAllocGran,
                                     size_t &smemAllocGran, size_t &maxBlocksPerSM)
{
    if(deviceProps.major == 1) {
        smemAllocGran = 512;
        warpAllocGran = 2;
        regAllocGran = (deviceProps.minor < 2) ? 256 : 512;
        maxBlocksPerSM = 8;
    }
    else if(deviceProps.major == 2) {
        smemAllocGran = 128;
        warpAllocGran = 2;
        regAllocGran = 64;
        maxBlocksPerSM = 8;
    }
    else if(deviceProps.major == 3) {
        smemAllocGran = 256;
        warpAllocGran = 4;
        regAllocGran = 256;
        maxBlocksPerSM = 16;
    }
    else if(deviceProps.major == 5) {
        smemAllocGran = 256;
        warpAllocGran = 4;
        regAllocGran = 256;
        maxBlocksPerSM = 32;
    }
    else if(deviceProps.major == 6) {
        smemAllocGran = 256;
        warpAllocGran = (deviceProps.minor == 0) ? 2 : 4;
        regAllocGran = 256;
        maxBlocksPerSM = 32;
    }
    else if(deviceProps.major == 7) {
        smemAllocGran = 256;
        warpAllocGran = 4;
        regAllocGran = 256;
        maxBlocksPerSM = (deviceProps.minor == 5) ? 16 : 32;
    }
    else if (deviceProps.major == 8) {
        smemAllocGran = 128;
        warpAllocGran = 4;
        regAllocGran = 256;

        if (deviceProps.minor == 0) {
            maxBlocksPerSM = 32;
        }
        else if (deviceProps.minor == 9) {
            maxBlocksPerSM = 24;
        }
        else {
            maxBlocksPerSM = 16;
        }
    }
    else {
        smemAllocGran = 128;
        warpAllocGran = 4;
        regAllocGran = 256;
        maxBlocksPerSM = 32;
        if(deviceProps.minor != 0) {
            LOGW_BACKEND << "Unsupported CUDA device version: 9." << deviceProps.minor;
            LOGW_BACKEND << "This is a bug! Please report it at https://github.com/genn-team/genn.";
            LOGW_BACKEND << "Falling back to SM 9.0 parameters.";
        }
        if(deviceProps.major > 9) {
            LOGW_BACKEND << "Unsupported CUDA device major version: " << deviceProps.major;
            LOGW_BACKEND << "This is a bug! Please report it at https://github.com/genn-team/genn.";
            LOGW_BACKEND << "Falling back to next latest SM version parameters.";
        }
    }
}
//--------------------------------------------------------------------------
template<typename G, typename C>
void addGroupSizes(const std::vector<G> &mergedGroups, Kernel kernel, std::vector<size_t> (&groupSizes)[KernelMax],
                   C getGroupSizeFn)
{
    // Loop through merged groups
    for(const auto &mg : mergedGroups) {
        // Loop through groups
        for(const auto &g : mg.getGroups()) {
            groupSizes[kernel].push_back(getGroupSizeFn(g.get()));
        }
    }
}
//--------------------------------------------------------------------------
template<typename G>
void addNumGroups(const std::vector<G> &mergedGroups, Kernel kernel, std::vector<size_t> (&groupSizes)[KernelMax])
{
    // Loop through merged groups
    for(const auto &mg : mergedGroups) {
        groupSizes[kernel].push_back(mg.getGroups().size());
    }
}
//--------------------------------------------------------------------------
void calcGroupSizes(const CUDA::Backend &backend, const ModelSpecMerged &modelMerged,
                    std::vector<size_t> (&groupSizes)[KernelMax])
{
    const auto &model = modelMerged.getModel();

    // Add neurons
    addGroupSizes(modelMerged.getMergedNeuronUpdateGroups(), KernelNeuronUpdate, groupSizes,
                  [&backend, &model](const auto &g){ return model.getBatchSize() * backend.getPaddedNeuronUpdateThreads(g, model.getTypeContext()); });
    addGroupSizes(modelMerged.getMergedNeuronInitGroups(), KernelInitialize, groupSizes,
                  [](const auto &g){ return g.getNumNeurons(); });
    addGroupSizes(modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups(), KernelNeuronPrevSpikeTimeUpdate, groupSizes,
                  [&model](const auto &g){ return model.getBatchSize() * g.getNumNeurons(); });
    addNumGroups(modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), KernelNeuronSpikeQueueUpdate, groupSizes);
    
    // Add custom updates
    addGroupSizes(modelMerged.getMergedCustomUpdateGroups(), KernelCustomUpdate, groupSizes,
                  [&backend, &model](const auto &g){ return backend.getPaddedNumCustomUpdateThreads(g, model.getBatchSize()); });
    addGroupSizes(modelMerged.getMergedCustomUpdateInitGroups(), KernelInitialize, groupSizes,
                  [](const auto &g){ return g.getNumNeurons(); });

    // Add custom WU updates
    addGroupSizes(modelMerged.getMergedCustomUpdateWUGroups(), KernelCustomUpdate, groupSizes,
                  [&backend, &model](const auto &g){ return backend.getPaddedNumCustomUpdateWUThreads(g, model.getBatchSize()); });
    addGroupSizes(modelMerged.getMergedCustomUpdateTransposeWUGroups(), KernelCustomTransposeUpdate, groupSizes,
                  [&backend, &model](const auto &g){ return backend.getPaddedNumCustomUpdateTransposeWUThreads(g, model.getBatchSize()); });
    addGroupSizes(modelMerged.getMergedCustomWUUpdateInitGroups(), KernelInitialize, groupSizes,
                  [&backend](const auto &g){ return backend.getNumInitThreads(g); });
    addGroupSizes(modelMerged.getMergedCustomWUUpdateSparseInitGroups(), KernelInitializeSparse, groupSizes,
                  [&backend](const auto &g){ return g.getSynapseGroup()->getMaxConnections(); });
    
    // Add custom connectivity updates
    addGroupSizes(modelMerged.getMergedCustomConnectivityUpdateGroups(), KernelCustomUpdate, groupSizes,
                  [](const auto &g){ return g.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(); });
    addGroupSizes(modelMerged.getMergedCustomConnectivityUpdatePreInitGroups(), KernelInitialize, groupSizes,
                  [](const auto &g){ return g.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(); });
    addGroupSizes(modelMerged.getMergedCustomConnectivityUpdatePostInitGroups(), KernelInitialize, groupSizes,
                  [](const auto &g){ return g.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(); });
    addGroupSizes(modelMerged.getMergedCustomConnectivityUpdateSparseInitGroups(), KernelInitializeSparse, groupSizes,
                  [](const auto &g){ return g.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons() * g.getSynapseGroup()->getMaxConnections(); });

    // Add synapse groups
    addGroupSizes(modelMerged.getMergedPresynapticUpdateGroups(), KernelPresynapticUpdate, groupSizes,
                  [&backend, &model](const auto &g){ return model.getBatchSize() * backend.getNumPresynapticUpdateThreads(g, model.getTypeContext()); });
    addGroupSizes(modelMerged.getMergedPostsynapticUpdateGroups(), KernelPostsynapticUpdate, groupSizes,
                  [&backend, &model](const auto &g){ return model.getBatchSize() * backend.getNumPostsynapticUpdateThreads(g); });
    addGroupSizes(modelMerged.getMergedSynapseDynamicsGroups(), KernelSynapseDynamicsUpdate, groupSizes,
                  [&backend, &model](const auto &g){ return model.getBatchSize() * backend.getNumSynapseDynamicsThreads(g); });
    addGroupSizes(modelMerged.getMergedSynapseInitGroups(), KernelInitialize, groupSizes,
                  [&backend, &model](const auto &g){ return backend.getNumInitThreads(g); });
    addGroupSizes(modelMerged.getMergedSynapseSparseInitGroups(), KernelInitializeSparse, groupSizes,
                  [](const auto &g){ return g.getSrcNeuronGroup()->getNumNeurons(); });
    addNumGroups(modelMerged.getMergedSynapseDendriticDelayUpdateGroups(), KernelSynapseDendriticDelayUpdate, groupSizes);
}
//--------------------------------------------------------------------------
void analyseModule(const Module &module, unsigned int r, CUcontext context, 
                   boost::uuids::detail::sha1::digest_type hashDigest, const filesystem::path &outputPath, const filesystem::path &nvccPath, const Backend &backend,
                   const std::set<std::string> &customUpdateKernels, const std::set<std::string> &customTransposeUpdateKernels, 
                   int (&krnlSharedSizeBytes)[2][KernelMax], int (&krnlNumRegs)[2][KernelMax], KernelOptimisationOutput &kernelsToOptimise, std::mutex &kernelsToOptimiseMutex)
{
    // Build source and module paths from module name
    const std::string sourcePath = (outputPath / (module.name + "CUDAOptim.cc")).str();
    const std::string moduleSHAPath = (outputPath / (module.name + "CUDA" + std::to_string(r) + ".sha")).str();

    LOGD_BACKEND << "\tModule " << module.name;
    try {
        // Open SHA file
        std::ifstream is(moduleSHAPath);

        // Throw exceptions in case of all errors
        is.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);

        // Read previous hash as hash
        boost::uuids::detail::sha1::digest_type previousHashDigest;
        is >> std::hex;
        for(auto &d : previousHashDigest) {
            is >> d;
        }

        // If hash matches
        if(previousHashDigest == hashDigest) {
            // Loop through kernels in module
            is >> std::dec;
            for(Kernel k : module.kernels) {
                // Read shared memory size and number of registers
                is >> krnlSharedSizeBytes[r][k] >> krnlNumRegs[r][k];

                // If this kernel requires any registers (and hence exists), add to map of kernels to optimier
                if(krnlNumRegs[r][k] > 0) {
                    std::lock_guard<std::mutex> l(kernelsToOptimiseMutex);
                    kernelsToOptimise.emplace(std::piecewise_construct,
                                              std::forward_as_tuple(k),
                                              std::forward_as_tuple(false, 0));
                }
            }

            // Remove tempory source file
            if(std::remove(sourcePath.c_str())) {
                LOGW_BACKEND << "Cannot remove dry-run source file";
            }
            LOGD_BACKEND << "\tModule unchanged - re-using shared memory and register usage";
            return;
        }
        // Otherwise, module needs analysing
        else {
            LOGD_BACKEND << "\tModule changed - re-analysing";
        }
    }
    catch(const std::ios_base::failure&) {
        LOGD_BACKEND << "\tUnable to read previous hash - re-analysing";
    }

    // Set context for this thread
    cuCtxSetCurrent(context);

#ifdef _WIN32
    // **YUCK** extra outer quotes required to workaround gross windowsness https://stackoverflow.com/questions/9964865/c-system-not-working-when-there-are-spaces-in-two-different-parameters
    const std::string nvccCommand = "\"\"" + nvccPath.str() + "\" -cubin " + backend.getNVCCFlags() + " -o \"" + sourcePath + ".cubin\" \"" + sourcePath + "\"\"";
#else
    const std::string nvccCommand = "\"" + nvccPath.str() + "\" -cubin " + backend.getNVCCFlags() + " -o \"" + sourcePath + ".cubin\" \"" + sourcePath + "\"";
#endif

    if(system(nvccCommand.c_str()) != 0) {
        throw std::runtime_error("optimizeBlockSize: NVCC failed");
    }

    // Load compiled module
    CUmodule loadedModule;
    CHECK_CU_ERRORS(cuModuleLoad(&loadedModule, (sourcePath + ".cubin").c_str()));

    // Loop through kernels that might be in this module
    for(Kernel k : module.kernels) {
        // If this kernel is a custom update
        // **YUCK** this mechanism is really not very nice but to fix it properly would require
        // replacing the block sizes std::array with a std::map to handle different custom update kernels
        //  which would break backward compatibility. For now just use worst case to pick block sizes
        if(k == KernelCustomUpdate || k == KernelCustomTransposeUpdate) {
            // Loop through all kernels of this type
            const auto &kernels = (k == KernelCustomUpdate) ? customUpdateKernels : customTransposeUpdateKernels;
            for(const std::string &c : kernels) {
                // If kernel is found, update maximum shared memory size and register count
                int sharedSizeBytes = 0;
                int numRegisters = 0;
                if(getKernelResourceUsage(loadedModule, Backend::KernelNames[k] + c, sharedSizeBytes, numRegisters)) {
                    krnlSharedSizeBytes[r][k] = std::max(krnlSharedSizeBytes[r][k], sharedSizeBytes);
                    krnlNumRegs[r][k] = std::max(krnlNumRegs[r][k], numRegisters);
                }
            }

            // If any kernels were found, add this type of custom update kernel to map
            if(krnlSharedSizeBytes[r][k] > 0 || krnlNumRegs[r][k] > 0) {
                std::lock_guard<std::mutex> g(kernelsToOptimiseMutex);
                kernelsToOptimise.emplace(std::piecewise_construct,
                                          std::forward_as_tuple(k),
                                          std::forward_as_tuple(false, 0));
            }
        }
        // Otherwise, if kernel is found, add to map of kernels to optimise
        else if(getKernelResourceUsage(loadedModule, Backend::KernelNames[k], krnlSharedSizeBytes[r][k], krnlNumRegs[r][k])) {
            std::lock_guard<std::mutex> g(kernelsToOptimiseMutex);
            kernelsToOptimise.emplace(std::piecewise_construct,
                                      std::forward_as_tuple(k),
                                      std::forward_as_tuple(false, 0));
        }
    }

    // Unload module
    CHECK_CU_ERRORS(cuModuleUnload(loadedModule));

    // Open sha file
    std::ofstream os(moduleSHAPath);

    // Write digest as hex with each word seperated by a space
    os << std::hex;
    for(const auto d : hashDigest) {
        os << d << " ";
    }
    os << std::endl;

    // Loop through kernels in this module and write shared memory size and number of registers
    os << std::dec;
    for(Kernel k : module.kernels) {
        os << krnlSharedSizeBytes[r][k] << " " << krnlNumRegs[r][k] << std::endl;
    }

    // Remove tempory cubin file
    if(std::remove((sourcePath + ".cubin").c_str())) {
        LOGW_BACKEND << "Cannot remove dry-run cubin file";
    }

    // Remove tempory source file
    if(std::remove(sourcePath.c_str())) {
        LOGW_BACKEND << "Cannot remove dry-run source file";
    }
}
//--------------------------------------------------------------------------
KernelOptimisationOutput optimizeBlockSize(int deviceID, const cudaDeviceProp &deviceProps, const ModelSpecInternal &model,
                                           KernelBlockSize &blockSize, const Preferences &preferences,
                                           const filesystem::path &outputPath)
{
    // Create directory for generated code
    filesystem::create_directory(outputPath);

    // Select device
    cudaSetDevice(deviceID);

    // Get names of custom update kernels
    const std::set<std::string> customUpdateKernels = model.getCustomUpdateGroupNames(false, true);
    const std::set<std::string> customTransposeUpdateKernels = model.getCustomUpdateGroupNames(true, false);

    // Create CUDA drive API device and context for accessing kernel attributes
    CUdevice cuDevice;
    CUcontext cuContext;
    CHECK_CU_ERRORS(cuDeviceGet(&cuDevice, deviceID));
    CHECK_CU_ERRORS(cuCtxCreate(&cuContext, 0, cuDevice));

    // Get CUDA_PATH environment variable
    // **NOTE** adding CUDA_PATH/bin to path is a REQUIRED post-installation action when installing CUDA so this shouldn't be required
    filesystem::path nvccPath;
    if(const char *cudaPath = std::getenv("CUDA_PATH")) {
        // Build path to NVCC using this
#ifdef _WIN32
        nvccPath = filesystem::path(cudaPath) / "bin" / "nvcc.exe";
#else
        nvccPath = filesystem::path(cudaPath) / "bin" / "nvcc";
#endif
    }
    else {
        throw std::runtime_error("CUDA_PATH environment variable not set - ");
    }

    // Arrays of kernel attributes gathered across block sizes
    int krnlSharedSizeBytes[2][KernelMax] = {};
    int krnlNumRegs[2][KernelMax] = {};

    // Map of kernels that are present in compiled 
    // modules and mutex to protect access to it
    KernelOptimisationOutput kernelsToOptimise;
    std::mutex kernelsToOptimiseMutex;

    // Do two repetitions with different candidate kernel size
    const size_t warpSize = 32;
    const size_t repBlockSizes[2] = {warpSize, warpSize * 2};
    std::vector<size_t> groupSizes[KernelMax];
    for(unsigned int r = 0; r < 2; r++) {
        LOGD_BACKEND  << "Generating code with block size:" << repBlockSizes[r];

        // Start with all group sizes set to warp size
        std::fill(blockSize.begin(), blockSize.end(), repBlockSizes[r]);

        // Create backend
        Backend backend(blockSize, preferences, deviceID, model.zeroCopyInUse());

        // Create merged model
        ModelSpecMerged modelMerged(backend, model);

        // Calculate group sizes for first block size
        if(r == 0) {
            calcGroupSizes(backend, modelMerged, groupSizes);
        }

        // Get memory spaces available to this backend
        // **NOTE** Memory spaces are given out on a first-come, first-serve basis so subsequent groups are in preferential order
        auto memorySpaces = backend.getMergedGroupMemorySpaces(modelMerged);

        // Generate code with suffix so it doesn't interfere with primary generated code
        // **NOTE** we don't really need to generate all the code but, on windows, generating code selectively seems to result in werid b
        const std::string dryRunSuffix = "CUDAOptim";
        {
            std::list<std::ofstream> fileStreams;
            auto fileStreamCreator =
                [&dryRunSuffix, &fileStreams, &outputPath](const std::string &title, const std::string &extension) -> std::ostream &
                {
                    fileStreams.emplace_back((outputPath / (title + "CUDAOptim." + extension)).str());
                    return fileStreams.back();
                };
            generateSynapseUpdate(fileStreamCreator, modelMerged, backend, memorySpaces, dryRunSuffix);
            generateNeuronUpdate(fileStreamCreator, modelMerged, backend, memorySpaces, dryRunSuffix);
            generateCustomUpdate(fileStreamCreator, modelMerged, backend, memorySpaces, dryRunSuffix);
            generateInit(fileStreamCreator, modelMerged, backend, memorySpaces, dryRunSuffix);
            generateRunner(outputPath, modelMerged, backend, dryRunSuffix);
        }

        // Loop through modules
        std::vector<std::thread> threads;
        for(const auto &m : modules) {
            // Calculate module's hash digest
            // **NOTE** this COULD be done in thread functions but, because when using GeNN from Python,
            // this will call into Python code it would require whole Python interface to be made thread-safe
            const auto hashDigest = std::invoke(m.getArchetypeHashDigest, modelMerged);

            // Launch thread to analyse kernels in this module (if required)
            threads.emplace_back(analyseModule, std::cref(m), r, cuContext, hashDigest, std::cref(outputPath), std::cref(nvccPath),
                                 std::cref(backend), std::cref(customUpdateKernels), std::cref(customTransposeUpdateKernels),
                                 std::ref(krnlSharedSizeBytes), std::ref(krnlNumRegs), std::ref(kernelsToOptimise), std::ref(kernelsToOptimiseMutex));
        }

        // Join all threads
        for(auto &t : threads) {
            t.join();
        }

        // Remove tempory source file
        if(std::remove((outputPath / ("runner" + dryRunSuffix + ".cc")).str().c_str())) {
            LOGW_BACKEND << "Cannot remove dry-run source file";
        }
        if(std::remove((outputPath / ("definitions" + dryRunSuffix + ".h")).str().c_str())) {
            LOGW_BACKEND << "Cannot remove dry-run source file";
        }
    }

    // Destroy context
    CHECK_CU_ERRORS(cuCtxDestroy(cuContext));

    // Get properties of device architecture
    size_t warpAllocGran;
    size_t regAllocGran;
    size_t smemAllocGran;
    size_t maxBlocksPerSM;
    getDeviceArchitectureProperties(deviceProps, warpAllocGran, regAllocGran, smemAllocGran, maxBlocksPerSM);

    // Zero block sizes
    std::fill(blockSize.begin(), blockSize.end(), 0);

    // Loop through kernels to optimise
    for(auto &k : kernelsToOptimise) {
        LOGD_BACKEND << "Kernel '" << Backend::KernelNames[k.first] << "':";

        // Get required number of registers per thread and shared memory bytes for this kernel
        // **NOTE** register requirements are assumed to remain constant as they're vector-width
        const size_t reqNumRegs = (size_t)krnlNumRegs[0][k.first];
        const size_t reqSharedMemBytes[2] = {(size_t)krnlSharedSizeBytes[0][k.first], (size_t)krnlSharedSizeBytes[1][k.first]};

        // Calculate coefficients for requiredSharedMemBytes = (A * blockThreads) + B model
        const size_t reqSharedMemBytesA = (reqSharedMemBytes[1] - reqSharedMemBytes[0]) / (repBlockSizes[1] - repBlockSizes[0]);
        const size_t reqSharedMemBytesB = reqSharedMemBytes[0] - (reqSharedMemBytesA * repBlockSizes[0]);

        // Loop through possible
        const size_t maxBlockWarps = deviceProps.maxThreadsPerBlock / warpSize;
        for(size_t blockWarps = 1; blockWarps < maxBlockWarps; blockWarps++) {
            const size_t blockThreads = blockWarps * warpSize;
            LOGD_BACKEND << "\tCandidate block size:" << blockThreads;

            // Estimate shared memory for block size and padd
            const size_t reqSharedMemBytes = padSize((reqSharedMemBytesA * blockThreads) + reqSharedMemBytesB, smemAllocGran);
            LOGD_BACKEND << "\t\tEstimated shared memory required:" << reqSharedMemBytes << " bytes (padded)";

            // Calculate number of blocks the groups used by this kernel will require
            const size_t reqBlocks = std::accumulate(groupSizes[k.first].begin(), groupSizes[k.first].end(), size_t{0},
                                                     [blockThreads](size_t acc, size_t size)
                                                     {
                                                         return acc + ceilDivide(size, blockThreads);
                                                     });
            LOGD_BACKEND << "\t\tBlocks required (according to padded sum):" << reqBlocks;

            // Start estimating SM block limit - the number of blocks of this size that can run on a single SM
            size_t smBlockLimit = deviceProps.maxThreadsPerMultiProcessor / blockThreads;
            LOGD_BACKEND << "\t\tSM block limit due to maxThreadsPerMultiProcessor:" << smBlockLimit;

            smBlockLimit = std::min(smBlockLimit, maxBlocksPerSM);
            LOGD_BACKEND << "\t\tSM block limit corrected for maxBlocksPerSM:" << smBlockLimit;

            // If register allocation is per-block
            if (deviceProps.major == 1) {
                // Pad size of block based on warp allocation granularity
                const size_t paddedNumBlockWarps = padSize(blockWarps, warpAllocGran);

                // Calculate number of registers per block and pad with register allocation granularity
                const size_t paddedNumRegPerBlock = padSize(paddedNumBlockWarps * reqNumRegs * warpSize, regAllocGran);

                // Update limit based on maximum registers available on SM
                smBlockLimit = std::min(smBlockLimit, deviceProps.regsPerBlock / paddedNumRegPerBlock);
            }
            // Otherwise, if register allocation is per-warp
            else {
                // Caculate number of registers per warp and pad with register allocation granularity
                const size_t paddedNumRegPerWarp = padSize(reqNumRegs * warpSize, regAllocGran);

                // Determine how many warps can therefore be simultaneously run on SM
                const size_t paddedNumWarpsPerSM = padSize(deviceProps.regsPerBlock / paddedNumRegPerWarp, warpAllocGran);

                // Update limit based on the number of warps required
                smBlockLimit = std::min(smBlockLimit, paddedNumWarpsPerSM / blockWarps);
            }
            LOGD_BACKEND << "\t\tSM block limit corrected for registers:" << smBlockLimit;

            // If this kernel requires any shared memory, update limit to reflect shared memory available in each multiprocessor
            // **NOTE** this used to be sharedMemPerBlock but that seems incorrect
            if(reqSharedMemBytes != 0) {
                smBlockLimit = std::min(smBlockLimit, deviceProps.sharedMemPerMultiprocessor / reqSharedMemBytes);
                LOGD_BACKEND << "\t\tSM block limit corrected for shared memory:" << smBlockLimit;
            }

            // Calculate occupancy
            const size_t newOccupancy = blockWarps * smBlockLimit * deviceProps.multiProcessorCount;

            // Use a small block size if it allows all groups to occupy the device concurrently
            if (reqBlocks <= (smBlockLimit * deviceProps.multiProcessorCount)) {
                blockSize[k.first] = blockThreads;
                k.second.second = newOccupancy;
                k.second.first = true;

                LOGD_BACKEND << "\t\tSmall model situation detected - block size:" << blockSize[k.first];

                // For small model the first (smallest) block size allowing it is chosen
                break;
            }
            // Otherwise, if we've improved on previous best occupancy
            else if(newOccupancy > k.second.second) {
                blockSize[k.first] = blockThreads;
                k.second.second = newOccupancy;

                LOGD_BACKEND << "\t\tNew highest occupancy: " << newOccupancy << ", block size:" << blockSize[k.first];
            }

        }

        LOGI_BACKEND << "Kernel: " << Backend::KernelNames[k.first] << ", block size:" << blockSize[k.first];
    }

    // Return optimisation data
    return kernelsToOptimise;
}
//--------------------------------------------------------------------------
int chooseOptimalDevice(const ModelSpecInternal &model, KernelBlockSize &blockSize,
                        const Preferences &preferences, const filesystem::path &outputPath)
{
    // Get number of devices
    int deviceCount;
    CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
    if(deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found");
    }

    // Loop through devices
    typedef std::tuple<int, size_t, size_t, KernelBlockSize> Device;
    std::vector<Device> devices;
    devices.reserve(deviceCount);
    for(int d = 0; d < deviceCount; d++) {
        // Get properties
        cudaDeviceProp deviceProps;
        CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&deviceProps, d));
        const int smVersion = (deviceProps.major * 10) + deviceProps.minor;

        // Optimise block size for this device
        KernelBlockSize optimalBlockSize;
        const auto kernels = optimizeBlockSize(d, deviceProps, model, optimalBlockSize, preferences, outputPath);

        // Sum up occupancy of each kernel
        const size_t totalOccupancy = std::accumulate(kernels.begin(), kernels.end(), size_t{0},
                                                      [](size_t acc, const KernelOptimisationOutput::value_type &kernel)
                                                      {
                                                          return acc + kernel.second.second;
                                                      });

        // Count number of kernels that count as small models
        const size_t numSmallModelKernels = std::accumulate(kernels.begin(), kernels.end(), size_t{0},
                                                            [](size_t acc, const KernelOptimisationOutput::value_type &kernel)
                                                            {
                                                                return acc + (kernel.second.first ? 1 : 0);
                                                            });

        LOGD_BACKEND << "Device " << d << " - total occupancy:" << totalOccupancy << ", number of small models:" << numSmallModelKernels << ", SM version:" << smVersion;
        devices.emplace_back(smVersion, totalOccupancy, numSmallModelKernels, optimalBlockSize);
    }

    // Find best device
    const auto bestDevice = std::min_element(devices.cbegin(), devices.cend(),
        [](const Device &a, const Device &b)
        {
            // If 'a' have a higher number of small model kernels -  return true - it is better
            const size_t numSmallModelKernelsA = std::get<2>(a);
            const size_t numSmallModelKernelsB = std::get<2>(b);
            if (numSmallModelKernelsA > numSmallModelKernelsB) {
                return true;
            }
            // Otherwise, if the two devices have an identical small model kernel count
            else if(numSmallModelKernelsA == numSmallModelKernelsB) {
                // If 'a' has a higher total occupancy - return true - it is better
                const size_t totalOccupancyA = std::get<1>(a);
                const size_t totalOccupancyB = std::get<1>(b);
                if(totalOccupancyA > totalOccupancyB) {
                    return true;
                }
                // Otherwise, if the two devices have identical occupancy
                else if(totalOccupancyA == totalOccupancyB) {
                    // If 'a' has a higher SM version - return true - it's better
                    const int smVersionA = std::get<0>(a);
                    const int smVersionB = std::get<0>(b);
                    if(smVersionA > smVersionB) {
                        return true;
                    }
                }
            }

            // 'a' is not better - return false
            return false;
        });

    // Find ID of best device
    const int bestDeviceID = (int)std::distance(devices.cbegin(), bestDevice);
    LOGI_BACKEND << "Optimal device " << bestDeviceID << " - total occupancy:" << std::get<1>(*bestDevice) << ", number of small models:" << std::get<2>(*bestDevice) << ", SM version:" << std::get<0>(*bestDevice);

    // Get optimal block size from best device
    blockSize = std::get<3>(*bestDevice);


    // Return ID of best device
    return bestDeviceID;
}
//--------------------------------------------------------------------------
int chooseDeviceWithMostGlobalMemory()
{
    // Get number of devices
    int deviceCount;
    CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
    if(deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found");
    }

    // Loop through devices
    size_t mostGlobalMemory = 0;
    int bestDevice = -1;
    for(int d = 0; d < deviceCount; d++) {
        // Get properties
        cudaDeviceProp deviceProps;
        CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&deviceProps, d));

        // If this device improves on previous best
        if(deviceProps.totalGlobalMem > mostGlobalMemory) {
            mostGlobalMemory = deviceProps.totalGlobalMem;
            bestDevice = d;
        }
    }

    LOGI_BACKEND << "Using device " << bestDevice << " which has " << mostGlobalMemory << " bytes of global memory";
    return bestDevice;
}
}   // anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator::Backends::Optimiser
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::CUDA::Optimiser
{
Backend createBackend(const ModelSpecInternal &model, const filesystem::path &outputPath, 
                      plog::Severity backendLevel, plog::IAppender *backendAppender, 
                      const Preferences &preferences)
{
    // If there isn't already a plog instance, initialise one
    if(plog::get<Logging::CHANNEL_BACKEND>() == nullptr) {
        plog::init<Logging::CHANNEL_BACKEND>(backendLevel, backendAppender);
    }
    // Otherwise, set it's max severity from GeNN preferences
    else {
        plog::get<Logging::CHANNEL_BACKEND>()->setMaxSeverity(backendLevel);
    }


    // If optimal device should be chosen
    if(preferences.deviceSelectMethod == DeviceSelect::OPTIMAL) {
        // Assert that block size selection method is set to occupancy as these two processes are linked
        assert(preferences.blockSizeSelectMethod == BlockSizeSelect::OCCUPANCY);

        // Choose optimal device
        KernelBlockSize cudaBlockSize;
        const int deviceID = chooseOptimalDevice(model, cudaBlockSize, preferences, outputPath);

        // Create backend
        return Backend(cudaBlockSize, preferences, deviceID, model.zeroCopyInUse());
    }
    // Otherwise
    else {
        // If we should select device with most memory, do so otherwise use manually selected device
        const int deviceID = (preferences.deviceSelectMethod == DeviceSelect::MOST_MEMORY)
            ? chooseDeviceWithMostGlobalMemory() : preferences.manualDeviceID;

        // If we should pick kernel block sizes based on occupancy
        if(preferences.blockSizeSelectMethod == BlockSizeSelect::OCCUPANCY) {
            // Get properties
            cudaDeviceProp deviceProps;
            CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&deviceProps, deviceID));

            // Optimise block size
            KernelBlockSize cudaBlockSize;
            optimizeBlockSize(deviceID, deviceProps, model, cudaBlockSize, preferences, outputPath);

            // Create backend
            return Backend(cudaBlockSize, preferences, deviceID, model.zeroCopyInUse());
        }
        // Otherwise, create backend using manual block sizes specified in preferences
        else {
            return Backend(preferences.manualBlockSizes, preferences, deviceID, model.zeroCopyInUse());
        }

    }
}
}   // namespace CodeGenerator::Backends::Optimiser

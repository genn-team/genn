#include "optimiser.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
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
#include "code_generator/generateCustomUpdate.h"
#include "code_generator/generateInit.h"
#include "code_generator/generateNeuronUpdate.h"
#include "code_generator/generateRunner.h"
#include "code_generator/generateSynapseUpdate.h"
#include "code_generator/generateSupportCode.h"
#include "code_generator/modelSpecMerged.h"

// CUDA backend includes
#include "utils.h"

using namespace CodeGenerator;
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

//! Table of module names to functions to get their archetype hash digests and the kernel IDs they might contain
const std::vector<std::tuple<std::string, GetArchetypeHashDigestFn, std::vector<Kernel>>> modules = {
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
        maxBlocksPerSM = (deviceProps.minor == 0) ? 32 : 16;
    }
    else {
        smemAllocGran = 128;
        warpAllocGran = 4;
        regAllocGran = 256;
        maxBlocksPerSM = (deviceProps.minor == 0) ? 32 : 16;

        if(deviceProps.major > 8) {
            LOGW_BACKEND << "Unsupported CUDA device major version: " << deviceProps.major;
            LOGW_BACKEND << "This is a bug! Please report it at https://github.com/genn-team/genn.";
            LOGW_BACKEND << "Falling back to next latest SM version parameters.";
        }
    }
}
//--------------------------------------------------------------------------
void calcGroupSizes(const CUDA::Preferences &preferences, const ModelSpecInternal &model,
                    std::vector<size_t> (&groupSizes)[KernelMax], std::set<std::string> &customUpdateKernels,
                    std::set<std::string> &customTransposeUpdateKernels)
{
    // Loop through neuron groups
    for(const auto &n : model.getNeuronGroups()) {
        // Add number of neurons to vector of neuron kernels
        groupSizes[KernelNeuronUpdate].push_back(model.getBatchSize() * n.second.getNumNeurons());

        // Add number of neurons to initialisation kernel (all neuron groups at least require spike counts initialising)
        groupSizes[KernelInitialize].push_back(n.second.getNumNeurons());

        // If neuron group requires previous spike or spike-like-event times to be reset after update 
        // i.e. in the pre-neuron reset kernel, add number of neurons to kernel
        if(n.second.isPrevSpikeTimeRequired() || n.second.isPrevSpikeEventTimeRequired()) {
            groupSizes[KernelNeuronPrevSpikeTimeUpdate].push_back((size_t)model.getBatchSize() * n.second.getNumNeurons());
        }
    }

    // Loop through custom updates, add size to vector of custom update groups and update group name to set
    for(const auto &c : model.getCustomUpdates()) {
        groupSizes[KernelCustomUpdate].push_back(c.second.isBatched() ? (model.getBatchSize() * c.second.getSize()) : c.second.getSize());
        customUpdateKernels.insert(c.second.getUpdateGroupName());
    }

     // Loop through custom updates add size to vector of custom update groups and update group name to set
    for(const auto &c : model.getCustomWUUpdates()) {
        const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(c.second.getSynapseGroup());
        if(c.second.isTransposeOperation()) {
            const size_t numCopies = c.second.isBatched() ? model.getBatchSize() : 1;
            const size_t size = numCopies * sgInternal->getSrcNeuronGroup()->getNumNeurons() * sgInternal->getTrgNeuronGroup()->getNumNeurons();
            groupSizes[KernelCustomTransposeUpdate].push_back(size);
            customTransposeUpdateKernels.insert(c.second.getUpdateGroupName());
        }
        else {
            customUpdateKernels.insert(c.second.getUpdateGroupName());

            const size_t numCopies = c.second.isBatched() ? model.getBatchSize() : 1;
            if(sgInternal->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                groupSizes[KernelCustomUpdate].push_back(numCopies * sgInternal->getSrcNeuronGroup()->getNumNeurons() * sgInternal->getMaxConnections());
            }
            else {
                groupSizes[KernelCustomUpdate].push_back(numCopies * sgInternal->getSrcNeuronGroup()->getNumNeurons() * sgInternal->getTrgNeuronGroup()->getNumNeurons());
            }
        }
    }

    // Loop through synapse groups
    size_t numPreSynapseResetGroups = 0;
    for(const auto &s : model.getSynapseGroups()) {
        if(s.second.isSpikeEventRequired() || s.second.isTrueSpikeRequired()) {
            groupSizes[KernelPresynapticUpdate].push_back(model.getBatchSize() * Backend::getNumPresynapticUpdateThreads(s.second, preferences));
        }

        if(!s.second.getWUModel()->getLearnPostCode().empty()) {
            groupSizes[KernelPostsynapticUpdate].push_back(model.getBatchSize() * Backend::getNumPostsynapticUpdateThreads(s.second));
        }

        if(!s.second.getWUModel()->getSynapseDynamicsCode().empty()) {
            groupSizes[KernelSynapseDynamicsUpdate].push_back(model.getBatchSize() * Backend::getNumSynapseDynamicsThreads(s.second));
        }

        // If synapse group has individual weights and needs device initialisation
        if((s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && s.second.isWUVarInitRequired()) {
            const size_t numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
            const size_t numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();
            // **FIXME**
            if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                groupSizes[KernelInitializeSparse].push_back(numSrcNeurons);
            }
            else {
                groupSizes[KernelInitialize].push_back(numSrcNeurons * numTrgNeurons);
            }
        }

        // If this synapse group requires dendritic delay, it requires a pre-synapse reset
        if(s.second.isDendriticDelayRequired()) {
            numPreSynapseResetGroups++;
        }
    }

    // Add group sizes for reset kernels
    groupSizes[KernelNeuronSpikeQueueUpdate].push_back(model.getNeuronGroups().size());
    groupSizes[KernelSynapseDendriticDelayUpdate].push_back(numPreSynapseResetGroups);
}
//--------------------------------------------------------------------------
void analyseModule(const std::tuple<std::string, GetArchetypeHashDigestFn, std::vector<Kernel>> &module, unsigned int r, CUcontext context, 
                   const filesystem::path &outputPath, const filesystem::path &nvccPath, const ModelSpecMerged &modelMerged, const Backend &backend,
                   const std::set<std::string> &customUpdateKernels, const std::set<std::string> &customTransposeUpdateKernels, 
                   int (&krnlSharedSizeBytes)[2][KernelMax], int (&krnlNumRegs)[2][KernelMax], KernelOptimisationOutput &kernelsToOptimise, std::mutex &kernelsToOptimiseMutex)
{
    // Build source and module paths from module name
    const std::string sourcePath = (outputPath / (std::get<0>(module) + "CUDAOptim.cc")).str();
    const std::string moduleSHAPath = (outputPath / (std::get<0>(module) + "CUDA" + std::to_string(r) + ".sha")).str();

    // Calculate modules hash digest
    const auto hashDigest = (modelMerged.*std::get<1>(module))();

    LOGD_BACKEND << "\tModule " << std::get<0>(module);
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
            for(Kernel k : std::get<2>(module)) {
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
    const std::string nvccCommand = "\"\"" + nvccPath.str() + "\" -cubin " + backend.getNVCCFlags() + " -DBUILDING_GENERATED_CODE -o \"" + sourcePath + ".cubin\" \"" + sourcePath + "\"\"";
#else
    const std::string nvccCommand = "\"" + nvccPath.str() + "\" -cubin " + backend.getNVCCFlags() + " -DBUILDING_GENERATED_CODE -o \"" + sourcePath + ".cubin\" \"" + sourcePath + "\"";
#endif
            
    if(system(nvccCommand.c_str()) != 0) {
        throw std::runtime_error("optimizeBlockSize: NVCC failed");
    }

    // Load compiled module
    CUmodule loadedModule;
    CHECK_CU_ERRORS(cuModuleLoad(&loadedModule, (sourcePath + ".cubin").c_str()));

    // Loop through kernels that might be in this module
    for(Kernel k : std::get<2>(module)) {
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
    for(Kernel k : std::get<2>(module)) {
        os << krnlSharedSizeBytes[r][k] << " " << krnlNumRegs[r][k] << std::endl;
    }

    // Remove tempory source file
    if(std::remove((sourcePath + ".cubin").c_str())) {
        LOGW_BACKEND << "Cannot remove dry-run cubin file";
    }

    // Remove tempory cubin file
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

    // Calculate model group sizes
    std::set<std::string> customUpdateKernels;
    std::set<std::string> customTransposeUpdateKernels;
    std::vector<size_t> groupSizes[KernelMax];
    calcGroupSizes(preferences, model, groupSizes, customUpdateKernels, customTransposeUpdateKernels);

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

    // Do two repititions with different candidate kernel size
    const size_t warpSize = 32;
    const size_t repBlockSizes[2] = {warpSize, warpSize * 2};
    for(unsigned int r = 0; r < 2; r++) {
        LOGD_BACKEND  << "Generating code with block size:" << repBlockSizes[r];

        // Start with all group sizes set to warp size
        std::fill(blockSize.begin(), blockSize.end(), repBlockSizes[r]);

        // Create backend
        Backend backend(blockSize, preferences, model.getPrecision(), deviceID);

        // Create merged model
        ModelSpecMerged modelMerged(model, backend);

        // Generate code with suffix so it doesn't interfere with primary generated code
        // **NOTE** we don't really need to generate all the code but, on windows, generating code selectively seems to result in werid b
        const std::string dryRunSuffix = "CUDAOptim";
        generateRunner(outputPath, modelMerged, backend, dryRunSuffix);
        generateSynapseUpdate(outputPath, modelMerged, backend, dryRunSuffix);
        generateNeuronUpdate(outputPath, modelMerged, backend, dryRunSuffix);
        generateCustomUpdate(outputPath, modelMerged, backend, dryRunSuffix);
        generateInit(outputPath, modelMerged, backend, dryRunSuffix);

        // Generate support code module if the backend supports namespaces
        if (backend.supportsNamespace()) {
            generateSupportCode(outputPath, modelMerged, dryRunSuffix);
        }

        // Loop through modules and launch threads to analyse kernels if required
        std::vector<std::thread> threads;
        for(const auto &m : modules) {
            threads.emplace_back(analyseModule, std::cref(m), r, cuContext, std::cref(outputPath), std::cref(nvccPath),
                                 std::cref(modelMerged), std::cref(backend), std::cref(customUpdateKernels), std::cref(customTransposeUpdateKernels),
                                 std::ref(krnlSharedSizeBytes), std::ref(krnlNumRegs), std::ref(kernelsToOptimise), std::ref(kernelsToOptimiseMutex));
            //analyseModule(m, r, cuContext, outputPath, nvccPath, modelMerged, backend, customUpdateKernels, customTransposeUpdateKernels,
            //              krnlSharedSizeBytes, krnlNumRegs, kernelsToOptimise, kernelsToOptimiseMutex);
        }

        // Join all threads
        for(auto &t : threads) {
            t.join();
        }

        // Remove tempory source file
        if(std::remove((outputPath / ("runner" + dryRunSuffix + ".cc")).str().c_str())) {
            LOGW_BACKEND << "Cannot remove dry-run source file";
        }
        if(std::remove((outputPath / ("supportCode" + dryRunSuffix + ".h")).str().c_str())) {
            LOGW_BACKEND << "Cannot remove dry-run source file";
        }
        if(std::remove((outputPath / ("definitions" + dryRunSuffix + ".h")).str().c_str())) {
            LOGW_BACKEND << "Cannot remove dry-run source file";
        }
        if(std::remove((outputPath / ("definitionsInternal" + dryRunSuffix + ".h")).str().c_str())) {
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
    LOGI_BACKEND << "Optimal  device " << bestDeviceID << " - total occupancy:" << std::get<1>(*bestDevice) << ", number of small models:" << std::get<2>(*bestDevice) << ", SM version:" << std::get<0>(*bestDevice);

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
}
// CodeGenerator::Backends::Optimiser
namespace CodeGenerator
{
namespace CUDA
{
namespace Optimiser
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
        return Backend(cudaBlockSize, preferences, model.getPrecision(), deviceID);
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
            return Backend(cudaBlockSize, preferences, model.getPrecision(), deviceID);
        }
        // Otherwise, create backend using manual block sizes specified in preferences
        else {
            return Backend(preferences.manualBlockSizes, preferences, model.getPrecision(), deviceID);
        }

    }
}
}   // namespace Optimiser
}   // namespace CUDA
}   // namespace CodeGenerator

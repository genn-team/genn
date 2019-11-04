#include "optimiser.h"

// Standard C++ includes
#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>

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
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/generateAll.h"

// CUDA backend includes
#include "utils.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
typedef std::map<unsigned int, std::pair<bool, size_t>> KernelOptimisationOutput;

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
    else {
        smemAllocGran = 256;
        warpAllocGran = 4;
        regAllocGran = 256;
        maxBlocksPerSM = 32;

        if(deviceProps.major > 7) {
            LOGW << "Unsupported CUDA device major version: " << deviceProps.major;
            LOGW << "This is a bug! Please report it at https://github.com/genn-team/genn.";
            LOGW << "Falling back to next latest SM version parameters.";
        }
    }
}
//--------------------------------------------------------------------------
void calcGroupSizes(const cudaDeviceProp &deviceProps, const ModelSpecInternal &model, std::vector<size_t> (&groupSizes)[CodeGenerator::CUDA::KernelMax])
{
    using namespace CodeGenerator;
    using namespace CUDA;

    // Loop through neuron groups
    for(const auto &n : model.getLocalNeuronGroups()) {
        // Add number of neurons to vector of neuron kernels
        groupSizes[KernelNeuronUpdate].push_back(n.second.getNumNeurons());

        // Add number of neurons to initialisation kernel (all neuron groups at least require spike counts initialising)
        groupSizes[KernelInitialize].push_back(n.second.getNumNeurons());
    }

    // Loop through synapse groups
    size_t numPreSynapseResetGroups = 0;
    for(const auto &s : model.getLocalSynapseGroups()) {
        groupSizes[KernelPresynapticUpdate].push_back(Backend::getNumPresynapticUpdateThreads(s.second, deviceProps));

        if(!s.second.getWUModel()->getLearnPostCode().empty()) {
            groupSizes[KernelPostsynapticUpdate].push_back(Backend::getNumPostsynapticUpdateThreads(s.second));
        }

        if (!s.second.getWUModel()->getLearnPostCode().empty()) {
            groupSizes[KernelSynapseDynamicsUpdate].push_back(Backend::getNumSynapseDynamicsThreads(s.second));
        }

        // If synapse group has individual weights and needs device initialisation
        if((s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && s.second.isWUVarInitRequired()) {
            const size_t numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
            const size_t numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();
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
    groupSizes[KernelPreNeuronReset].push_back(model.getLocalNeuronGroups().size());
    groupSizes[KernelPreSynapseReset].push_back(numPreSynapseResetGroups);
}
//--------------------------------------------------------------------------
KernelOptimisationOutput optimizeBlockSize(int deviceID, const cudaDeviceProp &deviceProps, const ModelSpecInternal &model,
                                           CodeGenerator::CUDA::KernelBlockSize &blockSize, const CodeGenerator::CUDA::Preferences &preferences,
                                           int localHostID, const filesystem::path &outputPath)
{
    using namespace CodeGenerator;
    using namespace CUDA;

    // Select device
    cudaSetDevice(deviceID);

    // Calculate model group sizes
    std::vector<size_t> groupSizes[KernelMax];
    calcGroupSizes(deviceProps, model, groupSizes);

    // Create CUDA drive API device and context for accessing kernel attributes
    CUdevice cuDevice;
    CUcontext cuContext;
    CHECK_CU_ERRORS(cuDeviceGet(&cuDevice, deviceID));
    CHECK_CU_ERRORS(cuCtxCreate(&cuContext, 0, cuDevice));

    // Bitset to mark which kernels are present and array of their attributes for each repetition
    int krnlSharedSizeBytes[2][KernelMax];
    int krnlNumRegs[2][KernelMax];

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
    
    // Do two repititions with different candidate kernel size
    const size_t warpSize = 32;
    const size_t repBlockSizes[2] = {warpSize, warpSize * 2};
    KernelOptimisationOutput kernelsToOptimise;
    for(unsigned int r = 0; r < 2; r++) {
        LOGD  << "Generating code with block size:" << repBlockSizes[r];

        // Start with all group sizes set to warp size
        std::fill(blockSize.begin(), blockSize.end(), repBlockSizes[r]);

        // Create backend
        Backend backend(blockSize, preferences, localHostID, model.getPrecision(), deviceID);

        // Generate code
        const auto moduleNames = generateAll(model, backend, outputPath, true);

        // Set context
        // **NOTE** CUDA calls in code generation seem to lose driver context
        CHECK_CU_ERRORS(cuCtxSetCurrent(cuContext));

        // Loop through generated modules
        for(const auto &m : moduleNames) {
            // Build module
            const std::string modulePath = (outputPath / m).str();
            
#ifdef _WIN32
            // **YUCK** extra outer quotes required to workaround gross windowsness https://stackoverflow.com/questions/9964865/c-system-not-working-when-there-are-spaces-in-two-different-parameters
            const std::string nvccCommand = "\"\"" + nvccPath.str() + "\" -cubin " + backend.getNVCCFlags() + " -DBUILDING_GENERATED_CODE -o \"" + modulePath + ".cubin\" \"" + modulePath + ".cc\"\"";
#else
            const std::string nvccCommand = "\"" + nvccPath.str() + "\" -cubin " + backend.getNVCCFlags() + " -DBUILDING_GENERATED_CODE -o \"" + modulePath + ".cubin\" \"" + modulePath + ".cc\"";
 #endif
            if(system(nvccCommand.c_str()) != 0) {
                throw std::runtime_error("optimizeBlockSize: NVCC failed");
            }

            // Load compiled module
            CUmodule module;
            CHECK_CU_ERRORS(cuModuleLoad(&module, (modulePath + ".cubin").c_str()));

            // Loop through kernels
            for (unsigned int k = 0; k < KernelMax; k++) {
                // If function is found
                CUfunction kern;
                CUresult res = cuModuleGetFunction(&kern, module, Backend::KernelNames[k]);
                if (res == CUDA_SUCCESS) {
                    LOGD << "\tKernel '" << Backend::KernelNames[k] << "' found";

                    // Read function's shared memory size and register counand add blank entry to map of kernels to optimise
                    CHECK_CU_ERRORS(cuFuncGetAttribute(&krnlSharedSizeBytes[r][k], CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kern));
                    CHECK_CU_ERRORS(cuFuncGetAttribute(&krnlNumRegs[r][k], CU_FUNC_ATTRIBUTE_NUM_REGS , kern));

                    //CHECK_CUDA_ERRORS(cudaFuncGetAttributes(&krnlAttr[r][k], kern));
                    kernelsToOptimise.emplace(std::piecewise_construct,
                                              std::forward_as_tuple(k),
                                              std::forward_as_tuple(false, 0));

                    LOGD << "\t\tShared memory bytes:" << krnlSharedSizeBytes[r][k];
                    LOGD << "\t\tNum registers:" << krnlNumRegs[r][k];
                }
            }

            // Unload module
            CHECK_CU_ERRORS(cuModuleUnload(module));

            // Remove tempory cubin file
            if(std::remove((modulePath + ".cubin").c_str())) {
                LOGW << "Cannot remove dry-run cubin file";
            }
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
        LOGD << "Kernel '" << Backend::KernelNames[k.first] << "':";

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
            LOGD << "\tCandidate block size:" << blockThreads;

            // Estimate shared memory for block size and padd
            const size_t reqSharedMemBytes = Utils::padSize((reqSharedMemBytesA * blockThreads) + reqSharedMemBytesB, smemAllocGran);
            LOGD << "\t\tEstimated shared memory required:" << reqSharedMemBytes << " bytes (padded)";

            // Calculate number of blocks the groups used by this kernel will require
            const size_t reqBlocks = std::accumulate(groupSizes[k.first].begin(), groupSizes[k.first].end(), size_t{0},
                                                        [blockThreads](size_t acc, size_t size)
                                                        {
                                                            return acc + Utils::ceilDivide(size, blockThreads);
                                                        });
            LOGD << "\t\tBlocks required (according to padded sum):" << reqBlocks;

            // Start estimating SM block limit - the number of blocks of this size that can run on a single SM
            size_t smBlockLimit = deviceProps.maxThreadsPerMultiProcessor / blockThreads;
            LOGD << "\t\tSM block limit due to maxThreadsPerMultiProcessor:" << smBlockLimit;

            smBlockLimit = std::min(smBlockLimit, maxBlocksPerSM);
            LOGD << "\t\tSM block limit corrected for maxBlocksPerSM:" << smBlockLimit;

            // If register allocation is per-block
            if (deviceProps.major == 1) {
                // Pad size of block based on warp allocation granularity
                const size_t paddedNumBlockWarps = Utils::padSize(blockWarps, warpAllocGran);

                // Calculate number of registers per block and pad with register allocation granularity
                const size_t paddedNumRegPerBlock = Utils::padSize(paddedNumBlockWarps * reqNumRegs * warpSize, regAllocGran);

                // Update limit based on maximum registers available on SM
                smBlockLimit = std::min(smBlockLimit, deviceProps.regsPerBlock / paddedNumRegPerBlock);
            }
            // Otherwise, if register allocation is per-warp
            else {
                // Caculate number of registers per warp and pad with register allocation granularity
                const size_t paddedNumRegPerWarp = Utils::padSize(reqNumRegs * warpSize, regAllocGran);

                // Determine how many warps can therefore be simultaneously run on SM
                const size_t paddedNumWarpsPerSM = Utils::padSize(deviceProps.regsPerBlock / paddedNumRegPerWarp, warpAllocGran);

                // Update limit based on the number of warps required
                smBlockLimit = std::min(smBlockLimit, paddedNumWarpsPerSM / blockWarps);
            }
            LOGD << "\t\tSM block limit corrected for registers:" << smBlockLimit;

            // If this kernel requires any shared memory, update limit to reflect shared memory available in each multiprocessor
            // **NOTE** this used to be sharedMemPerBlock but that seems incorrect
            if(reqSharedMemBytes != 0) {
                smBlockLimit = std::min(smBlockLimit, deviceProps.sharedMemPerMultiprocessor / reqSharedMemBytes);
                LOGD << "\t\tSM block limit corrected for shared memory:" << smBlockLimit;
            }

            // Calculate occupancy
            const size_t newOccupancy = blockWarps * smBlockLimit * deviceProps.multiProcessorCount;

            // Use a small block size if it allows all groups to occupy the device concurrently
            if (reqBlocks <= (smBlockLimit * deviceProps.multiProcessorCount)) {
                blockSize[k.first] = blockThreads;
                k.second.second = newOccupancy;
                k.second.first = true;

                LOGD << "\t\tSmall model situation detected - block size:" << blockSize[k.first];

                // For small model the first (smallest) block size allowing it is chosen
                break;
            }
            // Otherwise, if we've improved on previous best occupancy
            else if(newOccupancy > k.second.second) {
                blockSize[k.first] = blockThreads;
                k.second.second = newOccupancy;

                LOGD << "\t\tNew highest occupancy: " << newOccupancy << ", block size:" << blockSize[k.first];
            }

        }

        LOGI << "Kernel: " << Backend::KernelNames[k.first] << ", block size:" << blockSize[k.first];
    }

    // Return optimisation data
    return kernelsToOptimise;
}
//--------------------------------------------------------------------------
int chooseOptimalDevice(const ModelSpecInternal &model, CodeGenerator::CUDA::KernelBlockSize &blockSize,
                        const CodeGenerator::CUDA::Preferences &preferences, int localHostID, const filesystem::path &outputPath)
{
    using namespace CodeGenerator;
    using namespace CUDA;
    
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
        const auto kernels = optimizeBlockSize(d, deviceProps, model, optimalBlockSize, preferences, localHostID, outputPath);

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

        LOGD << "Device " << d << " - total occupancy:" << totalOccupancy << ", number of small models:" << numSmallModelKernels << ", SM version:" << smVersion;
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
    LOGI << "Optimal  device " << bestDeviceID << " - total occupancy:" << std::get<1>(*bestDevice) << ", number of small models:" << std::get<2>(*bestDevice) << ", SM version:" << std::get<0>(*bestDevice);

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

    LOGI << "Using device " << bestDevice << " which has " << mostGlobalMemory << " bytes of global memory";
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
Backend createBackend(const ModelSpecInternal &model, const filesystem::path &outputPath, int localHostID,
                      const Preferences &preferences)
{
    // If optimal device should be chosen
    if(preferences.deviceSelectMethod == DeviceSelect::OPTIMAL) {
        // Assert that block size selection method is set to occupancy as these two processes are linked
        assert(preferences.blockSizeSelectMethod == BlockSizeSelect::OCCUPANCY);

        // Choose optimal device
        KernelBlockSize cudaBlockSize;
        const int deviceID = chooseOptimalDevice(model, cudaBlockSize, preferences, localHostID, outputPath);

        // Create backend
        return Backend(cudaBlockSize, preferences, localHostID, model.getPrecision(), deviceID);
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
            optimizeBlockSize(deviceID, deviceProps, model, cudaBlockSize, preferences, localHostID, outputPath);

            // Create backend
            return Backend(cudaBlockSize, preferences, localHostID, model.getPrecision(), deviceID);
        }
        // Otherwise, create backend using manual block sizes specified in preferences
        else {
            return Backend(preferences.manualBlockSizes, preferences, localHostID, model.getPrecision(), deviceID);
        }

    }
}
}   // namespace Optimiser
}   // namespace CUDA
}   // namespace CodeGenerator

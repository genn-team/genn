#include <bitset>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <tuple>

// PLOG includes
#include <plog/Log.h>
#include <plog/Appenders/ConsoleAppender.h>

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

// NuGeNN includes
#include "generateInit.h"
#include "generateMakefile.h"
#include "generateNeuronUpdate.h"
#include "generateSynapseUpdate.h"
#include "generateRunner.h"
#include "backends/base.h"
#include "backends/cuda.h"
#include "backends/singleThreadedCPU.h"
#include "third_party/path.h"

// GeNN robotics includes
#include "genn_models/exp_curr.h"
#include "genn_models/lif.h"

#include "common/vogels_2011.h"

using namespace CodeGenerator;

typedef std::map<unsigned int, std::pair<bool, size_t>> KernelOptimisationOutput;

enum Log
{
    LogDefault,
    LogBackend,
    LogOptimiser,
};

// Anonymous namespace
namespace
{
/*void modelDefinition(NNmodel &model)
{
    GENN_PREFERENCES::buildSharedLibrary = true;
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::defaultSparseConnectivityMode = VarMode::LOC_DEVICE_INIT_DEVICE;
    initGeNN();
    model.setDT(1.0);
    model.setName("tutorial2");

    // Izhikevich model parameters
    NeuronModels::Izhikevich::ParamValues izkParams(
        0.02,   // 0 - A
        0.2,    // 1 - B
        -65.0,  // 2 - C
        8.0);   // 3 - D

    // Izhikevich initial conditions
    InitVarSnippet::Uniform::ParamValues uDist(
        0.0,    // 0 - min
        20.0);  // 1 - max
    NeuronModels::Izhikevich::VarValues ikzInit(
        -65.0,                                      // 0 - V
        initVar<InitVarSnippet::Uniform>(uDist));   // 1 - U

    model.addNeuronPopulation<NeuronModels::Izhikevich>("Exc", 8000, izkParams, ikzInit);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Inh", 2000, izkParams, ikzInit);

    // DC current source parameters
    CurrentSourceModels::DC::ParamValues currentSourceParamVals(4.0);  // 0 - magnitude
    model.addCurrentSource<CurrentSourceModels::DC>("ExcStim", "Exc", currentSourceParamVals, {});
    model.addCurrentSource<CurrentSourceModels::DC>("InhStim", "Inh", currentSourceParamVals, {});

    WeightUpdateModels::StaticPulse::VarValues excSynInitValues(0.05);
    WeightUpdateModels::StaticPulse::VarValues inhSynInitValues(-0.25);

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(0.1); // 0 - prob
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Exc_Exc", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "Exc", "Exc",
        {}, excSynInitValues,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Exc_Inh", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "Exc", "Inh",
        {}, excSynInitValues,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Inh_Inh", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "Inh", "Inh",
        {}, inhSynInitValues,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Inh_Exc", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "Inh", "Exc",
        {}, inhSynInitValues,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    model.finalize();
}*/
void modelDefinition(NNmodel &model)
{
    using namespace BoBRobotics;
    GENN_PREFERENCES::buildSharedLibrary = true;
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::defaultSparseConnectivityMode = VarMode::LOC_DEVICE_INIT_DEVICE;

    initGeNN();
    model.setDT(1.0);
    model.setName("vogels_2011");
    model.setTiming(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitVarSnippet::Uniform::ParamValues vDist(
        -60.0,  // 0 - min
        -50.0); // 1 - max

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(
        0.02); // 0 - prob

    // LIF model parameters
    GeNNModels::LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.2,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    GeNNModels::LIF::VarValues lifInit(
        initVar<InitVarSnippet::Uniform>(vDist),    // 0 - V
        0.0);                                       // 1 - RefracTime

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues excitatoryStaticSynapseInit(
        0.03);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues inhibitoryStaticSynapseInit(
        -0.03);    // 0 - Wij (nA)

    // Additive STDP synapse parameters
    Vogels2011::ParamValues vogels2011AdditiveSTDPParams(
        20.0,   // 0 - Tau
        0.12,   // 1 - rho
        0.005,  // 2 - eta
        -1.0,    // 3 - Wmin
        0.0);    // 4 - Wmax

    Vogels2011::VarValues vogels2011AdditiveSTDPInit(
        0.0);  // 0 - g

    // Exponential current parameters
    GeNNModels::ExpCurr::ParamValues excitatoryExpCurrParams(
        5.0);  // 0 - TauSyn (ms)

    GeNNModels::ExpCurr::ParamValues inhibitoryExpCurrParams(
        10.0);  // 0 - TauSyn (ms)

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<GeNNModels::LIF>("E", 2000, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<GeNNModels::LIF>("I", 500, lifParams, lifInit);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, GeNNModels::ExpCurr>(
        "EE", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "E", "E",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, GeNNModels::ExpCurr>(
        "EI", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "E", "I",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, GeNNModels::ExpCurr>(
        "II", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "I", "I",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    auto *ie = model.addSynapsePopulation<Vogels2011, GeNNModels::ExpCurr>(
        "IE", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
        "I", "E",
        vogels2011AdditiveSTDPParams, vogels2011AdditiveSTDPInit,
        inhibitoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    // Configure plastic weight variables they can be downloaded to host
    ie->setWUVarMode("g", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    ie->setSparseConnectivityVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    // Configure spike variables so that they can be downloaded to host
    e->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    i->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    model.finalize();
}

std::vector<std::string> generateCode(const NNmodel &model, const Backends::Base &backend)
{
    // Create directory for generated code
    filesystem::create_directory("generated_code");

    // Open output file streams for generated code files
    std::ofstream definitionsStream("generated_code/definitions.h");
    std::ofstream neuronUpdateStream("generated_code/neuronUpdate.cc");
    std::ofstream synapseUpdateStream("generated_code/synapseUpdate.cc");
    std::ofstream initStream("generated_code/init.cc");
    std::ofstream runnerStream("generated_code/runner.cc");

    // Wrap output file streams in CodeStreams for formatting
    CodeStream definitions(definitionsStream);
    CodeStream neuronUpdate(neuronUpdateStream);
    CodeStream synapseUpdate(synapseUpdateStream);
    CodeStream init(initStream);
    CodeStream runner(runnerStream);

    // Generate modules
    generateNeuronUpdate(neuronUpdate, model, backend);
    generateSynapseUpdate(synapseUpdate, model, backend);
    generateInit(init, model, backend);
    generateRunner(definitions, runner, model, backend, 0);

    // Return names of generated modules
    return {"neuronUpdate", "synapseUpdate", "init", "runner"};
}

// **TODO** move somewhere
size_t ceilDivide(size_t numerator, size_t denominator)
{
    return ((numerator + denominator - 1) / denominator);
}
//--------------------------------------------------------------------------
size_t padSize(size_t size, size_t blockSize)
{
    return ceilDivide(size, blockSize) * blockSize;
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
    else {
        smemAllocGran = 256;
        warpAllocGran = 4;
        regAllocGran = 256;
        maxBlocksPerSM = 32;

        if(deviceProps.major > 7) {
            LOGW << "Unsupported CUDA device major version: " << deviceProp[theDevice].major;
            LOGW << "This is a bug! Please report it at https://github.com/genn-team/genn.";
            LOGW << "Falling back to next latest SM version parameters.";
        }
    }
}

void calcGroupSizes(const NNmodel &model, std::vector<unsigned int> (&groupSizes)[Backends::CUDA::KernelMax])
{
    using namespace Backends;

    // **TODO** this belongs in code generator somewhere

    // Loop through neuron groups
    for(const auto &n : model.getLocalNeuronGroups()) {
        // Add number of neurons to vector of neuron kernels
        groupSizes[CUDA::KernelNeuronUpdate].push_back(n.second.getNumNeurons());

        // If this neuron group requires on-device initialisation
        if(n.second.isSimRNGRequired() || n.second.isDeviceVarInitRequired()) {
            groupSizes[CUDA::KernelInitialize].push_back(n.second.getNumNeurons());
        }
    }

    // Loop through synapse groups
    for(const auto &s : model.getLocalSynapseGroups()) {
        groupSizes[CUDA::KernelPresynapticUpdate].push_back(CUDA::getNumPresynapticUpdateThreads(s.second));

        if(!s.second.getWUModel()->getLearnPostCode().empty()) {
            groupSizes[CUDA::KernelPostsynapticUpdate].push_back(CUDA::getNumPostsynapticUpdateThreads(s.second));
        }

        if (!s.second.getWUModel()->getLearnPostCode().empty()) {
            groupSizes[CUDA::KernelSynapseDynamicsUpdate].push_back(CUDA::getNumSynapseDynamicsThreads(s.second));
        }

        // If synapse group has individual weights and needs device initialisation
        if((s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && s.second.isWUDeviceVarInitRequired()) {
            const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
            const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();
            if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                groupSizes[CUDA::KernelInitializeSparse].push_back(numSrcNeurons);
            }
            else {
                groupSizes[CUDA::KernelInitialize].push_back(numSrcNeurons * numTrgNeurons);
            }
        }
    }

    // Add group sizes for reset kernels
    groupSizes[CUDA::KernelPreNeuronReset].push_back(model.getLocalNeuronGroups().size());
    groupSizes[CUDA::KernelPreSynapseReset].push_back(model.getNumPreSynapseResetRequiredGroups());
}


KernelOptimisationOutput optimizeBlockSize(int deviceID, const NNmodel &model, Backends::CUDA::KernelBlockSize &blockSize)
{
    using namespace Backends;

    // Calculate model group sizes
    std::vector<unsigned int> groupSizes[Backends::CUDA::KernelMax];
    calcGroupSizes(model, groupSizes);

    // Create CUDA drive API device and context for accessing kernel attributes
    CUdevice cuDevice;
    CUcontext cuContext;
    CHECK_CU_ERRORS(cuDeviceGet(&cuDevice, deviceID));
    CHECK_CU_ERRORS(cuCtxCreate(&cuContext, 0, cuDevice));

    // Bitset to mark which kernels are present and array of their attributes for each repetition
    cudaFuncAttributes krnlAttr[2][CUDA::KernelMax];

    // Do two repititions with different candidate kernel size
    const size_t warpSize = 32;
    const size_t repBlockSizes[2] = {warpSize, warpSize * 2};
    KernelOptimisationOutput kernelsToOptimise;
    for(unsigned int r = 0; r < 2; r++) {
        LOGD_(LogOptimiser)  << "Generating code with block size:" << repBlockSizes[r];

        // Start with all group sizes set to warp size
        std::fill(blockSize.begin(), blockSize.end(), repBlockSizes[r]);

        // Create backends
        Backends::SingleThreadedCPU cpuBackend(0);
        Backends::CUDA backend(blockSize, 0, deviceID, cpuBackend);

        // Generate code
        const auto moduleNames = generateCode(model, backend);

        // Set context
        // **NOTE** CUDA calls in code generation seem to lose driver context
        CHECK_CU_ERRORS(cuCtxSetCurrent(cuContext));

        // Loop through generated modules
        for(const auto &m : moduleNames) {
            // Build module
            const std::string nvccCommand = "nvcc -cubin " + backend.getNVCCFlags() + " -o generated_code/" + m + ".cubin generated_code/" + m + ".cc";
            if(system(nvccCommand.c_str()) != 0) {
                throw std::runtime_error("optimizeBlockSize: NVCC failed");
            }

            // Load compiled module
            CUmodule module;
            CHECK_CU_ERRORS(cuModuleLoad(&module, ("generated_code/" + m + ".cubin").c_str()));

            // Loop through kernels
            for (unsigned int k = 0; k < CUDA::KernelMax; k++) {
                // If function is found
                CUfunction kern;
                CUresult res = cuModuleGetFunction(&kern, module, CUDA::KernelNames[k]);
                if (res == CUDA_SUCCESS) {
                    LOGD_(LogOptimiser) << "\tKernel '" << CUDA::KernelNames[k] << "' found";

                    // Read it's attributes and add blank entry to map of kernels to optimise
                    cudaFuncGetAttributesDriver(&krnlAttr[r][k], kern);
                    kernelsToOptimise.emplace(std::piecewise_construct,
                                              std::forward_as_tuple(k),
                                              std::forward_as_tuple(false, 0));

                    LOGD_(LogOptimiser) << "\t\tShared memory bytes:" << krnlAttr[r][k].sharedSizeBytes;
                    LOGD_(LogOptimiser) << "\t\tNum registers:" << krnlAttr[r][k].numRegs;
                }
            }

            // Unload module
            CHECK_CU_ERRORS(cuModuleUnload(module));
        }
    }

    // Destroy context
    CHECK_CU_ERRORS(cuCtxDestroy(cuContext));

    // Get device properties
    cudaDeviceProp deviceProps;
    CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&deviceProps, deviceID));

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
        LOGD_(LogOptimiser) << "Kernel '" << CUDA::KernelNames[k.first] << "':";

        // Get required number of registers and shared memory bytes for this kernel
        // **NOTE** register requirements are assumed to remain constant as they're vector-width
        const size_t reqNumRegs = krnlAttr[0][k.first].numRegs;
        const size_t reqSharedMemBytes[2] = {krnlAttr[0][k.first].sharedSizeBytes, krnlAttr[1][k.first].sharedSizeBytes};

        // Calculate coefficients for requiredSharedMemBytes = (A * blockThreads) + B model
        const size_t reqSharedMemBytesA = (reqSharedMemBytes[1] - reqSharedMemBytes[0]) / (repBlockSizes[1] - repBlockSizes[0]);
        const size_t reqSharedMemBytesB = reqSharedMemBytes[0] - (reqSharedMemBytesA * repBlockSizes[0]);

        // Loop through possible
        const size_t maxBlockWarps = deviceProps.maxThreadsPerBlock / warpSize;
        for(size_t blockWarps = 1; blockWarps < maxBlockWarps; blockWarps++) {
            const size_t blockThreads = blockWarps * warpSize;
            LOGD_(LogOptimiser) << "\tCandidate block size:" << blockThreads;

            // Estimate shared memory for block size and padd
            const size_t reqSharedMemBytes = padSize((reqSharedMemBytesA * blockThreads) + reqSharedMemBytesB, smemAllocGran);
            LOGD_(LogOptimiser) << "\t\tEstimated shared memory required:" << reqSharedMemBytes << " bytes (padded)";

            // Calculate number of blocks the groups used by this kernel will require
            const size_t reqBlocks = std::accumulate(groupSizes[k.first].begin(), groupSizes[k.first].end(), 0,
                                                        [blockThreads](size_t acc, size_t size)
                                                        {
                                                            return acc + ceilDivide(size, blockThreads);
                                                        });
            LOGD_(LogOptimiser) << "\t\tBlocks required (according to padded sum):" << reqBlocks;

            // Start estimating SM block limit - the number of blocks of this size that can run on a single SM
            size_t smBlockLimit = deviceProps.maxThreadsPerMultiProcessor / blockThreads;
            LOGD_(LogOptimiser) << "\t\tSM block limit due to maxThreadsPerMultiProcessor:" << smBlockLimit;

            smBlockLimit = std::min(smBlockLimit, maxBlocksPerSM);
            LOGD_(LogOptimiser) << "\t\tSM block limit corrected for maxBlocksPerSM:" << smBlockLimit;

            // If register allocation is per-block
            if (deviceProps.major == 1) {
                // Pad size of block based on warp allocation granularity
                const size_t paddedNumBlockWarps = padSize(blockWarps, warpAllocGran);

                // Calculate number of registers per block and pad with register allocation granularity
                const size_t paddedNumRegPerBlock = padSize(paddedNumBlockWarps * reqNumRegs * warpSize, regAllocGran);

                // Update limit based on maximum registers per block
                // **NOTE** this doesn't quite make sense either
                smBlockLimit = std::min(smBlockLimit, deviceProps.regsPerBlock / paddedNumRegPerBlock);
            }
            // Otherwise, if register allocation is per-warp
            else {
                // Caculate number of registers per warp and pad with register allocation granularity
                const size_t paddedNumRegPerWarp = padSize(reqNumRegs * warpSize, regAllocGran);

                // **THINK** I don't understand this
                //blockLimit = floor(deviceProps.regsPerBlock / (paddedNumRegPerWarp * warpAllocGran)*warpAllocGran;

                // **NOTE** this doesn't quite make sense either
                //smBlockLimit = std::min(smBlockLimit, blockLimit / blockWarps);
            }
            LOGD_(LogOptimiser) << "\t\tSM block limit corrected for registers:" << smBlockLimit;

            // If this kernel requires any shared memory, update limit to reflect shared memory available in each multiprocessor
            // **NOTE** this used to be sharedMemPerBlock but that seems incorrect
            if(reqSharedMemBytes != 0) {
                smBlockLimit = std::min(smBlockLimit, deviceProps.sharedMemPerMultiprocessor / reqSharedMemBytes);
                LOGD_(LogOptimiser) << "\t\tSM block limit corrected for shared memory:" << smBlockLimit;
            }

            // Calculate occupancy
            const size_t newOccupancy = blockWarps * smBlockLimit * deviceProps.multiProcessorCount;

            // Use a small block size if it allows all groups to occupy the device concurrently
            if (reqBlocks <= (smBlockLimit * deviceProps.multiProcessorCount)) {
                blockSize[k.first] = blockThreads;
                k.second.second = newOccupancy;
                k.second.first = true;

                LOGD_(LogOptimiser) << "\t\tSmall model situation detected - block size:" << blockSize[k.first];

                // For small model the first (smallest) block size allowing it is chosen
                break;
            }
            // Otherwise, if we've improved on previous best occupancy
            else if(newOccupancy > k.second.second) {
                blockSize[k.first] = blockThreads;
                k.second.second = newOccupancy;

                LOGD_(LogOptimiser) << "\t\tNew highest occupancy: " << newOccupancy << ", block size:" << blockSize[k.first];
            }

        }

        LOGI_(LogOptimiser) << "Kernel: " << CUDA::KernelNames[k.first] << ", block size:" << blockSize[k.first];
    }

    // Return optimisation data
    return kernelsToOptimise;
}

int chooseOptimalDevice(const NNmodel &model, Backends::CUDA::KernelBlockSize &blockSize)
{
    // Get number of devices
    int deviceCount;
    CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
    if(deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found");
    }

    // Loop through devices
    typedef std::tuple<int, size_t, size_t, Backends::CUDA::KernelBlockSize> Device;
    std::vector<Device> devices;
    devices.reserve(deviceCount);
    for(int d = 0; d < deviceCount; d++) {
        // Get properties
        cudaDeviceProp deviceProps;
        CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&deviceProps, d));
        const int smVersion = (deviceProps.major * 10) + deviceProps.minor;

        // Optimise block size for this device
        Backends::CUDA::KernelBlockSize optimalBlockSize;
        const auto kernels = optimizeBlockSize(d, model, optimalBlockSize);

        // Sum up occupancy of each kernel
        const size_t totalOccupancy = std::accumulate(kernels.begin(), kernels.end(), 0,
                                                      [](size_t acc, const KernelOptimisationOutput::value_type &kernel)
                                                      {
                                                          return acc + kernel.second.second;
                                                      });

        // Count number of kernels that count as small models
        const size_t numSmallModelKernels = std::accumulate(kernels.begin(), kernels.end(), 0,
                                                            [](size_t acc, const KernelOptimisationOutput::value_type &kernel)
                                                            {
                                                                return acc + (kernel.second.first ? 1 : 0);
                                                            });

        LOGD_(LogOptimiser) << "Device " << d << " - total occupancy:" << totalOccupancy << ", number of small models:" << numSmallModelKernels << ", SM version:" << smVersion;
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
    const int bestDeviceID = std::distance(devices.cbegin(), bestDevice);
    LOGI_(LogOptimiser) << "Optimal  device " << bestDeviceID << " - total occupancy:" << std::get<1>(*bestDevice) << ", number of small models:" << std::get<2>(*bestDevice) << ", SM version:" << std::get<0>(*bestDevice);

    // Get optimal block size from best device
    blockSize = std::get<3>(*bestDevice);


    // Return ID of best device
    return bestDeviceID;
}

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

    LOGI_(LogOptimiser) << "Using device " << bestDevice << " which has " << mostGlobalMemory << " bytes of global memory";
    return bestDevice;
}
}   // Anonymous namespace

int main()
{
    // Initialise log channels, appending all to console
    // **TODO** de-crud standard logger
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init<LogDefault>(plog::info, &consoleAppender);
    plog::init<LogBackend>(plog::info, &consoleAppender);
    plog::init<LogOptimiser>(plog::info, &consoleAppender);

    NNmodel model;
    modelDefinition(model);

    const int localHostID = 0;

    // Choose the device with the most memory
    /*const int deviceID = chooseDeviceWithMostGlobalMemory();

    // Optimise block sizes
    Backends::CUDA::KernelBlockSize cudaBlockSize;
    optimizeBlockSize(deviceID, model, cudaBlockSize);*/
    Backends::CUDA::KernelBlockSize cudaBlockSize;
    const int deviceID = chooseOptimalDevice(model, cudaBlockSize);

     // Create backends
    Backends::SingleThreadedCPU cpuBackend(localHostID);
    Backends::CUDA backend(cudaBlockSize, localHostID, deviceID, cpuBackend);
    //Backends::SingleThreadedCPU backend(localHostID);

    // Generate code
    const auto moduleNames = generateCode(model, backend);

    // Create makefile to compile and link all generated modules
    std::ofstream makefile("generated_code/Makefile");
    generateMakefile(makefile, backend, moduleNames);

    return EXIT_SUCCESS;
}

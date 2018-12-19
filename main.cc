#include <bitset>
#include <fstream>
#include <iostream>
#include <string>

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

        /*if (model.isSynapseGroupDynamicsRequired(s.first)) {
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                groupSizes[KernelCalcSynapseDynamics].push_back(numSrcNeurons * maxConnections);
            }
            else {
                groupSizes[KernelCalcSynapseDynamics].push_back(numSrcNeurons * numTrgNeurons);
            }
        }*/

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

void optimizeBlockSize(int deviceID, const NNmodel &model, Backends::CUDA::KernelBlockSize &blockSize)
{
    using namespace Backends;

    // Calculate model group sizes
    std::vector<unsigned int> groupSizes[Backends::CUDA::KernelMax];
    calcGroupSizes(model, groupSizes);

    // obtaining ptxas info.
    CUdevice cuDevice;
    CUcontext cuContext;
    CHECK_CU_ERRORS(cuDeviceGet(&cuDevice, deviceID));
    CHECK_CU_ERRORS(cuCtxCreate(&cuContext, 0, cuDevice));

    // Bitset to mark which kernels are present and array of their attributes for each repetition
    std::bitset<CUDA::KernelMax> kernelExists(false);
    cudaFuncAttributes krnlAttr[2][CUDA::KernelMax];

    // Do two repititions with different candidate kernel size
    const size_t warpSize = 32;
    for(unsigned int rep = 0; rep < 2; rep++) {
        const size_t repBlockSize = warpSize * (rep + 1);
        LOGD_(LogOptimiser)  << "Generating code with block size:" << repBlockSize << std::endl;

        // Start with all group sizes set to warp size
        std::fill(blockSize.begin(), blockSize.end(), repBlockSize);

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
            for (unsigned int i = 0; i < CUDA::KernelMax; i++) {

                // If function is found
                CUfunction kern;
                CUresult res = cuModuleGetFunction(&kern, module, CUDA::KernelNames[i]);
                if (res == CUDA_SUCCESS) {
                    LOGD_(LogOptimiser) << "Function " << CUDA::KernelNames[i] << " found" << std::endl;

                    // Read it's attributes and mark it as existing
                    cudaFuncGetAttributesDriver(&krnlAttr[rep][i], kern);
                    kernelExists[i] = true;
                }
            }

            // Unload module
            CHECK_CU_ERRORS(cuModuleUnload(module));
        }
    }

    // Destroy context
    CHECK_CU_ERRORS(cuCtxDestroy(cuContext));
}
}   // Anonymous namespace

int main()
{
    // Initialise log channels, appending all to console
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init<LogDefault>(plog::debug, &consoleAppender);
    plog::init<LogBackend>(plog::debug, &consoleAppender);
    plog::init<LogOptimiser>(plog::debug, &consoleAppender);

    NNmodel model;
    modelDefinition(model);
    
    Backends::CUDA::KernelBlockSize cudaBlockSize{128, 128, 64, 64, 64, 32, 32};
    optimizeBlockSize(0, model, cudaBlockSize);

     // Create backends
    Backends::SingleThreadedCPU cpuBackend(0);
    Backends::CUDA backend(cudaBlockSize, 0, 0, cpuBackend);
    //Backends::SingleThreadedCPU backend(0);

    const auto moduleNames = generateCode(model, backend);

    // Create makefile to compile and link all generated modules
    std::ofstream makefile("generated_code/Makefile");
    generateMakefile(makefile, backend, moduleNames);

    return EXIT_SUCCESS;
}

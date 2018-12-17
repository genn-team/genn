#include <fstream>
#include <iostream>
#include <string>

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

using namespace CodeGenerator;

// Anonymous namespace
namespace
{
void modelDefinition(NNmodel &model)
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
}
}

int main()
{
    NNmodel model;
    modelDefinition(model);
    
    // Create backends
    Backends::SingleThreadedCPU cpuBackend(0);
    Backends::CUDA backend(128, 128, 64, 64, 32, 32, 0, cpuBackend);

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

    // Create makefile to compile and link all generated modules
    std::ofstream makefile("generated_code/Makefile");
    generateMakefile(makefile, backend, {"neuronUpdate", "synapseUpdate", "init", "runner"});

    return EXIT_SUCCESS;
}

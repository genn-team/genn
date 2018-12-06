#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <vector>

// GeNN includes
#include "codeGenUtils.h"
#include "codeStream.h"
#include "modelSpec.h"
#include "standardSubstitutions.h"

// NuGeNN includes
#include "substitution_stack.h"
#include "tee_stream.h"
#include "generateInit.h"
#include "generateNeuronUpdate.h"
#include "generateRunner.h"
#include "tempSubstitutions.h"
#include "backends/base.h"
#include "backends/cuda.h"
#include "backends/singleThreadedCPU.h"

using namespace CodeGenerator;


/*string &wCode,                  //!< the code string to work on
    const SynapseGroup *sg,         //!< the synapse group connecting the pre and postsynaptic neuron populations whose parameters might need to be substituted
    const string &preIdx,           //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx,          //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix,        //!< device prefix, "dd_" for GPU, nothing for CPU
    double dt,                      //!< simulation timestep (ms)
    const string &preVarPrefix,     //!< prefix to be used for presynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const string &preVarSuffix,     //!< suffix to be used for presynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
    const string &postVarPrefix,    //!< prefix to be used for postsynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const string &postVarSuffix*/

void generatePresynapticUpdateKernel(CodeStream &os, const NNmodel &model, const Backends::Base &backend)
{
    // Presynaptic update kernel
    backend.genPresynapticUpdate(os, model,
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            // code substitutions ----
            const WeightUpdateModels::Base *wu = sg.getWUModel();
            std::string code = wu->getEventThresholdConditionCode();
            applyWeightUpdateModelSubstitutions(code, sg, backend.getVarPrefix(),
                                                sg.getName() + "[" + baseSubs.getVarSubstitution("id_syn") + "]", "");
            neuron_substitutions_in_synaptic_code(code, &sg, baseSubs.getVarSubstitution("id_pre"),
                                                  baseSubs.getVarSubstitution("id_post"), backend.getVarPrefix(),
                                                  model.getDT());
            baseSubs.apply(code);
            code= ensureFtype(code, model.getPrecision());
            checkUnreplacedVariables(code, sg.getName() + " : evntThreshold");
            os << code;
        },
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            const WeightUpdateModels::Base *wu = sg.getWUModel();
            std::string code = wu->getSimCode(); //**TODO** pass through truespikeness
            baseSubs.apply(code);
    
            applyWeightUpdateModelSubstitutions(code, sg, backend.getVarPrefix(),
                                                sg.getName() + "[" + baseSubs.getVarSubstitution("id_syn") + "]", "");
            neuron_substitutions_in_synaptic_code(code, &sg, baseSubs.getVarSubstitution("id_pre"),
                                                  baseSubs.getVarSubstitution("id_post"), backend.getVarPrefix(),
                                                  model.getDT());
            baseSubs.apply(code);
            code= ensureFtype(code, model.getPrecision());
            checkUnreplacedVariables(code, sg.getName() + " : simCode");
            os << code;
        }
    );
}




int main()
{
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::defaultSparseConnectivityMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;
    initGeNN();

    NNmodel model;
    model.setDT(0.1);
    model.setName("izk_regimes");

    // Izhikevich model parameters
    NeuronModels::Izhikevich::ParamValues paramValues(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues initValues(-65.0, -20.0);

    WeightUpdateModels::StaticPulse::VarValues wumVar(0.5);

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(0.1); // 0 - prob

    // Create population of Izhikevich neurons
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 4, paramValues, initValues);
    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
        "Neurons", "Neurons",
        {}, wumVar,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    //syn->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);
    model.finalize();
    
    CodeStream output(std::cout);
    
    Backends::SingleThreadedCPU cpuBackend(0);
    Backends::CUDA backend(128, 128, 64, 64, 0, cpuBackend);
    
    generateNeuronUpdate(output, model, backend);
    generatePresynapticUpdateKernel(output, model, backend);
    generateInit(output, model, backend);

    std::stringstream definitions;
    std::stringstream runner;
    CodeStream definitionsStream(definitions);
    CodeStream runnerStream(runner);
    generateRunner(definitionsStream, runnerStream, model, backend, 0);
    
    std::cout << definitions.str() << std::endl;
    std::cout << runner.str() << std::endl;
    return EXIT_SUCCESS;
}

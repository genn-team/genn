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
#include "generateNeuronUpdate.h"
#include "generateRunner.h"
#include "tempSubstitutions.h"
#include "backends/base.h"
#include "backends/cuda.h"
#include "backends/singleThreadedCPU.h"

using namespace CodeGenerator;




void generatePresynapticUpdateKernel(CodeStream &os, const NNmodel &model, const Backends::Base &codeGenerator)
{
    // Presynaptic update kernel
    codeGenerator.genPresynapticUpdateKernel(os, model,
        [&codeGenerator, &model](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            // code substitutions ----
            const WeightUpdateModels::Base *wu = sg.getWUModel();
            std::string code = wu->getEventThresholdConditionCode();
            baseSubs.apply(code);
   
            applyWeightUpdateModelSubstitutions(code, sg, codeGenerator.getVarPrefix());
           
            code= ensureFtype(code, model.getPrecision());
            checkUnreplacedVariables(code, sg.getName() + " : evntThreshold");
            os << code;
        },
        [&codeGenerator, &model](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            const WeightUpdateModels::Base *wu = sg.getWUModel();
            std::string code = wu->getSimCode(); //**TODO** pass through truespikeness
            baseSubs.apply(code);
    
            applyWeightUpdateModelSubstitutions(code, sg, codeGenerator.getVarPrefix());

            code= ensureFtype(code, model.getPrecision());
            checkUnreplacedVariables(code, sg.getName() + " : simCode");
            os << code;
        }
    );
}
// ------------------------------------------------------------------------
template<typename I, typename M, typename Q>
void genInitNeuronVarCode(CodeStream &os, const Backends::Base &codeGenerator, const Substitutions &kernelSubs, const NewModels::Base::StringPairVec &vars,
                          size_t count, size_t numDelaySlots, const std::string &popName, const std::string &ftype,
                          I getVarInitialiser, M getVarMode, Q isVarQueueRequired)
{
    for (size_t k = 0; k < vars.size(); k++) {
        const auto &varInit = getVarInitialiser(k);
        const VarMode varMode = getVarMode(k);

        // If this variable has any initialisation code
        if(!varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);

            // Generate target-specific code to initialise variable
            codeGenerator.genVariableInit(os, varMode, count, kernelSubs, 
                [&codeGenerator, &vars, &varInit, &popName, &ftype, k, count, isVarQueueRequired, numDelaySlots]
                (CodeStream &os, Substitutions &varSubs)
                {
                    // If variable requires a queue
                    if (isVarQueueRequired(k)) {
                        // Generate initial value into temporary variable
                        os << vars[k].second << " initVal;" << std::endl;
                        varSubs.addVarSubstitution("value", "initVal");

                        std::string code = varInit.getSnippet()->getCode();
                        varSubs.apply(code);
                        applyVarInitSnippetSubstitutions(code, varInit);
                        code = ensureFtype(code, ftype);
                        checkUnreplacedVariables(code, "initVar");
                        os << code << std::endl;

                        // Copy this into all delay slots
                        os << "for (unsigned int d = 0; d < " << numDelaySlots << "; d++)";
                        {
                            CodeStream::Scope b(os);
                            os << codeGenerator.getVarPrefix() << vars[k].first << popName << "[(d * " << count << ") + i] = initVal;" << std::endl;
                        }
                    }
                    else {
                        varSubs.addVarSubstitution("value", vars[k].first + popName + "[" + varSubs.getVarSubstitution("id") + "]");
                                
                        std::string code = varInit.getSnippet()->getCode();
                        varSubs.apply(code);
                        applyVarInitSnippetSubstitutions(code, varInit);
                        code = ensureFtype(code, ftype);
                        checkUnreplacedVariables(code, "initVar");
                        os << code << std::endl;
                    }
                });
        }
    }
}
//------------------------------------------------------------------------
template<typename I, typename M>
void genInitNeuronVarCode(CodeStream &os, const Backends::Base &codeGenerator, const Substitutions &kernelSubs, const NewModels::Base::StringPairVec &vars,
                          size_t count, const std::string &popName, const std::string &ftype,
                          I getVarInitialiser, M getVarMode)
{
    genInitNeuronVarCode(os, codeGenerator, kernelSubs, vars, count, 0, popName, ftype, getVarInitialiser, getVarMode,
                         [](size_t){ return false; });
}

void genInitKernel(CodeStream &os, const NNmodel &model, const Backends::Base &codeGenerator)
{
    
    codeGenerator.genInitKernel(os, model,
        [&codeGenerator, &model](CodeStream &os, const NeuronGroup &ng, const Substitutions &kernelSubs)
        {
            // Initialise neuron variables
            genInitNeuronVarCode(os, codeGenerator, kernelSubs, ng.getNeuronModel()->getVars(), ng.getNumNeurons(), ng.getNumDelaySlots(),
                                 ng.getName(),  model.getPrecision(),
                                 [&ng](size_t i){ return ng.getVarInitialisers()[i]; },
                                 [&ng](size_t i){ return ng.getVarMode(i); },
                                 [&ng](size_t i){ return ng.isVarQueueRequired(i); });

            // Loop through incoming synaptic populations
            for(const auto &m : ng.getMergedInSyn()) {
                const auto *sg = m.first;

                // If this synapse group's input variable should be initialised on device
                // Generate target-specific code to initialise variable
                codeGenerator.genVariableInit(os, sg->getInSynVarMode(), ng.getNumNeurons(), kernelSubs, 
                    [&codeGenerator, &model, sg] (CodeStream &os, Substitutions &varSubs)
                    {
                        os << codeGenerator.getVarPrefix() << "inSyn" << sg->getPSModelTargetName() << "[" << varSubs.getVarSubstitution("id") << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                    });

                // If dendritic delays are required
                if(sg->isDendriticDelayRequired()) {
                    codeGenerator.genVariableInit(os, sg->getDendriticDelayVarMode(), ng.getNumNeurons(), kernelSubs, 
                        [&codeGenerator, &model, sg](CodeStream &os, Substitutions &varSubs)
                        {
                            os << "for (unsigned int d = 0; d < " << sg->getMaxDendriticDelayTimesteps() << "; d++)";
                            {
                                CodeStream::Scope b(os);
                                const std::string denDelayIndex = "(d * " + std::to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + ") + " + varSubs.getVarSubstitution("id");
                                os << codeGenerator.getVarPrefix() << "denDelay" << sg->getPSModelTargetName() << "[" << denDelayIndex << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                            }
                        });
                }

                // If postsynaptic model variables should be individual
                if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    genInitNeuronVarCode(os, codeGenerator, kernelSubs, sg->getPSModel()->getVars(), ng.getNumNeurons(), sg->getName(), model.getPrecision(),
                                         [sg](size_t i){ return sg->getPSVarInitialisers()[i]; },
                                         [sg](size_t i){ return sg->getPSVarMode(i); });
                }
            }
                    
            // Loop through incoming synaptic populations
            for(const auto *s : ng.getInSyn()) {
                genInitNeuronVarCode(os, codeGenerator, kernelSubs, s->getWUModel()->getPostVars(), ng.getNumNeurons(), s->getTrgNeuronGroup()->getNumDelaySlots(), s->getName(), model.getPrecision(),
                                     [&s](size_t i){ return s->getWUPostVarInitialisers()[i]; },
                                     [&s](size_t i){ return s->getWUPostVarMode(i); },
                                     [&s](size_t){ return (s->getBackPropDelaySteps() != NO_DELAY); });
            }

            // Loop through outgoing synaptic populations
            for(const auto *s : ng.getOutSyn()) {
                // **NOTE** number of delay slots is based on the source neuron (for simplicity) but whether delay is required is based on the synapse group
                genInitNeuronVarCode(os, codeGenerator, kernelSubs, s->getWUModel()->getPreVars(), ng.getNumNeurons(), s->getSrcNeuronGroup()->getNumDelaySlots(), s->getName(), model.getPrecision(),
                                     [&s](size_t i){ return s->getWUPreVarInitialisers()[i]; },
                                     [&s](size_t i){ return s->getWUPreVarMode(i); },
                                     [&s](size_t){ return (s->getDelaySteps() != NO_DELAY); });
            }

            // Loop through current sources
            os << "// current source variables" << std::endl;
            for (auto const *cs : ng.getCurrentSources()) {
                genInitNeuronVarCode(os, codeGenerator, kernelSubs, cs->getCurrentSourceModel()->getVars(), ng.getNumNeurons(), cs->getName(), model.getPrecision(),
                                     [cs](size_t i){ return cs->getVarInitialisers()[i]; },
                                     [cs](size_t i){ return cs->getVarMode(i); });
            }
        },
        [](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            //**TODO** think about $(id_pre) and $(id_post); and looping over sg.getSrcNeuronGroup()->getNumNeurons()
            // alternative to genVariableInit COULD solve both
        });
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

    // Create population of Izhikevich neurons
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 4, paramValues, initValues);
    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Syn", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
                                                                                                           "Neurons", "Neurons",
                                                                                                           {}, wumVar,
                                                                                                           {}, {});
    //syn->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);
    model.finalize();
    
    CodeStream output(std::cout);
    
    Backends::SingleThreadedCPU cpuBackend(0);
    Backends::CUDA backend(128, 128, 64, 0, cpuBackend);
    
    generateNeuronUpdate(output, model, backend);
    generatePresynapticUpdateKernel(output, model, backend);
    genInitKernel(output, model, backend);

    std::stringstream definitions;
    std::stringstream runner;
    CodeStream definitionsStream(definitions);
    CodeStream runnerStream(runner);
    generateRunner(definitionsStream, runnerStream, model, backend, 0);
    
    std::cout << definitions.str() << std::endl;
    std::cout << runner.str() << std::endl;
    return EXIT_SUCCESS;
}

#include "generateInit.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

// NuGeNN includes
#include "tempSubstitutions.h"
#include "substitution_stack.h"
#include "backends/base.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
template<typename I, typename M, typename Q>
void genInitNeuronVarCode(CodeStream &os, const CodeGenerator::Backends::Base &backend, const Substitutions &popSubs, const NewModels::Base::StringPairVec &vars,
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
            backend.genVariableInit(os, varMode, count, "id", popSubs,
                [&backend, &vars, &varInit, &popName, &ftype, k, count, isVarQueueRequired, numDelaySlots]
                (CodeStream &os, Substitutions &varSubs)
                {
                    // If variable requires a queue
                    if (isVarQueueRequired(k)) {
                        // Generate initial value into temporary variable
                        os << vars[k].second << " initVal;" << std::endl;
                        varSubs.addVarSubstitution("value", "initVal");

                        std::string code = varInit.getSnippet()->getCode();
                        varSubs.apply(code);
                        CodeGenerator::applyVarInitSnippetSubstitutions(code, varInit);
                        code = ensureFtype(code, ftype);
                        checkUnreplacedVariables(code, "initVar");
                        os << code << std::endl;

                        // Copy this into all delay slots
                        os << "for (unsigned int d = 0; d < " << numDelaySlots << "; d++)";
                        {
                            CodeStream::Scope b(os);
                            os << backend.getVarPrefix() << vars[k].first << popName << "[(d * " << count << ") + i] = initVal;" << std::endl;
                        }
                    }
                    else {
                        varSubs.addVarSubstitution("value", vars[k].first + popName + "[" + varSubs.getVarSubstitution("id") + "]");

                        std::string code = varInit.getSnippet()->getCode();
                        varSubs.apply(code);
                        CodeGenerator::applyVarInitSnippetSubstitutions(code, varInit);
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
void genInitNeuronVarCode(CodeStream &os, const CodeGenerator::Backends::Base &backend, const Substitutions &popSubs, const NewModels::Base::StringPairVec &vars,
                          size_t count, const std::string &popName, const std::string &ftype,
                          I getVarInitialiser, M getVarMode)
{
    genInitNeuronVarCode(os, backend, popSubs, vars, count, 0, popName, ftype, getVarInitialiser, getVarMode,
                         [](size_t){ return false; });
}
//------------------------------------------------------------------------
// Initialise one row of weight update model variables
void genInitWUVarCode(CodeStream &os, const CodeGenerator::Backends::Base &backend,const Substitutions &popSubs,
                      const SynapseGroup &sg, size_t count, const std::string &ftype)
{
    const auto vars = sg.getWUModel()->getVars();
    for (size_t k = 0; k < vars.size(); k++) {
        const auto &varInit = sg.getWUVarInitialisers()[k];
        const VarMode varMode = sg.getWUVarMode(k);

        // If this variable has any initialisation code
        if(!varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);

            // Generate target-specific code to initialise variable
            backend.genVariableInit(os, varMode, count, "id_post", popSubs,
                [&backend, &vars, &varInit, &sg, &ftype, k, count]
                (CodeStream &os, Substitutions &varSubs)
                {
                    varSubs.addVarSubstitution("value", vars[k].first + sg.getName() + "[(" + varSubs.getVarSubstitution("id_pre") + " * " + std::to_string(count) + ") + " + varSubs.getVarSubstitution("id_post") + "]");

                    std::string code = varInit.getSnippet()->getCode();
                    varSubs.apply(code);
                    CodeGenerator::applyVarInitSnippetSubstitutions(code, varInit);
                    code = ensureFtype(code, ftype);
                    checkUnreplacedVariables(code, "initVar");
                    os << code << std::endl;
                });
        }
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateInit(CodeStream &os, const NNmodel &model, const Backends::Base &backend)
{

    backend.genInit(os, model,
        [&backend, &model](CodeStream &os, const NeuronGroup &ng, Substitutions &popSubs)
        {
            /*// If delay is required and spike vars, spike event vars or spike times should be initialised on device
            if(n.second.isDelayRequired() &&
                ((shouldInitSpikeVar && n.second.isTrueSpikeRequired()) || shouldInitSpikeEventVar || shouldInitSpikeTimeVar))
            {
                // Build string to use for delayed variable index
                const std::string delayedIndex = "(i * " + std::to_string(n.second.getNumNeurons()) + ") + lid";

                // Loop through delay slots
                os << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++)";
                {
                    CodeStream::Scope b(os);

                    if(shouldInitSpikeVar && n.second.isTrueSpikeRequired()) {
                        os << "dd_glbSpk" << n.first << "[" << delayedIndex << "] = 0;" << std::endl;
                    }

                    if(shouldInitSpikeEventVar) {
                        os << "dd_glbSpkEvnt" << n.first << "[" << delayedIndex << "] = 0;" << std::endl;
                    }

                    if(shouldInitSpikeTimeVar) {
                        os << "dd_sT" << n.first << "[" << delayedIndex << "] = -TIME_MAX;" << std::endl;
                    }
                }
            }

            if(shouldInitSpikeVar && !(n.second.isTrueSpikeRequired() && n.second.isDelayRequired())) {
                os << "dd_glbSpk" << n.first << "[lid] = 0;" << std::endl;
            }

            if(!n.second.isDelayRequired()) {
                if(shouldInitSpikeEventVar) {
                    os << "dd_glbSpkEvnt" << n.first << "[lid] = 0;" << std::endl;
                }

                if(shouldInitSpikeTimeVar) {
                    os << "dd_sT" << n.first << "[lid] = -TIME_MAX;" << std::endl;
                }
            }*/
            // Initialise neuron variables
            genInitNeuronVarCode(os, backend, popSubs, ng.getNeuronModel()->getVars(), ng.getNumNeurons(), ng.getNumDelaySlots(),
                                 ng.getName(),  model.getPrecision(),
                                 [&ng](size_t i){ return ng.getVarInitialisers()[i]; },
                                 [&ng](size_t i){ return ng.getVarMode(i); },
                                 [&ng](size_t i){ return ng.isVarQueueRequired(i); });

            // Loop through incoming synaptic populations
            for(const auto &m : ng.getMergedInSyn()) {
                const auto *sg = m.first;

                // If this synapse group's input variable should be initialised on device
                // Generate target-specific code to initialise variable
                backend.genVariableInit(os, sg->getInSynVarMode(), ng.getNumNeurons(), "id", popSubs,
                    [&backend, &model, sg] (CodeStream &os, Substitutions &varSubs)
                    {
                        os << backend.getVarPrefix() << "inSyn" << sg->getPSModelTargetName() << "[" << varSubs.getVarSubstitution("id") << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                    });

                // If dendritic delays are required
                if(sg->isDendriticDelayRequired()) {
                    backend.genVariableInit(os, sg->getDendriticDelayVarMode(), ng.getNumNeurons(), "id", popSubs,
                        [&backend, &model, sg](CodeStream &os, Substitutions &varSubs)
                        {
                            os << "for (unsigned int d = 0; d < " << sg->getMaxDendriticDelayTimesteps() << "; d++)";
                            {
                                CodeStream::Scope b(os);
                                const std::string denDelayIndex = "(d * " + std::to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + ") + " + varSubs.getVarSubstitution("id");
                                os << backend.getVarPrefix() << "denDelay" << sg->getPSModelTargetName() << "[" << denDelayIndex << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                            }
                        });
                }

                // If postsynaptic model variables should be individual
                if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    genInitNeuronVarCode(os, backend, popSubs, sg->getPSModel()->getVars(), ng.getNumNeurons(), sg->getName(), model.getPrecision(),
                                         [sg](size_t i){ return sg->getPSVarInitialisers()[i]; },
                                         [sg](size_t i){ return sg->getPSVarMode(i); });
                }
            }

            // Loop through incoming synaptic populations
            for(const auto *s : ng.getInSyn()) {
                genInitNeuronVarCode(os, backend, popSubs, s->getWUModel()->getPostVars(), ng.getNumNeurons(), s->getTrgNeuronGroup()->getNumDelaySlots(), s->getName(), model.getPrecision(),
                                     [&s](size_t i){ return s->getWUPostVarInitialisers()[i]; },
                                     [&s](size_t i){ return s->getWUPostVarMode(i); },
                                     [&s](size_t){ return (s->getBackPropDelaySteps() != NO_DELAY); });
            }

            // Loop through outgoing synaptic populations
            for(const auto *s : ng.getOutSyn()) {
                // **NOTE** number of delay slots is based on the source neuron (for simplicity) but whether delay is required is based on the synapse group
                genInitNeuronVarCode(os, backend, popSubs, s->getWUModel()->getPreVars(), ng.getNumNeurons(), s->getSrcNeuronGroup()->getNumDelaySlots(), s->getName(), model.getPrecision(),
                                     [&s](size_t i){ return s->getWUPreVarInitialisers()[i]; },
                                     [&s](size_t i){ return s->getWUPreVarMode(i); },
                                     [&s](size_t){ return (s->getDelaySteps() != NO_DELAY); });
            }

            // Loop through current sources
            os << "// current source variables" << std::endl;
            for (auto const *cs : ng.getCurrentSources()) {
                genInitNeuronVarCode(os, backend, popSubs, cs->getCurrentSourceModel()->getVars(), ng.getNumNeurons(), cs->getName(), model.getPrecision(),
                                     [cs](size_t i){ return cs->getVarInitialisers()[i]; },
                                     [cs](size_t i){ return cs->getVarMode(i); });
            }
        },
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, Substitutions &popSubs)
        {
            // Loop through rows
            os << "for(unsigned int i = 0; i < " << sg.getSrcNeuronGroup()->getNumNeurons() << "; i++)";
            {
                popSubs.addVarSubstitution("id_pre", "i");
                genInitWUVarCode(os, backend, popSubs, sg, sg.getTrgNeuronGroup()->getNumNeurons(), model.getPrecision());

            }
            //**TODO** think about $(id_pre) and $(id_post); and looping over sg.getSrcNeuronGroup()->getNumNeurons()
            // alternative to genVariableInit COULD solve both
        },
        [](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            //**TODO** think about $(id_pre) and $(id_post); and looping over sg.getSrcNeuronGroup()->getNumNeurons()
            // alternative to genVariableInit COULD solve both
        });
}
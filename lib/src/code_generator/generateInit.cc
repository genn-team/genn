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
void genInitSpikeCount(CodeStream &os, const CodeGenerator::Backends::Base &backend, const Substitutions &popSubs,
                       const NeuronGroup &ng, bool spikeEvent)
{
    // Is initialisation required at all
    const bool initRequired = spikeEvent ? ng.isSpikeEventRequired() : true;
    if(initRequired) {
        // Get variable mode
        const VarMode varMode = spikeEvent ? ng.getSpikeEventVarMode() : ng.getSpikeVarMode();

        // Generate variable initialisation code
        backend.genPopVariableInit(os, varMode, popSubs,
            [&backend, &ng, spikeEvent] (CodeStream &os, Substitutions &)
            {
                // Get variable name
                const char *spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";

                // Is delay required
                const bool delayRequired = spikeEvent ?
                    ng.isDelayRequired() :
                    (ng.isTrueSpikeRequired() && ng.isDelayRequired());

                if(delayRequired) {
                    os << "for (unsigned int d = 0; d < " << ng.getNumDelaySlots() << "; d++)";
                    {
                        CodeStream::Scope b(os);
                        os << backend.getVarPrefix() << spikeCntPrefix << ng.getName() << "[d] = 0;" << std::endl;
                    }
                }
                else {
                    os << backend.getVarPrefix() << spikeCntPrefix << ng.getName() << "[0] = 0;" << std::endl;
                }
            });
    }

}
//--------------------------------------------------------------------------
void genInitSpikes(CodeStream &os, const CodeGenerator::Backends::Base &backend, const Substitutions &popSubs,
                   const NeuronGroup &ng, bool spikeEvent)
{
    // Is initialisation required at all
    const bool initRequired = spikeEvent ? ng.isSpikeEventRequired() : true;
    if(initRequired) {
        // Get variable mode
        const VarMode varMode = spikeEvent ? ng.getSpikeEventVarMode() : ng.getSpikeVarMode();

        // Generate variable initialisation code
        backend.genVariableInit(os, varMode, ng.getNumNeurons(), "id", popSubs,
            [&backend, &ng, spikeEvent] (CodeStream &os, Substitutions &varSubs)
            {
                // Get variable name
                const char *spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

                // Is delay required
                const bool delayRequired = spikeEvent ?
                    ng.isDelayRequired() :
                    (ng.isTrueSpikeRequired() && ng.isDelayRequired());

                if(delayRequired) {
                    os << "for (unsigned int d = 0; d < " << ng.getNumDelaySlots() << "; d++)";
                    {
                        CodeStream::Scope b(os);
                        os << backend.getVarPrefix() << spikePrefix << ng.getName() << "[(d * " << ng.getNumNeurons() << ") + " + varSubs.getVarSubstitution("id") + "] = 0;" << std::endl;
                    }
                }
                else {
                    os << backend.getVarPrefix() << spikePrefix << ng.getName() << "[" << varSubs.getVarSubstitution("id") << "] = 0;" << std::endl;
                }
            });
    }
}
//------------------------------------------------------------------------
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
                        CodeGenerator::applyVarInitSnippetSubstitutions(code, varInit);
                        varSubs.apply(code);
                        code = ensureFtype(code, ftype);
                        checkUnreplacedVariables(code, "initVar");
                        os << code << std::endl;

                        // Copy this into all delay slots
                        os << "for (unsigned int d = 0; d < " << numDelaySlots << "; d++)";
                        {
                            CodeStream::Scope b(os);
                            os << backend.getVarPrefix() << vars[k].first << popName << "[(d * " << count << ") + " + varSubs.getVarSubstitution("id") + "] = initVal;" << std::endl;
                        }
                    }
                    else {
                        varSubs.addVarSubstitution("value", backend.getVarPrefix() + vars[k].first + popName + "[" + varSubs.getVarSubstitution("id") + "]");

                        std::string code = varInit.getSnippet()->getCode();
                        CodeGenerator::applyVarInitSnippetSubstitutions(code, varInit);
                        varSubs.apply(code);
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
                    varSubs.addVarSubstitution("value", backend.getVarPrefix() + vars[k].first + sg.getName() + "[(" + varSubs.getVarSubstitution("id_pre") + " * " + std::to_string(count) + ") + " + varSubs.getVarSubstitution("id_post") + "]");

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
    os << "#include \"definitions.h\"" << std::endl;

    backend.genInit(os, model,
        // Local neuron group initialisation
        [&backend, &model](CodeStream &os, const NeuronGroup &ng, Substitutions &popSubs)
        {
            // Initialise spike counts
            genInitSpikeCount(os, backend, popSubs, ng, false);
            genInitSpikeCount(os, backend, popSubs, ng, true);

            // Initialise spikes
            genInitSpikes(os, backend, popSubs, ng, false);
            genInitSpikes(os, backend, popSubs, ng, true);

            // If spike times are required
            if(ng.isSpikeTimeRequired()) {
                // Generate variable initialisation code
                backend.genVariableInit(os, ng.getSpikeTimeVarMode(), ng.getNumNeurons(), "id", popSubs,
                    [&backend, &ng] (CodeStream &os, Substitutions &varSubs)
                    {
                        // Is delay required
                        if(ng.isDelayRequired()) {
                            os << "for (unsigned int d = 0; d < " << ng.getNumDelaySlots() << "; d++)";
                            {
                                CodeStream::Scope b(os);
                                os << backend.getVarPrefix() << "sT" << ng.getName() << "[(d * " << ng.getNumNeurons() << ") + " + varSubs.getVarSubstitution("id") + "] = -TIME_MAX;" << std::endl;
                            }
                        }
                        else {
                            os << backend.getVarPrefix() << "sT" << ng.getName() << "[" << varSubs.getVarSubstitution("id") << "] = -TIME_MAX;" << std::endl;
                        }
                    });
            }

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
        // Remote neuron group initialisation
        [&backend, &model](CodeStream &os, const NeuronGroup &ng, Substitutions &popSubs)
        {
            // Initialise spike counts and spikes
            genInitSpikeCount(os, backend, popSubs, ng, false);
            genInitSpikes(os, backend, popSubs, ng, false);
        },
        // Dense syanptic matrix variable initialisation
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, Substitutions &popSubs)
        {
            // Loop through rows
            os << "for(unsigned int i = 0; i < " << sg.getSrcNeuronGroup()->getNumNeurons() << "; i++)";
            {
                popSubs.addVarSubstitution("id_pre", "i");
                genInitWUVarCode(os, backend, popSubs, sg, sg.getTrgNeuronGroup()->getNumNeurons(), model.getPrecision());

            }
        },
        // Sparse synaptic matrix connectivity initialisation
        [&model](CodeStream &os, const SynapseGroup &sg, Substitutions &popSubs)
        {
            popSubs.addVarSubstitution("num_post", std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()));
            popSubs.addFuncSubstitution("endRow", 0, "break");

            // Initialise row building state variables and loop on generated code to initialise sparse connectivity
            const auto &connectInit = sg.getConnectivityInitialiser();
            os << "// Build sparse connectivity" << std::endl;
            for(const auto &a : connectInit.getSnippet()->getRowBuildStateVars()) {
                os << a.second.first << " " << a.first << " = " << a.second.second << ";" << std::endl;
            }
            os << "while(true)";
            {
                CodeStream::Scope b(os);

                // Apply substitutions
                std::string code = connectInit.getSnippet()->getRowBuildCode();
                applySparsConnectInitSnippetSubstitutions(code, sg);
                popSubs.apply(code);
                code = ensureFtype(code, model.getPrecision());
                checkUnreplacedVariables(code, "initSparseConnectivity");

                // Write out code
                os << code << std::endl;
            }
        },
        // Sparse synaptic matrix var initialisation
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, Substitutions &popSubs)
        {
            // If this synapse group has individual variables
            if(sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                // **TODO** need row length in string to go into count
                genInitWUVarCode(os, backend, popSubs, sg, 0, model.getPrecision());
            }
        });
}
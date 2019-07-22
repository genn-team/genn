#include "code_generator/generateInit.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "models.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/substitutions.h"
#include "code_generator/backendBase.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void applyVarInitSnippetSubstitutions(std::string &code, const Models::VarInit &varInit)
{
    using namespace CodeGenerator;

    // Substitue derived and standard parameters into init code
    DerivedParamNameIterCtx viDerivedParams(varInit.getSnippet()->getDerivedParams());
    value_substitutions(code, varInit.getSnippet()->getParamNames(), varInit.getParams());
    value_substitutions(code, viDerivedParams.nameBegin, viDerivedParams.nameEnd, varInit.getDerivedParams());
}
//--------------------------------------------------------------------------
void applySparsConnectInitSnippetSubstitutions(std::string &code, const SynapseGroupInternal &sg)
{
    using namespace CodeGenerator;

    const auto connectInit = sg.getConnectivityInitialiser();

    // Substitue derived and standard parameters into init code
    DerivedParamNameIterCtx viDerivedParams(connectInit.getSnippet()->getDerivedParams());
    EGPNameIterCtx viExtraGlobalParams(connectInit.getSnippet()->getExtraGlobalParams());
    value_substitutions(code, connectInit.getSnippet()->getParamNames(), connectInit.getParams());
    value_substitutions(code, viDerivedParams.nameBegin, viDerivedParams.nameEnd, connectInit.getDerivedParams());
    name_substitutions(code, "", viExtraGlobalParams.nameBegin, viExtraGlobalParams.nameEnd, sg.getName());
}
//--------------------------------------------------------------------------
void genInitSpikeCount(CodeGenerator::CodeStream &os, const CodeGenerator::BackendBase &backend,
                       const CodeGenerator::Substitutions &popSubs, const NeuronGroupInternal &ng, bool spikeEvent)
{
    using namespace CodeGenerator;

    // Is initialisation required at all
    const bool initRequired = spikeEvent ? ng.isSpikeEventRequired() : true;
    if(initRequired) {
        // Get spike location
        const VarLocation varLoc = spikeEvent ? ng.getSpikeEventLocation() : ng.getSpikeLocation();

        // Generate variable initialisation code
        backend.genPopVariableInit(os, varLoc, popSubs,
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
void genInitSpikes(CodeGenerator::CodeStream &os, const CodeGenerator::BackendBase &backend,
                   const CodeGenerator::Substitutions &popSubs, const NeuronGroupInternal &ng, bool spikeEvent)
{
    using namespace CodeGenerator;

    // Is initialisation required at all
    const bool initRequired = spikeEvent ? ng.isSpikeEventRequired() : true;
    if(initRequired) {
        // Get spike location
        const VarLocation varLoc = spikeEvent ? ng.getSpikeEventLocation() : ng.getSpikeLocation();

        // Generate variable initialisation code
        backend.genVariableInit(os, varLoc, ng.getNumNeurons(), "id", popSubs,
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
                        os << backend.getVarPrefix() << spikePrefix << ng.getName() << "[(d * " << ng.getNumNeurons() << ") + " + varSubs["id"] + "] = 0;" << std::endl;
                    }
                }
                else {
                    os << backend.getVarPrefix() << spikePrefix << ng.getName() << "[" << varSubs["id"] << "] = 0;" << std::endl;
                }
            });
    }
}
//------------------------------------------------------------------------
template<typename I, typename M, typename Q>
void genInitNeuronVarCode(CodeGenerator::CodeStream &os, const CodeGenerator::BackendBase &backend, const CodeGenerator::Substitutions &popSubs,
                          const Models::Base::VarVec &vars, size_t count, size_t numDelaySlots, const std::string &popName, const std::string &ftype,
                          I getVarInitialiser, M getVarLocation, Q isVarQueueRequired)
{
    using namespace CodeGenerator;

    for (size_t k = 0; k < vars.size(); k++) {
        const auto &varInit = getVarInitialiser(k);
        const VarLocation varLoc = getVarLocation(k);

        // If this variable has any initialisation code
        if(!varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);

            // Generate target-specific code to initialise variable
            backend.genVariableInit(os, varLoc, count, "id", popSubs,
                [&backend, &vars, &varInit, &popName, &ftype, k, count, isVarQueueRequired, numDelaySlots]
                (CodeStream &os, Substitutions &varSubs)
                {
                    // If variable requires a queue
                    if (isVarQueueRequired(k)) {
                        // Generate initial value into temporary variable
                        os << vars[k].type << " initVal;" << std::endl;
                        varSubs.addVarSubstitution("value", "initVal");

                        std::string code = varInit.getSnippet()->getCode();
                        applyVarInitSnippetSubstitutions(code, varInit);
                        varSubs.apply(code);
                        code = ensureFtype(code, ftype);
                        checkUnreplacedVariables(code, "initVar");
                        os << code << std::endl;

                        // Copy this into all delay slots
                        os << "for (unsigned int d = 0; d < " << numDelaySlots << "; d++)";
                        {
                            CodeStream::Scope b(os);
                            os << backend.getVarPrefix() << vars[k].name << popName << "[(d * " << count << ") + " + varSubs["id"] + "] = initVal;" << std::endl;
                        }
                    }
                    else {
                        varSubs.addVarSubstitution("value", backend.getVarPrefix() + vars[k].name + popName + "[" + varSubs["id"] + "]");

                        std::string code = varInit.getSnippet()->getCode();
                        applyVarInitSnippetSubstitutions(code, varInit);
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
void genInitNeuronVarCode(CodeGenerator::CodeStream &os, const CodeGenerator::BackendBase &backend, const CodeGenerator::Substitutions &popSubs,
                          const Models::Base::VarVec &vars, size_t count, const std::string &popName, const std::string &ftype,
                          I getVarInitialiser, M getVarMode)
{
    genInitNeuronVarCode(os, backend, popSubs, vars, count, 0, popName, ftype, getVarInitialiser, getVarMode,
                         [](size_t){ return false; });
}
//------------------------------------------------------------------------
// Initialise one row of weight update model variables
void genInitWUVarCode(CodeGenerator::CodeStream &os, const CodeGenerator::BackendBase &backend,
                      const CodeGenerator::Substitutions &popSubs, const SynapseGroupInternal &sg, const std::string &ftype)
{
    using namespace CodeGenerator;

    const auto vars = sg.getWUModel()->getVars();
    for (size_t k = 0; k < vars.size(); k++) {
        const auto &varInit = sg.getWUVarInitialisers().at(k);
        const VarLocation varLoc = sg.getWUVarLocation(k);

        // If this variable has any initialisation code
        if(!varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);

            // Generate target-specific code to initialise variable
            backend.genSynapseVariableRowInit(os, varLoc, sg, popSubs,
                [&backend, &vars, &varInit, &sg, &ftype, k]
                (CodeStream &os, Substitutions &varSubs)
                {
                    varSubs.addVarSubstitution("value", backend.getVarPrefix() + vars[k].name + sg.getName() + "[" + varSubs["id_syn"] +  "]");

                    std::string code = varInit.getSnippet()->getCode();
                    varSubs.apply(code);
                    applyVarInitSnippetSubstitutions(code, varInit);
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
void CodeGenerator::generateInit(CodeStream &os, const ModelSpecInternal &model, const BackendBase &backend,
                                 bool standaloneModules)
{
    if(standaloneModules) {
        os << "#include \"runner.cc\"" << std::endl;
    }
    else {
        os << "#include \"definitionsInternal.h\"" << std::endl;
    }

    backend.genInit(os, model,
        // Local neuron group initialisation
        [&backend, &model](CodeStream &os, const NeuronGroupInternal &ng, Substitutions &popSubs)
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
                backend.genVariableInit(os, ng.getSpikeTimeLocation(), ng.getNumNeurons(), "id", popSubs,
                    [&backend, &ng] (CodeStream &os, Substitutions &varSubs)
                    {
                        // Is delay required
                        if(ng.isDelayRequired()) {
                            os << "for (unsigned int d = 0; d < " << ng.getNumDelaySlots() << "; d++)";
                            {
                                CodeStream::Scope b(os);
                                os << backend.getVarPrefix() << "sT" << ng.getName() << "[(d * " << ng.getNumNeurons() << ") + " + varSubs["id"] + "] = -TIME_MAX;" << std::endl;
                            }
                        }
                        else {
                            os << backend.getVarPrefix() << "sT" << ng.getName() << "[" << varSubs["id"] << "] = -TIME_MAX;" << std::endl;
                        }
                    });
            }

            // Initialise neuron variables
            genInitNeuronVarCode(os, backend, popSubs, ng.getNeuronModel()->getVars(), ng.getNumNeurons(), ng.getNumDelaySlots(),
                                 ng.getName(),  model.getPrecision(),
                                 [&ng](size_t i){ return ng.getVarInitialisers().at(i); },
                                 [&ng](size_t i){ return ng.getVarLocation(i); },
                                 [&ng](size_t i){ return ng.isVarQueueRequired(i); });

            // Loop through incoming synaptic populations
            for(const auto &m : ng.getMergedInSyn()) {
                const auto *sg = m.first;

                // If this synapse group's input variable should be initialised on device
                // Generate target-specific code to initialise variable
                backend.genVariableInit(os, sg->getInSynLocation(), ng.getNumNeurons(), "id", popSubs,
                    [&backend, &model, sg] (CodeStream &os, Substitutions &varSubs)
                    {
                        os << backend.getVarPrefix() << "inSyn" << sg->getPSModelTargetName() << "[" << varSubs["id"] << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                    });

                // If dendritic delays are required
                if(sg->isDendriticDelayRequired()) {
                    backend.genVariableInit(os, sg->getDendriticDelayLocation(), ng.getNumNeurons(), "id", popSubs,
                        [&backend, &model, sg](CodeStream &os, Substitutions &varSubs)
                        {
                            os << "for (unsigned int d = 0; d < " << sg->getMaxDendriticDelayTimesteps() << "; d++)";
                            {
                                CodeStream::Scope b(os);
                                const std::string denDelayIndex = "(d * " + std::to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + ") + " + varSubs["id"];
                                os << backend.getVarPrefix() << "denDelay" << sg->getPSModelTargetName() << "[" << denDelayIndex << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                            }
                        });
                }

                // If postsynaptic model variables should be individual
                if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    genInitNeuronVarCode(os, backend, popSubs, sg->getPSModel()->getVars(), ng.getNumNeurons(), sg->getName(), model.getPrecision(),
                                         [sg](size_t i){ return sg->getPSVarInitialisers().at(i); },
                                         [sg](size_t i){ return sg->getPSVarLocation(i); });
                }
            }

            // Loop through incoming synaptic populations
            for(const auto *s : ng.getInSyn()) {
                genInitNeuronVarCode(os, backend, popSubs, s->getWUModel()->getPostVars(), ng.getNumNeurons(), s->getTrgNeuronGroup()->getNumDelaySlots(), s->getName(), model.getPrecision(),
                                     [&s](size_t i){ return s->getWUPostVarInitialisers().at(i); },
                                     [&s](size_t i){ return s->getWUPostVarLocation(i); },
                                     [&s](size_t){ return (s->getBackPropDelaySteps() != NO_DELAY); });
            }

            // Loop through outgoing synaptic populations
            for(const auto *s : ng.getOutSyn()) {
                // **NOTE** number of delay slots is based on the source neuron (for simplicity) but whether delay is required is based on the synapse group
                genInitNeuronVarCode(os, backend, popSubs, s->getWUModel()->getPreVars(), ng.getNumNeurons(), s->getSrcNeuronGroup()->getNumDelaySlots(), s->getName(), model.getPrecision(),
                                     [&s](size_t i){ return s->getWUPreVarInitialisers().at(i); },
                                     [&s](size_t i){ return s->getWUPreVarLocation(i); },
                                     [&s](size_t){ return (s->getDelaySteps() != NO_DELAY); });
            }

            // Loop through current sources
            os << "// current source variables" << std::endl;
            for (auto const *cs : ng.getCurrentSources()) {
                genInitNeuronVarCode(os, backend, popSubs, cs->getCurrentSourceModel()->getVars(), ng.getNumNeurons(), cs->getName(), model.getPrecision(),
                                     [cs](size_t i){ return cs->getVarInitialisers().at(i); },
                                     [cs](size_t i){ return cs->getVarLocation(i); });
            }
        },
        // Remote neuron group initialisation
        [&backend, &model](CodeStream &os, const NeuronGroupInternal &ng, Substitutions &popSubs)
        {
            // Initialise spike counts and spikes
            genInitSpikeCount(os, backend, popSubs, ng, false);
            genInitSpikes(os, backend, popSubs, ng, false);
        },
        // Dense syanptic matrix variable initialisation
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, Substitutions &popSubs)
        {
            // Loop through rows
            os << "for(unsigned int i = 0; i < " << sg.getSrcNeuronGroup()->getNumNeurons() << "; i++)";
            {
                CodeStream::Scope b(os);
                popSubs.addVarSubstitution("id_pre", "i");
                genInitWUVarCode(os, backend, popSubs, sg, model.getPrecision());

            }
        },
        // Sparse synaptic matrix connectivity initialisation
        [&model](CodeStream &os, const SynapseGroupInternal &sg, Substitutions &popSubs)
        {
            popSubs.addVarSubstitution("num_post", std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()));
            popSubs.addFuncSubstitution("endRow", 0, "break");

            // Initialise row building state variables and loop on generated code to initialise sparse connectivity
            const auto &connectInit = sg.getConnectivityInitialiser();
            os << "// Build sparse connectivity" << std::endl;
            for(const auto &a : connectInit.getSnippet()->getRowBuildStateVars()) {
                os << a.type << " " << a.name << " = " << a.value << ";" << std::endl;
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
        [&backend, &model](CodeStream &os, const SynapseGroupInternal &sg, Substitutions &popSubs)
        {
            genInitWUVarCode(os, backend, popSubs, sg, model.getPrecision());
        });
}

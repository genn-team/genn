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

#include "standardGeneratedSections.h"
#include "standardSubstitutions.h"

// NuGeNN includes
#include "substitution_stack.h"
#include "tee_stream.h"
#include "backends/backend.h"
#include "backends/cudaBackend.h"
#include "backends/singleThreadedCPUBackend.h"

using namespace CodeGenerator;

// **TODO** move into NeuronModels::Base
void applyNeuronModelSubstitutions(std::string &code, const NeuronGroup &ng, 
                                   const std::string &varPrefix, const std::string &varSuffix = "", const std::string &varExt = "")
{
    const NeuronModels::Base *nm = ng.getNeuronModel();

     // Create iteration context to iterate over the variables; derived and extra global parameters
    VarNameIterCtx nmVars(nm->getVars());
    DerivedParamNameIterCtx nmDerivedParams(nm->getDerivedParams());
    ExtraGlobalParamNameIterCtx nmExtraGlobalParams(nm->getExtraGlobalParams());

    name_substitutions(code, varPrefix, nmVars.nameBegin, nmVars.nameEnd, varSuffix, varExt);
    value_substitutions(code, nm->getParamNames(), ng.getParams());
    value_substitutions(code, nmDerivedParams.nameBegin, nmDerivedParams.nameEnd, ng.getDerivedParams());
    name_substitutions(code, "", nmExtraGlobalParams.nameBegin, nmExtraGlobalParams.nameEnd, ng.getName());
}

void applyPostsynapticModelSubstitutions(std::string &code, const SynapseGroup &sg, const std::string &varPrefix)
{
    const auto *psm = sg.getPSModel();

    // Create iterators to iterate over the names of the postsynaptic model's initial values
    VarNameIterCtx psmVars(psm->getVars());
    DerivedParamNameIterCtx psmDerivedParams(psm->getDerivedParams());

    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
        name_substitutions(code, varPrefix, psmVars.nameBegin, psmVars.nameEnd, sg.getName());
    }
    else {
        value_substitutions(code, psmVars.nameBegin, psmVars.nameEnd, sg.getPSConstInitVals());
    }
    value_substitutions(code, psm->getParamNames(), sg.getPSParams());

    // Create iterators to iterate over the names of the postsynaptic model's derived parameters
    value_substitutions(code, psmDerivedParams.nameBegin, psmDerivedParams.nameEnd, sg.getPSDerivedParams());
}

void applyWeightUpdateModelSubstitutions(std::string &code, const SynapseGroup &sg, const std::string &varPrefix)
{
    const auto *wu = sg.getWUModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());
    VarNameIterCtx wuPreVars(wu->getPreVars());
    VarNameIterCtx wuPostVars(wu->getPostVars());

    value_substitutions(code, sg.getWUModel()->getParamNames(), sg.getWUParams());
    value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
    name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg.getName());

    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        // **TODO** prePos = syn_address
        name_substitutions(code, varPrefix, wuVars.nameBegin, wuVars.nameEnd, sg.getName() + "[prePos]");
    }

    //**TODO** preIdx = id_pre and postIdx = id_post
    // neuron_substitutions_in_synaptic_code(eCode, &sg, preIdx, postIdx, devPrefix);
}

void applyVarInitSnippetSubstitutions(std::string &code, const NewModels::VarInit &varInit)
{
    // Substitue derived and standard parameters into init code
    DerivedParamNameIterCtx viDerivedParams(varInit.getSnippet()->getDerivedParams());
    value_substitutions(code, varInit.getSnippet()->getParamNames(), varInit.getParams());
    value_substitutions(code, viDerivedParams.nameBegin, viDerivedParams.nameEnd, varInit.getDerivedParams());
}

void generateNeuronUpdateKernel(CodeStream &os, const NNmodel &model, const Backends::Base &codeGenerator)
{
    // Neuron update kernel
    codeGenerator.genNeuronUpdateKernel(os, model,
        [&codeGenerator, &model](CodeStream &os, const NeuronGroup &ng, Substitutions &popSubs)
        {
            const NeuronModels::Base *nm = ng.getNeuronModel();

            // Generate code to copy neuron state into local variable
            // **TODO** basic behaviour could exist in NewModels::Base, NeuronModels::Base could add queuing logic
            for(const auto &v : nm->getVars()) {
                os << v.second << " l" << v.first << " = ";
                os << codeGenerator.getVarPrefix() << v.first << ng.getName() << "[";
                if (ng.isVarQueueRequired(v.first) && ng.isDelayRequired()) {
                    os << "(delaySlot * " << ng.getNumNeurons() << ") + ";
                }
                os << popSubs.getVarSubstitution("id") << "];" << std::endl;
            }

            if ((nm->getSimCode().find("$(sT)") != std::string::npos)
                || (nm->getThresholdConditionCode().find("$(sT)") != std::string::npos)
                || (nm->getResetCode().find("$(sT)") != std::string::npos)) 
            { 
                // load sT into local variable
                os << model.getPrecision() << " lsT= " << codeGenerator.getVarPrefix() << "sT" << ng.getName() << "[";
                if (ng.isDelayRequired()) {
                    os << "(delaySlot * " << ng.getNumNeurons() << ") + ";
                }
                os << popSubs.getVarSubstitution("id") << "];" << std::endl;
            }
            os << std::endl;

            if (!ng.getMergedInSyn().empty() || (nm->getSimCode().find("Isyn") != std::string::npos)) {
                os << model.getPrecision() << " Isyn = 0;" << std::endl;
            }
            
            popSubs.addVarSubstitution("Isyn", "Isyn");
            popSubs.addVarSubstitution("sT", "lsT");

            // Initialise any additional input variables supported by neuron model
            for (const auto &a : nm->getAdditionalInputVars()) {
                os << a.second.first << " " << a.first << " = " << a.second.second << ";" << std::endl;
            }

            for (const auto &m : ng.getMergedInSyn()) {
                const auto *sg = m.first;
                const auto *psm = sg->getPSModel();

                os << "// pull inSyn values in a coalesced access" << std::endl;
                os << model.getPrecision() << " linSyn" << sg->getName() << " = " << codeGenerator.getVarPrefix() << "inSyn" << sg->getName() << "[" << popSubs.getVarSubstitution("id") << "];" << std::endl;

                // If dendritic delay is required
                if (sg->isDendriticDelayRequired()) {
                    // Get reference to dendritic delay buffer input for this timestep
                    os << model.getPrecision() << " &denDelay" << sg->getName() << " = " << codeGenerator.getVarPrefix() << "denDelay" + sg->getName() + "[" + sg->getDendriticDelayOffset("") + "n];" << std::endl;

                    // Add delayed input from buffer into inSyn
                    os << "linSyn" + sg->getName() + " += denDelay" << sg->getName() << ";" << std::endl;

                    // Zero delay buffer slot
                    os << "denDelay" << sg->getName() << " = " << model.scalarExpr(0.0) << ";" << std::endl;
                }

                // If synapse group has individual postsynaptic variables, also pull these in a coalesced access
                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    // **TODO** base behaviour from NewModels::Base
                    for (const auto &v : psm->getVars()) {
                        os << v.second << " lps" << v.first << sg->getName();
                        os << " = " << codeGenerator.getVarPrefix() << v.first << sg->getName() << "[n];" << std::endl;
                    }
                }

                Substitutions inSynSubs(&popSubs);
                inSynSubs.addVarSubstitution("inSyn", "linSyn" + sg->getName());
                
                // Apply substitutions to current converter code
                string psCode = psm->getApplyInputCode();
                inSynSubs.apply(psCode);
         
                applyNeuronModelSubstitutions(psCode, ng, "l");
                applyPostsynapticModelSubstitutions(psCode, *sg, "lps");
                
                psCode = ensureFtype(psCode, model.getPrecision());
                checkUnreplacedVariables(psCode, sg->getName() + " : postSyntoCurrent");

                if (!psm->getSupportCode().empty()) {
                    os << CodeStream::OB(29) << " using namespace " << sg->getName() << "_postsyn;" << std::endl;
                }
                os << psCode << std::endl;
                if (!psm->getSupportCode().empty()) {
                    os << CodeStream::CB(29) << " // namespace bracket closed" << std::endl;
                }
            }

            if (!nm->getSupportCode().empty()) {
                os << " using namespace " << ng.getName() << "_neuron;" << std::endl;
            }

            string thCode = nm->getThresholdConditionCode();
            if (thCode.empty()) { // no condition provided
                cerr << "Warning: No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << ng.getName() << "\" was provided. There will be no spikes detected in this population!" << endl;
            }
            else {
                os << "// test whether spike condition was fulfilled previously" << std::endl;
                popSubs.apply(thCode);
                
                applyNeuronModelSubstitutions(thCode, ng, "l");
                
                thCode= ensureFtype(thCode, model.getPrecision());
                checkUnreplacedVariables(thCode, ng.getName() + " : thresholdConditionCode");
                                
                if (GENN_PREFERENCES::autoRefractory) {
                    os << "const bool oldSpike= (" << thCode << ");" << std::endl;
                }
            }

            os << "// calculate membrane potential" << std::endl;
            string sCode = nm->getSimCode();
            popSubs.apply(sCode);

            applyNeuronModelSubstitutions(sCode, ng, "l");
            
            sCode = ensureFtype(sCode, model.getPrecision());
            checkUnreplacedVariables(sCode, ng.getName() + " : neuron simCode");
                            
            os << sCode << std::endl;

            // look for spike type events first.
            if (ng.isSpikeEventRequired()) {
                // Create local variable
                os << "bool spikeLikeEvent = false;" << std::endl;

                // Loop through outgoing synapse populations that will contribute to event condition code
                for(const auto &spkEventCond : ng.getSpikeEventCondition()) {
                    // Replace of parameters, derived parameters and extraglobalsynapse parameters
                    string eCode = spkEventCond.first;

                    // code substitutions ----
                    popSubs.apply(eCode);

                    applyNeuronModelSubstitutions(eCode, ng, "l", "", "_pre");
                   
                    eCode = ensureFtype(eCode, model.getPrecision());
                    checkUnreplacedVariables(eCode, ng.getName() + " : neuronSpkEvntCondition");

                    // Open scope for spike-like event test
                    os << CodeStream::OB(31);

                    // Use synapse population support code namespace if required
                    if (!spkEventCond.second.empty()) {
                        os << " using namespace " << spkEventCond.second << ";" << std::endl;
                    }

                    // Combine this event threshold test with
                    os << "spikeLikeEvent |= (" << eCode << ");" << std::endl;

                    // Close scope for spike-like event test
                    os << CodeStream::CB(31);
                }

                os << "// register a spike-like event" << std::endl;
                os << "if (spikeLikeEvent)";
                {
                    CodeStream::Scope b(os);
                    codeGenerator.genEmitSpikeLikeEvent(os, model, ng, popSubs);
                }
            }

            // test for true spikes if condition is provided
            if (!thCode.empty()) {
                os << "// test for and register a true spike" << std::endl;
                if (GENN_PREFERENCES::autoRefractory) {
                    os << "if ((" << thCode << ") && !(oldSpike))";
                }
                else {
                    os << "if (" << thCode << ")";
                }
                {
                    CodeStream::Scope b(os);

                    codeGenerator.genEmitTrueSpike(os, model, ng, popSubs);

                    // add after-spike reset if provided
                    if (!nm->getResetCode().empty()) {
                        string rCode = nm->getResetCode();
                        popSubs.apply(rCode);
             
                        applyNeuronModelSubstitutions(rCode, ng, "l");
                        
                        rCode = ensureFtype(rCode, model.getPrecision());
                        checkUnreplacedVariables(rCode, ng.getName() + " : resetCode");

                        os << "// spike reset code" << std::endl;
                        os << rCode << std::endl;
                    }
                }
            }

            // store the defined parts of the neuron state into the global state variables dd_V etc
            for(const auto &v : nm->getVars()) {
                os << codeGenerator.getVarPrefix() << v.first << ng.getName() << "[";

                if (ng.isVarQueueRequired(v.first) && ng.isDelayRequired()) {
                    os << "readDelayOffset + ";
                }
                os << popSubs.getVarSubstitution("id") << "] = l" << v.first << ";" << std::endl;
            }

            for (const auto &m : ng.getMergedInSyn()) {
                const auto *sg = m.first;
                const auto *psm = sg->getPSModel();

                Substitutions inSynSubs(&popSubs);
                inSynSubs.addVarSubstitution("inSyn", "linSyn" + sg->getName());

                string pdCode = psm->getDecayCode();
                inSynSubs.apply(pdCode);
     
                applyNeuronModelSubstitutions(pdCode, ng, "l");
                applyPostsynapticModelSubstitutions(pdCode, *sg, "lps");
                
                pdCode = ensureFtype(pdCode, model.getPrecision());
                checkUnreplacedVariables(pdCode, sg->getName() + " : postSynDecay");

                os << "// the post-synaptic dynamics" << std::endl;
                if (!psm->getSupportCode().empty()) {
                    os << CodeStream::OB(29) << " using namespace " << sg->getName() << "_postsyn;" << std::endl;
                }
                os << pdCode << std::endl;
                if (!psm->getSupportCode().empty()) {
                    os << CodeStream::CB(29) << " // namespace bracket closed" << endl;
                }

                os << codeGenerator.getVarPrefix() << "inSyn"  << sg->getName() << "[" << inSynSubs.getVarSubstitution("id") << "] = linSyn" << sg->getName() << ";" << std::endl;
                for (const auto &v : psm->getVars()) {
                    os << codeGenerator.getVarPrefix() << v.first << sg->getName() << "[n]" << " = lps" << v.first << sg->getName() << ";" << std::endl;
                }
            }
        }
    );    
}

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

void genDefinitions(CodeStream &definitions, CodeStream &runner, const NNmodel &model, const Backends::Base &codeGenerator, int localHostID)
{
    // Create codestreams to generate different sections of runner
    std::stringstream runnerVarDeclStream;
    std::stringstream runnerAllocStream;
    std::stringstream runnerFreeStream;
    CodeStream runnerVarDecl(runnerVarDeclStream);
    CodeStream runnerAlloc(runnerAllocStream);
    CodeStream runnerFree(runnerFreeStream);

    // Create a teestream to allow simultaneous writing to both streams
    TeeStream allStreams(definitions, runnerVarDecl, runnerAlloc, runnerFree);

    // Begin extern C block around variable declarations
    if(GENN_PREFERENCES::buildSharedLibrary) {
        runnerVarDecl << "extern \"C\" {" << std::endl;
    }

    // In windows making variables extern isn't enough to export then as DLL symbols - you need to add __declspec(dllexport)
#ifdef _WIN32
    const std::string varExportPrefix = GENN_PREFERENCES::buildSharedLibrary ? "__declspec(dllexport) extern" : "extern";
#else
    const std::string varExportPrefix = "extern";
#endif

    //---------------------------------
    // REMOTE NEURON GROUPS
    allStreams << "// ------------------------------------------------------------------------" << std::endl;
    allStreams << "// remote neuron groups" << std::endl;
    allStreams << std::endl;

    // Loop through remote neuron groups
    for(const auto &n : model.getRemoteNeuronGroups()) {
        // Write macro so whether a neuron group is remote or not can be determined at compile time
        // **NOTE** we do this for REMOTE groups so #ifdef GROUP_NAME_REMOTE is backward compatible
        definitions << "#define " << n.first << "_REMOTE" << std::endl;

        // If this neuron group has outputs to local host
        if(n.second.hasOutputToHost(localHostID)) {
            // Check that, whatever variable mode is set for these variables,
            // they are instantiated on host so they can be copied using MPI
            if(!(n.second.getSpikeVarMode() & VarLocation::HOST)) {
                gennError("Remote neuron group '" + n.first + "' has its spike variable mode set so it is not instantiated on the host - this is not supported");
            }

            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpkCnt"+n.first, n.second.getSpikeVarMode(),
                                   n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1);
            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpk"+n.first, n.second.getSpikeVarMode(),
                                   n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());
        }
    }
    allStreams << std::endl;

    //---------------------------------
    // LOCAL NEURON VARIABLES
    allStreams << "// ------------------------------------------------------------------------" << std::endl;
    allStreams << "// local neuron groups" << std::endl;
    allStreams << std::endl;

    for(const auto &n : model.getLocalNeuronGroups()) {
        codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpkCnt"+n.first, n.second.getSpikeVarMode(),
                               n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1);
        codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpk"+n.first, n.second.getSpikeVarMode(),
                               n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());
        
        if (n.second.isSpikeEventRequired()) {
            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpkCntEvnt"+n.first, n.second.getSpikeEventVarMode(),
                                   n.second.getNumDelaySlots());
            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpkEvnt"+n.first, n.second.getSpikeEventVarMode(),
                                   n.second.getNumNeurons() * n.second.getNumDelaySlots());
        }
        if (n.second.isDelayRequired()) {
            //**FIXME**
            definitions << varExportPrefix << " unsigned int spkQuePtr" << n.first << ";" << std::endl;
            runnerVarDecl << "unsigned int spkQuePtr" << n.first << ";" << std::endl;
#ifndef CPU_ONLY
            runnerVarDecl << "__device__ volatile unsigned int dd_spkQuePtr" << n.first << ";" << std::endl;
#endif
        }
        if (n.second.isSpikeTimeRequired()) {
            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, model.getTimePrecision()+" *", "sT"+n.first, n.second.getSpikeTimeVarMode(),
                                   n.second.getNumNeurons() * n.second.getNumDelaySlots());
        }
#ifndef CPU_ONLY
        //**FIXME**
        if(n.second.isSimRNGRequired()) {
            definitions << "extern curandState *d_rng" << n.first << ";" << std::endl;
            runnerVarDecl << "curandState *d_rng" << n.first << ";" << std::endl;
            runnerVarDecl << "__device__ curandState *dd_rng" << n.first << ";" << std::endl;
        }
#endif
        auto neuronModel = n.second.getNeuronModel();
        for(auto const &v : neuronModel->getVars()) {
            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, v.second, v.first + n.first, n.second.getVarMode(v.first),
                                   n.second.isVarQueueRequired(v.first) ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());
        }
        for(auto const &v : neuronModel->getExtraGlobalParams()) {
            definitions << "extern " << v.second << " " << v.first + n.first << ";" << std::endl;
            runnerVarDecl << v.second << " " <<  v.first << n.first << ";" << std::endl;
        }

        if(!n.second.getCurrentSources().empty()) {
            allStreams << "// current source variables" << std::endl;
        }
        for (auto const *cs : n.second.getCurrentSources()) {
            auto csModel = cs->getCurrentSourceModel();
            for(auto const &v : csModel->getVars()) {
                codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, v.second, v.first + cs->getName(), cs->getVarMode(v.first),
                                       n.second.getNumNeurons());
            }
            for(auto const &v : csModel->getExtraGlobalParams()) {
                definitions << "extern " << v.second << " " <<  v.first << cs->getName() << ";" << std::endl;
                runnerVarDecl << v.second << " " <<  v.first << cs->getName() << ";" << std::endl;
            }
        }
    }
    allStreams << std::endl;

    //----------------------------------
    // POSTSYNAPTIC VARIABLES
    allStreams << "// ------------------------------------------------------------------------" << std::endl;
    allStreams << "// postsynaptic variables" << std::endl;
    allStreams << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        // Loop through incoming synaptic populations
        for(const auto &m : n.second.getMergedInSyn()) {
            const auto *sg = m.first;

            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, model.getPrecision(), "inSyn" + sg->getPSModelTargetName(), sg->getInSynVarMode(),
                                   sg->getTrgNeuronGroup()->getNumNeurons());

            if (sg->isDendriticDelayRequired()) {
                codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, model.getPrecision(), "denDelay" + sg->getPSModelTargetName(), sg->getDendriticDelayVarMode(),
                                       sg->getMaxDendriticDelayTimesteps() * sg->getTrgNeuronGroup()->getNumNeurons());
                
                //**FIXME**
                runnerVarDecl << varExportPrefix << " unsigned int denDelayPtr" << sg->getPSModelTargetName() << ";" << std::endl;
                runnerVarDecl << "unsigned int denDelayPtr" << sg->getPSModelTargetName() << ";" << std::endl;
#ifndef CPU_ONLY
                runnerVarDecl << "__device__ volatile unsigned int dd_denDelayPtr" << sg->getPSModelTargetName() << ";" << std::endl;
#endif
            }

            if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                for(const auto &v : sg->getPSModel()->getVars()) {
                    codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, v.second, v.first + sg->getPSModelTargetName(), sg->getPSVarMode(v.first),
                                           sg->getTrgNeuronGroup()->getNumNeurons());
                }
            }
        }
    }
    allStreams << std::endl;

    //----------------------------------
    // SYNAPSE VARIABLE
    allStreams << "// ------------------------------------------------------------------------" << std::endl;
    allStreams << "// synapse variables" << std::endl;
    allStreams << std::endl;
    for(const auto &s : model.getLocalSynapseGroups()) {
        const auto *wu = s.second.getWUModel();

        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            const size_t gpSize = ((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * (size_t)s.second.getTrgNeuronGroup()->getNumNeurons()) / 32 + 1;
            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                    "uint32_t", "gp" + s.first, s.second.getSparseConnectivityVarMode(), gpSize);
        }
        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
            const VarMode varMode = s.second.getSparseConnectivityVarMode();
            const size_t size = s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getMaxConnections();
            
            // Row lengths
            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                   "unsigned int", "rowLength" + s.first, varMode, s.second.getSrcNeuronGroup()->getNumNeurons());
            
            // Target indices
            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                   "unsigned int", "ind" + s.first, varMode, size);
            
            // **TODO** remap is not always required
            if(!s.second.getWUModel()->getSynapseDynamicsCode().empty()) {
                // Allocate synRemap
                // **THINK** this is over-allocating
                codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                       "unsigned int", "synRemap" + s.first, varMode, size + 1);
            }
            
            // **TODO** remap is not always required
            if(!s.second.getWUModel()->getLearnPostCode().empty()) {
                const size_t postSize = (size_t)s.second.getTrgNeuronGroup()->getNumNeurons() * (size_t)s.second.getMaxSourceConnections();
                
                // Allocate column lengths
                codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                       "unsigned int", "colLength" + s.first, varMode, s.second.getTrgNeuronGroup()->getNumNeurons());
                
                // Allocate remap
                codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                       "unsigned int", "remap" + s.first, varMode, postSize);
                
            }
            
            // If weight update variables should be individual
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                for(const auto &v : wu->getVars()) {
                    codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                           v.second, v.first + s.first, s.second.getWUVarMode(v.first), size);
                }
            }
        }
        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) {
            const size_t size = s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumNeurons();
            
            // If weight update variables should be individual
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                for(const auto &v : wu->getVars()) {
                    codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                           v.second, v.first + s.first, s.second.getWUVarMode(v.first), size);
                }
            }

        }

         const size_t preSize = (s.second.getDelaySteps() == NO_DELAY)
                ? s.second.getSrcNeuronGroup()->getNumNeurons()
                : s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getSrcNeuronGroup()->getNumDelaySlots();
        for(const auto &v : wu->getPreVars()) {
            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                   v.second, v.first + s.first, s.second.getWUPreVarMode(v.first), preSize);
        }

        const size_t postSize = (s.second.getBackPropDelaySteps() == NO_DELAY)
                ? s.second.getTrgNeuronGroup()->getNumNeurons()
                : s.second.getTrgNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumDelaySlots();
        for(const auto &v : wu->getPostVars()) {
            codeGenerator.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                   v.second, v.first + s.first, s.second.getWUPostVarMode(v.first), postSize);
        }

        for(const auto &v : wu->getExtraGlobalParams()) {
            definitions << "extern " << v.second << " " << v.first + s.first << ";" << std::endl;
            runnerVarDecl << v.second << " " <<  v.first << s.first << ";" << std::endl;
        }

        for(auto const &p : s.second.getConnectivityInitialiser().getSnippet()->getExtraGlobalParams()) {
            definitions << "extern " << p.second << " initSparseConn" << p.first + s.first << ";" << std::endl;
            runnerVarDecl << p.second << " initSparseConn" << p.first + s.first << ";" << std::endl;
        }
    }
    allStreams << std::endl;
    // End extern C block around variable declarations
    if(GENN_PREFERENCES::buildSharedLibrary) {
        runnerVarDecl << "}\t// extern \"C\"" << std::endl;
    }
    
    // Write variable declarations to runner
    runner << runnerVarDeclStream.str();
    
    // ---------------------------------------------------------------------
    // Function for setting the CUDA device and the host's global variables.
    // Also estimates memory usage on device ...
    runner << "void allocateMem()";
    {
        CodeStream::Scope b(runner);
#ifndef CPU_ONLY
        // **TODO** move to code generator
        runner << "CHECK_CUDA_ERRORS(cudaSetDevice(" << theDevice << "));" << std::endl;

        // If the model requires zero-copy
        if(model.zeroCopyInUse()) {
            // If device doesn't support mapping host memory error
            if(!deviceProp[theDevice].canMapHostMemory) {
                gennError("Device does not support mapping CPU host memory!");
            }

            // set appropriate device flags
            runner << "CHECK_CUDA_ERRORS(cudaSetDeviceFlags(cudaDeviceMapHost));" << std::endl;
        }

        // If RNG is required, allocate memory for global philox RNG
        if(model.isDeviceRNGRequired()) {
            //allocate_device_variable(os, "curandStatePhilox4_32_10_t", "rng", VarMode::LOC_DEVICE_INIT_DEVICE, 1);
        }
#endif
        runner << runnerAllocStream.str();
    }
    runner << std::endl;

    // ------------------------------------------------------------------------
    // Function to free all global memory structures
    runner << "void freeMem()";
    {
        CodeStream::Scope b(runner);

        runner << runnerFreeStream.str();
    }
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
    
    generateNeuronUpdateKernel(output, model, backend);
    generatePresynapticUpdateKernel(output, model, backend);
    genInitKernel(output, model, backend);

    std::stringstream definitions;
    std::stringstream runner;
    CodeStream definitionsStream(definitions);
    CodeStream runnerStream(runner);
    genDefinitions(definitionsStream, runnerStream, model, backend, 0);
    
    std::cout << definitions.str() << std::endl;
    std::cout << runner.str() << std::endl;
    return EXIT_SUCCESS;
}

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "codeGenUtils.h"
#include "codeStream.h"
#include "modelSpec.h"

#include "standardGeneratedSections.h"
#include "standardSubstitutions.h"

// NuGeNN includes
#include "code_generator.h"
#include "cuda_code_generator.h"
#include "substitution_stack.h"

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

void generateNeuronUpdateKernel(CodeStream &os, const NNmodel &model, const CodeGenerator::Base &codeGenerator)
{
    // Neuron update kernel
    codeGenerator.genNeuronUpdateKernel(os, model,
        [](CodeStream &os, const CodeGenerator::Base &codeGenerator, const NNmodel &model, const NeuronGroup &ng, const Substitutions &baseSubs)
        {
            Substitutions subs(&baseSubs);
            const NeuronModels::Base *nm = ng.getNeuronModel();

            // Generate code to copy neuron state into local variable
            // **TODO** basic behaviour could exist in NewModels::Base, NeuronModels::Base could add queuing logic
            for(const auto &v : nm->getVars()) {
                os << v.second << " l" << v.first << " = ";
                os << codeGenerator.getVarPrefix() << v.first << ng.getName() << "[";
                if (ng.isVarQueueRequired(v.first) && ng.isDelayRequired()) {
                    os << "(delaySlot * " << ng.getNumNeurons() << ") + ";
                }
                os << subs.getVarSubstitution("id") << "];" << std::endl;
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
                os << subs.getVarSubstitution("id") << "];" << std::endl;
            }
            os << std::endl;

            if (!ng.getMergedInSyn().empty() || (nm->getSimCode().find("Isyn") != std::string::npos)) {
                os << model.getPrecision() << " Isyn = 0;" << std::endl;
            }

            
            subs.addVarSubstitution("Isyn", "Isyn");
            subs.addVarSubstitution("sT", "lsT");

            // Initialise any additional input variables supported by neuron model
            for (const auto &a : nm->getAdditionalInputVars()) {
                os << a.second.first << " " << a.first << " = " << a.second.second << ";" << std::endl;
            }

            for (const auto &m : ng.getMergedInSyn()) {
                const auto *sg = m.first;
                const auto *psm = sg->getPSModel();

                os << "// pull inSyn values in a coalesced access" << std::endl;
                os << model.getPrecision() << " linSyn" << sg->getName() << " = " << codeGenerator.getVarPrefix() << "inSyn" << sg->getName() << "[" << subs.getVarSubstitution("id") << "];" << std::endl;

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

                Substitutions subs(&subs);
                subs.addVarSubstitution("inSyn", "linSyn" + sg->getName());
                
                // Apply substitutions to current converter code
                string psCode = psm->getApplyInputCode();
                subs.apply(psCode);
                functionSubstitutions(psCode, model.getPrecision(), codeGenerator.getFunctions());

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
                subs.apply(thCode);
                functionSubstitutions(thCode, model.getPrecision(), codeGenerator.getFunctions());

                applyNeuronModelSubstitutions(thCode, ng, "l");
                
                thCode= ensureFtype(thCode, model.getPrecision());
                checkUnreplacedVariables(thCode, ng.getName() + " : thresholdConditionCode");
                                
                if (GENN_PREFERENCES::autoRefractory) {
                    os << "const bool oldSpike= (" << thCode << ");" << std::endl;
                }
            }

            os << "// calculate membrane potential" << std::endl;
            string sCode = nm->getSimCode();
            subs.apply(sCode);

            functionSubstitutions(sCode, model.getPrecision(), codeGenerator.getFunctions());

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
                    subs.apply(eCode);

                    functionSubstitutions(eCode, model.getPrecision(), codeGenerator.getFunctions());
                    
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
                    codeGenerator.genEmitSpikeLikeEvent(os, model, ng, subs);
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

                    codeGenerator.genEmitTrueSpike(os, model, ng, subs);

                    // add after-spike reset if provided
                    if (!nm->getResetCode().empty()) {
                        string rCode = nm->getResetCode();
                        subs.apply(rCode);
                        functionSubstitutions(rCode, model.getPrecision(), codeGenerator.getFunctions());

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
                if (ng.isVarQueueRequired(v.first)) {
                    os << codeGenerator.getVarPrefix() << v.first << ng.getName() << "[" << ng.getQueueOffset(codeGenerator.getVarPrefix()) << subs.getVarSubstitution("id") << "] = l" << v.first << ";" << std::endl;
                }
                else {
                    os << codeGenerator.getVarPrefix() << v.first << ng.getName() << "[" << subs.getVarSubstitution("id") << "] = l" << v.first << ";" << std::endl;
                }
            }

            for (const auto &m : ng.getMergedInSyn()) {
                const auto *sg = m.first;
                const auto *psm = sg->getPSModel();

                Substitutions subs(&subs);
                subs.addVarSubstitution("inSyn", "linSyn" + sg->getName());

                string pdCode = psm->getDecayCode();
                subs.apply(pdCode);
                functionSubstitutions(pdCode, model.getPrecision(), codeGenerator.getFunctions());

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

                os << codeGenerator.getVarPrefix() << "inSyn"  << sg->getName() << "[" << subs.getVarSubstitution("id") << "] = linSyn" << sg->getName() << ";" << std::endl;
                for (const auto &v : psm->getVars()) {
                    os << codeGenerator.getVarPrefix() << v.first << sg->getName() << "[n]" << " = lps" << v.first << sg->getName() << ";" << std::endl;
                }
            }
        }
    );    
}

void generatePresynapticUpdateKernel(CodeStream &os, const NNmodel &model, const CodeGenerator::Base &codeGenerator)
{
    // Presynaptic update kernel
    codeGenerator.genPresynapticUpdateKernel(os, model,
        [](CodeStream &os, const CodeGenerator::Base &codeGenerator, const NNmodel &model, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            // code substitutions ----
            const WeightUpdateModels::Base *wu = sg.getWUModel();
            std::string code = wu->getEventThresholdConditionCode();
            baseSubs.apply(code);
            functionSubstitutions(code, model.getPrecision(), codeGenerator.getFunctions());

            applyWeightUpdateModelSubstitutions(code, sg, codeGenerator.getVarPrefix());
           
            code= ensureFtype(code, model.getPrecision());
            checkUnreplacedVariables(code, sg.getName() + " : evntThreshold");
            os << code;
        },
        [](CodeStream &os, const CodeGenerator::Base &codeGenerator, const NNmodel &model, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            const WeightUpdateModels::Base *wu = sg.getWUModel();
            std::string code = wu->getSimCode(); //**TODO** pass through truespikeness
            baseSubs.apply(code);
            functionSubstitutions(code, model.getPrecision(), codeGenerator.getFunctions());

            applyWeightUpdateModelSubstitutions(code, sg, codeGenerator.getVarPrefix());

            code= ensureFtype(code, model.getPrecision());
            checkUnreplacedVariables(code, sg.getName() + " : simCode");
            os << code;
        }
    );
}

int main()
{
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
    syn->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);
    model.finalize();
    
    CodeStream output(std::cout);
    
    CUDA::CodeGenerator codeGenerator(128, 128);

    
    generateNeuronUpdateKernel(output, model, codeGenerator);
    generatePresynapticUpdateKernel(output, model, codeGenerator);

    return EXIT_SUCCESS;
}

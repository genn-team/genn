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
    auto psmVars = VarNameIterCtx(psm->getVars());
    auto psmDerivedParams = DerivedParamNameIterCtx(psm->getDerivedParams());

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

void generateNeuronUpdateKernel(CodeStream &os, const NNmodel &model, const CodeGenerator::Base &codeGenerator)
{
    // Neuron update kernel
    codeGenerator.genNeuronUpdateKernel(os, model,
        [](CodeStream &os, const CodeGenerator::Base &codeGenerator, const NNmodel &model, const NeuronGroup &ng, 
           const std::string &neuronID, const std::string &rngName)
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
                os << neuronID << "];" << std::endl;
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
                os << neuronID << "];" << std::endl;
            }
            os << std::endl;

            if (!ng.getMergedInSyn().empty() || (nm->getSimCode().find("Isyn") != std::string::npos)) {
                os << model.getPrecision() << " Isyn = 0;" << std::endl;
            }

            // Initialise any additional input variables supported by neuron model
            for (const auto &a : nm->getAdditionalInputVars()) {
                os << a.second.first << " " << a.first << " = " << a.second.second << ";" << std::endl;
            }

            for (const auto &m : ng.getMergedInSyn()) {
                const auto *sg = m.first;
                const auto *psm = sg->getPSModel();

                os << "// pull inSyn values in a coalesced access" << std::endl;
                os << model.getPrecision() << " linSyn" << sg->getName() << " = " << codeGenerator.getVarPrefix() << "inSyn" << sg->getName() << "[" << neuronID << "];" << std::endl;

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

                // Apply substitutions to current converter code
                string psCode = psm->getApplyInputCode();
                substitute(psCode, "$(id)", neuronID);
                substitute(psCode, "$(inSyn)", "linSyn" + sg->getName());
                substitute(psCode, "$(t)", "t");
                substitute(psCode, "$(Isyn)", "Isyn");
                substitute(psCode, "$(rng)", rngName);
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
                substitute(thCode, "$(id)", neuronID);
                substitute(thCode, "$(t)", "t");
                substitute(thCode, "$(Isyn)", "Isyn");
                substitute(thCode, "$(sT)", "lsT");
                substitute(thCode, "$(rng)", rngName);
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
            substitute(sCode, "$(id)", neuronID);
            substitute(sCode, "$(t)", "t");
            substitute(sCode, "$(Isyn)", "Isyn");
            substitute(sCode, "$(sT)", "lsT");  // ?
            substitute(sCode, "$(rng)", rngName);

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
                    substitute(eCode, "$(id)", "n");
                    substitute(eCode, "$(t)", "t");
                    substitute(eCode, "$(rng)", rngName);

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
                    codeGenerator.genEmitSpikeLikeEvent(os, model, ng, neuronID);
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

                    codeGenerator.genEmitTrueSpike(os, model, ng, neuronID);

                    // add after-spike reset if provided
                    if (!nm->getResetCode().empty()) {
                        string rCode = nm->getResetCode();
                        substitute(rCode, "$(id)", neuronID);
                        substitute(rCode, "$(t)", "t");
                        substitute(rCode, "$(Isyn)", "Isyn");
                        substitute(rCode, "$(sT)", "lsT");  // ?
                        substitute(rCode, "$(rng)", rngName);
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
                    os << codeGenerator.getVarPrefix() << v.first << ng.getName() << "[" << ng.getQueueOffset(codeGenerator.getVarPrefix()) << neuronID << "] = l" << v.first << ";" << std::endl;
                }
                else {
                    os << codeGenerator.getVarPrefix() << v.first << ng.getName() << "[" << neuronID << "] = l" << v.first << ";" << std::endl;
                }
            }

            for (const auto &m : ng.getMergedInSyn()) {
                const auto *sg = m.first;
                const auto *psm = sg->getPSModel();

                string pdCode = psm->getDecayCode();
                substitute(pdCode, "$(id)", neuronID);
                substitute(pdCode, "$(inSyn)", "linSyn" + sg->getName());
                substitute(pdCode, "$(t)", "t");
                substitute(pdCode, "$(rng)", rngName);
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

                os << codeGenerator.getVarPrefix() << "inSyn"  << sg->getName() << "[" << neuronID << "] = linSyn" << sg->getName() << ";" << std::endl;
                for (const auto &v : psm->getVars()) {
                    os << codeGenerator.getVarPrefix() << v.first << sg->getName() << "[n]" << " = lps" << v.first << sg->getName() << ";" << std::endl;
                }
            }
        }
    );    
}
/*
void generatePresynapticUpdateKernel(CodeStream &os, const NNmodel &model, const CodeGenerator::Base &codeGenerator)
{
    // Neuron update kernel
    codeGenerator.genPresynapticUpdateKernel(os, model,
        [](CodeStream &os, const CodeGenerator::Base &codeGenerator, const NNmodel &model, const SynapseGroup &sg,
            const std::string &preIdx, const std::string &postIdx)
        {
            const WeightUpdateModel::Base *wu = sg.getWUModel();
            
            // Create iteration context to iterate over the variables; derived and extra global parameters
            DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
            ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
            VarNameIterCtx wuVars(wu->getVars());
            
            if (!wu->getSimSupportCode().empty()) {
                os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
            }
            
            // Code substitutions ----------------------------------------------------------------------------------
            string wCode = (evnt ? wu->getEventCode() : wu->getSimCode());
            substitute(wCode, "$(t)", "t");
            if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                if (shouldAccumulateInSharedMemory(sg)) {
                    substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
                    substitute(wCode, "$(inSyn)", "shLg[ipost]");
                }
                else {
                    substitute(wCode, "$(updatelinsyn)", getFloatAtomicAdd(model.getPrecision()) + "(&$(inSyn), $(addtoinSyn))");
                    substitute(wCode, "$(inSyn)", "dd_inSyn" + sg.getName() + "[ipost]");
                }

                if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                    name_substitutions(wCode, "dd_", wuVars.nameBegin, wuVars.nameEnd,
                                            sg.getName() + "[prePos]");
                }
            }
            else {
                substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
                substitute(wCode, "$(inSyn)", "linSyn");
                if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                    name_substitutions(wCode, codeGenerator.getVarPrefix(), wuVars.nameBegin, wuVars.nameEnd,
                                        sg.getName() + "[shSpk" + postfix + "[j] * " + to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + "+ ipost]");
                }
            }

            StandardSubstitutions::weightUpdateSim(wCode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                "shSpk" + postfix + "[j]", "ipost", codeGenerator.getVarPrefix(),
                                                cudaFunctions, model.getPrecision());
            // end Code substitutions -------------------------------------------------------------------------
            os << wCode << std::endl;
        }
        }
    );
}*/

int main()
{
    initGeNN();

    NNmodel model;
    model.setDT(0.1);
    model.setName("izk_regimes");

    // Izhikevich model parameters
    NeuronModels::Izhikevich::ParamValues paramValues(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues initValues(-65.0, -20.0);

    // Create population of Izhikevich neurons
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 4, paramValues, initValues);
    model.finalize();
    
    CodeStream output(std::cout);
    
    CUDA::CodeGenerator codeGenerator(128, 128);

    
    generateNeuronUpdateKernel(output, model, codeGenerator);

    // Neuron groups
    /*for (const auto &neuroUpdateGroup : neuronUpdateKernel.getNeuronUpdateGroups()) {
        CUDACodeGenerator::PaddedThreadGroup t(output, neuronUpdateKernel.getThreadID(), neuroUpdateGroup);

        // **TODO** model business
    }*/

    // 
 // Presynaptic update kernel
    {
        const size_t blockSize = 32;

        /*CUDACodeGenerator::PresynapticUpdateKernel presynapticUpdateKernel(output, model, blockSize);

        // Synapse groups
        for(const auto &synapseUpdateGroup : presynapticUpdateKernel.getUpdateGroups()) {
            CUDACodeGenerator::PaddedThreadGroup t(output, neuronUpdateKernel.getThreadID(), synapseUpdateGroup);
        }*/
    }
    return EXIT_SUCCESS;
}

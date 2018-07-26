#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"
#include "standardGeneratedSections.h"
#include "standardSubstitutions.h"

// NuGeNN includes
#include "code_generator.h"
#include "cuda_code_generator.h"


void generateNeuronUpdateKernel(CodeStream &os, const NNmodel &model, const CodeGenerator::Base &codeGenerator)
{
    // Neuron update kernel
    codeGenerator.genNeuronUpdateKernel(os, model,
        [](CodeStream &os, const CodeGenerator::Base &codeGenerator, const NNmodel &model, const NeuronGroup &ng, 
           const std::string &neuronID, const std::string &rngName)
        {
            const NeuronModels::Base *nm = ng.getNeuronModel();

            // Create iteration context to iterate over the variables; derived and extra global parameters
            VarNameIterCtx nmVars(nm->getVars());
            DerivedParamNameIterCtx nmDerivedParams(nm->getDerivedParams());
            ExtraGlobalParamNameIterCtx nmExtraGlobalParams(nm->getExtraGlobalParams());

            // Generate code to copy neuron state into local variable
            StandardGeneratedSections::neuronLocalVarInit(os, ng, nmVars, codeGenerator.getVarPrefix(), neuronID);

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
                    for (const auto &v : psm->getVars()) {
                        os << v.second << " lps" << v.first << sg->getName();
                        os << " = " << codeGenerator.getVarPrefix() << v.first << sg->getName() << "[n];" << std::endl;
                    }
                }

                // Apply substitutions to current converter code
                string psCode = psm->getApplyInputCode();
                substitute(psCode, "$(id)", neuronID);
                substitute(psCode, "$(inSyn)", "linSyn" + sg->getName());
                StandardSubstitutions::postSynapseApplyInput(psCode, sg, ng,
                                                                nmVars, nmDerivedParams, nmExtraGlobalParams, 
                                                                codeGenerator.getFunctions(), model.getPrecision(), rngName);

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
                StandardSubstitutions::neuronThresholdCondition(thCode, ng,
                                                                nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                                codeGenerator.getFunctions(), model.getPrecision(), rngName);
                                
                if (GENN_PREFERENCES::autoRefractory) {
                    os << "const bool oldSpike= (" << thCode << ");" << std::endl;
                }
            }

            os << "// calculate membrane potential" << std::endl;
            string sCode = nm->getSimCode();
            substitute(sCode, "$(id)", neuronID);
            StandardSubstitutions::neuronSim(sCode, ng,
                                                nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                codeGenerator.getFunctions(), model.getPrecision(), rngName);
                            
            os << sCode << std::endl;

            // look for spike type events first.
            if (ng.isSpikeEventRequired()) {
                // Generate spike event test
                StandardGeneratedSections::neuronSpikeEventTest(os, ng,
                                                                nmVars, nmExtraGlobalParams, neuronID,
                                                                codeGenerator.getFunctions(), model.getPrecision(), rngName);

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
                        StandardSubstitutions::neuronReset(rCode, ng,
                                                            nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                            codeGenerator.getFunctions(), model.getPrecision(), rngName);

                        os << "// spike reset code" << std::endl;
                        os << rCode << std::endl;
                    }
                }
            }

            // store the defined parts of the neuron state into the global state variables V etc
            StandardGeneratedSections::neuronLocalVarWrite(os, ng, nmVars, codeGenerator.getVarPrefix(), neuronID);

            for (const auto &m : ng.getMergedInSyn()) {
                const auto *sg = m.first;
                const auto *psm = sg->getPSModel();

                string pdCode = psm->getDecayCode();
                substitute(pdCode, "$(id)", neuronID);
                substitute(pdCode, "$(inSyn)", "linSyn" + sg->getName());
                StandardSubstitutions::postSynapseDecay(pdCode, sg, ng,
                                                        nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                        codeGenerator.getFunctions(), model.getPrecision(), rngName);
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

void generatePresynapticUpdateKernel(CodeStream &os, const NNmodel &model, const CodeGenerator::Base &codeGenerator)
{
    // Neuron update kernel
    codeGenerator.genPresynapticUpdateKernel(os, model,
        [](CodeStream &os, const CodeGenerator::Base &codeGenerator, const NNmodel &model, const SynapseGroup &sg,
            const std::string &preIdx, const std::string &postIdx,
        )
        {
            const WeightUpdateModel::Base *wu = sg.getWeightUpdateModel();
            
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

    // Create population of Izhikevich neurons
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 4, paramValues, initValues);
    model.finalize();
    
    CodeStream output(std::cout);
    
    CUDA::CodeGenerator codeGenerator(128);

    
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

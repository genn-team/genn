#include "code_generator/generateNeuronUpdate.h"

// Standard C++ includes
#include <iostream>
#include <string>

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "models.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/substitutions.h"
#include "code_generator/teeStream.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void addNeuronModelSubstitutions(CodeGenerator::Substitutions &substitution, const CodeGenerator::NeuronUpdateGroupMerged &ng,
                                 const std::string &sourceSuffix = "", const std::string &destSuffix = "")
{
    const NeuronModels::Base *nm = ng.getArchetype().getNeuronModel();
    substitution.addVarNameSubstitution(nm->getVars(), sourceSuffix, "l", destSuffix);
    substitution.addParamValueSubstitution(nm->getParamNames(), ng.getArchetype().getParams(), 
                                           [&ng](size_t i) { return ng.isParamHeterogeneous(i);  },
                                           sourceSuffix, "group->");
    substitution.addVarValueSubstitution(nm->getDerivedParams(), ng.getArchetype().getDerivedParams(), 
                                         [&ng](size_t i) { return ng.isDerivedParamHeterogeneous(i);  },
                                         sourceSuffix, "group->");
    substitution.addVarNameSubstitution(nm->getExtraGlobalParams(), sourceSuffix, "group->");
}
//--------------------------------------------------------------------------
void generateWUVarUpdate(CodeGenerator::CodeStream &os, const CodeGenerator::Substitutions &popSubs,
                         const CodeGenerator::NeuronUpdateGroupMerged &ng, const std::string &fieldPrefixStem,
                         const std::string &precision, const std::string &sourceSuffix,
                         const std::vector<SynapseGroupInternal*> &archetypeSyn,
                         unsigned int(SynapseGroupInternal::*getDelaySteps)(void) const,
                         Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                         std::string(WeightUpdateModels::Base::*getCode)(void) const,
                         bool(CodeGenerator::NeuronUpdateGroupMerged::*isParamHeterogeneous)(size_t, size_t) const,
                         bool(CodeGenerator::NeuronUpdateGroupMerged::*isDerivedParamHeterogeneous)(size_t, size_t) const)
{
    using namespace CodeGenerator;

    // Loop through synaptic populations
    for(size_t i = 0; i < archetypeSyn.size(); i++) {
        const SynapseGroupInternal *sg = archetypeSyn[i];
        Substitutions subs(&popSubs);
        CodeStream::Scope b(os);

        // Fetch variables from global memory
        os << "// perform WUM update required for merged" << i << std::endl;
        const auto vars = (sg->getWUModel()->*getVars)();
        const bool delayed = ((sg->*getDelaySteps)() != NO_DELAY);
        for(const auto &v : vars) {
            if(v.access == VarAccess::READ_ONLY) {
                os << "const ";
            }
            os << v.type << " l" << v.name << " = group->" << v.name << fieldPrefixStem << i << "[";
            if(delayed) {
                os << "readDelayOffset + ";
            }
            os << subs["id"] << "];" << std::endl;
        }

        subs.addParamValueSubstitution(sg->getWUModel()->getParamNames(), sg->getWUParams(),
                                       [i, isParamHeterogeneous , &ng](size_t k) { return (ng.*isParamHeterogeneous)(i, k); },
                                       "", "group->", fieldPrefixStem + std::to_string(i));
        subs.addVarValueSubstitution(sg->getWUModel()->getDerivedParams(), sg->getWUDerivedParams(),
                                     [i, isDerivedParamHeterogeneous , &ng](size_t k) { return (ng.*isDerivedParamHeterogeneous)(i, k); },
                                     "", "group->", fieldPrefixStem + std::to_string(i));
        subs.addVarNameSubstitution(sg->getWUModel()->getExtraGlobalParams(), "", "group->", fieldPrefixStem + std::to_string(i));
        subs.addVarNameSubstitution(vars, "", "l");

        const std::string offset = ng.getArchetype().isDelayRequired() ? "readDelayOffset + " : "";
        neuronSubstitutionsInSynapticCode(subs, &ng.getArchetype(), offset, "", subs["id"], sourceSuffix, "", "", "",
                                          [&ng](size_t paramIndex) { return ng.isParamHeterogeneous(paramIndex); },
                                          [&ng](size_t derivedParamIndex) { return ng.isDerivedParamHeterogeneous(derivedParamIndex); });

        // Perform standard substitutions
        std::string code = (sg->getWUModel()->*getCode)();
        subs.applyCheckUnreplaced(code, "spikeCode : merged" + std::to_string(i));
        code = ensureFtype(code, precision);
        os << code;

        // Write back presynaptic variables into global memory
        for(const auto &v : vars) {
            // If state variables is read/write - meaning that it may have been updated - or it is delayed -
            // meaning that it needs to be copied into next delay slot whatever - copy neuron state variables
            // back to global state variables dd_V etc 
            if((v.access == VarAccess::READ_WRITE) || delayed) {
                os << "group->" << v.name << fieldPrefixStem << i << "[";
                if(delayed) {
                    os << "writeDelayOffset + ";
                }
                os << popSubs["id"] << "] = l" << v.name << ";" << std::endl;
            }
        }
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateNeuronUpdate(CodeStream &os, BackendBase::MemorySpaces &memorySpaces,
                                         const ModelSpecMerged &modelMerged, const BackendBase &backend)
{
    os << "#include \"definitionsInternal.h\"" << std::endl;
    if (backend.supportsNamespace()) {
        os << "#include \"supportCode.h\"" << std::endl;
    }
    os << std::endl;

    // Neuron update kernel
    backend.genNeuronUpdate(os, modelMerged, memorySpaces,
        // Preamble handler
        [&modelMerged, &backend](CodeStream &os)
        {
            // Generate functions to push merged neuron group structures
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedNeuronUpdateGroups(), backend);
        },
        // Sim handler
        [&backend, &modelMerged](CodeStream &os, const NeuronUpdateGroupMerged &ng, Substitutions &popSubs,
                                 BackendBase::GroupHandler<NeuronUpdateGroupMerged> genEmitTrueSpike,
                                 BackendBase::GroupHandler<NeuronUpdateGroupMerged> genEmitSpikeLikeEvent)
        {
            const ModelSpecInternal &model = modelMerged.getModel();
            const NeuronModels::Base *nm = ng.getArchetype().getNeuronModel();

            // Generate code to copy neuron state into local variable
            for(const auto &v : nm->getVars()) {
                if(v.access == VarAccess::READ_ONLY) {
                    os << "const ";
                }
                os << v.type << " l" << v.name << " = ";
                os << "group->" << v.name << "[";
                if (ng.getArchetype().isVarQueueRequired(v.name) && ng.getArchetype().isDelayRequired()) {
                    os << "readDelayOffset + ";
                }
                os << popSubs["id"] << "];" << std::endl;
            }
    
            // Also read spike time into local variable
            if(ng.getArchetype().isSpikeTimeRequired()) {
                os << model.getTimePrecision() << " lsT = group->sT[";
                if (ng.getArchetype().isDelayRequired()) {
                    os << "readDelayOffset + ";
                }
                os << popSubs["id"] << "];" << std::endl;
            }
            os << std::endl;

            // If neuron model sim code references ISyn (could still be the case if there are no incoming synapses)
            // OR any incoming synapse groups have post synaptic models which reference $(Isyn), declare it
            if (nm->getSimCode().find("$(Isyn)") != std::string::npos ||
                std::any_of(ng.getArchetype().getMergedInSyn().cbegin(), ng.getArchetype().getMergedInSyn().cend(),
                            [](const std::pair<SynapseGroupInternal*, std::vector<SynapseGroupInternal*>> &p)
                            {
                                return (p.first->getPSModel()->getApplyInputCode().find("$(Isyn)") != std::string::npos
                                        || p.first->getPSModel()->getDecayCode().find("$(Isyn)") != std::string::npos);
                            }))
            {
                os << model.getPrecision() << " Isyn = 0;" << std::endl;
            }

            Substitutions neuronSubs(&popSubs);
            neuronSubs.addVarSubstitution("Isyn", "Isyn");
            neuronSubs.addVarSubstitution("sT", "lsT");
            neuronSubs.addVarNameSubstitution(nm->getAdditionalInputVars());
            addNeuronModelSubstitutions(neuronSubs, ng);

            // Initialise any additional input variables supported by neuron model
            genParamValVecInit(os, neuronSubs, nm->getAdditionalInputVars(),
                               "neuron additional input var : merged" + std::to_string(ng.getIndex()));

            // Loop through incoming synapse groups
            for(size_t i = 0; i < ng.getArchetype().getMergedInSyn().size(); i++) {
                CodeStream::Scope b(os);

                const auto *sg = ng.getArchetype().getMergedInSyn()[i].first;;
                const auto *psm = sg->getPSModel();

                os << "// pull inSyn values in a coalesced access" << std::endl;
                os << model.getPrecision() << " linSyn = group->inSynInSyn" << i << "[" << popSubs["id"] << "];" << std::endl;

                // If dendritic delay is required
                if (sg->isDendriticDelayRequired()) {
                    // Get reference to dendritic delay buffer input for this timestep
                    os << backend.getPointerPrefix() << model.getPrecision() << " *denDelayFront = ";
                    os << "&group->denDelayInSyn" << i << "[(*group->denDelayPtrInSyn" << i << " * group->numNeurons) + " << popSubs["id"] << "];" << std::endl;

                    // Add delayed input from buffer into inSyn
                    os << "linSyn += *denDelayFront;" << std::endl;

                    // Zero delay buffer slot
                    os << "*denDelayFront = " << model.scalarExpr(0.0) << ";" << std::endl;
                }

                // If synapse group has individual postsynaptic variables, also pull these in a coalesced access
                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    // **TODO** base behaviour from Models::Base
                    for (const auto &v : psm->getVars()) {
                        if(v.access == VarAccess::READ_ONLY) {
                            os << "const ";
                        }
                        os << v.type << " lps" << v.name;
                        os << " = group->" << v.name << "InSyn" << i << "[" << neuronSubs["id"] << "];" << std::endl;
                    }
                }

                Substitutions inSynSubs(&neuronSubs);
                inSynSubs.addVarSubstitution("inSyn", "linSyn");

                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    inSynSubs.addVarNameSubstitution(psm->getVars(), "", "lps");
                }
                else {
                    inSynSubs.addVarValueSubstitution(psm->getVars(), sg->getPSConstInitVals(),
                                                      [i, &ng](size_t p) { return ng.isPSMGlobalVarHeterogeneous(i, p); },
                                                      "", "group->", "InSyn" + std::to_string(i));
                }

                inSynSubs.addParamValueSubstitution(psm->getParamNames(), sg->getPSParams(),
                                                    [i, &ng](size_t p) { return ng.isPSMParamHeterogeneous(i, p);  },
                                                    "", "group->", "InSyn" + std::to_string(i));
                inSynSubs.addVarValueSubstitution(psm->getDerivedParams(), sg->getPSDerivedParams(),
                                                  [i, &ng](size_t p) { return ng.isPSMDerivedParamHeterogeneous(i, p);  },
                                                  "", "group->", "InSyn" + std::to_string(i));
                inSynSubs.addVarNameSubstitution(psm->getExtraGlobalParams(), "", "group->", "InSyn" + std::to_string(i));

                // Apply substitutions to current converter code
                std::string psCode = psm->getApplyInputCode();
                inSynSubs.applyCheckUnreplaced(psCode, "postSyntoCurrent : merged " + std::to_string(i));
                psCode = ensureFtype(psCode, model.getPrecision());

                // Apply substitutions to decay code
                std::string pdCode = psm->getDecayCode();
                inSynSubs.applyCheckUnreplaced(pdCode, "decayCode : merged " + std::to_string(i));
                pdCode = ensureFtype(pdCode, model.getPrecision());

                if (!psm->getSupportCode().empty() && backend.supportsNamespace()) {
                    os << "using namespace " << modelMerged.getPostsynapticDynamicsSupportCodeNamespace(psm->getSupportCode()) <<  ";" << std::endl;
                }

                if (!psm->getSupportCode().empty() && !backend.supportsNamespace()) {
                    psCode = disambiguateNamespaceFunction(psm->getSupportCode(), psCode, modelMerged.getPostsynapticDynamicsSupportCodeNamespace(psm->getSupportCode()));
                    pdCode = disambiguateNamespaceFunction(psm->getSupportCode(), pdCode, modelMerged.getPostsynapticDynamicsSupportCodeNamespace(psm->getSupportCode()));
                }

                os << psCode << std::endl;
                os << pdCode << std::endl;

                if (!psm->getSupportCode().empty()) {
                    os << CodeStream::CB(29) << " // namespace bracket closed" << std::endl;
                }

                // Write back linSyn
                os << "group->inSynInSyn"  << i << "[" << inSynSubs["id"] << "] = linSyn;" << std::endl;

                // Copy any non-readonly postsynaptic model variables back to global state variables dd_V etc
                for (const auto &v : psm->getVars()) {
                    if(v.access == VarAccess::READ_WRITE) {
                        os << "group->" << v.name << "InSyn" << i << "[" << inSynSubs["id"] << "]" << " = lps" << v.name << ";" << std::endl;
                    }
                }
            }

            // Loop through all of neuron group's current sources
            for(size_t i = 0; i < ng.getArchetype().getCurrentSources().size(); i++) {
                const auto *cs = ng.getArchetype().getCurrentSources()[i];

                os << "// current source " << i << std::endl;
                CodeStream::Scope b(os);

                const auto *csm = cs->getCurrentSourceModel();

                // Read current source variables into registers
                for(const auto &v : csm->getVars()) {
                    if(v.access == VarAccess::READ_ONLY) {
                        os << "const ";
                    }
                    os << v.type << " lcs" << v.name << " = " << "group->" << v.name << "CS" << i <<"[" << popSubs["id"] << "];" << std::endl;
                }

                Substitutions currSourceSubs(&popSubs);
                currSourceSubs.addFuncSubstitution("injectCurrent", 1, "Isyn += $(0)");
                currSourceSubs.addVarNameSubstitution(csm->getVars(), "", "lcs");
                currSourceSubs.addParamValueSubstitution(csm->getParamNames(), cs->getParams(),
                                                         [&ng, i](size_t p) { return ng.isCurrentSourceParamHeterogeneous(i, p);  },
                                                         "", "group->", "CS" + std::to_string(i));
                currSourceSubs.addVarValueSubstitution(csm->getDerivedParams(), cs->getDerivedParams(),
                                                       [&ng, i](size_t p) { return ng.isCurrentSourceDerivedParamHeterogeneous(i, p);  },
                                                       "", "group->", "CS" + std::to_string(i));
                currSourceSubs.addVarNameSubstitution(csm->getExtraGlobalParams(), "", "group->", "CS" + std::to_string(i));

                std::string iCode = csm->getInjectionCode();
                currSourceSubs.applyCheckUnreplaced(iCode, "injectionCode : merged" + std::to_string(i));
                iCode = ensureFtype(iCode, model.getPrecision());
                os << iCode << std::endl;

                // Write read/write variables back to global memory
                for(const auto &v : csm->getVars()) {
                    if(v.access == VarAccess::READ_WRITE) {
                        os << "group->" << v.name << "CS" << i << "[" << currSourceSubs["id"] << "] = lcs" << v.name << ";" << std::endl;
                    }
                }
            }

            if (!nm->getSupportCode().empty() && backend.supportsNamespace()) {
                os << "using namespace " << modelMerged.getNeuronUpdateSupportCodeNamespace(nm->getSupportCode()) <<  ";" << std::endl;
            }

            // If a threshold condition is provided
            std::string thCode = nm->getThresholdConditionCode();
            if (!thCode.empty()) {
                os << "// test whether spike condition was fulfilled previously" << std::endl;

                neuronSubs.applyCheckUnreplaced(thCode, "thresholdConditionCode : merged" + std::to_string(ng.getIndex()));
                thCode= ensureFtype(thCode, model.getPrecision());

                if (!nm->getSupportCode().empty() && !backend.supportsNamespace()) {
                    thCode = disambiguateNamespaceFunction(nm->getSupportCode(), thCode, modelMerged.getNeuronUpdateSupportCodeNamespace(nm->getSupportCode()));
                }

                if (nm->isAutoRefractoryRequired()) {
                    os << "const bool oldSpike= (" << thCode << ");" << std::endl;
                }
            }
            // Otherwise, if any outgoing synapse groups have spike-processing code
            /*else if(std::any_of(ng.getOutSyn().cbegin(), ng.getOutSyn().cend(),
                                [](const SynapseGroupInternal *sg){ return !sg->getWUModel()->getSimCode().empty(); }))
            {
                LOGW_CODE_GEN << "No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << ng.getName() << "\" was provided. There will be no spikes detected in this population!";
            }*/

            os << "// calculate membrane potential" << std::endl;
            std::string sCode = nm->getSimCode();
            neuronSubs.applyCheckUnreplaced(sCode, "simCode : merged" + std::to_string(ng.getIndex()));
            sCode = ensureFtype(sCode, model.getPrecision());

            if (!nm->getSupportCode().empty() && !backend.supportsNamespace()) {
                sCode = disambiguateNamespaceFunction(nm->getSupportCode(), sCode, modelMerged.getNeuronUpdateSupportCodeNamespace(nm->getSupportCode()));
            }

            os << sCode << std::endl;

            // look for spike type events first.
            if (ng.getArchetype().isSpikeEventRequired()) {
                // Create local variable
                os << "bool spikeLikeEvent = false;" << std::endl;

                // Loop through outgoing synapse populations that will contribute to event condition code
                size_t i = 0;
                for(const auto &spkEventCond : ng.getArchetype().getSpikeEventCondition()) {
                    // Replace of parameters, derived parameters and extraglobalsynapse parameters
                    Substitutions spkEventCondSubs(&popSubs);

                    // If this spike event condition requires EGPS, substitute them
                    if(spkEventCond.egpInThresholdCode) {
                        spkEventCondSubs.addVarNameSubstitution(spkEventCond.synapseGroup->getWUModel()->getExtraGlobalParams(), "", "group->", "EventThresh" + std::to_string(i));
                        i++;
                    }
                    addNeuronModelSubstitutions(spkEventCondSubs, ng, "_pre");

                    std::string eCode = spkEventCond.eventThresholdCode;
                    spkEventCondSubs.applyCheckUnreplaced(eCode, "neuronSpkEvntCondition : merged" + std::to_string(ng.getIndex()));
                    eCode = ensureFtype(eCode, model.getPrecision());

                    // Open scope for spike-like event test
                    os << CodeStream::OB(31);

                    // Use presynaptic update namespace if required
                    if (!spkEventCond.supportCode.empty() && backend.supportsNamespace()) {
                        os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(spkEventCond.supportCode) << ";" << std::endl;
                    }

                    // Substitute with namespace functions
                    if (!spkEventCond.supportCode.empty() && !backend.supportsNamespace()) {
                        eCode = disambiguateNamespaceFunction(spkEventCond.supportCode, eCode, modelMerged.getPresynapticUpdateSupportCodeNamespace(spkEventCond.supportCode));
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
                    genEmitSpikeLikeEvent(os, ng, popSubs);
                }
            }

            // test for true spikes if condition is provided
            if (!thCode.empty()) {
                os << "// test for and register a true spike" << std::endl;
                if (nm->isAutoRefractoryRequired()) {
                    os << "if ((" << thCode << ") && !(oldSpike))";
                }
                else {
                    os << "if (" << thCode << ")";
                }
                {
                    CodeStream::Scope b(os);
                    genEmitTrueSpike(os, ng, popSubs);

                    // add after-spike reset if provided
                    if (!nm->getResetCode().empty()) {
                        std::string rCode = nm->getResetCode();
                        neuronSubs.applyCheckUnreplaced(rCode, "resetCode : merged" + std::to_string(ng.getIndex()));
                        rCode = ensureFtype(rCode, model.getPrecision());

                        os << "// spike reset code" << std::endl;
                        os << rCode << std::endl;
                    }
                }

                // Spike triggered variables don't need to be copied
                // if delay isn't required as there's only one copy of them
                if(ng.getArchetype().isDelayRequired()) {
                    const auto outSynWithPreCode = ng.getArchetype().getOutSynWithPreCode();
                    const auto inSynWithPostCode = ng.getArchetype().getInSynWithPostCode();

                    // Are there any outgoing synapse groups with axonal delay and presynaptic WUM variables?
                    const bool preVars = std::any_of(outSynWithPreCode.cbegin(), outSynWithPreCode.cend(),
                                                     [](const SynapseGroupInternal *sg){ return (sg->getDelaySteps() != NO_DELAY); });

                    // Are there any incoming synapse groups with back-propagation delay and postsynaptic WUM variables?
                    const bool postVars = std::any_of(inSynWithPostCode.cbegin(), inSynWithPostCode.cend(),
                                                      [](const SynapseGroupInternal *sg){ return (sg->getBackPropDelaySteps() != NO_DELAY); });

                    // If spike times, presynaptic variables or postsynaptic variables are required, add if clause
                    if(ng.getArchetype().isSpikeTimeRequired() || preVars || postVars) {
                        os << "else";
                        CodeStream::Scope b(os);

                        // If spike timing is required, copy spike time from register
                        if(ng.getArchetype().isSpikeTimeRequired()) {
                            os << "group->sT[writeDelayOffset + " << popSubs["id"] << "] = lsT;" << std::endl;
                        }

                        // Copy presynaptic WUM variables between delay slots
                        for(size_t i = 0; i < outSynWithPreCode.size(); i++) {
                            const auto *sg = outSynWithPreCode[i];
                            if(sg->getDelaySteps() != NO_DELAY) {
                                for(const auto &v : sg->getWUModel()->getPreVars()) {
                                    os << "group->" << v.name << "WUPre" << i << "[writeDelayOffset + " << popSubs["id"] <<  "] = ";
                                    os << "group->" << v.name << "WUPre" << i << "[readDelayOffset + " << popSubs["id"] << "];" << std::endl;
                                }
                            }
                        }


                        // Copy postsynaptic WUM variables between delay slots
                        for(size_t i = 0; i < inSynWithPostCode.size(); i++) {
                            const auto *sg = inSynWithPostCode[i];
                            if(sg->getBackPropDelaySteps() != NO_DELAY) {
                                for(const auto &v : sg->getWUModel()->getPostVars()) {
                                    os << "group->" << v.name << "WUPost" << i << "[writeDelayOffset + " << popSubs["id"] <<  "] = ";
                                    os << "group->" << v.name << "WUPost" << i << "[readDelayOffset + " << popSubs["id"] << "];" << std::endl;
                                }
                            }
                        }
                    }
                }
            }

            // Loop through neuron state variables
            for(const auto &v : nm->getVars()) {
                // If state variables is read/writes - meaning that it may have been updated - or it is delayed -
                // meaning that it needs to be copied into next delay slot whatever - copy neuron state variables
                // back to global state variables dd_V etc  
                const bool delayed = (ng.getArchetype().isVarQueueRequired(v.name) && ng.getArchetype().isDelayRequired());
                if((v.access == VarAccess::READ_WRITE) || delayed) {
                    os << "group->" << v.name << "[";

                    if (delayed) {
                        os << "writeDelayOffset + ";
                    }
                    os << popSubs["id"] << "] = l" << v.name << ";" << std::endl;
                }
            }
        },
        // WU var update handler
        [&modelMerged](CodeStream &os, const NeuronUpdateGroupMerged &ng, Substitutions &popSubs)
        {
            // Generate var update for outgoing synaptic populations with presynaptic update code
            generateWUVarUpdate(os, popSubs, ng, "WUPre", modelMerged.getModel().getPrecision(), "_pre",
                                ng.getArchetype().getOutSynWithPreCode(), &SynapseGroupInternal::getDelaySteps,
                                &WeightUpdateModels::Base::getPreVars, &WeightUpdateModels::Base::getPreSpikeCode,
                                &NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous, 
                                &NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous);
            

            // Generate var update for incoming synaptic populations with postsynaptic code
            generateWUVarUpdate(os, popSubs, ng, "WUPost", modelMerged.getModel().getPrecision(), "_post",
                                ng.getArchetype().getInSynWithPostCode(), &SynapseGroupInternal::getBackPropDelaySteps,
                                &WeightUpdateModels::Base::getPostVars, &WeightUpdateModels::Base::getPostSpikeCode,
                                &NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous,
                                &NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous);
        },
        // Push EGP handler
        [&backend, &modelMerged](CodeStream &os)
        {
            modelMerged.genScalarEGPPush(os, "NeuronUpdate", backend);
        });
}

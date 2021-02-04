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

using namespace CodeGenerator;

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
                         bool useLocalNeuronVars, unsigned int batchSize, 
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

        // If this code string isn't empty
        std::string code = (sg->getWUModel()->*getCode)();
        if(!code.empty()) {
            Substitutions subs(&popSubs);
            CodeStream::Scope b(os);

            // Fetch variables from global memory
            os << "// perform WUM update required for merged" << i << std::endl;
            const auto vars = (sg->getWUModel()->*getVars)();
            const bool delayed = ((sg->*getDelaySteps)() != NO_DELAY);
            for(const auto &v : vars) {
                if(v.access & VarAccessMode::READ_ONLY) {
                    os << "const ";
                }
                os << v.type << " l" << v.name << " = group->" << v.name << fieldPrefixStem << i << "[";
                os << ng.getReadVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), subs["id"]) << "];" << std::endl;
            }

            subs.addParamValueSubstitution(sg->getWUModel()->getParamNames(), sg->getWUParams(),
                                        [i, isParamHeterogeneous , &ng](size_t k) { return (ng.*isParamHeterogeneous)(i, k); },
                                        "", "group->", fieldPrefixStem + std::to_string(i));
            subs.addVarValueSubstitution(sg->getWUModel()->getDerivedParams(), sg->getWUDerivedParams(),
                                        [i, isDerivedParamHeterogeneous , &ng](size_t k) { return (ng.*isDerivedParamHeterogeneous)(i, k); },
                                        "", "group->", fieldPrefixStem + std::to_string(i));
            subs.addVarNameSubstitution(sg->getWUModel()->getExtraGlobalParams(), "", "group->", fieldPrefixStem + std::to_string(i));
            subs.addVarNameSubstitution(vars, "", "l");

            neuronSubstitutionsInSynapticCode(subs, &ng.getArchetype(), "", sourceSuffix, "", "", "", useLocalNeuronVars,
                                              [&ng](size_t paramIndex) { return ng.isParamHeterogeneous(paramIndex); },
                                              [&ng](size_t derivedParamIndex) { return ng.isDerivedParamHeterogeneous(derivedParamIndex); },
                                              [&subs, &ng, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                              {
                                                  return ng.getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                              },
                                              [&subs, &ng, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                              { 
                                                  return ng.getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                              });

            // Perform standard substitutions
            subs.applyCheckUnreplaced(code, "spikeCode : merged" + std::to_string(i));
            code = ensureFtype(code, precision);
            os << code;

            // Write back presynaptic variables into global memory
            for(const auto &v : vars) {
                // If state variables is read/write - meaning that it may have been updated - or it is delayed -
                // meaning that it needs to be copied into next delay slot whatever - copy neuron state variables
                // back to global state variables dd_V etc
                if((v.access & VarAccessMode::READ_WRITE) || delayed) {
                    os << "group->" << v.name << fieldPrefixStem << i << "[";
                    os << ng.getWriteVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), subs["id"]) << "] = l" << v.name << ";" << std::endl;
                }
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
            const unsigned int batchSize = model.getBatchSize();
            const NeuronModels::Base *nm = ng.getArchetype().getNeuronModel();

            // Generate code to copy neuron state into local variable
            for(const auto &v : nm->getVars()) {
                if(v.access & VarAccessMode::READ_ONLY) {
                    os << "const ";
                }
                os << v.type << " l" << v.name << " = group->" << v.name << "[";
                const bool delayed = (ng.getArchetype().isVarQueueRequired(v.name) && ng.getArchetype().isDelayRequired());
                os << ng.getReadVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "];" << std::endl;
            }
    
            // Also read spike and spike-like-event times into local variables if required
            if(ng.getArchetype().isSpikeTimeRequired()) {
                os << "const " << model.getTimePrecision() << " lsT = group->sT[";
                os << ng.getReadVarIndex(ng.getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
            }
            if(ng.getArchetype().isPrevSpikeTimeRequired()) {
                os << "const " << model.getTimePrecision() << " lprevST = group->prevST[";
                os << ng.getReadVarIndex(ng.getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
            }
            if(ng.getArchetype().isSpikeEventTimeRequired()) {
                os << "const " << model.getTimePrecision() << " lseT = group->seT[";
                os << ng.getReadVarIndex(ng.getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
            }
            if(ng.getArchetype().isPrevSpikeEventTimeRequired()) {
                os <<  "const " << model.getTimePrecision() << " lprevSET = group->prevSET[";
                os << ng.getReadVarIndex(ng.getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
            }
            os << std::endl;

            // If neuron model sim code references ISyn (could still be the case if there are no incoming synapses)
            // OR any incoming synapse groups have post synaptic models which reference $(Isyn), declare it
            if (nm->getSimCode().find("$(Isyn)") != std::string::npos ||
                std::any_of(ng.getArchetype().getMergedInSyn().cbegin(), ng.getArchetype().getMergedInSyn().cend(),
                            [](const SynapseGroupInternal *sg)
                            {
                                return (sg->getPSModel()->getApplyInputCode().find("$(Isyn)") != std::string::npos
                                        || sg->getPSModel()->getDecayCode().find("$(Isyn)") != std::string::npos);
                            }))
            {
                os << model.getPrecision() << " Isyn = 0;" << std::endl;
            }

            Substitutions neuronSubs(&popSubs);
            neuronSubs.addVarSubstitution("Isyn", "Isyn");

            if(ng.getArchetype().isSpikeTimeRequired()) {
                neuronSubs.addVarSubstitution("sT", "lsT");
            }
            if(ng.getArchetype().isPrevSpikeTimeRequired()) {
                neuronSubs.addVarSubstitution("prev_sT", "lprevST");
            }
            if(ng.getArchetype().isSpikeEventTimeRequired()) {
                neuronSubs.addVarSubstitution("seT", "lseT");
            }
            if(ng.getArchetype().isPrevSpikeEventTimeRequired()) {
                neuronSubs.addVarSubstitution("prev_seT", "lprevSET");
            }
            neuronSubs.addVarNameSubstitution(nm->getAdditionalInputVars());
            addNeuronModelSubstitutions(neuronSubs, ng);

            // Initialise any additional input variables supported by neuron model
            for (const auto &a : nm->getAdditionalInputVars()) {
                // Apply substitutions to value
                std::string value = a.value;
                neuronSubs.applyCheckUnreplaced(value, "neuron additional input var : merged" + std::to_string(ng.getIndex()));

                os << a.type << " " << a.name << " = " << value << ";" << std::endl;
            }

            // Loop through incoming synapse groups
            for(size_t i = 0; i < ng.getArchetype().getMergedInSyn().size(); i++) {
                CodeStream::Scope b(os);

                const auto *sg = ng.getArchetype().getMergedInSyn()[i];
                const auto *psm = sg->getPSModel();

                os << "// pull inSyn values in a coalesced access" << std::endl;
                os << model.getPrecision() << " linSyn = group->inSynInSyn" << i << "[";
                os << ng.getVarIndex(batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;

                // If dendritic delay is required
                if (sg->isDendriticDelayRequired()) {
                    // Get reference to dendritic delay buffer input for this timestep
                    os << backend.getPointerPrefix() << model.getPrecision() << " *denDelayFront = ";
                    os << "&group->denDelayInSyn" << i << "[(*group->denDelayPtrInSyn" << i << " * group->numNeurons) + ";
                    os << ng.getVarIndex(batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;

                    // Add delayed input from buffer into inSyn
                    os << "linSyn += *denDelayFront;" << std::endl;

                    // Zero delay buffer slot
                    os << "*denDelayFront = " << model.scalarExpr(0.0) << ";" << std::endl;
                }

                // If synapse group has individual postsynaptic variables, also pull these in a coalesced access
                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    // **TODO** base behaviour from Models::Base
                    for (const auto &v : psm->getVars()) {
                        if(v.access & VarAccessMode::READ_ONLY) {
                            os << "const ";
                        }
                        os << v.type << " lps" << v.name << " = group->" << v.name << "InSyn" << i << "[";
                        os << ng.getVarIndex(batchSize, getVarAccessDuplication(v.access), neuronSubs["id"]) << "];" << std::endl;
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
                os << "group->inSynInSyn" << i << "[";
                os << ng.getVarIndex(batchSize, VarAccessDuplication::DUPLICATE, inSynSubs["id"]) << "] = linSyn;" << std::endl;

                // Copy any non-readonly postsynaptic model variables back to global state variables dd_V etc
                for (const auto &v : psm->getVars()) {
                    if(v.access & VarAccessMode::READ_WRITE) {
                        os << "group->" << v.name << "InSyn" << i << "[";
                        os << ng.getVarIndex(batchSize, getVarAccessDuplication(v.access), inSynSubs["id"]) << "]" << " = lps" << v.name << ";" << std::endl;
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
                    if(v.access & VarAccessMode::READ_ONLY) {
                        os << "const ";
                    }
                    os << v.type << " lcs" << v.name << " = " << "group->" << v.name << "CS" << i << "[";
                    os << ng.getVarIndex(batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "];" << std::endl;
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
                    if(v.access & VarAccessMode::READ_WRITE) {
                        os << "group->" << v.name << "CS" << i << "[";
                        os << ng.getVarIndex(batchSize, getVarAccessDuplication(v.access), currSourceSubs["id"]) << "] = lcs" << v.name << ";" << std::endl;
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
                    os << "const bool oldSpike = (" << thCode << ");" << std::endl;
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

            // Generate var update for outgoing synaptic populations with presynaptic update code
            generateWUVarUpdate(os, popSubs, ng, "WUPre", modelMerged.getModel().getPrecision(), "_pre", true, batchSize,
                                ng.getArchetype().getOutSynWithPreCode(), &SynapseGroupInternal::getDelaySteps,
                                &WeightUpdateModels::Base::getPreVars, &WeightUpdateModels::Base::getPreDynamicsCode,
                                &NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous,
                                &NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous);


            // Generate var update for incoming synaptic populations with postsynaptic code
            generateWUVarUpdate(os, popSubs, ng, "WUPost", modelMerged.getModel().getPrecision(), "_post", true, batchSize,
                                ng.getArchetype().getInSynWithPostCode(), &SynapseGroupInternal::getBackPropDelaySteps,
                                &WeightUpdateModels::Base::getPostVars, &WeightUpdateModels::Base::getPostDynamicsCode,
                                &NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous,
                                &NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous);

            // look for spike type events first.
            if (ng.getArchetype().isSpikeEventRequired()) {
                // Create local variable
                os << "bool spikeLikeEvent = false;" << std::endl;

                // Loop through outgoing synapse populations that will contribute to event condition code
                size_t i = 0;
                for(const auto &spkEventCond : ng.getArchetype().getSpikeEventCondition()) {
                    // Replace of parameters, derived parameters and extraglobalsynapse parameters
                    Substitutions spkEventCondSubs(&popSubs);

                    // If this spike event condition requires synapse state
                    if(spkEventCond.synapseStateInThresholdCode) {
                        // Substitute EGPs
                        spkEventCondSubs.addVarNameSubstitution(spkEventCond.synapseGroup->getWUModel()->getExtraGlobalParams(), "", "group->", "EventThresh" + std::to_string(i));

                        // Substitute presynaptic variables
                        const bool delayed = (spkEventCond.synapseGroup->getDelaySteps() != NO_DELAY);
                        spkEventCondSubs.addVarNameSubstitution(spkEventCond.synapseGroup->getWUModel()->getPreVars(), "", "group->",
                                                                [&ng, &popSubs, batchSize, delayed, i](VarAccess a) 
                                                                { 
                                                                    return "EventThresh" + std::to_string(i) + "[" + ng.getReadVarIndex(delayed, batchSize, getVarAccessDuplication(a), popSubs["id"]) + "]";
                                                                });
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

                // If spike-like-event timing is required and they aren't updated after update, copy spike-like-event time from register
                if(ng.getArchetype().isDelayRequired() && (ng.getArchetype().isSpikeEventTimeRequired() || ng.getArchetype().isPrevSpikeEventTimeRequired())) {
                    os << "else";
                    CodeStream::Scope b(os);

                    if(ng.getArchetype().isSpikeEventTimeRequired()) {
                        os << "group->seT[" << ng.getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lseT;" << std::endl;
                    }
                    if(ng.getArchetype().isPrevSpikeEventTimeRequired()) {
                        os << "group->prevSET[" << ng.getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lprevSET;" << std::endl;
                    }
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
                    // **FIXME** there is a corner case here where, if pre or postsynaptic variables have no update code
                    // but there are delays they won't get copied. It might make more sense (and tidy up several things
                    // to instead build merged neuron update groups based on inSynWithPostVars/outSynWithPreVars instead.
                    const auto outSynWithPreCode = ng.getArchetype().getOutSynWithPreCode();
                    const auto inSynWithPostCode = ng.getArchetype().getInSynWithPostCode();

                    // Are there any outgoing synapse groups with presynaptic code
                    // which have axonal delay and no presynaptic dynamics
                    const bool preVars = std::any_of(outSynWithPreCode.cbegin(), outSynWithPreCode.cend(),
                                                     [](const SynapseGroupInternal *sg)
                                                     {
                                                         return ((sg->getDelaySteps() != NO_DELAY)
                                                                 && sg->getWUModel()->getPreDynamicsCode().empty());
                                                     });

                    // Are there any incoming synapse groups with postsynaptic code
                    // which have back-propagation delay and no postsynaptic dynamics
                    const bool postVars = std::any_of(inSynWithPostCode.cbegin(), inSynWithPostCode.cend(),
                                                      [](const SynapseGroupInternal *sg)
                                                      {
                                                          return ((sg->getBackPropDelaySteps() != NO_DELAY)
                                                                   && sg->getWUModel()->getPostDynamicsCode().empty());
                                                      });

                    // If spike times, presynaptic variables or postsynaptic variables are required, add if clause
                    if(ng.getArchetype().isSpikeTimeRequired() || ng.getArchetype().isPrevSpikeTimeRequired() || preVars || postVars) {
                        os << "else";
                        CodeStream::Scope b(os);

                        // If spike times are required, copy times from register
                        if(ng.getArchetype().isSpikeTimeRequired()) {
                            os << "group->sT[" << ng.getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lsT;" << std::endl;
                        }

                        // If previous spike times are required, copy times from register
                        if(ng.getArchetype().isPrevSpikeTimeRequired()) {
                            os << "group->prevST[" << ng.getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lprevST;" << std::endl;
                        }

                        // Loop through outgoing synapse groups with some sort of presynaptic code
                        for(size_t i = 0; i < outSynWithPreCode.size(); i++) {
                            const auto *sg = outSynWithPreCode[i];
                            // If this group has a delay and no presynaptic dynamics (which will already perform this copying)
                            if(sg->getDelaySteps() != NO_DELAY && sg->getWUModel()->getPreDynamicsCode().empty()) {
                                // Loop through variables and copy between read and write delay slots
                                for(const auto &v : sg->getWUModel()->getPreVars()) {
                                    if(v.access & VarAccessMode::READ_WRITE) {
                                        os << "group->" << v.name << "WUPre" << i << "[" << ng.getWriteVarIndex(true, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "] = ";
                                        os << "group->" << v.name << "WUPre" << i << "[" << ng.getReadVarIndex(true, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "];" << std::endl;
                                    }
                                }
                            }
                        }

                        // Loop through outgoing synapse groups with some sort of postsynaptic code
                        for(size_t i = 0; i < inSynWithPostCode.size(); i++) {
                            const auto *sg = inSynWithPostCode[i];
                            // If this group has a delay and no postsynaptic dynamics (which will already perform this copying)
                            if(sg->getBackPropDelaySteps() != NO_DELAY && sg->getWUModel()->getPostDynamicsCode().empty()) {
                                // Loop through variables and copy between read and write delay slots
                                for(const auto &v : sg->getWUModel()->getPostVars()) {
                                    if(v.access & VarAccessMode::READ_WRITE) {
                                        os << "group->" << v.name << "WUPost" << i << "[" << ng.getWriteVarIndex(true, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "] = ";
                                        os << "group->" << v.name << "WUPost" << i << "[" << ng.getReadVarIndex(true, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "];" << std::endl;
                                    }
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
                if((v.access & VarAccessMode::READ_WRITE) || delayed) {
                    os << "group->" << v.name << "[";
                    os << ng.getWriteVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "] = l" << v.name << ";" << std::endl;
                }
            }
        },
        // WU var update handler
        [&modelMerged](CodeStream &os, const NeuronUpdateGroupMerged &ng, Substitutions &popSubs)
        {
            // Generate var update for outgoing synaptic populations with presynaptic update code
            const unsigned int batchSize = modelMerged.getModel().getBatchSize();
            generateWUVarUpdate(os, popSubs, ng, "WUPre", modelMerged.getModel().getPrecision(), "_pre", false, batchSize,
                                ng.getArchetype().getOutSynWithPreCode(), &SynapseGroupInternal::getDelaySteps,
                                &WeightUpdateModels::Base::getPreVars, &WeightUpdateModels::Base::getPreSpikeCode,
                                &NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous, 
                                &NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous);
            

            // Generate var update for incoming synaptic populations with postsynaptic code
            generateWUVarUpdate(os, popSubs, ng, "WUPost", modelMerged.getModel().getPrecision(), "_post", false, batchSize,
                                ng.getArchetype().getInSynWithPostCode(), &SynapseGroupInternal::getBackPropDelaySteps,
                                &WeightUpdateModels::Base::getPostVars, &WeightUpdateModels::Base::getPostSpikeCode,
                                &NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous,
                                &NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous);
        },
        // Push EGP handler
        [&backend, &modelMerged](CodeStream &os)
        {
            modelMerged.genScalarEGPPush<NeuronUpdateGroupMerged>(os, backend);
        });
}

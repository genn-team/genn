#include "code_generator/neuronUpdateGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

using namespace CodeGenerator;

//----------------------------------------------------------------------------
// CodeGenerator::NeuronUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronUpdateGroupMerged::name = "NeuronUpdate";
//----------------------------------------------------------------------------
NeuronUpdateGroupMerged::NeuronUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                                 const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, precision, timePrecision, backend, false, groups)
{
    // Build vector of vectors containing each child group's incoming synapse groups
    // with postsynaptic updates, ordered to match those of the archetype group
    orderGroupChildren(m_SortedInSynWithPostCode, &NeuronGroupInternal::getFusedInSynWithPostCode,
                       &SynapseGroupInternal::getWUPostHashDigest);

    // Build vector of vectors containing each child group's outgoing synapse groups
    // with presynaptic synaptic updates, ordered to match those of the archetype group
    orderGroupChildren(m_SortedOutSynWithPreCode, &NeuronGroupInternal::getFusedOutSynWithPreCode,
                       &SynapseGroupInternal::getWUPreHashDigest);

    // Generate struct fields for incoming synapse groups with postsynaptic update code
    generateWUVar("WUPost", m_SortedInSynWithPostCode,
                  &WeightUpdateModels::Base::getPostVars, &NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous,
                  &NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous);

    // Generate struct fields for outgoing synapse groups with presynaptic update code
    generateWUVar("WUPre", m_SortedOutSynWithPreCode,
                  &WeightUpdateModels::Base::getPreVars, &NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous,
                  &NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous);

    // Loop through neuron groups
    std::vector<std::vector<SynapseGroupInternal *>> eventThresholdSGs;
    for(const auto &g : getGroups()) {
        // Reserve vector for this group's children
        eventThresholdSGs.emplace_back();

        // Add synapse groups 
        for(const auto &s : g.get().getSpikeEventCondition()) {
            if(s.synapseStateInThresholdCode) {
                eventThresholdSGs.back().push_back(s.synapseGroup);
            }
        }
    }

    // Loop through all spike event conditions
    size_t i = 0;
    for(const auto &s : getArchetype().getSpikeEventCondition()) {
        // If threshold condition references any synapse state
        if(s.synapseStateInThresholdCode) {
            const auto wum = s.synapseGroup->getWUModel();

            // Loop through all EGPs in synapse group 
            const auto sgEGPs = wum->getExtraGlobalParams();
            for(const auto &egp : sgEGPs) {
                // If EGP is referenced in event threshold code
                if(s.eventThresholdCode.find("$(" + egp.name + ")") != std::string::npos) {
                    const bool isPointer = Utils::isTypePointer(egp.type);
                    const std::string prefix = isPointer ? getDeviceVarPrefix() : "";
                    addField(egp.type, egp.name + "EventThresh" + std::to_string(i),
                             [eventThresholdSGs, prefix, egp, i](const NeuronGroupInternal &, size_t groupIndex, const MergedRunnerMap &map)
                             {
                                 return map.findGroup(*eventThresholdSGs.at(groupIndex).at(i)) + "." + prefix + egp.name;
                             },
                             Utils::isTypePointer(egp.type) ? FieldType::PointerEGP : FieldType::ScalarEGP);
                }
            }

            // Loop through all presynaptic variables in synapse group 
            const auto sgPreVars = wum->getPreVars();
            for(const auto &var : sgPreVars) {
                // If variable is referenced in event threshold code
                if(s.eventThresholdCode.find("$(" + var.name + ")") != std::string::npos) {
                    addField(var.type + "*", var.name + "EventThresh" + std::to_string(i),
                             [this, eventThresholdSGs, var, i](const NeuronGroupInternal &, size_t groupIndex, const MergedRunnerMap &map)
                             {
                                 return map.findGroup(*eventThresholdSGs.at(groupIndex).at(i)) + "." + getDeviceVarPrefix() + var.name;
                             });
                }
            }
            i++;
        }
    }

    if(getArchetype().isSpikeRecordingEnabled()) {
        // Add field for spike recording
        // **YUCK** this mechanism needs to be renamed from PointerEGP to RuntimeAlloc
        addField("uint32_t*", "recordSpk",
                 [this](const NeuronGroupInternal &ng, size_t, const MergedRunnerMap &map) 
                 { 
                     return map.findGroup(ng) + "." + getDeviceVarPrefix() + "recordSpk";
                 },
                 FieldType::PointerEGP);
    }

    if(getArchetype().isSpikeEventRecordingEnabled()) {
        // Add field for spike event recording
        // **YUCK** this mechanism needs to be renamed from PointerEGP to RuntimeAlloc
        addField("uint32_t*", "recordSpkEvent",
                 [this](const NeuronGroupInternal &ng, size_t, const MergedRunnerMap &map)
                 {
                     return map.findGroup(ng) + "." + getDeviceVarPrefix() + "recordSpkEvent";
                 },
                 FieldType::PointerEGP);
    }

}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous(size_t childIndex, const std::string &paramName) const
{
    return (isInSynWUMParamReferenced(childIndex, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedInSynWithPostCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous(size_t childIndex, const std::string &paramName) const
{
    return (isInSynWUMParamReferenced(childIndex, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedInSynWithPostCode,
                                           [](const SynapseGroupInternal *s) { return s->getWUDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous(size_t childIndex, const std::string &paramName) const
{
    return (isOutSynWUMParamReferenced(childIndex, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedOutSynWithPreCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous(size_t childIndex, const std::string &paramName) const
{
    return (isOutSynWUMParamReferenced(childIndex, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedOutSynWithPreCode,
                                           [](const SynapseGroupInternal *s) { return s->getWUDerivedParams(); }));
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronUpdateGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with generic neuron group data
    updateBaseHash(false, hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getHashDigest(), hash);

    // Update hash with each group's parameters and derived parameters
    updateHash([](const NeuronGroupInternal &g) { return g.getParams(); }, hash);
    updateHash([](const NeuronGroupInternal &g) { return g.getDerivedParams(); }, hash);
        
    // Loop through child incoming synapse groups with postsynaptic update code
    for(size_t i = 0; i < getSortedArchetypeInSynWithPostCode().size(); i++) {
        updateChildParamHash<NeuronUpdateGroupMerged>(m_SortedInSynWithPostCode, i, &NeuronUpdateGroupMerged::isInSynWUMParamReferenced, 
                                                      &SynapseGroupInternal::getWUParams, hash);
        updateChildDerivedParamHash<NeuronUpdateGroupMerged>(m_SortedInSynWithPostCode, i, &NeuronUpdateGroupMerged::isInSynWUMParamReferenced, 
                                                             &SynapseGroupInternal::getWUDerivedParams, hash);
    }

    // Loop through child outgoing synapse groups with presynaptic update code
    for(size_t i = 0; i < getSortedArchetypeOutSynWithPreCode().size(); i++) {
        updateChildParamHash<NeuronUpdateGroupMerged>(m_SortedOutSynWithPreCode, i, &NeuronUpdateGroupMerged::isOutSynWUMParamReferenced, 
                                                      &SynapseGroupInternal::getWUParams, hash);
        updateChildDerivedParamHash<NeuronUpdateGroupMerged>( m_SortedOutSynWithPreCode, i, &NeuronUpdateGroupMerged::isOutSynWUMParamReferenced, 
                                                             &SynapseGroupInternal::getWUDerivedParams, hash);
    }

    return hash.get_digest();
}
//--------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateNeuronUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs,
                                                   BackendBase::GroupHandler<NeuronUpdateGroupMerged> genEmitTrueSpike,
                                                   BackendBase::GroupHandler<NeuronUpdateGroupMerged> genEmitSpikeLikeEvent) const
{
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const NeuronModels::Base *nm = getArchetype().getNeuronModel();

    // Generate code to copy neuron state into local variable
    for(const auto &v : nm->getVars()) {
        if(v.access & VarAccessMode::READ_ONLY) {
            os << "const ";
        }
        os << v.type << " l" << v.name << " = group->" << v.name << "[";
        const bool delayed = (getArchetype().isVarQueueRequired(v.name) && getArchetype().isDelayRequired());
        os << getReadVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "];" << std::endl;
    }

    // Also read spike and spike-like-event times into local variables if required
    if(getArchetype().isSpikeTimeRequired()) {
        os << "const " << model.getTimePrecision() << " lsT = group->sT[";
        os << getReadVarIndex(getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
    }
    if(getArchetype().isPrevSpikeTimeRequired()) {
        os << "const " << model.getTimePrecision() << " lprevST = group->prevST[";
        os << getReadVarIndex(getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
    }
    if(getArchetype().isSpikeEventTimeRequired()) {
        os << "const " << model.getTimePrecision() << " lseT = group->seT[";
        os << getReadVarIndex(getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        os <<  "const " << model.getTimePrecision() << " lprevSET = group->prevSET[";
        os << getReadVarIndex(getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
    }
    os << std::endl;

    // If neuron model sim code references ISyn (could still be the case if there are no incoming synapses)
    // OR any incoming synapse groups have post synaptic models which reference $(Isyn), declare it
    if (nm->getSimCode().find("$(Isyn)") != std::string::npos ||
        std::any_of(getArchetype().getFusedPSMInSyn().cbegin(), getArchetype().getFusedPSMInSyn().cend(),
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

    if(getArchetype().isSpikeTimeRequired()) {
        neuronSubs.addVarSubstitution("sT", "lsT");
    }
    if(getArchetype().isPrevSpikeTimeRequired()) {
        neuronSubs.addVarSubstitution("prev_sT", "lprevST");
    }
    if(getArchetype().isSpikeEventTimeRequired()) {
        neuronSubs.addVarSubstitution("seT", "lseT");
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        neuronSubs.addVarSubstitution("prev_seT", "lprevSET");
    }
    neuronSubs.addVarNameSubstitution(nm->getAdditionalInputVars());
    addNeuronModelSubstitutions(neuronSubs);

    // Initialise any additional input variables supported by neuron model
    for (const auto &a : nm->getAdditionalInputVars()) {
        // Apply substitutions to value
        std::string value = a.value;
        neuronSubs.applyCheckUnreplaced(value, "neuron additional input var : merged" + std::to_string(getIndex()));
        value = ensureFtype(value, modelMerged.getModel().getPrecision());

        os << a.type << " " << a.name << " = " << value << ";" << std::endl;
    }

    // Loop through incoming synapse groups
    for(size_t i = 0; i < getSortedArchetypeMergedInSyns().size(); i++) {
        CodeStream::Scope b(os);

        const auto *sg = getSortedArchetypeMergedInSyns().at(i);
        const auto *psm = sg->getPSModel();

        os << "// pull inSyn values in a coalesced access" << std::endl;
        os << model.getPrecision() << " linSyn = group->inSynInSyn" << i << "[";
        os << getVarIndex(batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;

        // If dendritic delay is required
        if (sg->isDendriticDelayRequired()) {
            // Get reference to dendritic delay buffer input for this timestep
            os << backend.getPointerPrefix() << model.getPrecision() << " *denDelayFront = ";
            os << "&group->denDelayInSyn" << i << "[(*group->denDelayPtrInSyn" << i << " * group->numNeurons) + ";
            os << getVarIndex(batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;

            // Add delayed input from buffer into inSyn
            os << "linSyn += *denDelayFront;" << std::endl;

            // Zero delay buffer slot
            os << "*denDelayFront = " << model.scalarExpr(0.0) << ";" << std::endl;
        }

        // Pull postsynaptic model variables in a coalesced access
        for (const auto &v : psm->getVars()) {
            if(v.access & VarAccessMode::READ_ONLY) {
                os << "const ";
            }
            os << v.type << " lps" << v.name << " = group->" << v.name << "InSyn" << i << "[";
            os << getVarIndex(batchSize, getVarAccessDuplication(v.access), neuronSubs["id"]) << "];" << std::endl;
        }

        Substitutions inSynSubs(&neuronSubs);
        inSynSubs.addVarSubstitution("inSyn", "linSyn");
        
        // Allow synapse group's PS output var to override what Isyn points to
        inSynSubs.addVarSubstitution("Isyn", sg->getPSTargetVar(), true);
        inSynSubs.addVarNameSubstitution(psm->getVars(), "", "lps");

        inSynSubs.addParamValueSubstitution(psm->getParamNames(), sg->getPSParams(),
                                            [i, this](const std::string &p) { return isPSMParamHeterogeneous(i, p);  },
                                            "", "group->", "InSyn" + std::to_string(i));
        inSynSubs.addVarValueSubstitution(psm->getDerivedParams(), sg->getPSDerivedParams(),
                                            [i, this](const std::string &p) { return isPSMDerivedParamHeterogeneous(i, p);  },
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
        os << getVarIndex(batchSize, VarAccessDuplication::DUPLICATE, inSynSubs["id"]) << "] = linSyn;" << std::endl;

        // Copy any non-readonly postsynaptic model variables back to global state variables dd_V etc
        for (const auto &v : psm->getVars()) {
            if(v.access & VarAccessMode::READ_WRITE) {
                os << "group->" << v.name << "InSyn" << i << "[";
                os << getVarIndex(batchSize, getVarAccessDuplication(v.access), inSynSubs["id"]) << "]" << " = lps" << v.name << ";" << std::endl;
            }
        }
    }

    // Loop through outgoing synapse groups with presynaptic output
    for(size_t i = 0; i < getSortedArchetypeMergedPreOutputOutSyns().size(); i++) {
        CodeStream::Scope b(os);
        const auto *sg = getSortedArchetypeMergedPreOutputOutSyns().at(i);
     
        os << sg->getPreTargetVar() << "+= ";
        os << "group->revInSynOutSyn" << i << "[";
        os << getVarIndex(batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
        os << "group->revInSynOutSyn" << i << "[";
        os << getVarIndex(batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "]= 0.0;" << std::endl;        
    }

    // Loop through all of neuron group's current sources
    for(size_t i = 0; i < getSortedArchetypeCurrentSources().size(); i++) {
        const auto *cs = getSortedArchetypeCurrentSources().at(i);

        os << "// current source " << i << std::endl;
        CodeStream::Scope b(os);

        const auto *csm = cs->getCurrentSourceModel();

        // Read current source variables into registers
        for(const auto &v : csm->getVars()) {
            if(v.access & VarAccessMode::READ_ONLY) {
                os << "const ";
            }
            os << v.type << " lcs" << v.name << " = " << "group->" << v.name << "CS" << i << "[";
            os << getVarIndex(batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "];" << std::endl;
        }

        Substitutions currSourceSubs(&popSubs);
        currSourceSubs.addFuncSubstitution("injectCurrent", 1, "Isyn += $(0)");
        currSourceSubs.addVarNameSubstitution(csm->getVars(), "", "lcs");
        currSourceSubs.addParamValueSubstitution(csm->getParamNames(), cs->getParams(),
                                                    [i, this](const std::string &p) { return isCurrentSourceParamHeterogeneous(i, p);  },
                                                    "", "group->", "CS" + std::to_string(i));
        currSourceSubs.addVarValueSubstitution(csm->getDerivedParams(), cs->getDerivedParams(),
                                                [i, this](const std::string &p) { return isCurrentSourceDerivedParamHeterogeneous(i, p);  },
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
                os << getVarIndex(batchSize, getVarAccessDuplication(v.access), currSourceSubs["id"]) << "] = lcs" << v.name << ";" << std::endl;
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

        neuronSubs.applyCheckUnreplaced(thCode, "thresholdConditionCode : merged" + std::to_string(getIndex()));
        thCode= ensureFtype(thCode, model.getPrecision());

        if (!nm->getSupportCode().empty() && !backend.supportsNamespace()) {
            thCode = disambiguateNamespaceFunction(nm->getSupportCode(), thCode, modelMerged.getNeuronUpdateSupportCodeNamespace(nm->getSupportCode()));
        }

        if (nm->isAutoRefractoryRequired()) {
            os << "const bool oldSpike = (" << thCode << ");" << std::endl;
        }
    }
    // Otherwise, if any outgoing synapse groups have spike-processing code
    /*else if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                        [](const SynapseGroupInternal *sg){ return !sg->getWUModel()->getSimCode().empty(); }))
    {
        LOGW_CODE_GEN << "No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << getName() << "\" was provided. There will be no spikes detected in this population!";
    }*/

    os << "// calculate membrane potential" << std::endl;
    std::string sCode = nm->getSimCode();
    neuronSubs.applyCheckUnreplaced(sCode, "simCode : merged" + std::to_string(getIndex()));
    sCode = ensureFtype(sCode, model.getPrecision());

    if (!nm->getSupportCode().empty() && !backend.supportsNamespace()) {
        sCode = disambiguateNamespaceFunction(nm->getSupportCode(), sCode, modelMerged.getNeuronUpdateSupportCodeNamespace(nm->getSupportCode()));
    }

    os << sCode << std::endl;

    // Generate var update for outgoing synaptic populations with presynaptic update code
    generateWUVarUpdate(os, popSubs, "WUPre", modelMerged.getModel().getPrecision(), "_pre", true, batchSize,
                        getSortedArchetypeOutSynWithPreCode(), &SynapseGroupInternal::getDelaySteps,
                        &WeightUpdateModels::Base::getPreVars, &WeightUpdateModels::Base::getPreDynamicsCode,
                        &NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous,
                        &NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous);


    // Generate var update for incoming synaptic populations with postsynaptic code
    generateWUVarUpdate(os, popSubs, "WUPost", modelMerged.getModel().getPrecision(), "_post", true, batchSize,
                        getSortedArchetypeInSynWithPostCode(), &SynapseGroupInternal::getBackPropDelaySteps,
                        &WeightUpdateModels::Base::getPostVars, &WeightUpdateModels::Base::getPostDynamicsCode,
                        &NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous,
                        &NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous);

    // look for spike type events first.
    if (getArchetype().isSpikeEventRequired()) {
        // Create local variable
        os << "bool spikeLikeEvent = false;" << std::endl;

        // Loop through outgoing synapse populations that will contribute to event condition code
        size_t i = 0;
        for(const auto &spkEventCond : getArchetype().getSpikeEventCondition()) {
            // Replace of parameters, derived parameters and extraglobalsynapse parameters
            Substitutions spkEventCondSubs(&popSubs);

            // If this spike event condition requires synapse state
            if(spkEventCond.synapseStateInThresholdCode) {
                // Substitute EGPs
                spkEventCondSubs.addVarNameSubstitution(spkEventCond.synapseGroup->getWUModel()->getExtraGlobalParams(), "", "group->", "EventThresh" + std::to_string(i));

                // Substitute presynaptic variables
                const bool delayed = (spkEventCond.synapseGroup->getDelaySteps() != NO_DELAY);
                spkEventCondSubs.addVarNameSubstitution(spkEventCond.synapseGroup->getWUModel()->getPreVars(), "", "group->",
                                                        [&popSubs, batchSize, delayed, i, this](VarAccess a) 
                                                        { 
                                                            return "EventThresh" + std::to_string(i) + "[" + getReadVarIndex(delayed, batchSize, getVarAccessDuplication(a), popSubs["id"]) + "]";
                                                        });
                i++;
            }
            addNeuronModelSubstitutions(spkEventCondSubs, "_pre");

            std::string eCode = spkEventCond.eventThresholdCode;
            spkEventCondSubs.applyCheckUnreplaced(eCode, "neuronSpkEvntCondition : merged" + std::to_string(getIndex()));
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
            genEmitSpikeLikeEvent(os, *this, popSubs);
        }

        // If spike-like-event timing is required and they aren't updated after update, copy spike-like-event time from register
        if(getArchetype().isDelayRequired() && (getArchetype().isSpikeEventTimeRequired() || getArchetype().isPrevSpikeEventTimeRequired())) {
            os << "else";
            CodeStream::Scope b(os);

            if(getArchetype().isSpikeEventTimeRequired()) {
                os << "group->seT[" << getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lseT;" << std::endl;
            }
            if(getArchetype().isPrevSpikeEventTimeRequired()) {
                os << "group->prevSET[" << getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lprevSET;" << std::endl;
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
            genEmitTrueSpike(os, *this, popSubs);

            // add after-spike reset if provided
            if (!nm->getResetCode().empty()) {
                std::string rCode = nm->getResetCode();
                neuronSubs.applyCheckUnreplaced(rCode, "resetCode : merged" + std::to_string(getIndex()));
                rCode = ensureFtype(rCode, model.getPrecision());

                os << "// spike reset code" << std::endl;
                os << rCode << std::endl;
            }
        }

        // Spike triggered variables don't need to be copied
        // if delay isn't required as there's only one copy of them
        if(getArchetype().isDelayRequired()) {
            // **FIXME** there is a corner case here where, if pre or postsynaptic variables have no update code
            // but there are delays they won't get copied. It might make more sense (and tidy up several things
            // to instead build merged neuron update groups based on inSynWithPostVars/outSynWithPreVars instead.
            
            // Are there any outgoing synapse groups with presynaptic code
            // which have axonal delay and no presynaptic dynamics
            const bool preVars = std::any_of(getSortedArchetypeOutSynWithPreCode().cbegin(), getSortedArchetypeOutSynWithPreCode().cend(),
                                             [](const SynapseGroupInternal *sg)
                                             {
                                                 return ((sg->getDelaySteps() != NO_DELAY)
                                                         && sg->getWUModel()->getPreDynamicsCode().empty());
                                             });

            // Are there any incoming synapse groups with postsynaptic code
            // which have back-propagation delay and no postsynaptic dynamics
            const bool postVars = std::any_of(getSortedArchetypeInSynWithPostCode().cbegin(), getSortedArchetypeInSynWithPostCode().cend(),
                                              [](const SynapseGroupInternal *sg)
                                              {
                                                  return ((sg->getBackPropDelaySteps() != NO_DELAY)
                                                          && sg->getWUModel()->getPostDynamicsCode().empty());
                                              });

            // If spike times, presynaptic variables or postsynaptic variables are required, add if clause
            if(getArchetype().isSpikeTimeRequired() || getArchetype().isPrevSpikeTimeRequired() || preVars || postVars) {
                os << "else";
                CodeStream::Scope b(os);

                // If spike times are required, copy times from register
                if(getArchetype().isSpikeTimeRequired()) {
                    os << "group->sT[" << getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lsT;" << std::endl;
                }

                // If previous spike times are required, copy times from register
                if(getArchetype().isPrevSpikeTimeRequired()) {
                    os << "group->prevST[" << getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lprevST;" << std::endl;
                }

                // Loop through outgoing synapse groups with some sort of presynaptic code
                for(size_t i = 0; i < getSortedArchetypeOutSynWithPreCode().size(); i++) {
                    const auto *sg = getSortedArchetypeOutSynWithPreCode().at(i);
                    // If this group has a delay and no presynaptic dynamics (which will already perform this copying)
                    if(sg->getDelaySteps() != NO_DELAY && sg->getWUModel()->getPreDynamicsCode().empty()) {
                        // Loop through variables and copy between read and write delay slots
                        for(const auto &v : sg->getWUModel()->getPreVars()) {
                            if(v.access & VarAccessMode::READ_WRITE) {
                                os << "group->" << v.name << "WUPre" << i << "[" << getWriteVarIndex(true, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "] = ";
                                os << "group->" << v.name << "WUPre" << i << "[" << getReadVarIndex(true, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "];" << std::endl;
                            }
                        }
                    }
                }

                // Loop through outgoing synapse groups with some sort of postsynaptic code
                for(size_t i = 0; i < getSortedArchetypeInSynWithPostCode().size(); i++) {
                    const auto *sg = getSortedArchetypeInSynWithPostCode().at(i);
                    // If this group has a delay and no postsynaptic dynamics (which will already perform this copying)
                    if(sg->getBackPropDelaySteps() != NO_DELAY && sg->getWUModel()->getPostDynamicsCode().empty()) {
                        // Loop through variables and copy between read and write delay slots
                        for(const auto &v : sg->getWUModel()->getPostVars()) {
                            if(v.access & VarAccessMode::READ_WRITE) {
                                os << "group->" << v.name << "WUPost" << i << "[" << getWriteVarIndex(true, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "] = ";
                                os << "group->" << v.name << "WUPost" << i << "[" << getReadVarIndex(true, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "];" << std::endl;
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
        const bool delayed = (getArchetype().isVarQueueRequired(v.name) && getArchetype().isDelayRequired());
        if((v.access & VarAccessMode::READ_WRITE) || delayed) {
            os << "group->" << v.name << "[";
            os << getWriteVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "] = l" << v.name << ";" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateWUVarUpdate(const BackendBase&, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Generate var update for outgoing synaptic populations with presynaptic update code
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    generateWUVarUpdate(os, popSubs, "WUPre", modelMerged.getModel().getPrecision(), "_pre", false, batchSize,
                        getSortedArchetypeOutSynWithPreCode(), &SynapseGroupInternal::getDelaySteps,
                        &WeightUpdateModels::Base::getPreVars, &WeightUpdateModels::Base::getPreSpikeCode,
                        &NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous, 
                        &NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous);
    

    // Generate var update for incoming synaptic populations with postsynaptic code
    generateWUVarUpdate(os, popSubs, "WUPost", modelMerged.getModel().getPrecision(), "_post", false, batchSize,
                        getSortedArchetypeInSynWithPostCode(), &SynapseGroupInternal::getBackPropDelaySteps,
                        &WeightUpdateModels::Base::getPostVars, &WeightUpdateModels::Base::getPostSpikeCode,
                        &NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous,
                        &NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous);
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1) ? "" : "batchOffset + ") + index;
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getReadVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    if(delay) {
        return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1) ? "readDelayOffset + " : "readBatchDelayOffset + ") + index;
    }
    else {
        return getVarIndex(batchSize, varDuplication, index);
    }
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getWriteVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    if(delay) {
        return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1) ? "writeDelayOffset + " : "writeBatchDelayOffset + ") + index;
    }
    else {
        return getVarIndex(batchSize, varDuplication, index);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateWUVar(const std::string &fieldPrefixStem, const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                                            Models::Base::VarVec (WeightUpdateModels::Base::*getVars)(void) const,
                                            bool(NeuronUpdateGroupMerged::*isParamHeterogeneous)(size_t, const std::string&) const,
                                            bool(NeuronUpdateGroupMerged::*isDerivedParamHeterogeneous)(size_t, const std::string&) const)
{
    // Loop through synapse groups
    const auto &archetypeSyns = sortedSyn.front();
    for(size_t i = 0; i < archetypeSyns.size(); i++) {
        const auto *sg = archetypeSyns.at(i);

        // Loop through variables
        const auto vars = std::invoke(getVars, sg->getWUModel());
        for(size_t v = 0; v < vars.size(); v++) {
            // Add pointers to state variable
            const auto var = vars[v];
            assert(!Utils::isTypePointer(var.type));
            addChildPointerField(var.type, var.name, sortedSyn, i, fieldPrefixStem);
        }

        // Add any heterogeneous parameters
        addHeterogeneousChildParams<NeuronUpdateGroupMerged>(sg->getWUModel()->getParamNames(), sortedSyn, i, fieldPrefixStem,
                                                             isParamHeterogeneous, &SynapseGroupInternal::getWUParams);

        // Add any heterogeneous derived parameters
        addHeterogeneousChildDerivedParams<NeuronUpdateGroupMerged>(sg->getWUModel()->getDerivedParams(), sortedSyn, i, fieldPrefixStem,
                                                                    isDerivedParamHeterogeneous, &SynapseGroupInternal::getWUDerivedParams);

        // Add EGPs
        addChildEGPs(sg->getWUModel()->getExtraGlobalParams(), sortedSyn, i, fieldPrefixStem);
    }
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isInSynWUMParamReferenced(size_t childIndex, const std::string &paramName) const
{
    const auto *wum = getSortedArchetypeInSynWithPostCode().at(childIndex)->getWUModel();
    return isParamReferenced({wum->getPostSpikeCode(), wum->getPostDynamicsCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isOutSynWUMParamReferenced(size_t childIndex, const std::string &paramName) const
{
    const auto *wum = getSortedArchetypeOutSynWithPreCode().at(childIndex)->getWUModel();
    return isParamReferenced({wum->getPreSpikeCode(), wum->getPreDynamicsCode()}, paramName);
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::addNeuronModelSubstitutions(Substitutions &substitution, const std::string &sourceSuffix, const std::string &destSuffix) const
{
    const NeuronModels::Base *nm = getArchetype().getNeuronModel();
    substitution.addVarNameSubstitution(nm->getVars(), sourceSuffix, "l", destSuffix);
    substitution.addParamValueSubstitution(nm->getParamNames(), getArchetype().getParams(), 
                                           [this](const std::string &p) { return isParamHeterogeneous(p);  },
                                           sourceSuffix, "group->");
    substitution.addVarValueSubstitution(nm->getDerivedParams(), getArchetype().getDerivedParams(), 
                                         [this](const std::string &p) { return isDerivedParamHeterogeneous(p);  },
                                         sourceSuffix, "group->");
    substitution.addVarNameSubstitution(nm->getExtraGlobalParams(), sourceSuffix, "group->");
}
//--------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateWUVarUpdate(CodeStream &os, const Substitutions &popSubs,
                                                  const std::string &fieldPrefixStem, const std::string &precision, const std::string &sourceSuffix, 
                                                  bool useLocalNeuronVars, unsigned int batchSize, 
                                                  const std::vector<SynapseGroupInternal*> &archetypeSyn,
                                                  unsigned int(SynapseGroupInternal::*getDelaySteps)(void) const,
                                                  Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                                                  std::string(WeightUpdateModels::Base::*getCode)(void) const,
                                                  bool(NeuronUpdateGroupMerged::*isParamHeterogeneous)(size_t, const std::string&) const,
                                                  bool(NeuronUpdateGroupMerged::*isDerivedParamHeterogeneous)(size_t, const std::string&) const) const
{
    // Loop through synaptic populations
    for(size_t i = 0; i < archetypeSyn.size(); i++) {
        const SynapseGroupInternal *sg = archetypeSyn[i];

        // If this code string isn't empty
        std::string code = std::invoke(getCode, sg->getWUModel());
        if(!code.empty()) {
            Substitutions subs(&popSubs);
            CodeStream::Scope b(os);

            // Fetch variables from global memory
            os << "// perform WUM update required for merged" << i << std::endl;
            const auto vars = std::invoke(getVars, sg->getWUModel());
            const bool delayed = (std::invoke(getDelaySteps, sg) != NO_DELAY);
            for(const auto &v : vars) {
                if(v.access & VarAccessMode::READ_ONLY) {
                    os << "const ";
                }
                os << v.type << " l" << v.name << " = group->" << v.name << fieldPrefixStem << i << "[";
                os << getReadVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), subs["id"]) << "];" << std::endl;
            }

            subs.addParamValueSubstitution(sg->getWUModel()->getParamNames(), sg->getWUParams(),
                                           [i, isParamHeterogeneous , this](const std::string &p) { return std::invoke(isParamHeterogeneous, this, i, p); },
                                           "", "group->", fieldPrefixStem + std::to_string(i));
            subs.addVarValueSubstitution(sg->getWUModel()->getDerivedParams(), sg->getWUDerivedParams(),
                                         [i, isDerivedParamHeterogeneous, this](const std::string &p) { return std::invoke(isDerivedParamHeterogeneous, this, i, p); },
                                         "", "group->", fieldPrefixStem + std::to_string(i));
            subs.addVarNameSubstitution(sg->getWUModel()->getExtraGlobalParams(), "", "group->", fieldPrefixStem + std::to_string(i));
            subs.addVarNameSubstitution(vars, "", "l");

            neuronSubstitutionsInSynapticCode(subs, &getArchetype(), "", sourceSuffix, "", "", "", useLocalNeuronVars,
                                              [this](const std::string &p) { return this->isParamHeterogeneous(p); },
                                              [this](const std::string &p) { return this->isDerivedParamHeterogeneous(p); },
                                              [&subs, batchSize, this](bool delay, VarAccessDuplication varDuplication) 
                                              {
                                                  return getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                              },
                                              [&subs, batchSize, this](bool delay, VarAccessDuplication varDuplication) 
                                              { 
                                                  return getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
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
                    os << getWriteVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), subs["id"]) << "] = l" << v.name << ";" << std::endl;
                }
            }
        }
    }
}

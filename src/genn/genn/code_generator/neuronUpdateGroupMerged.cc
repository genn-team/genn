#include "code_generator/neuronUpdateGroupMerged.h"

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
    orderNeuronGroupChildren(m_SortedInSynWithPostCode, &NeuronGroupInternal::getFusedInSynWithPostCode,
                             &SynapseGroupInternal::getWUPostHashDigest);

    // Build vector of vectors containing each child group's outgoing synapse groups
    // with presynaptic synaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedOutSynWithPreCode, &NeuronGroupInternal::getFusedOutSynWithPreCode,
                             &SynapseGroupInternal::getWUPreHashDigest);

    // Generate struct fields for incoming synapse groups with postsynaptic update code
    generateWUVar(backend, "WUPost", m_SortedInSynWithPostCode,
                  &WeightUpdateModels::Base::getPostVars, &NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous,
                  &NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous,
                  &SynapseGroupInternal::getFusedWUPostVarSuffix);

    // Generate struct fields for outgoing synapse groups with presynaptic update code
    generateWUVar(backend, "WUPre", m_SortedOutSynWithPreCode,
                  &WeightUpdateModels::Base::getPreVars, &NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous,
                  &NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous,
                  &SynapseGroupInternal::getFusedWUPreVarSuffix);

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
                    const std::string prefix = isPointer ? backend.getDeviceVarPrefix() : "";
                    addField(egp.type, egp.name + "EventThresh" + std::to_string(i),
                             [eventThresholdSGs, prefix, egp, i](const NeuronGroupInternal &, size_t groupIndex)
                             {
                                 return prefix + egp.name + eventThresholdSGs.at(groupIndex).at(i)->getName();
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
                             [&backend, eventThresholdSGs, var, i](const NeuronGroupInternal &, size_t groupIndex)
                             {
                                 return backend.getDeviceVarPrefix() + var.name + eventThresholdSGs.at(groupIndex).at(i)->getName();
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
                 [&backend](const NeuronGroupInternal &ng, size_t) 
                 { 
                     return backend.getDeviceVarPrefix() + "recordSpk" + ng.getName(); 
                 },
                 FieldType::PointerEGP);
    }

    if(getArchetype().isSpikeEventRecordingEnabled()) {
        // Add field for spike event recording
        // **YUCK** this mechanism needs to be renamed from PointerEGP to RuntimeAlloc
        addField("uint32_t*", "recordSpkEvent",
                 [&backend](const NeuronGroupInternal &ng, size_t)
                 {
                     return backend.getDeviceVarPrefix() + "recordSpkEvent" + ng.getName(); 
                 },
                 FieldType::PointerEGP);
    }

}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    return (isInSynWUMParamReferenced(childIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedInSynWithPostCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    return (isInSynWUMDerivedParamReferenced(childIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedInSynWithPostCode,
                                           [](const SynapseGroupInternal *s) { return s->getWUDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    return (isOutSynWUMParamReferenced(childIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedOutSynWithPreCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    return (isOutSynWUMDerivedParamReferenced(childIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedOutSynWithPreCode,
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
        updateChildDerivedParamHash<NeuronUpdateGroupMerged>(m_SortedInSynWithPostCode, i, &NeuronUpdateGroupMerged::isInSynWUMDerivedParamReferenced, 
                                                             &SynapseGroupInternal::getWUDerivedParams, hash);
    }

    // Loop through child outgoing synapse groups with presynaptic update code
    for(size_t i = 0; i < getSortedArchetypeOutSynWithPreCode().size(); i++) {
        updateChildParamHash<NeuronUpdateGroupMerged>(m_SortedOutSynWithPreCode, i, &NeuronUpdateGroupMerged::isOutSynWUMParamReferenced, 
                                                      &SynapseGroupInternal::getWUParams, hash);
        updateChildDerivedParamHash<NeuronUpdateGroupMerged>( m_SortedOutSynWithPreCode, i, &NeuronUpdateGroupMerged::isOutSynWUMDerivedParamReferenced, 
                                                             &SynapseGroupInternal::getWUDerivedParams, hash);
    }

    return hash.get_digest();
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
void NeuronUpdateGroupMerged::generateWUVar(const BackendBase &backend,  const std::string &fieldPrefixStem, 
                                            const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                                            Models::Base::VarVec (WeightUpdateModels::Base::*getVars)(void) const,
                                            bool(NeuronUpdateGroupMerged::*isParamHeterogeneous)(size_t, size_t) const,
                                            bool(NeuronUpdateGroupMerged::*isDerivedParamHeterogeneous)(size_t, size_t) const,
                                            const std::string&(SynapseGroupInternal::*getFusedVarSuffix)(void) const)
{
    // Loop through synapse groups
    const auto &archetypeSyns = sortedSyn.front();
    for(size_t i = 0; i < archetypeSyns.size(); i++) {
        const auto *sg = archetypeSyns.at(i);

        // Loop through variables
        const auto vars = (sg->getWUModel()->*getVars)();
        for(size_t v = 0; v < vars.size(); v++) {
            // Add pointers to state variable
            const auto var = vars[v];
            assert(!Utils::isTypePointer(var.type));
            addField(var.type + "*", var.name + fieldPrefixStem + std::to_string(i),
                     [i, var, &backend, &sortedSyn, getFusedVarSuffix](const NeuronGroupInternal &, size_t groupIndex)
                     {
                         const std::string &varMergeSuffix = (sortedSyn.at(groupIndex).at(i)->*getFusedVarSuffix)();
                         return backend.getDeviceVarPrefix() + var.name + varMergeSuffix;
                     });
        }

        // Add any heterogeneous parameters
        addHeterogeneousChildParams<NeuronUpdateGroupMerged>(sg->getWUModel()->getParamNames(), sortedSyn, i, fieldPrefixStem,
                                                             isParamHeterogeneous, &SynapseGroupInternal::getWUParams);

        // Add any heterogeneous derived parameters
        addHeterogeneousChildDerivedParams<NeuronUpdateGroupMerged>(sg->getWUModel()->getDerivedParams(), sortedSyn, i, fieldPrefixStem,
                                                                    isDerivedParamHeterogeneous, &SynapseGroupInternal::getWUDerivedParams);

        // Add EGPs
        addChildEGPs(sg->getWUModel()->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), fieldPrefixStem,
                     [&sortedSyn](size_t groupIndex, size_t childIndex)
                     {
                         return sortedSyn.at(groupIndex).at(childIndex)->getName();
                     });
    }
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isInSynWUMParamReferenced(size_t childIndex, size_t paramIndex) const
{
    const auto *wum = getSortedArchetypeInSynWithPostCode().at(childIndex)->getWUModel();
    const std::string paramName = wum->getParamNames().at(paramIndex);
    return isParamReferenced({wum->getPostSpikeCode(), wum->getPostDynamicsCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isInSynWUMDerivedParamReferenced(size_t childIndex, size_t paramIndex) const
{
    const auto *wum = getSortedArchetypeInSynWithPostCode().at(childIndex)->getWUModel();
    const std::string derivedParamName = wum->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({wum->getPostSpikeCode(), wum->getPostDynamicsCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isOutSynWUMParamReferenced(size_t childIndex, size_t paramIndex) const
{
    const auto *wum = getSortedArchetypeOutSynWithPreCode().at(childIndex)->getWUModel();
    const std::string paramName = wum->getParamNames().at(paramIndex);
    return isParamReferenced({wum->getPreSpikeCode(), wum->getPreDynamicsCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isOutSynWUMDerivedParamReferenced(size_t childIndex, size_t paramIndex) const
{
    const auto *wum = getSortedArchetypeOutSynWithPreCode().at(childIndex)->getWUModel();
    const std::string derivedParamName = wum->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({wum->getPreSpikeCode(), wum->getPreDynamicsCode()}, derivedParamName);
}

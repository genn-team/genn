#include "code_generator/groupMerged.h"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"

using namespace CodeGenerator;

//----------------------------------------------------------------------------
// CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronSpikeQueueUpdateGroupMerged::name = "NeuronSpikeQueueUpdate";
//----------------------------------------------------------------------------
NeuronSpikeQueueUpdateGroupMerged::NeuronSpikeQueueUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                     const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   RuntimeGroupMerged<NeuronGroupInternal>(index, precision, backend, groups)
{
    if(getArchetype().isDelayRequired()) {
        addPointerField("unsigned int", "spkQuePtr", "", true);
    } 

    addPointerField("unsigned int", "spkCnt");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField("unsigned int", "spkCntEvnt");
    }
}
//----------------------------------------------------------------------------
void NeuronSpikeQueueUpdateGroupMerged::genMergedGroupSpikeCountReset(CodeStream &os, unsigned int batchSize) const
{
    if(getArchetype().isSpikeEventRequired()) {
        if(getArchetype().isDelayRequired()) {
            os << "group->spkCntEvnt[*group->spkQuePtr";
            if(batchSize > 1) {
                os << " + (batch * " << getArchetype().getNumDelaySlots() << ")";
            }
            os << "] = 0; " << std::endl;
        }
        else {
            os << "group->spkCntEvnt[" << ((batchSize > 1) ? "batch" : "0") << "] = 0;" << std::endl;
        }
    }

    if(getArchetype().isTrueSpikeRequired() && getArchetype().isDelayRequired()) {
        os << "group->spkCnt[*group->spkQuePtr";
        if(batchSize > 1) {
            os << " + (batch * " << getArchetype().getNumDelaySlots() << ")";
        }
        os << "] = 0; " << std::endl;
    }
    else {
        os << "group->spkCnt[" << ((batchSize > 1) ? "batch" : "0") << "] = 0;" << std::endl;
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronPrevSpikeTimeUpdateGroupMerged::name = "NeuronPrevSpikeTimeUpdate";
//----------------------------------------------------------------------------
NeuronPrevSpikeTimeUpdateGroupMerged::NeuronPrevSpikeTimeUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                                                           const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   RuntimeGroupMerged<NeuronGroupInternal>(index, precision, backend, groups)
{
    if(getArchetype().isDelayRequired()) {
        addPointerField("unsigned int", "spkQuePtr", "", true);
    } 

    addPointerField("unsigned int", "spkCnt");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField("unsigned int", "spkCntEvnt");
    }

    if(getArchetype().isPrevSpikeTimeRequired()) {
        addPointerField("unsigned int", "spk");
        addPointerField(timePrecision, "prevST");
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        addPointerField("unsigned int", "spkEvnt");
        addPointerField(timePrecision, "prevSET");
    }

    if(getArchetype().isDelayRequired()) {
        addField("unsigned int", "numNeurons",
                 [](const NeuronGroupInternal &ng, size_t, const MergedRunnerMap&) 
                 {
                     return std::to_string(ng.getNumNeurons()); 
                 });
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::NeuronGroupMergedBase
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const NeuronGroupInternal &ng) { return ng.getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isDerivedParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const NeuronGroupInternal &ng) { return ng.getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName,
                                      [varName](const NeuronGroupInternal &sg) { return sg.getVarInitialisers().at(varName).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName,
                                      [varName](const NeuronGroupInternal &sg){ return sg.getVarInitialisers().at(varName).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceParamHeterogeneous(size_t childIndex, const std::string &paramName) const
{
    return (isCurrentSourceParamReferenced(childIndex, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedCurrentSources,
                                           [](const CurrentSourceInternal *cs) { return cs->getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceDerivedParamHeterogeneous(size_t childIndex, const std::string &paramName) const
{
    return (isCurrentSourceParamReferenced(childIndex, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedCurrentSources,
                                           [](const CurrentSourceInternal *cs) { return cs->getDerivedParams(); }));
 
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceVarInitParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const
{
    return (isCurrentSourceVarInitParamReferenced(childIndex, varName, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedCurrentSources,
                                           [varName](const CurrentSourceInternal *cs) { return cs->getVarInitialisers().at(varName).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceVarInitDerivedParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const
{
    return (isCurrentSourceVarInitParamReferenced(childIndex, varName, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedCurrentSources,
                                           [varName](const CurrentSourceInternal *cs) { return cs->getVarInitialisers().at(varName).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMParamHeterogeneous(size_t childIndex, const std::string &paramName) const
{
    return (isPSMParamReferenced(childIndex, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedMergedInSyns,
                                           [](const SynapseGroupInternal *inSyn) { return inSyn->getPSParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMDerivedParamHeterogeneous(size_t childIndex, const std::string &paramName) const
{
    return (isPSMParamReferenced(childIndex, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedMergedInSyns, 
                                           [](const SynapseGroupInternal *inSyn) { return inSyn->getPSDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMVarInitParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const
{
    return (isPSMVarInitParamReferenced(childIndex, varName, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedMergedInSyns,
                                           [varName](const SynapseGroupInternal *inSyn){ return inSyn->getPSVarInitialisers().at(varName).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMVarInitDerivedParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const
{    
    return (isPSMVarInitParamReferenced(childIndex, varName, paramName) &&
            isChildParamValueHeterogeneous(childIndex, paramName, m_SortedMergedInSyns,
                                          [varName](const SynapseGroupInternal *inSyn){ return inSyn->getPSVarInitialisers().at(varName).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
NeuronGroupMergedBase::NeuronGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                             bool init, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   RuntimeGroupMerged<NeuronGroupInternal>(index, precision, backend, groups)
{
    // Build vector of vectors containing each child group's merged in syns, ordered to match those of the archetype group
    orderGroupChildren(m_SortedMergedInSyns, &NeuronGroupInternal::getFusedPSMInSyn,
                       init ? &SynapseGroupInternal::getPSInitHashDigest : &SynapseGroupInternal::getPSHashDigest);

    // Build vector of vectors containing each child group's merged out syns with pre output, ordered to match those of the archetype group
    orderGroupChildren(m_SortedMergedPreOutputOutSyns, &NeuronGroupInternal::getFusedPreOutputOutSyn,
                       init ? &SynapseGroupInternal::getPreOutputInitHashDigest : &SynapseGroupInternal::getPreOutputHashDigest);

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    orderGroupChildren(m_SortedCurrentSources, &NeuronGroupInternal::getCurrentSources,
                       init ? &CurrentSourceInternal::getInitHashDigest : &CurrentSourceInternal::getHashDigest);

    addField("unsigned int", "numNeurons",
              [](const NeuronGroupInternal &ng, size_t, const MergedRunnerMap&) { return std::to_string(ng.getNumNeurons()); });

    addPointerField("unsigned int", "spkCnt");
    addPointerField("unsigned int", "spk");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField("unsigned int", "spkCntEvnt");
        addPointerField("unsigned int", "spkEvnt");
    }

    if(getArchetype().isDelayRequired()) {
        addPointerField("unsigned int", "spkQuePtr", "", true);
    }

    if(getArchetype().isSpikeTimeRequired()) {
        addPointerField(timePrecision, "sT");
    }
    if(getArchetype().isSpikeEventTimeRequired()) {
        addPointerField(timePrecision, "seT");
    }

    if(getArchetype().isPrevSpikeTimeRequired()) {
        addPointerField(timePrecision, "prevST");
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        addPointerField(timePrecision, "prevSET", backend.getDeviceVarPrefix() + "prevSET");
    }

    // If this backend initialises population RNGs on device and this group requires on for simulation
    if(backend.isPopulationRNGRequired() && getArchetype().isSimRNGRequired() 
       && (!init || backend.isPopulationRNGInitialisedOnDevice())) 
    {
        addPointerField(backend.getMergedGroupSimRNGType(), "rng", backend.getDeviceVarPrefix() + "rng");
    }

    // Loop through variables
    const NeuronModels::Base *nm = getArchetype().getNeuronModel();
    const auto vars = nm->getVars();
    const auto &varInit = getArchetype().getVarInitialisers();
    for(const auto &var : vars) {
        // If we're not initialising or if there is initialization code for this variable
        if(!init || !varInit.at(var.name).getSnippet()->getCode().empty()) {
            addPointerField(var.type, var.name);
        }

        // If we're initializing, add any var init EGPs to structure
        if(init) {
            addEGPs(varInit.at(var.name).getSnippet()->getExtraGlobalParams(), var.name);
        }
    }

    // If we're generating a struct for initialization
    if(init) {
        // Add heterogeneous var init parameters
        addHeterogeneousVarInitParams<NeuronGroupMergedBase>(
            &NeuronGroupInternal::getVarInitialisers,
            &NeuronGroupMergedBase::isVarInitParamHeterogeneous);

        addHeterogeneousVarInitDerivedParams<NeuronGroupMergedBase>(
            &NeuronGroupInternal::getVarInitialisers,
            &NeuronGroupMergedBase::isVarInitDerivedParamHeterogeneous);
    }
    // Otherwise
    else {
        addEGPs(nm->getExtraGlobalParams());

        // Add heterogeneous neuron model parameters
        addHeterogeneousParams<NeuronGroupMergedBase>(
            getArchetype().getNeuronModel()->getParamNames(), "",
            [](const NeuronGroupInternal &ng) { return ng.getParams(); },
            &NeuronGroupMergedBase::isParamHeterogeneous);

        // Add heterogeneous neuron model derived parameters
        addHeterogeneousDerivedParams<NeuronGroupMergedBase>(
            getArchetype().getNeuronModel()->getDerivedParams(), "",
            [](const NeuronGroupInternal &ng) { return ng.getDerivedParams(); },
            &NeuronGroupMergedBase::isDerivedParamHeterogeneous);
    }

    // Loop through merged synaptic inputs to archetypical neuron group (0) in sorted order
    for(size_t i = 0; i < getSortedArchetypeMergedInSyns().size(); i++) {
        const SynapseGroupInternal *sg = getSortedArchetypeMergedInSyns().at(i);

        // Add pointer to insyn
        addChildPointerField(precision, "inSyn", m_SortedMergedInSyns, i, "InSyn");

        // Add pointer to dendritic delay buffer if required
        if(sg->isDendriticDelayRequired()) {
            addChildPointerField(precision, "denDelay", m_SortedMergedInSyns, i, "InSyn", true);
            addChildPointerField("unsigned int", "denDelayPtr", m_SortedMergedInSyns, i, "InSyn");
        }

        // Loop through variables
        const auto &varInit = sg->getPSVarInitialisers();
        for(const auto &var : sg->getPSModel()->getVars()) {
            // Add pointers to state variable
            if(!init || !varInit.at(var.name).getSnippet()->getCode().empty()) {
                addChildPointerField(var.type, var.name, m_SortedMergedInSyns, i, "InSyn");
            }

            // If we're generating an initialization structure, also add any heterogeneous parameters, derived parameters or extra global parameters required for initializers
            if(init) {
                const auto *varInitSnippet = varInit.at(var.name).getSnippet();
                addHeterogeneousChildVarInitParams(varInitSnippet->getParamNames(), m_SortedMergedInSyns, i, var.name, "InSyn",
                                                    &NeuronGroupMergedBase::isPSMVarInitParamHeterogeneous, &SynapseGroupInternal::getPSVarInitialisers);
                addHeterogeneousChildVarInitDerivedParams(varInitSnippet->getDerivedParams(), m_SortedMergedInSyns, i, var.name, "InSyn",
                                                          &NeuronGroupMergedBase::isPSMVarInitDerivedParamHeterogeneous, &SynapseGroupInternal::getPSVarInitialisers);
                addChildEGPs(varInitSnippet->getExtraGlobalParams(), m_SortedMergedInSyns, i, "InSyn", var.name);
            }
        }

        if(!init) {
            // Add any heterogeneous postsynaptic model parameters
            const auto paramNames = sg->getPSModel()->getParamNames();
            addHeterogeneousChildParams(paramNames, m_SortedMergedInSyns, i, "InSyn",
                                        &NeuronGroupMergedBase::isPSMParamHeterogeneous,
                                        &SynapseGroupInternal::getPSParams);

            // Add any heterogeneous postsynaptic mode derived parameters
            const auto derivedParams = sg->getPSModel()->getDerivedParams();
            addHeterogeneousChildDerivedParams(derivedParams, m_SortedMergedInSyns, i, "InSyn",
                                               &NeuronGroupMergedBase::isPSMDerivedParamHeterogeneous,
                                               &SynapseGroupInternal::getPSDerivedParams);

            // Add EGPs
            addChildEGPs(sg->getPSModel()->getExtraGlobalParams(), m_SortedMergedInSyns, i, "InSyn");
        }
    }

    // Loop through merged output synapses with presynaptic output of archetypical neuron group (0) in sorted order
    for(size_t i = 0; i < getSortedArchetypeMergedPreOutputOutSyns().size(); i++) {
        addChildPointerField(precision, "revInSyn", m_SortedMergedPreOutputOutSyns, i, "OutSyn");
    }
    
    // Loop through current sources to archetypical neuron group in sorted order
    for(size_t i = 0; i < getSortedArchetypeCurrentSources().size(); i++) {
        const auto *cs = getSortedArchetypeCurrentSources().at(i);

        // Loop through variables
        const auto &varInit = cs->getVarInitialisers();
        for(const auto &var : cs->getCurrentSourceModel()->getVars()) {
            // Add pointers to state variable
            if(!init || !varInit.at(var.name).getSnippet()->getCode().empty()) {
                assert(!Utils::isTypePointer(var.type));
                addChildPointerField(var.type, var.name, m_SortedCurrentSources, i, "CS");
            }

            // If we're generating an initialization structure, also add any heterogeneous parameters, derived parameters or extra global parameters required for initializers
            if(init) {
                const auto *varInitSnippet = varInit.at(var.name).getSnippet();
                addHeterogeneousChildVarInitParams(varInitSnippet->getParamNames(), m_SortedCurrentSources, i, var.name, "CS",
                                                   &NeuronGroupMergedBase::isCurrentSourceVarInitParamHeterogeneous, &CurrentSourceInternal::getVarInitialisers);
                addHeterogeneousChildVarInitDerivedParams(varInitSnippet->getDerivedParams(), m_SortedCurrentSources, i, var.name, "CS",
                                                          &NeuronGroupMergedBase::isCurrentSourceVarInitDerivedParamHeterogeneous,  &CurrentSourceInternal::getVarInitialisers);
                addChildEGPs(varInitSnippet->getExtraGlobalParams(), m_SortedCurrentSources, i, "CS", var.name);
            }
        }

        if(!init) {
            // Add any heterogeneous current source parameters
            const auto paramNames = cs->getCurrentSourceModel()->getParamNames();
            addHeterogeneousChildParams(paramNames, m_SortedCurrentSources, i, "CS",
                                        &NeuronGroupMergedBase::isCurrentSourceParamHeterogeneous,
                                        &CurrentSourceInternal::getParams);

            // Add any heterogeneous current source derived parameters
            const auto derivedParams = cs->getCurrentSourceModel()->getDerivedParams();
            addHeterogeneousChildDerivedParams(derivedParams, m_SortedCurrentSources, i, "CS",
                                               &NeuronGroupMergedBase::isCurrentSourceDerivedParamHeterogeneous,
                                               &CurrentSourceInternal::getDerivedParams);

            // Add EGPs
            addChildEGPs(cs->getCurrentSourceModel()->getExtraGlobalParams(), m_SortedCurrentSources, i, "CS");
        }
    }
}
//----------------------------------------------------------------------------
void NeuronGroupMergedBase::updateBaseHash(bool init, boost::uuids::detail::sha1 &hash) const
{
    // Update hash with each group's neuron count
    updateHash([](const NeuronGroupInternal &g) { return g.getNumNeurons(); }, hash);

    // **YUCK** it would be much nicer to have this in derived classes
    if(init) {
        // Loop through child current sources
        for(size_t c = 0; c < getSortedArchetypeCurrentSources().size(); c++) {
            const auto *cs = getSortedArchetypeCurrentSources().at(c);

            // Loop through variables and update hash with variable initialisation parameters and derived parameters
            for(const auto &v : cs->getVarInitialisers()) {
                updateChildVarInitParamsHash(m_SortedCurrentSources, c, v.first,
                                             &NeuronGroupMergedBase::isCurrentSourceVarInitParamReferenced, 
                                             &CurrentSourceInternal::getVarInitialisers, hash);
                updateChildVarInitDerivedParamsHash(m_SortedCurrentSources, c, v.first,
                                                    &NeuronGroupMergedBase::isCurrentSourceVarInitParamReferenced, 
                                                    &CurrentSourceInternal::getVarInitialisers, hash);
            }
        }

        // Loop through child merged insyns
        for(size_t c = 0; c < getSortedArchetypeMergedInSyns().size(); c++) {
            const auto *sg = getSortedArchetypeMergedInSyns().at(c);

            // Loop through variables and update hash with variable initialisation parameters and derived parameters
            for(const auto &v : sg->getPSVarInitialisers()) {
                updateChildVarInitParamsHash(m_SortedMergedInSyns, c, v.first,
                                                &NeuronGroupMergedBase::isPSMVarInitParamReferenced,
                                                &SynapseGroupInternal::getPSVarInitialisers, hash);
                updateChildVarInitDerivedParamsHash(m_SortedMergedInSyns, c, v.first,
                                                    &NeuronGroupMergedBase::isPSMVarInitParamReferenced,
                                                    &SynapseGroupInternal::getPSVarInitialisers, hash);
            }
        }
    }
    else {
        // Loop through child current sources
        for(size_t i = 0; i < getSortedArchetypeCurrentSources().size(); i++) {
            updateChildParamHash(m_SortedCurrentSources, i, &NeuronGroupMergedBase::isCurrentSourceParamReferenced, 
                                 &CurrentSourceInternal::getParams, hash);
            updateChildDerivedParamHash(m_SortedCurrentSources, i, &NeuronGroupMergedBase::isCurrentSourceParamReferenced, 
                                        &CurrentSourceInternal::getDerivedParams, hash);
        }

        // Loop through child merged insyns
        for(size_t i = 0; i < getSortedArchetypeMergedInSyns().size(); i++) {
            updateChildParamHash(m_SortedMergedInSyns, i, &NeuronGroupMergedBase::isPSMParamReferenced, 
                                 &SynapseGroupInternal::getPSParams, hash);
            updateChildDerivedParamHash(m_SortedMergedInSyns, i, &NeuronGroupMergedBase::isPSMParamReferenced, 
                                        &SynapseGroupInternal::getPSDerivedParams, hash);
        }
    }
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const
{
    const auto *varInitSnippet = getArchetype().getVarInitialisers().at(varName).getSnippet();
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceParamReferenced(size_t childIndex, const std::string &paramName) const
{
    const auto *csm = getSortedArchetypeCurrentSources().at(childIndex)->getCurrentSourceModel();
    return isParamReferenced({csm->getInjectionCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceVarInitParamReferenced(size_t childIndex, const std::string &varName, const std::string &paramName) const
{
    const auto *varInitSnippet = getSortedArchetypeCurrentSources().at(childIndex)->getVarInitialisers().at(varName).getSnippet();
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMParamReferenced(size_t childIndex, const std::string &paramName) const
{
    const auto *psm = getSortedArchetypeMergedInSyns().at(childIndex)->getPSModel();
    return isParamReferenced({psm->getApplyInputCode(), psm->getDecayCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMVarInitParamReferenced(size_t childIndex, const std::string &varName, const std::string &paramName) const
{
    const auto *varInitSnippet = getSortedArchetypeMergedInSyns().at(childIndex)->getPSVarInitialisers().at(varName).getSnippet();
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}


//----------------------------------------------------------------------------
// CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDendriticDelayUpdateGroupMerged::name = "SynapseDendriticDelayUpdate";
//----------------------------------------------------------------------------
SynapseDendriticDelayUpdateGroupMerged::SynapseDendriticDelayUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   RuntimeGroupMerged<SynapseGroupInternal>(index, precision, backend, groups)
{
    addPointerField("unsigned int", "denDelayPtr", "", true);
}

// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseConnectivityHostInitGroupMerged::name = "SynapseConnectivityHostInit";
//------------------------------------------------------------------------
SynapseConnectivityHostInitGroupMerged::SynapseConnectivityHostInitGroupMerged(size_t index, const std::string &precision, const std::string&, const BackendBase &backend,
                                                                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   RuntimeGroupMerged<SynapseGroupInternal>(index, precision, backend, groups, true)
{
    // **TODO** these could be generic
    addField("unsigned int", "numSrcNeurons",
             [](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap&) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "numTrgNeurons",
             [](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap&) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "rowStride",
             [&backend](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap&) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });

    // Add heterogeneous connectivity initialiser model parameters
    addHeterogeneousParams<SynapseConnectivityHostInitGroupMerged>(
        getArchetype().getSparseConnectivityInitialiser().getSnippet()->getParamNames(), "",
        [](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getParams(); },
        &SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous);

    // Add heterogeneous connectivity initialiser derived parameters
    addHeterogeneousDerivedParams<SynapseConnectivityHostInitGroupMerged>(
        getArchetype().getSparseConnectivityInitialiser().getSnippet()->getDerivedParams(), "",
        [](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getDerivedParams(); },
        &SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous);

    // Add EGP pointers to struct for both host and device EGPs if they are seperate
    const auto egps = getArchetype().getSparseConnectivityInitialiser().getSnippet()->getExtraGlobalParams();
    for(const auto &e : egps) {
        addField(e.type + "*", e.name,
                 [e](const SynapseGroupInternal &g, size_t, const MergedRunnerMap &map) 
                 { 
                     return "&" + map.findGroup(g) + "." + e.name; 
                 },
                 FieldType::Host);

        if(!backend.getDeviceVarPrefix().empty()) {
            addField(e.type + "*", backend.getDeviceVarPrefix() + e.name,
                     [this, e](const SynapseGroupInternal &g, size_t, const MergedRunnerMap &map)
                     {
                         return "&" + map.findGroup(g) + "." + getDeviceVarPrefix() + e.name;
                     });
        }
        if(!backend.getHostVarPrefix().empty()) {
            addField(e.type + "*", backend.getHostVarPrefix() + e.name,
                     [this, e, &backend](const SynapseGroupInternal &g, size_t, const MergedRunnerMap &map)
                     {
                         return "&" + map.findGroup(g) + "." + backend.getHostVarPrefix() + e.name;
                     });
        }
    }
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous(const std::string &paramName) const
{
    return (isSparseConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg){ return sg.getSparseConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isSparseConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isSparseConnectivityInitParamReferenced(const std::string &paramName) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *connectInitSnippet = getArchetype().getSparseConnectivityInitialiser().getSnippet();
    return isParamReferenced({connectInitSnippet->getHostInitCode()}, paramName);
}

//----------------------------------------------------------------------------
// CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUParamHeterogeneous(const std::string &paramName) const
{
    return (isWUParamReferenced(paramName) && 
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isWUParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUGlobalVarHeterogeneous(const std::string &varName) const
{
    return (isWUGlobalVarReferenced(varName) &&
            isParamValueHeterogeneous(varName, [](const SynapseGroupInternal &sg) { return sg.getWUConstInitVals(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isWUVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName, [varName](const SynapseGroupInternal &sg){ return sg.getWUVarInitialisers().at(varName).getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isWUVarInitParamReferenced(varName, paramName) && 
            isParamValueHeterogeneous(paramName, [varName](const SynapseGroupInternal &sg) { return sg.getWUVarInitialisers().at(varName).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitParamHeterogeneous(const std::string &paramName) const
{
    return (isSparseConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isSparseConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitParamHeterogeneous(const std::string &paramName) const
{
    return (isToeplitzConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isToeplitzConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSrcNeuronParamHeterogeneous(const std::string &paramName) const
{
    return (isSrcNeuronParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSrcNeuronDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isSrcNeuronParamReferenced(paramName) &&  
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isTrgNeuronParamHeterogeneous(const std::string &paramName) const
{
    return (isTrgNeuronParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isTrgNeuronDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isTrgNeuronParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isKernelSizeHeterogeneous(size_t dimensionIndex) const
{
    // Get size of this kernel dimension for archetype
    const unsigned archetypeValue = getArchetype().getKernelSize().at(dimensionIndex);

    // Return true if any of the other groups have a different value
    return std::any_of(getGroups().cbegin(), getGroups().cend(),
                       [archetypeValue, dimensionIndex](const GroupInternal &g)
                       {
                           return (g.getKernelSize().at(dimensionIndex) != archetypeValue);
                       });
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getKernelSize(size_t dimensionIndex) const
{
    // If kernel size if heterogeneous in this dimension, return group structure entry
    if(isKernelSizeHeterogeneous(dimensionIndex)) {
        return "group->kernelSize" + std::to_string(dimensionIndex);
    }
    // Otherwise, return literal
    else {
        return std::to_string(getArchetype().getKernelSize().at(dimensionIndex));
    }
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::genKernelIndex(std::ostream &os, const CodeGenerator::Substitutions &subs) const
{
    // Loop through kernel dimensions to calculate array index
    const auto &kernelSize = getArchetype().getKernelSize();
    for(size_t i = 0; i < kernelSize.size(); i++) {
        os << "(" << subs["id_kernel_" + std::to_string(i)];
        // Loop through remainining dimensions of kernel and multiply
        for(size_t j = i + 1; j < kernelSize.size(); j++) {
            os << " * " << getKernelSize(j);
        }
        os << ")";

        // If this isn't the last dimension, add +
        if(i != (kernelSize.size() - 1)) {
            os << " + ";
        }
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPreSlot(unsigned int batchSize) const
{
    if(getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        return  (batchSize == 1) ? "preDelaySlot" : "preBatchDelaySlot";
    }
    else {
        return (batchSize == 1) ? "0" : "batch";
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostSlot(unsigned int batchSize) const
{
    if(getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
        return  (batchSize == 1) ? "postDelaySlot" : "postBatchDelaySlot";
    }
    else {
        return (batchSize == 1) ? "0" : "batch";
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostDenDelayIndex(unsigned int batchSize, const std::string &index, const std::string &offset) const
{
    assert(getArchetype().isDendriticDelayRequired());

    const std::string batchID = ((batchSize == 1) ? "" : "postBatchOffset + ") + index;

    if(offset.empty()) {
        return "(*group->denDelayPtr * group->numTrgNeurons) + " + batchID;
    }
    else {
        return "(((*group->denDelayPtr + " + offset + ") % " + std::to_string(getArchetype().getMaxDendriticDelayTimesteps()) + ") * group->numTrgNeurons) + " + batchID;
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPreVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
    if(delay) {
        return (singleBatch ? "preDelayOffset + " : "preBatchDelayOffset + ") + index;
    }
    else {
        return (singleBatch ? "" : "preBatchOffset + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
    if(delay) {
        return (singleBatch ? "postDelayOffset + " : "postBatchDelayOffset + ") + index;
    }
    else {
        return (singleBatch ? "" : "postBatchOffset + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPrePrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
   
    if(delay) {
        return (singleBatch ? "prePrevSpikeTimeDelayOffset + " : "prePrevSpikeTimeBatchDelayOffset + ") + index;
    }
    else {
        return (singleBatch ? "" : "preBatchOffset + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostPrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
   
    if(delay) {
        return (singleBatch ? "postPrevSpikeTimeDelayOffset + " : "postPrevSpikeTimeBatchDelayOffset + ") + index;
    }
    else {
        return (singleBatch ? "" : "postBatchOffset + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getSynVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
    return (singleBatch ? "" : "synBatchOffset + ") + index;
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getKernelVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
    return (singleBatch ? "" : "kernBatchOffset + ") + index;
}
//----------------------------------------------------------------------------
SynapseGroupMergedBase::SynapseGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                               Role role, const std::string &archetypeCode, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   RuntimeGroupMerged<SynapseGroupInternal>(index, precision, backend, groups), m_ArchetypeCode(archetypeCode)
{
    const bool updateRole = ((role == Role::PresynapticUpdate)
                             || (role == Role::PostsynapticUpdate)
                             || (role == Role::SynapseDynamics));
    const WeightUpdateModels::Base *wum = getArchetype().getWUModel();

    if(role != Role::KernelInit) {
        addField("unsigned int", "rowStride",
                 [&backend](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap&) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });
        addField("unsigned int", "numSrcNeurons",
                 [](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap&) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
        addField("unsigned int", "numTrgNeurons",
                [](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap&) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    }
    
    if(role == Role::PostsynapticUpdate || role == Role::SparseInit) {
        addField("unsigned int", "colStride",
                 [](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap&) { return std::to_string(sg.getMaxSourceConnections()); });
    }
    
    // If this role is one where postsynaptic input can be provided
    if(role == Role::PresynapticUpdate || role == Role::SynapseDynamics) {
        if(getArchetype().isDendriticDelayRequired()) {
            addPSPointerField(precision, "denDelay");
            addPSPointerField("unsigned int", "denDelayPtr", true);
        }
        else {
            addPSPointerField(precision, "inSyn");
        }
    }
    // for all types of roles
    if(getArchetype().isPresynapticOutputRequired()) {
        addPreOutputPointerField(precision, "revInSyn");
    }

    if(role == Role::PresynapticUpdate) {
        if(getArchetype().isTrueSpikeRequired()) {
            addPrePointerField("unsigned int", "spkCnt");
            addPrePointerField("unsigned int", "spk");
        }

        if(getArchetype().isSpikeEventRequired()) {
            addPrePointerField("unsigned int", "spkCntEvnt");
            addPrePointerField("unsigned int", "spkEvnt");
        }
    }
    else if(role == Role::PostsynapticUpdate) {
        addPostPointerField("unsigned int", "spkCnt");
        addPostPointerField("unsigned int", "spk");
    }

    // If this structure is used for updating rather than initializing
    if(updateRole) {
        // If presynaptic population has delay buffers
        if(getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
            addPrePointerField("unsigned int", "spkQuePtr", true);
        }

        // If postsynaptic population has delay buffers
        if(getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
            addPostPointerField("unsigned int", "spkQuePtr", true);
        }

        // Add heterogeneous presynaptic neuron model parameters
        addHeterogeneousParams<SynapseGroupMergedBase>(
            getArchetype().getSrcNeuronGroup()->getNeuronModel()->getParamNames(), "Pre",
            [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getParams(); },
            &SynapseGroupMergedBase::isSrcNeuronParamHeterogeneous);

        // Add heterogeneous presynaptic neuron model derived parameters
        addHeterogeneousDerivedParams<SynapseGroupMergedBase>(
            getArchetype().getSrcNeuronGroup()->getNeuronModel()->getDerivedParams(), "Pre",
            [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getDerivedParams(); },
            &SynapseGroupMergedBase::isSrcNeuronDerivedParamHeterogeneous);

        // Add heterogeneous postsynaptic neuron model parameters
        addHeterogeneousParams<SynapseGroupMergedBase>(
            getArchetype().getTrgNeuronGroup()->getNeuronModel()->getParamNames(), "Post",
            [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getParams(); },
            &SynapseGroupMergedBase::isTrgNeuronParamHeterogeneous);

        // Add heterogeneous postsynaptic neuron model derived parameters
        addHeterogeneousDerivedParams<SynapseGroupMergedBase>(
            getArchetype().getTrgNeuronGroup()->getNeuronModel()->getDerivedParams(), "Post",
            [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getDerivedParams(); },
            &SynapseGroupMergedBase::isTrgNeuronDerivedParamHeterogeneous);

        // Get correct code string
        const std::string code = getArchetypeCode();

        // Loop through variables in presynaptic neuron model
        const auto preVars = getArchetype().getSrcNeuronGroup()->getNeuronModel()->getVars();
        for(const auto &v : preVars) {
            // If variable is referenced in code string, add source pointer
            if(code.find("$(" + v.name + "_pre)") != std::string::npos) {
                addPrePointerField(v.type, v.name);
            }
        }

        // Loop through variables in postsynaptic neuron model
        const auto postVars = getArchetype().getTrgNeuronGroup()->getNeuronModel()->getVars();
        for(const auto &v : postVars) {
            // If variable is referenced in code string, add target pointer
            if(code.find("$(" + v.name + "_post)") != std::string::npos) {
                addPostPointerField(v.type, v.name);
            }
        }

        // Loop through extra global parameters in presynaptic neuron model
        const auto preEGPs = getArchetype().getSrcNeuronGroup()->getNeuronModel()->getExtraGlobalParams();
        for(const auto &e : preEGPs) {
            if(code.find("$(" + e.name + "_pre)") != std::string::npos) {
                const bool isPointer = Utils::isTypePointer(e.type);
                const std::string prefix = isPointer ? backend.getDeviceVarPrefix() : "";
                addField(e.type, e.name + "Pre",
                         [e, this](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap &map) 
                         { 
                             return map.findGroup(*sg.getSrcNeuronGroup()) + "." + getDeviceVarPrefix() + e.name;
                         },
                         isPointer ? FieldType::PointerEGP : FieldType::ScalarEGP);
            }
        }

        // Loop through extra global parameters in postsynaptic neuron model
        const auto postEGPs = getArchetype().getTrgNeuronGroup()->getNeuronModel()->getExtraGlobalParams();
        for(const auto &e : postEGPs) {
            if(code.find("$(" + e.name + "_post)") != std::string::npos) {
                const bool isPointer = Utils::isTypePointer(e.type);
                const std::string prefix = isPointer ? backend.getDeviceVarPrefix() : "";
                addField(e.type, e.name + "Post",
                         [e, this](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap &map)
                         { 
                             return map.findGroup(*sg.getTrgNeuronGroup()) + "." + getDeviceVarPrefix() + e.name;
                         },
                         isPointer ? FieldType::PointerEGP : FieldType::ScalarEGP);
            }
        }

        // Add spike times if required
        if(wum->isPreSpikeTimeRequired()) {
            addPrePointerField(timePrecision, "sT");
        }
        if(wum->isPostSpikeTimeRequired()) {
            addPostPointerField(timePrecision, "sT");
        }
        if(wum->isPreSpikeEventTimeRequired()) {
            addPrePointerField(timePrecision, "seT");
        }
        if(wum->isPrevPreSpikeTimeRequired()) {
            addPrePointerField(timePrecision, "prevST");
        }
        if(wum->isPrevPostSpikeTimeRequired()) {
            addPostPointerField(timePrecision, "prevST");
        }
        if(wum->isPrevPreSpikeEventTimeRequired()) {
            addPrePointerField(timePrecision, "prevSET");
        }
        // Add heterogeneous weight update model parameters
        addHeterogeneousParams<SynapseGroupMergedBase>(
            wum->getParamNames(), "",
            [](const SynapseGroupInternal &sg) { return sg.getWUParams(); },
            &SynapseGroupMergedBase::isWUParamHeterogeneous);

        // Add heterogeneous weight update model derived parameters
        addHeterogeneousDerivedParams<SynapseGroupMergedBase>(
            wum->getDerivedParams(), "",
            [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); },
            &SynapseGroupMergedBase::isWUDerivedParamHeterogeneous);

        // Add presynaptic variables to struct
        for(const auto &v : wum->getPreVars()) {
            addField(v.type + "*", v.name, 
                     [v, this](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap &map) 
                     { 
                         assert(!sg.isWUPreModelFused());
                         return map.findGroup(sg) + "." + getDeviceVarPrefix() + v.name;
                     });
        }
        
        // Add presynaptic variables to struct
        for(const auto &v : wum->getPostVars()) {
            addField(v.type + "*", v.name, 
                     [v, this](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap &map) 
                     { 
                         assert(!sg.isWUPostModelFused());
                         return map.findGroup(sg) + "." + getDeviceVarPrefix() + v.name;
                     });
        }

        // Add EGPs to struct
        addEGPs(wum->getExtraGlobalParams());
    }

    // Add pointers to connectivity data
    if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        addPointerField("unsigned int", "rowLength");
        addPointerField(getArchetype().getSparseIndType(), "ind");

        // Add additional structure for postsynaptic access
        if(backend.isPostsynapticRemapRequired() && !wum->getLearnPostCode().empty()
           && (role == Role::PostsynapticUpdate || role == Role::SparseInit))
        {
            addPointerField("unsigned int", "colLength");
            addPointerField("unsigned int", "remap");
        }

        // Add additional structure for synapse dynamics access if required
        if((role == Role::SynapseDynamics || role == Role::SparseInit) &&
           backend.isSynRemapRequired(getArchetype()))
        {
            addPointerField("unsigned int", "synRemap");
        }
    }
    else if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        addPointerField("uint32_t", "gp");
    }

    // If we're updating a group with procedural connectivity or initialising connectivity
    if((getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) || (role == Role::ConnectivityInit)) {
        // Add heterogeneous sparse connectivity initialiser model parameters
        addHeterogeneousParams<SynapseGroupMergedBase>(
            getArchetype().getSparseConnectivityInitialiser().getSnippet()->getParamNames(), "",
            [](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getParams(); },
            &SynapseGroupMergedBase::isSparseConnectivityInitParamHeterogeneous);


        // Add heterogeneous sparse connectivity initialiser derived parameters
        addHeterogeneousDerivedParams<SynapseGroupMergedBase>(
            getArchetype().getSparseConnectivityInitialiser().getSnippet()->getDerivedParams(), "",
            [](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getDerivedParams(); },
            &SynapseGroupMergedBase::isSparseConnectivityInitDerivedParamHeterogeneous);

        addEGPs(getArchetype().getSparseConnectivityInitialiser().getSnippet()->getExtraGlobalParams(),
                backend.getDeviceVarPrefix());
    }

    // If we're updating a group with Toeplitz connectivity
    if(updateRole && (getArchetype().getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ)) {
        // Add heterogeneous toeplitz connectivity initialiser model parameters
        addHeterogeneousParams<SynapseGroupMergedBase>(
            getArchetype().getToeplitzConnectivityInitialiser().getSnippet()->getParamNames(), "",
            [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getParams(); },
            &SynapseGroupMergedBase::isToeplitzConnectivityInitParamHeterogeneous);


        // Add heterogeneous toeplitz initialiser derived parameters
        addHeterogeneousDerivedParams<SynapseGroupMergedBase>(
            getArchetype().getToeplitzConnectivityInitialiser().getSnippet()->getDerivedParams(), "",
            [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getDerivedParams(); },
            &SynapseGroupMergedBase::isToeplitzConnectivityInitDerivedParamHeterogeneous);

        addEGPs(getArchetype().getToeplitzConnectivityInitialiser().getSnippet()->getExtraGlobalParams(),
                backend.getDeviceVarPrefix());
    }

    // If WU variables are global
    const auto &varInit = getArchetype().getWUVarInitialisers();
    if(getArchetype().getMatrixType() & SynapseMatrixWeight::GLOBAL) {
        // If this is an update role
        // **NOTE **global variable values aren't useful during initialization
        if(updateRole) {
            for(const auto &var : wum->getVars()) {
                // If variable should be implemented heterogeneously, add scalar field
                if(isWUGlobalVarHeterogeneous(var.name)) {
                    addScalarField(var.name,
                                   [var](const SynapseGroupInternal &sg, size_t)
                                   {
                                       return Utils::writePreciseString(sg.getWUConstInitVals().at(var.name));
                                   });
                }
            }
        }
    }
    // Otherwise (weights are individual or procedural)
    else {
        const bool connectInitRole = (role == Role::ConnectivityInit);
        const bool varInitRole = (role == Role::DenseInit || role == Role::SparseInit || role == Role::KernelInit);
        const bool proceduralWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL);
        const bool kernelWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL);
        const bool individualWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);

        // If synapse group has a kernel and has kernel weights or initialising individual weights
        if(!getArchetype().getKernelSize().empty() && ((proceduralWeights && updateRole) || kernelWeights || (connectInitRole && individualWeights))) {
            // Loop through kernel size dimensions
            for(size_t d = 0; d < getArchetype().getKernelSize().size(); d++) {
                // If this dimension has a heterogeneous size, add it to struct
                if(isKernelSizeHeterogeneous(d)) {
                    addField("unsigned int", "kernelSize" + std::to_string(d),
                             [d](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap&) { return std::to_string(sg.getKernelSize().at(d)); });
                }
            }
        }

        // If weights are procedural, we're initializing individual variables or we're initialising variables in a kernel
        // **NOTE** some of these won't actually be required - could do this per-variable in loop over vars
        if((proceduralWeights && updateRole) || (connectInitRole && !getArchetype().getKernelSize().empty()) 
           || (varInitRole && (individualWeights || kernelWeights))) 
        {
            // Add heterogeneous variable initialization parameters and derived parameters
            addHeterogeneousVarInitParams<SynapseGroupMergedBase>(
                &SynapseGroupInternal::getWUVarInitialisers,
                &SynapseGroupMergedBase::isWUVarInitParamHeterogeneous);

            addHeterogeneousVarInitDerivedParams<SynapseGroupMergedBase>(
                &SynapseGroupInternal::getWUVarInitialisers,
                &SynapseGroupMergedBase::isWUVarInitDerivedParamHeterogeneous);
        }

        // Loop through variables
        for(const auto &var : wum->getVars()) {
            // Variable initialisation is required if we're performing connectivity init and var init snippet requires a kernel or
            // We're performing some other sort of initialisation, the snippet DOESN'T require a kernel but has SOME code
            const auto *snippet = varInit.at(var.name).getSnippet();
            const bool varInitRequired = ((connectInitRole && snippet->requiresKernel()) 
                                          || (varInitRole && individualWeights && !snippet->requiresKernel() && !snippet->getCode().empty())
                                          || (varInitRole && kernelWeights && !snippet->getCode().empty()));

            // If we're performing an update with individual weights; or this variable should be initialised
            if((updateRole && individualWeights) || (kernelWeights && updateRole) || varInitRequired) {
                addPointerField(var.type, var.name);
            }

            // If we're performing a procedural update or this variable should be initialised, add any var init EGPs to structure
            if((proceduralWeights && updateRole) || varInitRequired) {
                addEGPs(snippet->getExtraGlobalParams(), var.name);
            }
        }
    }
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroupMergedBase::getHashDigest(Role role) const
{
    const bool updateRole = ((role == Role::PresynapticUpdate)
                             || (role == Role::PostsynapticUpdate)
                             || (role == Role::SynapseDynamics));

    // Update hash with archetype's hash
    boost::uuids::detail::sha1 hash;
    if(updateRole) {
        Utils::updateHash(getArchetype().getWUHashDigest(), hash);
    }
    else {
        Utils::updateHash(getArchetype().getWUInitHashDigest(), hash);
    }

    // Update hash with number of neurons in pre and postsynaptic population
    updateHash([](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxSourceConnections(); }, hash);
    // **NOTE** ideally we'd include the row stride but this needs a backend and it SHOULDN'T be necessary
    // as I can't think of any way of changing this without changing the hash in other places

    if(updateRole) {
        // Update hash with weight update model parameters and derived parameters
        updateHash([](const SynapseGroupInternal &g) { return g.getWUParams(); }, hash);
        updateHash([](const SynapseGroupInternal &g) { return g.getWUDerivedParams(); }, hash);

        // Update hash with presynaptic neuron population parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSrcNeuronParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getParams(); }, hash);
        
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSrcNeuronParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getDerivedParams(); }, hash);

        // Update hash with postsynaptic neuron population parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isTrgNeuronParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getParams(); }, hash);
        
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isTrgNeuronParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getDerivedParams(); }, hash);
    }


    // If we're updating a hash for a group with procedural connectivity or initialising connectivity
    if((getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) || (role == Role::ConnectivityInit)) {
        // Update hash with connectivity parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSparseConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getParams(); }, hash);

        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSparseConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getSparseConnectivityInitialiser().getDerivedParams(); }, hash);
    }

    // If we're updating a hash for a group with Toeplitz connectivity
    if((getArchetype().getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) && updateRole) {
        // Update hash with connectivity parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isToeplitzConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getParams(); }, hash);

        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isToeplitzConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getDerivedParams(); }, hash);
    }

    if(getArchetype().getMatrixType() & SynapseMatrixWeight::GLOBAL) {
        // If this is an update role
        // **NOTE **global variable values aren't useful during initialization
        if(updateRole) {
            updateParamHash<SynapseGroupMergedBase>(
                &SynapseGroupMergedBase::isWUGlobalVarReferenced,
                [](const SynapseGroupInternal &sg) { return sg.getWUConstInitVals();  }, hash);
        }
    }
    // Otherwise (weights are individual or procedural)
    else {
        const bool connectInitRole = (role == Role::ConnectivityInit);
        const bool varInitRole = (role == Role::DenseInit || role == Role::SparseInit);
        const bool proceduralWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL);
        const bool individualWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);
        const bool kernelWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);

        // If synapse group has a kernel and we're either updating with procedural  
        // weights or initialising individual weights, update hash with kernel size
        if(!getArchetype().getKernelSize().empty() && 
            ((proceduralWeights && updateRole) || (connectInitRole && individualWeights) || (kernelWeights && !updateRole))) 
        {
            updateHash([](const SynapseGroupInternal &g) { return g.getKernelSize(); }, hash);
        }

        // If weights are procedural, we're initializing individual variables or we're initialising variables in a kernel
        // **NOTE** some of these won't actually be required - could do this per-variable in loop over vars
        if((proceduralWeights && updateRole) || (connectInitRole && !getArchetype().getKernelSize().empty())
           || (varInitRole && individualWeights) || (varInitRole && kernelWeights))
        {
            // Update hash with each group's variable initialisation parameters and derived parameters
            updateVarInitParamHash<SynapseGroupMergedBase>(&SynapseGroupInternal::getWUVarInitialisers, 
                                                           &SynapseGroupMergedBase::isWUVarInitParamReferenced, hash);
            updateVarInitDerivedParamHash<SynapseGroupMergedBase>(&SynapseGroupInternal::getWUVarInitialisers,
                                                                  &SynapseGroupMergedBase::isWUVarInitParamReferenced, hash);
        }
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addPSPointerField(const std::string &type, const std::string &name, bool scalar)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name, 
             [name, scalar, this](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap &map) 
             { 
                 assert(!sg.isPSModelFused());
                 if(scalar && isDeviceScalarRequired()) {
                     return "&" + map.findGroup(sg) + "." + getDeviceVarPrefix() + name;
                 }
                 else {
                     return map.findGroup(sg) + "." + getDeviceVarPrefix() + name;
                 }
             });
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addPreOutputPointerField(const std::string &type, const std::string &name, bool scalar)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name, 
             [name, scalar, this](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap &map) 
             { 
                 assert(!sg.isPreOutputModelFused());
                 if(scalar && isDeviceScalarRequired()) {
                     return "&" + map.findGroup(sg) + "." + getDeviceVarPrefix() + name;
                 }
                 else {
                     return map.findGroup(sg) + "." + getDeviceVarPrefix() + name;
                 }
             });
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addPrePointerField(const std::string &type, const std::string &name, bool scalar)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name + "Pre", 
             [name, scalar, this](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap &map) 
             { 
                 if(scalar && isDeviceScalarRequired()) {
                     return "&" + map.findGroup(*sg.getSrcNeuronGroup()) + "." + getDeviceVarPrefix() + name;
                 }
                 else {
                     return map.findGroup(*sg.getSrcNeuronGroup()) + "." + getDeviceVarPrefix() + name;
                 }
             });
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addPostPointerField(const std::string &type, const std::string &name, bool scalar)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name + "Post", 
             [name, scalar, this](const SynapseGroupInternal &sg, size_t, const MergedRunnerMap &map) 
             { 
                 if(scalar && isDeviceScalarRequired()) {
                     return "&" + map.findGroup(*sg.getTrgNeuronGroup()) + "." + getDeviceVarPrefix() + name;
                 }
                 else {
                     return map.findGroup(*sg.getTrgNeuronGroup()) + "." + getDeviceVarPrefix() + name;
                 }
             });
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUParamReferenced(const std::string &paramName) const
{
    return isParamReferenced({getArchetypeCode()}, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUGlobalVarReferenced(const std::string &varName) const
{
    // If synapse group has global WU variables
    if(getArchetype().getMatrixType() & SynapseMatrixWeight::GLOBAL) {
        return isParamReferenced({getArchetypeCode()}, varName);
    }
    // Otherwise, return false
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUVarInitParamReferenced(const std::string &varName, const std::string &paramName) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *varInitSnippet = getArchetype().getWUVarInitialisers().at(varName).getSnippet();
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitParamReferenced(const std::string &paramName) const
{
    const auto *snippet = getArchetype().getSparseConnectivityInitialiser().getSnippet();
    const auto rowBuildStateVars = snippet->getRowBuildStateVars();
    const auto colBuildStateVars = snippet->getColBuildStateVars();

    // Build list of code strings containing row build code and any row build state variable values
    std::vector<std::string> codeStrings{snippet->getRowBuildCode(), snippet->getColBuildCode()};
    std::transform(rowBuildStateVars.cbegin(), rowBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });
    std::transform(colBuildStateVars.cbegin(), colBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });

    return isParamReferenced(codeStrings, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitParamReferenced(const std::string &paramName) const
{
    const auto *snippet = getArchetype().getToeplitzConnectivityInitialiser().getSnippet();
    const auto diagonalBuildStateVars = snippet->getDiagonalBuildStateVars();

    // Build list of code strings containing diagonal build code and any diagonal build state variable values
    std::vector<std::string> codeStrings{snippet->getDiagonalBuildCode()};
    std::transform(diagonalBuildStateVars.cbegin(), diagonalBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });
   
    return isParamReferenced(codeStrings, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSrcNeuronParamReferenced(const std::string &paramName) const
{
    return isParamReferenced({getArchetypeCode()}, paramName + "_pre");
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isTrgNeuronParamReferenced(const std::string &paramName) const
{
    return isParamReferenced({getArchetypeCode()}, paramName +  "_post");
}

// ----------------------------------------------------------------------------
// CustomUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateHostReductionGroupMerged::name = "CustomUpdateHostReduction";
//----------------------------------------------------------------------------
CustomUpdateHostReductionGroupMerged::CustomUpdateHostReductionGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                           const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups)
:   CustomUpdateHostReductionGroupMergedBase<CustomUpdateInternal>(index, precision, backend, groups, true)
{
    addField("unsigned int", "size",
             [](const CustomUpdateInternal &c, size_t, const MergedRunnerMap&) { return std::to_string(c.getSize()); });

    // If some variables are delayed, add delay pointer
    // **NOTE** this is HOST delay pointer
    if(getArchetype().getDelayNeuronGroup() != nullptr) {
        addField("unsigned int*", "spkQuePtr", 
                 [&](const CustomUpdateInternal &cg, size_t, const MergedRunnerMap &map) 
                 { 
                     if(isDeviceScalarRequired()) {
                         return "&" + map.findGroup(*cg.getDelayNeuronGroup()) + "." + getDeviceVarPrefix() + "spkQuePtr";
                     }
                     else {
                         return map.findGroup(*cg.getDelayNeuronGroup()) + "." + getDeviceVarPrefix() + "spkQuePtr";
                     }
                 });
    }
}

// ----------------------------------------------------------------------------
// CustomWUUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateHostReductionGroupMerged::name = "CustomWUUpdateHostReduction";
//----------------------------------------------------------------------------
CustomWUUpdateHostReductionGroupMerged::CustomWUUpdateHostReductionGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                               const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   CustomUpdateHostReductionGroupMergedBase<CustomUpdateWUInternal>(index, precision, backend, groups, true)
{
    addField("unsigned int", "size",
             [&backend](const CustomUpdateWUInternal &cg, size_t, const MergedRunnerMap&) 
             {
                 return std::to_string(cg.getSynapseGroup()->getMaxConnections() * (size_t)cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); 
             });
}

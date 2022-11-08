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
:   GroupMerged<NeuronGroupInternal>(index, precision, groups)
{
    if(getArchetype().isDelayRequired()) {
        addPointerField("unsigned int", "spkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
    } 

    addPointerField("unsigned int", "spkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField("unsigned int", "spkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
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
:   GroupMerged<NeuronGroupInternal>(index, precision, groups)
{
    if(getArchetype().isDelayRequired()) {
        addPointerField("unsigned int", "spkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
    } 

    addPointerField("unsigned int", "spkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField("unsigned int", "spkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
    }

    if(getArchetype().isPrevSpikeTimeRequired()) {
        addPointerField("unsigned int", "spk", backend.getDeviceVarPrefix() + "glbSpk");
        addPointerField(timePrecision, "prevST", backend.getDeviceVarPrefix() + "prevST");
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        addPointerField("unsigned int", "spkEvnt", backend.getDeviceVarPrefix() + "glbSpkEvnt");
        addPointerField(timePrecision, "prevSET", backend.getDeviceVarPrefix() + "prevSET");
    }

    if(getArchetype().isDelayRequired()) {
        addField("unsigned int", "numNeurons",
                 [](const NeuronGroupInternal &ng, size_t) { return std::to_string(ng.getNumNeurons()); });
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::NeuronGroupMergedBase
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const NeuronGroupInternal &ng) { return ng.getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isDerivedParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const NeuronGroupInternal &ng) { return ng.getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isVarInitParamHeterogeneous(size_t varIndex, size_t paramIndex) const
{
    return (isVarInitParamReferenced(varIndex, paramIndex) &&
            isParamValueHeterogeneous(paramIndex,
                                      [varIndex](const NeuronGroupInternal &sg) { return sg.getVarInitialisers().at(varIndex).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isVarInitDerivedParamHeterogeneous(size_t varIndex, size_t paramIndex) const
{
    return (isVarInitDerivedParamReferenced(varIndex, paramIndex) &&
            isParamValueHeterogeneous(paramIndex,
                                      [varIndex](const NeuronGroupInternal &sg){ return sg.getVarInitialisers().at(varIndex).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    return (isCurrentSourceParamReferenced(childIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedCurrentSources,
                                           [](const CurrentSourceInternal *cs) { return cs->getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    return (isCurrentSourceDerivedParamReferenced(childIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedCurrentSources,
                                           [](const CurrentSourceInternal *cs) { return cs->getDerivedParams(); }));
 
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isCurrentSourceVarInitParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedCurrentSources,
                                           [varIndex](const CurrentSourceInternal *cs) { return cs->getVarInitialisers().at(varIndex).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isCurrentSourceVarInitDerivedParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedCurrentSources,
                                           [varIndex](const CurrentSourceInternal *cs) { return cs->getVarInitialisers().at(varIndex).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    return (isPSMParamReferenced(childIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedMergedInSyns,
                                           [](const SynapseGroupInternal *inSyn) { return inSyn->getPSParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    return (isPSMDerivedParamReferenced(childIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedMergedInSyns, 
                                           [](const SynapseGroupInternal *inSyn) { return inSyn->getPSDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMGlobalVarHeterogeneous(size_t childIndex, size_t varIndex) const
{
    return (isPSMGlobalVarReferenced(childIndex, varIndex) &&
            isChildParamValueHeterogeneous(childIndex, varIndex, m_SortedMergedInSyns, 
                                           [](const SynapseGroupInternal *inSyn) { return inSyn->getPSConstInitVals(); }));

}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isPSMVarInitParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedMergedInSyns,
                                           [varIndex](const SynapseGroupInternal *inSyn){ return inSyn->getPSVarInitialisers().at(varIndex).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{    
    return (isPSMVarInitDerivedParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedMergedInSyns,
                                          [varIndex](const SynapseGroupInternal *inSyn){ return inSyn->getPSVarInitialisers().at(varIndex).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
NeuronGroupMergedBase::NeuronGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                             bool init, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   GroupMerged<NeuronGroupInternal>(index, precision, groups)
{
    // Build vector of vectors containing each child group's merged in syns, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedMergedInSyns, &NeuronGroupInternal::getFusedPSMInSyn,
                             init ? &SynapseGroupInternal::getPSInitHashDigest : &SynapseGroupInternal::getPSHashDigest);

    // Build vector of vectors containing each child group's merged out syns with pre output, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedMergedPreOutputOutSyns, &NeuronGroupInternal::getFusedPreOutputOutSyn,
                             init ? &SynapseGroupInternal::getPreOutputInitHashDigest : &SynapseGroupInternal::getPreOutputHashDigest);

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedCurrentSources, &NeuronGroupInternal::getCurrentSources,
                             init ? &CurrentSourceInternal::getInitHashDigest : &CurrentSourceInternal::getHashDigest);

    addField("unsigned int", "numNeurons",
              [](const NeuronGroupInternal &ng, size_t) { return std::to_string(ng.getNumNeurons()); });

    addPointerField("unsigned int", "spkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");
    addPointerField("unsigned int", "spk", backend.getDeviceVarPrefix() + "glbSpk");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField("unsigned int", "spkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
        addPointerField("unsigned int", "spkEvnt", backend.getDeviceVarPrefix() + "glbSpkEvnt");
    }

    if(getArchetype().isDelayRequired()) {
        addPointerField("unsigned int", "spkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
    }

    if(getArchetype().isSpikeTimeRequired()) {
        addPointerField(timePrecision, "sT", backend.getDeviceVarPrefix() + "sT");
    }
    if(getArchetype().isSpikeEventTimeRequired()) {
        addPointerField(timePrecision, "seT", backend.getDeviceVarPrefix() + "seT");
    }

    if(getArchetype().isPrevSpikeTimeRequired()) {
        addPointerField(timePrecision, "prevST", backend.getDeviceVarPrefix() + "prevST");
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
    assert(vars.size() == varInit.size());
    for(size_t v = 0; v < vars.size(); v++) {
        // If we're not initialising or if there is initialization code for this variable
        const auto var = vars[v];
        if(!init || !varInit[v].getSnippet()->getCode().empty()) {
            addPointerField(var.type, var.name, backend.getDeviceVarPrefix() + var.name);
        }

        // If we're initializing, add any var init EGPs to structure
        if(init) {
            addEGPs(varInit[v].getSnippet()->getExtraGlobalParams(), backend.getDeviceVarPrefix(), var.name);
        }
    }

    // If we're generating a struct for initialization
    if(init) {
        // Add heterogeneous var init parameters
        addHeterogeneousVarInitParams<NeuronGroupMergedBase, NeuronVarAdapter>(
            &NeuronGroupMergedBase::isVarInitParamHeterogeneous);

        addHeterogeneousVarInitDerivedParams<NeuronGroupMergedBase, NeuronVarAdapter>(
            &NeuronGroupMergedBase::isVarInitDerivedParamHeterogeneous);
    }
    // Otherwise
    else {
        addEGPs(nm->getExtraGlobalParams(), backend.getDeviceVarPrefix());

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
        addMergedInSynPointerField(precision, "inSynInSyn", i, backend.getDeviceVarPrefix() + "inSyn");

        // Add pointer to dendritic delay buffer if required
        if(sg->isDendriticDelayRequired()) {
            addMergedInSynPointerField(precision, "denDelayInSyn", i, backend.getDeviceVarPrefix() + "denDelay");
            addMergedInSynPointerField("unsigned int", "denDelayPtrInSyn", i, backend.getScalarAddressPrefix() + "denDelayPtr");
        }

        // Loop through variables
        const auto vars = sg->getPSModel()->getVars();
        const auto &varInit = sg->getPSVarInitialisers();
        for(size_t v = 0; v < vars.size(); v++) {
            // If PSM has individual variables
            const auto var = vars[v];
            if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                // Add pointers to state variable
                if(!init || !varInit[v].getSnippet()->getCode().empty()) {
                    addMergedInSynPointerField(var.type, var.name + "InSyn", i, backend.getDeviceVarPrefix() + var.name);
                }

                // If we're generating an initialization structure, also add any heterogeneous parameters, derived parameters or extra global parameters required for initializers
                if(init) {
                    const auto *varInitSnippet = varInit.at(v).getSnippet();
                    addHeterogeneousChildVarInitParams(varInitSnippet->getParamNames(), m_SortedMergedInSyns, i, v, var.name + "InSyn",
                                                       &NeuronGroupMergedBase::isPSMVarInitParamHeterogeneous, &SynapseGroupInternal::getPSVarInitialisers);
                    addHeterogeneousChildVarInitDerivedParams(varInitSnippet->getDerivedParams(), m_SortedMergedInSyns, i, v, var.name + "InSyn",
                                                              &NeuronGroupMergedBase::isPSMVarInitDerivedParamHeterogeneous, &SynapseGroupInternal::getPSVarInitialisers);
                    addChildEGPs(varInitSnippet->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), var.name + "InSyn",
                                 [var, this](size_t groupIndex, size_t childIndex)
                                 {
                                     return var.name + m_SortedMergedInSyns.at(groupIndex).at(childIndex)->getFusedPSVarSuffix();
                                 });
                }
            }
            // Otherwise, if postsynaptic model variables are global and we're updating 
            // **NOTE** global variable values aren't useful during initialization
            else if(!init) {
                // If GLOBALG variable should be implemented heterogeneously, add value
                if(isPSMGlobalVarHeterogeneous(i, v)) {
                    addScalarField(var.name + "InSyn" + std::to_string(i),
                                   [this, i, v](const NeuronGroupInternal &, size_t groupIndex)
                                   {
                                       const double val = m_SortedMergedInSyns.at(groupIndex).at(i)->getPSConstInitVals().at(v);
                                       return Utils::writePreciseString(val);
                                   });
                }
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
            addChildEGPs(sg->getPSModel()->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), "InSyn",
                         [this](size_t groupIndex, size_t childIndex)
                         {
                             return m_SortedMergedInSyns.at(groupIndex).at(childIndex)->getFusedPSVarSuffix();
                         });
        }
    }

    // Loop through merged output synapses with presynaptic output of archetypical neuron group (0) in sorted order
    for(size_t i = 0; i < getSortedArchetypeMergedPreOutputOutSyns().size(); i++) {
        // Add pointer to revInSyn
        addMergedPreOutputOutSynPointerField(precision, "revInSynOutSyn", i, backend.getDeviceVarPrefix() + "revInSyn");
    }
    
    // Loop through current sources to archetypical neuron group in sorted order
    for(size_t i = 0; i < getSortedArchetypeCurrentSources().size(); i++) {
        const auto *cs = getSortedArchetypeCurrentSources().at(i);

        // Loop through variables
        const auto vars = cs->getCurrentSourceModel()->getVars();
        const auto &varInit = cs->getVarInitialisers();
        for(size_t v = 0; v < vars.size(); v++) {
            // Add pointers to state variable
            const auto var = vars[v];
            if(!init || !varInit[v].getSnippet()->getCode().empty()) {
                assert(!Utils::isTypePointer(var.type));
                addField(var.type + "*", var.name + "CS" + std::to_string(i),
                         [&backend, i, var, this](const NeuronGroupInternal &, size_t groupIndex)
                         {
                             return backend.getDeviceVarPrefix() + var.name + m_SortedCurrentSources.at(groupIndex).at(i)->getName();
                         });
            }

            // If we're generating an initialization structure, also add any heterogeneous parameters, derived parameters or extra global parameters required for initializers
            if(init) {
                const auto *varInitSnippet = varInit.at(v).getSnippet();
                addHeterogeneousChildVarInitParams(varInitSnippet->getParamNames(), m_SortedCurrentSources, i, v, var.name + "CS",
                                                   &NeuronGroupMergedBase::isCurrentSourceVarInitParamHeterogeneous, &CurrentSourceInternal::getVarInitialisers);
                addHeterogeneousChildVarInitDerivedParams(varInitSnippet->getDerivedParams(), m_SortedCurrentSources, i, v, var.name + "CS",
                                                          &NeuronGroupMergedBase::isCurrentSourceVarInitDerivedParamHeterogeneous,  &CurrentSourceInternal::getVarInitialisers);
                addChildEGPs(varInitSnippet->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), var.name + "CS",
                             [var, this](size_t groupIndex, size_t childIndex)
                             {
                                 return var.name + m_SortedCurrentSources.at(groupIndex).at(childIndex)->getName();
                             });
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
            addChildEGPs(cs->getCurrentSourceModel()->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), "CS",
                         [this](size_t groupIndex, size_t childIndex)
                         {
                             return m_SortedCurrentSources.at(groupIndex).at(childIndex)->getName();
                         });

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
            const auto &varInit = cs->getVarInitialisers();
            for(size_t v = 0; v < varInit.size(); v++) {
                updateChildVarInitParamsHash<CurrentSourceVarAdapter>(
                    m_SortedCurrentSources, c, v, &NeuronGroupMergedBase::isCurrentSourceVarInitParamReferenced, hash);
                updateChildVarInitDerivedParamsHash<CurrentSourceVarAdapter>(
                    m_SortedCurrentSources, c, v, &NeuronGroupMergedBase::isCurrentSourceVarInitDerivedParamReferenced, hash);
            }
        }

        // Loop through child merged insyns
        for(size_t c = 0; c < getSortedArchetypeMergedInSyns().size(); c++) {
            const auto *sg = getSortedArchetypeMergedInSyns().at(c);

            // Loop through variables and update hash with variable initialisation parameters and derived parameters
            const auto &varInit = sg->getPSVarInitialisers();
            for(size_t v = 0; v < varInit.size(); v++) {
                if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    updateChildVarInitParamsHash<SynapsePSMVarAdapter>(
                        m_SortedMergedInSyns, c, v, &NeuronGroupMergedBase::isPSMVarInitParamReferenced, hash);
                    updateChildVarInitDerivedParamsHash<SynapsePSMVarAdapter>(
                        m_SortedMergedInSyns, c, v, &NeuronGroupMergedBase::isPSMVarInitDerivedParamReferenced, hash);
                }
            }
        }
    }
    else {
        // Loop through child current sources
        for(size_t i = 0; i < getSortedArchetypeCurrentSources().size(); i++) {
            updateChildParamHash(m_SortedCurrentSources, i, &NeuronGroupMergedBase::isCurrentSourceParamReferenced, 
                                 &CurrentSourceInternal::getParams, hash);
            updateChildDerivedParamHash(m_SortedCurrentSources, i, &NeuronGroupMergedBase::isCurrentSourceDerivedParamReferenced, 
                                        &CurrentSourceInternal::getDerivedParams, hash);
        }

        // Loop through child merged insyns
        for(size_t i = 0; i < getSortedArchetypeMergedInSyns().size(); i++) {
            const auto *sg = getSortedArchetypeMergedInSyns().at(i);

            updateChildParamHash(m_SortedMergedInSyns, i, &NeuronGroupMergedBase::isPSMParamReferenced, 
                                 &SynapseGroupInternal::getPSParams, hash);
            updateChildDerivedParamHash(m_SortedMergedInSyns, i, &NeuronGroupMergedBase::isPSMDerivedParamReferenced, 
                                        &SynapseGroupInternal::getPSDerivedParams, hash);

            if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM)) {
                updateChildParamHash(m_SortedMergedInSyns, i, &NeuronGroupMergedBase::isPSMGlobalVarReferenced,
                                     &SynapseGroupInternal::getPSConstInitVals, hash);
            }
        }
    }
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isVarInitParamReferenced(size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isVarInitDerivedParamReferenced(size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({varInitSnippet->getCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceParamReferenced(size_t childIndex, size_t paramIndex) const
{
    const auto *csm = getSortedArchetypeCurrentSources().at(childIndex)->getCurrentSourceModel();
    const std::string paramName = csm->getParamNames().at(paramIndex);
    return isParamReferenced({csm->getInjectionCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceDerivedParamReferenced(size_t childIndex, size_t paramIndex) const
{
    const auto *csm = getSortedArchetypeCurrentSources().at(childIndex)->getCurrentSourceModel();
    const std::string derivedParamName = csm->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({csm->getInjectionCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceVarInitParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeCurrentSources().at(childIndex)->getVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isCurrentSourceVarInitDerivedParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeCurrentSources().at(childIndex)->getVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({varInitSnippet->getCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMParamReferenced(size_t childIndex, size_t paramIndex) const
{
    const auto *psm = getSortedArchetypeMergedInSyns().at(childIndex)->getPSModel();
    const std::string paramName = psm->getParamNames().at(paramIndex);
    return isParamReferenced({psm->getApplyInputCode(), psm->getDecayCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMDerivedParamReferenced(size_t childIndex, size_t paramIndex) const
{
    const auto *psm = getSortedArchetypeMergedInSyns().at(childIndex)->getPSModel();
    const std::string derivedParamName = psm->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({psm->getApplyInputCode(), psm->getDecayCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMGlobalVarReferenced(size_t childIndex, size_t varIndex) const
{
    // If synapse group doesn't have individual PSM variables to start with, return false
    const auto *sg = getSortedArchetypeMergedInSyns().at(childIndex);
    if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
        return false;
    }
    else {
        const auto *psm = sg->getPSModel();
        const std::string varName = psm->getVars().at(varIndex).name;
        return isParamReferenced({psm->getApplyInputCode(), psm->getDecayCode()}, varName);
    }
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMVarInitParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeMergedInSyns().at(childIndex)->getPSVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isPSMVarInitDerivedParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeMergedInSyns().at(childIndex)->getPSVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({varInitSnippet->getCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
void NeuronGroupMergedBase::addMergedInSynPointerField(const std::string &type, const std::string &name, 
                                                       size_t archetypeIndex, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name + std::to_string(archetypeIndex),
             [prefix, archetypeIndex, this](const NeuronGroupInternal &, size_t groupIndex)
             {
                 return prefix + m_SortedMergedInSyns.at(groupIndex).at(archetypeIndex)->getFusedPSVarSuffix();
             });
}
//----------------------------------------------------------------------------
void NeuronGroupMergedBase::addMergedPreOutputOutSynPointerField(const std::string &type, const std::string &name, 
                                                       size_t archetypeIndex, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name + std::to_string(archetypeIndex),
             [prefix, archetypeIndex, this](const NeuronGroupInternal &, size_t groupIndex)
             {
                 return prefix + m_SortedMergedPreOutputOutSyns.at(groupIndex).at(archetypeIndex)->getFusedPreOutputSuffix();
             });
}

//----------------------------------------------------------------------------
// CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUParamHeterogeneous(size_t paramIndex) const
{
    return (isWUParamReferenced(paramIndex) && 
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getWUParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUDerivedParamHeterogeneous(size_t paramIndex) const
{
    return (isWUDerivedParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUGlobalVarHeterogeneous(size_t varIndex) const
{
    return (isWUGlobalVarReferenced(varIndex) &&
            isParamValueHeterogeneous(varIndex, [](const SynapseGroupInternal &sg) { return sg.getWUConstInitVals(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUVarInitParamHeterogeneous(size_t varIndex, size_t paramIndex) const
{
    return (isWUVarInitParamReferenced(varIndex, paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [varIndex](const SynapseGroupInternal &sg){ return sg.getWUVarInitialisers().at(varIndex).getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUVarInitDerivedParamHeterogeneous(size_t varIndex, size_t paramIndex) const
{
    return (isWUVarInitDerivedParamReferenced(varIndex, paramIndex) && 
            isParamValueHeterogeneous(paramIndex, [varIndex](const SynapseGroupInternal &sg) { return sg.getWUVarInitialisers().at(varIndex).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitParamHeterogeneous(size_t paramIndex) const
{
    return (isSparseConnectivityInitParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const
{
    return (isSparseConnectivityInitDerivedParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitParamHeterogeneous(size_t paramIndex) const
{
    return (isToeplitzConnectivityInitParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const
{
    return (isToeplitzConnectivityInitDerivedParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSrcNeuronParamHeterogeneous(size_t paramIndex) const
{
    return (isSrcNeuronParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSrcNeuronDerivedParamHeterogeneous(size_t paramIndex) const
{
    return (isSrcNeuronDerivedParamReferenced(paramIndex) &&  
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isTrgNeuronParamHeterogeneous(size_t paramIndex) const
{
    return (isTrgNeuronParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isTrgNeuronDerivedParamHeterogeneous(size_t paramIndex) const
{
    return (isTrgNeuronDerivedParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getDerivedParams(); }));
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
std::string SynapseGroupMergedBase::getPreVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    return getVarIndex(delay, batchSize, varDuplication, index, "pre");
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
   return getVarIndex(delay, batchSize, varDuplication, index, "post");
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPrePrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
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
std::string SynapseGroupMergedBase::getPostPrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
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
std::string SynapseGroupMergedBase::getSynVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
    return (singleBatch ? "" : "synBatchOffset + ") + index;
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getKernelVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
    return (singleBatch ? "" : "kernBatchOffset + ") + index;
}
//----------------------------------------------------------------------------
SynapseGroupMergedBase::SynapseGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                               Role role, const std::string &archetypeCode, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, precision, groups), m_ArchetypeCode(archetypeCode)
{
    const bool updateRole = ((role == Role::PresynapticUpdate)
                             || (role == Role::PostsynapticUpdate)
                             || (role == Role::SynapseDynamics));
    const WeightUpdateModels::Base *wum = getArchetype().getWUModel();

    // If role isn't an init role or weights aren't kernel
    if(role != Role::Init || !(getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL)) {
        addField("unsigned int", "rowStride",
                 [&backend](const SynapseGroupInternal &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });
        addField("unsigned int", "numSrcNeurons",
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
        addField("unsigned int", "numTrgNeurons",
                [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    }
    
    if(role == Role::PostsynapticUpdate || role == Role::SparseInit) {
        addField("unsigned int", "colStride",
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getMaxSourceConnections()); });
    }
    
    // If this role is one where postsynaptic input can be provided
    if(role == Role::PresynapticUpdate || role == Role::SynapseDynamics) {
        if(getArchetype().isDendriticDelayRequired()) {
            addPSPointerField(precision, "denDelay", backend.getDeviceVarPrefix() + "denDelay");
            addPSPointerField("unsigned int", "denDelayPtr", backend.getScalarAddressPrefix() + "denDelayPtr");
        }
        else {
            addPSPointerField(precision, "inSyn", backend.getDeviceVarPrefix() + "inSyn");
        }
    }

    if(role == Role::PresynapticUpdate) {
        if(getArchetype().isTrueSpikeRequired()) {
            addSrcPointerField("unsigned int", "srcSpkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");
            addSrcPointerField("unsigned int", "srcSpk", backend.getDeviceVarPrefix() + "glbSpk");
        }

        if(getArchetype().isSpikeEventRequired()) {
            addSrcPointerField("unsigned int", "srcSpkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
            addSrcPointerField("unsigned int", "srcSpkEvnt", backend.getDeviceVarPrefix() + "glbSpkEvnt");
        }
    }
    else if(role == Role::PostsynapticUpdate) {
        addTrgPointerField("unsigned int", "trgSpkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");
        addTrgPointerField("unsigned int", "trgSpk", backend.getDeviceVarPrefix() + "glbSpk");
    }

    // If this structure is used for updating rather than initializing
    if(updateRole) {
        // for all types of roles
        if (getArchetype().isPresynapticOutputRequired()) {
            addPreOutputPointerField(precision, "revInSyn", backend.getDeviceVarPrefix() + "revInSyn");
        }

        // If presynaptic population has delay buffers
        if(getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
            addSrcPointerField("unsigned int", "srcSpkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
        }

        // If postsynaptic population has delay buffers
        if(getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
            addTrgPointerField("unsigned int", "trgSpkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
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
                addSrcPointerField(v.type, v.name + "Pre", backend.getDeviceVarPrefix() + v.name);
            }
        }

        // Loop through variables in postsynaptic neuron model
        const auto postVars = getArchetype().getTrgNeuronGroup()->getNeuronModel()->getVars();
        for(const auto &v : postVars) {
            // If variable is referenced in code string, add target pointer
            if(code.find("$(" + v.name + "_post)") != std::string::npos) {
                addTrgPointerField(v.type, v.name + "Post", backend.getDeviceVarPrefix() + v.name);
            }
        }

        // Loop through extra global parameters in presynaptic neuron model
        const auto preEGPs = getArchetype().getSrcNeuronGroup()->getNeuronModel()->getExtraGlobalParams();
        for(const auto &e : preEGPs) {
            if(code.find("$(" + e.name + "_pre)") != std::string::npos) {
                const std::string prefix = Utils::isTypePointer(e.type) ? backend.getDeviceVarPrefix() : "";
                addField(e.type, e.name + "Pre",
                         [e, prefix](const SynapseGroupInternal &sg, size_t) { return prefix + e.name + sg.getSrcNeuronGroup()->getName(); },
                         FieldType::Dynamic);
            }
        }

        // Loop through extra global parameters in postsynaptic neuron model
        const auto postEGPs = getArchetype().getTrgNeuronGroup()->getNeuronModel()->getExtraGlobalParams();
        for(const auto &e : postEGPs) {
            if(code.find("$(" + e.name + "_post)") != std::string::npos) {
                const std::string prefix = Utils::isTypePointer(e.type) ? backend.getDeviceVarPrefix() : "";
                addField(e.type, e.name + "Post",
                         [e, prefix](const SynapseGroupInternal &sg, size_t) { return prefix + e.name + sg.getTrgNeuronGroup()->getName(); },
                         FieldType::Dynamic);
            }
        }

        // Add spike times if required
        if(wum->isPreSpikeTimeRequired()) {
            addSrcPointerField(timePrecision, "sTPre", backend.getDeviceVarPrefix() + "sT");
        }
        if(wum->isPostSpikeTimeRequired()) {
            addTrgPointerField(timePrecision, "sTPost", backend.getDeviceVarPrefix() + "sT");
        }
        if(wum->isPreSpikeEventTimeRequired()) {
            addSrcPointerField(timePrecision, "seTPre", backend.getDeviceVarPrefix() + "seT");
        }
        if(wum->isPrevPreSpikeTimeRequired()) {
            addSrcPointerField(timePrecision, "prevSTPre", backend.getDeviceVarPrefix() + "prevST");
        }
        if(wum->isPrevPostSpikeTimeRequired()) {
            addTrgPointerField(timePrecision, "prevSTPost", backend.getDeviceVarPrefix() + "prevST");
        }
        if(wum->isPrevPreSpikeEventTimeRequired()) {
            addSrcPointerField(timePrecision, "prevSETPre", backend.getDeviceVarPrefix() + "prevSET");
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
            const std::string prefix = backend.getDeviceVarPrefix() + v.name;
            addField(v.type + "*", v.name, [prefix](const SynapseGroupInternal &g, size_t) { return prefix + g.getFusedWUPreVarSuffix(); });
        }
        
        // Add presynaptic variables to struct
        for(const auto &v : wum->getPostVars()) {
            const std::string prefix = backend.getDeviceVarPrefix() + v.name;
            addField(v.type + "*", v.name, [prefix](const SynapseGroupInternal &g, size_t) { return prefix + g.getFusedWUPostVarSuffix(); });
        }

        // Add EGPs to struct
        addEGPs(wum->getExtraGlobalParams(), backend.getDeviceVarPrefix());
    }

    // Add pointers to connectivity data
    if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        addWeightSharingPointerField("unsigned int", "rowLength", backend.getDeviceVarPrefix() + "rowLength");
        addWeightSharingPointerField(getArchetype().getSparseIndType(), "ind", backend.getDeviceVarPrefix() + "ind");

        // Add additional structure for postsynaptic access
        if(backend.isPostsynapticRemapRequired() && !wum->getLearnPostCode().empty()
           && (role == Role::PostsynapticUpdate || role == Role::SparseInit))
        {
            addWeightSharingPointerField("unsigned int", "colLength", backend.getDeviceVarPrefix() + "colLength");
            addWeightSharingPointerField("unsigned int", "remap", backend.getDeviceVarPrefix() + "remap");
        }
    }
    else if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        addWeightSharingPointerField("uint32_t", "gp", backend.getDeviceVarPrefix() + "gp");
    }

    // If we're updating a group with procedural connectivity or initialising connectivity
    if((getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) || (role == Role::ConnectivityInit)) {
        // Add heterogeneous sparse connectivity initialiser model parameters
        addHeterogeneousParams<SynapseGroupMergedBase>(
            getArchetype().getConnectivityInitialiser().getSnippet()->getParamNames(), "",
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); },
            &SynapseGroupMergedBase::isSparseConnectivityInitParamHeterogeneous);


        // Add heterogeneous sparse connectivity initialiser derived parameters
        addHeterogeneousDerivedParams<SynapseGroupMergedBase>(
            getArchetype().getConnectivityInitialiser().getSnippet()->getDerivedParams(), "",
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); },
            &SynapseGroupMergedBase::isSparseConnectivityInitDerivedParamHeterogeneous);

        addEGPs(getArchetype().getConnectivityInitialiser().getSnippet()->getExtraGlobalParams(),
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
    const auto vars = wum->getVars();
    const auto &varInit = getArchetype().getWUVarInitialisers();
    if(getArchetype().getMatrixType() & SynapseMatrixWeight::GLOBAL) {
        // If this is an update role
        // **NOTE **global variable values aren't useful during initialization
        if(updateRole) {
            for(size_t v = 0; v < vars.size(); v++) {
                // If variable should be implemented heterogeneously, add scalar field
                if(isWUGlobalVarHeterogeneous(v)) {
                    addScalarField(vars[v].name,
                                   [v](const SynapseGroupInternal &sg, size_t)
                                   {
                                       return Utils::writePreciseString(sg.getWUConstInitVals().at(v));
                                   });
                }
            }
        }
    }
    // Otherwise (weights are individual or procedural)
    else {
        const bool connectInitRole = (role == Role::ConnectivityInit);
        const bool varInitRole = (role == Role::Init || role == Role::SparseInit);
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
                             [d](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getKernelSize().at(d)); });
                }
            }
        }

        // If weights are procedural, we're initializing individual variables or we're initialising variables in a kernel
        // **NOTE** some of these won't actually be required - could do this per-variable in loop over vars
        if((proceduralWeights && updateRole) || (connectInitRole && !getArchetype().getKernelSize().empty()) 
           || (varInitRole && (individualWeights || kernelWeights))) 
        {
            // Add heterogeneous variable initialization parameters and derived parameters
            addHeterogeneousVarInitParams<SynapseGroupMergedBase, SynapseWUVarAdapter>(
                &SynapseGroupMergedBase::isWUVarInitParamHeterogeneous);

            addHeterogeneousVarInitDerivedParams<SynapseGroupMergedBase, SynapseWUVarAdapter>(
                &SynapseGroupMergedBase::isWUVarInitDerivedParamHeterogeneous);
        }

        // Loop through variables
        for(size_t v = 0; v < vars.size(); v++) {
            // Variable initialisation is required if we're performing connectivity init and var init snippet requires a kernel or
            // We're performing some other sort of initialisation, the snippet DOESN'T require a kernel but has SOME code
            const auto var = vars[v];
            const auto *snippet = varInit.at(v).getSnippet();
            const bool varInitRequired = ((connectInitRole && snippet->requiresKernel()) 
                                          || (varInitRole && individualWeights && !snippet->requiresKernel() && !snippet->getCode().empty())
                                          || (varInitRole && kernelWeights && !snippet->getCode().empty()));

            // If we're performing an update with individual weights; or this variable should be initialised
            if((updateRole && individualWeights) || (kernelWeights && updateRole) || varInitRequired) {
                addWeightSharingPointerField(var.type, var.name, backend.getDeviceVarPrefix() + var.name);
            }

            // If we're performing a procedural update or this variable should be initialised, add any var init EGPs to structure
            if((proceduralWeights && updateRole) || varInitRequired) {
                const auto egps = snippet->getExtraGlobalParams();
                for(const auto &e : egps) {
                    const std::string prefix = Utils::isTypePointer(e.type) ? backend.getDeviceVarPrefix() : "";
                    addField(e.type, e.name + var.name,
                             [e, prefix, var](const SynapseGroupInternal &sg, size_t)
                             {
                                 if(sg.isWeightSharingSlave()) {
                                     return prefix + e.name + var.name + sg.getWeightSharingMaster()->getName();
                                 }
                                 else {
                                     return prefix + e.name + var.name + sg.getName();
                                 }
                             },
                             FieldType::Dynamic);
                }
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
    else if (role == Role::ConnectivityInit) {
        Utils::updateHash(getArchetype().getConnectivityInitHashDigest(), hash);
    }
    else {
        Utils::updateHash(getArchetype().getWUInitHashDigest(), hash);
    }

    // Update hash with number of neurons in pre and postsynaptic population
    updateHash([](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxConnections(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxSourceConnections(); }, hash);
    
    if(updateRole) {
        // Update hash with weight update model parameters and derived parameters
        updateHash([](const SynapseGroupInternal &g) { return g.getWUParams(); }, hash);
        updateHash([](const SynapseGroupInternal &g) { return g.getWUDerivedParams(); }, hash);

        // Update hash with presynaptic neuron population parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSrcNeuronParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getParams(); }, hash);
        
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSrcNeuronDerivedParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getDerivedParams(); }, hash);

        // Update hash with postsynaptic neuron population parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isTrgNeuronParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getParams(); }, hash);
        
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isTrgNeuronDerivedParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getDerivedParams(); }, hash);
    }


    // If we're updating a hash for a group with procedural connectivity or initialising connectivity
    if((getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) || (role == Role::ConnectivityInit)) {
        // Update hash with connectivity parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSparseConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); }, hash);

        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSparseConnectivityInitDerivedParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }, hash);
    }

    // If we're updating a hash for a group with Toeplitz connectivity
    if((getArchetype().getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) && updateRole) {
        // Update hash with connectivity parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isToeplitzConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getParams(); }, hash);

        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isToeplitzConnectivityInitDerivedParamReferenced,
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
        const bool varInitRole = (role == Role::Init || role == Role::SparseInit);
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
            updateVarInitParamHash<SynapseGroupMergedBase, SynapseWUVarAdapter>(
                &SynapseGroupMergedBase::isWUVarInitParamReferenced, hash);
            updateVarInitDerivedParamHash<SynapseGroupMergedBase, SynapseWUVarAdapter>(
                &SynapseGroupMergedBase::isWUVarInitDerivedParamReferenced, hash);
        }
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addPSPointerField(const std::string &type, const std::string &name, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getFusedPSVarSuffix(); });
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addPreOutputPointerField(const std::string &type, const std::string &name, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getFusedPreOutputSuffix(); });
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addSrcPointerField(const std::string &type, const std::string &name, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getSrcNeuronGroup()->getName(); });
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addTrgPointerField(const std::string &type, const std::string &name, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getTrgNeuronGroup()->getName(); });
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addWeightSharingPointerField(const std::string &type, const std::string &name, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name, 
             [prefix](const SynapseGroupInternal &sg, size_t)
             { 
                 if(sg.isWeightSharingSlave()) {
                     return prefix + sg.getWeightSharingMaster()->getName();
                 }
                 else {
                     return prefix + sg.getName();
                 }
             });
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication,
                                                const std::string &index, const std::string &prefix) const
{
    if (delay) {
        if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
            return prefix + ((batchSize == 1) ? "DelaySlot" : "BatchDelaySlot");
        }
        else if (varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
            return prefix + "DelayOffset + " + index;
        }
        else {
            return prefix + "BatchDelayOffset + " + index;
        }
    }
    else {
        if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
            return (batchSize == 1) ? "0" : "batch";
        }
        else if (varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
            return index;
        }
        else {
            return prefix + "BatchOffset + " + index;
        }
    }
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUParamReferenced(size_t paramIndex) const
{
    const auto *wum = getArchetype().getWUModel();
    const std::string paramName = wum->getParamNames().at(paramIndex);
    return isParamReferenced({getArchetypeCode()}, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUDerivedParamReferenced(size_t paramIndex) const
{
    const auto *wum = getArchetype().getWUModel();
    const std::string derivedParamName = wum->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({getArchetypeCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUGlobalVarReferenced(size_t varIndex) const
{
    // If synapse group has global WU variables
    if(getArchetype().getMatrixType() & SynapseMatrixWeight::GLOBAL) {
        const auto *wum = getArchetype().getWUModel();
        const std::string varName = wum->getVars().at(varIndex).name;
        return isParamReferenced({getArchetypeCode()}, varName);
    }
    // Otherwise, return false
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUVarInitParamReferenced(size_t varIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *varInitSnippet = getArchetype().getWUVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUVarInitDerivedParamReferenced(size_t varIndex, size_t paramIndex) const
{
    // If derived parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *varInitSnippet = getArchetype().getWUVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({varInitSnippet->getCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitParamReferenced(size_t paramIndex) const
{
    const auto *snippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const auto rowBuildStateVars = snippet->getRowBuildStateVars();
    const auto colBuildStateVars = snippet->getColBuildStateVars();

    // Build list of code strings containing row build code and any row build state variable values
    std::vector<std::string> codeStrings{snippet->getRowBuildCode(), snippet->getColBuildCode()};
    std::transform(rowBuildStateVars.cbegin(), rowBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });
    std::transform(colBuildStateVars.cbegin(), colBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });

    const std::string paramName = snippet->getParamNames().at(paramIndex);
    return isParamReferenced(codeStrings, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitDerivedParamReferenced(size_t paramIndex) const
{
    const auto *snippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const auto rowBuildStateVars = snippet->getRowBuildStateVars();
    const auto colBuildStateVars = snippet->getColBuildStateVars();

    // Build list of code strings containing row build code and any row build state variable values
    std::vector<std::string> codeStrings{snippet->getRowBuildCode(), snippet->getColBuildCode()};
    std::transform(rowBuildStateVars.cbegin(), rowBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });
    std::transform(colBuildStateVars.cbegin(), colBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });

    const std::string derivedParamName = snippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced(codeStrings, derivedParamName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitParamReferenced(size_t paramIndex) const
{
    const auto *snippet = getArchetype().getToeplitzConnectivityInitialiser().getSnippet();
    const auto diagonalBuildStateVars = snippet->getDiagonalBuildStateVars();

    // Build list of code strings containing diagonal build code and any diagonal build state variable values
    std::vector<std::string> codeStrings{snippet->getDiagonalBuildCode()};
    std::transform(diagonalBuildStateVars.cbegin(), diagonalBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });
   
    const std::string paramName = snippet->getParamNames().at(paramIndex);
    return isParamReferenced(codeStrings, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitDerivedParamReferenced(size_t paramIndex) const
{
    const auto *snippet = getArchetype().getToeplitzConnectivityInitialiser().getSnippet();
    const auto diagonalBuildStateVars = snippet->getDiagonalBuildStateVars();

    // Build list of code strings containing diagonal build code and any diagonal build state variable values
    std::vector<std::string> codeStrings{snippet->getDiagonalBuildCode()};
    std::transform(diagonalBuildStateVars.cbegin(), diagonalBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });

    const std::string derivedParamName = snippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced(codeStrings, derivedParamName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSrcNeuronParamReferenced(size_t paramIndex) const
{
    const auto *neuronModel = getArchetype().getSrcNeuronGroup()->getNeuronModel();
    const std::string paramName = neuronModel->getParamNames().at(paramIndex) + "_pre";
    return isParamReferenced({getArchetypeCode()}, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSrcNeuronDerivedParamReferenced(size_t paramIndex) const
{
    const auto *neuronModel = getArchetype().getSrcNeuronGroup()->getNeuronModel();
    const std::string derivedParamName = neuronModel->getDerivedParams().at(paramIndex).name + "_pre";
    return isParamReferenced({getArchetypeCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isTrgNeuronParamReferenced(size_t paramIndex) const
{
    const auto *neuronModel = getArchetype().getTrgNeuronGroup()->getNeuronModel();
    const std::string paramName = neuronModel->getParamNames().at(paramIndex) + "_post";
    return isParamReferenced({getArchetypeCode()}, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isTrgNeuronDerivedParamReferenced(size_t paramIndex) const
{
    const auto *neuronModel = getArchetype().getTrgNeuronGroup()->getNeuronModel();
    const std::string derivedParamName = neuronModel->getDerivedParams().at(paramIndex).name + "_post";
    return isParamReferenced({getArchetypeCode()}, derivedParamName);
}
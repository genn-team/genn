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
        addHeterogeneousVarInitParams<NeuronGroupMergedBase>(
            vars, &NeuronGroupInternal::getVarInitialisers,
            &NeuronGroupMergedBase::isVarInitParamHeterogeneous);

        addHeterogeneousVarInitDerivedParams<NeuronGroupMergedBase>(
            vars, &NeuronGroupInternal::getVarInitialisers,
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
                updateChildVarInitParamsHash(m_SortedCurrentSources, c, v,
                                             &NeuronGroupMergedBase::isCurrentSourceVarInitParamReferenced, 
                                             &CurrentSourceInternal::getVarInitialisers, hash);
                updateChildVarInitDerivedParamsHash(m_SortedCurrentSources, c, v,
                                                    &NeuronGroupMergedBase::isCurrentSourceVarInitDerivedParamReferenced, 
                                                    &CurrentSourceInternal::getVarInitialisers, hash);
            }
        }

        // Loop through child merged insyns
        for(size_t c = 0; c < getSortedArchetypeMergedInSyns().size(); c++) {
            const auto *sg = getSortedArchetypeMergedInSyns().at(c);

            // Loop through variables and update hash with variable initialisation parameters and derived parameters
            const auto &varInit = sg->getPSVarInitialisers();
            for(size_t v = 0; v < varInit.size(); v++) {
                if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    updateChildVarInitParamsHash(m_SortedMergedInSyns, c, v,
                                                 &NeuronGroupMergedBase::isPSMVarInitParamReferenced,
                                                 &SynapseGroupInternal::getPSVarInitialisers, hash);
                    updateChildVarInitDerivedParamsHash(m_SortedMergedInSyns, c, v,
                                                        &NeuronGroupMergedBase::isPSMVarInitDerivedParamReferenced,
                                                        &SynapseGroupInternal::getPSVarInitialisers, hash);
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

//----------------------------------------------------------------------------
// CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronInitGroupMerged::name = "NeuronInit";
//----------------------------------------------------------------------------
NeuronInitGroupMerged::NeuronInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                             const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, precision, timePrecision, backend, true, groups)
{
    // Build vector of vectors containing each child group's incoming 
    // synapse groups, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedInSynWithPostVars, &NeuronGroupInternal::getFusedInSynWithPostVars,
                             &SynapseGroupInternal::getWUPostInitHashDigest);

    // Build vector of vectors containing each child group's outgoing 
    // synapse groups, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedOutSynWithPreVars, &NeuronGroupInternal::getFusedOutSynWithPreVars,
                             &SynapseGroupInternal::getWUPreInitHashDigest);

    // Generate struct fields for incoming synapse groups with postsynaptic variables
    generateWUVar(backend, "WUPost", m_SortedInSynWithPostVars,
                  &WeightUpdateModels::Base::getPostVars, &SynapseGroupInternal::getWUPostVarInitialisers,
                  &NeuronInitGroupMerged::isInSynWUMVarInitParamHeterogeneous,
                  &NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamHeterogeneous,
                  &SynapseGroupInternal::getFusedWUPostVarSuffix);


    // Generate struct fields for outgoing synapse groups
    generateWUVar(backend, "WUPre", m_SortedOutSynWithPreVars,
                  &WeightUpdateModels::Base::getPreVars, &SynapseGroupInternal::getWUPreVarInitialisers,
                  &NeuronInitGroupMerged::isOutSynWUMVarInitParamHeterogeneous,
                  &NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamHeterogeneous,
                  &SynapseGroupInternal::getFusedWUPreVarSuffix);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isInSynWUMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isInSynWUMVarInitParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedInSynWithPostVars,
                                           [varIndex](const SynapseGroupInternal *s) { return s->getWUPostVarInitialisers().at(varIndex).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isInSynWUMVarInitDerivedParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedInSynWithPostVars,
                                           [varIndex](const SynapseGroupInternal *s) { return s->getWUPostVarInitialisers().at(varIndex).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isOutSynWUMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isOutSynWUMVarInitParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedOutSynWithPreVars,
                                           [varIndex](const SynapseGroupInternal *s) { return s->getWUPreVarInitialisers().at(varIndex).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isOutSynWUMVarInitDerivedParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedOutSynWithPreVars,
                                           [varIndex](const SynapseGroupInternal *s) { return s->getWUPreVarInitialisers().at(varIndex).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with generic neuron group data
    updateBaseHash(true, hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getInitHashDigest(), hash);

    // Update hash with each group's variable initialisation parameters and derived parameters
    updateVarInitParamHash<NeuronInitGroupMerged>(&NeuronGroupInternal::getVarInitialisers, 
                                                  &NeuronInitGroupMerged::isVarInitParamReferenced, hash);
    updateVarInitDerivedParamHash<NeuronInitGroupMerged>(&NeuronGroupInternal::getVarInitialisers,
                                                         &NeuronInitGroupMerged::isVarInitDerivedParamReferenced, hash);
    
    // Loop through child incoming synapse groups with postsynaptic variables
    for(size_t c = 0; c < getSortedArchetypeInSynWithPostVars().size(); c++) {
        const auto *sg = getSortedArchetypeInSynWithPostVars().at(c);

        // Loop through variables and update hash with variable initialisation parameters and derived parameters
        const auto &varInit = sg->getWUPostVarInitialisers();
        for(size_t v = 0; v < varInit.size(); v++) {
            updateChildVarInitParamsHash<NeuronInitGroupMerged>(
                m_SortedInSynWithPostVars, c, v, &NeuronInitGroupMerged::isInSynWUMVarInitParamReferenced, 
                &SynapseGroupInternal::getWUPostVarInitialisers, hash);
            updateChildVarInitDerivedParamsHash<NeuronInitGroupMerged>(
                m_SortedInSynWithPostVars, c, v, &NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamReferenced, 
                &SynapseGroupInternal::getWUPostVarInitialisers, hash);
        }
    }

    // Loop through child outgoing synapse groups with presynaptic variables
    for(size_t c = 0; c < getSortedArchetypeOutSynWithPreVars().size(); c++) {
        const auto *sg = getSortedArchetypeOutSynWithPreVars().at(c);

        // Loop through variables and update hash with variable initialisation parameters and derived parameters
        const auto &varInit = sg->getWUPreVarInitialisers();
        for(size_t v = 0; v < varInit.size(); v++) {
            updateChildVarInitParamsHash<NeuronInitGroupMerged>(
                m_SortedOutSynWithPreVars, c, v, &NeuronInitGroupMerged::isOutSynWUMVarInitParamReferenced, 
                &SynapseGroupInternal::getWUPreVarInitialisers, hash);
            updateChildVarInitDerivedParamsHash<NeuronInitGroupMerged>(
                m_SortedOutSynWithPreVars, c, v, &NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamReferenced, 
                &SynapseGroupInternal::getWUPreVarInitialisers, hash);
        }
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::generateWUVar(const BackendBase &backend,
                                          const std::string &fieldPrefixStem,
                                          const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                                          Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                                          const std::vector<Models::VarInit> &(SynapseGroupInternal:: *getVarInitialiserFn)(void) const,
                                          bool(NeuronInitGroupMerged::*isParamHeterogeneousFn)(size_t, size_t, size_t) const,
                                          bool(NeuronInitGroupMerged::*isDerivedParamHeterogeneousFn)(size_t, size_t, size_t) const,
                                          const std::string&(SynapseGroupInternal::*getFusedVarSuffix)(void) const)
{
    // Loop through synapse groups
    const auto &archetypeSyns = sortedSyn.front();
    for(size_t i = 0; i < archetypeSyns.size(); i++) {
        const auto *sg = archetypeSyns.at(i);

        // Loop through variables
        const auto vars = (sg->getWUModel()->*getVars)();
        const auto &varInit = (sg->*getVarInitialiserFn)();
        for(size_t v = 0; v < vars.size(); v++) {
            // Add pointers to state variable
            const auto var = vars.at(v);
            if(!varInit.at(v).getSnippet()->getCode().empty()) {
                assert(!Utils::isTypePointer(var.type));
                addField(var.type + "*", var.name + fieldPrefixStem + std::to_string(i),
                         [i, var, &backend, &sortedSyn, getFusedVarSuffix](const NeuronGroupInternal &, size_t groupIndex)
                         {
                             const std::string &varMergeSuffix = (sortedSyn.at(groupIndex).at(i)->*getFusedVarSuffix)();
                             return backend.getDeviceVarPrefix() + var.name + varMergeSuffix;
                         });
            }

            // Also add any heterogeneous, derived or extra global parameters required for initializers
            const auto *varInitSnippet = varInit.at(v).getSnippet();
            addHeterogeneousChildVarInitParams<NeuronInitGroupMerged>(varInitSnippet->getParamNames(), sortedSyn, i, v, var.name + fieldPrefixStem,
                                                                      isParamHeterogeneousFn, getVarInitialiserFn);
            addHeterogeneousChildVarInitDerivedParams<NeuronInitGroupMerged>(varInitSnippet->getDerivedParams(), sortedSyn, i, v, var.name + fieldPrefixStem,
                                                                             isDerivedParamHeterogeneousFn, getVarInitialiserFn);
            addChildEGPs(varInitSnippet->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), var.name + fieldPrefixStem,
                         [var, &sortedSyn](size_t groupIndex, size_t childIndex)
                         {
                             return var.name + sortedSyn.at(groupIndex).at(childIndex)->getName();
                         });
        }
    }
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isInSynWUMVarInitParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeInSynWithPostVars().at(childIndex)->getWUPostVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeInSynWithPostVars().at(childIndex)->getWUPostVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({varInitSnippet->getCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isOutSynWUMVarInitParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeOutSynWithPreVars().at(childIndex)->getWUPreVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeOutSynWithPreVars().at(childIndex)->getWUPreVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({varInitSnippet->getCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
// CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDendriticDelayUpdateGroupMerged::name = "SynapseDendriticDelayUpdate";
//----------------------------------------------------------------------------
SynapseDendriticDelayUpdateGroupMerged::SynapseDendriticDelayUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, precision, groups)
{
    addField("unsigned int*", "denDelayPtr", 
             [&backend](const SynapseGroupInternal &sg, size_t) 
             {
                 return backend.getScalarAddressPrefix() + "denDelayPtr" + sg.getFusedPSVarSuffix(); 
             });
}

// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseConnectivityHostInitGroupMerged::name = "SynapseConnectivityHostInit";
//------------------------------------------------------------------------
SynapseConnectivityHostInitGroupMerged::SynapseConnectivityHostInitGroupMerged(size_t index, const std::string &precision, const std::string&, const BackendBase &backend,
                                                                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, precision, groups)
{
    // **TODO** these could be generic
    addField("unsigned int", "numSrcNeurons",
             [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "numTrgNeurons",
             [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "rowStride",
             [&backend](const SynapseGroupInternal &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });

    // Add heterogeneous connectivity initialiser model parameters
    addHeterogeneousParams<SynapseConnectivityHostInitGroupMerged>(
        getArchetype().getConnectivityInitialiser().getSnippet()->getParamNames(), "",
        [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); },
        &SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous);

    // Add heterogeneous connectivity initialiser derived parameters
    addHeterogeneousDerivedParams<SynapseConnectivityHostInitGroupMerged>(
        getArchetype().getConnectivityInitialiser().getSnippet()->getDerivedParams(), "",
        [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); },
        &SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous);

    // Add EGP pointers to struct for both host and device EGPs if they are seperate
    const auto egps = getArchetype().getConnectivityInitialiser().getSnippet()->getExtraGlobalParams();
    for(const auto &e : egps) {
        addField(e.type + "*", e.name,
                 [e](const SynapseGroupInternal &g, size_t) { return "&" + e.name + g.getName(); },
                 FieldType::Host);

        if(!backend.getDeviceVarPrefix().empty()) {
            addField(e.type + "*", backend.getDeviceVarPrefix() + e.name,
                     [e, &backend](const SynapseGroupInternal &g, size_t)
                     {
                         return "&" + backend.getDeviceVarPrefix() + e.name + g.getName();
                     });
        }
        if(!backend.getHostVarPrefix().empty()) {
            addField(e.type + "*", backend.getHostVarPrefix() + e.name,
                     [e, &backend](const SynapseGroupInternal &g, size_t)
                     {
                         return "&" + backend.getHostVarPrefix() + e.name + g.getName();
                     });
        }
    }
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous(size_t paramIndex) const
{
    return (isConnectivityInitParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg){ return sg.getConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const
{
    return (isConnectivityInitDerivedParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamReferenced(size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *connectInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const std::string paramName = connectInitSnippet->getParamNames().at(paramIndex);
    return isParamReferenced({connectInitSnippet->getHostInitCode()}, paramName);
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamReferenced(size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *connectInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const std::string paramName = connectInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({connectInitSnippet->getHostInitCode()}, paramName);
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
bool SynapseGroupMergedBase::isConnectivityInitParamHeterogeneous(size_t paramIndex) const
{
    return (isConnectivityInitParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const
{
    return (isConnectivityInitDerivedParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }));
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
:   GroupMerged<SynapseGroupInternal>(index, precision, groups), m_ArchetypeCode(archetypeCode)
{
    const bool updateRole = ((role == Role::PresynapticUpdate)
                             || (role == Role::PostsynapticUpdate)
                             || (role == Role::SynapseDynamics));
    const WeightUpdateModels::Base *wum = getArchetype().getWUModel();

    addField("unsigned int", "rowStride",
             [&backend](const SynapseGroupInternal &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });
    if(role == Role::PostsynapticUpdate || role == Role::SparseInit) {
        addField("unsigned int", "colStride",
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getMaxSourceConnections()); });
    }

    addField("unsigned int", "numSrcNeurons",
             [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "numTrgNeurons",
             [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });

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
                const bool isPointer = Utils::isTypePointer(e.type);
                const std::string prefix = isPointer ? backend.getDeviceVarPrefix() : "";
                addField(e.type, e.name + "Pre",
                         [e, prefix](const SynapseGroupInternal &sg, size_t) { return prefix + e.name + sg.getSrcNeuronGroup()->getName(); },
                         Utils::isTypePointer(e.type) ? FieldType::PointerEGP : FieldType::ScalarEGP);
            }
        }

        // Loop through extra global parameters in postsynaptic neuron model
        const auto postEGPs = getArchetype().getTrgNeuronGroup()->getNeuronModel()->getExtraGlobalParams();
        for(const auto &e : postEGPs) {
            if(code.find("$(" + e.name + "_post)") != std::string::npos) {
                const bool isPointer = Utils::isTypePointer(e.type);
                const std::string prefix = isPointer ? backend.getDeviceVarPrefix() : "";
                addField(e.type, e.name + "Post",
                         [e, prefix](const SynapseGroupInternal &sg, size_t) { return prefix + e.name + sg.getTrgNeuronGroup()->getName(); },
                         Utils::isTypePointer(e.type) ? FieldType::PointerEGP : FieldType::ScalarEGP);
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

        // Add additional structure for synapse dynamics access if required
        if((role == Role::SynapseDynamics || role == Role::SparseInit) &&
           backend.isSynRemapRequired(getArchetype()))
        {
            addWeightSharingPointerField("unsigned int", "synRemap", backend.getDeviceVarPrefix() + "synRemap");
        }
    }
    else if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        addWeightSharingPointerField("uint32_t", "gp", backend.getDeviceVarPrefix() + "gp");
    }

    // If we're updating a group with procedural connectivity or initialising connectivity
    if((getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) || (role == Role::ConnectivityInit)) {
        // Add heterogeneous connectivity initialiser model parameters
        addHeterogeneousParams<SynapseGroupMergedBase>(
            getArchetype().getConnectivityInitialiser().getSnippet()->getParamNames(), "",
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); },
            &SynapseGroupMergedBase::isConnectivityInitParamHeterogeneous);


        // Add heterogeneous connectivity initialiser derived parameters
        addHeterogeneousDerivedParams<SynapseGroupMergedBase>(
            getArchetype().getConnectivityInitialiser().getSnippet()->getDerivedParams(), "",
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); },
            &SynapseGroupMergedBase::isConnectivityInitDerivedParamHeterogeneous);

        addEGPs(getArchetype().getConnectivityInitialiser().getSnippet()->getExtraGlobalParams(),
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
        const bool varInitRole = (role == Role::DenseInit || role == Role::SparseInit);
        const bool proceduralWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL);
        const bool kernelWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL);
        const bool individualWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);

        // If synapse group has a kernel and we're either updating 
        // with procedural weights or initialising individual weights
        if(!getArchetype().getKernelSize().empty() && ((proceduralWeights && updateRole) || (kernelWeights && updateRole) || (connectInitRole && individualWeights))) {
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
           || (varInitRole && individualWeights)) 
        {
            // Add heterogeneous variable initialization parameters and derived parameters
            addHeterogeneousVarInitParams<SynapseGroupMergedBase>(
                wum->getVars(), &SynapseGroupInternal::getWUVarInitialisers,
                &SynapseGroupMergedBase::isWUVarInitParamHeterogeneous);

            addHeterogeneousVarInitDerivedParams<SynapseGroupMergedBase>(
                wum->getVars(), &SynapseGroupInternal::getWUVarInitialisers,
                &SynapseGroupMergedBase::isWUVarInitDerivedParamHeterogeneous);
        }

        // Loop through variables
        for(size_t v = 0; v < vars.size(); v++) {
            // Variable initialisation is required if we're performing connectivity init and var init snippet requires a kernel or
            // We're performing some other sort of initialisation, the snippet DOESN'T require a kernel but has SOME code
            const auto var = vars[v];
            const auto *snippet = varInit.at(v).getSnippet();
            const bool varInitRequired = ((connectInitRole && snippet->requiresKernel()) 
                                          || (varInitRole && !snippet->requiresKernel() && !snippet->getCode().empty()));

            // If we're performing an update with individual weights; or this variable should be initialised
            if((updateRole && individualWeights) || (kernelWeights && updateRole) || varInitRequired) {
                addWeightSharingPointerField(var.type, var.name, backend.getDeviceVarPrefix() + var.name);
            }

            // If we're performing a procedural update or this variable should be initialised, add any var init EGPs to structure
            if((proceduralWeights && updateRole) || varInitRequired) {
                const auto egps = snippet->getExtraGlobalParams();
                for(const auto &e : egps) {
                    const bool isPointer = Utils::isTypePointer(e.type);
                    const std::string prefix = isPointer ? backend.getDeviceVarPrefix() : "";
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
                             isPointer ? FieldType::PointerEGP : FieldType::ScalarEGP);
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
            &SynapseGroupMergedBase::isConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); }, hash);

        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isConnectivityInitDerivedParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }, hash);
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

        // If synapse group has a kernel and we're either updating with procedural  
        // weights or initialising individual weights, update hash with kernel size
        if(!getArchetype().getKernelSize().empty() && ((proceduralWeights && updateRole) || (connectInitRole && individualWeights))) {
            updateHash([](const SynapseGroupInternal &g) { return g.getKernelSize(); }, hash);
        }

        // If weights are procedural, we're initializing individual variables or we're initialising variables in a kernel
        // **NOTE** some of these won't actually be required - could do this per-variable in loop over vars
        if((proceduralWeights && updateRole) || (connectInitRole && !getArchetype().getKernelSize().empty())
           || (varInitRole && individualWeights))
        {
            // Update hash with each group's variable initialisation parameters and derived parameters
            updateVarInitParamHash<SynapseGroupMergedBase>(&SynapseGroupInternal::getWUVarInitialisers, 
                                                           &SynapseGroupMergedBase::isWUVarInitParamReferenced, hash);
            updateVarInitDerivedParamHash<SynapseGroupMergedBase>(&SynapseGroupInternal::getWUVarInitialisers,
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
bool SynapseGroupMergedBase::isConnectivityInitParamReferenced(size_t paramIndex) const
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
bool SynapseGroupMergedBase::isConnectivityInitDerivedParamReferenced(size_t paramIndex) const
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

//----------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string PresynapticUpdateGroupMerged::name = "PresynapticUpdate";

//----------------------------------------------------------------------------
// CodeGenerator::PostsynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string PostsynapticUpdateGroupMerged::name = "PostsynapticUpdate";

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDynamicsGroupMerged::name = "SynapseDynamics";

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDenseInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseDenseInitGroupMerged::name = "SynapseDenseInit";

//----------------------------------------------------------------------------
// CodeGenerator::SynapseSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseSparseInitGroupMerged::name = "SynapseSparseInit";

// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseConnectivityInitGroupMerged::name = "SynapseConnectivityInit";

//----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateGroupMerged::name = "CustomUpdate";
//----------------------------------------------------------------------------
CustomUpdateGroupMerged::CustomUpdateGroupMerged(size_t index, const std::string &precision, const std::string&, const BackendBase &backend,
                                                 const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups)
:   GroupMerged<CustomUpdateInternal>(index, precision, groups)
{
    addField("unsigned int", "size",
             [](const CustomUpdateInternal &c, size_t) { return std::to_string(c.getSize()); });
    
    // If some variables are delayed, add delay pointer
    if(getArchetype().getDelayNeuronGroup() != nullptr) {
        addField("unsigned int*", "spkQuePtr", 
                 [&backend](const CustomUpdateInternal &cg, size_t) 
                 { 
                     return backend.getScalarAddressPrefix() + "spkQuePtr" + cg.getDelayNeuronGroup()->getName(); 
                 });
    }

    // Add heterogeneous custom update model parameters
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    addHeterogeneousParams<CustomUpdateGroupMerged>(
        cm->getParamNames(), "",
        [](const CustomUpdateInternal &cg) { return cg.getParams(); },
        &CustomUpdateGroupMerged::isParamHeterogeneous);

    // Add heterogeneous weight update model derived parameters
    addHeterogeneousDerivedParams<CustomUpdateGroupMerged>(
        cm->getDerivedParams(), "",
        [](const CustomUpdateInternal &cg) { return cg.getDerivedParams(); },
        &CustomUpdateGroupMerged::isDerivedParamHeterogeneous);

    // Add variables to struct
    addVars(cm->getVars(), backend.getDeviceVarPrefix());

    // Add variable references to struct
    addVarReferences(cm->getVarRefs(), backend.getDeviceVarPrefix(),
                    [](const CustomUpdateInternal &cg) { return cg.getVarReferences(); });

    // Add EGPs to struct
    this->addEGPs(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());
}
//----------------------------------------------------------------------------
bool CustomUpdateGroupMerged::isParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomUpdateInternal &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------    
bool CustomUpdateGroupMerged::isDerivedParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomUpdateInternal &cg) { return cg.getDerivedParams(); });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getHashDigest(), hash);

    // Update hash with each group's custom update size
    updateHash([](const CustomUpdateInternal &cg) { return cg.getSize(); }, hash);

    // Update hash with each group's parameters, derived parameters and variable references
    updateHash([](const CustomUpdateInternal &cg) { return cg.getParams(); }, hash);
    updateHash([](const CustomUpdateInternal &cg) { return cg.getDerivedParams(); }, hash);
    updateHash([](const CustomUpdateInternal &cg) { return cg.getVarReferences(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
std::string CustomUpdateGroupMerged::getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    // If variable is shared, the batch size is one or this custom update isn't batched, batch offset isn't required
    return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1 || !getArchetype().isBatched()) ? "" : "batchOffset + ") + index;
}
//----------------------------------------------------------------------------
std::string CustomUpdateGroupMerged::getVarRefIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    // If delayed, variable is shared, the batch size is one or this custom update isn't batched, batch delay offset isn't required
    if(delay) {
        return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1 || !getArchetype().isBatched()) ? "delayOffset + " : "batchDelayOffset + ") + index;
    }
    else {
        return getVarIndex(batchSize, varDuplication, index);
    }    
}

// ----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateWUGroupMergedBase
//----------------------------------------------------------------------------
bool CustomUpdateWUGroupMergedBase::isParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomUpdateWUInternal &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------
bool CustomUpdateWUGroupMergedBase::isDerivedParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomUpdateWUInternal &cg) { return cg.getDerivedParams(); });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateWUGroupMergedBase::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getHashDigest(), hash);

    // Update hash with sizes of pre and postsynaptic neuron groups
    updateHash([](const CustomUpdateWUInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const CustomUpdateWUInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    // Update hash with each group's parameters, derived parameters and variable referneces
    updateHash([](const CustomUpdateWUInternal &cg) { return cg.getParams(); }, hash);
    updateHash([](const CustomUpdateWUInternal &cg) { return cg.getDerivedParams(); }, hash);
    updateHash([](const CustomUpdateWUInternal &cg) { return cg.getVarReferences(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
std::string CustomUpdateWUGroupMergedBase::getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1 || !getArchetype().isBatched()) ? "" : "batchOffset + ") + index;
}
//----------------------------------------------------------------------------
std::string CustomUpdateWUGroupMergedBase::getVarRefIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1 || !getArchetype().isBatched()) ? "" : "batchOffset + ") + index;
}
//----------------------------------------------------------------------------
CustomUpdateWUGroupMergedBase::CustomUpdateWUGroupMergedBase(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                             const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   GroupMerged<CustomUpdateWUInternal>(index, precision, groups)
{
    addField("unsigned int", "rowStride",
             [&backend](const CustomUpdateWUInternal &cg, size_t) 
             { 
                 const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                 return std::to_string(backend.getSynapticMatrixRowStride(*sgInternal)); 
             });
    
    addField("unsigned int", "numSrcNeurons",
             [](const CustomUpdateWUInternal &cg, size_t) 
             {
                 const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                 return std::to_string(sgInternal->getSrcNeuronGroup()->getNumNeurons()); 
             });

    addField("unsigned int", "numTrgNeurons",
             [](const CustomUpdateWUInternal &cg, size_t)
             { 
                 const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                 return std::to_string(sgInternal->getTrgNeuronGroup()->getNumNeurons()); 
             });

    // If synapse group has sparse connectivity
    if(getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        addField(getArchetype().getSynapseGroup()->getSparseIndType() + "*", "ind", 
                 [&backend](const CustomUpdateWUInternal &cg, size_t) 
                 { 
                     return backend.getDeviceVarPrefix() + "ind" + cg.getSynapseGroup()->getName(); 
                 });

        // If the referenced synapse group requires synaptic remapping and matrix type is sparse, add field
        if(backend.isSynRemapRequired(*getArchetype().getSynapseGroup())) {
            addField("unsigned int*", "synRemap", 
                     [&backend](const CustomUpdateWUInternal &cg, size_t) 
                     { 
                         return backend.getDeviceVarPrefix() + "synRemap" + cg.getSynapseGroup()->getName(); 
                     });
        }
        // Otherwise, add row length
        else {
            addField("unsigned int*", "rowLength",
                     [&backend](const CustomUpdateWUInternal &cg, size_t) 
                     { 
                         return backend.getDeviceVarPrefix() + "rowLength" + cg.getSynapseGroup()->getName(); 
                     });
        }
    }

    // Add heterogeneous custom update model parameters
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    addHeterogeneousParams<CustomUpdateWUGroupMerged>(
        cm->getParamNames(), "",
        [](const CustomUpdateWUInternal &cg) { return cg.getParams(); },
        &CustomUpdateWUGroupMergedBase::isParamHeterogeneous);

    // Add heterogeneous weight update model derived parameters
    addHeterogeneousDerivedParams<CustomUpdateWUGroupMerged>(
        cm->getDerivedParams(), "",
        [](const CustomUpdateWUInternal &cg) { return cg.getDerivedParams(); },
        &CustomUpdateWUGroupMergedBase::isDerivedParamHeterogeneous);

    // Add variables to struct
    addVars(cm->getVars(), backend.getDeviceVarPrefix());

    // Add variable references to struct
    const auto varRefs = cm->getVarRefs();
    addVarReferences(varRefs, backend.getDeviceVarPrefix(),
                    [](const CustomUpdateWUInternal &cg) { return cg.getVarReferences(); });

     // Loop through variables
    for(size_t v = 0; v < varRefs.size(); v++) {
        // If variable has a transpose 
        if(getArchetype().getVarReferences().at(v).getTransposeSynapseGroup() != nullptr) {
            // Add field with transpose suffix, pointing to transpose var
            addField(varRefs[v].type + "*", varRefs[v].name + "Transpose",
                     [&backend, v](const CustomUpdateWUInternal &g, size_t)
                     {
                         const auto varRef = g.getVarReferences().at(v);
                         return backend.getDeviceVarPrefix() + varRef.getTransposeVar().name + varRef.getTransposeTargetName();
                     });
            }
    }
    // Add EGPs to struct
    this->addEGPs(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());
}
// ----------------------------------------------------------------------------
// CustomUpdateWUGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateWUGroupMerged::name = "CustomUpdateWU";

//----------------------------------------------------------------------------
// CustomUpdateTransposeWUGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateTransposeWUGroupMerged::name = "CustomUpdateTransposeWU";

// ----------------------------------------------------------------------------
// CustomUpdateInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateInitGroupMerged::name = "CustomUpdateInit";
//----------------------------------------------------------------------------
CustomUpdateInitGroupMerged::CustomUpdateInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                         const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomUpdateInternal>(index, precision, backend, groups)
{
    addField("unsigned int", "size",
             [](const CustomUpdateInternal &c, size_t) { return std::to_string(c.getSize()); });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    
    // Update hash with generic custom update init data
    updateBaseHash(hash);

    // Update hash with size of custom update
    updateHash([](const CustomUpdateInternal &cg) { return cg.getSize(); }, hash);

    return hash.get_digest();
}

// ----------------------------------------------------------------------------
// CustomWUUpdateDenseInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateDenseInitGroupMerged::name = "CustomWUUpdateDenseInit";
//----------------------------------------------------------------------------
CustomWUUpdateDenseInitGroupMerged::CustomWUUpdateDenseInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                       const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal>(index, precision, backend, groups)
{
    addField("unsigned int", "rowStride",
             [&backend](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(*cg.getSynapseGroup())); });
  
    addField("unsigned int", "numSrcNeurons",
             [](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "numTrgNeurons",
             [](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomWUUpdateDenseInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    
    // Update hash with generic custom update init data
    updateBaseHash(hash);

    // Update hash with sizes of pre and postsynaptic neuron groups
    updateHash([](const CustomUpdateWUInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const CustomUpdateWUInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
               }, hash);


    // **TODO** rowstride

    return hash.get_digest();
}

// ----------------------------------------------------------------------------
// CustomWUUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateSparseInitGroupMerged::name = "CustomWUUpdateSparseInit";
//----------------------------------------------------------------------------
CustomWUUpdateSparseInitGroupMerged::CustomWUUpdateSparseInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                         const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal>(index, precision, backend, groups)
{
    addField("unsigned int", "rowStride",
             [&backend](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(*cg.getSynapseGroup())); });

    addField("unsigned int", "numSrcNeurons",
             [](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "numTrgNeurons",
             [](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); });

    addField("unsigned int*", "rowLength", 
             [&backend](const CustomUpdateWUInternal &cg, size_t) 
             { 
                 const SynapseGroupInternal *sg = cg.getSynapseGroup();
                 return backend.getDeviceVarPrefix() + "rowLength" + sg->getName();
             });
    addField(getArchetype().getSynapseGroup()->getSparseIndType() + "*", "ind", 
             [&backend](const CustomUpdateWUInternal &cg, size_t) 
             { 
                 const SynapseGroupInternal *sg = cg.getSynapseGroup();
                 return backend.getDeviceVarPrefix() + "ind" + sg->getName();
             });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomWUUpdateSparseInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    
    // Update hash with generic custom update init data
    updateBaseHash(hash);

    // Update hash with sizes of pre and postsynaptic neuron groups
    updateHash([](const CustomUpdateWUInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const CustomUpdateWUInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
               }, hash);


    // **TODO** rowstride

    return hash.get_digest();
}

// ----------------------------------------------------------------------------
// CustomUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateHostReductionGroupMerged::name = "CustomUpdateHostReduction";
//----------------------------------------------------------------------------
CustomUpdateHostReductionGroupMerged::CustomUpdateHostReductionGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                           const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups)
:   CustomUpdateHostReductionGroupMergedBase<CustomUpdateInternal>(index, precision, backend, groups)
{
    addField("unsigned int", "size",
             [](const CustomUpdateInternal &c, size_t) { return std::to_string(c.getSize()); });

    // If some variables are delayed, add delay pointer
    // **NOTE** this is HOST delay pointer
    if(getArchetype().getDelayNeuronGroup() != nullptr) {
        addField("unsigned int*", "spkQuePtr", 
                 [&](const CustomUpdateInternal &cg, size_t) 
                 { 
                     return "spkQuePtr" + cg.getDelayNeuronGroup()->getName(); 
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
:   CustomUpdateHostReductionGroupMergedBase<CustomUpdateWUInternal>(index, precision, backend, groups)
{
    addField("unsigned int", "size",
             [&backend](const CustomUpdateWUInternal &cg, size_t) 
             {
                 return std::to_string(cg.getSynapseGroup()->getMaxConnections() * (size_t)cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); 
             });
}

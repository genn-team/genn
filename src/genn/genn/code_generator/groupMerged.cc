#include "code_generator/groupMerged.h"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronSpikeQueueUpdateGroupMerged::name = "NeuronSpikeQueueUpdate";
//----------------------------------------------------------------------------
NeuronSpikeQueueUpdateGroupMerged::NeuronSpikeQueueUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                     const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   GroupMerged<NeuronGroupInternal>(index, typeContext, groups)
{
    using namespace Type;

    if(getArchetype().isDelayRequired()) {
        addPointerField<Uint32>("spkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
    } 

    addPointerField<Uint32>("spkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField<Uint32>("spkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
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
// GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronPrevSpikeTimeUpdateGroupMerged::name = "NeuronPrevSpikeTimeUpdate";
//----------------------------------------------------------------------------
NeuronPrevSpikeTimeUpdateGroupMerged::NeuronPrevSpikeTimeUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                           const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   GroupMerged<NeuronGroupInternal>(index, typeContext, groups)
{
    using namespace Type;

    if(getArchetype().isDelayRequired()) {
        addPointerField<Uint32>("spkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
    } 

    addPointerField<Uint32>("spkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField<Uint32>("spkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
    }

    if(getArchetype().isPrevSpikeTimeRequired()) {
        addPointerField<Uint32>("spk", backend.getDeviceVarPrefix() + "glbSpk");
        addPointerField(getTimeType(), "prevST", backend.getDeviceVarPrefix() + "prevST");
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        addPointerField<Uint32>("spkEvnt", backend.getDeviceVarPrefix() + "glbSpkEvnt");
        addPointerField(getTimeType(), "prevSET", backend.getDeviceVarPrefix() + "prevSET");
    }

    if(getArchetype().isDelayRequired()) {
        addField<Uint32>("numNeurons",
                         [](const auto &ng, size_t) { return std::to_string(ng.getNumNeurons()); });
    }
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronGroupMergedBase
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
NeuronGroupMergedBase::NeuronGroupMergedBase(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend, 
                                             bool init, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   GroupMerged<NeuronGroupInternal>(index, typeContext, groups)
{
    using namespace Type;

    // Build vector of vectors containing each child group's merged in syns, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedMergedInSyns, &NeuronGroupInternal::getFusedPSMInSyn,
                             init ? &SynapseGroupInternal::getPSInitHashDigest : &SynapseGroupInternal::getPSHashDigest);

    // Build vector of vectors containing each child group's merged out syns with pre output, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedMergedPreOutputOutSyns, &NeuronGroupInternal::getFusedPreOutputOutSyn,
                             init ? &SynapseGroupInternal::getPreOutputInitHashDigest : &SynapseGroupInternal::getPreOutputHashDigest);

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedCurrentSources, &NeuronGroupInternal::getCurrentSources,
                             init ? &CurrentSourceInternal::getInitHashDigest : &CurrentSourceInternal::getHashDigest);

    addField<Uint32>("numNeurons",
                     [](const auto &ng, size_t) { return std::to_string(ng.getNumNeurons()); });

    addPointerField<Uint32>("spkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");
    addPointerField<Uint32>("spk", backend.getDeviceVarPrefix() + "glbSpk");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField<Uint32>("spkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
        addPointerField<Uint32>("spkEvnt", backend.getDeviceVarPrefix() + "glbSpkEvnt");
    }

    if(getArchetype().isDelayRequired()) {
        addPointerField<Uint32>("spkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
    }

    if(getArchetype().isSpikeTimeRequired()) {
        addPointerField(getTimeType(), "sT", backend.getDeviceVarPrefix() + "sT");
    }
    if(getArchetype().isSpikeEventTimeRequired()) {
        addPointerField(getTimeType(), "seT", backend.getDeviceVarPrefix() + "seT");
    }

    if(getArchetype().isPrevSpikeTimeRequired()) {
        addPointerField(getTimeType(), "prevST", backend.getDeviceVarPrefix() + "prevST");
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        addPointerField(getTimeType(), "prevSET", backend.getDeviceVarPrefix() + "prevSET");
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
            addPointerField(var.type, var.name, backend.getDeviceVarPrefix() + var.name);
        }

        // If we're initializing, add any var init EGPs to structure
        if(init) {
            addEGPs(varInit.at(var.name).getSnippet()->getExtraGlobalParams(), backend.getDeviceVarPrefix(), var.name);
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
        addMergedInSynPointerField(getScalarType(), "inSynInSyn", i, backend.getDeviceVarPrefix() + "inSyn");

        // Add pointer to dendritic delay buffer if required
        if(sg->isDendriticDelayRequired()) {
            addMergedInSynPointerField(getScalarType(), "denDelayInSyn", i, backend.getDeviceVarPrefix() + "denDelay");
            addMergedInSynPointerField(Uint32::getInstance(), "denDelayPtrInSyn", i, backend.getScalarAddressPrefix() + "denDelayPtr");
        }

        // Loop through variables
        const auto &varInit = sg->getPSVarInitialisers();
        for(const auto &var : sg->getPSModel()->getVars()) {
            // Add pointers to state variable
            if(!init || !varInit.at(var.name).getSnippet()->getCode().empty()) {
                addMergedInSynPointerField(var.type, var.name + "InSyn", i, backend.getDeviceVarPrefix() + var.name);
            }

            // If we're generating an initialization structure, also add any heterogeneous parameters, derived parameters or extra global parameters required for initializers
            if(init) {
                const auto *varInitSnippet = varInit.at(var.name).getSnippet();
                addHeterogeneousChildVarInitParams(varInitSnippet->getParamNames(), m_SortedMergedInSyns, i, var.name, "InSyn",
                                                    &NeuronGroupMergedBase::isPSMVarInitParamHeterogeneous, &SynapseGroupInternal::getPSVarInitialisers);
                addHeterogeneousChildVarInitDerivedParams(varInitSnippet->getDerivedParams(), m_SortedMergedInSyns, i, var.name, "InSyn",
                                                            &NeuronGroupMergedBase::isPSMVarInitDerivedParamHeterogeneous, &SynapseGroupInternal::getPSVarInitialisers);
                addChildEGPs(varInitSnippet->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), var.name + "InSyn",
                                [var, this](size_t groupIndex, size_t childIndex)
                                {
                                    return var.name + m_SortedMergedInSyns.at(groupIndex).at(childIndex)->getFusedPSVarSuffix();
                                });
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
        addMergedPreOutputOutSynPointerField(getScalarType(), "revInSynOutSyn", i, backend.getDeviceVarPrefix() + "revInSyn");
    }
    
    // Loop through current sources to archetypical neuron group in sorted order
    for(size_t i = 0; i < getSortedArchetypeCurrentSources().size(); i++) {
        const auto *cs = getSortedArchetypeCurrentSources().at(i);

        // Loop through variables
        const auto &varInit = cs->getVarInitialisers();
        for(const auto &var : cs->getCurrentSourceModel()->getVars()) {
            // Add pointers to state variable
            if(!init || !varInit.at(var.name).getSnippet()->getCode().empty()) {
                addField(var.type->getPointerType(), var.name + "CS" + std::to_string(i),
                         [&backend, i, var, this](const auto&, size_t groupIndex)
                         {
                             return backend.getDeviceVarPrefix() + var.name + m_SortedCurrentSources.at(groupIndex).at(i)->getName();
                         });
            }

            // If we're generating an initialization structure, also add any heterogeneous parameters, derived parameters or extra global parameters required for initializers
            if(init) {
                const auto *varInitSnippet = varInit.at(var.name).getSnippet();
                addHeterogeneousChildVarInitParams(varInitSnippet->getParamNames(), m_SortedCurrentSources, i, var.name, "CS",
                                                   &NeuronGroupMergedBase::isCurrentSourceVarInitParamHeterogeneous, &CurrentSourceInternal::getVarInitialisers);
                addHeterogeneousChildVarInitDerivedParams(varInitSnippet->getDerivedParams(), m_SortedCurrentSources, i, var.name, "CS",
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
            for(const auto &v : cs->getVarInitialisers()) {
                updateChildVarInitParamsHash<CurrentSourceVarAdapter>(
                    m_SortedCurrentSources, c, v.first, &NeuronGroupMergedBase::isCurrentSourceVarInitParamReferenced, hash);
                updateChildVarInitDerivedParamsHash<CurrentSourceVarAdapter>(
                    m_SortedCurrentSources, c, v.first, &NeuronGroupMergedBase::isCurrentSourceVarInitParamReferenced, hash);
            }
        }

        // Loop through child merged insyns
        for(size_t c = 0; c < getSortedArchetypeMergedInSyns().size(); c++) {
            const auto *sg = getSortedArchetypeMergedInSyns().at(c);

            // Loop through variables and update hash with variable initialisation parameters and derived parameters
            for(const auto &v : sg->getPSVarInitialisers()) {
                updateChildVarInitParamsHash<SynapsePSMVarAdapter>(
                    m_SortedMergedInSyns, c, v.first, &NeuronGroupMergedBase::isPSMVarInitParamReferenced, hash);
                updateChildVarInitDerivedParamsHash<SynapsePSMVarAdapter>(
                    m_SortedMergedInSyns, c, v.first, &NeuronGroupMergedBase::isPSMVarInitParamReferenced, hash);
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
void NeuronGroupMergedBase::addMergedInSynPointerField(const Type::NumericBase *type, const std::string &name, 
                                                       size_t archetypeIndex, const std::string &prefix)
{
    addField(type->getPointerType(), name + std::to_string(archetypeIndex),
             [prefix, archetypeIndex, this](const auto&, size_t groupIndex)
             {
                 return prefix + m_SortedMergedInSyns.at(groupIndex).at(archetypeIndex)->getFusedPSVarSuffix();
             });
}
//----------------------------------------------------------------------------
void NeuronGroupMergedBase::addMergedPreOutputOutSynPointerField(const Type::NumericBase *type, const std::string &name, 
                                                                 size_t archetypeIndex, const std::string &prefix)
{
    addField(type->getPointerType(), name + std::to_string(archetypeIndex),
             [prefix, archetypeIndex, this](const auto&, size_t groupIndex)
             {
                 return prefix + m_SortedMergedPreOutputOutSyns.at(groupIndex).at(archetypeIndex)->getFusedPreOutputSuffix();
             });
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseGroupMergedBase
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
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isSparseConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }));
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
SynapseGroupMergedBase::SynapseGroupMergedBase(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                               Role role, const std::string &archetypeCode, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups), m_ArchetypeCode(archetypeCode)
{
    using namespace Type;

    const bool updateRole = ((role == Role::PresynapticUpdate)
                             || (role == Role::PostsynapticUpdate)
                             || (role == Role::SynapseDynamics));
    const WeightUpdateModels::Base *wum = getArchetype().getWUModel();

    // If role isn't an init role or weights aren't kernel
    if(role != Role::Init || !(getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL)) {
        addField<Uint32>("rowStride",
                         [&backend](const SynapseGroupInternal &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });
        addField<Uint32>("numSrcNeurons",
                         [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
        addField<Uint32>("numTrgNeurons",
                         [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    }
    
    if(role == Role::PostsynapticUpdate || role == Role::SparseInit) {
        addField<Uint32>("colStride",
                 [](const auto &sg, size_t) { return std::to_string(sg.getMaxSourceConnections()); });
    }
    
    // If this role is one where postsynaptic input can be provided
    if(role == Role::PresynapticUpdate || role == Role::SynapseDynamics) {
        if(getArchetype().isDendriticDelayRequired()) {
            addPSPointerField(getScalarType(), "denDelay", backend.getDeviceVarPrefix() + "denDelay");
            addPSPointerField(Uint32::getInstance(), "denDelayPtr", backend.getScalarAddressPrefix() + "denDelayPtr");
        }
        else {
            addPSPointerField(getScalarType(), "inSyn", backend.getDeviceVarPrefix() + "inSyn");
        }
    }

    if(role == Role::PresynapticUpdate) {
        if(getArchetype().isTrueSpikeRequired()) {
            addSrcPointerField(Uint32::getInstance(), "srcSpkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");
            addSrcPointerField(Uint32::getInstance(), "srcSpk", backend.getDeviceVarPrefix() + "glbSpk");
        }

        if(getArchetype().isSpikeEventRequired()) {
            addSrcPointerField(Uint32::getInstance(), "srcSpkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
            addSrcPointerField(Uint32::getInstance(), "srcSpkEvnt", backend.getDeviceVarPrefix() + "glbSpkEvnt");
        }
    }
    else if(role == Role::PostsynapticUpdate) {
        addTrgPointerField(Uint32::getInstance(), "trgSpkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");
        addTrgPointerField(Uint32::getInstance(), "trgSpk", backend.getDeviceVarPrefix() + "glbSpk");
    }

    // If this structure is used for updating rather than initializing
    if(updateRole) {
        // for all types of roles
        if (getArchetype().isPresynapticOutputRequired()) {
            addPreOutputPointerField(getScalarType(), "revInSyn", backend.getDeviceVarPrefix() + "revInSyn");
        }

        // If presynaptic population has delay buffers
        if(getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
            addSrcPointerField(Uint32::getInstance(), "srcSpkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
        }

        // If postsynaptic population has delay buffers
        if(getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
            addTrgPointerField(Uint32::getInstance(), "trgSpkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
        }

        // Add heterogeneous presynaptic neuron model parameters
        addHeterogeneousParams<SynapseGroupMergedBase>(
            getArchetype().getSrcNeuronGroup()->getNeuronModel()->getParamNames(), "Pre",
            [](const auto &sg) { return sg.getSrcNeuronGroup()->getParams(); },
            &SynapseGroupMergedBase::isSrcNeuronParamHeterogeneous);

        // Add heterogeneous presynaptic neuron model derived parameters
        addHeterogeneousDerivedParams<SynapseGroupMergedBase>(
            getArchetype().getSrcNeuronGroup()->getNeuronModel()->getDerivedParams(), "Pre",
            [](const auto &sg) { return sg.getSrcNeuronGroup()->getDerivedParams(); },
            &SynapseGroupMergedBase::isSrcNeuronDerivedParamHeterogeneous);

        // Add heterogeneous postsynaptic neuron model parameters
        addHeterogeneousParams<SynapseGroupMergedBase>(
            getArchetype().getTrgNeuronGroup()->getNeuronModel()->getParamNames(), "Post",
            [](const auto &sg) { return sg.getTrgNeuronGroup()->getParams(); },
            &SynapseGroupMergedBase::isTrgNeuronParamHeterogeneous);

        // Add heterogeneous postsynaptic neuron model derived parameters
        addHeterogeneousDerivedParams<SynapseGroupMergedBase>(
            getArchetype().getTrgNeuronGroup()->getNeuronModel()->getDerivedParams(), "Post",
            [](const auto &sg) { return sg.getTrgNeuronGroup()->getDerivedParams(); },
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
                const std::string prefix = backend.getDeviceVarPrefix();
                addField(parseNumericPtr(e.type), e.name + "Pre",
                         [e, prefix](const auto &sg, size_t) { return prefix + e.name + sg.getSrcNeuronGroup()->getName(); },
                         GroupMergedFieldType::DYNAMIC);
            }
        }

        // Loop through extra global parameters in postsynaptic neuron model
        const auto postEGPs = getArchetype().getTrgNeuronGroup()->getNeuronModel()->getExtraGlobalParams();
        for(const auto &e : postEGPs) {
            if(code.find("$(" + e.name + "_post)") != std::string::npos) {
                const std::string prefix = backend.getDeviceVarPrefix();
                addField(parseNumericPtr(e.type), e.name + "Post",
                         [e, prefix](const auto &sg, size_t) { return prefix + e.name + sg.getTrgNeuronGroup()->getName(); },
                         GroupMergedFieldType::DYNAMIC);
            }
        }

        // Add spike times if required
        if(wum->isPreSpikeTimeRequired()) {
            addSrcPointerField(getTimeType(), "sTPre", backend.getDeviceVarPrefix() + "sT");
        }
        if(wum->isPostSpikeTimeRequired()) {
            addTrgPointerField(getTimeType(), "sTPost", backend.getDeviceVarPrefix() + "sT");
        }
        if(wum->isPreSpikeEventTimeRequired()) {
            addSrcPointerField(getTimeType(), "seTPre", backend.getDeviceVarPrefix() + "seT");
        }
        if(wum->isPrevPreSpikeTimeRequired()) {
            addSrcPointerField(getTimeType(), "prevSTPre", backend.getDeviceVarPrefix() + "prevST");
        }
        if(wum->isPrevPostSpikeTimeRequired()) {
            addTrgPointerField(getTimeType(), "prevSTPost", backend.getDeviceVarPrefix() + "prevST");
        }
        if(wum->isPrevPreSpikeEventTimeRequired()) {
            addSrcPointerField(getTimeType(), "prevSETPre", backend.getDeviceVarPrefix() + "prevSET");
        }
        // Add heterogeneous weight update model parameters
        addHeterogeneousParams<SynapseGroupMergedBase>(
            wum->getParamNames(), "",
            [](const auto &sg) { return sg.getWUParams(); },
            &SynapseGroupMergedBase::isWUParamHeterogeneous);

        // Add heterogeneous weight update model derived parameters
        addHeterogeneousDerivedParams<SynapseGroupMergedBase>(
            wum->getDerivedParams(), "",
            [](const auto &sg) { return sg.getWUDerivedParams(); },
            &SynapseGroupMergedBase::isWUDerivedParamHeterogeneous);

        // Add presynaptic variables to struct
        for(const auto &v : wum->getPreVars()) {
            const std::string prefix = backend.getDeviceVarPrefix() + v.name;
            addField(v.type->getPointerType(), v.name, 
                     [prefix](const auto &g, size_t) { return prefix + g.getFusedWUPreVarSuffix(); });
        }
        
        // Add presynaptic variables to struct
        for(const auto &v : wum->getPostVars()) {
            const std::string prefix = backend.getDeviceVarPrefix() + v.name;
            addField(v.type->getPointerType(), v.name, 
                     [prefix](const auto &g, size_t) { return prefix + g.getFusedWUPostVarSuffix(); });
        }

        // Add EGPs to struct
        addEGPs(wum->getExtraGlobalParams(), backend.getDeviceVarPrefix());
    }

    // Add pointers to connectivity data
    if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        addPointerField<Uint32>("rowLength", backend.getDeviceVarPrefix() + "rowLength");
        addPointerField(getArchetype().getSparseIndType(), "ind", backend.getDeviceVarPrefix() + "ind");

        // Add additional structure for postsynaptic access
        if(backend.isPostsynapticRemapRequired() && !wum->getLearnPostCode().empty()
           && (role == Role::PostsynapticUpdate || role == Role::SparseInit))
        {
            addPointerField<Uint32>("colLength", backend.getDeviceVarPrefix() + "colLength");
            addPointerField<Uint32>("remap", backend.getDeviceVarPrefix() + "remap");
        }
    }
    else if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        addPointerField<Uint32>("gp", backend.getDeviceVarPrefix() + "gp");
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
                                       return sg.getWUConstInitVals().at(var.name);
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
                    addField<Uint32>("kernelSize" + std::to_string(d),
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
        for(const auto &var : wum->getVars()) {
            // Variable initialisation is required if we're performing connectivity init and var init snippet requires a kernel or
            // We're performing some other sort of initialisation, the snippet DOESN'T require a kernel but has SOME code
            const auto *snippet = varInit.at(var.name).getSnippet();
            const bool varInitRequired = ((connectInitRole && snippet->requiresKernel()) 
                                          || (varInitRole && individualWeights && !snippet->requiresKernel() && !snippet->getCode().empty())
                                          || (varInitRole && kernelWeights && !snippet->getCode().empty()));

            // If we're performing an update with individual weights; or this variable should be initialised
            if((updateRole && individualWeights) || (kernelWeights && updateRole) || varInitRequired) {
                addPointerField(var.type, var.name, backend.getDeviceVarPrefix() + var.name);
            }

            // If we're performing a procedural update or this variable should be initialised, add any var init EGPs to structure
            if((proceduralWeights && updateRole) || varInitRequired) {
                const auto egps = snippet->getExtraGlobalParams();
                for(const auto &e : egps) {
                    const std::string prefix = backend.getDeviceVarPrefix();
                    addField(parseNumericPtr(e.type), e.name + var.name,
                             [e, prefix, var](const SynapseGroupInternal &sg, size_t)
                             {
                                 return prefix + e.name + var.name + sg.getName();
                             },
                             GroupMergedFieldType::DYNAMIC);
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
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); }, hash);

        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSparseConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }, hash);
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
                &SynapseGroupMergedBase::isWUVarInitParamReferenced, hash);
        }
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addPSPointerField(const Type::NumericBase *type, const std::string &name, const std::string &prefix)
{
    addField(type->getPointerType(), name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getFusedPSVarSuffix(); });
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addPreOutputPointerField(const Type::NumericBase *type, const std::string &name, const std::string &prefix)
{
    addField(type->getPointerType(), name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getFusedPreOutputSuffix(); });
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addSrcPointerField(const Type::NumericBase *type, const std::string &name, const std::string &prefix)
{
    addField(type->getPointerType(), name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getSrcNeuronGroup()->getName(); });
}
//----------------------------------------------------------------------------
void SynapseGroupMergedBase::addTrgPointerField(const Type::NumericBase *type, const std::string &name, const std::string &prefix)
{
    addField(type->getPointerType(), name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getTrgNeuronGroup()->getName(); });
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
    const auto *snippet = getArchetype().getConnectivityInitialiser().getSnippet();
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

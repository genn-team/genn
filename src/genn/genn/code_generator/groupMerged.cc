#include "code_generator/groupMerged.h"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"

//----------------------------------------------------------------------------
// CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::NeuronSpikeQueueUpdateGroupMerged::name = "NeuronSpikeQueueUpdate";
//----------------------------------------------------------------------------
CodeGenerator::NeuronSpikeQueueUpdateGroupMerged::NeuronSpikeQueueUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                                                                    const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   GroupMerged<NeuronGroupInternal>(index, precision, groups)
{
    if(getArchetype().isDelayRequired()) {
        addField("unsigned int", "numDelaySlots",
                 [](const NeuronGroupInternal &ng, size_t) { return std::to_string(ng.getNumDelaySlots()); });

        addPointerField("unsigned int", "spkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
    } 

    addPointerField("unsigned int", "spkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField("unsigned int", "spkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
    }

    if(getArchetype().isPrevSpikeTimeRequired() || getArchetype().isPrevSpikeEventTimeRequired()) {
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
}
//----------------------------------------------------------------------------
void CodeGenerator::NeuronSpikeQueueUpdateGroupMerged::genMergedGroupSpikeCountReset(CodeStream &os, unsigned int batchSize) const
{
    if(getArchetype().isDelayRequired()) { // with delay
        assert(batchSize == 1);

        if(getArchetype().isSpikeEventRequired()) {
            os << "group->spkCntEvnt[*group->spkQuePtr] = 0;" << std::endl;
        }
        if(getArchetype().isTrueSpikeRequired()) {
            os << "group->spkCnt[*group->spkQuePtr] = 0;" << std::endl;
        }
        else {
            os << "group->spkCnt[0] = 0;" << std::endl;
        }
    }
    else { // no delay
        if(getArchetype().isSpikeEventRequired()) {
            os << "group->spkCntEvnt[" << ((batchSize > 1) ? "batch" : "0") << "] = 0;" << std::endl;
        }
        os << "group->spkCnt[" << ((batchSize > 1) ? "batch" : "0") << "] = 0;" << std::endl;
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::NeuronGroupMergedBase
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const NeuronGroupInternal &ng) { return ng.getParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isDerivedParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const NeuronGroupInternal &ng) { return ng.getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isVarInitParamHeterogeneous(size_t varIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *varInitSnippet = getArchetype().getVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isParamValueHeterogeneous({varInitSnippet->getCode()}, paramName, paramIndex,
                                     [varIndex](const NeuronGroupInternal &sg)
                                     {
                                         return sg.getVarInitialisers().at(varIndex).getParams();
                                     });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isVarInitDerivedParamHeterogeneous(size_t varIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *varInitSnippet = getArchetype().getVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamValueHeterogeneous({varInitSnippet->getCode()}, derivedParamName, paramIndex,
                                     [varIndex](const NeuronGroupInternal &sg)
                                     {
                                         return sg.getVarInitialisers().at(varIndex).getDerivedParams();
                                     });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isCurrentSourceParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *csm = getArchetype().getCurrentSources().at(childIndex)->getCurrentSourceModel();
    const std::string paramName = csm->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({csm->getInjectionCode()}, paramName, childIndex, paramIndex, m_SortedCurrentSources,
                                          [](const CurrentSourceInternal *cs) { return cs->getParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isCurrentSourceDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If derived parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *csm = getArchetype().getCurrentSources().at(childIndex)->getCurrentSourceModel();
    const std::string derivedParamName = csm->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({csm->getInjectionCode()}, derivedParamName, childIndex, paramIndex, m_SortedCurrentSources,
                                          [](const CurrentSourceInternal *cs) { return cs->getDerivedParams(); });
 
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isCurrentSourceVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getCurrentSources().at(childIndex)->getVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({varInitSnippet->getCode()}, paramName, childIndex, paramIndex, m_SortedCurrentSources,
                                          [varIndex](const CurrentSourceInternal *cs) { return cs->getVarInitialisers().at(varIndex).getParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isCurrentSourceVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getCurrentSources().at(childIndex)->getVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({varInitSnippet->getCode()}, derivedParamName, childIndex, paramIndex, m_SortedCurrentSources,
                                          [varIndex](const CurrentSourceInternal *cs) { return cs->getVarInitialisers().at(varIndex).getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isPSMParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{  
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *psm = getArchetype().getMergedInSyn().at(childIndex)->getPSModel();
    const std::string paramName = psm->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({psm->getApplyInputCode(), psm->getDecayCode()}, paramName, childIndex, paramIndex, m_SortedMergedInSyns,
                                          [](const SynapseGroupInternal *inSyn)
                                          {
                                              return inSyn->getPSParams();
                                          });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isPSMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *psm = getArchetype().getMergedInSyn().at(childIndex)->getPSModel();
    const std::string derivedParamName = psm->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({psm->getApplyInputCode(), psm->getDecayCode()}, derivedParamName, childIndex, paramIndex, m_SortedMergedInSyns,
                                          [](const SynapseGroupInternal *inSyn)
                                          {
                                              return inSyn->getPSDerivedParams();
                                          });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isPSMGlobalVarHeterogeneous(size_t childIndex, size_t varIndex) const
{
    // If synapse group doesn't have individual PSM variables to start with, return false
    const auto *sg = getArchetype().getMergedInSyn().at(childIndex);
    if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
        return false;
    }
    else {
        const auto *psm = sg->getPSModel();
        const std::string varName = psm->getVars().at(varIndex).name;
        return isChildParamValueHeterogeneous({psm->getApplyInputCode(), psm->getDecayCode()}, varName, childIndex, varIndex, m_SortedMergedInSyns,
                                              [](const SynapseGroupInternal *inSyn)
                                              {
                                                  return inSyn->getPSConstInitVals();
                                              });
    }
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isPSMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getMergedInSyn().at(childIndex)->getPSVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({varInitSnippet->getCode()}, paramName, childIndex, paramIndex, m_SortedMergedInSyns,
                                          [varIndex](const SynapseGroupInternal *inSyn)
                                          { 
                                              return inSyn->getPSVarInitialisers().at(varIndex).getParams();
                                          });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isPSMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getMergedInSyn().at(childIndex)->getPSVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({varInitSnippet->getCode()}, derivedParamName, childIndex, paramIndex, m_SortedMergedInSyns,
                                          [varIndex](const SynapseGroupInternal *inSyn)
                                          { 
                                              return inSyn->getPSVarInitialisers().at(varIndex).getDerivedParams();
                                          });
}
//----------------------------------------------------------------------------
CodeGenerator::NeuronGroupMergedBase::NeuronGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                                            bool init, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   CodeGenerator::GroupMerged<NeuronGroupInternal>(index, precision, groups)
{
    // Build vector of vectors containing each child group's merged in syns, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedMergedInSyns, &NeuronGroupInternal::getMergedInSyn,
                             [init](const SynapseGroupInternal *a, const SynapseGroupInternal *b)
                             {
                                 return init ? a->canPSInitBeMerged(*b) : a->canPSBeMerged(*b);
                             });

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedCurrentSources, &NeuronGroupInternal::getCurrentSources,
                             [init](const CurrentSourceInternal *a, const CurrentSourceInternal *b)
                             {
                                 return init ? a->canInitBeMerged(*b) : a->canBeMerged(*b);
                             });

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

    // Loop through merged synaptic inputs in archetypical neuron group
    for(size_t i = 0; i < getArchetype().getMergedInSyn().size(); i++) {
        const SynapseGroupInternal *sg = getArchetype().getMergedInSyn()[i];

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
                    auto getVarInitialiserFn = [this](size_t groupIndex, size_t childIndex)
                                               {
                                                   return m_SortedMergedInSyns.at(groupIndex).at(childIndex)->getPSVarInitialisers();
                                               };
                    addHeterogeneousChildVarInitParams(varInitSnippet->getParamNames(), i, v, var.name + "InSyn",
                                                       &NeuronGroupMergedBase::isPSMVarInitParamHeterogeneous, getVarInitialiserFn);
                    addHeterogeneousChildVarInitDerivedParams(varInitSnippet->getDerivedParams(), i, v, var.name + "InSyn",
                                                              &NeuronGroupMergedBase::isPSMVarInitDerivedParamHeterogeneous, getVarInitialiserFn);
                    addChildEGPs(varInitSnippet->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), var.name + "InSyn",
                                 [var, this](size_t groupIndex, size_t childIndex)
                                 {
                                     return var.name + m_SortedMergedInSyns.at(groupIndex).at(childIndex)->getPSModelTargetName();
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
            addHeterogeneousChildParams(paramNames, i, "InSyn", &NeuronGroupMergedBase::isPSMParamHeterogeneous,
                                        [this](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                        {
                                            return m_SortedMergedInSyns.at(groupIndex).at(childIndex)->getPSParams().at(paramIndex);
                                        });

            // Add any heterogeneous postsynaptic mode derived parameters
            const auto derivedParams = sg->getPSModel()->getDerivedParams();
            addHeterogeneousChildDerivedParams(derivedParams, i, "InSyn", &NeuronGroupMergedBase::isPSMDerivedParamHeterogeneous,
                                               [this](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                               {
                                                    return m_SortedMergedInSyns.at(groupIndex).at(childIndex)->getPSDerivedParams().at(paramIndex);
                                               });
            // Add EGPs
            addChildEGPs(sg->getPSModel()->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), "InSyn",
                         [this](size_t groupIndex, size_t childIndex)
                         {
                             return m_SortedMergedInSyns.at(groupIndex).at(childIndex)->getPSModelTargetName();
                         });
        }
    }

    // Loop through current sources in archetypical neuron group
    for(size_t i = 0; i < getArchetype().getCurrentSources().size(); i++) {
        const auto *cs = getArchetype().getCurrentSources()[i];

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
                auto getVarInitialiserFn = [this](size_t groupIndex, size_t childIndex)
                {
                    return m_SortedCurrentSources.at(groupIndex).at(childIndex)->getVarInitialisers();
                };
                addHeterogeneousChildVarInitParams(varInitSnippet->getParamNames(), i, v, var.name + "CS",
                                                   &NeuronGroupMergedBase::isCurrentSourceVarInitParamHeterogeneous, getVarInitialiserFn);
                addHeterogeneousChildVarInitDerivedParams(varInitSnippet->getDerivedParams(), i, v, var.name + "CS",
                                                          &NeuronGroupMergedBase::isCurrentSourceVarInitDerivedParamHeterogeneous, getVarInitialiserFn);
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
            addHeterogeneousChildParams(paramNames, i, "CS", &NeuronGroupMergedBase::isCurrentSourceParamHeterogeneous,
                                        [this](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                        {
                                            return m_SortedCurrentSources.at(groupIndex).at(childIndex)->getParams().at(paramIndex);
                                        });

            // Add any heterogeneous current source derived parameters
            const auto derivedParams = cs->getCurrentSourceModel()->getDerivedParams();
            addHeterogeneousChildDerivedParams(derivedParams, i, "CS", &NeuronGroupMergedBase::isCurrentSourceDerivedParamHeterogeneous,
                                               [this](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                                {
                                                    return m_SortedCurrentSources.at(groupIndex).at(childIndex)->getDerivedParams().at(paramIndex);
                                                });

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
void CodeGenerator::NeuronGroupMergedBase::addMergedInSynPointerField(const std::string &type, const std::string &name, 
                                                                      size_t archetypeIndex, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name + std::to_string(archetypeIndex),
             [prefix, archetypeIndex, this](const NeuronGroupInternal &, size_t groupIndex)
             {
                 return prefix + m_SortedMergedInSyns.at(groupIndex).at(archetypeIndex)->getPSModelTargetName();
             });
}

//----------------------------------------------------------------------------
// CodeGenerator::NeuronUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::NeuronUpdateGroupMerged::name = "NeuronUpdate";
//----------------------------------------------------------------------------
CodeGenerator::NeuronUpdateGroupMerged::NeuronUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                                                const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, precision, timePrecision, backend, false, groups)
{
    // Build vector of vectors containing each child group's incoming synapse groups
    // with postsynaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(getArchetype().getInSynWithPostCode(), m_SortedInSynWithPostCode, &NeuronGroupInternal::getInSynWithPostCode,
                             [](const SynapseGroupInternal *a, const SynapseGroupInternal *b){ return a->canWUPostBeMerged(*b); });

    // Build vector of vectors containing each child group's outgoing synapse groups
    // with presynaptic synaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(getArchetype().getOutSynWithPreCode(), m_SortedOutSynWithPreCode, &NeuronGroupInternal::getOutSynWithPreCode,
                             [](const SynapseGroupInternal *a, const SynapseGroupInternal *b){ return a->canWUPreBeMerged(*b); });

    // Generate struct fields for incoming synapse groups with postsynaptic update code
    const auto inSynWithPostCode = getArchetype().getInSynWithPostCode();
    generateWUVar(backend, "WUPost", inSynWithPostCode, m_SortedInSynWithPostCode,
                  &WeightUpdateModels::Base::getPostVars, &NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous,
                  &NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous);

    // Generate struct fields for outgoing synapse groups with presynaptic update code
    const auto outSynWithPreCode = getArchetype().getOutSynWithPreCode();
    generateWUVar(backend, "WUPre", outSynWithPreCode, m_SortedOutSynWithPreCode,
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
bool CodeGenerator::NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *wum = getArchetype().getInSynWithPostCode().at(childIndex)->getWUModel();
    const std::string paramName = wum->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({wum->getPostSpikeCode(), wum->getPostDynamicsCode()}, paramName, childIndex, paramIndex, m_SortedInSynWithPostCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If derived parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *wum = getArchetype().getInSynWithPostCode().at(childIndex)->getWUModel();
    const std::string derivedParamName = wum->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({wum->getPostSpikeCode(), wum->getPostDynamicsCode()}, derivedParamName, childIndex, paramIndex, m_SortedInSynWithPostCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUDerivedParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *wum = getArchetype().getOutSynWithPreCode().at(childIndex)->getWUModel();
    const std::string paramName = wum->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({wum->getPreSpikeCode(), wum->getPreDynamicsCode()}, paramName, childIndex, paramIndex, m_SortedOutSynWithPreCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If derived parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *wum = getArchetype().getOutSynWithPreCode().at(childIndex)->getWUModel();
    const std::string derivedParamName = wum->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({wum->getPreSpikeCode(), wum->getPreDynamicsCode()}, derivedParamName, childIndex, paramIndex, m_SortedOutSynWithPreCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUDerivedParams(); });
}
 std::string CodeGenerator::NeuronUpdateGroupMerged::getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    return ((varDuplication & VarAccessDuplication::SHARED || batchSize == 1) ? "" : "batchOffset + ") + index;
}
//--------------------------------------------------------------------------
std::string CodeGenerator::NeuronUpdateGroupMerged::getReadVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    if(delay) {
        return ((varDuplication & VarAccessDuplication::SHARED || batchSize == 1) ? "readDelayOffset + " : "readBatchDelayOffset + ") + index;
    }
    else {
        return getVarIndex(batchSize, varDuplication, index);
    }
}
//--------------------------------------------------------------------------
std::string CodeGenerator::NeuronUpdateGroupMerged::getWriteVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    if(delay) {
        return ((varDuplication & VarAccessDuplication::SHARED || batchSize == 1) ? "writeDelayOffset + " : "writeBatchDelayOffset + ") + index;
    }
    else {
        return getVarIndex(batchSize, varDuplication, index);
    }
}
//----------------------------------------------------------------------------
void CodeGenerator::NeuronUpdateGroupMerged::generateWUVar(const BackendBase &backend,  const std::string &fieldPrefixStem, 
                                                           const std::vector<SynapseGroupInternal *> &archetypeSyn,
                                                           const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                                                           Models::Base::VarVec (WeightUpdateModels::Base::*getVars)(void) const,
                                                           bool(NeuronUpdateGroupMerged::*isParamHeterogeneous)(size_t, size_t) const,
                                                           bool(NeuronUpdateGroupMerged::*isDerivedParamHeterogeneous)(size_t, size_t) const)
{
    // Loop through synapse groups
    for(size_t i = 0; i < archetypeSyn.size(); i++) {
        const auto *sg = archetypeSyn[i];

        // Loop through variables
        const auto vars = (sg->getWUModel()->*getVars)();
        for(size_t v = 0; v < vars.size(); v++) {
            // Add pointers to state variable
            const auto var = vars[v];
            assert(!Utils::isTypePointer(var.type));
            addField(var.type + "*", var.name + fieldPrefixStem + std::to_string(i),
                     [i, var, &backend, &sortedSyn](const NeuronGroupInternal &, size_t groupIndex)
                     {
                         return backend.getDeviceVarPrefix() + var.name + sortedSyn.at(groupIndex).at(i)->getName();
                     });
        }

        // Add any heterogeneous parameters
        addHeterogeneousChildParams<NeuronUpdateGroupMerged>(sg->getWUModel()->getParamNames(), i, fieldPrefixStem, isParamHeterogeneous,
                                                             [&sortedSyn](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                                             {
                                                                 return sortedSyn.at(groupIndex).at(childIndex)->getWUParams().at(paramIndex);
                                                             });

        // Add any heterogeneous derived parameters
        addHeterogeneousChildDerivedParams<NeuronUpdateGroupMerged>(sg->getWUModel()->getDerivedParams(), i, fieldPrefixStem, isDerivedParamHeterogeneous,
                                                                    [&sortedSyn](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                                                    {
                                                                        return sortedSyn.at(groupIndex).at(childIndex)->getWUDerivedParams().at(paramIndex);
                                                                    });

        // Add EGPs
        addChildEGPs(sg->getWUModel()->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), fieldPrefixStem,
                     [&sortedSyn](size_t groupIndex, size_t childIndex)
                     {
                         return sortedSyn.at(groupIndex).at(childIndex)->getName();
                     });
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::NeuronInitGroupMerged::name = "NeuronInit";
//----------------------------------------------------------------------------
CodeGenerator::NeuronInitGroupMerged::NeuronInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                                            const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, precision, timePrecision, backend, true, groups)
{
    // Build vector of vectors containing each child group's incoming 
    // synapse groups, ordered to match those of the archetype group
    orderNeuronGroupChildren(getArchetype().getInSynWithPostVars(), m_SortedInSynWithPostVars, &NeuronGroupInternal::getInSynWithPostVars,
                             [](const SynapseGroupInternal *a, const SynapseGroupInternal *b) { return a->canWUPostInitBeMerged(*b); });

    // Build vector of vectors containing each child group's outgoing 
    // synapse groups, ordered to match those of the archetype group
    orderNeuronGroupChildren(getArchetype().getOutSynWithPreVars(), m_SortedOutSynWithPreVars, &NeuronGroupInternal::getOutSynWithPreVars,
                             [](const SynapseGroupInternal *a, const SynapseGroupInternal *b){ return a->canWUPreInitBeMerged(*b); });

    // Generate struct fields for incoming synapse groups with postsynaptic variables
    const auto inSynWithPostVars = getArchetype().getInSynWithPostVars();
    generateWUVar(backend, "WUPost", inSynWithPostVars, m_SortedInSynWithPostVars,
                  &WeightUpdateModels::Base::getPostVars, &SynapseGroupInternal::getWUPostVarInitialisers,
                  &NeuronInitGroupMerged::isInSynWUMVarInitParamHeterogeneous,
                  &NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamHeterogeneous);


    // Generate struct fields for outgoing synapse groups
    const auto outSynWithPreVars = getArchetype().getOutSynWithPreVars();
    generateWUVar(backend, "WUPre", outSynWithPreVars, m_SortedOutSynWithPreVars,
                  &WeightUpdateModels::Base::getPreVars, &SynapseGroupInternal::getWUPreVarInitialisers,
                  &NeuronInitGroupMerged::isOutSynWUMVarInitParamHeterogeneous,
                  &NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamHeterogeneous);
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronInitGroupMerged::isInSynWUMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getInSynWithPostVars().at(childIndex)->getWUPostVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({varInitSnippet->getCode()}, paramName, childIndex, paramIndex, m_SortedInSynWithPostVars,
                                          [varIndex](const SynapseGroupInternal *s) { return s->getWUPostVarInitialisers().at(varIndex).getParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getInSynWithPostVars().at(childIndex)->getWUPostVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({varInitSnippet->getCode()}, derivedParamName, childIndex, paramIndex, m_SortedInSynWithPostVars,
                                          [varIndex](const SynapseGroupInternal *s) { return s->getWUPostVarInitialisers().at(varIndex).getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronInitGroupMerged::isOutSynWUMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getOutSynWithPreVars().at(childIndex)->getWUPreVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({varInitSnippet->getCode()}, paramName, childIndex, paramIndex, m_SortedOutSynWithPreVars,
                                          [varIndex](const SynapseGroupInternal *s) { return s->getWUPreVarInitialisers().at(varIndex).getParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getOutSynWithPreVars().at(childIndex)->getWUPreVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({varInitSnippet->getCode()}, derivedParamName, childIndex, paramIndex, m_SortedOutSynWithPreVars,
                                          [varIndex](const SynapseGroupInternal *s) { return s->getWUPreVarInitialisers().at(varIndex).getDerivedParams(); });
}
//----------------------------------------------------------------------------
void CodeGenerator::NeuronInitGroupMerged::generateWUVar(const BackendBase &backend,
                                                         const std::string &fieldPrefixStem,
                                                         const std::vector<SynapseGroupInternal *> &archetypeSyn,
                                                         const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                                                         Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                                                         const std::vector<Models::VarInit> &(SynapseGroupInternal:: *getVarInitialisers)(void) const,
                                                         bool(NeuronInitGroupMerged::*isParamHeterogeneous)(size_t, size_t, size_t) const,
                                                         bool(NeuronInitGroupMerged::*isDerivedParamHeterogeneous)(size_t, size_t, size_t) const)
{
    // Loop through synapse groups
    for(size_t i = 0; i < archetypeSyn.size(); i++) {
        const auto *sg = archetypeSyn.at(i);

        // Loop through variables
        const auto vars = (sg->getWUModel()->*getVars)();
        const auto &varInit = (sg->*getVarInitialisers)();
        for(size_t v = 0; v < vars.size(); v++) {
            // Add pointers to state variable
            const auto var = vars.at(v);
            if(!varInit.at(v).getSnippet()->getCode().empty()) {
                assert(!Utils::isTypePointer(var.type));
                addField(var.type + "*", var.name + fieldPrefixStem + std::to_string(i),
                         [i, var, &backend, &sortedSyn](const NeuronGroupInternal &, size_t groupIndex)
                         {
                             return backend.getDeviceVarPrefix() + var.name + sortedSyn.at(groupIndex).at(i)->getName();
                         });
            }

            // Also add any heterogeneous, derived or extra global parameters required for initializers
            const auto *varInitSnippet = varInit.at(v).getSnippet();
            auto getVarInitialiserFn = [&sortedSyn](size_t groupIndex, size_t childIndex)
                                       {
                                           return sortedSyn.at(groupIndex).at(childIndex)->getWUPreVarInitialisers();
                                       };
            addHeterogeneousChildVarInitParams<NeuronInitGroupMerged>(varInitSnippet->getParamNames(), i, v, var.name + fieldPrefixStem,
                                                                      isParamHeterogeneous, getVarInitialiserFn);
            addHeterogeneousChildVarInitDerivedParams<NeuronInitGroupMerged>(varInitSnippet->getDerivedParams(), i, v, var.name + fieldPrefixStem,
                                                                             isDerivedParamHeterogeneous, getVarInitialiserFn);
            addChildEGPs(varInitSnippet->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), var.name + fieldPrefixStem,
                         [var, &sortedSyn](size_t groupIndex, size_t childIndex)
                         {
                             return var.name + sortedSyn.at(groupIndex).at(childIndex)->getName();
                         });
        }
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::SynapseDendriticDelayUpdateGroupMerged::name = "SynapseDendriticDelayUpdate";
//----------------------------------------------------------------------------
CodeGenerator::SynapseDendriticDelayUpdateGroupMerged::SynapseDendriticDelayUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    : GroupMerged<SynapseGroupInternal>(index, precision, groups)
{
    addField("unsigned int*", "denDelayPtr", 
             [&backend](const SynapseGroupInternal &sg, size_t) 
             {
                 return backend.getScalarAddressPrefix() + "denDelayPtr" + sg.getPSModelTargetName(); 
             });
}

// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::SynapseConnectivityHostInitGroupMerged::name = "SynapseConnectivityHostInit";
//------------------------------------------------------------------------
CodeGenerator::SynapseConnectivityHostInitGroupMerged::SynapseConnectivityHostInitGroupMerged(size_t index, const std::string &precision, const std::string&, const BackendBase &backend,
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
bool CodeGenerator::SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous(size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *connectInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const std::string paramName = connectInitSnippet->getParamNames().at(paramIndex);
    return isParamValueHeterogeneous({connectInitSnippet->getHostInitCode()}, paramName, paramIndex,
                                     [](const SynapseGroupInternal &sg)
                                     {
                                         return sg.getConnectivityInitialiser().getParams();
                                     });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *connectInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const std::string paramName = connectInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamValueHeterogeneous({connectInitSnippet->getHostInitCode()}, paramName, paramIndex,
                                     [](const SynapseGroupInternal &sg)
                                     {
                                         return sg.getConnectivityInitialiser().getDerivedParams();
                                     });
}

//----------------------------------------------------------------------------
// CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPresynapticAxonalDelaySlot() const
{
    assert(getArchetype().getSrcNeuronGroup()->isDelayRequired());

    const unsigned int numDelaySteps = getArchetype().getDelaySteps();
    if(numDelaySteps == 0) {
        return "(*group->srcSpkQuePtr)";
    }
    else {
        const unsigned int numSrcDelaySlots = getArchetype().getSrcNeuronGroup()->getNumDelaySlots();
        return "((*group->srcSpkQuePtr + " + std::to_string(numSrcDelaySlots - numDelaySteps) + ") % " + std::to_string(numSrcDelaySlots) + ")";
    }
}
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPrevPresynapticSpikeTimeAxonalDelaySlot() const
{
    // Always read from previous delay slot
    assert(getArchetype().getSrcNeuronGroup()->isDelayRequired());

    const unsigned int numDelaySteps = getArchetype().getDelaySteps();
    const unsigned int numSrcDelaySlots = getArchetype().getSrcNeuronGroup()->getNumDelaySlots();
    return "((*group->srcSpkQuePtr + " + std::to_string(numSrcDelaySlots - numDelaySteps - 1) + ") % " + std::to_string(numSrcDelaySlots) + ")";
}
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPostsynapticBackPropDelaySlot() const
{
    assert(getArchetype().getTrgNeuronGroup()->isDelayRequired());

    const unsigned int numBackPropDelaySteps = getArchetype().getBackPropDelaySteps();
    if(numBackPropDelaySteps == 0) {
        return "(*group->trgSpkQuePtr)";
    }
    else {
        const unsigned int numTrgDelaySlots = getArchetype().getTrgNeuronGroup()->getNumDelaySlots();
        return "((*group->trgSpkQuePtr + " + std::to_string(numTrgDelaySlots - numBackPropDelaySteps) + ") % " + std::to_string(numTrgDelaySlots) + ")";
    }
}
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPrevPostsynapticSpikeTimeBackPropDelaySlot() const
{
    // Always read from previous delay slot
    assert(getArchetype().getTrgNeuronGroup()->isDelayRequired());

    const unsigned int numBackPropDelaySteps = getArchetype().getBackPropDelaySteps();
    const unsigned int numTrgDelaySlots = getArchetype().getTrgNeuronGroup()->getNumDelaySlots();
    return "((*group->trgSpkQuePtr + " + std::to_string(numTrgDelaySlots - numBackPropDelaySteps - 1) + ") % " + std::to_string(numTrgDelaySlots) + ")";
}
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getDendriticDelayOffset(const std::string &offset) const
{
    assert(getArchetype().isDendriticDelayRequired());

    if(offset.empty()) {
        return "(*group->denDelayPtr * group->numTrgNeurons) + ";
    }
    else {
        return "(((*group->denDelayPtr + " + offset + ") % " + std::to_string(getArchetype().getMaxDendriticDelayTimesteps()) + ") * group->numTrgNeurons) + ";
    }
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isWUParamHeterogeneous(size_t paramIndex) const
{
    const auto *wum = getArchetype().getWUModel();
    const std::string paramName = wum->getParamNames().at(paramIndex);
    return isParamValueHeterogeneous({getArchetypeCode()}, paramName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getWUParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isWUDerivedParamHeterogeneous(size_t paramIndex) const
{
    const auto *wum = getArchetype().getWUModel();
    const std::string derivedParamName = wum->getDerivedParams().at(paramIndex).name;
    return isParamValueHeterogeneous({getArchetypeCode()}, derivedParamName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isWUGlobalVarHeterogeneous(size_t varIndex) const
{
    // If synapse group has global WU variables
    if(getArchetype().getMatrixType() & SynapseMatrixWeight::GLOBAL) {
        const auto *wum = getArchetype().getWUModel();
        const std::string varName = wum->getVars().at(varIndex).name;
        return isParamValueHeterogeneous({getArchetypeCode()}, varName, varIndex,
                                         [](const SynapseGroupInternal &sg) { return sg.getWUConstInitVals(); });
    }
    // Otherwise, return false
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isWUVarInitParamHeterogeneous(size_t varIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *varInitSnippet = getArchetype().getWUVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isParamValueHeterogeneous({varInitSnippet->getCode()}, paramName, paramIndex,
                                     [varIndex](const SynapseGroupInternal &sg)
                                     {
                                         return sg.getWUVarInitialisers().at(varIndex).getParams();
                                     });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isWUVarInitDerivedParamHeterogeneous(size_t varIndex, size_t paramIndex) const
{
    // If derived parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *varInitSnippet = getArchetype().getWUVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamValueHeterogeneous({varInitSnippet->getCode()}, derivedParamName, paramIndex,
                                     [varIndex](const SynapseGroupInternal &sg)
                                     {
                                         return sg.getWUVarInitialisers().at(varIndex).getDerivedParams();
                                     });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isConnectivityInitParamHeterogeneous(size_t paramIndex) const
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
    return isParamValueHeterogeneous(codeStrings, paramName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const
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
    return isParamValueHeterogeneous(codeStrings, derivedParamName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isSrcNeuronParamHeterogeneous(size_t paramIndex) const
{
    const auto *neuronModel = getArchetype().getSrcNeuronGroup()->getNeuronModel();
    const std::string paramName = neuronModel->getParamNames().at(paramIndex) + "_pre";
    return isParamValueHeterogeneous({getArchetypeCode()}, paramName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isSrcNeuronDerivedParamHeterogeneous(size_t paramIndex) const
{
    const auto *neuronModel = getArchetype().getSrcNeuronGroup()->getNeuronModel();
    const std::string derivedParamName = neuronModel->getDerivedParams().at(paramIndex).name + "_pre";
    return isParamValueHeterogeneous({getArchetypeCode()}, derivedParamName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isTrgNeuronParamHeterogeneous(size_t paramIndex) const
{
    const auto *neuronModel = getArchetype().getTrgNeuronGroup()->getNeuronModel();
    const std::string paramName = neuronModel->getParamNames().at(paramIndex) + "_post";
    return isParamValueHeterogeneous({getArchetypeCode()}, paramName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isTrgNeuronDerivedParamHeterogeneous(size_t paramIndex) const
{
    const auto *neuronModel = getArchetype().getTrgNeuronGroup()->getNeuronModel();
    const std::string derivedParamName = neuronModel->getDerivedParams().at(paramIndex).name + "_post";
    return isParamValueHeterogeneous({getArchetypeCode()}, derivedParamName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isKernelSizeHeterogeneous(size_t dimensionIndex) const
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
std::string CodeGenerator::SynapseGroupMergedBase::getPreSlot(bool delay, unsigned int batchSize)
{
    if(delay) {
        assert(batchSize == 1);
        return "preDelaySlot";
    }
    else {
        return (batchSize == 1) ? "0" : "batch";
    }
}
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPostSlot(bool delay, unsigned int batchSize)
{
    if(delay) {
        assert(batchSize == 1);
        return "postDelaySlot";
    }
    else {
        return (batchSize == 1) ? "0" : "batch";
    }
}
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPreVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication & VarAccessDuplication::SHARED || batchSize == 1);
    if(delay) {
        return (singleBatch ? "preDelayOffset + " : "preBatchDelayOffset + ") + index;
    }
    else {
        return (singleBatch ? "" : "preBatchOffset + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPostVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication & VarAccessDuplication::SHARED || batchSize == 1);
    if(delay) {
        return (singleBatch ? "postDelayOffset + " : "postBatchDelayOffset + ") + index;
    }
    else {
        return (singleBatch ? "" : "postBatchOffset + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPrePrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication & VarAccessDuplication::SHARED || batchSize == 1);
   
    if(delay) {
        assert(singleBatch);
        return (singleBatch ? "prePrevSpikeTimeDelayOffset + " : "prePrevSpikeTimeBatchDelayOffset + ") + index;
    }
    else {
        return (singleBatch ? "" : "preBatchOffset + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPostPrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication & VarAccessDuplication::SHARED || batchSize == 1);
   
    if(delay) {
        assert(singleBatch);
        return (singleBatch ? "postPrevSpikeTimeDelayOffset + " : "postPrevSpikeTimeBatchDelayOffset + ") + index;
    }
    else {
        return (singleBatch ? "" : "postBatchOffset + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getSynVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index)
{
    const bool singleBatch = (varDuplication & VarAccessDuplication::SHARED || batchSize == 1);
    return (singleBatch ? "" : "synBatchOffset + ") + index;
}
//----------------------------------------------------------------------------
CodeGenerator::SynapseGroupMergedBase::SynapseGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
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

        // Add pre and postsynaptic variables to struct
        addVars(wum->getPreVars(), backend.getDeviceVarPrefix());
        addVars(wum->getPostVars(), backend.getDeviceVarPrefix());

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

        // Add additional structure for synapse dynamics access
        if(backend.isSynRemapRequired() && !wum->getSynapseDynamicsCode().empty()
           && (role == Role::SynapseDynamics || role == Role::SparseInit))
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
        const bool individualWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);

        // If synapse group has a kernel and we're either updating 
        // with procedural weights or initialising individual weights
        if(!getArchetype().getKernelSize().empty() && ((proceduralWeights && updateRole) || (connectInitRole && individualWeights))) {
            // Loop through kernel size dimensions
            for(size_t d = 0; d < getArchetype().getKernelSize().size(); d++) {
                // If this dimension has a heterogeneous size, add it to struct
                if(isKernelSizeHeterogeneous(d)) {
                    addField("unsigned int", "kernelSize" + std::to_string(d),
                             [d](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getKernelSize().at(d)); });
                }
            }
        }

        // If weights are procedura, we're initializing individual variables or we're initialising variables in a kernel
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
            if((updateRole && individualWeights) || varInitRequired) {
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
void CodeGenerator::SynapseGroupMergedBase::addPSPointerField(const std::string &type, const std::string &name, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getPSModelTargetName(); });
}
//----------------------------------------------------------------------------
void CodeGenerator::SynapseGroupMergedBase::addSrcPointerField(const std::string &type, const std::string &name, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getSrcNeuronGroup()->getName(); });
}
//----------------------------------------------------------------------------
void CodeGenerator::SynapseGroupMergedBase::addTrgPointerField(const std::string &type, const std::string &name, const std::string &prefix)
{
    assert(!Utils::isTypePointer(type));
    addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getTrgNeuronGroup()->getName(); });
}
//----------------------------------------------------------------------------
void CodeGenerator::SynapseGroupMergedBase::addWeightSharingPointerField(const std::string &type, const std::string &name, const std::string &prefix)
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
// CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::PresynapticUpdateGroupMerged::name = "PresynapticUpdate";

//----------------------------------------------------------------------------
// CodeGenerator::PostsynapticUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::PostsynapticUpdateGroupMerged::name = "PostsynapticUpdate";

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::SynapseDynamicsGroupMerged::name = "SynapseDynamics";

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDenseInitGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::SynapseDenseInitGroupMerged::name = "SynapseDenseInit";

//----------------------------------------------------------------------------
// CodeGenerator::SynapseSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::SynapseSparseInitGroupMerged::name = "SynapseSparseInit";

// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityInitGroupMerged
//----------------------------------------------------------------------------
const std::string CodeGenerator::SynapseConnectivityInitGroupMerged::name = "SynapseConnectivityInit";
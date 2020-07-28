#include "code_generator/groupMerged.h"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/mergedStructGenerator.h"

//----------------------------------------------------------------------------
// CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
void CodeGenerator::NeuronSpikeQueueUpdateGroupMerged::generate(const BackendBase &backend, CodeStream &definitionsInternal,
                                                                CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                                                                CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                                                                MergedStructData &mergedStructData, const std::string &precision) const
{
    MergedStructGenerator<NeuronSpikeQueueUpdateGroupMerged> gen(*this, precision);

    if(getArchetype().isDelayRequired()) {
        gen.addField("unsigned int", "numDelaySlots",
                     [](const NeuronGroupInternal &ng, size_t) { return std::to_string(ng.getNumDelaySlots()); });

        gen.addField("volatile unsigned int*", "spkQuePtr",
                     [&backend](const NeuronGroupInternal &ng, size_t)
                     {
                         return "getSymbolAddress(" + backend.getScalarPrefix() + "spkQuePtr" + ng.getName() + ")";
                     });
    }

    gen.addPointerField("unsigned int", "spkCnt", backend.getArrayPrefix() + "glbSpkCnt");

    if(getArchetype().isSpikeEventRequired()) {
        gen.addPointerField("unsigned int", "spkCntEvnt", backend.getArrayPrefix() + "glbSpkCntEvnt");
    }


    // Generate structure definitions and instantiation
    gen.generate(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                 mergedStructData, "NeuronSpikeQueueUpdate");
}
//----------------------------------------------------------------------------
void CodeGenerator::NeuronSpikeQueueUpdateGroupMerged::genMergedGroupSpikeCountReset(CodeStream &os) const
{
    if(getArchetype().isDelayRequired()) { // with delay
        if(getArchetype().isSpikeEventRequired()) {
            os << "group.spkCntEvnt[*group.spkQuePtr] = 0;" << std::endl;
        }
        if(getArchetype().isTrueSpikeRequired()) {
            os << "group.spkCnt[*group.spkQuePtr] = 0;" << std::endl;
        }
        else {
            os << "group.spkCnt[0] = 0;" << std::endl;
        }
    }
    else { // no delay
        if(getArchetype().isSpikeEventRequired()) {
            os << "group.spkCntEvnt[0] = 0;" << std::endl;
        }
        os << "group.spkCnt[0] = 0;" << std::endl;
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
    const auto *psm = getArchetype().getMergedInSyn().at(childIndex).first->getPSModel();
    const std::string paramName = psm->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({psm->getApplyInputCode(), psm->getDecayCode()}, paramName, childIndex, paramIndex, m_SortedMergedInSyns,
                                          [](const std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>> &inSyn)
                                          {
                                              return inSyn.first->getPSParams();
                                          });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isPSMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *psm = getArchetype().getMergedInSyn().at(childIndex).first->getPSModel();
    const std::string derivedParamName = psm->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({psm->getApplyInputCode(), psm->getDecayCode()}, derivedParamName, childIndex, paramIndex, m_SortedMergedInSyns,
                                          [](const std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>> &inSyn)
                                          {
                                              return inSyn.first->getPSDerivedParams();
                                          });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isPSMGlobalVarHeterogeneous(size_t childIndex, size_t varIndex) const
{
    // If synapse group doesn't have individual PSM variables to start with, return false
    const auto *sg = getArchetype().getMergedInSyn().at(childIndex).first;
    if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
        return false;
    }
    else {
        const auto *psm = getArchetype().getMergedInSyn().at(childIndex).first->getPSModel();
        const std::string varName = psm->getVars().at(varIndex).name;
        return isChildParamValueHeterogeneous({psm->getApplyInputCode(), psm->getDecayCode()}, varName, childIndex, varIndex, m_SortedMergedInSyns,
                                              [](const std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>> &inSyn)
                                              {
                                                  return inSyn.first->getPSConstInitVals();
                                              });
    }
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isPSMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getMergedInSyn().at(childIndex).first->getPSVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({varInitSnippet->getCode()}, paramName, childIndex, paramIndex, m_SortedMergedInSyns,
                                          [varIndex](const std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>> &inSyn) 
                                          { 
                                              return inSyn.first->getPSVarInitialisers().at(varIndex).getParams();
                                          });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronGroupMergedBase::isPSMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getArchetype().getMergedInSyn().at(childIndex).first->getPSVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({varInitSnippet->getCode()}, derivedParamName, childIndex, paramIndex, m_SortedMergedInSyns,
                                          [varIndex](const std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>> &inSyn) 
                                          { 
                                              return inSyn.first->getPSVarInitialisers().at(varIndex).getDerivedParams();
                                          });
}
//----------------------------------------------------------------------------
CodeGenerator::NeuronGroupMergedBase::NeuronGroupMergedBase(size_t index, bool init, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
    : CodeGenerator::GroupMerged<NeuronGroupInternal>(index, groups)
{
    // Build vector of vectors containing each child group's merged in syns, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedMergedInSyns, &NeuronGroupInternal::getMergedInSyn,
                             [init](const std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>> &a,
                                    const std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>> &b)
                             {
                                 return init ? a.first->canPSInitBeMerged(*b.first) : a.first->canPSBeMerged(*b.first);
                             });

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedCurrentSources, &NeuronGroupInternal::getCurrentSources,
                             [init](const CurrentSourceInternal *a, const CurrentSourceInternal *b)
                             {
                                 return init ? a->canInitBeMerged(*b) : a->canBeMerged(*b);
                             });
}
//----------------------------------------------------------------------------
void CodeGenerator::NeuronGroupMergedBase::generate(MergedStructGenerator<NeuronGroupMergedBase> &gen, const BackendBase &backend, 
                                                    const std::string &precision, const std::string &timePrecision, bool init) const
{
    gen.addField("unsigned int", "numNeurons",
                 [](const NeuronGroupInternal &ng, size_t) { return std::to_string(ng.getNumNeurons()); });

    gen.addPointerField("unsigned int", "spkCnt", backend.getArrayPrefix() + "glbSpkCnt");
    gen.addPointerField("unsigned int", "spk", backend.getArrayPrefix() + "glbSpk");

    if(getArchetype().isSpikeEventRequired()) {
        gen.addPointerField("unsigned int", "spkCntEvnt", backend.getArrayPrefix() + "glbSpkCntEvnt");
        gen.addPointerField("unsigned int", "spkEvnt", backend.getArrayPrefix() + "glbSpkEvnt");
    }

    if(getArchetype().isDelayRequired()) {
        gen.addField("volatile unsigned int*", "spkQuePtr",
                     [&backend](const NeuronGroupInternal &ng, size_t)
                     {
                         return "getSymbolAddress(" + backend.getScalarPrefix() + "spkQuePtr" + ng.getName() + ")";
                     });
    }

    if(getArchetype().isSpikeTimeRequired()) {
        gen.addPointerField(timePrecision, "sT", backend.getArrayPrefix() + "sT");
    }

    if(backend.isPopulationRNGRequired() && getArchetype().isSimRNGRequired()) {
        gen.addPointerField("curandState", "rng", backend.getArrayPrefix() + "rng");
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
            gen.addPointerField(var.type, var.name, backend.getArrayPrefix() + var.name);
        }

        // If we're initializing, add any var init EGPs to structure
        if(init) {
            gen.addEGPs(varInit[v].getSnippet()->getExtraGlobalParams(), backend.getArrayPrefix(), var.name);
        }
    }

    // If we're generating a struct for initialization
    if(init) {
        // Add heterogeneous var init parameters
        gen.addHeterogeneousVarInitParams(vars, &NeuronGroupInternal::getVarInitialisers,
                                          &NeuronGroupMergedBase::isVarInitParamHeterogeneous);

        gen.addHeterogeneousVarInitDerivedParams(vars, &NeuronGroupInternal::getVarInitialisers,
                                                 &NeuronGroupMergedBase::isVarInitDerivedParamHeterogeneous);
    }
    // Otherwise
    else {
        gen.addEGPs(nm->getExtraGlobalParams(), backend.getArrayPrefix());

        // Add heterogeneous neuron model parameters
        gen.addHeterogeneousParams(getArchetype().getNeuronModel()->getParamNames(), "",
                                   [](const NeuronGroupInternal &ng) { return ng.getParams(); },
                                   &NeuronGroupMergedBase::isParamHeterogeneous);

        // Add heterogeneous neuron model derived parameters
        gen.addHeterogeneousDerivedParams(getArchetype().getNeuronModel()->getDerivedParams(), "",
                                          [](const NeuronGroupInternal &ng) { return ng.getDerivedParams(); },
                                          &NeuronGroupMergedBase::isDerivedParamHeterogeneous);
    }

    // Loop through merged synaptic inputs in archetypical neuron group
    for(size_t i = 0; i < getArchetype().getMergedInSyn().size(); i++) {
        const SynapseGroupInternal *sg = getArchetype().getMergedInSyn()[i].first;

        // Add pointer to insyn
        addMergedInSynPointerField(gen, precision, "inSynInSyn", i, backend.getArrayPrefix() + "inSyn");

        // Add pointer to dendritic delay buffer if required
        if(sg->isDendriticDelayRequired()) {
            addMergedInSynPointerField(gen, precision, "denDelayInSyn", i, backend.getArrayPrefix() + "denDelay");

            gen.addField("volatile unsigned int*", "denDelayPtrInSyn" + std::to_string(i),
                         [&backend, i, this](const NeuronGroupInternal &, size_t groupIndex)
                         {
                             const std::string &targetName = m_SortedMergedInSyns.at(groupIndex).at(i).first->getPSModelTargetName();
                             return "getSymbolAddress(" + backend.getScalarPrefix() + "denDelayPtr" + targetName + ")";
                         });
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
                    addMergedInSynPointerField(gen, var.type, var.name + "InSyn", i, backend.getArrayPrefix() + var.name);
                }

                // If we're generating an initialization structure, also add any heterogeneous parameters, derived parameters or extra global parameters required for initializers
                if(init) {
                    const auto *varInitSnippet = varInit.at(v).getSnippet();
                    auto getVarInitialiserFn = [this](size_t groupIndex, size_t childIndex)
                    {
                        return m_SortedMergedInSyns.at(groupIndex).at(childIndex).first->getPSVarInitialisers();
                    };
                    addHeterogeneousChildVarInitParams(gen, varInitSnippet->getParamNames(), i, v, var.name + "InSyn",
                                                       &NeuronGroupMergedBase::isPSMVarInitParamHeterogeneous, getVarInitialiserFn);
                    addHeterogeneousChildVarInitDerivedParams(gen, varInitSnippet->getDerivedParams(), i, v, var.name + "InSyn",
                                                              &NeuronGroupMergedBase::isPSMVarInitDerivedParamHeterogeneous, getVarInitialiserFn);
                    addChildEGPs(gen, varInitSnippet->getExtraGlobalParams(), i, backend.getArrayPrefix(), var.name + "InSyn",
                                 [var, this](size_t groupIndex, size_t childIndex)
                                 {
                                     return var.name + m_SortedMergedInSyns.at(groupIndex).at(childIndex).first->getPSModelTargetName();
                                 });
                }
            }
            // Otherwise, if postsynaptic model variables are global and we're updating 
            // **NOTE** global variable values aren't useful during initialization
            else if(!init) {
                // If GLOBALG variable should be implemented heterogeneously, add value
                if(isPSMGlobalVarHeterogeneous(i, v)) {
                    gen.addScalarField(var.name + "InSyn" + std::to_string(i),
                                       [this, i, v](const NeuronGroupInternal &, size_t groupIndex)
                                       {
                                           const double val = m_SortedMergedInSyns.at(groupIndex).at(i).first->getPSConstInitVals().at(v);
                                           return Utils::writePreciseString(val);
                                       });
                }
            }
        }

        if(!init) {
            // Add any heterogeneous postsynaptic model parameters
            const auto paramNames = sg->getPSModel()->getParamNames();
            addHeterogeneousChildParams(gen, paramNames, i, "InSyn", &NeuronGroupMergedBase::isPSMParamHeterogeneous,
                                        [this](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                        {
                                            return m_SortedMergedInSyns.at(groupIndex).at(childIndex).first->getPSParams().at(paramIndex);
                                        });

            // Add any heterogeneous postsynaptic mode derived parameters
            const auto derivedParams = sg->getPSModel()->getDerivedParams();
            addHeterogeneousChildDerivedParams(gen, derivedParams, i, "InSyn", &NeuronGroupMergedBase::isPSMDerivedParamHeterogeneous,
                                               [this](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                               {
                                                   return m_SortedMergedInSyns.at(groupIndex).at(childIndex).first->getPSDerivedParams().at(paramIndex);
                                               });
            // Add EGPs
            addChildEGPs(gen, sg->getPSModel()->getExtraGlobalParams(), i, backend.getArrayPrefix(), "InSyn",
                         [this](size_t groupIndex, size_t childIndex)
                         {
                             return m_SortedMergedInSyns.at(groupIndex).at(childIndex).first->getPSModelTargetName();
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
                gen.addField(var.type + "*", var.name + "CS" + std::to_string(i),
                             [&backend, i, var, this](const NeuronGroupInternal &, size_t groupIndex)
                             {
                                 return backend.getArrayPrefix() + var.name + m_SortedCurrentSources.at(groupIndex).at(i)->getName();
                             });
            }

            // If we're generating an initialization structure, also add any heterogeneous parameters, derived parameters or extra global parameters required for initializers
            if(init) {
                const auto *varInitSnippet = varInit.at(v).getSnippet();
                auto getVarInitialiserFn = [this](size_t groupIndex, size_t childIndex)
                {
                    return m_SortedCurrentSources.at(groupIndex).at(childIndex)->getVarInitialisers();
                };
                addHeterogeneousChildVarInitParams(gen, varInitSnippet->getParamNames(), i, v, var.name + "CS",
                                                   &NeuronGroupMergedBase::isCurrentSourceVarInitParamHeterogeneous, getVarInitialiserFn);
                addHeterogeneousChildVarInitDerivedParams(gen, varInitSnippet->getDerivedParams(), i, v, var.name + "CS",
                                                          &NeuronGroupMergedBase::isCurrentSourceVarInitDerivedParamHeterogeneous, getVarInitialiserFn);
                addChildEGPs(gen, varInitSnippet->getExtraGlobalParams(), i, backend.getArrayPrefix(), var.name + "CS", 
                             [var, this](size_t groupIndex, size_t childIndex)
                             {
                                return var.name + m_SortedCurrentSources.at(groupIndex).at(childIndex)->getName();
                             });
            }
        }

        if(!init) {
            // Add any heterogeneous current source parameters
            const auto paramNames = cs->getCurrentSourceModel()->getParamNames();
            addHeterogeneousChildParams(gen, paramNames, i, "CS", &NeuronGroupMergedBase::isCurrentSourceParamHeterogeneous,
                                        [this](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                        {
                                            return m_SortedCurrentSources.at(groupIndex).at(childIndex)->getParams().at(paramIndex);
                                        });

            // Add any heterogeneous current source derived parameters
            const auto derivedParams = cs->getCurrentSourceModel()->getDerivedParams();
            addHeterogeneousChildDerivedParams(gen, derivedParams, i, "CS", &NeuronGroupMergedBase::isCurrentSourceDerivedParamHeterogeneous,
                                               [this](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                               {
                                                   return m_SortedCurrentSources.at(groupIndex).at(childIndex)->getDerivedParams().at(paramIndex);
                                               });

            // Add EGPs
            addChildEGPs(gen, cs->getCurrentSourceModel()->getExtraGlobalParams(), i, backend.getArrayPrefix(), "CS",
                         [this](size_t groupIndex, size_t childIndex) 
                         { 
                             return m_SortedCurrentSources.at(groupIndex).at(childIndex)->getName(); 
                         });
            
        }
    }

    // Loop through neuron groups
    std::vector<std::vector<SynapseGroupInternal *>> eventThresholdSGs;
    for(const auto &g : getGroups()) {
        // Reserve vector for this group's children
        eventThresholdSGs.emplace_back();

        // Add synapse groups 
        for(const auto &s : g.get().getSpikeEventCondition()) {
            if(s.egpInThresholdCode) {
                eventThresholdSGs.back().push_back(s.synapseGroup);
            }
        }
    }

    // Loop through all spike event conditions
    using FieldType = std::remove_reference<decltype(gen)>::type::FieldType;
    size_t i = 0;
    for(const auto &s : getArchetype().getSpikeEventCondition()) {
        // If threshold condition references any EGPs
        if(s.egpInThresholdCode) {
            // Loop through all EGPs in synapse group and add to merged group
            // **TODO** should only be ones referenced
            const auto sgEGPs = s.synapseGroup->getWUModel()->getExtraGlobalParams();
            for(const auto &egp : sgEGPs) {
                const bool isPointer = Utils::isTypePointer(egp.type);
                const std::string prefix = isPointer ? backend.getArrayPrefix() : "";
                gen.addField(egp.type, egp.name + "EventThresh" + std::to_string(i),
                             [eventThresholdSGs, prefix, egp, i](const NeuronGroupInternal &, size_t groupIndex)
                             {
                                 return prefix + egp.name + eventThresholdSGs.at(groupIndex).at(i)->getName();
                             },
                             Utils::isTypePointer(egp.type) ? FieldType::PointerEGP : FieldType::ScalarEGP);
            }
            i++;
        }
    }

    
}
//----------------------------------------------------------------------------
void CodeGenerator::NeuronGroupMergedBase::addMergedInSynPointerField(MergedStructGenerator<NeuronGroupMergedBase> &gen,
                                                                      const std::string &type, const std::string &name, 
                                                                      size_t archetypeIndex, const std::string &prefix) const
{
    assert(!Utils::isTypePointer(type));
    gen.addField(type + "*", name + std::to_string(archetypeIndex),
                 [prefix, archetypeIndex, this](const NeuronGroupInternal &, size_t groupIndex)
                 {
                     return prefix + m_SortedMergedInSyns.at(groupIndex).at(archetypeIndex).first->getPSModelTargetName();
                 });
}

//----------------------------------------------------------------------------
// CodeGenerator::NeuronUpdateGroupMerged
//----------------------------------------------------------------------------
CodeGenerator::NeuronUpdateGroupMerged::NeuronUpdateGroupMerged(size_t index, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, false, groups)
{
    // Build vector of vectors containing each child group's incoming synapse groups
    // with postsynaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(getArchetype().getInSynWithPostCode(), m_SortedInSynWithPostCode, &NeuronGroupInternal::getInSynWithPostCode,
                             [](const SynapseGroupInternal *a, const SynapseGroupInternal *b){ return a->canWUPostBeMerged(*b); });

    // Build vector of vectors containing each child group's outgoing synapse groups
    // with presynaptic synaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(getArchetype().getOutSynWithPreCode(), m_SortedOutSynWithPreCode, &NeuronGroupInternal::getOutSynWithPreCode,
                             [](const SynapseGroupInternal *a, const SynapseGroupInternal *b){ return a->canWUPreBeMerged(*b); });
}
//----------------------------------------------------------------------------
std::string CodeGenerator::NeuronUpdateGroupMerged::getCurrentQueueOffset() const
{
    assert(getArchetype().isDelayRequired());
    return "(*group.spkQuePtr * group.numNeurons)";
}
//----------------------------------------------------------------------------
std::string CodeGenerator::NeuronUpdateGroupMerged::getPrevQueueOffset() const
{
    assert(getArchetype().isDelayRequired());
    return "(((*group.spkQuePtr + " + std::to_string(getArchetype().getNumDelaySlots() - 1) + ") % " + std::to_string(getArchetype().getNumDelaySlots()) + ") * group.numNeurons)";
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *wum = getArchetype().getInSynWithPostCode().at(childIndex)->getWUModel();
    const std::string paramName = wum->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({wum->getPostSpikeCode()}, paramName, childIndex, paramIndex, m_SortedInSynWithPostCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If derived parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *wum = getArchetype().getInSynWithPostCode().at(childIndex)->getWUModel();
    const std::string derivedParamName = wum->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({wum->getPostSpikeCode()}, derivedParamName, childIndex, paramIndex, m_SortedInSynWithPostCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUDerivedParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *wum = getArchetype().getOutSynWithPreCode().at(childIndex)->getWUModel();
    const std::string paramName = wum->getParamNames().at(paramIndex);
    return isChildParamValueHeterogeneous({wum->getPreSpikeCode()}, paramName, childIndex, paramIndex, m_SortedOutSynWithPreCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
{
    // If derived parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *wum = getArchetype().getOutSynWithPreCode().at(childIndex)->getWUModel();
    const std::string derivedParamName = wum->getDerivedParams().at(paramIndex).name;
    return isChildParamValueHeterogeneous({wum->getPreSpikeCode()}, derivedParamName, childIndex, paramIndex, m_SortedOutSynWithPreCode,
                                          [](const SynapseGroupInternal *s) { return s->getWUDerivedParams(); });
}
//----------------------------------------------------------------------------
void CodeGenerator::NeuronUpdateGroupMerged::generate(const BackendBase &backend, CodeStream &definitionsInternal,
                                                      CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                                                      CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                                                      MergedStructData &mergedStructData, const std::string &precision,
                                                      const std::string &timePrecision) const
{
    // Create merged struct generator
    MergedStructGenerator<NeuronGroupMergedBase> gen(*this, precision);

    // Build generic struct
    NeuronGroupMergedBase::generate(gen, backend, precision, timePrecision, false);

    // Generate struct fields for incoming synapse groups with postsynaptic update code
    const auto inSynWithPostCode = getArchetype().getInSynWithPostCode();
    generateWUVar(gen, backend, "WUPost", inSynWithPostCode, m_SortedInSynWithPostCode,
                  &WeightUpdateModels::Base::getPostVars, &NeuronUpdateGroupMerged::isInSynWUMParamHeterogeneous,
                  &NeuronUpdateGroupMerged::isInSynWUMDerivedParamHeterogeneous);
    
    // Generate struct fields for outgoing synapse groups with presynaptic update code
    const auto outSynWithPreCode = getArchetype().getOutSynWithPreCode();
    generateWUVar(gen, backend, "WUPre", outSynWithPreCode, m_SortedOutSynWithPreCode,
                  &WeightUpdateModels::Base::getPreVars, &NeuronUpdateGroupMerged::isOutSynWUMParamHeterogeneous,
                  &NeuronUpdateGroupMerged::isOutSynWUMDerivedParamHeterogeneous);

    // Generate structure definitions and instantiation
    gen.generate(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                 mergedStructData, "NeuronUpdate");
}
//----------------------------------------------------------------------------
void CodeGenerator::NeuronUpdateGroupMerged::generateWUVar(MergedStructGenerator<NeuronGroupMergedBase> &gen, const BackendBase &backend, 
                                                           const std::string &fieldPrefixStem, const std::vector<SynapseGroupInternal *> &archetypeSyn,
                                                           const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                                                           Models::Base::VarVec (WeightUpdateModels::Base::*getVars)(void) const,
                                                           bool(NeuronUpdateGroupMerged::*isParamHeterogeneous)(size_t, size_t) const,
                                                           bool(NeuronUpdateGroupMerged::*isDerivedParamHeterogeneous)(size_t, size_t) const) const
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
            gen.addField(var.type + "*", var.name + fieldPrefixStem + std::to_string(i),
                         [i, var, &backend, &sortedSyn](const NeuronGroupInternal &, size_t groupIndex)
                         {
                             return backend.getArrayPrefix() + var.name + sortedSyn.at(groupIndex).at(i)->getName();
                         });
        }

        // Add any heterogeneous parameters
        addHeterogeneousChildParams<NeuronUpdateGroupMerged>(gen, sg->getWUModel()->getParamNames(), i, fieldPrefixStem, isParamHeterogeneous,
                                                             [&sortedSyn](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                                             {
                                                                 return sortedSyn.at(groupIndex).at(childIndex)->getWUParams().at(paramIndex);
                                                             });

        // Add any heterogeneous derived parameters
        addHeterogeneousChildDerivedParams<NeuronUpdateGroupMerged>(gen, sg->getWUModel()->getDerivedParams(), i, fieldPrefixStem, isDerivedParamHeterogeneous,
                                                                    [&sortedSyn](size_t groupIndex, size_t childIndex, size_t paramIndex)
                                                                    {
                                                                        return sortedSyn.at(groupIndex).at(childIndex)->getWUDerivedParams().at(paramIndex);
                                                                    });

        // Add EGPs
        addChildEGPs(gen, sg->getWUModel()->getExtraGlobalParams(), i, backend.getArrayPrefix(), fieldPrefixStem,
                     [&sortedSyn](size_t groupIndex, size_t childIndex)
                     {
                         return sortedSyn.at(groupIndex).at(childIndex)->getName();
                     });
    }
}

//----------------------------------------------------------------------------
// CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
CodeGenerator::NeuronInitGroupMerged::NeuronInitGroupMerged(size_t index, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, true, groups)
{
    // Build vector of vectors containing each child group's incoming 
    // synapse groups, ordered to match those of the archetype group
    orderNeuronGroupChildren(getArchetype().getInSynWithPostVars(), m_SortedInSynWithPostVars, &NeuronGroupInternal::getInSynWithPostVars,
                             [](const SynapseGroupInternal *a, const SynapseGroupInternal *b) { return a->canWUPostInitBeMerged(*b); });

    // Build vector of vectors containing each child group's outgoing 
    // synapse groups, ordered to match those of the archetype group
    orderNeuronGroupChildren(getArchetype().getOutSynWithPreVars(), m_SortedOutSynWithPreVars, &NeuronGroupInternal::getOutSynWithPreVars,
                             [](const SynapseGroupInternal *a, const SynapseGroupInternal *b){ return a->canWUPreInitBeMerged(*b); });
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
void CodeGenerator::NeuronInitGroupMerged::generate(const BackendBase &backend, CodeStream &definitionsInternal,
                                                    CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                                                    CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                                                    MergedStructData &mergedStructData, const std::string &precision,
                                                    const std::string &timePrecision) const
{
    // Create merged struct generator
    MergedStructGenerator<NeuronGroupMergedBase> gen(*this, precision);

    // Build generic struct
    NeuronGroupMergedBase::generate(gen, backend, precision, timePrecision, true);

    // Generate struct fields for incoming synapse groups with postsynaptic variables
    const auto inSynWithPostVars = getArchetype().getInSynWithPostVars();
    generateWUVar(gen, backend, "WUPost", inSynWithPostVars, m_SortedInSynWithPostVars,
                  &WeightUpdateModels::Base::getPostVars, &SynapseGroupInternal::getWUPostVarInitialisers,
                  &NeuronInitGroupMerged::isInSynWUMVarInitParamHeterogeneous,
                  &NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamHeterogeneous);
    

    // Generate struct fields for outgoing synapse groups
    const auto outSynWithPreVars = getArchetype().getOutSynWithPreVars();
    generateWUVar(gen, backend, "WUPre", outSynWithPreVars, m_SortedOutSynWithPreVars,
                  &WeightUpdateModels::Base::getPreVars, &SynapseGroupInternal::getWUPreVarInitialisers,
                  &NeuronInitGroupMerged::isOutSynWUMVarInitParamHeterogeneous,
                  &NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamHeterogeneous);

    // Generate structure definitions and instantiation
    gen.generate(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                 mergedStructData, "NeuronInit");
}
//----------------------------------------------------------------------------
void CodeGenerator::NeuronInitGroupMerged::generateWUVar(MergedStructGenerator<NeuronGroupMergedBase> &gen, const BackendBase &backend,
                                                         const std::string &fieldPrefixStem,
                                                         const std::vector<SynapseGroupInternal *> &archetypeSyn,
                                                         const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                                                         Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                                                         const std::vector<Models::VarInit> &(SynapseGroupInternal:: *getVarInitialisers)(void) const,
                                                         bool(NeuronInitGroupMerged::*isParamHeterogeneous)(size_t, size_t, size_t) const,
                                                         bool(NeuronInitGroupMerged::*isDerivedParamHeterogeneous)(size_t, size_t, size_t) const) const
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
                gen.addField(var.type + "*", var.name + fieldPrefixStem + std::to_string(i),
                             [i, var, &backend, &sortedSyn](const NeuronGroupInternal &, size_t groupIndex)
                             {
                                 return backend.getArrayPrefix() + var.name + sortedSyn.at(groupIndex).at(i)->getName();
                             });
            }

            // Also add any heterogeneous, derived or extra global parameters required for initializers
            const auto *varInitSnippet = varInit.at(v).getSnippet();
            auto getVarInitialiserFn = [&sortedSyn](size_t groupIndex, size_t childIndex)
                                       {
                                           return sortedSyn.at(groupIndex).at(childIndex)->getWUPreVarInitialisers();
                                       };
            addHeterogeneousChildVarInitParams<NeuronInitGroupMerged>(gen, varInitSnippet->getParamNames(), i, v, var.name + fieldPrefixStem,
                                                                      isParamHeterogeneous, getVarInitialiserFn);
            addHeterogeneousChildVarInitDerivedParams<NeuronInitGroupMerged>(gen, varInitSnippet->getDerivedParams(), i, v, var.name + fieldPrefixStem,
                                                                             isDerivedParamHeterogeneous, getVarInitialiserFn);
            addChildEGPs(gen, varInitSnippet->getExtraGlobalParams(), i, backend.getArrayPrefix(), var.name + fieldPrefixStem,
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
void CodeGenerator::SynapseDendriticDelayUpdateGroupMerged::generate(const BackendBase &backend, CodeStream &definitionsInternal,
                                                                     CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                                                                     CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                                                                     MergedStructData &mergedStructData, const std::string &precision) const
{
    MergedStructGenerator<SynapseDendriticDelayUpdateGroupMerged> gen(*this, precision);

    gen.addField("volatile unsigned int*", "denDelayPtr",
                 [&backend](const SynapseGroupInternal &sg, size_t)
                 {
                     return "getSymbolAddress(" + backend.getScalarPrefix() + "denDelayPtr" + sg.getPSModelTargetName() + ")";
                 });

    // Generate structure definitions and instantiation
    gen.generate(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                 mergedStructData, "SynapseDendriticDelayUpdate");
}

// ----------------------------------------------------------------------------
// SynapseConnectivityHostInitGroupMerged
//------------------------------------------------------------------------
void CodeGenerator::SynapseConnectivityHostInitGroupMerged::generate(const BackendBase &backend, CodeStream &definitionsInternal,
                                                                     CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                                                                     CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                                                                     MergedStructData &mergedStructData, const std::string &precision) const
{
    MergedStructGenerator<SynapseConnectivityHostInitGroupMerged> gen(*this, precision);

    // **TODO** these could be generic
    gen.addField("unsigned int", "numSrcNeurons",
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
    gen.addField("unsigned int", "numTrgNeurons",
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    gen.addField("unsigned int", "rowStride",
                 [&backend](const SynapseGroupInternal &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });

    // Add heterogeneous connectivity initialiser model parameters
    gen.addHeterogeneousParams(getArchetype().getConnectivityInitialiser().getSnippet()->getParamNames(), "",
                               [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); },
                               &SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous);


    // Add heterogeneous connectivity initialiser derived parameters
    gen.addHeterogeneousDerivedParams(getArchetype().getConnectivityInitialiser().getSnippet()->getDerivedParams(), "",
                                      [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); },
                                      &SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous);

    // Add EGP pointers to struct for both host and device EGPs if they are seperate
    const auto egps = getArchetype().getConnectivityInitialiser().getSnippet()->getExtraGlobalParams();
    for(const auto &e : egps) {
        gen.addField(e.type + "*", e.name,
                     [e](const SynapseGroupInternal &g, size_t) { return "&" + e.name + g.getName(); });

        if(!backend.getArrayPrefix().empty()) {
            gen.addField(e.type + "*", backend.getArrayPrefix() + e.name,
                         [e, &backend](const SynapseGroupInternal &g, size_t) 
                         { 
                             return "&" + backend.getArrayPrefix() + e.name + g.getName();
                         });
        }
    }

    // Generate structure definitions and instantiation
    gen.generate(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                 mergedStructData,  "SynapseConnectivityHostInit", true);
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

// ----------------------------------------------------------------------------
// SynapseConnectivityInitGroupMerged
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseConnectivityInitGroupMerged::isConnectivityInitParamHeterogeneous(size_t paramIndex) const
{
    const auto *connectivityInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const auto rowBuildStateVars = connectivityInitSnippet->getRowBuildStateVars();
    
    // Build list of code strings containing row build code and any row build state variable values
    std::vector<std::string> codeStrings{connectivityInitSnippet->getRowBuildCode()};
    std::transform(rowBuildStateVars.cbegin(), rowBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });
    
    const std::string paramName = connectivityInitSnippet->getParamNames().at(paramIndex);
    return isParamValueHeterogeneous(codeStrings, paramName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseConnectivityInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const
{
    const auto *connectivityInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const auto rowBuildStateVars = connectivityInitSnippet->getRowBuildStateVars();

    // Build list of code strings containing row build code and any row build state variable values
    std::vector<std::string> codeStrings{connectivityInitSnippet->getRowBuildCode()};
    std::transform(rowBuildStateVars.cbegin(), rowBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });

    const std::string derivedParamName = connectivityInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamValueHeterogeneous(codeStrings, derivedParamName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); });
}
//------------------------------------------------------------------------
void CodeGenerator::SynapseConnectivityInitGroupMerged::generate(const BackendBase &backend, CodeStream &definitionsInternal,
                                                                 CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                                                                 CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                                                                 MergedStructData &mergedStructData, const std::string &precision) const
{
    MergedStructGenerator<SynapseConnectivityInitGroupMerged> gen(*this, precision);

    // **TODO** these could be generic
    gen.addField("unsigned int", "numSrcNeurons",
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
    gen.addField("unsigned int", "numTrgNeurons",
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    gen.addField("unsigned int", "rowStride",
                 [&backend](const SynapseGroupInternal &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });

    // Add heterogeneous connectivity initialiser model parameters
    gen.addHeterogeneousParams(getArchetype().getConnectivityInitialiser().getSnippet()->getParamNames(), "",
                               [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); },
                               &SynapseConnectivityInitGroupMerged::isConnectivityInitParamHeterogeneous);


    // Add heterogeneous connectivity initialiser derived parameters
    gen.addHeterogeneousDerivedParams(getArchetype().getConnectivityInitialiser().getSnippet()->getDerivedParams(), "",
                                      [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); },
                                      &SynapseConnectivityInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous);

    if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        gen.addPointerField("unsigned int", "rowLength", backend.getArrayPrefix() + "rowLength");
        gen.addPointerField(getArchetype().getSparseIndType(), "ind", backend.getArrayPrefix() + "ind");
    }
    else if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        gen.addPointerField("uint32_t", "gp", backend.getArrayPrefix() + "gp");
    }

    // Add EGPs to struct
    gen.addEGPs(getArchetype().getConnectivityInitialiser().getSnippet()->getExtraGlobalParams(),
                backend.getArrayPrefix());

    // Generate structure definitions and instantiation
    gen.generate(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                 mergedStructData, "SynapseConnectivityInit");
}

//----------------------------------------------------------------------------
// CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPresynapticAxonalDelaySlot() const
{
    assert(getArchetype().getSrcNeuronGroup()->isDelayRequired());

    const unsigned int numDelaySteps = getArchetype().getDelaySteps();
    if(numDelaySteps == 0) {
        return "(*group.srcSpkQuePtr)";
    }
    else {
        const unsigned int numSrcDelaySlots = getArchetype().getSrcNeuronGroup()->getNumDelaySlots();
        return "((*group.srcSpkQuePtr + " + std::to_string(numSrcDelaySlots - numDelaySteps) + ") % " + std::to_string(numSrcDelaySlots) + ")";
    }
}
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getPostsynapticBackPropDelaySlot() const
{
    assert(getArchetype().getTrgNeuronGroup()->isDelayRequired());

    const unsigned int numBackPropDelaySteps = getArchetype().getBackPropDelaySteps();
    if(numBackPropDelaySteps == 0) {
        return "(*group.trgSpkQuePtr)";
    }
    else {
        const unsigned int numTrgDelaySlots = getArchetype().getTrgNeuronGroup()->getNumDelaySlots();
        return "((*group.trgSpkQuePtr + " + std::to_string(numTrgDelaySlots - numBackPropDelaySteps) + ") % " + std::to_string(numTrgDelaySlots) + ")";
    }
}
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMergedBase::getDendriticDelayOffset(const std::string &offset) const
{
    assert(getArchetype().isDendriticDelayRequired());

    if(offset.empty()) {
        return "(*group.denDelayPtr * group.numTrgNeurons) + ";
    }
    else {
        return "(((*group.denDelayPtr + " + offset + ") % " + std::to_string(getArchetype().getMaxDendriticDelayTimesteps()) + ") * group.numTrgNeurons) + ";
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
    const auto *connectivityInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const auto rowBuildStateVars = connectivityInitSnippet->getRowBuildStateVars();

    // Build list of code strings containing row build code and any row build state variable values
    std::vector<std::string> codeStrings{connectivityInitSnippet->getRowBuildCode()};
    std::transform(rowBuildStateVars.cbegin(), rowBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });

    const std::string paramName = connectivityInitSnippet->getParamNames().at(paramIndex);
    return isParamValueHeterogeneous(codeStrings, paramName, paramIndex,
                                     [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool CodeGenerator::SynapseGroupMergedBase::isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const
{
    const auto *connectivityInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const auto rowBuildStateVars = connectivityInitSnippet->getRowBuildStateVars();

    // Build list of code strings containing row build code and any row build state variable values
    std::vector<std::string> codeStrings{connectivityInitSnippet->getRowBuildCode()};
    std::transform(rowBuildStateVars.cbegin(), rowBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });

    const std::string derivedParamName = connectivityInitSnippet->getDerivedParams().at(paramIndex).name;
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
void CodeGenerator::SynapseGroupMergedBase::generate(const BackendBase &backend, CodeStream &definitionsInternal,
                                                     CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                                                     CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                                                     MergedStructData &mergedStructData, const std::string &precision, 
                                                     const std::string &timePrecision, const std::string &name, Role role) const
{
    const bool updateRole = ((role == Role::PresynapticUpdate)
                             || (role == Role::PostsynapticUpdate)
                             || (role == Role::SynapseDynamics));
    const WeightUpdateModels::Base *wum = getArchetype().getWUModel();

    MergedStructGenerator<SynapseGroupMergedBase> gen(*this, precision);

    gen.addField("unsigned int", "rowStride",
                 [&backend](const SynapseGroupInternal &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });
    if(role == Role::PostsynapticUpdate || role == Role::SparseInit) {
        gen.addField("unsigned int", "colStride",
                     [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getMaxSourceConnections()); });
    }

    gen.addField("unsigned int", "numSrcNeurons",
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
    gen.addField("unsigned int", "numTrgNeurons",
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });

    // If this role is one where postsynaptic input can be provided
    if(role == Role::PresynapticUpdate || role == Role::SynapseDynamics) {
        if(getArchetype().isDendriticDelayRequired()) {
            addPSPointerField(gen, precision, "denDelay", backend.getArrayPrefix() + "denDelay");
            gen.addField("volatile unsigned int*", "denDelayPtr",
                         [&backend](const SynapseGroupInternal &sg, size_t)
                         {
                             return "getSymbolAddress(" + backend.getScalarPrefix() + "denDelayPtr" + sg.getPSModelTargetName() + ")";
                         });
        }
        else {
            addPSPointerField(gen, precision, "inSyn", backend.getArrayPrefix() + "inSyn");
        }
    }

    if(role == Role::PresynapticUpdate) {
        if(getArchetype().isTrueSpikeRequired()) {
            addSrcPointerField(gen, "unsigned int", "srcSpkCnt", backend.getArrayPrefix() + "glbSpkCnt");
            addSrcPointerField(gen, "unsigned int", "srcSpk", backend.getArrayPrefix() + "glbSpk");
        }

        if(getArchetype().isSpikeEventRequired()) {
            addSrcPointerField(gen, "unsigned int", "srcSpkCntEvnt", backend.getArrayPrefix() + "glbSpkCntEvnt");
            addSrcPointerField(gen, "unsigned int", "srcSpkEvnt", backend.getArrayPrefix() + "glbSpkEvnt");
        }
    }
    else if(role == Role::PostsynapticUpdate) {
        addTrgPointerField(gen, "unsigned int", "trgSpkCnt", backend.getArrayPrefix() + "glbSpkCnt");
        addTrgPointerField(gen, "unsigned int", "trgSpk", backend.getArrayPrefix() + "glbSpk");
    }

    // If this structure is used for updating rather than initializing
    if(updateRole) {
        // If presynaptic population has delay buffers
        if(getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
            gen.addField("volatile unsigned int*", "srcSpkQuePtr",
                         [&backend](const SynapseGroupInternal &sg, size_t)
                         {
                             return "getSymbolAddress(" + backend.getScalarPrefix() + "spkQuePtr" + sg.getSrcNeuronGroup()->getName() + ")";
                         });
        }

        // If postsynaptic population has delay buffers
        if(getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
            gen.addField("volatile unsigned int*", "trgSpkQuePtr",
                         [&backend](const SynapseGroupInternal &sg, size_t)
                         {
                             return "getSymbolAddress(" + backend.getScalarPrefix() + "spkQuePtr" + sg.getTrgNeuronGroup()->getName() + ")";
                         });
        }

        // Add heterogeneous presynaptic neuron model parameters
        gen.addHeterogeneousParams(getArchetype().getSrcNeuronGroup()->getNeuronModel()->getParamNames(), "Pre",
                                   [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getParams(); },
                                   &SynapseGroupMergedBase::isSrcNeuronParamHeterogeneous);

        // Add heterogeneous presynaptic neuron model derived parameters
        gen.addHeterogeneousDerivedParams(getArchetype().getSrcNeuronGroup()->getNeuronModel()->getDerivedParams(), "Pre",
                                          [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getDerivedParams(); },
                                          &SynapseGroupMergedBase::isSrcNeuronDerivedParamHeterogeneous);

        // Add heterogeneous postsynaptic neuron model parameters
        gen.addHeterogeneousParams(getArchetype().getTrgNeuronGroup()->getNeuronModel()->getParamNames(), "Post",
                                   [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getParams(); },
                                   &SynapseGroupMergedBase::isTrgNeuronParamHeterogeneous);

        // Add heterogeneous postsynaptic neuron model derived parameters
        gen.addHeterogeneousDerivedParams(getArchetype().getTrgNeuronGroup()->getNeuronModel()->getDerivedParams(), "Post",
                                          [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getDerivedParams(); },
                                          &SynapseGroupMergedBase::isTrgNeuronDerivedParamHeterogeneous);

        // Get correct code string
        const std::string code = getArchetypeCode();

        // Loop through variables in presynaptic neuron model
        const auto preVars = getArchetype().getSrcNeuronGroup()->getNeuronModel()->getVars();
        for(const auto &v : preVars) {
            // If variable is referenced in code string, add source pointer
            if(code.find("$(" + v.name + "_pre)") != std::string::npos) {
                addSrcPointerField(gen, v.type, v.name + "Pre", backend.getArrayPrefix() + v.name);
            }
        }

        // Loop through variables in postsynaptic neuron model
        const auto postVars = getArchetype().getTrgNeuronGroup()->getNeuronModel()->getVars();
        for(const auto &v : postVars) {
            // If variable is referenced in code string, add target pointer
            if(code.find("$(" + v.name + "_post)") != std::string::npos) {
                addTrgPointerField(gen, v.type, v.name + "Post", backend.getArrayPrefix() + v.name);
            }
        }

        // Loop through extra global parameters in presynaptic neuron model
        const auto preEGPs = getArchetype().getSrcNeuronGroup()->getNeuronModel()->getExtraGlobalParams();
        for(const auto &e : preEGPs) {
            if(code.find("$(" + e.name + "_pre)") != std::string::npos) {
                const bool isPointer = Utils::isTypePointer(e.type);
                const std::string prefix = isPointer ? backend.getArrayPrefix() : "";
                gen.addField(e.type, e.name + "Pre",
                             [e, prefix](const SynapseGroupInternal &sg, size_t) { return prefix + e.name + sg.getSrcNeuronGroup()->getName(); },
                             Utils::isTypePointer(e.type) ? decltype(gen)::FieldType::PointerEGP : decltype(gen)::FieldType::ScalarEGP);
            }
        }

        // Loop through extra global parameters in postsynaptic neuron model
        const auto postEGPs = getArchetype().getTrgNeuronGroup()->getNeuronModel()->getExtraGlobalParams();
        for(const auto &e : postEGPs) {
            if(code.find("$(" + e.name + "_post)") != std::string::npos) {
                const bool isPointer = Utils::isTypePointer(e.type);
                const std::string prefix = isPointer ? backend.getArrayPrefix() : "";
                gen.addField(e.type, e.name + "Post",
                             [e, prefix](const SynapseGroupInternal &sg, size_t) { return prefix + e.name + sg.getTrgNeuronGroup()->getName(); },
                             Utils::isTypePointer(e.type) ? decltype(gen)::FieldType::PointerEGP : decltype(gen)::FieldType::ScalarEGP);
            }
        }

        // Add spike times if required
        if(wum->isPreSpikeTimeRequired()) {
            addSrcPointerField(gen, timePrecision, "sTPre", backend.getArrayPrefix() + "sT");
        }
        if(wum->isPostSpikeTimeRequired()) {
            addTrgPointerField(gen, timePrecision, "sTPost", backend.getArrayPrefix() + "sT");
        }

        // Add heterogeneous weight update model parameters
        gen.addHeterogeneousParams(wum->getParamNames(), "",
                                   [](const SynapseGroupInternal &sg) { return sg.getWUParams(); },
                                   &SynapseGroupMergedBase::isWUParamHeterogeneous);

        // Add heterogeneous weight update model derived parameters
        gen.addHeterogeneousDerivedParams(wum->getDerivedParams(), "",
                                          [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); },
                                          &SynapseGroupMergedBase::isWUDerivedParamHeterogeneous);

        // Add pre and postsynaptic variables to struct
        gen.addVars(wum->getPreVars(), backend.getArrayPrefix());
        gen.addVars(wum->getPostVars(), backend.getArrayPrefix());

        // Add EGPs to struct
        gen.addEGPs(wum->getExtraGlobalParams(), backend.getArrayPrefix());

        // If we're updating a group with procedural connectivity
        if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) {
            // Add heterogeneous connectivity initialiser model parameters
            gen.addHeterogeneousParams(getArchetype().getConnectivityInitialiser().getSnippet()->getParamNames(), "",
                                       [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); },
                                       &SynapseGroupMergedBase::isConnectivityInitParamHeterogeneous);


            // Add heterogeneous connectivity initialiser derived parameters
            gen.addHeterogeneousDerivedParams(getArchetype().getConnectivityInitialiser().getSnippet()->getDerivedParams(), "",
                                              [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); },
                                              &SynapseGroupMergedBase::isConnectivityInitDerivedParamHeterogeneous);
        }
    }

    // Add pointers to connectivity data
    if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        addWeightSharingPointerField(gen, "unsigned int", "rowLength", backend.getArrayPrefix() + "rowLength");
        addWeightSharingPointerField(gen, getArchetype().getSparseIndType(), "ind", backend.getArrayPrefix() + "ind");

        // Add additional structure for postsynaptic access
        if(backend.isPostsynapticRemapRequired() && !wum->getLearnPostCode().empty()
           && (role == Role::PostsynapticUpdate || role == Role::SparseInit))
        {
            addWeightSharingPointerField(gen, "unsigned int", "colLength", backend.getArrayPrefix() + "colLength");
            addWeightSharingPointerField(gen, "unsigned int", "remap", backend.getArrayPrefix() + "remap");
        }

        // Add additional structure for synapse dynamics access
        if(backend.isSynRemapRequired() && !wum->getSynapseDynamicsCode().empty()
           && (role == Role::SynapseDynamics || role == Role::SparseInit))
        {
            addWeightSharingPointerField(gen, "unsigned int", "synRemap", backend.getArrayPrefix() + "synRemap");
        }
    }
    else if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        addWeightSharingPointerField(gen, "uint32_t", "gp", backend.getArrayPrefix() + "gp");
    }
    else if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) {
        gen.addEGPs(getArchetype().getConnectivityInitialiser().getSnippet()->getExtraGlobalParams(),
                    backend.getArrayPrefix());
    }

    // If WU variables are procedural and this is an update or WU variables are individual
    const auto vars = wum->getVars();
    const auto &varInit = getArchetype().getWUVarInitialisers();
    const bool proceduralWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL);
    const bool individualWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);
    if((proceduralWeights && updateRole) || individualWeights) {
        // If we're performing a procedural update or we're initializing individual variables
        if((proceduralWeights && updateRole) || !updateRole) {
            // Add heterogeneous variable initialization parameters and derived parameters
            gen.addHeterogeneousVarInitParams(wum->getVars(), &SynapseGroupInternal::getWUVarInitialisers,
                                              &SynapseGroupMergedBase::isWUVarInitParamHeterogeneous);

            gen.addHeterogeneousVarInitDerivedParams(wum->getVars(), &SynapseGroupInternal::getWUVarInitialisers,
                                                     &SynapseGroupMergedBase::isWUVarInitDerivedParamHeterogeneous);
        }

        // Loop through variables
        for(size_t v = 0; v < vars.size(); v++) {
            // If we're updating or if there is initialization code for this variable 
            // (otherwise, it's not needed during initialization)
            const auto var = vars[v];
            if(individualWeights && (updateRole || !varInit.at(v).getSnippet()->getCode().empty())) {
                addWeightSharingPointerField(gen, var.type, var.name, backend.getArrayPrefix() + var.name);
            }

            // If we're performing a procedural update or we're initializing, add any var init EGPs to structure
            if((proceduralWeights && updateRole) || !updateRole) {
                const auto egps = varInit.at(v).getSnippet()->getExtraGlobalParams();
                for(const auto &e : egps) {
                    const bool isPointer = Utils::isTypePointer(e.type);
                    const std::string prefix = isPointer ? backend.getArrayPrefix() : "";
                    gen.addField(e.type, e.name + var.name,
                                 [e, prefix, var](const SynapseGroupInternal &sg, size_t) 
                                 {
                                     if(sg.isWeightSharingSlave()) {
                                         return prefix + e.name + var.name + sg.getWeightSharingMaster()->getName();
                                     }
                                     else {
                                         return prefix + e.name + var.name + sg.getName();
                                     }
                                
                                 },
                                 isPointer ? decltype(gen)::FieldType::PointerEGP : decltype(gen)::FieldType::ScalarEGP);
                }
            }
        }
    }
    // Otherwise, if WU variables are global and this is an update kernel
    // **NOTE** global variable values aren't useful during initialization
    else if(getArchetype().getMatrixType() & SynapseMatrixWeight::GLOBAL && updateRole) {
        for(size_t v = 0; v < vars.size(); v++) {
            // If variable should be implemented heterogeneously, add scalar field
            if(isWUGlobalVarHeterogeneous(v)) {
                gen.addScalarField(vars[v].name,
                                   [v](const SynapseGroupInternal &sg, size_t)
                                   {
                                       return Utils::writePreciseString(sg.getWUConstInitVals().at(v));
                                   });
            }
        }
    }

    // Generate structure definitions and instantiation
    gen.generate(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                 mergedStructData, name);
}
//----------------------------------------------------------------------------
void CodeGenerator::SynapseGroupMergedBase::addPSPointerField(MergedStructGenerator<SynapseGroupMergedBase> &gen,
                                                              const std::string &type, const std::string &name, const std::string &prefix) const
{
    assert(!Utils::isTypePointer(type));
    gen.addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getPSModelTargetName(); });
}
//----------------------------------------------------------------------------
void CodeGenerator::SynapseGroupMergedBase::addSrcPointerField(MergedStructGenerator<SynapseGroupMergedBase> &gen,
                                                               const std::string &type, const std::string &name, const std::string &prefix) const
{
    assert(!Utils::isTypePointer(type));
    gen.addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getSrcNeuronGroup()->getName(); });
}
//----------------------------------------------------------------------------
void CodeGenerator::SynapseGroupMergedBase::addTrgPointerField(MergedStructGenerator<SynapseGroupMergedBase> &gen,
                                                               const std::string &type, const std::string &name, const std::string &prefix) const
{
    assert(!Utils::isTypePointer(type));
    gen.addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t) { return prefix + sg.getTrgNeuronGroup()->getName(); });
}
//----------------------------------------------------------------------------
void CodeGenerator::SynapseGroupMergedBase::addWeightSharingPointerField(MergedStructGenerator<SynapseGroupMergedBase> &gen,
                                                                         const std::string &type, const std::string &name, const std::string &prefix) const
{
    assert(!Utils::isTypePointer(type));
    gen.addField(type + "*", name, 
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
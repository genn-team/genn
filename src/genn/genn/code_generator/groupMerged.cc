#include "code_generator/groupMerged.h"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "modelSpecInternal.h"

//----------------------------------------------------------------------------
// CodeGenerator::NeuronGroupMerged
//----------------------------------------------------------------------------
const SynapseGroupInternal *CodeGenerator::NeuronGroupMerged::getCompatibleMergedInSyn(size_t archetypeMergedInSyn, const NeuronGroupInternal &ng) const
{
    const SynapseGroupInternal *archetypeSG = getArchetype().getMergedInSyn()[archetypeMergedInSyn].first;
    const auto otherSyn = std::find_if(ng.getMergedInSyn().cbegin(), ng.getMergedInSyn().cend(),
                                       [archetypeSG](const std::pair<SynapseGroupInternal*, std::vector<SynapseGroupInternal*>> &m)
                                       {
                                           return m.first->canPSBeMerged(*archetypeSG);
                                       });
    assert(otherSyn != ng.getMergedInSyn().cend());
    return otherSyn->first;
}
//----------------------------------------------------------------------------
const SynapseGroupInternal *CodeGenerator::NeuronGroupMerged::getCompatibleInitMergedInSyn(size_t archetypeMergedInSyn, const NeuronGroupInternal &ng) const
{
    const SynapseGroupInternal *archetypeSG = getArchetype().getMergedInSyn()[archetypeMergedInSyn].first;
    const auto otherSyn = std::find_if(ng.getMergedInSyn().cbegin(), ng.getMergedInSyn().cend(),
                                       [archetypeSG](const std::pair<SynapseGroupInternal*, std::vector<SynapseGroupInternal*>> &m)
                                       {
                                           return m.first->canPSInitBeMerged(*archetypeSG);
                                       });
    assert(otherSyn != ng.getMergedInSyn().cend());
    return otherSyn->first;
}
//----------------------------------------------------------------------------
const CurrentSourceInternal *CodeGenerator::NeuronGroupMerged::getCompatibleCurrentSource(size_t archetypeCurrentSource, const NeuronGroupInternal &ng) const
{
    const CurrentSourceInternal *archetypeCS = getArchetype().getCurrentSources()[archetypeCurrentSource];
    const auto otherCS = std::find_if(ng.getCurrentSources().cbegin(), ng.getCurrentSources().cend(),
                                      [archetypeCS](const CurrentSourceInternal *m)
                                      {
                                          return m->canBeMerged(*archetypeCS);
                                      });
    assert(otherCS != ng.getCurrentSources().cend());
    return *otherCS;
}
//----------------------------------------------------------------------------
const CurrentSourceInternal *CodeGenerator::NeuronGroupMerged::getCompatibleInitCurrentSource(size_t archetypeCurrentSource, const NeuronGroupInternal &ng) const
{
    const CurrentSourceInternal *archetypeCS = getArchetype().getCurrentSources()[archetypeCurrentSource];
    const auto otherCS = std::find_if(ng.getCurrentSources().cbegin(), ng.getCurrentSources().cend(),
                                      [archetypeCS](const CurrentSourceInternal *m)
                                      {
                                          return m->canInitBeMerged(*archetypeCS);
                                      });
    assert(otherCS != ng.getCurrentSources().cend());
    return *otherCS;
}

//----------------------------------------------------------------------------
// CodeGenerator::SynapseGroupMerged
//----------------------------------------------------------------------------
std::string CodeGenerator::SynapseGroupMerged::getPresynapticAxonalDelaySlot() const
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
std::string CodeGenerator::SynapseGroupMerged::getPostsynapticBackPropDelaySlot() const
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
std::string CodeGenerator::SynapseGroupMerged::getDendriticDelayOffset(const std::string &offset) const
{
    assert(getArchetype().isDendriticDelayRequired());

    if(offset.empty()) {
        return "(*group.denDelayPtr * group.numTrgNeurons) + ";
    }
    else {
        return "(((*group.denDelayPtr + " + offset + ") % " + std::to_string(getArchetype().getMaxDendriticDelayTimesteps()) + ") * group.numTrgNeurons) + ";
    }
}

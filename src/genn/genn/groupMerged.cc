#include "groupMerged.h"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "modelSpecInternal.h"

//----------------------------------------------------------------------------
// NeuronGroupMerged
//----------------------------------------------------------------------------
const SynapseGroupInternal *NeuronGroupMerged::getCompatibleMergedInSyn(size_t archetypeMergedInSyn, const NeuronGroupInternal &ng) const
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
// SynapseGroupMerged
//----------------------------------------------------------------------------
std::string SynapseGroupMerged::getPresynapticAxonalDelaySlot() const
{
    assert(getArchetype().getSrcNeuronGroup()->isDelayRequired());

    const unsigned int numDelaySteps = getArchetype().getDelaySteps();
    if(numDelaySteps == 0) {
        return "(*synapseGroup.srcSpkQuePtr)";
    }
    else {
        const unsigned int numSrcDelaySlots = getArchetype().getSrcNeuronGroup()->getNumDelaySlots();
        return "((*synapseGroup.srcSpkQuePtr + " + std::to_string(numSrcDelaySlots - numDelaySteps) + ") % " + std::to_string(numSrcDelaySlots) + ")";
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMerged::getPostsynapticBackPropDelaySlot() const
{
    assert(getArchetype().getTrgNeuronGroup()->isDelayRequired());

    const unsigned int numBackPropDelaySteps = getArchetype().getBackPropDelaySteps();
    if(numBackPropDelaySteps == 0) {
        return "(*synapseGroup.trgSpkQuePtr)";
    }
    else {
        const unsigned int numTrgDelaySlots = getArchetype().getTrgNeuronGroup()->getNumDelaySlots();
        return "((*synapseGroup.trgSpkQuePtr + " + std::to_string(numTrgDelaySlots - numBackPropDelaySteps) + ") % " + std::to_string(numTrgDelaySlots) + ")";
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMerged::getDendriticDelayOffset(const std::string &offset) const
{
    assert(getArchetype().isDendriticDelayRequired());

    if(offset.empty()) {
        return "(*synapseGroup.denDelayPtr * synapseGroup.numTrgNeurons) + ";
    }
    else {
        return "(((*synapseGroup.denDelayPtr + " + offset + ") % " + std::to_string(getArchetype().getMaxDendriticDelayTimesteps()) + ") * synapseGroup.numTrgNeurons) + ";
    }
}

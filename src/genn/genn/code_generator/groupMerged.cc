#include "code_generator/groupMerged.h"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "modelSpecInternal.h"

//----------------------------------------------------------------------------
// CodeGenerator::NeuronGroupMerged
//----------------------------------------------------------------------------
std::string CodeGenerator::NeuronGroupMerged::getCurrentQueueOffset() const
{
    assert(getArchetype().isDelayRequired());
    return "(*group.spkQuePtr * group.numNeurons)";
}
//----------------------------------------------------------------------------
std::string CodeGenerator::NeuronGroupMerged::getPrevQueueOffset() const
{
    assert(getArchetype().isDelayRequired());
    return "(((*group.spkQuePtr + " + std::to_string(getArchetype().getNumDelaySlots() - 1) + ") % " + std::to_string(getArchetype().getNumDelaySlots()) + ") * group.numNeurons)";
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

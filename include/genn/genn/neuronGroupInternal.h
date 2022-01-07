#pragma once

// GeNN includes
#include "neuronGroup.h"

//------------------------------------------------------------------------
// NeuronGroupInternal
//------------------------------------------------------------------------
class NeuronGroupInternal : public NeuronGroup
{
public:
    NeuronGroupInternal(const std::string &name, int numNeurons, const NeuronModels::Base *neuronModel,
                        const std::unordered_map<std::string, double> &params, const std::unordered_map<std::string, Models::VarInit> &varInitialisers,
                        VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   NeuronGroup(name, numNeurons, neuronModel, params, varInitialisers,
                    defaultVarLocation, defaultExtraGlobalParamLocation)
    {
    }
    
    using NeuronGroup::checkNumDelaySlots;
    using NeuronGroup::updatePreVarQueues;
    using NeuronGroup::updatePostVarQueues;
    using NeuronGroup::addSpkEventCondition;
    using NeuronGroup::addInSyn;
    using NeuronGroup::addOutSyn;
    using NeuronGroup::initDerivedParams;
    using NeuronGroup::fusePrePostSynapses;
    using NeuronGroup::injectCurrent;
    using NeuronGroup::getFusedPSMInSyn;
    using NeuronGroup::getFusedWUPostInSyn;
    using NeuronGroup::getFusedPreOutputOutSyn;
    using NeuronGroup::getFusedWUPreOutSyn;
    using NeuronGroup::getOutSyn;
    using NeuronGroup::getCurrentSources;
    using NeuronGroup::getDerivedParams;
    using NeuronGroup::getSpikeEventCondition;
    using NeuronGroup::getFusedInSynWithPostCode;
    using NeuronGroup::getFusedOutSynWithPreCode;
    using NeuronGroup::getFusedInSynWithPostVars;
    using NeuronGroup::getFusedOutSynWithPreVars;
    using NeuronGroup::isVarQueueRequired;
    using NeuronGroup::getHashDigest;
    using NeuronGroup::getInitHashDigest;
    using NeuronGroup::getSpikeQueueUpdateHashDigest;
    using NeuronGroup::getPrevSpikeTimeUpdateHashDigest;
    using NeuronGroup::getVarLocationHashDigest;
};

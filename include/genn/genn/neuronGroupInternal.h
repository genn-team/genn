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
                        const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
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
    using NeuronGroup::mergePrePostSynapses;
    using NeuronGroup::injectCurrent;
    using NeuronGroup::getMergedPSMInSyn;
    using NeuronGroup::getMergedWUPostInSyn;
    using NeuronGroup::getMergedWUPreOutSyn;
    using NeuronGroup::getOutSyn;
    using NeuronGroup::getCurrentSources;
    using NeuronGroup::getDerivedParams;
    using NeuronGroup::getSpikeEventCondition;
    using NeuronGroup::getInSynWithPostCode;
    using NeuronGroup::getOutSynWithPreCode;
    using NeuronGroup::getInSynWithPostVars;
    using NeuronGroup::getOutSynWithPreVars;
    using NeuronGroup::isVarQueueRequired;
    using NeuronGroup::getHashDigest;
    using NeuronGroup::getInitHashDigest;
    using NeuronGroup::getSpikeQueueUpdateHashDigest;
    using NeuronGroup::getPrevSpikeTimeUpdateHashDigest;
    using NeuronGroup::getVarLocationHashDigest;
};

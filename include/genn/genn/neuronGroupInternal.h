#pragma once

// GeNN includes
#include "neuronGroup.h"

//------------------------------------------------------------------------
// GeNN::NeuronGroupInternal
//------------------------------------------------------------------------
namespace GeNN
{
class NeuronGroupInternal : public NeuronGroup
{
public:
    using GroupExternal = NeuronGroup;

    NeuronGroupInternal(const std::string &name, int numNeurons, const NeuronModels::Base *neuronModel,
                        const std::map<std::string, Type::NumericValue> &params, const std::map<std::string, InitVarSnippet::Init> &varInitialisers,
                        VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   NeuronGroup(name, numNeurons, neuronModel, params, varInitialisers,
                    defaultVarLocation, defaultExtraGlobalParamLocation)
    {
    }
    
    using NeuronGroup::checkNumDelaySlots;
    using NeuronGroup::setVarQueueRequired;
    using NeuronGroup::setSpikeQueueRequired;
    using NeuronGroup::setSpikeEventQueueRequired;
    using NeuronGroup::addInSyn;
    using NeuronGroup::addOutSyn;
    using NeuronGroup::finalise;
    using NeuronGroup::fusePrePostSynapses;
    using NeuronGroup::injectCurrent;
    using NeuronGroup::getFusedPSMInSyn;
    using NeuronGroup::getFusedWUPostInSyn;
    using NeuronGroup::getFusedPreOutputOutSyn;
    using NeuronGroup::getFusedWUPreOutSyn;
    using NeuronGroup::getFusedSpike;
    using NeuronGroup::getFusedSpikeEvent;
    using NeuronGroup::getOutSyn;
    using NeuronGroup::getCurrentSources;
    using NeuronGroup::getDerivedParams;
    using NeuronGroup::getFusedInSynWithPostCode;
    using NeuronGroup::getFusedOutSynWithPreCode;
    using NeuronGroup::getFusedInSynWithPostVars;
    using NeuronGroup::getFusedOutSynWithPreVars;
    using NeuronGroup::getSimCodeTokens;
    using NeuronGroup::getThresholdConditionCodeTokens;
    using NeuronGroup::getResetCodeTokens;
    using NeuronGroup::isSimRNGRequired;
    using NeuronGroup::isInitRNGRequired;
    using NeuronGroup::isRecordingEnabled;
    using NeuronGroup::isVarInitRequired;
    using NeuronGroup::isVarQueueRequired;
    using NeuronGroup::isSpikeQueueRequired;
    using NeuronGroup::isSpikeEventQueueRequired;
    using NeuronGroup::isSpikeDelayRequired;
    using NeuronGroup::isSpikeEventDelayRequired;
    using NeuronGroup::getHashDigest;
    using NeuronGroup::getInitHashDigest;
    using NeuronGroup::getSpikeQueueUpdateHashDigest;
    using NeuronGroup::getPrevSpikeTimeUpdateHashDigest;
    using NeuronGroup::getVarLocationHashDigest;
};

//----------------------------------------------------------------------------
// NeuronVarAdapter
//----------------------------------------------------------------------------
class NeuronVarAdapter
{
public:
    NeuronVarAdapter(const NeuronGroupInternal &ng) : m_NG(ng)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_NG.getVarLocation(varName); }
    
    auto getDefs() const{ return m_NG.getModel()->getVars(); }

    const auto &getInitialisers() const{ return m_NG.getVarInitialisers(); }

    std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const
    { 
        if(m_NG.isDelayRequired() && m_NG.isVarQueueRequired(varName)) {
            return m_NG.getNumDelaySlots();
        }
        else {
            return std::nullopt; 
        }
    }

    const NeuronGroup &getTarget() const{ return m_NG; }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return getVarAccessDim(var.access); }
    
private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const NeuronGroupInternal &m_NG;
};

//----------------------------------------------------------------------------
// NeuronEGPAdapter
//----------------------------------------------------------------------------
class NeuronEGPAdapter
{
public:
    NeuronEGPAdapter(const NeuronGroupInternal &ng) : m_NG(ng)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_NG.getExtraGlobalParamLocation(varName); }

    auto getDefs() const{ return m_NG.getModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const NeuronGroupInternal &m_NG;
};
}   // namespace GeNN

#pragma once

// GeNN includes
#include "neuronGroupInternal.h"
#include "synapseGroup.h"

//------------------------------------------------------------------------
// GeNN::SynapseGroupInternal
//------------------------------------------------------------------------
namespace GeNN
{
class SynapseGroupInternal : public SynapseGroup
{
public:
    SynapseGroupInternal(const std::string &name, SynapseMatrixType matrixType, unsigned int delaySteps,
                         const WeightUpdateModels::Base *wu, const std::unordered_map<std::string, double> &wuParams, const std::unordered_map<std::string, Models::VarInit> &wuVarInitialisers, const std::unordered_map<std::string, Models::VarInit> &wuPreVarInitialisers, const std::unordered_map<std::string, Models::VarInit> &wuPostVarInitialisers,
                         const PostsynapticModels::Base *ps, const std::unordered_map<std::string, double> &psParams, const std::unordered_map<std::string, Models::VarInit> &psVarInitialisers,
                         NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                         const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                         const InitToeplitzConnectivitySnippet::Init &toeplitzConnectivityInitialiser,
                         VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                         VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled)
    :   SynapseGroup(name, matrixType, delaySteps, wu, wuParams, wuVarInitialisers, wuPreVarInitialisers, wuPostVarInitialisers,
                     ps, psParams, psVarInitialisers, srcNeuronGroup, trgNeuronGroup,
                     connectivityInitialiser, toeplitzConnectivityInitialiser, defaultVarLocation, defaultExtraGlobalParamLocation,
                     defaultSparseConnectivityLocation, defaultNarrowSparseIndEnabled)
    {
        // Add references to target and source neuron groups
        trgNeuronGroup->addInSyn(this);
        srcNeuronGroup->addOutSyn(this);
    }

    using SynapseGroup::getSrcNeuronGroup;
    using SynapseGroup::getTrgNeuronGroup;
    using SynapseGroup::getWeightSharingMaster;
    using SynapseGroup::getWUDerivedParams;
    using SynapseGroup::getPSDerivedParams;
    using SynapseGroup::setEventThresholdReTestRequired;
    using SynapseGroup::setFusedPSVarSuffix;
    using SynapseGroup::setFusedPreOutputSuffix;
    using SynapseGroup::setFusedWUPreVarSuffix;
    using SynapseGroup::setFusedWUPostVarSuffix;
    using SynapseGroup::initDerivedParams;
    using SynapseGroup::addCustomUpdateReference;
    using SynapseGroup::isEventThresholdReTestRequired;
    using SynapseGroup::getFusedPSVarSuffix;
    using SynapseGroup::getFusedPreOutputSuffix;
    using SynapseGroup::getFusedWUPreVarSuffix;
    using SynapseGroup::getFusedWUPostVarSuffix;
    using SynapseGroup::getSparseIndType;
    using SynapseGroup::getCustomConnectivityUpdateReferences;
    using SynapseGroup::getCustomUpdateReferences;
    using SynapseGroup::canPSBeFused;
    using SynapseGroup::canWUMPreUpdateBeFused;
    using SynapseGroup::canWUMPostUpdateBeFused;
    using SynapseGroup::canPreOutputBeFused;
    using SynapseGroup::isPSModelFused;
    using SynapseGroup::isWUPreModelFused;
    using SynapseGroup::isWUPostModelFused;
    using SynapseGroup::getWUHashDigest;
    using SynapseGroup::getWUPreHashDigest;
    using SynapseGroup::getWUPostHashDigest;
    using SynapseGroup::getPSHashDigest;
    using SynapseGroup::getPSFuseHashDigest;
    using SynapseGroup::getPreOutputHashDigest;
    using SynapseGroup::getWUPreFuseHashDigest;
    using SynapseGroup::getWUPostFuseHashDigest;
    using SynapseGroup::getDendriticDelayUpdateHashDigest;
    using SynapseGroup::getWUInitHashDigest;
    using SynapseGroup::getWUPreInitHashDigest;
    using SynapseGroup::getWUPostInitHashDigest;
    using SynapseGroup::getPSInitHashDigest;
    using SynapseGroup::getPreOutputInitHashDigest;
    using SynapseGroup::getConnectivityInitHashDigest;
    using SynapseGroup::getConnectivityHostInitHashDigest;
    using SynapseGroup::getVarLocationHashDigest;
};


//----------------------------------------------------------------------------
// SynapsePSMVarAdapter
//----------------------------------------------------------------------------
class SynapsePSMVarAdapter
{
public:
    SynapsePSMVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_SG.getPSVarLocation(varName); }

    Models::Base::VarVec getDefs() const{ return m_SG.getPSModel()->getVars(); }

    const std::unordered_map<std::string, Models::VarInit> &getInitialisers() const{ return m_SG.getPSVarInitialisers(); }

    const std::string &getNameSuffix() const{ return m_SG.getFusedPSVarSuffix(); }

    bool isVarDelayed(const std::string &) const { return false; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUVarAdapter
//----------------------------------------------------------------------------
class SynapseWUVarAdapter
{
public:
    SynapseWUVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_SG.getWUVarLocation(varName); }
    
    Models::Base::VarVec getDefs() const{ return m_SG.getWUModel()->getVars(); }

    const std::unordered_map<std::string, Models::VarInit> &getInitialisers() const{ return m_SG.getWUVarInitialisers(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPreVarAdapter
//----------------------------------------------------------------------------
class SynapseWUPreVarAdapter
{
public:
    SynapseWUPreVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_SG.getWUPreVarLocation(varName); }

    Models::Base::VarVec getDefs() const{ return m_SG.getWUModel()->getPreVars(); }

    const std::unordered_map<std::string, Models::VarInit> &getInitialisers() const{ return m_SG.getWUPreVarInitialisers(); }

    const std::string &getNameSuffix() const{ return m_SG.getFusedWUPreVarSuffix(); }

    bool isVarDelayed(const std::string&) const{ return (m_SG.getDelaySteps() != 0); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPostVarAdapter
//----------------------------------------------------------------------------
class SynapseWUPostVarAdapter
{
public:
    SynapseWUPostVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_SG.getWUPostVarLocation(varName); }

    Models::Base::VarVec getDefs() const{ return m_SG.getWUModel()->getPostVars(); }

    const std::unordered_map<std::string, Models::VarInit> &getInitialisers() const{ return m_SG.getWUPostVarInitialisers(); }

    const std::string &getNameSuffix() const{ return m_SG.getFusedWUPostVarSuffix(); }

    bool isVarDelayed(const std::string&) const{ return (m_SG.getBackPropDelaySteps() != 0); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};


//----------------------------------------------------------------------------
// SynapseWUEGPAdapter
//----------------------------------------------------------------------------
class SynapseWUEGPAdapter
{
public:
    SynapseWUEGPAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_SG.getWUExtraGlobalParamLocation(varName); }
    
    Snippet::Base::EGPVec getDefs() const{ return m_SG.getWUModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};
}   // namespace GeNN

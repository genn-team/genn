#pragma once

// GeNN includes
#include "neuronGroupInternal.h"
#include "synapseGroup.h"

//------------------------------------------------------------------------
// SynapseGroupInternal
//------------------------------------------------------------------------
class SynapseGroupInternal : public SynapseGroup
{
public:
    SynapseGroupInternal(const std::string &name, const SynapseGroupInternal *weightSharingMaster, SynapseMatrixType matrixType, unsigned int delaySteps,
                         const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<Models::VarInit> &wuVarInitialisers, const std::vector<Models::VarInit> &wuPreVarInitialisers, const std::vector<Models::VarInit> &wuPostVarInitialisers,
                         const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<Models::VarInit> &psVarInitialisers,
                         NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                         const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                         const InitToeplitzConnectivitySnippet::Init &toeplitzConnectivityInitialiser,
                         VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                         VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled)
    :   SynapseGroup(name, matrixType, delaySteps, wu, wuParams, wuVarInitialisers, wuPreVarInitialisers, wuPostVarInitialisers,
                     ps, psParams, psVarInitialisers, srcNeuronGroup, trgNeuronGroup, weightSharingMaster,
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
    VarLocation getVarLocation(const std::string &varName) const{ return m_SG.getPSVarLocation(varName); }

    VarLocation getVarLocation(size_t index) const{ return m_SG.getPSVarLocation(index); }
    
    Models::Base::VarVec getVars() const{ return m_SG.getPSModel()->getVars(); }

    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_SG.getPSVarInitialisers(); }

    const std::string &getFusedVarSuffix() const{ return m_SG.getFusedPSVarSuffix(); }

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
    VarLocation getVarLocation(const std::string &varName) const{ return m_SG.getWUVarLocation(varName); }

    VarLocation getVarLocation(size_t index) const{ return m_SG.getWUVarLocation(index); }
    
    Models::Base::VarVec getVars() const{ return m_SG.getWUModel()->getVars(); }

    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_SG.getWUVarInitialisers(); }

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
    VarLocation getVarLocation(const std::string &varName) const{ return m_SG.getWUPreVarLocation(varName); }

    VarLocation getVarLocation(size_t index) const{ return m_SG.getWUPreVarLocation(index); }
    
    Models::Base::VarVec getVars() const{ return m_SG.getWUModel()->getPreVars(); }

    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_SG.getWUPreVarInitialisers(); }

    const std::string &getFusedVarSuffix() const{ return m_SG.getFusedWUPreVarSuffix(); }

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
    VarLocation getVarLocation(const std::string &varName) const{ return m_SG.getWUPostVarLocation(varName); }

    VarLocation getVarLocation(size_t index) const{ return m_SG.getWUPostVarLocation(index); }
    
    Models::Base::VarVec getVars() const{ return m_SG.getWUModel()->getPostVars(); }

    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_SG.getWUPostVarInitialisers(); }

    const std::string &getFusedVarSuffix() const{ return m_SG.getFusedWUPostVarSuffix(); }

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
    VarLocation getEGPLocation(const std::string &varName) const{ return m_SG.getWUExtraGlobalParamLocation(varName); }

    VarLocation getEGPLocation(size_t index) const{ return m_SG.getWUExtraGlobalParamLocation(index); }
    
    Snippet::Base::EGPVec getEGPs() const{ return m_SG.getWUModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

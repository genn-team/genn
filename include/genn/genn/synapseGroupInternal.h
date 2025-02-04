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
    using GroupExternal = SynapseGroup;

    SynapseGroupInternal(const std::string &name, SynapseMatrixType matrixType,
                         const WeightUpdateModels::Init &wumInitialiser, const PostsynapticModels::Init &psmInitialiser,
                         NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                         const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                         const InitToeplitzConnectivitySnippet::Init &toeplitzConnectivityInitialiser,
                         VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                         VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled)
    :   SynapseGroup(name, matrixType, wumInitialiser, psmInitialiser,
                     srcNeuronGroup, trgNeuronGroup, connectivityInitialiser, 
                     toeplitzConnectivityInitialiser, defaultVarLocation, defaultExtraGlobalParamLocation,
                     defaultSparseConnectivityLocation, defaultNarrowSparseIndEnabled)
    {
        // Add references to target and source neuron groups
        trgNeuronGroup->addInSyn(this);
        srcNeuronGroup->addOutSyn(this);
    }

    using SynapseGroup::getSrcNeuronGroup;
    using SynapseGroup::getTrgNeuronGroup;
    using SynapseGroup::setFusedPSTarget;
    using SynapseGroup::setFusedSpikeTarget;
    using SynapseGroup::setFusedSpikeEventTarget;
    using SynapseGroup::setFusedPreOutputTarget;
    using SynapseGroup::setFusedWUPrePostTarget;
    using SynapseGroup::finalise;
    using SynapseGroup::addCustomUpdateReference;
    using SynapseGroup::getFusedPSTarget;
    using SynapseGroup::getFusedSpikeTarget;
    using SynapseGroup::getFusedSpikeEventTarget;
    using SynapseGroup::getFusedPreOutputTarget;
    using SynapseGroup::getFusedWUPreTarget;
    using SynapseGroup::getFusedWUPostTarget;
    using SynapseGroup::getSparseIndType;
    using SynapseGroup::getCustomConnectivityUpdateReferences;
    using SynapseGroup::getCustomUpdateReferences;
    using SynapseGroup::canPSBeFused;
    using SynapseGroup::canSpikeBeFused;
    using SynapseGroup::canWUMPrePostUpdateBeFused;
    using SynapseGroup::canWUSpikeEventBeFused;
    using SynapseGroup::canPreOutputBeFused;
    using SynapseGroup::isPSModelFused;
    using SynapseGroup::isPreSpikeFused;
    using SynapseGroup::isWUPreModelFused;
    using SynapseGroup::isWUPostModelFused;
    using SynapseGroup::isDendriticOutputDelayRequired;
    using SynapseGroup::isWUPostVarHeterogeneouslyDelayed;
    using SynapseGroup::areAnyWUPostVarHeterogeneouslyDelayed;
    using SynapseGroup::isPresynapticOutputRequired; 
    using SynapseGroup::isPostsynapticOutputRequired;
    using SynapseGroup::isProceduralConnectivityRNGRequired;
    using SynapseGroup::isWUInitRNGRequired;
    using SynapseGroup::isPSVarInitRequired;
    using SynapseGroup::isWUVarInitRequired;
    using SynapseGroup::isWUPreVarInitRequired;
    using SynapseGroup::isWUPostVarInitRequired;
    using SynapseGroup::isSparseConnectivityInitRequired;
    using SynapseGroup::getWUHashDigest;
    using SynapseGroup::getWUPrePostHashDigest;
    using SynapseGroup::getWUSpikeEventHashDigest;
    using SynapseGroup::getPSHashDigest;
    using SynapseGroup::getPSFuseHashDigest;
    using SynapseGroup::getSpikeHashDigest;
    using SynapseGroup::getPreOutputHashDigest;
    using SynapseGroup::getWUPrePostFuseHashDigest;
    using SynapseGroup::getWUSpikeEventFuseHashDigest;
    using SynapseGroup::getDendriticDelayUpdateHashDigest;
    using SynapseGroup::getWUInitHashDigest;
    using SynapseGroup::getWUPrePostInitHashDigest;
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

    auto getDefs() const{ return m_SG.getPSInitialiser().getSnippet()->getVars(); }

    const auto &getInitialisers() const{ return m_SG.getPSInitialiser().getVarInitialisers(); }

    const SynapseGroup &getTarget() const{ return m_SG.getFusedPSTarget(); }

    std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const{ return std::nullopt; }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return getVarAccessDim(var.access); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapsePSMEGPAdapter
//----------------------------------------------------------------------------
class SynapsePSMEGPAdapter
{
public:
    SynapsePSMEGPAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_SG.getPSExtraGlobalParamLocation(varName); }
    
    auto getDefs() const{ return m_SG.getPSInitialiser().getSnippet()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapsePSMNeuronVarRefAdapter
//----------------------------------------------------------------------------
class SynapsePSMNeuronVarRefAdapter
{
public:
    SynapsePSMNeuronVarRefAdapter(const SynapseGroupInternal &cs) : m_SG(cs)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    auto getDefs() const{ return m_SG.getPSInitialiser().getSnippet()->getNeuronVarRefs(); }

    const auto &getInitialisers() const{ return m_SG.getPSInitialiser().getNeuronVarReferences(); }

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
    
    auto getDefs() const{ return m_SG.getWUInitialiser().getSnippet()->getVars(); }

    const auto &getInitialisers() const{ return m_SG.getWUInitialiser().getVarInitialisers(); }

    const SynapseGroup &getTarget() const{ return m_SG; }

    std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const{ return std::nullopt; }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return getVarAccessDim(var.access); }

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

    auto getDefs() const{ return m_SG.getWUInitialiser().getSnippet()->getPreVars(); }

    const auto &getInitialisers() const{ return m_SG.getWUInitialiser().getPreVarInitialisers(); }

    const SynapseGroup &getTarget() const{ return m_SG.getFusedWUPreTarget(); }

    std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const
    {
        if(m_SG.getAxonalDelaySteps() != 0) {
            return m_SG.getSrcNeuronGroup()->getNumDelaySlots();
        }
        else {
            return std::nullopt;
        }
    }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return getVarAccessDim(var.access); }

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

    auto getDefs() const{ return m_SG.getWUInitialiser().getSnippet()->getPostVars(); }

    const auto &getInitialisers() const{ return m_SG.getWUInitialiser().getPostVarInitialisers(); }

    const SynapseGroup &getTarget() const{ return m_SG.getFusedWUPostTarget(); }

    std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const
    {
        if(m_SG.getBackPropDelaySteps() != 0 || m_SG.isWUPostVarHeterogeneouslyDelayed(varName)) {
            return m_SG.getTrgNeuronGroup()->getNumDelaySlots();
        }
        else {
            return std::nullopt;
        }
    }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return getVarAccessDim(var.access); }

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
    
    auto getDefs() const{ return m_SG.getWUInitialiser().getSnippet()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPreNeuronVarRefAdapter
//----------------------------------------------------------------------------
class SynapseWUPreNeuronVarRefAdapter
{
public:
    SynapseWUPreNeuronVarRefAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    auto getDefs() const{ return m_SG.getWUInitialiser().getSnippet()->getPreNeuronVarRefs(); }

    const auto &getInitialisers() const{ return m_SG.getWUInitialiser().getPreNeuronVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPostNeuronVarRefAdapter
//----------------------------------------------------------------------------
class SynapseWUPostNeuronVarRefAdapter
{
public:
    SynapseWUPostNeuronVarRefAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    auto getDefs() const{ return m_SG.getWUInitialiser().getSnippet()->getPostNeuronVarRefs(); }

    const auto &getInitialisers() const{ return m_SG.getWUInitialiser().getPostNeuronVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};
}   // namespace GeNN

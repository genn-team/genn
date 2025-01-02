#pragma once

// GeNN includes
#include "adapters.h"
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
class SynapsePSMVarAdapter : public VarAdapter
{
public:
    SynapsePSMVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getPSVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_SG.getPSInitialiser().getSnippet()->getVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_SG.getPSInitialiser().getVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const override final { return std::nullopt; }
    
    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    const SynapseGroup &getTarget() const{ return m_SG.getFusedPSTarget(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapsePSMEGPAdapter
//----------------------------------------------------------------------------
class SynapsePSMEGPAdapter : public EGPAdapter
{
public:
    SynapsePSMEGPAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // EGPAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getPSExtraGlobalParamLocation(varName); }
    
    virtual Snippet::Base::EGPVec getDefs() const override final { return m_SG.getPSInitialiser().getSnippet()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapsePSMNeuronVarRefAdapter
//----------------------------------------------------------------------------
class SynapsePSMNeuronVarRefAdapter : public VarRefAdapter
{
public:
    SynapsePSMNeuronVarRefAdapter(const SynapseGroupInternal &cs) : m_SG(cs)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_SG.getPSInitialiser().getSnippet()->getNeuronVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_SG.getPSInitialiser().getNeuronVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUVarAdapter
//----------------------------------------------------------------------------
class SynapseWUVarAdapter : public VarAdapter
{
public:
    SynapseWUVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getWUVarLocation(varName); }
    
    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_SG.getWUInitialiser().getVarInitialisers(); }
    
    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const{ return std::nullopt; }

    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    // Public API
    const SynapseGroup &getTarget() const{ return m_SG; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPreVarAdapter
//----------------------------------------------------------------------------
class SynapseWUPreVarAdapter : public VarAdapter
{
public:
    SynapseWUPreVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getWUPreVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getPreVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_SG.getWUInitialiser().getPreVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const override final
    {
        if(m_SG.getAxonalDelaySteps() != 0) {
            return m_SG.getSrcNeuronGroup()->getNumDelaySlots();
        }
        else {
            return std::nullopt;
        }
    }
    
    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    const SynapseGroup &getTarget() const{ return m_SG.getFusedWUPreTarget(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPostVarAdapter
//----------------------------------------------------------------------------
class SynapseWUPostVarAdapter : public VarAdapter
{
public:
    SynapseWUPostVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getWUPostVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getPostVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_SG.getWUInitialiser().getPostVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final
    {
        if(m_SG.getBackPropDelaySteps() != 0 || m_SG.isWUPostVarHeterogeneouslyDelayed(varName)) {
            return m_SG.getTrgNeuronGroup()->getNumDelaySlots();
        }
        else {
            return std::nullopt;
        }
    }
    
    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    const SynapseGroup &getTarget() const{ return m_SG.getFusedWUPostTarget(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUEGPAdapter
//----------------------------------------------------------------------------
class SynapseWUEGPAdapter : public EGPAdapter
{
public:
    SynapseWUEGPAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // EGPAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getWUExtraGlobalParamLocation(varName); }
    
    virtual Snippet::Base::EGPVec getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPreNeuronVarRefAdapter
//----------------------------------------------------------------------------
class SynapseWUPreNeuronVarRefAdapter : public VarRefAdapter
{
public:
    SynapseWUPreNeuronVarRefAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getPreNeuronVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_SG.getWUInitialiser().getPreNeuronVarReferences(); }
    
    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final{ throw std::runtime_error("Not implemented"); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPostNeuronVarRefAdapter
//----------------------------------------------------------------------------
class SynapseWUPostNeuronVarRefAdapter : public VarRefAdapter
{
public:
    SynapseWUPostNeuronVarRefAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getPostNeuronVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_SG.getWUInitialiser().getPostNeuronVarReferences(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final{ throw std::runtime_error("Not implemented"); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};
}   // namespace GeNN

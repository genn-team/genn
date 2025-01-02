#pragma once

// GeNN includes
#include "adapters.h"
#include "customConnectivityUpdate.h"
#include "synapseGroupInternal.h"

//------------------------------------------------------------------------
// CustomUpdateInternal
//------------------------------------------------------------------------
namespace GeNN
{
class CustomConnectivityUpdateInternal : public CustomConnectivityUpdate
{
public:
    using GroupExternal = CustomConnectivityUpdate;

    CustomConnectivityUpdateInternal(const std::string &name, const std::string &updateGroupName, SynapseGroupInternal *synapseGroup, 
                                     const CustomConnectivityUpdateModels::Base *customConnectivityUpdateModel, 
                                     const std::map<std::string, Type::NumericValue> &params, const std::map<std::string, InitVarSnippet::Init> &varInitialisers,
                                     const std::map<std::string, InitVarSnippet::Init> &preVarInitialisers, const std::map<std::string, InitVarSnippet::Init> &postVarInitialisers,
                                     const std::map<std::string, Models::WUVarReference> &varReferences, const std::map<std::string, Models::VarReference> &preVarReferences,
                                     const std::map<std::string, Models::VarReference> &postVarReferences, const std::map<std::string, Models::EGPReference> &egpReferences,
                                     VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   CustomConnectivityUpdate(name, updateGroupName, synapseGroup, customConnectivityUpdateModel, params, varInitialisers, preVarInitialisers, postVarInitialisers,
                                 varReferences, preVarReferences, postVarReferences, egpReferences, defaultVarLocation, defaultExtraGlobalParamLocation)
    {
        getSynapseGroup()->addCustomUpdateReference(this);
    }

    using CustomConnectivityUpdate::getDerivedParams;
    using CustomConnectivityUpdate::isZeroCopyEnabled;
    using CustomConnectivityUpdate::getVarLocationHashDigest;
    using CustomConnectivityUpdate::getRowUpdateCodeTokens;
    using CustomConnectivityUpdate::getHostUpdateCodeTokens;
    using CustomConnectivityUpdate::getSynapseGroup;
    using CustomConnectivityUpdate::getDependentVariables;
    using CustomConnectivityUpdate::finalise;
    using CustomConnectivityUpdate::getHashDigest;
    using CustomConnectivityUpdate::getRemapHashDigest;
    using CustomConnectivityUpdate::getInitHashDigest;
    using CustomConnectivityUpdate::getPreDelayNeuronGroup;
    using CustomConnectivityUpdate::getPostDelayNeuronGroup;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdateVarAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdateVarAdapter : public VarAdapter
{
public:
    CustomConnectivityUpdateVarAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CU.getVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_CU.getModel()->getVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_CU.getVarInitialisers(); }
    
    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final { return std::nullopt; }

    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    const auto &getTarget() const{ return m_CU; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePreVarAdapter : public VarAdapter
{
public:
    CustomConnectivityUpdatePreVarAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CU.getPreVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_CU.getModel()->getPreVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_CU.getPreVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const override final { return std::nullopt; }

    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    const auto &getTarget() const{ return m_CU; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePostVarAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePostVarAdapter : public VarAdapter
{
public:
    CustomConnectivityUpdatePostVarAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CU.getPostVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_CU.getModel()->getPostVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_CU.getPostVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const override final { return std::nullopt; }
    
    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    const auto &getTarget() const{ return m_CU; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdateEGPAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdateEGPAdapter : public EGPAdapter
{
public:
    CustomConnectivityUpdateEGPAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // EGPAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CU.getExtraGlobalParamLocation(varName); }

    virtual Snippet::Base::EGPVec getDefs() const override final { return m_CU.getModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdateVarRefAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdateVarRefAdapter : public WUVarRefAdapter
{
public:
    CustomConnectivityUpdateVarRefAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::WUVarReference;

    //----------------------------------------------------------------------------
    // WUVarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_CU.getModel()->getVarRefs(); }

    virtual const std::map<std::string, Models::WUVarReference> &getInitialisers() const override final { return m_CU.getVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarRefAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePreVarRefAdapter : public VarRefAdapter
{
public:
    CustomConnectivityUpdatePreVarRefAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_CU.getModel()->getPreVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_CU.getPreVarReferences(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final{ throw std::runtime_error("Not implemented"); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePostVarRefAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePostVarRefAdapter : public VarRefAdapter
{
public:
    CustomConnectivityUpdatePostVarRefAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_CU.getModel()->getPostVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_CU.getPostVarReferences(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final{ throw std::runtime_error("Not implemented"); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};
}   // namespace GeNN

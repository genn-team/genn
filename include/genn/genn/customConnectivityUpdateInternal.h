#pragma once

// GeNN includes
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
class CustomConnectivityUpdateVarAdapter
{
public:
    CustomConnectivityUpdateVarAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_CU.getVarLocation(varName); }

    auto getDefs() const{ return m_CU.getModel()->getVars(); }

    const auto &getInitialisers() const{ return m_CU.getVarInitialisers(); }

    const auto &getTarget() const{ return m_CU; }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return getVarAccessDim(var.access); }
     
private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePreVarAdapter
{
public:
    CustomConnectivityUpdatePreVarAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_CU.getPreVarLocation(varName); }

    auto getDefs() const{ return m_CU.getModel()->getPreVars(); }

    const auto &getInitialisers() const{ return m_CU.getPreVarInitialisers(); }

    std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const{ return std::nullopt; }

    const auto &getTarget() const{ return m_CU; }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return getVarAccessDim(var.access); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePostVarAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePostVarAdapter
{
public:
    CustomConnectivityUpdatePostVarAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_CU.getPostVarLocation(varName); }

    auto getDefs() const{ return m_CU.getModel()->getPostVars(); }

    const auto &getInitialisers() const{ return m_CU.getPostVarInitialisers(); }

    std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const{ return std::nullopt; }

    const auto &getTarget() const{ return m_CU; }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return getVarAccessDim(var.access); }
    
private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};


//----------------------------------------------------------------------------
// CustomConnectivityUpdateEGPAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdateEGPAdapter
{
public:
    CustomConnectivityUpdateEGPAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_CU.getExtraGlobalParamLocation(varName); }

    Snippet::Base::EGPVec getDefs() const{ return m_CU.getModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdateVarRefAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdateVarRefAdapter
{
public:
    CustomConnectivityUpdateVarRefAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::WUVarReference;

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    auto getDefs() const{ return m_CU.getModel()->getVarRefs(); }

    const auto &getInitialisers() const{ return m_CU.getVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarRefAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePreVarRefAdapter
{
public:
    CustomConnectivityUpdatePreVarRefAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    auto getDefs() const{ return m_CU.getModel()->getPreVarRefs(); }

    const auto &getInitialisers() const{ return m_CU.getPreVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePostVarRefAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePostVarRefAdapter
{
public:
    CustomConnectivityUpdatePostVarRefAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    auto getDefs() const{ return m_CU.getModel()->getPostVarRefs(); }

    const auto &getInitialisers() const{ return m_CU.getPostVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};
}   // namespace GeNN

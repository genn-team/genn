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
                                     const std::unordered_map<std::string, double> &params, const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers,
                                     const std::unordered_map<std::string, InitVarSnippet::Init> &preVarInitialisers, const std::unordered_map<std::string, InitVarSnippet::Init> &postVarInitialisers,
                                     const std::unordered_map<std::string, Models::WUVarReference> &varReferences, const std::unordered_map<std::string, Models::VarReference> &preVarReferences,
                                     const std::unordered_map<std::string, Models::VarReference> &postVarReferences, VarLocation defaultVarLocation,
                                     VarLocation defaultExtraGlobalParamLocation)
    :   CustomConnectivityUpdate(name, updateGroupName, synapseGroup, customConnectivityUpdateModel, params, varInitialisers, preVarInitialisers, postVarInitialisers,
                                 varReferences, preVarReferences, postVarReferences, defaultVarLocation, defaultExtraGlobalParamLocation)
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

    Models::Base::VarVec getDefs() const{ return m_CU.getCustomConnectivityUpdateModel()->getVars(); }

    const std::unordered_map<std::string, InitVarSnippet::Init> &getInitialisers() const{ return m_CU.getVarInitialisers(); }

    const std::string &getNameSuffix() const{ return m_CU.getName(); }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return var.access.getDims<SynapseVarAccess>(); }

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

    Models::Base::VarVec getDefs() const{ return m_CU.getCustomConnectivityUpdateModel()->getPreVars(); }

    const std::unordered_map<std::string, InitVarSnippet::Init> &getInitialisers() const{ return m_CU.getPreVarInitialisers(); }

    bool isVarDelayed(const std::string &) const { return false; }

    const std::string &getNameSuffix() const{ return m_CU.getName(); }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return var.access.getDims<NeuronVarAccess>(); }

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

    Models::Base::VarVec getDefs() const{ return m_CU.getCustomConnectivityUpdateModel()->getPostVars(); }

    const std::unordered_map<std::string, InitVarSnippet::Init> &getInitialisers() const{ return m_CU.getPostVarInitialisers(); }

    bool isVarDelayed(const std::string &) const { return false; }

    const std::string &getNameSuffix() const{ return m_CU.getName(); }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return var.access.getDims<NeuronVarAccess>(); }
    
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
    VarLocation getLoc(const std::string&) const{ return VarLocation::HOST_DEVICE; }

    Snippet::Base::EGPVec getDefs() const{ return m_CU.getCustomConnectivityUpdateModel()->getExtraGlobalParams(); }

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
    Models::Base::VarRefVec getDefs() const{ return m_CU.getCustomConnectivityUpdateModel()->getVarRefs(); }

    const std::unordered_map<std::string, Models::WUVarReference> &getInitialisers() const{ return m_CU.getVarReferences(); }

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
    Models::Base::VarRefVec getDefs() const{ return m_CU.getCustomConnectivityUpdateModel()->getPreVarRefs(); }

    const std::unordered_map<std::string, Models::VarReference> &getInitialisers() const{ return m_CU.getPreVarReferences(); }

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
    Models::Base::VarRefVec getDefs() const{ return m_CU.getCustomConnectivityUpdateModel()->getPostVarRefs(); }

    const std::unordered_map<std::string, Models::VarReference> &getInitialisers() const{ return m_CU.getPostVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};
}   // namespace GeNN

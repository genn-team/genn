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
    CustomConnectivityUpdateInternal(const std::string &name, const std::string &updateGroupName, SynapseGroupInternal *synapseGroup, 
                                     const CustomConnectivityUpdateModels::Base *customConnectivityUpdateModel, 
                                     const std::unordered_map<std::string, double> &params, const std::unordered_map<std::string, Models::VarInit> &varInitialisers,
                                     const std::unordered_map<std::string, Models::VarInit> &preVarInitialisers, const std::unordered_map<std::string, Models::VarInit> &postVarInitialisers,
                                     const std::unordered_map<std::string, Models::WUVarReference> &varReferences, const std::unordered_map<std::string, Models::VarReference> &preVarReferences,
                                     const std::unordered_map<std::string, Models::VarReference> &postVarReferences, VarLocation defaultVarLocation,
                                     VarLocation defaultExtraGlobalParamLocation)
    :   CustomConnectivityUpdate(name, updateGroupName, synapseGroup, customConnectivityUpdateModel, params, varInitialisers, preVarInitialisers, postVarInitialisers,
                                 varReferences, preVarReferences, postVarReferences, defaultVarLocation, defaultExtraGlobalParamLocation)
    {
        getSynapseGroup()->addCustomUpdateReference(this);
    }

    using CustomConnectivityUpdate::initDerivedParams;
    using CustomConnectivityUpdate::getDerivedParams;
    using CustomConnectivityUpdate::isPreVarInitRNGRequired;
    using CustomConnectivityUpdate::isPostVarInitRNGRequired;
    using CustomConnectivityUpdate::isVarInitRNGRequired;
    using CustomConnectivityUpdate::isZeroCopyEnabled;
    using CustomConnectivityUpdate::getVarLocationHashDigest;
    using CustomConnectivityUpdate::getSynapseGroup;
    using CustomConnectivityUpdate::getDependentVariables;
    using CustomConnectivityUpdate::finalize;
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

    const std::unordered_map<std::string, Models::VarInit> &getInitialisers() const{ return m_CU.getVarInitialisers(); }

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

    const std::unordered_map<std::string, Models::VarInit> &getInitialisers() const{ return m_CU.getPreVarInitialisers(); }

    bool isVarDelayed(const std::string &) const { return false; }

    const std::string &getNameSuffix() const{ return m_CU.getName(); }

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

    const std::unordered_map<std::string, Models::VarInit> &getInitialisers() const{ return m_CU.getPostVarInitialisers(); }

    bool isVarDelayed(const std::string &) const { return false; }

    const std::string &getNameSuffix() const{ return m_CU.getName(); }

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
}   // namespace GeNN

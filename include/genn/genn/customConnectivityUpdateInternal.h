#pragma once

// GeNN includes
#include "customConnectivityUpdate.h"

//------------------------------------------------------------------------
// CustomUpdateInternal
//------------------------------------------------------------------------
class CustomConnectivityUpdateInternal : public CustomConnectivityUpdate
{
public:
    CustomConnectivityUpdateInternal(const std::string &name, const std::string &updateGroupName, const SynapseGroupInternal *synapseGroup, 
                                     const CustomConnectivityUpdateModels::Base *customConnectivityUpdateModel, 
                                     const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                                     const std::vector<Models::VarInit> &preVarInitialisers, const std::vector<Models::VarInit> &postVarInitialisers,
                                     const std::vector<Models::WUVarReference> &varReferences, const std::vector<Models::VarReference> &preVarReferences,
                                     const std::vector<Models::VarReference> &postVarReferences, VarLocation defaultVarLocation,
                                     VarLocation defaultExtraGlobalParamLocation)
    :   CustomConnectivityUpdate(name, updateGroupName, synapseGroup, customConnectivityUpdateModel, params, varInitialisers, preVarInitialisers, postVarInitialisers,
                                 varReferences, preVarReferences, postVarReferences, defaultVarLocation, defaultExtraGlobalParamLocation)
    {
    }

    using CustomConnectivityUpdate::initDerivedParams;
    using CustomConnectivityUpdate::getDerivedParams;
    using CustomConnectivityUpdate::isInitRNGRequired;
    using CustomConnectivityUpdate::isZeroCopyEnabled;
    using CustomConnectivityUpdate::getVarLocationHashDigest;
    using CustomConnectivityUpdate::getSynapseGroup;
    using CustomConnectivityUpdate::finalize;
    using CustomConnectivityUpdate::getHashDigest;
    using CustomConnectivityUpdate::getInitHashDigest;
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
    VarLocation getVarLocation(const std::string &varName) const{ return m_CU.getVarLocation(varName); }

    VarLocation getVarLocation(size_t index) const{ return m_CU.getVarLocation(index); }
    
    Models::Base::VarVec getVars() const{ return m_CU.getCustomConnectivityUpdateModel()->getVars(); }

    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_CU.getVarInitialisers(); }

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
    VarLocation getVarLocation(const std::string &varName) const{ return m_CU.getPreVarLocation(varName); }

    VarLocation getVarLocation(size_t index) const{ return m_CU.getPreVarLocation(index); }
    
    Models::Base::VarVec getVars() const{ return m_CU.getCustomConnectivityUpdateModel()->getPreVars(); }

    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_CU.getPreVarInitialisers(); }

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
    VarLocation getVarLocation(const std::string &varName) const{ return m_CU.getPostVarLocation(varName); }

    VarLocation getVarLocation(size_t index) const{ return m_CU.getPostVarLocation(index); }
    
    Models::Base::VarVec getVars() const{ return m_CU.getCustomConnectivityUpdateModel()->getPostVars(); }

    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_CU.getPostVarInitialisers(); }

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
    VarLocation getEGPLocation(const std::string&) const{ return VarLocation::HOST_DEVICE; }

    VarLocation getEGPLocation(size_t) const{ return VarLocation::HOST_DEVICE; }
    
    Snippet::Base::EGPVec getEGPs() const{ return m_CU.getCustomConnectivityUpdateModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};
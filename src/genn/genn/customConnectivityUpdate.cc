#include "customConnectivityUpdate.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "gennUtils.h"
#include "currentSource.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//------------------------------------------------------------------------
// CustomConnectivityUpdate
//------------------------------------------------------------------------
void CustomConnectivityUpdate::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomConnectivityUpdateModel()->getVarIndex(varName)] = loc;
}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::setPreVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomConnectivityUpdateModel()->getPreVarIndex(varName)] = loc;
}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::setPostVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomConnectivityUpdateModel()->getPostVarIndex(varName)] = loc;
}
//------------------------------------------------------------------------
VarLocation CustomConnectivityUpdate::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomConnectivityUpdateModel()->getVarIndex(varName)];
}
//------------------------------------------------------------------------
VarLocation CustomConnectivityUpdate::getPreVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomConnectivityUpdateModel()->getPreVarIndex(varName)];
}
//------------------------------------------------------------------------
VarLocation CustomConnectivityUpdate::getPostVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomConnectivityUpdateModel()->getPostVarIndex(varName)];
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isVarInitRequired() const
{

}
//------------------------------------------------------------------------
CustomConnectivityUpdate::CustomConnectivityUpdate(const std::string &name, const std::string &updateGroupName,
                                                   const CustomConnectivityUpdateModels::Base *customConnectivityUpdateModel,
                                                   const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                                                   const std::vector<Models::VarInit> &preVarInitialisers, const std::vector<Models::VarInit> &postVarInitialisers,
                                                   const std::vector<Models::WUVarReference> &varReferences, const std::vector<Models::VarReference> &preVarReferences,
                                                   const std::vector<Models::VarReference> &postVarReferences, VarLocation defaultVarLocation,
                                                   VarLocation defaultExtraGlobalParamLocation)
:   m_Name(name), m_UpdateGroupName(updateGroupName), m_CustomConnectivityUpdateModel(customConnectivityUpdateModel),
    m_Params(params), m_VarInitialisers(varInitialisers), m_PreVarInitialisers(preVarInitialisers), m_PostVarInitialisers(postVarInitialisers),
    m_VarLocation(varInitialisers.size(), defaultVarLocation), m_PreVarLocation(preVarInitialisers.size()), m_PostVarLocation(postVarInitialisers.size()),
    m_VarReferences(varReferences), m_PreVarReferences(preVarReferences), m_PostVarReferences(postVarReferences),
    m_SynapseGroup(m_VarReferences.empty() ? nullptr : static_cast<const SynapseGroupInternal*>(m_VarReferences.front().getSynapseGroup())),
    m_ExtraGlobalParamLocation(customConnectivityUpdateModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation), m_Batched(false)
{
    // Validate names
    Utils::validatePopName(name, "Custom connectivity update");
    Utils::validatePopName(updateGroupName, "Custom connectivity update group name");
    getCustomConnectivityUpdateModel()->validate();
}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::initDerivedParams(double dt)
{
    auto derivedParams = getCustomConnectivityUpdateModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_DerivedParams.reserve(derivedParams.size());

    // Loop through derived parameters
    for (const auto &d : derivedParams) {
        m_DerivedParams.push_back(d.func(getParams(), dt));
    }

    // Initialise derived parameters for synaptic variable initialisers
    for (auto &v : m_VarInitialisers) {
        v.initDerivedParams(dt);
    }

    // Initialise derived parameters for presynaptic variable initialisers
    for (auto &v : m_PreVarInitialisers) {
        v.initDerivedParams(dt);
    }

    // Initialise derived parameters for postsynaptic variable initialisers
    for (auto &v : m_PostVarInitialisers) {
        v.initDerivedParams(dt);
    }
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isInitRNGRequired() const
{

}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isZeroCopyEnabled() const
{

}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::updateHash(boost::uuids::detail::sha1 &hash) const
{

}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::updateInitHash(boost::uuids::detail::sha1 &hash) const
{

}
//------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdate::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(m_VarLocation, hash);
    Utils::updateHash(m_PreVarLocation, hash);
    Utils::updateHash(m_PostVarLocation, hash);
    Utils::updateHash(m_ExtraGlobalParamLocation, hash);
    return hash.get_digest();
}
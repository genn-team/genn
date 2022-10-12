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
    using CustomConnectivityUpdate::isBatched;
    using CustomConnectivityUpdate::getVarLocationHashDigest;

    //using CustomUpdate::finalize;
    using CustomConnectivityUpdate::getHashDigest;
    using CustomConnectivityUpdate::getInitHashDigest;
};

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
}   // namespace GeNN

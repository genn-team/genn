#pragma once

// GeNN includes
#include "currentSource.h"

//------------------------------------------------------------------------
// GeNN::CurrentSourceInternal
//------------------------------------------------------------------------
namespace GeNN
{
class CurrentSourceInternal : public CurrentSource
{
public:
    using GroupExternal = CurrentSource;

    CurrentSourceInternal(const std::string &name, const CurrentSourceModels::Base *currentSourceModel,
                          const std::map<std::string, Type::NumericValue> &params, const std::map<std::string, InitVarSnippet::Init> &varInitialisers,
                          const std::map<std::string, Models::VarReference> &neuronVarReferences, const NeuronGroupInternal *targetNeuronGroup, 
                          VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   CurrentSource(name, currentSourceModel, params, varInitialisers, neuronVarReferences, 
                      targetNeuronGroup, defaultVarLocation, defaultExtraGlobalParamLocation)
    {
    }

    using CurrentSource::getTrgNeuronGroup;
    using CurrentSource::finalise;
    using CurrentSource::getDerivedParams;
    using CurrentSource::isZeroCopyEnabled;
    using CurrentSource::isVarInitRequired;
    using CurrentSource::getHashDigest;
    using CurrentSource::getInitHashDigest;
    using CurrentSource::getVarLocationHashDigest;
    using CurrentSource::getInjectionCodeTokens;
};
}   // namespace GeNN

#pragma once

// GeNN includes
#include "currentSource.h"

//------------------------------------------------------------------------
// CurrentSourceInternal
//------------------------------------------------------------------------
class CurrentSourceInternal : public CurrentSource
{
public:
    CurrentSourceInternal(const std::string &name, const CurrentSourceModels::Base *currentSourceModel,
                          const std::unordered_map<std::string, double> &params, const std::unordered_map<std::string, Models::VarInit> &varInitialisers,
                          const NeuronGroupInternal *targetNeuronGroup, VarLocation defaultVarLocation, 
                          VarLocation defaultExtraGlobalParamLocation)
    :   CurrentSource(name, currentSourceModel, params, varInitialisers, targetNeuronGroup, 
                      defaultVarLocation, defaultExtraGlobalParamLocation)
    {
    }

    using CurrentSource::getTrgNeuronGroup;
    using CurrentSource::initDerivedParams;
    using CurrentSource::getDerivedParams;
    using CurrentSource::isSimRNGRequired;
    using CurrentSource::isInitRNGRequired;
    using CurrentSource::isZeroCopyEnabled;
    using CurrentSource::getHashDigest;
    using CurrentSource::getInitHashDigest;
    using CurrentSource::getRunnerHashDigest;
    using CurrentSource::getVarLocationHashDigest;
};

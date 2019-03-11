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
                          const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                          VarLocation defaultVarLocation)
    :   CurrentSource(name, currentSourceModel, params, varInitialisers, defaultVarLocation)
    {
    }

    using CurrentSource::initDerivedParams;
    using CurrentSource::getDerivedParams;
    using CurrentSource::isInitCodeRequired;
    using CurrentSource::isSimRNGRequired;
    using CurrentSource::isInitRNGRequired;
};

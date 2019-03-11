#pragma once

// GeNN includes
#include "currentSource.h"

//------------------------------------------------------------------------
// CurrentSourceInternal
//------------------------------------------------------------------------
class CurrentSourceInternal : public CurrentSource
{
public:
    using CurrentSource::CurrentSource;
    using CurrentSource::initDerivedParams;
    using CurrentSource::getDerivedParams;
    using CurrentSource::isInitCodeRequired;
    using CurrentSource::isSimRNGRequired;
    using CurrentSource::isInitRNGRequired;
};

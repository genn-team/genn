#pragma once


// GeNN includes
#include "modelSpec.h"

//------------------------------------------------------------------------
// ModelSpecInternal
//------------------------------------------------------------------------
class ModelSpecInternal : public ModelSpec
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    using ModelSpec::getLocalNeuronGroups;
    using ModelSpec::getRemoteNeuronGroups;
    using ModelSpec::getLocalSynapseGroups;
    using ModelSpec::getRemoteSynapseGroups;
    using ModelSpec::getLocalCurrentSources;
    using ModelSpec::getRemoteCurrentSources;

    using ModelSpec::finalize;

    using ModelSpec::scalarExpr;

    using ModelSpec::getGeneratedCodePath;

    using ModelSpec::zeroCopyInUse;
};

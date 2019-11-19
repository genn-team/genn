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
    using ModelSpec::getMergedLocalNeuronGroups;
    using ModelSpec::getMergedLocalSynapseGroups;
    using ModelSpec::getMergedLocalNeuronInitGroups;
    using ModelSpec::getMergedLocalSynapseConnectivityInitGroups;


    using ModelSpec::finalize;

    using ModelSpec::scalarExpr;

    using ModelSpec::zeroCopyInUse;
};

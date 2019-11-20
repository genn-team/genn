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
    using ModelSpec::getNeuronGroups;
    using ModelSpec::getSynapseGroups;
    using ModelSpec::getLocalCurrentSources;
    using ModelSpec::getMergedLocalNeuronGroups;
    using ModelSpec::getMergedLocalSynapseGroups;
    using ModelSpec::getMergedLocalNeuronInitGroups;
    using ModelSpec::getMergedLocalSynapseInitGroups;
    using ModelSpec::getMergedLocalSynapseConnectivityInitGroups;


    using ModelSpec::finalize;

    using ModelSpec::scalarExpr;

    using ModelSpec::zeroCopyInUse;
};

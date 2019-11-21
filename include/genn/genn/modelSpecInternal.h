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
    using ModelSpec::getMergedNeuronUpdateGroups;
    using ModelSpec::getMergedPresynapticUpdateGroups;
    using ModelSpec::getMergedPostsynapticUpdateGroups;
    using ModelSpec::getMergedSynapseDynamicsUpdateGroups;
    using ModelSpec::getMergedNeuronInitGroups;
    using ModelSpec::getMergedSynapseDenseInitGroups;
    using ModelSpec::getMergedSynapseConnectivityInitGroups;
    using ModelSpec::getMergedSynapseSparseInitGroups;

    using ModelSpec::finalize;

    using ModelSpec::scalarExpr;

    using ModelSpec::zeroCopyInUse;
};

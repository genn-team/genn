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
    using ModelSpec::getCustomUpdates;
    using ModelSpec::getCustomWUUpdates;

    using ModelSpec::finalize;

    using ModelSpec::scalarExpr;

    using ModelSpec::zeroCopyInUse;
    using ModelSpec::isRecordingInUse;
};

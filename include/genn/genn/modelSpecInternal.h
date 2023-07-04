#pragma once


// GeNN includes
#include "modelSpec.h"

//------------------------------------------------------------------------
// GeNN::ModelSpecInternal
//------------------------------------------------------------------------
namespace GeNN
{
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
    using ModelSpec::getCustomConnectivityUpdates;

    using ModelSpec::finalise;

    using ModelSpec::zeroCopyInUse;
    using ModelSpec::isRecordingInUse;
    using ModelSpec::getHashDigest;
};
}   // namespace GeNN

#include "weightUpdateModels.h"

IMPLEMENT_MODEL(WeightUpdateModels::StaticPulse);
IMPLEMENT_MODEL(WeightUpdateModels::StaticPulseDendriticDelay);
IMPLEMENT_MODEL(WeightUpdateModels::StaticGraded);
IMPLEMENT_MODEL(WeightUpdateModels::PiecewiseSTDP);

//----------------------------------------------------------------------------
// NeuronModels::Base
//----------------------------------------------------------------------------
bool WeightUpdateModels::Base::canBeMerged(const Base *other) const
{
    return (Models::Base::canBeMerged(other)
            && (getSimCode() == other->getSimCode())
            && (getEventCode() == other->getEventCode())
            && (getLearnPostCode() == other->getLearnPostCode())
            && (getSynapseDynamicsCode() == other->getSynapseDynamicsCode())
            && (getEventThresholdConditionCode() == other->getEventThresholdConditionCode())
            && (getSimSupportCode() == other->getSimSupportCode())
            && (getLearnPostSupportCode() == other->getLearnPostSupportCode())
            && (getSynapseDynamicsSuppportCode() == other->getSynapseDynamicsSuppportCode())
            && (getPreSpikeCode() == other->getPreSpikeCode())
            && (getPostSpikeCode() == other->getPostSpikeCode())
            && (getPreVars() == other->getPreVars())
            && (getPostVars() == other->getPostVars())
            && (isPreSpikeTimeRequired() == other->isPreSpikeTimeRequired())
            && (isPostSpikeTimeRequired() == other->isPostSpikeTimeRequired())
            && (isPrevPreSpikeTimeRequired() == other->isPrevPreSpikeTimeRequired())
            && (isPrevPostSpikeTimeRequired() == other->isPrevPostSpikeTimeRequired()));
}

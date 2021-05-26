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
            && (isPreSpikeEventTimeRequired() == other->isPreSpikeEventTimeRequired())
            && (isPrevPreSpikeTimeRequired() == other->isPrevPreSpikeTimeRequired())
            && (isPrevPostSpikeTimeRequired() == other->isPrevPostSpikeTimeRequired())
            && (isPrevPreSpikeEventTimeRequired() == other->isPrevPreSpikeEventTimeRequired()));
}

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
void WeightUpdateModels::updateHash(const Base &w, boost::uuids::detail::sha1 &hash)
{
    // Superclass
    Models::updateHash(w, hash);

    Utils::updateHash(w.getSimCode(), hash);
    Utils::updateHash(w.getEventCode(), hash);
    Utils::updateHash(w.getLearnPostCode(), hash);
    Utils::updateHash(w.getSynapseDynamicsCode(), hash);
    Utils::updateHash(w.getEventThresholdConditionCode(), hash);
    Utils::updateHash(w.getSimSupportCode(), hash);
    Utils::updateHash(w.getLearnPostSupportCode(), hash);
    Utils::updateHash(w.getSynapseDynamicsSuppportCode(), hash);
    Utils::updateHash(w.getPreSpikeCode(), hash);
    Utils::updateHash(w.getPostSpikeCode(), hash);
    Utils::updateHash(w.getPreVars(), hash);
    Utils::updateHash(w.getPostVars(), hash);
    Utils::updateHash(w.isPreSpikeTimeRequired(), hash);
    Utils::updateHash(w.isPostSpikeTimeRequired(), hash);
    Utils::updateHash(w.isPreSpikeEventTimeRequired(), hash);
    Utils::updateHash(w.isPrevPreSpikeTimeRequired(), hash);
    Utils::updateHash(w.isPrevPostSpikeTimeRequired(), hash);
    Utils::updateHash(w.isPrevPreSpikeEventTimeRequired(), hash);
}

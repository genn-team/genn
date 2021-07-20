#include "weightUpdateModels.h"

IMPLEMENT_MODEL(WeightUpdateModels::StaticPulse);
IMPLEMENT_MODEL(WeightUpdateModels::StaticPulseDendriticDelay);
IMPLEMENT_MODEL(WeightUpdateModels::StaticGraded);
IMPLEMENT_MODEL(WeightUpdateModels::PiecewiseSTDP);

//----------------------------------------------------------------------------
// WeightUpdateModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type WeightUpdateModels::Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getSimCode(), hash);
    Utils::updateHash(getEventCode(), hash);
    Utils::updateHash(getLearnPostCode(), hash);
    Utils::updateHash(getSynapseDynamicsCode(), hash);
    Utils::updateHash(getEventThresholdConditionCode(), hash);
    Utils::updateHash(getSimSupportCode(), hash);
    Utils::updateHash(getLearnPostSupportCode(), hash);
    Utils::updateHash(getSynapseDynamicsSuppportCode(), hash);
    Utils::updateHash(getPreSpikeCode(), hash);
    Utils::updateHash(getPostSpikeCode(), hash);
    Utils::updateHash(getPreVars(), hash);
    Utils::updateHash(getPostVars(), hash);
    Utils::updateHash(isPreSpikeTimeRequired(), hash);
    Utils::updateHash(isPostSpikeTimeRequired(), hash);
    Utils::updateHash(isPreSpikeEventTimeRequired(), hash);
    Utils::updateHash(isPrevPreSpikeTimeRequired(), hash);
    Utils::updateHash(isPrevPostSpikeTimeRequired(), hash);
    Utils::updateHash(isPrevPreSpikeEventTimeRequired(), hash);

    // Return digest
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void WeightUpdateModels::Base::validate() const
{
    // Superclass
    Models::Base::validate();

    Utils::validateVecNames(getPreVars(), "Presynaptic variable");
    Utils::validateVecNames(getPostVars(), "Presynaptic variable");
}

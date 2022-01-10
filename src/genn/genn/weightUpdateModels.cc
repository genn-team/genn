#include "weightUpdateModels.h"

IMPLEMENT_SNIPPET(WeightUpdateModels::StaticPulse);
IMPLEMENT_SNIPPET(WeightUpdateModels::StaticPulseDendriticDelay);
IMPLEMENT_SNIPPET(WeightUpdateModels::StaticGraded);
IMPLEMENT_SNIPPET(WeightUpdateModels::PiecewiseSTDP);

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

    // If any variables have a reduction access mode, give an error
    const auto vars = getVars();
    const auto preVars = getPreVars();
    const auto postVars = getPostVars();
    if(std::any_of(vars.cbegin(), vars.cend(),
                   [](const Models::Base::Var &v){ return (v.access & VarAccessModeAttribute::REDUCE); })
       || std::any_of(preVars.cbegin(), preVars.cend(),
                      [](const Models::Base::Var &v){ return (v.access & VarAccessModeAttribute::REDUCE); })
       || std::any_of(postVars.cbegin(), postVars.cend(),
                      [](const Models::Base::Var &v){ return (v.access & VarAccessModeAttribute::REDUCE); }))
    {
        throw std::runtime_error("Weight update models cannot include variables with REDUCE access modes - they are only supported by custom update models");
    }
}

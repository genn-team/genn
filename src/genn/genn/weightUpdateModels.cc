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
void WeightUpdateModels::Base::validate(const std::unordered_map<std::string, double> &paramValues, 
                                        const std::unordered_map<std::string, Models::VarInit> &varValues,
                                        const std::unordered_map<std::string, Models::VarInit> &preVarValues,
                                        const std::unordered_map<std::string, Models::VarInit> &postVarValues,
                                        const std::string &description) const
{
    // Superclass
    Models::Base::validate(paramValues, varValues, description);

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

    // Validate variable reference initialisers
    Utils::validateInitialisers(preVars, preVarValues, "presynaptic variable", description);

    // Validate variable reference initialisers
    Utils::validateInitialisers(postVars, postVarValues, "postsynaptic variable", description);

    // If any variables have shared neuron duplication mode, give an error
    if (std::any_of(vars.cbegin(), vars.cend(),
                    [](const Models::Base::Var &v) { return (v.access & VarAccessDuplication::SHARED_NEURON); }))
    {
        throw std::runtime_error("Weight update models cannot include variables with SHARED_NEURON access modes - they are only supported on pre, postsynaptic or neuron variables");
    }
}

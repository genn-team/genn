#include "neuronModels.h"

// Implement models
IMPLEMENT_SNIPPET(NeuronModels::RulkovMap);
IMPLEMENT_SNIPPET(NeuronModels::Izhikevich);
IMPLEMENT_SNIPPET(NeuronModels::IzhikevichVariable);
IMPLEMENT_SNIPPET(NeuronModels::LIF);
IMPLEMENT_SNIPPET(NeuronModels::SpikeSource);
IMPLEMENT_SNIPPET(NeuronModels::SpikeSourceArray);
IMPLEMENT_SNIPPET(NeuronModels::Poisson);
IMPLEMENT_SNIPPET(NeuronModels::PoissonNew);
IMPLEMENT_SNIPPET(NeuronModels::TraubMiles);
IMPLEMENT_SNIPPET(NeuronModels::TraubMilesFast);
IMPLEMENT_SNIPPET(NeuronModels::TraubMilesAlt);
IMPLEMENT_SNIPPET(NeuronModels::TraubMilesNStep);

//----------------------------------------------------------------------------
// NeuronModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronModels::Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getSimCode(), hash);
    Utils::updateHash(getThresholdConditionCode(), hash);
    Utils::updateHash(getResetCode(), hash);
    Utils::updateHash(getSupportCode(), hash);
    Utils::updateHash(isAutoRefractoryRequired(), hash);
    Utils::updateHash(getAdditionalInputVars(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void NeuronModels::Base::validate(const std::unordered_map<std::string, double> &paramValues, 
                                  const std::unordered_map<std::string, Models::VarInit> &varValues,
                                  const std::string &description) const
{
    // Superclass
    Models::Base::validate(paramValues, varValues, description);

    Utils::validateVecNames(getAdditionalInputVars(), "Additional input variable");

    // If any variables have a reduction access mode, give an error
    const auto vars = getVars();
    if(std::any_of(vars.cbegin(), vars.cend(),
                   [](const Models::Base::Var &v){ return (v.access & VarAccessModeAttribute::REDUCE); }))
    {
        throw std::runtime_error("Neuron models cannot include variables with REDUCE access modes - they are only supported by custom update models");
    }
}

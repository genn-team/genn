#include "neuronModels.h"

// Implement models
IMPLEMENT_MODEL(NeuronModels::RulkovMap);
IMPLEMENT_MODEL(NeuronModels::Izhikevich);
IMPLEMENT_MODEL(NeuronModels::IzhikevichVariable);
IMPLEMENT_MODEL(NeuronModels::LIF);
IMPLEMENT_MODEL(NeuronModels::SpikeSource);
IMPLEMENT_MODEL(NeuronModels::SpikeSourceArray);
IMPLEMENT_MODEL(NeuronModels::Poisson);
IMPLEMENT_MODEL(NeuronModels::PoissonNew);
IMPLEMENT_MODEL(NeuronModels::TraubMiles);
IMPLEMENT_MODEL(NeuronModels::TraubMilesFast);
IMPLEMENT_MODEL(NeuronModels::TraubMilesAlt);
IMPLEMENT_MODEL(NeuronModels::TraubMilesNStep);

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
void NeuronModels::Base::validate() const
{
    // Superclass
    Models::Base::validate();

    Utils::validateVecNames(getAdditionalInputVars(), "Additional input variable");

    // If any variables have a reduction access mode, give an error
    const auto vars = getVars();
    if(std::any_of(vars.cbegin(), vars.cend(),
                   [](const Models::Base::Var &v){ return (v.access & VarAccessModeAttribute::REDUCE); }))
    {
        throw std::runtime_error("Neuron models cannot include variables with REDUCE access modes - they are only supported by custom update models");
    }
}

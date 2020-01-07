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
bool NeuronModels::Base::canBeMerged(const Base *other) const
{
    return (Models::Base::canBeMerged(other)
            && (getSimCode() == other->getSimCode())
            && (getThresholdConditionCode() == other->getThresholdConditionCode())
            && (getResetCode() == other->getResetCode())
            && (getSupportCode() == other->getSupportCode())
            && (isAutoRefractoryRequired() == other->isAutoRefractoryRequired())
            && (getAdditionalInputVars() == other->getAdditionalInputVars()));
}

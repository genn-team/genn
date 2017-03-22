#include "newNeuronModels.h"

// Implement models
IMPLEMENT_MODEL(NeuronModels::Izhikevich);
IMPLEMENT_MODEL(NeuronModels::SpikeSource);
IMPLEMENT_MODEL(NeuronModels::Poisson);

//----------------------------------------------------------------------------
// NeuronModels::LegacyWrapper
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::getSimCode() const
{
    return nModels[m_LegacyTypeIndex].simCode;
}
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::getThresholdConditionCode() const
{
    return nModels[m_LegacyTypeIndex].thresholdConditionCode;
}
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::getResetCode() const
{
    return nModels[m_LegacyTypeIndex].resetCode;
}
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::getSupportCode() const
{
    return nModels[m_LegacyTypeIndex].supportCode;
}
//----------------------------------------------------------------------------
NewModels::Base::StringPairVec NeuronModels::LegacyWrapper::getExtraGlobalParams() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return zipStringVectors(nm.extraGlobalNeuronKernelParameters, nm.extraGlobalNeuronKernelParameterTypes);
}
//----------------------------------------------------------------------------
bool NeuronModels::LegacyWrapper::isPoisson() const
{
    return (m_LegacyTypeIndex == POISSONNEURON);
}
//----------------------------------------------------------------------------

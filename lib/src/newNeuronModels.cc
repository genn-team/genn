#include "newNeuronModels.h"

// Implement models
IMPLEMENT_MODEL(NeuronModels::Izhikevich);
IMPLEMENT_MODEL(NeuronModels::SpikeSource);

//----------------------------------------------------------------------------
// NeuronModels::LegacyWrapper
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::GetSimCode() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return nm.simCode;
}
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::GetThresholdConditionCode() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return nm.thresholdConditionCode;
}
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::GetResetCode() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return nm.resetCode;
}
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::GetSupportCode() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return nm.supportCode;
}
//----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string>> NeuronModels::LegacyWrapper::GetExtraGlobalParams() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return ZipStringVectors(nm.extraGlobalNeuronKernelParameters, nm.extraGlobalNeuronKernelParameterTypes);
}
//----------------------------------------------------------------------------
bool NeuronModels::LegacyWrapper::IsPoisson() const
{
    return (m_LegacyTypeIndex == POISSONNEURON);
}
//----------------------------------------------------------------------------

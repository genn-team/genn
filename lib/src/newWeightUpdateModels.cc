#include "newWeightUpdateModels.h"

IMPLEMENT_MODEL(WeightUpdateModels::StaticPulse);
IMPLEMENT_MODEL(WeightUpdateModels::StaticGraded);
IMPLEMENT_MODEL(WeightUpdateModels::Learn1);

//----------------------------------------------------------------------------
// WeightUpdateModels::LegacyWrapper
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::GetSimCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].simCode;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::GetEventCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].simCodeEvnt;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::GetLearnPostCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].simLearnPost;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::GetSynapseDynamicsCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].synapseDynamics;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::GetEventThresholdConditionCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].evntThreshold;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::GetSimSupportCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].simCode_supportCode;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::GetLearnPostSupportCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].simLearnPost_supportCode;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::GetSynapseDynamicsSuppportCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].synapseDynamics_supportCode;
}
//----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string>> WeightUpdateModels::LegacyWrapper::GetExtraGlobalParams() const
{
    const auto &wu = weightUpdateModels[m_LegacyTypeIndex];
    return ZipStringVectors(wu.extraGlobalSynapseKernelParameters, wu.extraGlobalSynapseKernelParameterTypes);
}
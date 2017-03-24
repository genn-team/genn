#include "newWeightUpdateModels.h"

IMPLEMENT_MODEL(WeightUpdateModels::StaticPulse);
IMPLEMENT_MODEL(WeightUpdateModels::StaticGraded);
IMPLEMENT_MODEL(WeightUpdateModels::Learn1);

//----------------------------------------------------------------------------
// WeightUpdateModels::LegacyWrapper
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::getSimCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].simCode;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::getEventCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].simCodeEvnt;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::getLearnPostCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].simLearnPost;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::getSynapseDynamicsCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].synapseDynamics;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::getEventThresholdConditionCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].evntThreshold;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::getSimSupportCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].simCode_supportCode;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::getLearnPostSupportCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].simLearnPost_supportCode;
}
//----------------------------------------------------------------------------
std::string WeightUpdateModels::LegacyWrapper::getSynapseDynamicsSuppportCode() const
{
    return weightUpdateModels[m_LegacyTypeIndex].synapseDynamics_supportCode;
}
//----------------------------------------------------------------------------
NewModels::Base::StringPairVec WeightUpdateModels::LegacyWrapper::getExtraGlobalParams() const
{
    const auto &wu = weightUpdateModels[m_LegacyTypeIndex];
    return zipStringVectors(wu.extraGlobalSynapseKernelParameters, wu.extraGlobalSynapseKernelParameterTypes);
}
//----------------------------------------------------------------------------
bool WeightUpdateModels::LegacyWrapper::isPreSpikeTimeRequired() const
{
    return weightUpdateModels[m_LegacyTypeIndex].needPreSt;
}
 //----------------------------------------------------------------------------
bool WeightUpdateModels::LegacyWrapper::isPostSpikeTimeRequired() const
{
    return weightUpdateModels[m_LegacyTypeIndex].needPostSt;
}
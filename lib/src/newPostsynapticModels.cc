#include "newPostsynapticModels.h"

// Implement models
IMPLEMENT_MODEL(PostsynapticModels::ExpCond);
IMPLEMENT_MODEL(PostsynapticModels::DeltaCurr);

//----------------------------------------------------------------------------
// PostsynapticModels::LegacyWrapper
//----------------------------------------------------------------------------
std::string PostsynapticModels::LegacyWrapper::getDecayCode() const
{
    const auto &ps = postSynModels[m_LegacyTypeIndex];
    return ps.postSynDecay;
}
//----------------------------------------------------------------------------
std::string PostsynapticModels::LegacyWrapper::getCurrentConverterCode() const
{
    const auto &ps = postSynModels[m_LegacyTypeIndex];
    return ps.postSyntoCurrent;
}
//----------------------------------------------------------------------------
std::string PostsynapticModels::LegacyWrapper::getSupportCode() const
{
    const auto &ps = postSynModels[m_LegacyTypeIndex];
    return ps.supportCode;
}
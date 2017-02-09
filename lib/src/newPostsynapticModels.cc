#include "newPostsynapticModels.h"

//----------------------------------------------------------------------------
// PostsynapticModels::LegacyWrapper
//----------------------------------------------------------------------------
std::string PostsynapticModels::LegacyWrapper::GetDecayCode() const
{
    const auto &ps = postSynModels[m_LegacyTypeIndex];
    return ps.postSynDecay;
}
//----------------------------------------------------------------------------
std::string PostsynapticModels::LegacyWrapper::GetCurrentConverterCode() const
{
    const auto &ps = postSynModels[m_LegacyTypeIndex];
    return ps.postSyntoCurrent;
}
//----------------------------------------------------------------------------
std::string PostsynapticModels::LegacyWrapper::GetSupportCode() const
{
    const auto &ps = postSynModels[m_LegacyTypeIndex];
    return ps.supportCode;
}
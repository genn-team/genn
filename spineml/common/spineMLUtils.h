#pragma once

// Standard C++ includes
#include <algorithm>

//----------------------------------------------------------------------------
// SpineMLCommon::SpineMLUtils
//----------------------------------------------------------------------------
namespace SpineMLCommon
{
namespace SpineMLUtils
{
inline std::string getSafeName(const std::string &name)
{
    std::string safeName = name;
    std::replace(safeName.begin(), safeName.end(), ' ', '_');
    return safeName;
}
}   // namespace Utils
}   // namespace SpineMLCommon
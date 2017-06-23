#pragma once

// Standard C++ includes
#include <algorithm>
#include <sstream>

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

template<typename Iter>
inline void readCSVIndices(const char *csvString, Iter outputIter)
{
     // **TODO** maybe move somewhere common
    std::stringstream indicesStream(csvString);
    while(indicesStream.good()) {
        std::string index;
        std::getline(indicesStream, index, ',');
        *outputIter++ = std::stoul(index);
    }
}
}   // namespace Utils
}   // namespace SpineMLCommon
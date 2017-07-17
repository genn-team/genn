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
    std::replace(safeName.begin(), safeName.end(), '-', '_');
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

template<typename Iter>
inline void readCSVValues(const char *csvString, Iter outputIter)
{
     // **TODO** maybe move somewhere common
    std::stringstream indicesStream(csvString);
    while(indicesStream.good()) {
        std::string index;
        std::getline(indicesStream, index, ',');
        *outputIter++ = std::stod(index);
    }
}

inline std::string xPathNodeHasSuffix(const std::string &suffix, const std::string &nodePath = "*")
{
    // Build XPath 1.0 query to match nodes whose name ends in specified suffix
    // https://stackoverflow.com/questions/4203119/xpath-wildcards-on-node-name
    std::stringstream queryStream;
    queryStream << nodePath << "[substring(name(), string-length(name()) - " << suffix.length() - 1 << ") = '" << suffix << "']";
    return queryStream.str();
}
}   // namespace Utils
}   // namespace SpineMLCommon
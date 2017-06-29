#pragma once

// Standard C++ includes
#include <map>
#include <set>
#include <string>

// Forward declarations
namespace filesystem
{
    class path;
}

namespace pugi
{
    class xml_node;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class ModelParams
{
public:
    ModelParams(const filesystem::path &basePath, const pugi::xml_node &node,
                std::map<std::string, double> &fixedParamVals);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const std::string &getURL() const{ return m_URL; }
    const std::set<std::string> &getVariableParams() const{ return m_VariableParams; }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    bool operator < (const ModelParams &other) const
    {
        return (std::tie(m_URL, m_VariableParams, m_InputPortMappings)
                < std::tie(other.m_URL, other.m_VariableParams, other.m_InputPortMappings));
    }

private:
    //----------------------------------------------------------------------------
    // Enumerations
    //----------------------------------------------------------------------------
    enum class PortSource
    {
        PRE_NEURON,
        POST_NEURON,
        POST_SYNAPSE,
        WEIGHT_UPDATE
    };

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::string m_URL;
    std::set<std::string> m_VariableParams;

    // Map of destination port names to their source component and port
    std::map<std::string, std::pair<PortSource, std::string>> m_InputPortMappings;
};
}   // namespace SpineMLGenerator
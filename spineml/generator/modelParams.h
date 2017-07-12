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
// SpineMLGenerator::ModelParams::Base
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
namespace ModelParams
{
class Base
{
public:
    //----------------------------------------------------------------------------
    // Enumerations
    //----------------------------------------------------------------------------
    enum class PortSource
    {
        PRESYNAPTIC_NEURON,
        POSTSYNAPTIC_NEURON,
        POSTSYNAPTIC_SYNAPSE,
        WEIGHT_UPDATE
    };

    Base(const filesystem::path &basePath, const pugi::xml_node &node,
         std::map<std::string, double> &fixedParamVals);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const std::string &getURL() const{ return m_URL; }
    const std::set<std::string> &getVariableParams() const{ return m_VariableParams; }
    const std::pair<PortSource, std::string> &getPortSrc(const std::string &dstPort) const;
    const std::string &getPortTrg(PortSource src, const std::string srcPort) const;

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    bool operator < (const Base &other) const
    {
        return (std::tie(m_URL, m_VariableParams, m_PortMappings)
                < std::tie(other.m_URL, other.m_VariableParams, other.m_PortMappings));
    }

protected:
    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    void addPortMapping(const std::string &dstPort, PortSource srcComponent, const std::string &srcPort);

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::string m_URL;
    std::set<std::string> m_VariableParams;

    // Map of destination port names to their source component and port
    std::map<std::string, std::pair<PortSource, std::string>> m_PortMappings;
};

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::Neuron
//----------------------------------------------------------------------------
class Neuron : public Base
{
public:
    Neuron(const filesystem::path &basePath, const pugi::xml_node &node,
           std::map<std::string, double> &fixedParamVals);
};

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::WeightUpdate
//----------------------------------------------------------------------------
class WeightUpdate : public Base
{
public:
    WeightUpdate(const filesystem::path &basePath, const pugi::xml_node &node,
                 const std::string &srcPopName, const std::string &trgPopName,
                 std::map<std::string, double> &fixedParamVals);
};

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::Postsynaptic
//----------------------------------------------------------------------------
class Postsynaptic : public Base
{
public:
    Postsynaptic(const filesystem::path &basePath, const pugi::xml_node &node,
                 const std::string &trgPopName,
                 std::map<std::string, double> &fixedParamVals);
};
}   // namespace ModelParams
}   // namespace SpineMLGenerator
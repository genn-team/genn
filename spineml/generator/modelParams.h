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

namespace NewModels
{
    class VarInit;
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
         std::map<std::string, NewModels::VarInit> &varInitialisers);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const std::string &getURL() const{ return m_URL; }
    const std::set<std::string> &getVariableParams() const{ return m_VariableParams; }
    const std::pair<PortSource, std::string> &getInputPortSrc(const std::string &dstPort) const;
    const std::pair<PortSource, std::string> &getOutputPortTrg(const std::string &srcPort) const;

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    bool operator < (const Base &other) const
    {
        return (std::tie(m_URL, m_VariableParams, m_InputPortSources, m_OutputPortTargets)
                < std::tie(other.m_URL, other.m_VariableParams, other.m_InputPortSources, other.m_OutputPortTargets));
    }

protected:
    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    void addInputPortMapping(const std::string &dstPort, PortSource srcComponent, const std::string &srcPort);
    void addOutputPortMapping(const std::string &srcPort, PortSource dstComponent, const std::string &dstPort);

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::string m_URL;
    std::set<std::string> m_VariableParams;

    // Map of destination port names to their source component and port
    std::map<std::string, std::pair<PortSource, std::string>> m_InputPortSources;
    std::map<std::string, std::pair<PortSource, std::string>> m_OutputPortTargets;
};

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::Neuron
//----------------------------------------------------------------------------
class Neuron : public Base
{
public:
    Neuron(const filesystem::path &basePath, const pugi::xml_node &node,
           std::map<std::string, NewModels::VarInit> &varInitialisers);
};

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::WeightUpdate
//----------------------------------------------------------------------------
class WeightUpdate : public Base
{
public:
    WeightUpdate(const filesystem::path &basePath, const pugi::xml_node &node,
                 const std::string &srcPopName, const std::string &trgPopName,
                 std::map<std::string, NewModels::VarInit> &varInitialisers);
};

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::Postsynaptic
//----------------------------------------------------------------------------
class Postsynaptic : public Base
{
public:
    Postsynaptic(const filesystem::path &basePath, const pugi::xml_node &node,
                 const std::string &trgPopName,
                 std::map<std::string, NewModels::VarInit> &varInitialisers);
};
}   // namespace ModelParams
}   // namespace SpineMLGenerator
#pragma once

// Standard C++ includes
#include <string>

// Forward declarations
namespace pugi
{
    class xml_node;
}

namespace filesystem
{
    class path;
}

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutput
//----------------------------------------------------------------------------
namespace SpineMLSimulator
{
class LogOutput
{
public:
    LogOutput(const pugi::xml_node &node, double dt, const filesystem::path &basePath);

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    // Record any data required during this timestep
    virtual void record(double dt, unsigned int timestep) = 0;

protected:
    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    bool shouldRecord(unsigned int timestep) const
    {
        return (timestep >= m_StartTimeStep && timestep < m_EndTimeStep);
    }

    const std::string &getName() const{ return m_Name; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    unsigned int m_StartTimeStep;
    unsigned int m_EndTimeStep;

    std::string m_Name;
};
}   // namespace SpineMLSimulator
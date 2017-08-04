#pragma once

// Standard C++ includes
#include <fstream>
#include <set>
#include <string>
#include <vector>

// SpineML simulator includes
#include "modelProperty.h"

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
// SpineMLSimulator::LogOutput::Base
//----------------------------------------------------------------------------
namespace SpineMLSimulator
{
namespace LogOutput
{
class Base
{
public:
    Base(const pugi::xml_node &node, double dt, unsigned int numTimeSteps);
    virtual ~Base(){}

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

    unsigned int getEndTimestep() const{ return m_EndTimeStep; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    unsigned int m_StartTimeStep;
    unsigned int m_EndTimeStep;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutput::Analogue
//----------------------------------------------------------------------------
class Analogue : public Base
{
public:
    Analogue(const pugi::xml_node &node, double dt, unsigned int numTimeSteps,
             const std::string &port, unsigned int popSize,
             const filesystem::path &basePath,
             const ModelProperty::Base *modelProperty);

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    // Record any data required during this timestep
    virtual void record(double dt, unsigned int timestep) override;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_File;

    const ModelProperty::Base *m_ModelProperty;

    std::vector<scalar> m_OutputBuffer;

    std::vector<unsigned int> m_Indices;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutput::Event
//----------------------------------------------------------------------------
class Event : public Base
{
public:
    Event(const pugi::xml_node &node, double dt, unsigned int numTimeSteps,
          const std::string &port, unsigned int popSize,
          const filesystem::path &basePath, unsigned int *spikeQueuePtr,
          unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
          unsigned int *hostSpikes, unsigned int *deviceSpikes);

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    // Record any data required during this timestep
    virtual void record(double dt, unsigned int timestep) override;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_File;

    const unsigned int m_PopSize;

    unsigned int *m_SpikeQueuePtr;
    unsigned int *m_HostSpikeCount;
    unsigned int *m_DeviceSpikeCount;

    unsigned int *m_HostSpikes;
    unsigned int *m_DeviceSpikes;

    std::set<unsigned int> m_Indices;
};
}   // namespace LogOutput
}   // namespace SpineMLSimulator
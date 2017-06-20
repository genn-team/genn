#pragma once

// Standard C++ includes
#include <fstream>

// SpineML simulator includes
#include "logOutput.h"

// Forward declarations
namespace pugi
{
    class xml_node;
}

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutputSpike
//----------------------------------------------------------------------------
namespace SpineMLSimulator
{
class LogOutputSpike : public LogOutput
{
public:
    LogOutputSpike(const pugi::xml_node &node, double dt,
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

    unsigned int *m_SpikeQueuePtr;
    unsigned int *m_HostSpikeCount;
    unsigned int *m_DeviceSpikeCount;

    unsigned int *m_HostSpikes;
    unsigned int *m_DeviceSpikes;
};
}   // namespace SpineMLSimulator
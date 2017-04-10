#include "logOutputSpike.h"

// GeNN includes
#include "utils.h"

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutputSpike
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutputSpike::LogOutputSpike(const pugi::xml_node &node, double dt, unsigned int *spikeQueuePtr,
                                                 unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
                                                 unsigned int *hostSpikes, unsigned int *deviceSpikes)
    : LogOutput(node, dt), m_SpikeQueuePtr(spikeQueuePtr),
      m_HostSpikeCount(hostSpikeCount), m_DeviceSpikeCount(deviceSpikeCount),
      m_HostSpikes(hostSpikes), m_DeviceSpikes(deviceSpikes)
{
    // Open CSV file and write header
    // **TEMP**
    m_File.open(getName() + ".csv");
    m_File << "Time(ms), Neuron ID" << std::endl;
}
//----------------------------------------------------------------------------
void SpineMLSimulator::LogOutputSpike::record(double dt, unsigned int timestep)
{
    // If we should be recording this timestep
    if(shouldRecord(timestep)) {
#ifndef CPU_ONLY
        // Copy spike count from spike queue position to host
        CHECK_CUDA_ERRORS(cudaMemcpy(&m_HostSpikeCount[*m_SpikeQueuePtr], &m_DeviceSpikeCount[*m_SpikeQueuePtr],
                                     sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // Copy this many spikes to host
        CHECK_CUDA_ERRORS(cudaMemcpy(m_HostSpikes, m_DeviceSpikes,
                                     sizeof(unsigned int) * m_HostSpikeCount[*m_SpikeQueuePtr], cudaMemcpyDeviceToHost));
#endif  // CPU_ONLY
        const double t = dt * (double)timestep;
        for(unsigned int i = 0; i < m_HostSpikeCount[*m_SpikeQueuePtr]; i++)
        {
            m_File << t << "," << m_HostSpikes[i] << std::endl;
        }
    }
}
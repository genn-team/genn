#include "logOutputSpike.h"

// GeNN includes
#include "utils.h"

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutputSpike
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutputSpike::LogOutputSpike(const pugi::xml_node &node, double dt,
                                                 const filesystem::path &basePath, unsigned int *spikeQueuePtr,
                                                 unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
                                                 unsigned int *hostSpikes, unsigned int *deviceSpikes)
    : LogOutput(node, dt, basePath), m_SpikeQueuePtr(spikeQueuePtr),
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
        // Determine current spike queue
        const unsigned int spikeQueueIndex = (m_SpikeQueuePtr == NULL) ? 0 : *m_SpikeQueuePtr;

#ifndef CPU_ONLY
        // Copy spike count from spike queue position to host
        CHECK_CUDA_ERRORS(cudaMemcpy(&m_HostSpikeCount[spikeQueueIndex], &m_DeviceSpikeCount[spikeQueueIndex],
                                     sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // Copy this many spikes to host
        CHECK_CUDA_ERRORS(cudaMemcpy(m_HostSpikes, m_DeviceSpikes,
                                     sizeof(unsigned int) * m_HostSpikeCount[spikeQueueIndex], cudaMemcpyDeviceToHost));
#endif  // CPU_ONLY
        const double t = dt * (double)timestep;
        for(unsigned int i = 0; i < m_HostSpikeCount[spikeQueueIndex]; i++)
        {
            m_File << t << "," << m_HostSpikes[i] << std::endl;
        }
    }
}
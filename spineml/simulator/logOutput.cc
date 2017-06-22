#include "logOutput.h"

// Standard C++ includes
#include <algorithm>
#include <limits>
#include <iostream>

// Standard C includes
#include <cmath>

// Filesystem includes
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "utils.h"

// SpineML simulator includes
#include "modelProperty.h"

//----------------------------------------------------------------------------
// SpineMLSimulator::Base::Base
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::Base::Base(const pugi::xml_node &node, double dt, const filesystem::path &basePath)
{
    // Combine node target and logger names to get file title
    std::string fileTitle = std::string(node.attribute("target").value()) + "_" + std::string(node.attribute("name").value());

    // Combine this with base path to get full file title
    m_Name = (basePath / fileTitle).str();
    std::cout << "Output log:" << m_Name << std::endl;

    // Read start time
    auto startAttr = node.attribute("start_time");
    if(startAttr.empty()) {
        m_StartTimeStep = 0;
    }
    else {
        m_StartTimeStep = (unsigned int)std::ceil(startAttr.as_double() / dt);
    }

    // Read duration
    auto durationAttr = node.attribute("duration");
    if(durationAttr.empty()) {
        m_EndTimeStep = std::numeric_limits<unsigned int>::max();
    }
    else {
        m_EndTimeStep = m_StartTimeStep + (unsigned int)std::ceil(durationAttr.as_double() / dt);
    }

    std::cout << "\tStart timestep:" << m_StartTimeStep << ", end timestep:" << m_EndTimeStep << std::endl;
}


//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutputAnalogue
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::Analogue::Analogue(const pugi::xml_node &node, double dt,
                                                const filesystem::path &basePath,
                                                const ModelProperty *modelProperty)
    : Base(node, dt, basePath), m_ModelProperty(modelProperty)
{
    // If indices are specified
    auto indices = node.attribute("indices");
    if(indices) {
        // **TODO** maybe move somewhere common
        std::stringstream indicesStream(indices.value());
        while(indicesStream.good()) {
            std::string index;
            std::getline(indicesStream, index, ',');
            m_Indices.push_back(std::stoul(index));
        }

        std::cout << "\tRecording " << m_Indices.size() << " values" << std::endl;
    }

    // Open CSV file and write header
    // **TEMP**
    m_File.open(getName() + ".csv");
    m_File << "Time(ms), Neuron ID, Value" << std::endl;
}
//----------------------------------------------------------------------------
void SpineMLSimulator::LogOutput::Analogue::record(double dt, unsigned int timestep)
{
    // If we should be recording this timestep
    if(shouldRecord(timestep)) {
        // Pull state variable from device
        // **TODO** simple min/max index optimisation
        m_ModelProperty->pullFromDevice();

        const double t = dt * (double)timestep;

        // If no indices are specified
        if(m_Indices.empty()) {
            // Loop through state variable values and write to file
            unsigned int i = 0;
            for(const scalar *v = m_ModelProperty->getHostStateVarBegin(); v != m_ModelProperty->getHostStateVarEnd(); v++, i++) {
                m_File << t << "," << i << "," << *v << std::endl;
            }
        }
        // Otherwise
        else {
            // Loop through indices and write selected values to file
            for(unsigned int i : m_Indices) {
                const scalar v = m_ModelProperty->getHostStateVarBegin()[i];
                m_File << t << "," << i << "," << v << std::endl;
            }
        }
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutputSpike
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::Event::Event(const pugi::xml_node &node, double dt,
                                          const filesystem::path &basePath, unsigned int *spikeQueuePtr,
                                          unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
                                          unsigned int *hostSpikes, unsigned int *deviceSpikes)
    : Base(node, dt, basePath), m_SpikeQueuePtr(spikeQueuePtr),
      m_HostSpikeCount(hostSpikeCount), m_DeviceSpikeCount(deviceSpikeCount),
      m_HostSpikes(hostSpikes), m_DeviceSpikes(deviceSpikes)
{
    // Open CSV file and write header
    m_File.open(getName() + ".csv");
}
//----------------------------------------------------------------------------
void SpineMLSimulator::LogOutput::Event::record(double dt, unsigned int timestep)
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
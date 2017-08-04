#include "logOutput.h"

// Standard C++ includes
#include <algorithm>
#include <limits>
#include <iostream>

// Standard C includes
#include <cassert>
#include <cmath>

// Filesystem includes
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "utils.h"

// SpineML common includes
#include "spineMLUtils.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
// Irritatingly typeid(float).name() isn't actually guaranteed to return float
// e.g. on GCC it returns "f" therefore we use a type trait to return correct SpineML type name
template<typename T>
struct SpineMLTypeName
{
};

template<>
struct SpineMLTypeName<float>
{
    static const char *name;
};

template<>
struct SpineMLTypeName<double>
{
    static const char *name;
};

// **YUCK** Visual C++ doesn't support constexpr so need to do this the old way
const char *SpineMLTypeName<float>::name = "float";
const char *SpineMLTypeName<double>::name = "double";
}

//----------------------------------------------------------------------------
// SpineMLSimulator::Base::Base
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::Base::Base(const pugi::xml_node &node, double dt, unsigned int numTimeSteps)
{
    std::cout << "Log '" << node.attribute("name").value() << "'" << std::endl;

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
        m_EndTimeStep = numTimeSteps;
    }
    else {
        m_EndTimeStep = m_StartTimeStep + (unsigned int)std::ceil(durationAttr.as_double() / dt);
    }

    std::cout << "\tStart timestep:" << m_StartTimeStep << ", end timestep:" << m_EndTimeStep << std::endl;
}


//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutputAnalogue
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::Analogue::Analogue(const pugi::xml_node &node, double dt, unsigned int numTimeSteps,
                                                const std::string &port, unsigned int popSize,
                                                const filesystem::path &basePath,
                                                const ModelProperty::Base *modelProperty)
    : Base(node, dt, numTimeSteps), m_ModelProperty(modelProperty)
{
    // If indices are specified
    auto indices = node.attribute("indices");
    if(indices) {
        // Read indices into vector
        SpineMLCommon::SpineMLUtils::readCSVIndices(indices.value(),
                                                    std::back_inserter(m_Indices));

        std::cout << "\tRecording " << m_Indices.size() << " values" << std::endl;

        // Resize output buffer to match indices
        m_OutputBuffer.resize(m_Indices.size());
    }

    // Combine node target and logger names to get file title
    std::string fileTitle = std::string(node.attribute("target").value()) + "_" + std::string(node.attribute("name").value());

    // Combine this with base path to get full file title
    std::string absoluteFileTitle = (basePath / fileTitle).str();

    // Create report document
    pugi::xml_document reportDoc;
    auto report = reportDoc.append_child("LogReport").append_child("AnalogLog");

    // Write standard report metadata here
    report.append_child("LogFile").text().set((fileTitle + ".bin").c_str());
    report.append_child("LogFileType").text().set("binary");
    report.append_child("LogEndTime").text().set((double)getEndTimestep() * dt);


    // If we're logging data from all neurons, add LogAll node to report
    if(m_Indices.empty()) {
        auto logAll = report.append_child("LogAll");
        logAll.append_attribute("size").set_value(popSize);
        logAll.append_attribute("headings").set_value(port.c_str());
        logAll.append_attribute("dims").set_value("");
        logAll.append_attribute("type").set_value(SpineMLTypeName<scalar>::name);
    }
    // Otherwise add LogCol node for each index
    else {
        for(unsigned int i : m_Indices) {
            auto logCol = report.append_child("LogCol");
            logCol.append_attribute("index").set_value(i);
            logCol.append_attribute("heading").set_value(port.c_str());
            logCol.append_attribute("dims").set_value("");
            logCol.append_attribute("type").set_value(SpineMLTypeName<scalar>::name);
        }
    }

    // Write timestamp
    report.append_child("TimeStep").append_attribute("dt").set_value(dt);

    // Save report
    reportDoc.save_file((absoluteFileTitle + "rep.xml").c_str());

    std::cout << "\tAnalogue log:" << absoluteFileTitle << ".bin" << std::endl;

    // Open file for binary writing
    m_File.open(absoluteFileTitle + ".bin", std::ios::binary);
}
//----------------------------------------------------------------------------
void SpineMLSimulator::LogOutput::Analogue::record(double, unsigned int timestep)
{
    // If we should be recording this timestep
    if(shouldRecord(timestep)) {
        // Pull state variable from device
        // **TODO** simple min/max index optimisation
        m_ModelProperty->pullFromDevice();

        // If no indices are specified, directly write out data from model property
        if(m_Indices.empty()) {
            m_File.write(reinterpret_cast<const char*>(m_ModelProperty->getHostStateVarBegin()), sizeof(scalar) * m_ModelProperty->getSize());
        }
        // Otherwise
        else {
            // Transform indexed variables into output buffer so they can be written in one call
            std::transform(m_Indices.begin(), m_Indices.end(), m_OutputBuffer.begin(),
                           [this](unsigned int i)
                           {
                               return m_ModelProperty->getHostStateVarBegin()[i];
                           });

            // Write output buffer to file
            m_File.write(reinterpret_cast<char*>(m_OutputBuffer.data()), sizeof(scalar) * m_OutputBuffer.size());
        }
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutputSpike
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::Event::Event(const pugi::xml_node &node, double dt, unsigned int numTimeSteps,
                                          const std::string &port, unsigned int popSize,
                                          const filesystem::path &basePath, unsigned int *spikeQueuePtr,
                                          unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
                                          unsigned int *hostSpikes, unsigned int *deviceSpikes)
    : Base(node, dt, numTimeSteps), m_PopSize(popSize), m_SpikeQueuePtr(spikeQueuePtr),
      m_HostSpikeCount(hostSpikeCount), m_DeviceSpikeCount(deviceSpikeCount),
      m_HostSpikes(hostSpikes), m_DeviceSpikes(deviceSpikes)
{
    // If indices are specified
    auto indices = node.attribute("indices");
    if(indices) {
        // Read indices into set
        SpineMLCommon::SpineMLUtils::readCSVIndices(indices.value(),
                                                    std::inserter(m_Indices, m_Indices.end()));

        std::cout << "\tRecording " << m_Indices.size() << " values" << std::endl;
    }

    // Combine node target and logger names to get file title
    std::string fileTitle = std::string(node.attribute("target").value()) + "_" + std::string(node.attribute("name").value());

    // Combine this with base path to get full file title
    std::string absoluteFileTitle = (basePath / fileTitle).str();

    // Create report document
    pugi::xml_document reportDoc;
    auto report = reportDoc.append_child("LogReport").append_child("EventLog");

    // Write standard report metadata here
    report.append_child("LogFile").text().set((fileTitle + ".csv").c_str());
    report.append_child("LogFileType").text().set("csv");
    report.append_child("LogPort").text().set(port.c_str());
    report.append_child("LogEndTime").text().set((double)getEndTimestep() * dt);

    // If we're logging events from all neurons, add LogAll node to report
    if(m_Indices.empty()) {
        auto logAll = report.append_child("LogAll");
        logAll.append_attribute("size").set_value(popSize);
        logAll.append_attribute("type").set_value("int");
        logAll.append_attribute("dims").set_value("");
    }
    // Otherwise add LogIndex node for each index
    else {
        for(unsigned int i : m_Indices) {
            report.append_child("LogIndex").text().set(i);
        }
    }

    // Add CSV time column
    auto logColT = report.append_child("LogCol");
    logColT.append_attribute("heading").set_value("t");
    logColT.append_attribute("dims").set_value("ms");
    logColT.append_attribute("type").set_value("double");

    // Add CSV neuron index column
    auto logColIndex = report.append_child("LogCol");
    logColIndex.append_attribute("heading").set_value("index");
    logColIndex.append_attribute("dims").set_value("");
    logColIndex.append_attribute("type").set_value("int");

    // Write timestamp
    report.append_child("TimeStep").append_attribute("dt").set_value(dt);

    // Save report
    reportDoc.save_file((absoluteFileTitle + "rep.xml").c_str());

    std::cout << "\tEvent log:" << absoluteFileTitle << ".csv" << std::endl;

    // Open CSV file
    m_File.open(absoluteFileTitle + ".csv");
}
//----------------------------------------------------------------------------
void SpineMLSimulator::LogOutput::Event::record(double dt, unsigned int timestep)
{
    // If we should be recording this timestep
    if(shouldRecord(timestep)) {
        // Determine current spike queue
        const unsigned int spikeQueueIndex = (m_SpikeQueuePtr == nullptr) ? 0 : *m_SpikeQueuePtr;
        const unsigned int spikeOffset = m_PopSize * spikeQueueIndex;
#ifndef CPU_ONLY
        // Copy spike count from spike queue position to host
        CHECK_CUDA_ERRORS(cudaMemcpy(&m_HostSpikeCount[spikeQueueIndex], &m_DeviceSpikeCount[spikeQueueIndex],
                                     sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // Copy this many spikes to host
        CHECK_CUDA_ERRORS(cudaMemcpy(&m_HostSpikes[spikeOffset], &m_DeviceSpikes[spikeOffset],
                                     sizeof(unsigned int) * m_HostSpikeCount[spikeQueueIndex], cudaMemcpyDeviceToHost));
#endif  // CPU_ONLY
        const double t = dt * (double)timestep;

        if(m_Indices.empty()) {
            for(unsigned int i = 0; i < m_HostSpikeCount[spikeQueueIndex]; i++)
            {
                m_File << t << "," << m_HostSpikes[spikeOffset + i] << std::endl;
            }
        }
        else {
            for(unsigned int i = 0; i < m_HostSpikeCount[spikeQueueIndex]; i++)
            {
                const unsigned int spikeID = m_HostSpikes[spikeOffset + i];
                if(m_Indices.find(spikeID) != m_Indices.end()) {
                    m_File << t << "," << spikeID << std::endl;
                }
            }
        }
    }
}
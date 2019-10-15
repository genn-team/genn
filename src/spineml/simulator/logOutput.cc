#include "logOutput.h"

// Standard C++ includes
#include <algorithm>
#include <limits>
#include <iostream>

// Standard C includes
#include <cassert>
#include <cmath>

// Filesystem includes
#include "path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// PLOG includes
#include <plog/Log.h>
#include <plog/Appenders/ConsoleAppender.h>

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

/*template<>
struct SpineMLTypeName<double>
{
    static const char *name;
};*/

// **YUCK** Visual C++ doesn't support constexpr so need to do this the old way
const char *SpineMLTypeName<float>::name = "float";
//const char *SpineMLTypeName<double>::name = "double";
}

//----------------------------------------------------------------------------
// SpineMLSimulator::Base::Base
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::Base::Base(const pugi::xml_node &node, double dt)
{
    LOGI << "Log '" << node.attribute("name").value() << "'";

    // Read start time
    auto startAttr = node.attribute("start_time");
    if(startAttr.empty()) {
        m_StartTimeStep = 0;
    }
    else {
        m_StartTimeStep = (unsigned long long)std::ceil(startAttr.as_double() / dt);
    }

    // Read duration
    auto durationAttr = node.attribute("duration");
    if(durationAttr.empty()) {
        m_EndTimeStep = std::numeric_limits<unsigned long long>::max();
    }
    else {
        m_EndTimeStep = m_StartTimeStep + (unsigned long long)std::ceil(durationAttr.as_double() / dt);
    }

    LOGD << "\tStart timestep:" << m_StartTimeStep << ", end timestep:" << m_EndTimeStep;
}

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutput::AnalogueBase
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::AnalogueBase::AnalogueBase(const pugi::xml_node &node, double dt,
                                                        const ModelProperty::Base *modelProperty)
    : Base(node, dt), m_ModelProperty(modelProperty)
{
    // If indices are specified
    auto indices = node.attribute("indices");
    if(indices) {
        // Read indices into vector
        SpineMLCommon::SpineMLUtils::readCSVIndices(indices.value(),
                                                    std::back_inserter(m_Indices));

        LOGD << "\tRecording " << m_Indices.size() << " values";
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutput::AnalogueFile
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::AnalogueFile::AnalogueFile(const pugi::xml_node &node, double dt, unsigned long long numTimeSteps,
                                                        const std::string &port, unsigned int popSize,
                                                        const filesystem::path &logPath,
                                                        const ModelProperty::Base *modelProperty)
    : AnalogueBase(node, dt, modelProperty)
{
    // If indices are specified, allocate output buffer to match indices
    if(!getIndices().empty()) {
        m_OutputBuffer.resize(getIndices().size());
    }

    // Combine node target and logger names to get file title
    std::string fileTitle = std::string(node.attribute("target").value()) + "_" + std::string(node.attribute("port").value());

    // Combine this with base path to get full file title
    std::string absoluteFileTitle = (logPath / fileTitle).str();

    // Create report document
    pugi::xml_document reportDoc;
    auto report = reportDoc.append_child("LogReport").append_child("AnalogLog");

    // Write standard report metadata here
    report.append_child("LogFile").text().set((fileTitle + "_log.bin").c_str());
    report.append_child("LogFileType").text().set("binary");
    report.append_child("LogEndTime").text().set((double)numTimeSteps * dt);

    // If we're logging data from all neurons, add LogAll node to report
    if(getIndices().empty()) {
        auto logAll = report.append_child("LogAll");
        logAll.append_attribute("size").set_value(popSize);
        logAll.append_attribute("headings").set_value(port.c_str());
        logAll.append_attribute("dims").set_value("");
        logAll.append_attribute("type").set_value(SpineMLTypeName<scalar>::name);
    }
    // Otherwise add LogCol node for each index
    else {
        for(unsigned int i : getIndices()) {
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
    reportDoc.save_file((absoluteFileTitle + "_logrep.xml").c_str());

    LOGD << "\tAnalogue file log:" << absoluteFileTitle << "_log.bin";

    // Open file for binary writing
    m_File.open(absoluteFileTitle + "_log.bin", std::ios::binary);
}
//----------------------------------------------------------------------------
void SpineMLSimulator::LogOutput::AnalogueFile::record(double, unsigned long long timestep)
{
    // If we should be recording this timestep
    if(shouldRecord(timestep)) {
        // Pull state variable from device
        // **TODO** simple min/max index optimisation
        pullModelPropertyFromDevice();

        // If no indices are specified, directly write out data from model property
        if(getIndices().empty()) {
            m_File.write(reinterpret_cast<const char*>(getStateVarBegin()), sizeof(scalar) * getModelPropertySize());
        }
        // Otherwise
        else {
            // Transform indexed variables into output buffer so they can be written in one call
            std::transform(getIndices().begin(), getIndices().end(), m_OutputBuffer.begin(),
                           [this](unsigned int i)
                           {
                               return getStateVarBegin()[i];
                           });

            // Write output buffer to file
            m_File.write(reinterpret_cast<char*>(m_OutputBuffer.data()), sizeof(scalar) * m_OutputBuffer.size());
        }
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutput::AnalogueExternal
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::AnalogueExternal::AnalogueExternal(const pugi::xml_node &node, double dt,
                                                                const std::string&, unsigned int,
                                                                const filesystem::path&,
                                                                const ModelProperty::Base *modelProperty)
    : AnalogueBase(node, dt, modelProperty), m_CurrentIntervalTimesteps(0)
{
    // If external timestep is zero then send every timestep
    const double externalTimestepMs = node.attribute("timestep").as_double();
    if(externalTimestepMs == 0.0) {
        m_IntervalTimesteps = 0;
    }
    // Otherwise
    else {
        // Check we're not trying to use an external timestep smaller than GeNN timestep
        assert(externalTimestepMs >= dt);

        // Calculate how many GeNN timesteps to count down before logging
        // **NOTE** subtract one because we are checking BEFORE we subtract
        m_IntervalTimesteps = ((unsigned int)std::round(externalTimestepMs / dt)) - 1;
        LOGD << "\tExternal timestep:" << externalTimestepMs << "ms - interval:" << m_IntervalTimesteps;
    }
}
//----------------------------------------------------------------------------
void SpineMLSimulator::LogOutput::AnalogueExternal::record(double, unsigned long long timestep)
{
    // If we should be recording this timestep
    if(shouldRecord(timestep)) {
        // If we should transmit this timestep
        if(m_CurrentIntervalTimesteps == 0) {
            // Pull state variable from device
            // **TODO** simple min/max index optimisation
            pullModelPropertyFromDevice();

            // Perform additional recording logic
            recordInternal();

            // Reset interval
            m_CurrentIntervalTimesteps = m_IntervalTimesteps;
        }
        // Otherwise decrement interval
        else {
            m_CurrentIntervalTimesteps--;
        }
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutput::AnalogueNetwork
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::AnalogueNetwork::AnalogueNetwork(const pugi::xml_node &node, double dt,
                                                              const std::string &port, unsigned int popSize,
                                                              const filesystem::path &logPath,
                                                              const ModelProperty::Base *modelProperty)
    : AnalogueExternal(node, dt, port, popSize, logPath, modelProperty)
{
    // Check size determined by indices/population size matches attribute
    const unsigned int size = getIndices().empty() ? popSize : (unsigned int)getIndices().size();
    //assert(size == node.attribute("size").as_uint());

    // Allocate output buffer
    m_OutputBuffer.resize(size);

    // Read connection stats
    const std::string connectionName = node.attribute("name").value();
    const std::string hostname = node.attribute("host").value();
    const unsigned int tcpPort = node.attribute("tcp_port").as_uint();

    LOGD << "\tAnalogue network log '" << connectionName << "' (" << hostname << ":" << tcpPort << ")";

    // Attempt to connect network client
    if(!m_Client.connect(hostname, tcpPort, size, NetworkClient::DataType::Analogue,
        NetworkClient::Mode::Source, connectionName))
    {
        throw std::runtime_error("Cannot connect network client");
    }
}
//----------------------------------------------------------------------------
void SpineMLSimulator::LogOutput::AnalogueNetwork::recordInternal()
{
    // If no indices are specified, transform all values in model property into double precision
    // **TODO** once precision is switchable this could be optimised out
    if(getIndices().empty()) {
        std::transform(getStateVarBegin(), getStateVarEnd(), m_OutputBuffer.begin(),
                    [](scalar x)
                    {
                        return static_cast<double>(x);
                    });
    }
    // Otherwise, transform indexed variables into output buffer
    else {
        std::transform(getIndices().begin(), getIndices().end(), m_OutputBuffer.begin(),
                    [this](unsigned int i)
                    {
                        return static_cast<double>(getStateVarBegin()[i]);
                    });
    }

    // Send output data over network client
    if(!m_Client.send(m_OutputBuffer)) {
        throw std::runtime_error("Cannot send data to socket");
    }
}


//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutput::Event
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::Event::Event(const pugi::xml_node &node, double dt, unsigned long long numTimeSteps,
                                          const std::string &port, unsigned int popSize,
                                          const filesystem::path &logPath, unsigned int *spikeQueuePtr,
                                          unsigned int *hostSpikeCount, unsigned int *hostSpikes,
                                          void (*pullCurrentSpikesFunc)(void))
    : Base(node, dt), m_PopSize(popSize), m_SpikeQueuePtr(spikeQueuePtr),
      m_HostSpikeCount(hostSpikeCount), m_HostSpikes(hostSpikes), m_PullCurrentSpikesFunc(pullCurrentSpikesFunc)
{
    // If indices are specified
    auto indices = node.attribute("indices");
    if(indices) {
        // Read indices into set
        SpineMLCommon::SpineMLUtils::readCSVIndices(indices.value(),
                                                    std::inserter(m_Indices, m_Indices.end()));

        LOGD << "\tRecording " << m_Indices.size() << " values";
    }

    // Combine node target and logger names to get file title
    std::string fileTitle = std::string(node.attribute("target").value()) + "_" + std::string(node.attribute("port").value());

    // Combine this with base path to get full file title
    std::string absoluteFileTitle = (logPath / fileTitle).str();

    // Create report document
    pugi::xml_document reportDoc;
    auto report = reportDoc.append_child("LogReport").append_child("EventLog");

    // Write standard report metadata here
    report.append_child("LogFile").text().set((fileTitle + "_log.csv").c_str());
    report.append_child("LogFileType").text().set("csv");
    report.append_child("LogPort").text().set(port.c_str());
    report.append_child("LogEndTime").text().set((double)numTimeSteps * dt);

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
    reportDoc.save_file((absoluteFileTitle + "_logrep.xml").c_str());

    LOGD << "\tEvent log:" << absoluteFileTitle << ".csv";

    // Open CSV file
    m_File.open(absoluteFileTitle + "_log.csv");
}
//----------------------------------------------------------------------------
void SpineMLSimulator::LogOutput::Event::record(double dt, unsigned long long timestep)
{
    // If we should be recording this timestep
    if(shouldRecord(timestep)) {
        // Determine current spike queue
        const unsigned int spikeQueueIndex = (m_SpikeQueuePtr == nullptr) ? 0 : *m_SpikeQueuePtr;
        const unsigned int spikeOffset = m_PopSize * spikeQueueIndex;

        // Pull current spikes from device
        m_PullCurrentSpikesFunc();

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

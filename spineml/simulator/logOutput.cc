#include "logOutput.h"

// Standard C++ includes
#include <limits>
#include <iostream>

// Standard C includes
#include <cmath>

// Filesystem includes
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutput
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutput::LogOutput(const pugi::xml_node &node, double dt, const filesystem::path &basePath)
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
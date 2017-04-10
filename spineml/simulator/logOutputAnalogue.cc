#include "logOutputAnalogue.h"

// Standard C++ includes
#include <algorithm>
#include <iostream>
#include <sstream>

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineML simulator includes
#include "modelProperty.h"

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutputAnalogue
//----------------------------------------------------------------------------
SpineMLSimulator::LogOutputAnalogue::LogOutputAnalogue(const pugi::xml_node &node, double dt,
                                                       const ModelProperty *modelProperty)
    : LogOutput(node, dt), m_ModelProperty(modelProperty)
{
    // If indices are specified
    auto indices = node.attribute("indices");
    if(indices) {
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
void SpineMLSimulator::LogOutputAnalogue::record(double dt, unsigned int timestep)
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
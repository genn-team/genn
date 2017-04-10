#include "modelPropertyFixed.h"

// Standard C++ includes
#include <algorithm>
#include <iostream>

// pugixml includes
#include "pugixml/pugixml.hpp"

//------------------------------------------------------------------------
// SpineMLSimulator::ModelPropertyFixed
//------------------------------------------------------------------------
SpineMLSimulator::ModelPropertyFixed::ModelPropertyFixed(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
    : ModelProperty(hostStateVar, deviceStateVar, size)
{
    setValue(node.attribute("value").as_double());
    std::cout << "\t\t\tFixed value:" << m_Value << std::endl;
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelPropertyFixed::setValue(scalar value)
{
    // Cache value
    m_Value = value;

    // Fill host state variable
    std::fill(getHostStateVarBegin(), getHostStateVarEnd(), m_Value);

    // Push to device
    pushToDevice();
}
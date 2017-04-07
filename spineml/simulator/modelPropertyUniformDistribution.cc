#include "modelPropertyUniformDistribution.h"

// Standard C++ includes
#include <algorithm>
#include <iostream>

// pugixml includes
#include "pugixml/pugixml.hpp"

//------------------------------------------------------------------------
// SpineMLSimulator::ModelPropertyUniformDistribution
//------------------------------------------------------------------------
SpineMLSimulator::ModelPropertyUniformDistribution::ModelPropertyUniformDistribution(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
    : ModelProperty(hostStateVar, deviceStateVar, size), m_RandomGenerator(m_RandomDevice())
{
    SetValue(node.attribute("minimum").as_double(), node.attribute("maximum").as_double());
    std::cout << "\t\t\tMin value:" << m_Distribution.min() << ", Max value:" << m_Distribution.max() << std::endl;
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelPropertyUniformDistribution::SetValue(scalar min, scalar max)
{
    // Create distribution
    m_Distribution = std::uniform_real_distribution<scalar>(min, max);

    // Generate uniformly distributed numbers to fill host array
    std::generate(getHostStateVarBegin(), getHostStateVarEnd(),
        [this](){ return m_Distribution(m_RandomGenerator); });

    // Push to device
    PushToDevice();
}
//------------------------------------------------------------------------
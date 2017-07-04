#include "modelProperty.h"

// Standard C++ includes
#include <algorithm>
#include <iostream>

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "utils.h"

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::Base
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::Base::pushToDevice() const
{
#ifndef CPU_ONLY
    CHECK_CUDA_ERRORS(cudaMemcpy(m_DeviceStateVar, m_HostStateVar, m_Size * sizeof(scalar), cudaMemcpyHostToDevice));
#endif  // CPU_ONLY
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::Base::pullFromDevice() const
{
#ifndef CPU_ONLY
    CHECK_CUDA_ERRORS(cudaMemcpy(m_HostStateVar, m_DeviceStateVar, m_Size * sizeof(scalar), cudaMemcpyDeviceToHost));
#endif  // CPU_ONLY
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::Fixed
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::Fixed::Fixed(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
    : Base(hostStateVar, deviceStateVar, size)
{
    setValue(node.attribute("value").as_double());
    std::cout << "\t\t\tFixed value:" << m_Value << std::endl;
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::Fixed::setValue(scalar value)
{
    // Cache value
    m_Value = value;

    // Fill host state variable
    std::fill(getHostStateVarBegin(), getHostStateVarEnd(), m_Value);

    // Push to device
    pushToDevice();
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::ValueList
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::ValueList::ValueList(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
    : Base(hostStateVar, deviceStateVar, size)
{
    // Copy values into vector
    std::vector<scalar> values(size);
    for(const auto v : node.children("Value")) {
        values[v.attribute("index").as_uint()] = v.attribute("value").as_double();
    }

    setValue(values);
    std::cout << "\t\t\tValue list" << std::endl;
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::ValueList::setValue(const std::vector<scalar> &values)
{
    // Cache value
    m_Values = values;

    // Copy vector of values into state variable
    std::copy(m_Values.begin(), m_Values.end(),
              getHostStateVarBegin());

    // Push to device
    pushToDevice();
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::UniformDistribution
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::UniformDistribution::UniformDistribution(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
    : Base(hostStateVar, deviceStateVar, size)
{
    setValue(node.attribute("minimum").as_double(), node.attribute("maximum").as_double());
    std::cout << "\t\t\tMin value:" << m_Distribution.min() << ", Max value:" << m_Distribution.max() << std::endl;

    // Seed RNG if required
    auto seed = node.attribute("seed");
    if(seed) {
        m_RandomGenerator.seed(seed.as_uint());
        std::cout << "\t\t\tSeed:" << seed.as_uint() << std::endl;
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::UniformDistribution::setValue(scalar min, scalar max)
{
    // Create distribution
    m_Distribution = std::uniform_real_distribution<scalar>(min, max);

    // Generate uniformly distributed numbers to fill host array
    std::generate(getHostStateVarBegin(), getHostStateVarEnd(),
        [this](){ return m_Distribution(m_RandomGenerator); });

    // Push to device
    pushToDevice();
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::NormalDistribution
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::NormalDistribution::NormalDistribution(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
    : Base(hostStateVar, deviceStateVar, size)
{
    setValue(node.attribute("mean").as_double(), node.attribute("variance").as_double());
    std::cout << "\t\t\tMean:" << m_Distribution.min() << ", Variance:" << m_Distribution.stddev() * m_Distribution.stddev() << std::endl;

    // Seed RNG if required
    auto seed = node.attribute("seed");
    if(seed) {
        m_RandomGenerator.seed(seed.as_uint());
        std::cout << "\t\t\tSeed:" << seed.as_uint() << std::endl;
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::NormalDistribution::setValue(scalar mean, scalar variance)
{
    // Create distribution
    m_Distribution = std::normal_distribution<scalar>(mean, std::sqrt(variance));

    // Generate uniformly distributed numbers to fill host array
    std::generate(getHostStateVarBegin(), getHostStateVarEnd(),
        [this](){ return m_Distribution(m_RandomGenerator); });

    // Push to device
    pushToDevice();
}
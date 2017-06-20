#include "inputValue.h"

// Standard C++ includes
#include <iostream>
#include <sstream>

// Standard C includes
#include <cmath>

// pugixml includes
#include "pugixml/pugixml.hpp"

//------------------------------------------------------------------------
// SpineMLSimulator::InputValue::Base
//------------------------------------------------------------------------
SpineMLSimulator::InputValue::Base::Base(const pugi::xml_node &node)
{
    // If indices are specified
    auto targetIndices = node.attribute("indices");
    if(targetIndices) {
        // **TODO** maybe move somewhere common
        std::stringstream targetIndicesStream(targetIndices.value());
        while(targetIndicesStream.good()) {
            std::string index;
            std::getline(targetIndicesStream, index, ',');
            m_TargetIndices.push_back(std::stoul(index));
        }

        std::cout << "\tTargetting " << m_TargetIndices.size() << " neurons" << std::endl;
    }
}

//------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ScalarBase
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::ScalarBase::applyScalar(unsigned int numNeurons, double value,
                                                           std::function<void(unsigned int, double)> applyValueFunc) const
{
     // If we have no target indices, apply constant value to all neurons
    if(getTargetIndices().empty()) {
        for(unsigned int i = 0; i < numNeurons; i++) {
            applyValueFunc(i, value);
        }
    }
    // Otherwise, apply to those in target indices
    else {
        for(unsigned int i : getTargetIndices()) {
            applyValueFunc(i, value);
        }
    }
}

//------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ArrayBase
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::ArrayBase::applyArray(unsigned int numNeurons, const std::vector<double> &values,
                                                          std::function<void(unsigned int, double)> applyValueFunc) const
{
    // If we have no target indices, apply array values to each neuron
    if(getTargetIndices().empty()) {
        for(unsigned int i = 0; i < numNeurons; i++) {
            applyValueFunc(i, values[i]);
        }
    }
    // Otherwise, apply to each target index
    else {
        for(unsigned int i = 0; i < getTargetIndices().size(); i++) {
            applyValueFunc(getTargetIndices()[i], values[i]);
        }
    }
}
//------------------------------------------------------------------------
bool SpineMLSimulator::InputValue::ArrayBase::checkArrayDimensions(unsigned int numNeurons, const std::vector<double> &values) const
{
    // If there are no target indices, check number of values matches number of neurons
    if(getTargetIndices().empty() && values.size() != numNeurons) {
        return false;
    }
    // Otherwise check number of values matches number of target indices
    else if(!getTargetIndices().empty() && values.size() != getTargetIndices().size()) {
        return false;
    }
    else {
        return true;
    }
}


//------------------------------------------------------------------------
// SpineMLSimulator::InputValue::Constant
//------------------------------------------------------------------------
SpineMLSimulator::InputValue::Constant::Constant(double, unsigned int, const pugi::xml_node &node)
: ScalarBase(node)
{
    m_Value = node.attribute("value").as_double();
    std::cout << "\tConstant value:" << m_Value << std::endl;
}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::Constant::updateValues(double, unsigned int timestep, unsigned int numNeurons,
                                                          std::function<void(unsigned int, double)> applyValueFunc) const
{
    // If this is the first timestep, apply constant value
    if(timestep == 0) {
        applyScalar(numNeurons, m_Value, applyValueFunc);
    }
}

//------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ConstantArray
//------------------------------------------------------------------------
SpineMLSimulator::InputValue::ConstantArray::ConstantArray(double, unsigned int numNeurons, const pugi::xml_node &node)
: ArrayBase(node)
{
    // If array values are specified
    auto arrayValue = node.attribute("array_value");
    if(arrayValue) {
        std::stringstream arrayValueStream(arrayValue.value());
        while(arrayValueStream.good()) {
            std::string value;
            std::getline(arrayValueStream, value, ',');
            m_Values.push_back(std::stod(value));
        }

        // Check dimensions of array
        if(!checkArrayDimensions(numNeurons, m_Values)) {
            throw std::runtime_error("Number of values passed to ConstantArrayInput does not match target indices/population size");
        }

        std::cout << "\tSpecified " << m_Values.size() << " constant values" << std::endl;
    }

}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::ConstantArray::updateValues(double, unsigned int timestep, unsigned int numNeurons,
                                                               std::function<void(unsigned int, double)> applyValueFunc) const
{
    // If this is the first timestep, apply array of values
    if(timestep == 0) {
        applyArray(numNeurons, m_Values, applyValueFunc);
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::TimeVarying
//----------------------------------------------------------------------------
SpineMLSimulator::InputValue::TimeVarying::TimeVarying(double dt, unsigned int, const pugi::xml_node &node)
: ScalarBase(node)
{
    // Loop through time points
    for(auto timePoint : node.children("TimePointValue")) {
        // Read time and value
        const double time = timePoint.attribute("time").as_double();
        const double value = timePoint.attribute("value").as_double();

        // Convert time to integer timestep and add timestep and associated value to map
        unsigned int timeStep = (unsigned int)std::floor(time / dt);
        m_TimeValues.insert(std::make_pair(timeStep, value));
    }

}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::TimeVarying::updateValues(double, unsigned int timestep, unsigned int numNeurons,
                                                             std::function<void(unsigned int, double)> applyValueFunc) const
{
    // If there is a time value to apply at this timestep, do so
    auto timeValue = m_TimeValues.find(timestep);
    if(timeValue != m_TimeValues.end()) {
        applyScalar(numNeurons, timeValue->second, applyValueFunc);
    }
}
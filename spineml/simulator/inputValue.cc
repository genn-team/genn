#include "inputValue.h"

// Standard C++ includes
#include <iostream>
#include <sstream>

// Standard C includes
#include <cmath>

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineML common includes
#include "spineMLUtils.h"

//------------------------------------------------------------------------
// SpineMLSimulator::InputValue::Base
//------------------------------------------------------------------------
SpineMLSimulator::InputValue::Base::Base(unsigned int numNeurons, const pugi::xml_node &node) : m_NumNeurons(numNeurons)
{
    std::cout << "Input value:" << std::endl;

    // If indices are specified
    auto targetIndices = node.attribute("indices");
    if(targetIndices) {
        SpineMLCommon::SpineMLUtils::readCSVIndices(targetIndices.value(),
                                                    std::back_inserter(m_TargetIndices));
        std::cout << "\tTargetting " << m_TargetIndices.size() << " neurons" << std::endl;
    }
}

//------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ScalarBase
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::ScalarBase::applyScalar(double value,
                                                           std::function<void(unsigned int, double)> applyValueFunc) const
{
     // If we have no target indices, apply constant value to all neurons
    if(getTargetIndices().empty()) {
        for(unsigned int i = 0; i < getNumNeurons(); i++) {
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
// SpineMLSimulator::InputValue::Constant
//------------------------------------------------------------------------
SpineMLSimulator::InputValue::Constant::Constant(double, unsigned int numNeurons, const pugi::xml_node &node)
: ScalarBase(numNeurons, node)
{
    m_Value = node.attribute("value").as_double();
    std::cout << "\tConstant value:" << m_Value << std::endl;
}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::Constant::update(double, unsigned int timestep,
                                                    std::function<void(unsigned int, double)> applyValueFunc) const
{
    // If this is the first timestep, apply constant value
    if(timestep == 0) {
        applyScalar(m_Value, applyValueFunc);
    }
}

//------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ConstantArray
//------------------------------------------------------------------------
SpineMLSimulator::InputValue::ConstantArray::ConstantArray(double, unsigned int numNeurons, const pugi::xml_node &node)
: Base(numNeurons, node)
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

         // If there are no target indices, check number of values matches number of neurons
        if(getTargetIndices().empty() && m_Values.size() != numNeurons) {
            throw std::runtime_error("Number of values passed to ConstantArrayInput does not match population size");
        }
        // Otherwise check number of values matches number of target indices
        else if(!getTargetIndices().empty() && m_Values.size() != getTargetIndices().size()) {
            throw std::runtime_error("Number of values passed to ConstantArrayInput does not match target indices size");
        }

        std::cout << "\tSpecified " << m_Values.size() << " constant values" << std::endl;
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::ConstantArray::update(double, unsigned int timestep,
                                                         std::function<void(unsigned int, double)> applyValueFunc) const
{
    // If this is the first timestep, apply array of values
    if(timestep == 0) {
        // If we have no target indices, apply array values to each neuron
        if(getTargetIndices().empty()) {
            for(unsigned int i = 0; i < getNumNeurons(); i++) {
                applyValueFunc(i, m_Values[i]);
            }
        }
        // Otherwise, apply to each target index
        else {
            for(unsigned int i = 0; i < getTargetIndices().size(); i++) {
                applyValueFunc(getTargetIndices()[i], m_Values[i]);
            }
        }
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::TimeVarying
//----------------------------------------------------------------------------
SpineMLSimulator::InputValue::TimeVarying::TimeVarying(double dt, unsigned int numNeurons, const pugi::xml_node &node)
: ScalarBase(numNeurons, node)
{
    // Loop through time points
    for(auto timePoint : node.children("TimePointValue")) {
        // Read time and value
        const double time = timePoint.attribute("time").as_double();
        const double value = timePoint.attribute("value").as_double();

        // Convert time to integer timestep and add timestep and associated value to map
        unsigned int timestep = (unsigned int)std::floor(time / dt);
        m_TimeValues.insert(std::make_pair(timestep, value));

        std::cout << "\tTime:" << time << "(timestep:" << timestep << "), value:" << value << std::endl;
    }

}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::TimeVarying::update(double, unsigned int timestep,
                                                       std::function<void(unsigned int, double)> applyValueFunc) const
{
    // If there is a time value to apply at this timestep, do so
    auto timeValue = m_TimeValues.find(timestep);
    if(timeValue != m_TimeValues.end()) {
        std::cout << "\tTimestep:" << timestep << ", applying:" << timeValue->second << std::endl;
        applyScalar(timeValue->second, applyValueFunc);
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::TimeVaryingArray
//----------------------------------------------------------------------------
SpineMLSimulator::InputValue::TimeVaryingArray::TimeVaryingArray(double dt, unsigned int numNeurons, const pugi::xml_node &node)
: Base(numNeurons, node)
{
    /*<TimePointArrayValue index="0" array_time="0,10" array_value="10,20"/>*/
    // Loop through time points
    for(auto timePoint : node.children("TimePointArrayValue")) {
        // Read time and value
        const unsigned int index = timePoint.attribute("index").as_uint();

        // Read array of times
        std::vector<unsigned int> times;
        auto arrayTime = timePoint.attribute("array_time");
        if(arrayTime) {
            SpineMLCommon::SpineMLUtils::readCSVIndices(arrayTime.value(),
                                                        std::back_inserter(times));
        }
        else {
            throw std::runtime_error("No array of times specified in TimePointArrayValue");
        }

        // If an array of values is specified
        auto arrayValue = timePoint.attribute("array_value");
        if(arrayValue) {
            // Read value array
            std::vector<double> values;
            SpineMLCommon::SpineMLUtils::readCSVValues(arrayValue.value(),
                                                       std::back_inserter(values));

            // Check sizes match
            if(times.size() != values.size()) {
                throw std::runtime_error("Number of times and values specified in each TimePointArrayValue must match");
            }

            // Loop through times and values
            std::vector<unsigned int>::const_iterator t;
            std::vector<double>::const_iterator v;
            for(t = times.cbegin(), v = values.cbegin(); t != times.cend() && v != values.end(); ++t, ++v) {
                // If there is not yet an entry for this time step, insert new array into map
                auto timeArray = m_TimeArrays.find(*t);
                if(timeArray == m_TimeArrays.end()) {
                    m_TimeArrays.insert(std::make_pair(*t, NeuronValueVec({std::make_pair(index, *v)})));
                }
                // Otherwise add index and value
                else {
                    timeArray->second.push_back(std::make_pair(index, *v));
                }
            }
        }
        // Otherwise
        else {
            // Loop through times
            for(unsigned int t : times) {
                // If there is not yet an entry for this time step, insert new array into map
                auto timeArray = m_TimeArrays.find(t);
                if(timeArray == m_TimeArrays.end()) {
                    m_TimeArrays.insert(std::make_pair(t, NeuronValueVec({std::make_pair(index, 0.0)})));
                }
                // Otherwise add index and zero value
                else {
                    timeArray->second.push_back(std::make_pair(index, 0.0));
                }
            }
        }
    }

    for(const auto t : m_TimeArrays) {
        std::cout << "\tTimestep:" << t.first << "," << t.second.size() << " values " << std::endl;
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::TimeVaryingArray::update(double, unsigned int timestep,
                                                            std::function<void(unsigned int, double)> applyValueFunc) const
{
    // If there is a time value to apply at this timestep, do so
    auto timeValue = m_TimeArrays.find(timestep);
    if(timeValue != m_TimeArrays.end()) {
        std::cout << "\tTimestep:" << timestep << ", applying " << timeValue->second.size() << " values" << std::endl;
        for(const auto &value : timeValue->second) {
            applyValueFunc(value.first, value.second);
        }
    }
}
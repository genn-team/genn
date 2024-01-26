#include "inputValue.h"

// Standard C++ includes
#include <iostream>
#include <sstream>

// Standard C includes
#include <cassert>
#include <cmath>
#include <cstring>

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineML common includes
#include "spineMLLogging.h"
#include "spineMLUtils.h"

//------------------------------------------------------------------------
// SpineMLSimulator::InputValue::Base
//------------------------------------------------------------------------
SpineMLSimulator::InputValue::Base::Base(unsigned int numNeurons, const pugi::xml_node &node) : m_NumNeurons(numNeurons)
{
    // If indices are specified
    auto targetIndices = node.attribute("target_indices");
    if(targetIndices) {
        SpineMLCommon::SpineMLUtils::readCSVIndices(targetIndices.value(),
                                                    std::back_inserter(m_TargetIndices));
        LOGD_SPINEML << "\tTargetting " << m_TargetIndices.size() << " neurons";
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
    LOGD_SPINEML << "\tConstant value:" << m_Value;
}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::Constant::update(double, unsigned long long timestep,
                                                    std::function<void(unsigned int, double)> applyValueFunc)
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

        LOGD_SPINEML << "\tSpecified " << m_Values.size() << " constant values";
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::ConstantArray::update(double, unsigned long long timestep,
                                                         std::function<void(unsigned int, double)> applyValueFunc)
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

        LOGD_SPINEML << "\tTime:" << time << "(timestep:" << timestep << "), value:" << value;
    }

}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::TimeVarying::update(double, unsigned long long timestep,
                                                       std::function<void(unsigned int, double)> applyValueFunc)
{
    // If there is a time value to apply at this timestep, do so
    auto timeValue = m_TimeValues.find(timestep);
    if(timeValue != m_TimeValues.end()) {
        LOGD_SPINEML << "\tTimestep:" << timestep << ", applying:" << timeValue->second;
        applyScalar(timeValue->second, applyValueFunc);
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::TimeVaryingArray
//----------------------------------------------------------------------------
SpineMLSimulator::InputValue::TimeVaryingArray::TimeVaryingArray(double dt, unsigned int numNeurons, const pugi::xml_node &node)
: Base(numNeurons, node)
{
    // Loop through time points
    for(auto timePoint : node.children("TimePointArrayValue")) {
        // Read time and value
        const unsigned int index = timePoint.attribute("index").as_uint();

        // If an array of times is specified
        std::vector<unsigned int> times;
        auto arrayTime = timePoint.attribute("array_time");
        if(arrayTime) {
            // Read array of times in milliseconds
            std::vector<double> timesMs;
            SpineMLCommon::SpineMLUtils::readCSVValues(arrayTime.value(),
                                                       std::back_inserter(timesMs));

            // Convert milliseconds to timesteps
            times.reserve(timesMs.size());
            std::transform(timesMs.cbegin(), timesMs.cend(), std::back_inserter(times),
                           [dt](double t)
                           {
                               return (unsigned int)std::floor(t / dt);
                           });
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

    for(const auto &t : m_TimeArrays) {
        LOGD_SPINEML << "\tTimestep:" << t.first << "," << t.second.size() << " values";
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::TimeVaryingArray::update(double, unsigned long long timestep,
                                                            std::function<void(unsigned int, double)> applyValueFunc)
{
    // If there is a time value to apply at this timestep, do so
    auto timeValue = m_TimeArrays.find(timestep);
    if(timeValue != m_TimeArrays.end()) {
        LOGD_SPINEML << "\tTimestep:" << timestep << ", applying " << timeValue->second.size() << " values";
        for(const auto &value : timeValue->second) {
            applyValueFunc(value.first, value.second);
        }
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::External
//----------------------------------------------------------------------------
SpineMLSimulator::InputValue::External::External(double dt, unsigned int numNeurons, const pugi::xml_node &node)
: Base(numNeurons, node), m_CurrentIntervalTimesteps(0)
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
        LOGD_SPINEML << "\tExternal timestep:" << externalTimestepMs << "ms - interval:" << m_IntervalTimesteps;
    }

    // Resize buffer
    m_Buffer.resize(getSize());
}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::External::update(double, unsigned long long,
                                                    std::function<void(unsigned int, double)> applyValueFunc)
{
    // If we should update this timestep
    if(m_CurrentIntervalTimesteps == 0) {
        // Perform additional update logic
        updateInternal();

        // If we have no target indices, apply array values to each neuron
        if(getTargetIndices().empty()) {
            for(unsigned int i = 0; i < getNumNeurons(); i++) {
                applyValueFunc(i, m_Buffer[i]);
            }
        }
        // Otherwise, apply to each target index
        else {
            for(unsigned int i = 0; i < getTargetIndices().size(); i++) {
                applyValueFunc(getTargetIndices()[i], m_Buffer[i]);
            }
        }

        // Reset interval
        m_CurrentIntervalTimesteps = m_IntervalTimesteps;
    }
    // Otherwise decrement interval
    else {
        m_CurrentIntervalTimesteps--;
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ExternalNetwork
//----------------------------------------------------------------------------
SpineMLSimulator::InputValue::ExternalNetwork::ExternalNetwork(double dt, unsigned int numNeurons, const pugi::xml_node &node)
: External(dt, numNeurons, node)
{
    // Read connection stats
    const std::string connectionName = node.attribute("name").value();
    const std::string hostname = node.attribute("host").value();
    const unsigned int port = node.attribute("tcp_port").as_uint();
    LOGD_SPINEML << "\tNetwork input '" << connectionName << "' (" << hostname << ":" << port << ")";

    // Attempt to connect network client
    if(!m_Client.connect(hostname, port, getSize(), NetworkClient::DataType::Analogue,
        NetworkClient::Mode::Target, connectionName))
    {
        throw std::runtime_error("Cannot connect network client");
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::InputValue::ExternalNetwork::updateInternal()
{
    // Read buffer from network client
    if(!m_Client.receive(getBuffer())) {
        throw std::runtime_error("Cannot receive data from socket");
    }
}

//------------------------------------------------------------------------
// SpineMLSimulator::InputValue
//------------------------------------------------------------------------
std::unique_ptr<SpineMLSimulator::InputValue::Base> SpineMLSimulator::InputValue::create(double dt, unsigned int numNeurons, const pugi::xml_node &node,
                                                                                         std::map<std::string, InputValue::External*> &externalInputs)
{
    if(strcmp(node.name(), "ConstantInput") == 0) {
        return std::unique_ptr<Base>(new Constant(dt, numNeurons, node));
    }
    else if(strcmp(node.name(), "ConstantArrayInput") == 0) {
        return std::unique_ptr<Base>(new ConstantArray(dt, numNeurons, node));
    }
    else if(strcmp(node.name(), "TimeVaryingInput") == 0) {
        return std::unique_ptr<Base>(new TimeVarying(dt, numNeurons, node));
    }
    else if(strcmp(node.name(), "TimeVaryingArrayInput") == 0) {
        return std::unique_ptr<Base>(new TimeVaryingArray(dt, numNeurons, node));
    }
    else if(strcmp(node.name(), "ExternalInput") == 0) {
        const std::string hostName = node.attribute("host").value();
        if(hostName == "0.0.0.0") {
            std::unique_ptr<External> inputValue(new External(dt, numNeurons, node));

            // Add to map of external inputs
            const std::string name = node.attribute("name").value();
            if(!externalInputs.emplace(name, inputValue.get()).second) {
                LOGW_SPINEML << "External input with duplicate name '" << name << "' encountered";
            }

            return inputValue;
        }
        else {
            return std::unique_ptr<Base>(new ExternalNetwork(dt, numNeurons, node));
        }
    }
    else {
        throw std::runtime_error("Input value type '" + std::string(node.name()) + "' not supported");
    }
}

#include "modelProperty.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>

// Standard C includes
#include <cassert>
#include <cstring>

// Filesystem includes
#include "path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// PLOG includes
#include <plog/Log.h>

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::Fixed
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::Fixed::Fixed(const pugi::xml_node &node, const StateVar<scalar> &stateVar, unsigned int size)
    : Fixed(node.attribute("value").as_double(), stateVar, size)
{
}
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::Fixed::Fixed(double value, const StateVar<scalar> &stateVar, unsigned int size)
    : Base(stateVar, size)
{
    setValue(value);
    LOGD << "\t\t\tFixed value:" << m_Value;
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::Fixed::setValue(scalar value)
{
    // Cache value
    m_Value = value;

    // Fill host state variable
    std::fill_n(getHostStateVar(), getSize(), m_Value);

    // Push to device
    pushToDevice();
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::ValueList
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::ValueList::ValueList(const pugi::xml_node &node, const filesystem::path &basePath,
                                                      const std::vector<unsigned int> *remapIndices, const StateVar<scalar> &stateVar, unsigned int size)
    : Base(stateVar, size)
{
    // Allocate vector to hold values
    // **NOTE** If we're remapping to a padded sparse matrix we want the size of
    // our values to match the number of connections rather than the padded size
    const size_t valueSize =  (remapIndices == nullptr) ? size : remapIndices->size();
    std::vector<scalar> values(valueSize);

    // If there's a binary file
    auto binaryFile = node.child("BinaryFile");
    if(binaryFile) {
        // Check number of elements matches
        const unsigned int numElements =  binaryFile.attribute("num_elements").as_uint();
        const std::string filename = (basePath / binaryFile.attribute("file_name").value()).str();

        // Open file for binary IO
        // **TODO** basepath here
        std::ifstream input(filename, std::ios::binary);
        if(!input.good()) {
            throw std::runtime_error("Cannot open binary model property file:" + filename);
        }

        // Loop through elements in file
        for(unsigned int i = 0; i < numElements; i++) {
            // Read index and value
            // **TODO** this is probably a very sub-optimal way of doing this
            uint32_t index;
            input.read(reinterpret_cast<char*>(&index), sizeof(uint32_t));
            double value;
            input.read(reinterpret_cast<char*>(&value), sizeof(double));

            // Check index is safe and set value
            assert(index < valueSize);
            values[index] = value;
        }

        LOGD << "\t\t\tValue list (from file)";
    }
    // Otherwise
    else {
        LOGD << "\t\t\tValue list (inline)";

        // Loop through inline values
        for(const auto v : node.children("Value")) {
            values[v.attribute("index").as_uint()] = v.attribute("value").as_double();
        }
    }

    setValue(values, remapIndices);
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::ValueList::setValue(const std::vector<scalar> &values, const std::vector<unsigned int> *remapIndices)
{
    // Cache value
    if(remapIndices == nullptr) {
        m_Values = values;
    }
    else {
        assert(remapIndices->size() == values.size());

        // Underlying values should be the size of the property
        m_Values.resize(getSize());

        // Remap values
        for(unsigned int i = 0; i < values.size(); i++) {
            m_Values[remapIndices->operator[](i)] = values[i];
        }
    }

    // Copy vector of values into state variable
    std::copy(m_Values.begin(), m_Values.end(),
              getHostStateVar());

    // Push to device
    pushToDevice();
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::UniformDistribution
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::UniformDistribution::UniformDistribution(const pugi::xml_node &node,
                                                                          const StateVar<scalar> &stateVar, unsigned int size)
    : Base(stateVar, size)
{
    setValue(node.attribute("minimum").as_double(), node.attribute("maximum").as_double());
    LOGD << "\t\t\tMin value:" << m_Distribution.min() << ", Max value:" << m_Distribution.max();

    // Seed RNG if required
    auto seed = node.attribute("seed");
    if(seed) {
        m_RandomGenerator.seed(seed.as_uint());
        LOGD << "\t\t\tSeed:" << seed.as_uint();
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::UniformDistribution::setValue(scalar min, scalar max)
{
    // Create distribution
    m_Distribution = std::uniform_real_distribution<scalar>(min, max);

    // Generate uniformly distributed numbers to fill host array
    std::generate_n(getHostStateVar(), getSize(),
        [this](){ return m_Distribution(m_RandomGenerator); });

    // Push to device
    pushToDevice();
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::NormalDistribution
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::NormalDistribution::NormalDistribution(const pugi::xml_node &node,
                                                                        const StateVar<scalar> &stateVar, unsigned int size)
    : Base(stateVar, size)
{
    setValue(node.attribute("mean").as_double(), node.attribute("variance").as_double());
    LOGD << "\t\t\tMean:" << m_Distribution.mean() << ", Variance:" << m_Distribution.stddev() * m_Distribution.stddev();

    // Seed RNG if required
    auto seed = node.attribute("seed");
    if(seed) {
        m_RandomGenerator.seed(seed.as_uint());
        LOGD << "\t\t\tSeed:" << seed.as_uint();
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::NormalDistribution::setValue(scalar mean, scalar variance)
{
    // Create distribution
    m_Distribution = std::normal_distribution<scalar>(mean, std::sqrt(variance));

    // Generate uniformly distributed numbers to fill host array
    std::generate_n(getHostStateVar(), getSize(),
        [this](){ return m_Distribution(m_RandomGenerator); });

    // Push to device
    pushToDevice();
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::ExponentialDistribution
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::ExponentialDistribution::ExponentialDistribution(const pugi::xml_node &node,
                                                                                  const StateVar<scalar> &stateVar, unsigned int size)
    : Base(stateVar, size)
{
    setValue(node.attribute("mean").as_double());
    LOGD << "\t\t\tLambda:" << m_Distribution.lambda();

    // Seed RNG if required
    auto seed = node.attribute("seed");
    if(seed) {
        m_RandomGenerator.seed(seed.as_uint());
        LOGD << "\t\t\tSeed:" << seed.as_uint();
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::ExponentialDistribution::setValue(scalar lambda)
{
    // Create distribution
    m_Distribution = std::exponential_distribution<scalar>(lambda);

    // Generate uniformly distributed numbers to fill host array
    std::generate_n(getHostStateVar(), getSize(),
        [this](){ return m_Distribution(m_RandomGenerator); });

    // Push to device
    pushToDevice();
}

//----------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty
//----------------------------------------------------------------------------
std::unique_ptr<SpineMLSimulator::ModelProperty::Base> SpineMLSimulator::ModelProperty::create(const pugi::xml_node &node, const StateVar<scalar> &stateVar,
                                                                                               unsigned int size, bool skipGeNNInitialised, const filesystem::path &basePath,
                                                                                               const std::string &valueNamespace, const std::vector<unsigned int> *remapIndices)
{
    // Prepend namespace onto names of various types of model property
    const std::string valueListName = valueNamespace + "ValueList";
    const std::string fixedValueName = valueNamespace + "FixedValue";
    const std::string uniformDistributionName = valueNamespace + "UniformDistribution";
    const std::string normalDistribution = valueNamespace + "NormalDistribution";
    const std::string poissonDistributionName = valueNamespace + "PoissonDistribution";

    // If this property has a child
    auto valueChild = node.first_child();
    if(valueChild) {
        // If this property is intialised with a list of values - create a value list model property to manually
        if(strcmp(valueChild.name(), valueListName.c_str()) == 0) {
            return std::unique_ptr<Base>(new ValueList(valueChild, basePath, remapIndices, stateVar, size));
        }
        // Otherwise if we can skip property types that GeNN can initialise
        else if(skipGeNNInitialised) {
            // If property type is one supported by GeNN, add standard model property
            if(strcmp(valueChild.name(), fixedValueName.c_str()) == 0 ||
                strcmp(valueChild.name(), uniformDistributionName.c_str()) == 0 ||
                strcmp(valueChild.name(), normalDistribution.c_str()) == 0 ||
                strcmp(valueChild.name(), poissonDistributionName.c_str()) == 0)
            {
                return std::unique_ptr<Base>(new Base(stateVar, size));
            }
        }
        // Otherwise, if we can't skip property types supported by GeNN i.e. when we are overriding model properties
        else if(!skipGeNNInitialised) {
            if(strcmp(valueChild.name(), fixedValueName.c_str()) == 0) {
                return std::unique_ptr<Base>(new Fixed(valueChild, stateVar, size));
            }
            else if(strcmp(valueChild.name(), uniformDistributionName.c_str()) == 0) {
                return std::unique_ptr<Base>(new UniformDistribution(valueChild, stateVar, size));
            }
            else if(strcmp(valueChild.name(), normalDistribution.c_str()) == 0) {
                return std::unique_ptr<Base>(new NormalDistribution(valueChild, stateVar, size));
            }
            // **NOTE** Poisson distribution isn't actually one - it is the exponential
            // distribution (which models the inter-event-interval of a Poisson PROCESS)
            else if(strcmp(valueChild.name(), poissonDistributionName.c_str()) == 0) {
                return std::unique_ptr<Base>(new ExponentialDistribution(valueChild, stateVar, size));
            }
        }
        throw std::runtime_error("Unsupported type '" + std::string(valueChild.name()) + "' for property '" + std::string(node.attribute("name").value()) + "'");
    }
    // Otherwise assume that GeNN has initialised variable to something sensible and add standard model property
    else {
        return std::unique_ptr<Base>(new Base(stateVar, size));
    }
}

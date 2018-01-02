#include "modelProperty.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>

// Standard C includes
#include <cassert>
#include <cstring>

// Filesystem includes
#include "filesystem/path.h"

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
    : Fixed(node.attribute("value").as_double(), hostStateVar, deviceStateVar, size)
{
}
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::Fixed::Fixed(double value, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
    : Base(hostStateVar, deviceStateVar, size)
{
    setValue(value);
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
SpineMLSimulator::ModelProperty::ValueList::ValueList(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size,
                                                      const filesystem::path &basePath, const std::vector<unsigned int> *remapIndices)
    : Base(hostStateVar, deviceStateVar, size)
{
    // Allocate vector
    std::vector<scalar> values(size);

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
            assert(index < size);
            values[index] = value;
        }

        std::cout << "\t\t\tValue list (from file)" << std::endl;
    }
    // Otherwise
    else {
        std::cout << "\t\t\tValue list (inline)" << std::endl;

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

        m_Values.resize(values.size());
        for(unsigned int i = 0; i < values.size(); i++) {
            m_Values[i] = values[(*remapIndices)[i]];
        }
    }

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
    std::cout << "\t\t\tMean:" << m_Distribution.mean() << ", Variance:" << m_Distribution.stddev() * m_Distribution.stddev() << std::endl;

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

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::ExponentialDistribution
//------------------------------------------------------------------------
SpineMLSimulator::ModelProperty::ExponentialDistribution::ExponentialDistribution(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
    : Base(hostStateVar, deviceStateVar, size)
{
    setValue(node.attribute("mean").as_double());
    std::cout << "\t\t\tLambda:" << m_Distribution.lambda() << std::endl;

    // Seed RNG if required
    auto seed = node.attribute("seed");
    if(seed) {
        m_RandomGenerator.seed(seed.as_uint());
        std::cout << "\t\t\tSeed:" << seed.as_uint() << std::endl;
    }
}
//------------------------------------------------------------------------
void SpineMLSimulator::ModelProperty::ExponentialDistribution::setValue(scalar lambda)
{
    // Create distribution
    m_Distribution = std::exponential_distribution<scalar>(lambda);

    // Generate uniformly distributed numbers to fill host array
    std::generate(getHostStateVarBegin(), getHostStateVarEnd(),
        [this](){ return m_Distribution(m_RandomGenerator); });

    // Push to device
    pushToDevice();
}

//----------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty
//----------------------------------------------------------------------------
std::unique_ptr<SpineMLSimulator::ModelProperty::Base> SpineMLSimulator::ModelProperty::create(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar,
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
            return std::unique_ptr<Base>(new ValueList(valueChild, hostStateVar, deviceStateVar, size,
                                                       basePath, remapIndices));
        }
        // Otherwise if we can skip property types that GeNN can initialise
        else if(skipGeNNInitialised) {
            // If property type is one supported by GeNN, add standard model property
            if(strcmp(valueChild.name(), fixedValueName.c_str()) == 0 ||
                strcmp(valueChild.name(), uniformDistributionName.c_str()) == 0 ||
                strcmp(valueChild.name(), normalDistribution.c_str()) == 0 ||
                strcmp(valueChild.name(), poissonDistributionName.c_str()) == 0)
            {
                return std::unique_ptr<Base>(new Base(hostStateVar, deviceStateVar, size));
            }
        }
        // Otherwise, if we can't skip property types supported by GeNN i.e. when we are overriding model properties
        else if(!skipGeNNInitialised) {
            if(strcmp(valueChild.name(), fixedValueName.c_str()) == 0) {
                return std::unique_ptr<Base>(new Fixed(valueChild, hostStateVar, deviceStateVar, size));
            }
            else if(strcmp(valueChild.name(), uniformDistributionName.c_str()) == 0) {
                return std::unique_ptr<Base>(new UniformDistribution(valueChild, hostStateVar, deviceStateVar, size));
            }
            else if(strcmp(valueChild.name(), normalDistribution.c_str()) == 0) {
                return std::unique_ptr<Base>(new NormalDistribution(valueChild, hostStateVar, deviceStateVar, size));
            }
            // **NOTE** Poisson distribution isn't actually one - it is the exponential
            // distribution (which models the inter-event-interval of a Poisson PROCESS)
            else if(strcmp(valueChild.name(), poissonDistributionName.c_str()) == 0) {
                return std::unique_ptr<Base>(new ExponentialDistribution(valueChild, hostStateVar, deviceStateVar, size));
            }
        }
        throw std::runtime_error("Unsupported type '" + std::string(valueChild.name()) + "' for property '" + std::string(node.attribute("name").value()) + "'");
    }
    // Otherwise assume that GeNN has initialised variable to something sensible and add standard model property
    else {
        return std::unique_ptr<Base>(new Base(hostStateVar, deviceStateVar, size));
    }
}
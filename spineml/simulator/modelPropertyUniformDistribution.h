#pragma once

// SpineML simulator includes
#include "modelProperty.h"

// Standard C++ includes
#include <random>

// Forward declarations
namespace pugi
{
    class xml_node;
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelPropertyUniformDistribution
//------------------------------------------------------------------------
namespace SpineMLSimulator
{
class ModelPropertyUniformDistribution : public ModelProperty
{
public:
    ModelPropertyUniformDistribution(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void SetValue(scalar min, scalar max);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::random_device m_RandomDevice;
    std::mt19937 m_RandomGenerator;

    std::uniform_real_distribution<scalar> m_Distribution;
};
} // namespace SpineMLSimulator
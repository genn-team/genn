#pragma once

// Standard C++ includes
#include <memory>
#include <random>
#include <vector>

// SpineML simulator includes
#include "stateVar.h"

// Forward declarations
namespace pugi
{
    class xml_node;
}

namespace filesystem
{
    class path;
}

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::Base
//------------------------------------------------------------------------
namespace SpineMLSimulator
{
typedef float scalar;   // **TODO** move this somewhere more sensible
namespace ModelProperty
{
class Base
{
public:
    Base(const StateVar<scalar> &stateVar, unsigned int size)
    :   m_StateVar(stateVar), m_Size(size)
    {
    }

    virtual ~Base(){}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    scalar *getHostStateVar()
    {
        return m_StateVar.get();
    }

    const scalar *getHostStateVar() const
    {
        return m_StateVar.get();
    }

    void pushToDevice() const
    {
        m_StateVar.push();
    }
    void pullFromDevice() const
    {
        m_StateVar.pull();
    }

    unsigned int getSize() const
    {
        return m_Size;
    }

private:
    //------------------------------------------------------------------------
    // Private members
    //------------------------------------------------------------------------
    StateVar<scalar> m_StateVar;
    unsigned int m_Size;
};

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::Fixed
//------------------------------------------------------------------------
class Fixed : public Base
{
public:
    Fixed(const pugi::xml_node &node, const StateVar<scalar> &stateVar, unsigned int size);
    Fixed(double value, const StateVar<scalar> &stateVar, unsigned int size);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void setValue(scalar value);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    scalar m_Value;
};

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::ValueList
//------------------------------------------------------------------------
class ValueList : public Base
{
public:
    ValueList(const pugi::xml_node &node, const filesystem::path &basePath, const std::vector<unsigned int> *remapIndices,
              const StateVar<scalar> &stateVar, unsigned int size);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void setValue(const std::vector<scalar> &values, const std::vector<unsigned int> *remapIndices);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<scalar> m_Values;
};

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::UniformDistribution
//------------------------------------------------------------------------
class UniformDistribution : public Base
{
public:
    UniformDistribution(const pugi::xml_node &node, const StateVar<scalar> &stateVar, unsigned int size);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void setValue(scalar min, scalar max);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::mt19937 m_RandomGenerator;
    std::uniform_real_distribution<scalar> m_Distribution;
};

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::NormalDistribution
//------------------------------------------------------------------------
class NormalDistribution : public Base
{
public:
    NormalDistribution(const pugi::xml_node &node, const StateVar<scalar> &stateVar, unsigned int size);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void setValue(scalar mean, scalar variance);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::mt19937 m_RandomGenerator;
    std::normal_distribution<scalar> m_Distribution;
};

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::ExponentialDistribution
//------------------------------------------------------------------------
class ExponentialDistribution : public Base
{
public:
    ExponentialDistribution(const pugi::xml_node &node, const StateVar<scalar> &stateVar,  unsigned int size);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void setValue(scalar lambda);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::mt19937 m_RandomGenerator;
    std::exponential_distribution<scalar> m_Distribution;
};

//----------------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------------
std::unique_ptr<Base> create(const pugi::xml_node &node,
                             const StateVar<scalar> &stateVar, unsigned int size,
                             bool skipGeNNInitialised, const filesystem::path &basePath,
                             const std::string &valueNamespace, const std::vector<unsigned int> *remapIndices);

}   // namespace ModelProperty
}   // namespace SpineMLSimulator

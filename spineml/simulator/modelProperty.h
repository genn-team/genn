#pragma once

// Standard C++ includes
#include <memory>
#include <random>

// Forward declarations
namespace pugi
{
    class xml_node;
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
    Base(scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
        : m_HostStateVar(hostStateVar), m_DeviceStateVar(deviceStateVar), m_Size(size)
    {
    }
    virtual ~Base(){}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    scalar *getHostStateVarBegin() { return &m_HostStateVar[0]; }
    scalar *getHostStateVarEnd() { return &m_HostStateVar[m_Size]; }

    const scalar *getHostStateVarBegin() const{ return &m_HostStateVar[0]; }
    const scalar *getHostStateVarEnd() const{ return &m_HostStateVar[m_Size]; }

    void pushToDevice() const;
    void pullFromDevice() const;

    unsigned int getSize() const{ return m_Size; }

private:
    //------------------------------------------------------------------------
    // Private members
    //------------------------------------------------------------------------
    scalar *m_HostStateVar;
    scalar *m_DeviceStateVar;
    unsigned int m_Size;
};

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::Fixed
//------------------------------------------------------------------------
class Fixed : public Base
{
public:
    Fixed(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size);

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
    ValueList(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void setValue(const std::vector<scalar> &values);

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
    UniformDistribution(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size);

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
    NormalDistribution(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size);

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
    ExponentialDistribution(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size);

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
std::unique_ptr<Base> create(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size);

}   // namespace ModelProperty
}   // namespace SpineMLSimulator
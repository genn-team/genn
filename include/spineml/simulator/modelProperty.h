#pragma once

// Standard C++ includes
#include <memory>
#include <random>
#include <vector>

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
    typedef void (*PushFunc)(bool);
    typedef void (*PullFunc)(void);

    Base(scalar *hostStateVar, PushFunc pushFunc, PullFunc pullFunc, unsigned int size)
        : m_HostStateVar(hostStateVar), m_PushFunc(pushFunc), m_PullFunc(pullFunc), m_Size(size)
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
    PushFunc m_PushFunc;
    PullFunc m_PullFunc;
    unsigned int m_Size;
};

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty::Fixed
//------------------------------------------------------------------------
class Fixed : public Base
{
public:
    Fixed(const pugi::xml_node &node,
          scalar *hostStateVar, PushFunc pushFunc, PullFunc pullFunc, unsigned int size);
    Fixed(double value,
          scalar *hostStateVar, PushFunc pushFunc, PullFunc pullFunc, unsigned int size);

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
              scalar *hostStateVar, PushFunc pushFunc, PullFunc pullFunc, unsigned int size);

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
    UniformDistribution(const pugi::xml_node &node,
                        scalar *hostStateVar, PushFunc pushFunc, PullFunc pullFunc, unsigned int size);

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
    NormalDistribution(const pugi::xml_node &node,
                       scalar *hostStateVar, PushFunc pushFunc, PullFunc pullFunc, unsigned int size);

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
    ExponentialDistribution(const pugi::xml_node &node,
                            scalar *hostStateVar, PushFunc pushFunc, PullFunc pullFunc, unsigned int size);

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
                             scalar *hostStateVar, Base::PushFunc pushFunc, Base::PullFunc pullFunc, unsigned int size,
                             bool skipGeNNInitialised, const filesystem::path &basePath,
                             const std::string &valueNamespace, const std::vector<unsigned int> *remapIndices);

}   // namespace ModelProperty
}   // namespace SpineMLSimulator

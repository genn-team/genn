#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <vector>

// Forward declarations
namespace pugi
{
    class xml_node;
}

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue
//----------------------------------------------------------------------------
namespace SpineMLSimulator
{
namespace InputValue
{
class Base
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void update(double dt, unsigned int timestep,
                        std::function<void(unsigned int, double)> applyValueFunc) const = 0;

protected:
    Base(unsigned int numNeurons, const pugi::xml_node &node);

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const std::vector<unsigned int> &getTargetIndices() const{ return m_TargetIndices; }
    unsigned int getNumNeurons() const{ return m_NumNeurons; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    unsigned int m_NumNeurons;
    std::vector<unsigned int> m_TargetIndices;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ScalarBase
//----------------------------------------------------------------------------
class ScalarBase : public Base
{
protected:
    ScalarBase(unsigned int numNeurons, const pugi::xml_node &node) : Base(numNeurons, node){}

    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    void applyScalar(double value,
                     std::function<void(unsigned int, double)> applyValueFunc) const;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ArrayBase
//----------------------------------------------------------------------------
class ArrayBase : public Base
{
protected:
    ArrayBase(unsigned int numNeurons, const pugi::xml_node &node) : Base(numNeurons, node){}

    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    void applyArray(const std::vector<double> &values,
                    std::function<void(unsigned int, double)> applyValueFunc) const;

    bool checkArrayDimensions(const std::vector<double> &values) const;

};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::Constant
//----------------------------------------------------------------------------
class Constant : public ScalarBase
{
public:
    Constant(double dt, unsigned int numNeurons, const pugi::xml_node &node);

    //------------------------------------------------------------------------
    // InputValue virtuals
    //------------------------------------------------------------------------
    virtual void update(double dt, unsigned int timestep,
                        std::function<void(unsigned int, double)> applyValueFunc) const override;
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    double m_Value;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ConstantArray
//----------------------------------------------------------------------------
class ConstantArray : public ArrayBase
{
public:
    ConstantArray(double dt, unsigned int numNeurons, const pugi::xml_node &node);

    //------------------------------------------------------------------------
    // InputValue virtuals
    //------------------------------------------------------------------------
    virtual void update(double dt, unsigned int timestep,
                        std::function<void(unsigned int, double)> applyValueFunc) const override;
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<double> m_Values;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::TimeVarying
//----------------------------------------------------------------------------
class TimeVarying : public ScalarBase
{
public:
    TimeVarying(double dt, unsigned int numNeurons, const pugi::xml_node &node);

    //------------------------------------------------------------------------
    // InputValue virtuals
    //------------------------------------------------------------------------
    virtual void update(double dt, unsigned int timestep,
                        std::function<void(unsigned int, double)> applyValueFunc) const override;
private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::map<unsigned int, double> m_TimeValues;
};
}   // namespace InputValue
}   // namespace SpineMLSimulator
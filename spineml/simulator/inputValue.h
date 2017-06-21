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
    virtual void updateValues(double dt, unsigned int timestep, unsigned int numNeurons,
                              std::function<void(unsigned int, double)> applyValueFunc) const = 0;

protected:
    Base(const pugi::xml_node &node);

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const std::vector<unsigned int> &getTargetIndices() const{ return m_TargetIndices; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<unsigned int> m_TargetIndices;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ScalarBase
//----------------------------------------------------------------------------
class ScalarBase : public Base
{
protected:
    ScalarBase(const pugi::xml_node &node) : Base(node){}

    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    void applyScalar(unsigned int numNeurons, double value,
                     std::function<void(unsigned int, double)> applyValueFunc) const;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ArrayBase
//----------------------------------------------------------------------------
class ArrayBase : public Base
{
protected:
    ArrayBase(const pugi::xml_node &node) : Base(node){}

    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    void applyArray(unsigned int numNeurons, const std::vector<double> &values,
                    std::function<void(unsigned int, double)> applyValueFunc) const;

    bool checkArrayDimensions(unsigned int numNeurons, const std::vector<double> &values) const;

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
    virtual void updateValues(double dt, unsigned int timestep, unsigned int numNeurons,
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
    virtual void updateValues(double dt, unsigned int timestep, unsigned int numNeurons,
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
    virtual void updateValues(double dt, unsigned int timestep, unsigned int numNeurons,
                              std::function<void(unsigned int, double)> applyValueFunc) const override;
private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::map<unsigned int, double> m_TimeValues;
};
}   // namespace InputValue
}   // namespace SpineMLSimulator
#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Simulator includes
#include "networkClient.h"

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
    virtual ~Base(){}

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void update(double dt, unsigned long long timestep,
                        std::function<void(unsigned int, double)> applyValueFunc) = 0;

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
// SpineMLSimulator::InputValue::Constant
//----------------------------------------------------------------------------
class Constant : public ScalarBase
{
public:
    Constant(double dt, unsigned int numNeurons, const pugi::xml_node &node);

    //------------------------------------------------------------------------
    // InputValue virtuals
    //------------------------------------------------------------------------
    virtual void update(double dt, unsigned long long timestep,
                        std::function<void(unsigned int, double)> applyValueFunc) override;
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    // Value to apply to targetted neurons throughout simulation
    double m_Value;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ConstantArray
//----------------------------------------------------------------------------
class ConstantArray : public Base
{
public:
    ConstantArray(double dt, unsigned int numNeurons, const pugi::xml_node &node);

    //------------------------------------------------------------------------
    // InputValue virtuals
    //------------------------------------------------------------------------
    virtual void update(double dt, unsigned long long timestep,
                        std::function<void(unsigned int, double)> applyValueFunc) override;
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    // Values to apply to targetted neurons throughout simulation
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
    virtual void update(double dt, unsigned long long timestep,
                        std::function<void(unsigned int, double)> applyValueFunc) override;
private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    // Values to apply to all neurons at certain timesteps
    std::map<unsigned int, double> m_TimeValues;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::TimeVaryingArray
//----------------------------------------------------------------------------
class TimeVaryingArray : public Base
{
public:
    TimeVaryingArray(double dt, unsigned int numNeurons, const pugi::xml_node &node);

    //------------------------------------------------------------------------
    // InputValue virtuals
    //------------------------------------------------------------------------
    virtual void update(double dt, unsigned long long timestep,
                        std::function<void(unsigned int, double)> applyValueFunc) override;

private:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef std::vector<std::pair<unsigned int, double>> NeuronValueVec;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    // Vector of neurons and values to apply at certain timesteps
    std::map<unsigned int, NeuronValueVec> m_TimeArrays;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::External
//----------------------------------------------------------------------------
class External : public Base
{
public:
    External(double dt, unsigned int numNeurons, const pugi::xml_node &node);

    //------------------------------------------------------------------------
    // InputValue virtuals
    //------------------------------------------------------------------------
    virtual void update(double dt, unsigned long long timestep,
                        std::function<void(unsigned int, double)> applyValueFunc) final;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    std::vector<double>::iterator getBufferBegin(){ return m_Buffer.begin(); }
    std::vector<double>::iterator getBufferEnd(){ return m_Buffer.end(); }

protected:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void updateInternal(){}

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    unsigned int getSize() const
    {
        return getTargetIndices().empty() ? getNumNeurons() : (unsigned int)getTargetIndices().size();
    }

    std::vector<double> &getBuffer(){ return m_Buffer; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<double> m_Buffer;

    // How many GeNN timesteps do we wait before updating
    unsigned int m_IntervalTimesteps;

    // Count down to next time we update
    unsigned int m_CurrentIntervalTimesteps;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::InputValue::ExternalNetwork
//----------------------------------------------------------------------------
class ExternalNetwork : public External
{
public:
    ExternalNetwork(double dt, unsigned int numNeurons, const pugi::xml_node &node);

protected:
    //------------------------------------------------------------------------
    // External virtuals
    //------------------------------------------------------------------------
    virtual void updateInternal() override;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    NetworkClient m_Client;
};

//----------------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------------
std::unique_ptr<Base> create(double dt, unsigned int numNeurons, const pugi::xml_node &node,
                             std::map<std::string, InputValue::External*> &externalInputs);

}   // namespace InputValue
}   // namespace SpineMLSimulator

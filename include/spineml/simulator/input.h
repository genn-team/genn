#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <memory>
#include <random>

// Forward declarations
namespace pugi
{
    class xml_node;
}

namespace SpineMLSimulator
{
    namespace InputValue
    {
        class Base;
    }

    namespace ModelProperty
    {
        class Base;
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::Base
//----------------------------------------------------------------------------
namespace SpineMLSimulator
{
namespace Input
{
class Base
{
public:
    virtual ~Base();

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual void apply(double dt, unsigned long long timestep) = 0;

protected:
     Base(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value);

    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    bool shouldApply(unsigned long long timestep) const
    {
        return (timestep >= m_StartTimeStep && timestep < m_EndTimeStep);
    }

    void updateValues(double dt, unsigned long long timestep,
                      std::function<void(unsigned int, double)> applyValueFunc) const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    unsigned long long m_StartTimeStep;
    unsigned long long m_EndTimeStep;

    std::unique_ptr<InputValue::Base> m_Value;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::SpikeBase
//----------------------------------------------------------------------------
class SpikeBase : public Base
{
protected:
    typedef void (*PushCurrentSpikesFunc)();

    SpikeBase(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value,
              unsigned int popSize, unsigned int *spikeQueuePtr, unsigned int *hostSpikeCount, unsigned int *hostSpikes,
              PushCurrentSpikesFunc pushCurrentSpikes);

    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    void injectSpike(unsigned int neuronID);

    void uploadSpikes();

private:
    //----------------------------------------------------------------------------
    // Private API
    //----------------------------------------------------------------------------
    unsigned int getSpikeQueueIndex() const
    {
        return (m_SpikeQueuePtr == nullptr) ? 0 : *m_SpikeQueuePtr;
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const unsigned int m_PopSize;

    unsigned int *m_SpikeQueuePtr;
    unsigned int *m_HostSpikeCount;
    unsigned int *m_HostSpikes;
    PushCurrentSpikesFunc m_PushCurrentSpikes;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::InterSpikeIntervalBase
//----------------------------------------------------------------------------
class InterSpikeIntervalBase : public SpikeBase
{
public:
    //----------------------------------------------------------------------------
    // Base virtuals
    //----------------------------------------------------------------------------
    virtual void apply(double dt, unsigned long long timestep) override;

protected:
    InterSpikeIntervalBase(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value,
                           unsigned int popSize, unsigned int *spikeQueuePtr, unsigned int *hostSpikeCount, unsigned int *hostSpikes,
                           PushCurrentSpikesFunc pushCurrentSpikes);

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual double getTimeToSpike(double isiMs) = 0;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::map<unsigned int, std::pair<double, double>> m_TimeToSpike;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::RegularSpikeRate
//----------------------------------------------------------------------------
class RegularSpikeRate : public InterSpikeIntervalBase
{
public:
    RegularSpikeRate(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value,
                     unsigned int popSize, unsigned int *spikeQueuePtr, unsigned int *hostSpikeCount, unsigned int *hostSpikes,
                     PushCurrentSpikesFunc pushCurrentSpikes);

protected:
    //----------------------------------------------------------------------------
    // InterSpikeIntervalBase virtuals
    //----------------------------------------------------------------------------
    virtual double getTimeToSpike(double isiMs) override;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::PoissonSpikeRate
//----------------------------------------------------------------------------
class PoissonSpikeRate : public InterSpikeIntervalBase
{
public:
    PoissonSpikeRate(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value,
                     unsigned int popSize, unsigned int *spikeQueuePtr, unsigned int *hostSpikeCount, unsigned int *hostSpikes,
                     PushCurrentSpikesFunc pushCurrentSpikes);

protected:
    //----------------------------------------------------------------------------
    // InterSpikeIntervalBase virtuals
    //----------------------------------------------------------------------------
    virtual double getTimeToSpike(double isiMs);

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::mt19937 m_RandomGenerator;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::SpikeTime
//----------------------------------------------------------------------------
class SpikeTime : public SpikeBase
{
public:
    SpikeTime(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value,
              unsigned int popSize, unsigned int *spikeQueuePtr, unsigned int *hostSpikeCount, unsigned int *hostSpikes,
              PushCurrentSpikesFunc pushCurrentSpikes);

    //----------------------------------------------------------------------------
    // SpikeBase virtuals
    //----------------------------------------------------------------------------
    virtual void apply(double dt, unsigned long long timestep) override;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::Analogue
//----------------------------------------------------------------------------
class Analogue : public Base
{
public:
    Analogue(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value,
             ModelProperty::Base *modelProperty);

    //----------------------------------------------------------------------------
    // Base virtuals
    //----------------------------------------------------------------------------
    virtual void apply(double dt, unsigned long long timestep) override;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    //! Has a change been made to values which needs applying to model property
    bool m_PropertyUpdateRequired;

    ModelProperty::Base *m_ModelProperty;

    // Current values to apply
    std::map<unsigned int, double> m_CurrentValues;
};

}   // namespace Input
}   // namespace SpineMLSimulator

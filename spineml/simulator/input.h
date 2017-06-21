#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <memory>

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
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual void apply(double dt, unsigned int timestep) = 0;

protected:
     Base(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value);

    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    bool shouldApply(unsigned int timestep) const
    {
        return (timestep >= m_StartTimeStep && timestep < m_EndTimeStep);
    }

    void updateValues(double dt, unsigned int timestep,
                      std::function<void(unsigned int, double)> applyValueFunc) const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    unsigned int m_StartTimeStep;
    unsigned int m_EndTimeStep;

    std::unique_ptr<InputValue::Base> m_Value;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::SpikeBase
//----------------------------------------------------------------------------
class SpikeBase : public Base
{
protected:
    SpikeBase(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value,
              unsigned int *spikeQueuePtr,
              unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
              unsigned int *hostSpikes, unsigned int *deviceSpikes);

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
        return (m_SpikeQueuePtr == NULL) ? 0 : *m_SpikeQueuePtr;
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    unsigned int *m_SpikeQueuePtr;
    unsigned int *m_HostSpikeCount;
    unsigned int *m_DeviceSpikeCount;

    unsigned int *m_HostSpikes;
    unsigned int *m_DeviceSpikes;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::RegularSpikeRate
//----------------------------------------------------------------------------
class RegularSpikeRate : public SpikeBase
{
public:
    RegularSpikeRate(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value,
                     unsigned int *spikeQueuePtr,
                     unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
                     unsigned int *hostSpikes, unsigned int *deviceSpikes);

    //----------------------------------------------------------------------------
    // SpikeBase virtuals
    //----------------------------------------------------------------------------
    virtual void apply(double dt, unsigned int timestep) override;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::map<unsigned int, std::pair<double, double>> m_TimeToSpike;
};

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::PoissonSpikeRate
//----------------------------------------------------------------------------
/*class PoissonSpikeRate : public SpikeBase
{
};*/

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::SpikeTime
//----------------------------------------------------------------------------
class SpikeTime : public SpikeBase
{
public:
    SpikeTime(double dt, const pugi::xml_node &node, std::unique_ptr<InputValue::Base> value,
              unsigned int *spikeQueuePtr,
              unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
              unsigned int *hostSpikes, unsigned int *deviceSpikes);

    //----------------------------------------------------------------------------
    // SpikeBase virtuals
    //----------------------------------------------------------------------------
    virtual void apply(double dt, unsigned int timestep) override;

private:

};

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::Analogue
//----------------------------------------------------------------------------
/*class Analogue : public Base
{
};*/

}   // namespace Input
}   // namespace SpineMLSimulator
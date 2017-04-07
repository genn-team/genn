#pragma once

//------------------------------------------------------------------------
// SpineMLSimulator::ModelProperty
//------------------------------------------------------------------------
namespace SpineMLSimulator
{
typedef float scalar;

class ModelProperty
{
public:
    ModelProperty(scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
        : m_HostStateVar(hostStateVar), m_DeviceStateVar(deviceStateVar), m_Size(size)
    {
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    scalar *getHostStateVarBegin() { return &m_HostStateVar[0]; }
    scalar *getHostStateVarEnd() { return &m_HostStateVar[m_Size]; }

    void PushToDevice();
    void PullFromDevice();
private:
    //------------------------------------------------------------------------
    // Private members
    //------------------------------------------------------------------------
    scalar *m_HostStateVar;
    scalar *m_DeviceStateVar;
    unsigned int m_Size;
};
}   // namespace SpineMLSimulator
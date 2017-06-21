#include "input.h"

// Standard C++ includes
#include <limits>
#include <iostream>

// Standard C includes
#include <cmath>

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "utils.h"

// SpineML simulator includes
#include "inputValue.h"

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::Base
//----------------------------------------------------------------------------
SpineMLSimulator::Input::Base::Base(double dt, const pugi::xml_node &node, const InputValue::Base &value) : m_Value(value)
{
    // Read start time
    auto startAttr = node.attribute("start_time");
    if(startAttr.empty()) {
        m_StartTimeStep = 0;
    }
    else {
        m_StartTimeStep = (unsigned int)std::ceil(startAttr.as_double() / dt);
    }

    // Read duration
    auto durationAttr = node.attribute("duration");
    if(durationAttr.empty()) {
        m_EndTimeStep = std::numeric_limits<unsigned int>::max();
    }
    else {
        m_EndTimeStep = m_StartTimeStep + (unsigned int)std::ceil(durationAttr.as_double() / dt);
    }

    std::cout << "\tStart timestep:" << m_StartTimeStep << ", end timestep:" << m_EndTimeStep << std::endl;
}
//----------------------------------------------------------------------------
void SpineMLSimulator::Input::Base::updateValues(double dt, unsigned int timestep, unsigned int numNeurons,
                                                 std::function<void(unsigned int, double)> applyValueFunc) const
{
    m_Value.updateValues(dt, timestep, numNeurons, applyValueFunc);
}

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::SpikeBase
//----------------------------------------------------------------------------
SpineMLSimulator::Input::SpikeBase::SpikeBase(double dt, const pugi::xml_node &node, const InputValue::Base &value,
                                              unsigned int *spikeQueuePtr,
                                              unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
                                              unsigned int *hostSpikes, unsigned int *deviceSpikes)
: Base(dt, node, value), m_SpikeQueuePtr(spikeQueuePtr),
  m_HostSpikeCount(hostSpikeCount), m_DeviceSpikeCount(deviceSpikeCount),
  m_HostSpikes(hostSpikes), m_DeviceSpikes(deviceSpikes)
{
}
//----------------------------------------------------------------------------
void SpineMLSimulator::Input::SpikeBase::injectSpike(unsigned int neuronID)
{
    m_HostSpikes[m_HostSpikeCount[*m_SpikeQueuePtr]++] = neuronID;
}
//----------------------------------------------------------------------------
void SpineMLSimulator::Input::SpikeBase::uploadSpikes()
{
#ifndef CPU_ONLY
    // If therea any spikes to inject
    if(m_HostSpikeCount[*m_SpikeQueuePtr] > 0) {
        // Copy spike count from spike queue position to device
        CHECK_CUDA_ERRORS(cudaMemcpy(&m_DeviceSpikeCount[*m_SpikeQueuePtr], &m_HostSpikeCount[*m_SpikeQueuePtr],
                                    sizeof(unsigned int), cudaMemcpyHostToDevice));

        // Copy this many spikes to device
        CHECK_CUDA_ERRORS(cudaMemcpy(m_DeviceSpikes, m_HostSpikes,
                                    sizeof(unsigned int) * m_HostSpikeCount[*m_SpikeQueuePtr], cudaMemcpyHostToDevice));
    }
#endif  // CPU_ONLY
}

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::RegularSpikeRate
//----------------------------------------------------------------------------
SpineMLSimulator::Input::RegularSpikeRate::RegularSpikeRate(double dt, const pugi::xml_node &node, const InputValue::Base &value,
                                                            unsigned int *spikeQueuePtr,
                                                            unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
                                                            unsigned int *hostSpikes, unsigned int *deviceSpikes)
: SpikeBase(dt, node, value, spikeQueuePtr, hostSpikeCount, deviceSpikeCount, hostSpikes, deviceSpikes)
{
}
//----------------------------------------------------------------------------
void SpineMLSimulator::Input::RegularSpikeRate::apply(double dt, unsigned int timestep, unsigned int numNeurons)
{
    // Determine if there are any update values this timestep (spikes)
    // **NOTE** even if we shouldn't be applying any input, rate updates still should happen
    updateValues(dt, timestep, numNeurons,
        [this, dt](unsigned int neuronID, double rate)
        {
            // Convert rate into interspike interval in timesteps
            const double isiMs = 1000.0 / rate;

            // If this neuron has not had a rate set before, add it to map
            auto neuronTTS = m_TimeToSpike.find(neuronID);
            if(neuronTTS == m_TimeToSpike.end()) {
                m_TimeToSpike.insert(std::make_pair(neuronID, std::make_pair(isiMs, isiMs)));
            }
            // Otherwise, update it's ISI to the new one and also reset it's time to spike
            else {
                neuronTTS->second.first = isiMs;
                neuronTTS->second.second = isiMs;
            }
        });

    // If we should be applying input during this timestep
    if(shouldApply(timestep)) {
        // Loop through all times to spike
        for(auto &tts : m_TimeToSpike) {
            // If this neuron isn't going to spike this timestep, decrement ISI
            if(tts.second.second > 0.0) {
                tts.second.second -= dt;
            }
            // Otherwise
            else {
                // Reset time-to-spike to interspike interval
                tts.second.second = tts.second.first;

                // Inject spike
                injectSpike(tts.first);
            }
        }

        // Upload spikes to GPU if required
        uploadSpikes();
    }
}

//----------------------------------------------------------------------------
// SpineMLSimulator::Input::SpikeTime
//----------------------------------------------------------------------------
SpineMLSimulator::Input::SpikeTime::SpikeTime(double dt, const pugi::xml_node &node, const InputValue::Base &value,
                                              unsigned int *spikeQueuePtr,
                                              unsigned int *hostSpikeCount, unsigned int *deviceSpikeCount,
                                              unsigned int *hostSpikes, unsigned int *deviceSpikes)
: SpikeBase(dt, node, value, spikeQueuePtr, hostSpikeCount, deviceSpikeCount, hostSpikes, deviceSpikes)
{
}
//----------------------------------------------------------------------------
void SpineMLSimulator::Input::SpikeTime::apply(double dt, unsigned int timestep, unsigned int numNeurons)
{
    // If we should be applying input during this timestep
    if(shouldApply(timestep)) {
        // Determine if there are any update values this timestep (spikes)
        updateValues(dt, timestep, numNeurons,
            [this](unsigned int neuronID, double)
            {
                injectSpike(neuronID);
            });

        // Upload spikes to GPU if required
        uploadSpikes();
    }
}
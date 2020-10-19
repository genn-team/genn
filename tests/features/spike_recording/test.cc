//--------------------------------------------------------------------------
/*! \file spike_recording/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "spike_recording_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
public:
    //------------------------------------------------------------------------
    // SimulationTest virtuals
    //------------------------------------------------------------------------
    virtual void Init() override
    {
        // Allocate spike recording buffers
        allocateRecordingBuffers(100);

        // Allocate enough memory for 2 spikes per neuron per source
        allocatespikeTimesPop(200);

        // Loop through neurons
        for(unsigned int n = 0; n < 100; n++) {
            startSpikePop[n] = (n * 2) + 0;

            // Configure neuron to spike twice during simulation
            const float timestep1 = (float)n;
            const float timestep2 = (float)(99 - n);
            spikeTimesPop[(n * 2) + 0] = std::min(timestep1, timestep2);
            spikeTimesPop[(n * 2) + 1] = std::max(timestep1, timestep2);

            endSpikePop[n] = (n * 2) + 2;
        }

        // Upload spike times
        pushspikeTimesPopToDevice(200);
    }
};

TEST_F(SimTest, SpikeRecording)
{
    // Simulate 100 timesteps
    while(iT < 100) {
        StepGeNN();
    }

    // Copy recording data from device
    pullRecordingBuffersFromDevice();

    // Loop through timesteps
    for(unsigned int t = 0; t < 100; t++) {
        // Calculate indices of neurons which should spike this timestep
        const unsigned int n1 = t;
        const unsigned int n2 = 99 - t;
        
        // Build bitset 
        uint32_t correct[4] = {0, 0, 0, 0};
        correct[n1 / 32] |= (1 << (n1 % 32));
        correct[n2 / 32] |= (1 << (n2 % 32));
        
        // Check that this matches actual recording
        EXPECT_TRUE(std::equal(&correct[0], &correct[4], &recordSpkPop[4 * t]));
    }
}

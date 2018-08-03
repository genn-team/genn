#pragma once

// Test includes
#include "simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestDecoderDenDelayMatrix
//----------------------------------------------------------------------------
class SimulationTestDecoderDenDelayMatrix : public SimulationTest
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        for (int i = 0; i < 100; i++)
        {
            // Active single neuron in each timestep
            const unsigned int activeNeuron = (i % 10);
            glbSpkCntPre[0] = 1;
            glbSpkPre[0] = activeNeuron;

#ifndef CPU_ONLY
            if(GetParam())
            {
                pushPreSpikesToDevice();
            }
#endif  // CPU_ONLY

            // Step GeNN
            StepGeNN();

            // If delay code should be active, return false if x isn't ten (all neurons)
            if(activeNeuron == 9) {
                if(fabs(xPost[0] - 10.0f) >= 1E-5) {
                    return false;
                }
            }
            // Otherwise, return false if x isn't zero
            else {
                if(fabs(xPost[0]) >= 1E-5) {
                    return false;
                }
            }

        }

        return true;
    }
};
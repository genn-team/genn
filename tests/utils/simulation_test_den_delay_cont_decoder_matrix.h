#pragma once

// Standard C includes
#include <cmath>

// Test includes
#include "simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestContDecoderDenDelayMatrix
//----------------------------------------------------------------------------
class SimulationTestContDecoderDenDelayMatrix : public SimulationTest
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        for (int i = 0; i < 100; i++) {
            // Active single neuron in each timestep
            const unsigned int activeNeuron = (i % 10);
            for(unsigned int j = 0; j < 10; j++) {
                xPre[j] = (j == activeNeuron) ? 1.0f : 0.0f;
            }

            pushPreStateToDevice();

            // Step GeNN
            StepGeNN();

            // If delay code should be active, return false if x isn't ten (all neurons)
            if(activeNeuron == 9) {
                if(std::fabs(xPost[0] - 10.0f) >= 1E-5) {
                    return false;
                }
            }
            // Otherwise, return false if x isn't zero
            else {
                if(std::fabs(xPost[0]) >= 1E-5) {
                    return false;
                }
            }

        }

        return true;
    }
};

#pragma once

// Standard C includes
#include <cmath>

// Test includes
#include "simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestContDecoderMatrix
//----------------------------------------------------------------------------
class SimulationTestContDecoderMatrix : public SimulationTest
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        for (int i = 0; i < 100; i++) {
            // What value should neurons be representing this time step?
            const unsigned int inValue = (i % 10) + 1;

            // Activate this neuron
            // **NOTE** neurons start from zero
            for(unsigned int j = 0; j < 10; j++) {
                xPre[j] = (j == (inValue - 1)) ? 1.0f : 0.0f;
            }

            pushPreStateToDevice();

            // Step GeNN
            StepGeNN();

            // Loop through output neurons
            unsigned int outValue = 0;
            for(unsigned int j = 0; j < 4; j++) {
                // If this neuron is representing 1 add value it represents to output
                if(std::fabs(xPost[j] - 1.0f) < 1E-5) {
                    outValue += (1 << j);
                }
            }

            // If input value isn't correctly decoded, return false
            if(outValue != inValue) {
                return false;
            }
        }

        return true;
    }
};

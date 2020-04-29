#pragma once

// Standard C includes
#include <cmath>

// Test includes
#include "simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestDecoderSharedMatrix
//----------------------------------------------------------------------------
class SimulationTestDecoderSharedMatrix : public SimulationTest
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        for (int i = 0; i < (int)(10.0f / DT); i++) {
            // What value should neurons be representing this time step?
            const unsigned int inValue = (i / 10) + 1;

            // Input spike representing value
            // **NOTE** neurons start from zero
            glbSpkCntPre[0] = 1;
            glbSpkPre[0] = (inValue - 1);

            // Push spikes to device
            pushPreSpikesToDevice();

            // Step GeNN
            StepGeNN();

            // Loop through output neurons
            unsigned int outValue1 = 0;
            unsigned int outValue2 = 0;
            for(unsigned int j = 0; j < 4; j++) {
                // If this neuron is representing 1 add value it represents to output
                if(std::fabs(xPost1[j] - 1.0f) < 1E-5) {
                    outValue1 += (1 << j);
                }
                
                // If this neuron is representing 1 add value it represents to output
                if(std::fabs(xPost2[j] - 1.0f) < 1E-5) {
                    outValue2 += (1 << j);
                }
            }

            // If input value isn't correctly decoded, return false
            if(outValue1 != inValue || outValue2 != inValue) {
                return false;
            }
        }

        return true;
    }
};

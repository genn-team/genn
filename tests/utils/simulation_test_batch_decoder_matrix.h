#pragma once

// Standard C includes
#include <cmath>

// Test includes
#include "simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestBatchDecoderMatrix
//----------------------------------------------------------------------------
class SimulationTestBatchDecoderMatrix : public SimulationTest
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        for (int i = 0; i < (int)(10.0f / DT); i++) {
            // What value should neurons be representing this time step?
            const unsigned int inValue1 = (i / 10) + 1;
            const unsigned int inValue2 = 10 - (i / 10);

            // Input spike representing value
            // **NOTE** neurons start from zero
            glbSpkCntPre[0] = 1;
            glbSpkCntPre[1] = 1;
            glbSpkPre[0] = (inValue1 - 1);
            glbSpkPre[10] = (inValue2 - 1);

            // Push spikes to device
            pushPreSpikesToDevice();

            // Step GeNN
            StepGeNN();

            // Loop through output neurons
            unsigned int outValue1 = 0;
            unsigned int outValue2 = 0;
            for(unsigned int j = 0; j < 4; j++) {
                // If this neuron is representing 1 add value it represents to output
                if(std::fabs(xPost[j] - 1.0f) < 1E-5) {
                    outValue1 += (1 << j);
                }
                
                // If this neuron is representing 1 add value it represents to output
                if(std::fabs(xPost[j + 4] - 1.0f) < 1E-5) {
                    outValue2 += (1 << j);
                }
            }

            // If input value isn't correctly decoded, return false
            if(outValue1 != inValue1 || outValue2 != inValue2) {
                std::cout << outValue1 << "!=" << inValue1 << " or " << outValue2 << "!=" << inValue2 << std::endl;
                return false;
            }
        }

        return true;
    }
};

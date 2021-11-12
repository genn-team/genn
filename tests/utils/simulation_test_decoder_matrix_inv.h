#pragma once

// Standard C includes
#include <cmath>

// Test includes
#include "simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestDecoderMatrixInv
//----------------------------------------------------------------------------
class SimulationTestDecoderMatrixInv : public SimulationTest
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
                xPost[j] = (j == (inValue - 1)) ? 1.0f : 0.0f;
            }

            pushPostStateToDevice();

            // Step GeNN
            StepGeNN();

	    // Loop through output neurons
	    unsigned int outValue = 0;
	    for(unsigned int j = 0; j < 4; j++) {
	        // If this neuron is representing 1 add value it represents to output
	        if(std::fabs(xPre[j] - 1.0f) < 1E-5) {
		    outValue += (1 << j);
	         }
	    }
	    pullPreCurrentSpikesFromDevice();
	    if (std::fabs(t/0.2f - (int) (t/0.2f)) < 1e-5) {
		// If input value isn't correctly decoded, return false
	      if(outValue != inValue) {
		    return false;
                }
	    }
	    else {
	        if(fabs(outValue) > 1e-5) {
		  return false;
	        }
	    }
        }

        return true;
    }
};

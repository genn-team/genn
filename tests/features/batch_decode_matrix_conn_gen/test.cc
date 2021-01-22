//--------------------------------------------------------------------------
/*! \file batch_decode_matrix_conn_genn/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "batch_decode_matrix_conn_gen_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{       
};

TEST_F(SimTest, BatchDecodeMatrixConnGen)
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
        EXPECT_EQ(outValue1, inValue1);
        EXPECT_EQ(outValue2, inValue2);
    }    
}

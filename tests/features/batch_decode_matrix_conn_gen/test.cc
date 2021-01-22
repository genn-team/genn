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
        const unsigned int inValue[2] = {(i / 10) + 1,
                                         10 - (i / 10)};

        // Input spike representing value in each batch
        // **NOTE** neurons start from zero
        for(unsigned int b = 0; b < 2; b++) {
            glbSpkCntPre[b] = 1;
            glbSpkPre[b * 10] = (inValue[b] - 1);
        }

        // Push spikes to device
        pushPreSpikesToDevice();

        // Step GeNN
        StepGeNN();

        // Loop through batches
        for(unsigned int b = 0; b < 2; b++) {
            // Loop through output neurons
            unsigned int outDense = 0;
            unsigned int outSparse = 0;
            unsigned int outProcedural = 0;
            unsigned int outBitmask = 0;
            for(unsigned int j = 0; j < 4; j++) {
                // If this neuron is representing 1 add value it represents to output
                if(std::fabs(xPostDense[j + (b * 4)] - 1.0f) < 1E-5) {
                    outDense += (1 << j);
                }
                if(std::fabs(xPostSparse[j + (b * 4)] - 1.0f) < 1E-5) {
                    outSparse += (1 << j);
                }
                if(std::fabs(xPostProcedural[j + (b * 4)] - 1.0f) < 1E-5) {
                    outProcedural += (1 << j);
                }
                if(std::fabs(xPostBitmask[j + (b * 4)] - 1.0f) < 1E-5) {
                    outBitmask += (1 << j);
                }
            }
            
            // If input value isn't correctly decoded, return false
            EXPECT_EQ(outDense, inValue[b]);
            EXPECT_EQ(outSparse, inValue[b]);
            EXPECT_EQ(outProcedural, inValue[b]);
            EXPECT_EQ(outBitmask, inValue[b]);
        }
        
    }    
}

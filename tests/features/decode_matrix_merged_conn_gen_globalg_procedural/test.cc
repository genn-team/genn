//--------------------------------------------------------------------------
/*! \file decode_matrix_merged_conn_gen_globalg_procedural/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "decode_matrix_merged_conn_gen_globalg_procedural_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
public:
    //----------------------------------------------------------------------------
    // SimulationTest virtuals
    //----------------------------------------------------------------------------
    virtual void Init()
    {
    }
    
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        bool allRowsMatch = true;
        for (int i = 0; i < (int)(10.0f / DT); i++) {
            // What value should neurons be representing this time step?
            const unsigned int inValue = (i / 10) + 1;

            // Input spike representing value
            // **NOTE** neurons start from zero
            glbSpkCntPre[0] = 1;
            glbSpkPre[0] = (inValue - 1);

            pushPreSpikesToDevice();

            // Step GeNN
            StepGeNN();

            // Loop through output neurons
            unsigned int outValue1 = 0;
            unsigned int outValue2 = 0;
            for(unsigned int j = 0; j < 32; j++) {
                // If output neuron values are the same, non-zero value connectivity is broken!
                if(std::fabs(xPost1[j] - 1.0f) < 1E-5) {
                    outValue1 += (1 << j);
                }

                // If this neuron is representing 1 add value it represents to output
                if(std::fabs(xPost2[j] - 1.0f) < 1E-5) {
                    outValue2 += (1 << j);
                }
            }

            // If output values differ, connectivity is at least partially randomized
            if(outValue1 != outValue2) {
                allRowsMatch = false;
            }
        }

        // All rows SHOULDN'T mathc
        return !allRowsMatch;
    }
};

TEST_F(SimTest, DecodeMatrixMergedConnGenGlobalgProcedural)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

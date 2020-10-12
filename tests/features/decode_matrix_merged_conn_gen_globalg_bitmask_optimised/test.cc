//--------------------------------------------------------------------------
/*! \file decode_matrix_merged_conn_gen_globalg_bitmask_optimised/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "decode_matrix_merged_conn_gen_globalg_bitmask_optimised_CODE/definitions.h"

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

TEST_F(SimTest, DecodeMatrixMergedConnGenGlobalgBitmaskOptimised)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

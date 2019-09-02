//--------------------------------------------------------------------------
/*! \file decode_matrix_merged_globalg_dense/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Standard C includes
#include <cmath>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "decode_matrix_merged_globalg_dense_CODE/definitions.h"

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
            // How many neurons should be active this timestep
            const unsigned int num_active_pre = i / 10;

            // Activate this many neurons
            glbSpkCntPre1[0] = num_active_pre;
            glbSpkCntPre2[0] = num_active_pre;
            for(unsigned int s = 0; s < num_active_pre; s++) {
                glbSpkPre1[s] = s;
                glbSpkPre2[s] = s;
            }

            pushPre1SpikesToDevice();
            pushPre2SpikesToDevice();

            // Step GeNN
            StepGeNN();

            // Loop through output neurons
            for(unsigned int j = 0; j < 4; j++) {
                // If activation of postsynaptic neuron is incorrect fail
                if(std::fabs(xPost[j] - (float)(2 * num_active_pre)) >= 1E-5) {
                    return false;
                }
            }

        }

        return true;
    }
};

TEST_F(SimTest, DecodeMatrixMergedGlobalgDense)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

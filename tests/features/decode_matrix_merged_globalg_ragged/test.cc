//--------------------------------------------------------------------------
/*! \file decode_matrix_merged_globalg_ragged/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
#include <cmath>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "decode_matrix_merged_globalg_ragged_CODE/definitions.h"

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
    virtual void Init() override
    {
        // Loop through presynaptic neurons
        for(unsigned int i = 0; i < 10; i++)
        {
            // Initially zero row length
            rowLengthSyn1[i] = 0;
            rowLengthSyn2[i] = 0;
            for(unsigned int j = 0; j < 4; j++)
            {
                // Get value this post synaptic neuron represents
                const unsigned int j_value = (1 << j);

                // If this postsynaptic neuron should be connected, add index
                if(((i + 1) & j_value) != 0)
                {
                    const unsigned int idx1 = (i * 4) + rowLengthSyn1[i]++;
                    const unsigned int idx2 = (i * 4) + rowLengthSyn2[i]++;
                    indSyn1[idx1] = j;
                    indSyn2[idx2] = j;
                }
            }
        }
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        for (int i = 0; i < (int)(10.0f / DT); i++) {
            // What value should neurons be representing this time step?
            const unsigned int in_value = (i / 10) + 1;

            // Input spikes representing value
            // **NOTE** neurons start from zero
            glbSpkCntPre1[0] = 1;
            glbSpkCntPre2[0] = 1;
            glbSpkPre1[0] = (in_value - 1);
            glbSpkPre2[0] = (in_value - 1);

            // Push spikes to device
            pushPre1SpikesToDevice();
            pushPre2SpikesToDevice();

            // Step GeNN
            StepGeNN();

            // Loop through output neurons
            unsigned int out_value = 0;
            for(unsigned int j = 0; j < 4; j++) {
                // If this neuron is representing 1 add value it represents to output
                if(std::fabs(xPost[j] - 2.0f) < 1E-5) {
                    out_value += (1 << j);
                }
            }

            // If input value isn't correctly decoded, return false
            if(out_value != in_value) {
                return false;
            }
        }

        return true;
    }
};

TEST_F(SimTest, DecodeMatrixMergedGlobalgRagged)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

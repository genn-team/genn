//--------------------------------------------------------------------------
/*! \file decode_matrix_globalg_ragged_hybrid/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "decode_matrix_globalg_ragged_hybrid_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_decoder_matrix.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTestDecoderMatrix
{
public:
    //----------------------------------------------------------------------------
    // SimulationTest virtuals
    //----------------------------------------------------------------------------
    virtual void Init()
    {
        // Loop through presynaptic neurons
        for(unsigned int i = 0; i < 10; i++)
        {
            // Initially zero row length
            rowLengthSyn[i] = 0;
            for(unsigned int j = 0; j < 4; j++)
            {
                // Get value this post synaptic neuron represents
                const unsigned int j_value = (1 << j);

                // If this postsynaptic neuron should be connected, add index
                if(((i + 1) & j_value) != 0)
                {
                    const unsigned int idx = (i * 4) + rowLengthSyn[i]++;
                    indSyn[idx] = j;
                }
            }
        }
    }
};

TEST_F(SimTest, DecodeMatrixGlobalgRaggedHybrid)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

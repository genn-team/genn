//--------------------------------------------------------------------------
/*! \file pre_output_decode_matrix_post_learn_individualg_dense/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "pre_output_decode_matrix_post_learn_individualg_dense_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_decoder_matrix_inv.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTestDecoderMatrixInv
{
public:
    //----------------------------------------------------------------------------
    // SimulationTest virtuals
    //----------------------------------------------------------------------------
    virtual void Init()
    {
        // Loop through presynaptic neurons
        unsigned int c = 0;
        for(unsigned int i = 0; i < 4; i++)
        {
            // Loop through postsynaptic neurons
            for(unsigned int j = 0; j < 10; j++)
            {
                // Get value this pre synaptic neuron represents
                const unsigned int i_value = (1 << i);

                // If this presynaptic neuron should be connected, add 1.0 otherwise 0.0
                gSyn[c++] = (((j + 1) & i_value) != 0) ? 1.0f : 0.0f;
            }
        }
    }
};

TEST_F(SimTest, PreOutputDecodeMatrixPostLearnIndividualgDense)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

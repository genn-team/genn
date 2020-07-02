//--------------------------------------------------------------------------
/*! \file decode_matrix_individualg_dense/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "decode_shared_matrix_individualg_dense_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_decoder_shared_matrix.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTestDecoderSharedMatrix
{
public:
    //----------------------------------------------------------------------------
    // SimulationTest virtuals
    //----------------------------------------------------------------------------
    virtual void Init()
    {
        // Loop through presynaptic neurons
        unsigned int c = 0;
        for(unsigned int i = 0; i < 10; i++)
        {
            // Set start index for this presynaptic neuron's weight matrix row
            for(unsigned int j = 0; j < 4; j++)
            {
                // Get value this post synaptic neuron represents
                const unsigned int j_value = (1 << j);

                // If this postsynaptic neuron should be connected, add 1.0 otherwise 0.0
                gSyn1[c++] = (((i + 1) & j_value) != 0) ? 1.0f : 0.0f;

            }
        }
    }
};

TEST_F(SimTest, DecodeSharedMatrixIndividualgDense)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

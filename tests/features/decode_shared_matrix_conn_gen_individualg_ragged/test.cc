//--------------------------------------------------------------------------
/*! \file decode_matrix_conn_gen_individualg_ragged/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "decode_shared_matrix_conn_gen_individualg_ragged_CODE/definitions.h"

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
    }
};

TEST_F(SimTest, DecodeSharedMatrixConnGenIndividualgRagged)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

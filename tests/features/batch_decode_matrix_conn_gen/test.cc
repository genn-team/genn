//--------------------------------------------------------------------------
/*! \file batch_decode_matrix_conn_genn_individualg_dense/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "batch_decode_matrix_conn_gen_individualg_dense_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_batch_decoder_matrix.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTestBatchDecoderMatrix
{
};

TEST_F(SimTest, BatchDecodeMatrixConnGenIndividualgDense)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

//--------------------------------------------------------------------------
/*! \file decode_matrix_conn_gen_globalg_ragged/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include DEFINITIONS_HEADER

// This test does't support building for GPU and testing on CPU
#define CPU_GPU_NOT_SUPPORTED

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
    }
};

TEST_P(SimTest, CorrectDecoding)
{
    // Initialize sparse arrays
    initdecode_matrix_conn_gen_globalg_ragged_new();

    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

#ifndef CPU_ONLY
auto simulatorBackends = ::testing::Values(true);
#else
auto simulatorBackends = ::testing::Values(false);
#endif

WRAPPED_INSTANTIATE_TEST_CASE_P(MODEL_NAME,
                                SimTest,
                                simulatorBackends);
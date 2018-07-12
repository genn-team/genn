// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include DEFINITIONS_HEADER

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_den_delay_cont_decoder_matrix.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTestContDecoderDenDelayMatrix
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
            // Set rowlength to 1
            CSyn.rowLength[i] = 1;

            // Connect row to output neuron with weight of one and dendritic delay of (9 - i)
            CSyn.ind[i] = 0;
            gSyn[i] = 1.0f;
            dSyn[i] = (uint8_t)(9 - i);
        }
    }
};

TEST_P(SimTest, CorrectDecoding)
{
    // Initialize sparse arrays
    initdecode_matrix_cont_den_delay_individualg_ragged_new();

    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

#ifndef CPU_ONLY
auto simulatorBackends = ::testing::Values(true, false);
#else
auto simulatorBackends = ::testing::Values(false);
#endif

WRAPPED_INSTANTIATE_TEST_CASE_P(MODEL_NAME,
                                SimTest,
                                simulatorBackends);

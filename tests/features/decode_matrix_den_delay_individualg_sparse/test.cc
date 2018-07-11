// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include DEFINITIONS_HEADER

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_den_delay_decoder_matrix.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTestDecoderDenDelayMatrix
{
public:
    //----------------------------------------------------------------------------
    // SimulationTest virtuals
    //----------------------------------------------------------------------------
    virtual void Init()
    {
        // Allocate sparse matrix
        allocateSyn(10);

        // Loop through presynaptic neurons
        for(unsigned int i = 0; i < 10; i++)
        {
            // Set start index for this presynaptic neuron's weight matrix row
            CSyn.indInG[i] = i;
            CSyn.ind[i] = 0;
            gSyn[i] = 1.0f;
            dSyn[i] = (uint8_t)(9 - i);
        }

        // Add end index
        CSyn.indInG[10] = 10;
    }
};

TEST_P(SimTest, CorrectDecoding)
{
#ifndef CPU_ONLY
    // Initialize sparse arrays
    initializeAllSparseArrays();
#endif  // CPU_ONLY

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
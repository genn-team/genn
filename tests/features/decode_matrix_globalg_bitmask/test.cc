// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "decode_matrix_globalg_bitmask_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_decoder_matrix.h"

//----------------------------------------------------------------------------
// SimulationTestGlobalGBitmask
//----------------------------------------------------------------------------
class SimulationTestGlobalGBitmask : public SimulationTestDecoderMatrix
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
                unsigned int gid = ((i * 4) + j);
                if(((i + 1) & j_value) != 0)
                {
                    setB(gpSyn[gid >> 5], gid & 31);
                }
                else
                {
                    delB(gpSyn[gid >> 5], gid & 31);
                }

            }
        }
    }
};

TEST_P(SimulationTestGlobalGBitmask, CorrectDecoding)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

#ifndef CPU_ONLY
auto simulatorBackends = ::testing::Values(true, false);
#else
auto simulatorBackends = ::testing::Values(false);
#endif

INSTANTIATE_TEST_CASE_P(DecodeMatrix,
                        SimulationTestGlobalGBitmask,
                        simulatorBackends);
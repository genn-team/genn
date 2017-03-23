// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include DEFINITIONS_HEADER

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_decoder_matrix.h"

//----------------------------------------------------------------------------
// SimulationTestIndividualGSparse
//----------------------------------------------------------------------------
class SimulationTestIndividualGSparse : public SimulationTestDecoderMatrix
{
public:
    //----------------------------------------------------------------------------
    // SimulationTest virtuals
    //----------------------------------------------------------------------------
    virtual void Init()
    {
        // Allocate sparse matrix
        allocateSyn(17);

        // Loop through presynaptic neurons
        unsigned int c = 0;
        for(unsigned int i = 0; i < 10; i++)
        {
            // Set start index for this presynaptic neuron's weight matrix row
            CSyn.indInG[i] = c;
            for(unsigned int j = 0; j < 4; j++)
            {
                // Get value this post synaptic neuron represents
                const unsigned int j_value = (1 << j);

                // If this postsynaptic neuron should be connected, add index
                if(((i + 1) & j_value) != 0)
                {
                    CSyn.ind[c++] = j;
                }
            }
        }

        // Add end index
        CSyn.indInG[10] = c;

        // Fill weights
        std::fill(&gSyn[0], &gSyn[17], 1.0f);
    }
};

TEST_P(SimulationTestIndividualGSparse, CorrectDecoding)
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

INSTANTIATE_TEST_CASE_P(DecodeMatrix,
                        SimulationTestIndividualGSparse,
                        simulatorBackends);
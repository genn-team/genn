// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "decode_matrix_globalg_dense_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestGlobalGDense
//----------------------------------------------------------------------------
class SimulationTestGlobalGDense : public SimulationTest
{
public:
    //----------------------------------------------------------------------------
    // SimulationTest virtuals
    //----------------------------------------------------------------------------
    virtual void Init()
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        for (int i = 0; i < (int)(10.0f / DT); i++)
        {
            // How many neurons should be active this timestep
            const unsigned int num_active_pre = i / 10;

            // Activate this many neurons
            glbSpkCntPre[0] = num_active_pre;
            for(unsigned int s = 0; s < num_active_pre; s++)
            {
                glbSpkPre[s] = s;
            }

#ifndef CPU_ONLY
            if(GetParam())
            {
                pushPreSpikesToDevice();
            }
#endif  // CPU_ONLY

            // Step GeNN
            StepGeNN();

            // Loop through output neurons
            for(unsigned int j = 0; j < 4; j++)
            {
                // If activation of postsynaptic neuron is incorrect fail
                if(fabs(xPost[j] - (float)num_active_pre) >= 1E-5)
                {
                    return false;
                }
            }

        }

        return true;
    }
};

TEST_P(SimulationTestGlobalGDense, CorrectDecoding)
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
                        SimulationTestGlobalGDense,
                        simulatorBackends);
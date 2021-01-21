//--------------------------------------------------------------------------
/*! \file batch_pull_current_var/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "batch_pull_current_var_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

void checkVar()
{
    // Determine which batch spike SHOULD be in
    // **NOTE** iT was advanced after simulation step
    const unsigned int correctSpikeBatch = (iT - 1)/ 10;
    const unsigned int correctSpikeNeuron = (iT - 1) % 10;
    
    // Loop through batches
    for(unsigned int b = 0; b < 10; b++) {
        const scalar *xPop = getCurrentxPop(b);
        const scalar *xPopDelay = getCurrentxPopDelay(b);
        
        for(unsigned int i = 0; i < 10; i++) {
            const float correct = (float)(((iT -1) * 100) + (b * 10) + i);
            
            EXPECT_FLOAT_EQ(correct, xPop[i]);
            EXPECT_FLOAT_EQ(correct, xPopDelay[i]);
        }
    }
    
}

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
public:
    virtual void Init() override
    {
    }
};

TEST_F(SimTest, BatchPullCurrentVar)
{
    while(iT < 100) {
        StepGeNN();
        
        // Download all spikes from device and check
        pullxPopFromDevice();
        pullxPopDelayFromDevice();
        checkVar();
        
        // Zero host data structures
        std::fill_n(xPop, 100, 0);
        std::fill_n(xPopDelay, 600, 0.0f);
        
        // Download current spikes from device
        pullCurrentxPopFromDevice();
        pullCurrentxPopDelayFromDevice();
        checkVar();
    }
}


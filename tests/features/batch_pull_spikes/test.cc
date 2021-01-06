//--------------------------------------------------------------------------
/*! \file batch_pull_spikes/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "batch_pull_spikes_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

void checkSpikes()
{
    // Determine which batch spike SHOULD be in
    // **NOTE** iT was advanced after simulation step
    const unsigned int correctSpikeBatch = (iT - 1)/ 10;
    const unsigned int correctSpikeNeuron = (iT - 1) % 10;
    
    // Loop through batches
    for(unsigned int b = 0; b < 10; b++) {
        // If spike should be in this batch
        if(b == correctSpikeBatch) {
            // Assert that only a single spike was emitted
            ASSERT_EQ(glbSpkCntPop[b], 1);
            
            ASSERT_EQ(glbSpkPop[b * 10], correctSpikeNeuron);
        }
        // Otherwise, check there are no spikes
        else {
            ASSERT_EQ(glbSpkCntPop[b], 0);
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
        // Allocate spike times
        allocatespikeTimesPop(100);
        
        // Loop through batches and neurons
        for(unsigned int b = 0; b < 10; b++) {
            for(unsigned int i = 0; i < 10; i++) {
                const unsigned int idx = (b * 10) + i;
                
                // Set spike time to index
                spikeTimesPop[idx] = (scalar)idx;
                
                // Configure spike source
                startSpikePop[idx] = idx;
                endSpikePop[idx] = idx + 1;
            }
        }
        
        // Upload spike times
        pushspikeTimesPopToDevice(100);
    }
};

TEST_F(SimTest, BatchPullSpikes)
{
    while(iT < 100) {
        StepGeNN();
        
        // Download all spikes from device and check
        pullPopSpikesFromDevice();
        checkSpikes();
        
        // Zero host data structures
        std::fill_n(glbSpkCntPop, 10, 0);
        std::fill_n(glbSpkPop, 100, 0);
        
        // Download current spikes from device
        pullPopCurrentSpikesFromDevice();
        checkSpikes();
    }
}


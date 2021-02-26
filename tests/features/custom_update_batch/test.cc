//--------------------------------------------------------------------------
/*! \file custom_update_batch/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Standard C++ includes
#include <array>
#include <numeric>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "custom_update_batch_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

TEST_F(SimTest, CustomUpdateBatch)
{
    while(iT < 1000) {
        StepGeNN();

        if((iT % 100) == 0) {
            // Perform custom update
            updateTest();

            // Pull variables
            pullVNeuronDuplicateSetTimeFromDevice();
            pullVNeuronFromDevice();
            pullUNeuronFromDevice();
           

            // Check all values match time of update
            EXPECT_TRUE(std::all_of(&UNeuron[0], &UNeuron[50],
                        [](scalar v) { return v == t; }));
            // Loop through batches
            for(unsigned int b = 0; b < 5; b++) {
                const unsigned int startIdx = b * 50;
                const unsigned int endIdx = startIdx + 50;
                const float bOffset = b * 1000.0f;
                EXPECT_TRUE(std::all_of(&VNeuronDuplicateSetTime[startIdx], &VNeuronDuplicateSetTime[endIdx],
                            [bOffset](scalar v) { return v == (bOffset + t); }));
                EXPECT_TRUE(std::all_of(&VNeuron[startIdx], &VNeuron[endIdx],
                            [bOffset](scalar v) { return v == (bOffset + t); }));
            }
        }
    }
}


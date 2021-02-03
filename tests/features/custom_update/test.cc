//--------------------------------------------------------------------------
/*! \file custom_init/test.cc

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
#include "custom_update_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

TEST_F(SimTest, CustomInit)
{
    while(iT < 1000) {
        StepGeNN();
        
        if((iT % 100) == 0) {
            // Perform custom update
            updateTest();
            
            // Pull variables
            pullVNeuronSetTimeFromDevice();
            pullVNeuronFromDevice();
            
            // Check all values match time of update
            EXPECT_TRUE(std::all_of(&VNeuronSetTime[0], &VNeuronSetTime[100],
                        [](scalar v) { return v == t; }));

        }
    }
}


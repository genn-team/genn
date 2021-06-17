//--------------------------------------------------------------------------
/*! \file custom_update_delay/test.cc

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
#include "custom_update_delay_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

TEST_F(SimTest, CustomUpdateDelay)
{
    while(iT < 1000) {
        StepGeNN();

        if((iT % 100) == 0) {
            // Perform custom update
            updateTest();

            // Pull variables
            pullCurrentVPreFromDevice();
            pullCurrentUPreFromDevice();
            
            // **YUCK** missing helper
            pullpreSyn2FromDevice();
            
            const scalar *currentVPre = getCurrentVPre();
            const scalar *currentUPre = getCurrentUPre();
            
            // **YUCK** missing helper
            const scalar *currentPreSyn2 = preSyn2 + (spkQuePtrPre * 100);
            
            // Check all values match time of update
            EXPECT_TRUE(std::all_of(&currentVPre[0], &currentVPre[100],
                        [](scalar v) { return v == t; }));
            EXPECT_TRUE(std::all_of(&currentUPre[0], &currentUPre[100],
                        [](scalar v) { return v == t; }));
            EXPECT_TRUE(std::all_of(&currentPreSyn2[0], &currentPreSyn2[100],
                        [](scalar v) { return v == t; }));

        }
    }
}


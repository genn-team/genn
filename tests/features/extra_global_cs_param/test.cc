//--------------------------------------------------------------------------
/*! \file extra_global_cs_param/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Standard C include
#include <cmath>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "extra_global_cs_param_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"



TEST_F(SimulationTest, ExtraGlobalCSParams)
{
    scalar error = 0.0;
    for(int i = 0; i < 10; i++) {
        kcs = iT;
        StepGeNN();
        
        for(int j = 0; j < 10; j++) {
            if(j == i) {
                error = fabs(xpop[j] - 1.0f);
            }
            else {
                error = fabs(xpop[j]);
            }
        }
    }
    
    // Check total error is less than some tolerance
    EXPECT_LT(error, 1e-6);
}

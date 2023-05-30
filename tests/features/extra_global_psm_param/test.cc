//--------------------------------------------------------------------------
/*! \file extra_global_psm_param/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Standard C include
#include <cmath>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "extra_global_psm_param_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"



TEST_F(SimulationTest, ExtraGlobalPSMParams)
{
    scalar error = 0.0;
    for(int i = 0; i < 11; i++) {
        ksyn = i - 1;
        StepGeNN();
        
        for(int j = 0; j < 10; j++) {
            if(i > 1 && j == (i - 1)) {
                error += fabs(xpost[j] - ((i - 1) * DT));
            }
            else {
                error += fabs(xpost[j]);
            }
        }
    }
    
    // Check total error is less than some tolerance
    EXPECT_LT(error, 1e-6);
}

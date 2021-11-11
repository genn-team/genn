//--------------------------------------------------------------------------
/*! \file batch_var_init/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "batch_var_init_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

TEST_F(SimTest, BatchVarInit)
{
    // Pull kernel from device
    pullgKernelFromDevice();
    
    scalar *kernel = gKernel;
    for(unsigned int b = 0; b < 10; b++) {
        for(unsigned int i = 0; i < 3; i++) {
            for(unsigned int j = 0; j < 3; j++) {
                for(unsigned int k = 0; k < 4; k++) {
                    const float check = std::sqrt((scalar)(i * i) + (scalar)(j * j) + (scalar)(k * k));
                    ASSERT_FLOAT_EQ(check, *kernel++);
                }
            }
        }
    }
}


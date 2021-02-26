//--------------------------------------------------------------------------
/*! \file custom_update_transpose/test.cc

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
#include "custom_update_transpose_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

TEST_F(SimTest, CustomUpdateTranspose)
{
    updateTest();
    pullgDense1FromDevice();
    pullgDense2FromDevice();
    pullgTranspose1FromDevice();
    pullgTranspose2FromDevice();

    for(unsigned int i = 0; i < 100; i++) {
        for(unsigned int j = 0; j < 100; j++) {
            ASSERT_FLOAT_EQ(gDense1[(i * 100) + j], gTranspose1[(j * 100) + i]);
            ASSERT_FLOAT_EQ(gDense2[(i * 100) + j], gTranspose2[(j * 100) + i]);
        }
    }
}


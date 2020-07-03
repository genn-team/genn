//--------------------------------------------------------------------------
/*! \file connect_init/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Standard C++ includes
#include <numeric>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "connect_init_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"


//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

TEST_F(SimTest, ConnectInit)
{
    // Pull connectivity back to host
    pullFixedNumberTotalConnectivityFromDevice();
    pullFixedNumberPostConnectivityFromDevice();

    // Test connectivity
    EXPECT_EQ(std::accumulate(&rowLengthFixedNumberTotal[0], &rowLengthFixedNumberTotal[100], 0u), 1000);
    EXPECT_TRUE(std::all_of(&rowLengthFixedNumberPost[0], &rowLengthFixedNumberPost[100],
                            [](unsigned int rowLength) { return rowLength == 10; }));

    // **TODO** we could also build a histogram of postsynaptic neurons and check that they are approximately uniformly distributed
}


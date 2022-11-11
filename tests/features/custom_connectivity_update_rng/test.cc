//--------------------------------------------------------------------------
/*! \file custom_connectivity_update_rng/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "custom_connectivity_update_rng_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"
#include "../../utils/stats.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
    virtual void Init() final
    {
        allocateOutputRNGTest(1000 * 1000);
    }
};


TEST_F(SimTest, CustomConnectivityUpdate)
{
    // Launch custom update to generate a bunch of random numbers
    updateRNGTest();
    
    pullOutputRNGTestFromDevice(1000 * 1000);

    // Perform Kolmogorov-Smirnov test
    double d;
    double prob;
    std::vector<double> samples(&OutputRNGTest[0], &OutputRNGTest[1000 * 1000]);
    std::tie(d, prob) = Stats::kolmogorovSmirnovTest(samples, Stats::uniformCDF);

    // Check p value passes 95% confidence interval
    EXPECT_GT(prob, 0.05);
    
}

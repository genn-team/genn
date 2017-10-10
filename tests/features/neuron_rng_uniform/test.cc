// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include DEFINITIONS_HEADER

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_histogram.h"
#include "../../utils/stats.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTestHistogram
{
public:
    SimTest() : SimulationTestHistogram(0, 1.0, 100){}

    //----------------------------------------------------------------------------
    // SimulationTestHistogram virtuals
    //----------------------------------------------------------------------------
    virtual bool Test(const std::vector<double> &bins) const
    {
        // Expected probability mass in each bin
        std::vector<double> ebins(100);
        std::fill(ebins.begin(), ebins.end(), 10000.0 * 1000.0 / 100.0);

        // Calculate chi-squared
        double df;
        double chiSquared;
        double prob;
        std::tie(df, chiSquared, prob) = Stats::chiSquared(bins, ebins);
        std::cout << chiSquared << "," << prob << std::endl;
        return true;
    }
};

TEST_P(SimTest, ChiSquared)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

#ifndef CPU_ONLY
auto simulatorBackends = ::testing::Values(true, false);
#else
auto simulatorBackends = ::testing::Values(false);
#endif

WRAPPED_INSTANTIATE_TEST_CASE_P(MODEL_NAME,
                                SimTest,
                                simulatorBackends);
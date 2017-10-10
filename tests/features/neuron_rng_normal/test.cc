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
    SimTest() : SimulationTestHistogram(-2.0, 2.0, 100){}

    //----------------------------------------------------------------------------
    // SimulationTestHistogram virtuals
    //----------------------------------------------------------------------------
    virtual bool Test(const std::vector<double> &bins) const
    {
        // Expected probability mass in each bin
        std::vector<double> ebins = {0.002248, 0.002431, 0.002625, 0.002830, 0.003046, 0.003274, 0.003512,
            0.003762, 0.004024, 0.004297, 0.004581, 0.004876, 0.005181, 0.005497, 0.005823, 0.006158, 0.006503,
            0.006855, 0.007215, 0.007582, 0.007955, 0.008332, 0.008714, 0.009099, 0.009485, 0.009872, 0.010259,
            0.010643, 0.011025, 0.011401, 0.011772, 0.012135, 0.012490, 0.012834, 0.013167, 0.013487, 0.013792,
            0.014082, 0.014355, 0.014610, 0.014845, 0.015061, 0.015255, 0.015426, 0.015575, 0.015700, 0.015801,
            0.015877, 0.015928, 0.015953, 0.015953, 0.015928, 0.015877, 0.015801, 0.015700, 0.015575, 0.015426,
            0.015255, 0.015061, 0.014845, 0.014610, 0.014355, 0.014082, 0.013792, 0.013487, 0.013167, 0.012834,
            0.012490, 0.012135, 0.011772, 0.011401, 0.011025, 0.010643, 0.010259, 0.009872, 0.009485, 0.009099,
            0.008714, 0.008332, 0.007955, 0.007582, 0.007215, 0.006855, 0.006503, 0.006158, 0.005823, 0.005497,
            0.005181, 0.004876, 0.004581, 0.004297, 0.004024, 0.003762, 0.003512, 0.003274, 0.003046, 0.002830,
            0.002625, 0.002431, 0.002248};

        // Scale to number of samples
        std::transform(ebins.begin(), ebins.end(), ebins.begin(),
                       [](double a){ return a * 1000.0 * 1000.0; });

        // Perform chi-squared test
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
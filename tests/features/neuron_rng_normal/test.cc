// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include DEFINITIONS_HEADER

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_samples.h"
#include "../../utils/stats.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTestSamples
{
public:
    //----------------------------------------------------------------------------
    // SimulationTestHistogram virtuals
    //----------------------------------------------------------------------------
    virtual bool Test(std::vector<double> &samples) const
    {
        // Perform Kolmogorov-Smirnov test
        double d;
        double prob;
        std::tie(d, prob) = Stats::kolmogorovSmirnovTest(samples,
            [](double x)
            {
                return 0.5 * (1.0 + erf(x / sqrt(2.0)));
            });

        return (prob > 0.05);
    }
};

TEST_P(SimTest, KolmogorovSmirnovTest)
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
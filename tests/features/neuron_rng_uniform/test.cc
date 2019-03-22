//--------------------------------------------------------------------------
/*! \file neuron_rng_uniform/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "neuron_rng_uniform_CODE/definitions.h"

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
    virtual double Test(std::vector<double> &samples) const
    {
        // Perform Kolmogorov-Smirnov test
        double d;
        double prob;
        std::tie(d, prob) = Stats::kolmogorovSmirnovTest(samples, Stats::uniformCDF);

        return prob;
    }
};

TEST_F(SimTest, NeuronRngUniform)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}

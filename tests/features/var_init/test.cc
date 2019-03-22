//--------------------------------------------------------------------------
/*! \file var_init/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "var_init_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"
#include "../../utils/stats.h"

double gammaCDF4(double x)
{
    return Stats::gammaCDF(4.0, x);
}

// Macro to generate full set of tests for a particular model
#define PROB_TEST(PREFIX, SUFFIX, N) \
    { \
        EXPECT_TRUE(std::all_of(&PREFIX##constant##SUFFIX[0], &PREFIX##constant##SUFFIX[N], [](scalar x){ return (x == 13.0); })); \
        const double PREFIX##uniform##SUFFIX##Prob = getProb(PREFIX##uniform##SUFFIX, N, Stats::uniformCDF); \
        EXPECT_GT(PREFIX##uniform##SUFFIX##Prob, p); \
        const double PREFIX##normal##SUFFIX##Prob = getProb(PREFIX##normal##SUFFIX, N, Stats::normalCDF); \
        EXPECT_GT(PREFIX##normal##SUFFIX##Prob, p); \
        const double PREFIX##exponential##SUFFIX##Prob = getProb(PREFIX##exponential##SUFFIX, N, Stats::exponentialCDF); \
        EXPECT_GT(PREFIX##exponential##SUFFIX##Prob, p); \
        const double PREFIX##gamma##SUFFIX##Prob = getProb(PREFIX##gamma##SUFFIX, N, gammaCDF4); \
        EXPECT_GT(PREFIX##gamma##SUFFIX##Prob, p); \
    } \

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
public:
    //----------------------------------------------------------------------------
    // SimulationTest virtuals
    //----------------------------------------------------------------------------
    virtual void Init()
    {
        // Build sparse connectors
        rowLengthSparse[0] = 10000;
        for(unsigned int i = 0; i < 10000; i++) {
            indSparse[i] = i;
        }
    }
};

template<typename F>
double getProb(scalar *data, unsigned int size, F cdf)
{
    // Convert to double and store in vector
    std::vector<double> doubleData;
    doubleData.reserve(size);
    std::copy_n(data, size, std::back_inserter(doubleData));

    // Perform Kolmogorov-Smirnow test
    double d;
    double prob;
    std::tie(d, prob) = Stats::kolmogorovSmirnovTest(doubleData, cdf);
    return prob;
}

TEST_F(SimTest, Vars)
{
    const double p = 0.02;

    // Pull vars back to host
    pullPopStateFromDevice();
    pullCurrSourceStateFromDevice();
    pullDenseStateFromDevice();
    pullSparseStateFromDevice();

    // Test host-generated vars
    PROB_TEST(, Pop, 10000)
    PROB_TEST(, CurrSource, 10000)
    PROB_TEST(p, Dense, 10000)
    PROB_TEST(, Dense, 10000)
    PROB_TEST(, Sparse, 10000)
}


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
        EXPECT_TRUE(std::all_of(&PREFIX##constant_val##SUFFIX[0], &PREFIX##constant_val##SUFFIX[N], [](scalar x){ return (x == 13.0); })); \
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
};

template<typename F>
double getProb(scalar *data, unsigned int size, F cdf)
{
    // Convert to double and store in vector
    std::vector<double> doubleData;
    doubleData.reserve(size);
    std::copy_n(data, size, std::back_inserter(doubleData));

    // Perform Kolmogorov-Smirnow test
    double prob;
    std::tie(std::ignore, prob) = Stats::kolmogorovSmirnovTest(doubleData, cdf);
    return prob;
}

TEST_F(SimTest, Vars)
{
    // **NOTE**
    // After considerable thought as to why these fail:
    // * Each distribution is tested in 12 different contexts
    // * This test is run using 5 different RNGs (OpenCL, CUDA, MSVC standard library, Clang standard library, GCC standard library)
    // = 60 permutations
    // We want the probability that one or more of the 60 tests fail simply by chance 
    // to be less than 2%; for significance level a the probability that none of the 
    // tests fail is (1-a)^60 which we want to be 0.98, i.e. 98% of the time the test 
    // passes if the algorithm is correct. Hence, a= 1- 0.98^(1/60) = 0.00034
    const double p = 0.00034;

    // Pull vars back to host
    pullPopStateFromDevice();
    pullCurrSourceStateFromDevice();
    pullDenseStateFromDevice();
    pullSparseStateFromDevice();
    pullNeuronCustomUpdateStateFromDevice();
    pullPSMCustomUpdateStateFromDevice();
    pullWUPreCustomUpdateStateFromDevice();
    pullWUPostCustomUpdateStateFromDevice();

    // Test host-generated vars
    PROB_TEST(, Pop, 50000);
    PROB_TEST(, CurrSource, 50000);
    PROB_TEST(p, Dense, 50000);
    PROB_TEST(, Dense, 50000);
    PROB_TEST(, Sparse, 50000);
    PROB_TEST(pre_, Sparse, 50000);
    PROB_TEST(post_, Sparse, 50000);
    PROB_TEST(, NeuronCustomUpdate, 50000);
    PROB_TEST(, CurrentSourceCustomUpdate, 50000);
    PROB_TEST(, PSMCustomUpdate, 50000);
    PROB_TEST(, WUPreCustomUpdate, 50000);
    PROB_TEST(, WUPostCustomUpdate, 50000);
}


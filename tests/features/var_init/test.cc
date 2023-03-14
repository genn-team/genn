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
        const double PREFIX##uniform##SUFFIX##Prob = getProb(PREFIX##uniform##SUFFIX, N, Stats::uniformCDF); \
        EXPECT_GT(PREFIX##uniform##SUFFIX##Prob, p); \
        const double PREFIX##normal##SUFFIX##Prob = getProb(PREFIX##normal##SUFFIX, N, Stats::normalCDF); \
        EXPECT_GT(PREFIX##normal##SUFFIX##Prob, p); \
        const double PREFIX##exponential##SUFFIX##Prob = getProb(PREFIX##exponential##SUFFIX, N, Stats::exponentialCDF); \
        EXPECT_GT(PREFIX##exponential##SUFFIX##Prob, p); \
        const double PREFIX##gamma##SUFFIX##Prob = getProb(PREFIX##gamma##SUFFIX, N, gammaCDF4); \
        EXPECT_GT(PREFIX##gamma##SUFFIX##Prob, p); \
    }

#define PROB_TEST_NEURON(PREFIX, SUFFIX, NUM) \
    { \
        EXPECT_TRUE(std::all_of(&PREFIX##num##SUFFIX[0], &PREFIX##num##SUFFIX[NUM], [](unsigned int x){ return (x == NUM); })); \
        EXPECT_TRUE(std::all_of(&PREFIX##num_batch##SUFFIX[0], &PREFIX##num_batch##SUFFIX[NUM], [](unsigned int x){ return (x == 1); })); \
        PROB_TEST(PREFIX, SUFFIX, NUM) \
    }

#define PROB_TEST_SYNAPSE(PREFIX, SUFFIX, NUM, NUM_PRE, NUM_POST) \
    { \
        EXPECT_TRUE(std::all_of(&PREFIX##num_pre##SUFFIX[0], &PREFIX##num_pre##SUFFIX[NUM], [](unsigned int x){ return (x == NUM_PRE); })); \
        EXPECT_TRUE(std::all_of(&PREFIX##num_post##SUFFIX[0], &PREFIX##num_post##SUFFIX[NUM], [](unsigned int x){ return (x == NUM_POST); })); \
        EXPECT_TRUE(std::all_of(&PREFIX##num_batch##SUFFIX[0], &PREFIX##num_batch##SUFFIX[NUM], [](unsigned int x){ return (x == 1); })); \
        PROB_TEST(PREFIX, SUFFIX, NUM) \
    }
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
    pullKernelStateFromDevice();
    pullNeuronCustomUpdateStateFromDevice();
    pullPSMCustomUpdateStateFromDevice();
    pullWUPreCustomUpdateStateFromDevice();
    pullWUPostCustomUpdateStateFromDevice();
    pullWUSparseCustomUpdateStateFromDevice();
    pullWUDenseCustomUpdateStateFromDevice();
    pullWUKernelCustomUpdateStateFromDevice();
    
    // Test host-generated vars
    PROB_TEST_NEURON(, Pop, 50000);
    PROB_TEST_NEURON(, CurrSource, 50000);
    PROB_TEST_NEURON(p, Dense, 50000);
    PROB_TEST_SYNAPSE(, Dense, 50000, 1, 50000);
    PROB_TEST_SYNAPSE(, Sparse, 50000, 50000, 50000);
    PROB_TEST_NEURON(pre_, Sparse, 50000);
    PROB_TEST_NEURON(post_, Sparse, 50000);
    PROB_TEST_SYNAPSE(, Kernel, 3 * 3 * 5 * 5, 50000, 50000);
    PROB_TEST_NEURON(, NeuronCustomUpdate, 50000);
    PROB_TEST_NEURON(, CurrentSourceCustomUpdate, 50000);
    PROB_TEST_NEURON(, PSMCustomUpdate, 50000);
    PROB_TEST_NEURON(, WUPreCustomUpdate, 50000);
    PROB_TEST_NEURON(, WUPostCustomUpdate, 50000);
    PROB_TEST_SYNAPSE(, WUDenseCustomUpdate, 50000, 1, 50000);
    PROB_TEST_SYNAPSE(, WUSparseCustomUpdate, 50000, 50000, 50000);
    PROB_TEST_SYNAPSE(, WUKernelCustomUpdate, 3 * 3 * 5 * 5, 50000, 50000);
}


//--------------------------------------------------------------------------
/*! \file var_init/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include DEFINITIONS_HEADER

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"
#include "../../utils/stats.h"

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
        allocateSparse(10000);
        CSparse.indInG[0] = 0;
        for(unsigned int i = 0; i < 10000; i++) {
            CSparse.ind[i] = i;
        }
        CSparse.indInG[1] = 10000;
#ifndef CPU_ONLY
        allocateSparseGPU(10000);
        CSparseGPU.indInG[0] = 0;
        for(unsigned int i = 0; i < 10000; i++) {
            CSparseGPU.ind[i] = i;
        }
        CSparseGPU.indInG[1] = 10000;
#endif

        // Build ragged connectors
        CRagged.rowLength[0] = 10000;
        for(unsigned int i = 0; i < 10000; i++) {
            CRagged.ind[i] = i;
        }
#ifndef CPU_ONLY
        CRaggedGPU.rowLength[0] = 10000;
        for(unsigned int i = 0; i < 10000; i++) {
            CRaggedGPU.ind[i] = i;
        }
#endif
        // Call sparse initialisation function
        initvar_init_new();
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

TEST_P(SimTest, Vars)
{
    const double p = 0.025;

    // Test host-generated vars
    PROB_TEST(, Pop, 10000)
    PROB_TEST(, CurrSource, 10000)
    PROB_TEST(p, Dense, 10000)
    PROB_TEST(, Dense, 10000)
    PROB_TEST(, Sparse, 10000)
    PROB_TEST(, Ragged, 10000)

#ifndef CPU_ONLY
    // Pull device-generated vars back to host
    pullPopGPUStateFromDevice();
    pullCurrSourceGPUStateFromDevice();
    pullDenseGPUStateFromDevice();
    pullSparseGPUStateFromDevice();
    pullRaggedGPUStateFromDevice();

    // Test device-generated vars
    PROB_TEST(, PopGPU, 10000)
    PROB_TEST(, CurrSourceGPU, 10000)
    PROB_TEST(p, DenseGPU, 10000)
    PROB_TEST(, DenseGPU, 10000)
    PROB_TEST(, SparseGPU, 10000)
    PROB_TEST(, RaggedGPU, 10000)
#endif
}


#ifndef CPU_ONLY
auto simulatorBackends = ::testing::Values(true);
#else
auto simulatorBackends = ::testing::Values(false);
#endif

WRAPPED_INSTANTIATE_TEST_CASE_P(MODEL_NAME,
                                SimTest,
                                simulatorBackends);
// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include DEFINITIONS_HEADER

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"
#include "../../utils/stats.h"

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
        // Build sparse connector
        allocateSparse(10000);
        CSparse.indInG[0] = 0;
        for(unsigned int i = 0; i < 10000; i++) {
            CSparse.ind[i] = i;
        }
        CSparse.indInG[10000] = 10000;

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
    const double p = 0.05;

    const double uniformNeuronProb = getProb(uniformPop, 10000, Stats::uniformCDF);
    EXPECT_GT(uniformNeuronProb, p);

    const double normalNeuronProb = getProb(normalPop, 10000, Stats::normalCDF);
    EXPECT_GT(normalNeuronProb, p);

    const double exponentialNeuronProb = getProb(exponentialPop, 10000, Stats::exponentialCDF);
    EXPECT_GT(exponentialNeuronProb, p);

    const double uniformPSProb = getProb(puniformDense, 10000, Stats::uniformCDF);
    EXPECT_GT(uniformPSProb, p);

    const double normalPSProb = getProb(pnormalDense, 10000, Stats::normalCDF);
    EXPECT_GT(normalPSProb, p);

    const double exponentialPSProb = getProb(pexponentialDense, 10000, Stats::exponentialCDF);
    EXPECT_GT(exponentialPSProb, p);

    const double uniformWUDenseProb = getProb(uniformDense, 10000, Stats::uniformCDF);
    EXPECT_GT(uniformWUDenseProb, p);

    const double normalWUDenseProb = getProb(normalDense, 10000, Stats::normalCDF);
    EXPECT_GT(normalWUDenseProb, p);

    const double exponentialWUDenseProb = getProb(exponentialDense, 10000, Stats::exponentialCDF);
    EXPECT_GT(exponentialWUDenseProb, p);

    const double uniformWUSparseProb = getProb(uniformSparse, 10000, Stats::uniformCDF);
    EXPECT_GT(uniformWUSparseProb, p);

    const double normalWUSparseProb = getProb(normalSparse, 10000, Stats::normalCDF);
    EXPECT_GT(normalWUSparseProb, p);

    const double exponentialWUSparseProb = getProb(exponentialSparse, 10000, Stats::exponentialCDF);
    EXPECT_GT(exponentialWUSparseProb, p);
}


#ifndef CPU_ONLY
auto simulatorBackends = ::testing::Values(true, false);
#else
auto simulatorBackends = ::testing::Values(false);
#endif

WRAPPED_INSTANTIATE_TEST_CASE_P(MODEL_NAME,
                                SimTest,
                                simulatorBackends);
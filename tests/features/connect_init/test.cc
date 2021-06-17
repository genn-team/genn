//--------------------------------------------------------------------------
/*! \file connect_init/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Standard C++ includes
#include <array>
#include <numeric>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "connect_init_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define CALC_ROW_LENGTH(NAME, HISTOGRAM) calcHistogram(rowLength##NAME, ind##NAME, maxRowLength##NAME, HISTOGRAM)

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

template<size_t N>
void calcHistogram(const unsigned int *rowLength, const uint32_t *ind,
                   unsigned int maxRowLength, std::array<unsigned int, N> &histogram)
{
    // Loop through rows
    for(unsigned int i = 0; i < N; i++) {
        // Loop through synapses
        for(unsigned int j = 0; j < rowLength[i]; j++) {
            // Increment histogram bin
            EXPECT_LT(ind[j],  N);
            histogram[ind[j]]++;
        }
        
        // Advance to next row
        ind += maxRowLength;
    }
}

TEST_F(SimTest, ConnectInit)
{
    // Pull connectivity back to host
    pullFixedNumberTotalConnectivityFromDevice();
    pullFixedNumberPostConnectivityFromDevice();
    pullFixedNumberPreConnectivityFromDevice();

    // Test that connectivity has required properties
    EXPECT_EQ(std::accumulate(&rowLengthFixedNumberTotal[0], &rowLengthFixedNumberTotal[100], 0u), 1000);
    EXPECT_TRUE(std::all_of(&rowLengthFixedNumberPost[0], &rowLengthFixedNumberPost[100],
                            [](unsigned int rowLength) { return rowLength == 10; }));

    std::array<unsigned int, 100> fixedNumPreHist{};
    CALC_ROW_LENGTH(FixedNumberPre, fixedNumPreHist);
    EXPECT_TRUE(std::all_of(fixedNumPreHist.cbegin(), fixedNumPreHist.cend(),
                            [](unsigned int colLength) { return colLength == 10; }));

    // **TODO** we could also build a histogram of postsynaptic neurons and check that they are approximately uniformly distributed
}


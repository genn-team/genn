//--------------------------------------------------------------------------
/*! \file custom_connectivity_update/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Standard includes
#include <bitset>

// Auto-generated simulation code includess
#include "custom_connectivity_update_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

TEST_F(SimTest, CustomConnectivityUpdate)
{
    // Download state
    pullgSynFromDevice();
    pulldSynFromDevice();
    pullSynConnectivityFromDevice();
    pullvCustomConnectivityUpdateFromDevice();
    
    // Loop through rows
    for(unsigned int i = 0; i < 64; i++) {
        // Check correct triangle row length
        ASSERT_EQ(rowLengthSyn[i], 63 - i);

        // Loop through row
        std::bitset<64> row;
        for(unsigned int s = 0; s < rowLengthSyn[i]; s++) {
            const unsigned int idx = (i * maxRowLengthSyn) + s;
            const unsigned int j = indSyn[idx];

            // Check that all variables are correct given the pre and postsynaptic index
            ASSERT_EQ(dSyn[idx], (j * 64) + i);
            ASSERT_FLOAT_EQ(gSyn[idx], (i * 64.0f) + j);
            ASSERT_FLOAT_EQ(vCustomConnectivityUpdate[idx], (i * 64.0f) + j);

            // Set bit in row bitset
            row.set(j);
        }
        const uint64_t correct = 0xFFFFFFFFFFFFFFFEULL << i;
        ASSERT_EQ(row.to_ullong(), correct);
    }

    // Launch custom update
    updateUpdate();

    // Download state
    pullgSynFromDevice();
    pulldSynFromDevice();
    pullSynConnectivityFromDevice();
    pullvCustomConnectivityUpdateFromDevice();
    
    // Loop through rows
    for(unsigned int i = 0; i < 64; i++) {
        // Check correct triangle row length
        ASSERT_EQ(rowLengthSyn[i], (i > 62) ? 0 : (62 - i));

        // Loop through row
        std::bitset<64> row;
        for(unsigned int s = 0; s < rowLengthSyn[i]; s++) {
            const unsigned int idx = (i * maxRowLengthSyn) + s;
            const unsigned int j = indSyn[idx];

            // Check that all variables are correct given the pre and postsynaptic index
            EXPECT_EQ(dSyn[idx], (j * 64) + i);
            EXPECT_FLOAT_EQ(gSyn[idx], (i * 64.0f) + j);
            EXPECT_FLOAT_EQ(vCustomConnectivityUpdate[idx], (i * 64.0f) + j);

            // Set bit in row bitset
            row.set(j);
        }
        
        const uint64_t correct = 0xFFFFFFFFFFFFFFFCULL << i;
        ASSERT_EQ(row.to_ullong(), correct);
    }
}
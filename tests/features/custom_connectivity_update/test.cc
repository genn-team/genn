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

template<typename R, typename B>
void checkConnectivity(R getRowLengthFn, B getCorrectRowWordFn)
{
    // Download state
    pullgSyn1FromDevice();
    pulldSyn1FromDevice();
    pullSyn1ConnectivityFromDevice();
    pullaRemoveSynapseFromDevice();
    
    // Loop through rows
    for(unsigned int i = 0; i < 64; i++) {
        // Check correct triangle row length
        ASSERT_EQ(rowLengthSyn2[i], getRowLengthFn(i));

        // Loop through row
        std::bitset<64> row;
        for(unsigned int s = 0; s < rowLengthSyn1[i]; s++) {
            const unsigned int idx = (i * maxRowLengthSyn1) + s;
            const unsigned int j = indSyn1[idx];

            // Check that all variables are correct given the pre and postsynaptic index
            ASSERT_EQ(dSyn1[idx], (j * 64) + i);
            ASSERT_FLOAT_EQ(gSyn1[idx], (i * 64.0f) + j);
            ASSERT_FLOAT_EQ(aRemoveSynapse[idx], (i * 64.0f) + j);

            // Set bit in row bitset
            row.set(j);
        }
        ASSERT_EQ(row.to_ullong(), getCorrectRowWordFn(i));
    }
}

TEST_F(SimTest, CustomConnectivityUpdate)
{
    // Check initial connectivity is correct
    checkConnectivity([](unsigned int i){ return 64 - i; },
                      [](unsigned int i){ return 0xFFFFFFFFFFFFFFFFULL << i; });
    
    // Launch custom update to remove first synapse from each row
    updateRemoveSynapse();

    // Check modified connectivity is correct
    checkConnectivity([](unsigned int i){ return (i > 63) ? 0 : (63 - i); },
                      [](unsigned int i){ return 0xFFFFFFFFFFFFFFFEULL << i; });

    // Launch custom update to re-add synapse again
    updateAddSynapse();

    // Check connectivity is restored
    checkConnectivity([](unsigned int i){ return 64 - i; },
                      [](unsigned int i){ return 0xFFFFFFFFFFFFFFFFULL << i; });
                      
}

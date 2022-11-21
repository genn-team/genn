//--------------------------------------------------------------------------
/*! \file custom_connectivity_update_delay/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


// Google test includes
#include "gtest/gtest.h"

// Standard includes
#include <bitset>

// Auto-generated simulation code includess
#include "custom_connectivity_update_delay_CODE/definitions.h"

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
void checkConnectivity1(R getRowLengthFn, B getCorrectRowWordFn)
{
    // Download state
    pullgSyn1FromDevice();
    pullSyn1ConnectivityFromDevice();
    
    // Loop through rows
    for(unsigned int i = 0; i < 64; i++) {
        // Check correct triangle row length
        ASSERT_EQ(rowLengthSyn1[i], getRowLengthFn(i));

        // Loop through row
        std::bitset<64> row;
        for(unsigned int s = 0; s < rowLengthSyn1[i]; s++) {
            const unsigned int idx = (i * maxRowLengthSyn1) + s;
            const unsigned int j = indSyn1[idx];

            // Check that all variables are correct given the pre and postsynaptic index
            ASSERT_FLOAT_EQ(gSyn1[idx], (i * 64.0f) + j);

            // Set bit in row bitset
            row.set(j);
        }
        ASSERT_EQ(row.to_ullong(), getCorrectRowWordFn(i));
    }
}

template<typename R, typename B>
void checkConnectivity2(R getRowLengthFn, B getCorrectRowWordFn)
{
    // Download state
    pullgSyn2FromDevice();
    pullSyn2ConnectivityFromDevice();
    
    // Loop through rows
    for(unsigned int i = 0; i < 64; i++) {
        // Check correct triangle row length
        ASSERT_EQ(rowLengthSyn2[i], getRowLengthFn(i));

        // Loop through row
        std::bitset<64> row;
        for(unsigned int s = 0; s < rowLengthSyn2[i]; s++) {
            const unsigned int idx = (i * maxRowLengthSyn2) + s;
            const unsigned int j = indSyn2[idx];

            // Check that all variables are correct given the pre and postsynaptic index
            ASSERT_FLOAT_EQ(gSyn2[idx], (i * 64.0f) + j);

            // Set bit in row bitset
            row.set(j);
        }
        ASSERT_EQ(row.to_ullong(), getCorrectRowWordFn(i));
    }
}

TEST_F(SimTest, CustomConnectivityUpdateDelay)
{
    // Check initial connectivity is correct
    checkConnectivity1([](unsigned int){ return 64; },
                       [](unsigned int i){ return 0xFFFFFFFFFFFFFFFFULL; });
    
    // Check initial connectivity is correct
    checkConnectivity2([](unsigned int){ return 64; },
                       [](unsigned int i){ return 0xFFFFFFFFFFFFFFFFULL; });
                       
    // Run for 5 timesteps
    while(iT < 5) {
        stepTime();
    }

    // Run update to remove selected timesteps
    updateRemoveSynapse();

    // Check modified connectivity is correct
    checkConnectivity1([](unsigned int){ return 63; },
                       [](unsigned int i)
                       { 
                           return 0xFFFFFFFFFFFFFFFFULL & ~(1ULL << ((i + 4) % 64)); 
                       });
    checkConnectivity2([](unsigned int){ return 63; },
                       [](unsigned int i)
                       { 
                           return 0xFFFFFFFFFFFFFFFFULL & ~(1ULL << ((i + 4) % 64)); 
                       });

}

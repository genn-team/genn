//--------------------------------------------------------------------------
/*! \file num/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
#include <algorithm>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "num_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

TEST_F(SimTest, Num)
{
    // Simulate timestep and trigger custom update
    StepGeNN();
    updateTest();

    // Copy all state from device
    copyStateFromDevice();

    // Neuron
    EXPECT_TRUE(std::all_of(&num_testPost[0], &num_testPost[4], 
                            [](unsigned int n){ return (n == 4); }));
    EXPECT_TRUE(std::all_of(&num_batch_testPost[0], &num_batch_testPost[4], 
                            [](unsigned int n){ return (n == 1); }));

    // PSM
    EXPECT_TRUE(std::all_of(&num_psm_testSyn[0], &num_psm_testSyn[4], 
                            [](unsigned int n){ return (n == 4); }));
    EXPECT_TRUE(std::all_of(&num_batch_psm_testSyn[0], &num_batch_psm_testSyn[4], 
                            [](unsigned int n){ return (n == 1); }));

    // Current source
    EXPECT_TRUE(std::all_of(&num_cs_testCurrSource[0], &num_cs_testCurrSource[4], 
                            [](unsigned int n){ return (n == 4); }));
    EXPECT_TRUE(std::all_of(&num_batch_cs_testCurrSource[0], &num_batch_cs_testCurrSource[4], 
                            [](unsigned int n){ return (n == 1); }));
    
    // WUM pre
    EXPECT_TRUE(std::all_of(&num_wum_pre_testSyn[0], &num_wum_pre_testSyn[2], 
                            [](unsigned int n){ return (n == 2); }));
    EXPECT_TRUE(std::all_of(&num_batch_wum_pre_testSyn[0], &num_batch_wum_pre_testSyn[2], 
                            [](unsigned int n){ return (n == 1); }));
    
    // WUM post
    EXPECT_TRUE(std::all_of(&num_wum_post_testSyn[0], &num_wum_post_testSyn[4], 
                            [](unsigned int n){ return (n == 4); }));
    EXPECT_TRUE(std::all_of(&num_batch_wum_post_testSyn[0], &num_batch_wum_post_testSyn[4], 
                            [](unsigned int n){ return (n == 1); }));
    
    // WUM syn
    EXPECT_TRUE(std::all_of(&num_pre_wum_syn_testSyn[0], &num_pre_wum_syn_testSyn[8], 
                            [](unsigned int n){ return (n == 2); }));
    EXPECT_TRUE(std::all_of(&num_post_wum_syn_testSyn[0], &num_post_wum_syn_testSyn[8], 
                            [](unsigned int n){ return (n == 4); }));
    EXPECT_TRUE(std::all_of(&num_batch_wum_syn_testSyn[0], &num_batch_wum_syn_testSyn[8], 
                            [](unsigned int n){ return (n == 1); }));
    
    // CU
    EXPECT_TRUE(std::all_of(&num_testCU[0], &num_testCU[4], 
                            [](unsigned int n){ return (n == 4); }));
    EXPECT_TRUE(std::all_of(&num_batch_testCU[0], &num_batch_testCU[4], 
                            [](unsigned int n){ return (n == 1); }));
    
    // CU WUM
    EXPECT_TRUE(std::all_of(&num_pre_testCUWM[0], &num_pre_testCUWM[8], 
                            [](unsigned int n){ return (n == 2); }));
    EXPECT_TRUE(std::all_of(&num_post_testCUWM[0], &num_post_testCUWM[8], 
                            [](unsigned int n){ return (n == 4); }));
    EXPECT_TRUE(std::all_of(&num_batch_testCUWM[0], &num_batch_testCUWM[8], 
                            [](unsigned int n){ return (n == 1); }));
}

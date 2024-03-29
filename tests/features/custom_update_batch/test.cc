//--------------------------------------------------------------------------
/*! \file custom_update_batch/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Standard C++ includes
#include <algorithm>
#include <array>
#include <numeric>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "custom_update_batch_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

template<typename Predicate>
void checkSparseVar(scalar *var, Predicate predicate)
{
    for(unsigned int i = 0; i < 50; i++) {
        const unsigned int rowStart = maxRowLengthSparse * i;
        const unsigned int rowLength = rowLengthSparse[i];
        ASSERT_TRUE(std::all_of(&var[rowStart], &var[rowStart + rowLength], predicate));
    }
}

TEST_F(SimTest, CustomUpdateBatch)
{
    pullSparseConnectivityFromDevice();

    while(iT < 1000) {
        StepGeNN();

        if((iT % 100) == 0) {
            // Perform custom update
            updateTest();

            // Pull variables
            pullVNeuronDuplicateSetTimeFromDevice();
            pullVWUMDenseDuplicateSetTimeFromDevice();
            pullVWUMSparseDuplicateSetTimeFromDevice();
            pullVWUMKernelDuplicateSetTimeFromDevice();
            pullVNeuronFromDevice();
            pullUNeuronFromDevice();
            pullSNeuronFromDevice();
            pullVDenseFromDevice();
            pullUDenseFromDevice();
            pullVSparseFromDevice();
            pullUSparseFromDevice();
            pullVKernelFromDevice();
            pullUKernelFromDevice();
            
            // Check shared neuron and synapse variables match time
            ASSERT_TRUE(std::all_of(&UNeuron[0], &UNeuron[50],
                        [](scalar v) { return v == t; }));
            ASSERT_TRUE(std::all_of(&UDense[0], &UDense[50 * 50],
                        [](scalar v) { return v == t; }));
            ASSERT_TRUE(std::all_of(&UKernel[0], &UKernel[3 * 3],
                        [](scalar v) { return v == t; }));
            checkSparseVar(USparse, [](scalar v){ return v == t; });
    
            // Loop through batches
            for(unsigned int b = 0; b < 5; b++) {
                const unsigned int startNeuronIdx = b * 50;
                const unsigned int endNeuronIdx = startNeuronIdx + 50;
                const unsigned int startDenseSynIdx = b * (50 * 50);
                const unsigned int endDenseSynIdx = startDenseSynIdx + (50 * 50);
                const unsigned int startKernelIdx = b * (3 * 3);
                const unsigned int endKernelIdx = startKernelIdx + (3 * 3);
                const unsigned int startSparseSynIdx = b * (50 * maxRowLengthSparse);
                const float batchOffset = b * 1000.0f;
                
                ASSERT_EQ(SNeuron[b], batchOffset + t);
                // Check batched variables match expectations
                ASSERT_TRUE(std::all_of(&VNeuronDuplicateSetTime[startNeuronIdx], &VNeuronDuplicateSetTime[endNeuronIdx],
                            [batchOffset](scalar v) { return v == (batchOffset + t); }));
                ASSERT_TRUE(std::all_of(&VNeuron[startNeuronIdx], &VNeuron[endNeuronIdx],
                            [batchOffset](scalar v) { return v == (batchOffset + t); }));
                
                ASSERT_TRUE(std::all_of(&VWUMDenseDuplicateSetTime[startDenseSynIdx], &VWUMDenseDuplicateSetTime[endDenseSynIdx],
                            [batchOffset](scalar v) { return v == (batchOffset + t); }));
                ASSERT_TRUE(std::all_of(&VDense[startDenseSynIdx], &VDense[endDenseSynIdx],
                            [batchOffset](scalar v) { return v == (batchOffset + t); }));

                ASSERT_TRUE(std::all_of(&VWUMKernelDuplicateSetTime[startKernelIdx], &VWUMKernelDuplicateSetTime[endKernelIdx],
                            [batchOffset](scalar v) { return v == (batchOffset + t); }));
                ASSERT_TRUE(std::all_of(&VKernel[startKernelIdx], &VKernel[endKernelIdx],
                            [batchOffset](scalar v) { return v == (batchOffset + t); }));

                checkSparseVar(&VSparse[startSparseSynIdx],
                               [batchOffset](scalar v) { return v == (batchOffset + t); });
                checkSparseVar(&VWUMSparseDuplicateSetTime[startSparseSynIdx],
                               [batchOffset](scalar v) { return v == (batchOffset + t); });
                
            }
        }
    }
}


//--------------------------------------------------------------------------
/*! \file custom_update_reduction_batch_one/test.cc

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
#include "custom_update_reduction_batch_one_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
public:
    virtual void Init()
    {
        // Initialise variables to reduce
        std::iota(&VNeuron[0], &VNeuron[50], 0.0f);
        std::iota(&VDense[0], &VDense[50 * 50], 0.0f);
        
        pullSparseConnectivityFromDevice();
        for(unsigned int i = 0; i < 50; i++) {
            const unsigned rowStartIdx = i * maxRowLengthSparse;
            for(unsigned int j = 0 ; j < rowLengthSparse[i]; j++) {
                const unsigned int synIdx = rowStartIdx + j;
                VSparse[synIdx] = (scalar)synIdx;
            }
        }
    }
};

TEST_F(SimTest, CustomUpdateReductionBatchOne)
{
    // Launch reduction
    updateTest();

    // Download reductions
    pullNeuronReduceStateFromDevice();
    pullWUMDenseReduceStateFromDevice();
    pullWUMSparseReduceStateFromDevice();

    // Check neuron reductions
    for(unsigned int i = 0; i < 50; i++) {
        ASSERT_EQ(SumNeuronReduce[i], (float)i);
        ASSERT_EQ(MaxNeuronReduce[i], (float)i);
    }

    // Check dense weight reductions
    for(unsigned int i = 0; i < (50 * 50); i++) {
        ASSERT_EQ(SumWUMDenseReduce[i], (float)i);
        ASSERT_EQ(MaxWUMDenseReduce[i], (float)i);
    }

    // Check sparse weight reductions
    for(unsigned int i = 0; i < 50; i++) {
        const unsigned rowStartIdx = i * maxRowLengthSparse;
        for(unsigned int j = 0 ; j < rowLengthSparse[i]; j++) {
            const unsigned int synIdx = rowStartIdx + j;
            ASSERT_EQ(SumWUMSparseReduce[synIdx], synIdx);
            ASSERT_EQ(MaxWUMSparseReduce[synIdx], synIdx);
        }
    }
}


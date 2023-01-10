//--------------------------------------------------------------------------
/*! \file custom_update_neuron_reduction_batch_one/test.cc

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
#include "custom_update_neuron_reduction_batch_one_CODE/definitions.h"

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
        std::iota(&YNeuron[0], &YNeuron[50], 0.0f);
    }
};

TEST_F(SimTest, CustomUpdateNeuronReductionBatchOne)
{
    // Perform three step softmax
    updateSoftmax1();
    updateSoftmax2();
    updateSoftmax3();

    // Check max reduction
    pullSoftmax1StateFromDevice();
    const float maxY = *std::max_element(&YNeuron[0], &YNeuron[50]);
    ASSERT_FLOAT_EQ(MaxYSoftmax1[0], maxY);
    
    // Calculate sum of exponentials
    pullSoftmax2StateFromDevice();
    const float sumExp = std::accumulate(&YNeuron[0], &YNeuron[50], 0.0f,
                                         [maxY](float acc, float y){ return acc + exp(y - maxY); });
    ASSERT_FLOAT_EQ(SumExpPiSoftmax2[0], sumExp);
    
    // Calculate final softmax values
    pullNeuronStateFromDevice();
    for(int i = 0; i < 50; i++) {
        ASSERT_FLOAT_EQ(exp(YNeuron[i] - maxY) / sumExp, PiNeuron[i]);
    }
    
   
}


//--------------------------------------------------------------------------
/*! \file custom_update_neuron_reduction_batch_one/test.cc

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
        std::iota(&VNeuron[0], &VNeuron[50], 0.0f);
    }
};

TEST_F(SimTest, CustomUpdateNeuronReductionBatchOne)
{
    // Launch reduction
    updateTest();

    // Download reductions
    pullNeuronReduceStateFromDevice();

    ASSERT_FLOAT_EQ(SumNeuronReduce[0], 1225.0);
    ASSERT_FLOAT_EQ(MaxNeuronReduce[0], 49.0f);
   
}


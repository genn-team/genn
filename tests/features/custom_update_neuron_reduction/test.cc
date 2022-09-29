//--------------------------------------------------------------------------
/*! \file custom_update_neuron_reduction/test.cc

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
#include "custom_update_neuron_reduction_CODE/definitions.h"

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
        std::iota(&VNeuron[0], &VNeuron[50 * 5], 0.0f);
    }
};

TEST_F(SimTest, CustomUpdateNeuronReduction)
{
    // Launch reduction
    updateTest();

    // Download reductions
    pullNeuronReduceStateFromDevice();

    for(unsigned int b = 0; b < 5; b++) {
        const float *begin = &VNeuron[b * 50];
        const float *end = begin + 50;
        ASSERT_FLOAT_EQ(SumNeuronReduce[b], std::accumulate(begin, end, 0.0f));
        ASSERT_FLOAT_EQ(MaxNeuronReduce[b], *std::max_element(begin, end));
    }
   
}


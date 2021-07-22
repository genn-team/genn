//--------------------------------------------------------------------------
/*! \file custom_update_reduction/test.cc

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
#include "custom_update_reduction_CODE/definitions.h"

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
        std::iota(&VNeuron[0], &VNeuron[50 * 5], 0);
    }
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

TEST_F(SimTest, CustomUpdateReduction)
{
    // Launch reduction
    updateTest();
    
    // Download reductions
    pullSumNeuronReduceAddFromDevice();
    pullMaxNeuronReduceMaxFromDevice();
    
    for(unsigned int i = 0; i < 50; i++) {
        ASSERT_EQ(SumNeuronReduceAdd[i], (5 * (i + (4 * 50) + i)) / 2);
    }
}


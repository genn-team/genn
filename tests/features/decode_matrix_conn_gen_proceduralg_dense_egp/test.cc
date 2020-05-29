//--------------------------------------------------------------------------
/*! \file decode_matrix_conn_gen_proceduralg_dense_egp/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
#include <numeric>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "decode_matrix_conn_gen_proceduralg_dense_egp_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
public:
    //------------------------------------------------------------------------
    // SimulationTest virtuals
    //------------------------------------------------------------------------
    virtual void Init() override
    {
        allocaterowWeightsgSyn(4);
        std::iota(&rowWeightsgSyn[0], &rowWeightsgSyn[4], 1);
        pushrowWeightsgSynToDevice(4);
    }
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    float Simulate()
    {
        float error = 0;
        for (int i = 0; i < (int)(10.0f / DT); i++) {
            // What value should neurons be representing this time step?
            const unsigned int inValue = (i / 10);

            // Input spike representing value
            // **NOTE** neurons start from zero
            glbSpkCntPre[0] = 1;
            glbSpkPre[0] = inValue;

            // Push spikes to device
            pushPreSpikesToDevice();

            // Step GeNN
            StepGeNN();

            // Loop through output neurons
            for(unsigned int j = 0; j < 4; j++) {
                error += std::fabs(xPost[j] - (float)(inValue * (j + 1)));
            }
        }

        return error;
    }
};

TEST_F(SimTest, DecodeMatrixConnGenProceduralgDenseEGP)
{
    // Check total error is less than some tolerance
    EXPECT_LT(Simulate(), 2e-2);
}

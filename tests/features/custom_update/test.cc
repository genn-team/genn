//--------------------------------------------------------------------------
/*! \file custom_update/test.cc

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
#include "custom_update_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
};

TEST_F(SimTest, CustomUpdate)
{
    while(iT < 1000) {
        StepGeNN();

        if((iT % 100) == 0) {
            // Perform custom update
            updateTest();

            // Pull variables
            pullVNeuronSetTimeFromDevice();
            pullVNeuronFromDevice();
            pullVCurrentSourceSetTimeFromDevice();
            pullCCurrentSourceFromDevice();
            pullVPSMSetTimeFromDevice();
            pullPDenseFromDevice();
            pullVWUPreSetTimeFromDevice();
            pullPreDenseFromDevice();
            pullVWUPostSetTimeFromDevice();
            pullPostSparseFromDevice();
            pullVWUDenseSetTimeFromDevice();
            pullgDenseFromDevice();
            pullVWUSparseSetTimeFromDevice();
            pullgSparseFromDevice();
            pullSparseConnectivityFromDevice();

            // Check all values match time of update
            EXPECT_TRUE(std::all_of(&VNeuron[0], &VNeuron[100],
                        [](scalar v) { return v == t; }));
            EXPECT_TRUE(std::all_of(&VNeuronSetTime[0], &VNeuronSetTime[100],
                        [](scalar v) { return v == t; }));

            EXPECT_TRUE(std::all_of(&VCurrentSourceSetTime[0], &VCurrentSourceSetTime[100],
                        [](scalar v) { return v == t; }));
            EXPECT_TRUE(std::all_of(&CCurrentSource[0], &CCurrentSource[100],
                        [](scalar v) { return v == t; }));

            EXPECT_TRUE(std::all_of(&VPSMSetTime[0], &VPSMSetTime[100],
                        [](scalar v) { return v == t; }));
            EXPECT_TRUE(std::all_of(&PDense[0], &PDense[100],
                        [](scalar v) { return v == t; }));

            EXPECT_TRUE(std::all_of(&VWUPreSetTime[0], &VWUPreSetTime[100],
                        [](scalar v) { return v == t; }));
            EXPECT_TRUE(std::all_of(&PreDense[0], &PreDense[100],
                        [](scalar v) { return v == t; }));

            EXPECT_TRUE(std::all_of(&VWUPostSetTime[0], &VWUPostSetTime[100],
                        [](scalar v) { return v == t; }));
            EXPECT_TRUE(std::all_of(&PostSparse[0], &PostSparse[100],
                        [](scalar v) { return v == t; }));

            EXPECT_TRUE(std::all_of(&VPSMSetTime[0], &VPSMSetTime[100],
                        [](scalar v) { return v == t; }));
            EXPECT_TRUE(std::all_of(&PDense[0], &PDense[100],
                        [](scalar v) { return v == t; }));

            EXPECT_TRUE(std::all_of(&VWUDenseSetTime[0], &VWUDenseSetTime[100 * 100],
                        [](scalar v) { return v == t; }));
            EXPECT_TRUE(std::all_of(&gDense[0], &gDense[100 * 100],
                        [](scalar v) { return v == t; }));

            for(unsigned int i = 0; i < 100; i++) {
                const unsigned int rowStartIdx = maxRowLengthSparse * i;
                const unsigned int rowEndIdx = rowStartIdx + rowLengthSparse[i];

                EXPECT_TRUE(std::all_of(&VWUSparseSetTime[rowStartIdx], &VWUSparseSetTime[rowEndIdx],
                            [](scalar v) { return v == t; }));
                EXPECT_TRUE(std::all_of(&gSparse[rowStartIdx], &gSparse[rowEndIdx],
                            [](scalar v) { return v == t; }));
            }

        }
    }
}


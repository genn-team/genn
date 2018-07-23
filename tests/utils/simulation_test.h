#pragma once

// Google test includes
#include "gtest/gtest.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
// **YUCK** in order to pass prefix from
#define WRAPPED_INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator) INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator)

// **YUCK** need wrapper to paste together init and name macro
#define TOKENPASTE(x, y) x ## y
#define INIT_SPARSE(N) TOKENPASTE(init, N)()

//----------------------------------------------------------------------------
// SimulationTest
//----------------------------------------------------------------------------
class SimulationTest : public ::testing::TestWithParam<bool>
{
protected:
    //--------------------------------------------------------------------------
    // test virtuals
    //--------------------------------------------------------------------------
    virtual void SetUp()
    {
        // Perform GeNN initialization
        allocateMem();
        initialize();

        Init();

#ifndef CPU_ONLY
        if(GetParam())
        {
            // Copy state, initialised on host to device
            // **NOTE** this is done at the end of initialize/init_model_name function so is PROBABLY unnecessary
            copyStateToDevice(true);
        }
#endif  // CPU_ONLY
    }

    virtual void TearDown()
    {
        freeMem();
    }

    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    virtual void Init() = 0;

    //--------------------------------------------------------------------------
    // Protected methods
    //--------------------------------------------------------------------------
    void StepGeNN()
    {
#ifdef CPU_ONLY
        stepTimeCPU();
#else   // CPU_ONLY
    #ifndef CPU_GPU_NOT_SUPPORTED
        if(!GetParam()) {
            stepTimeCPU();
        }
        else
    #else   // CPU_NOT_SUPPORTED
        {
            stepTimeGPU();
            copyStateFromDevice();
        }
    #endif  // CPU_NOT_SUPPORTED
#endif  // CPU_ONLY
    }
};

#pragma once

// Google test includes
#include "gtest/gtest.h"

//----------------------------------------------------------------------------
// SimulationTest
//----------------------------------------------------------------------------
class SimulationTest : public ::testing::Test
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

        initializeSparse();
    }

    virtual void TearDown()
    {
        freeMem();
    }

    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    virtual void Init(){}

    //--------------------------------------------------------------------------
    // Protected methods
    //--------------------------------------------------------------------------
    void StepGeNN()
    {
        stepTime();
        copyStateFromDevice();
    }
};

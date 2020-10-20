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
        
        PreInit();
        
        initialize();

        Init();

        initializeSparse();
        copyStateFromDevice();
    }

    virtual void TearDown()
    {
        freeMem();
    }

    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    virtual void PreInit(){}
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

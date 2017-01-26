#pragma once

// Google test includes
#include "gtest/gtest.h"

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
      copyStateToDevice();
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
  void Step()
  {
#ifndef CPU_ONLY
    if(GetParam())
    {
      stepTimeGPU();
      copyStateFromDevice();
    }
    else
#endif  // CPU_ONLY
    {
      stepTimeCPU();
    }
  }
};

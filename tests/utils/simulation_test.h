#pragma once

// Google test includes
#include "gtest/gtest.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
// **YUCK** in order to pass prefix from
#define WRAPPED_INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator) INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator)

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
  void StepGeNN()
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

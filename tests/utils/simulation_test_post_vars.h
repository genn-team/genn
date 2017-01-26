#pragma once

#include "simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestPostVars
//----------------------------------------------------------------------------
class SimulationTestPostVars : public SimulationTest
{
protected:
  //--------------------------------------------------------------------------
  // SimulationTest virtuals
  //--------------------------------------------------------------------------
  virtual void Init()
  {
    // Initialise neuron parameters
    for (int i = 0; i < 10; i++)
    {
      shiftpre[i] = i * 10.0f;
      shiftpost[i] = i * 10.0f;
    }

    m_TheW[0] = wsyn0;
    m_TheW[1] = wsyn1;
    m_TheW[2] = wsyn2;
    m_TheW[3] = wsyn3;
    m_TheW[4] = wsyn4;
    m_TheW[5] = wsyn5;
    m_TheW[6] = wsyn6;
    m_TheW[7] = wsyn7;
    m_TheW[8] = wsyn8;
    m_TheW[9] = wsyn9;
  };

  //--------------------------------------------------------------------------
  // Protected methods
  //--------------------------------------------------------------------------
  float *GetTheW(unsigned int delay) const
  {
      return m_TheW[delay];
  }

  void SetTheW(unsigned int i, unsigned int j, float value)
  {
      m_TheW[i][j] = value;
  }

private:
  //--------------------------------------------------------------------------
  // Members
  //--------------------------------------------------------------------------
  float *m_TheW[10];
};

#pragma once

// Standard includes
#include <functional>
#include <numeric>

// Test includes
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
  template<typename ShouldUpdateFn>
  float Simulate(ShouldUpdateFn shouldUpdate)
  {
      float err = 0.0f;
      float x[10][100];
      for (int i = 0; i < (int)(20.0f / DT); i++)
      {
        t = i * DT;

        // for each delay
        for (int d = 0; d < 10; d++)
        {
            // for all pre-synaptic neurons
            for (int j = 0; j < 10; j++)
            {
                // for all post-syn neurons
                for (int k = 0; k < 10; k++)
                {
                    if (shouldUpdate(t))
                    {
                        x[d][(j * 10) + k] = t-2*DT+10*k;
                    }
                    else if(i == 0)
                    {
                        x[d][(j * 10) + k] = 0.0f;
                    }
                }
            }

            // Add error for this time step to total
            err += std::inner_product(&x[d][0], &x[d][100],
                                      GetTheW(d),
                                      0.0,
                                      std::plus<float>(),
                                      [](double a, double b){ return abs(a - b); });
        }

        // Step simulation
        Step();
      }

      return err;
  }

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

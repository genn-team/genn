#pragma once

#include "simulation_test_post_vars.h"


//----------------------------------------------------------------------------
// SimulationTestPostVarsSparse
//----------------------------------------------------------------------------
class SimulationTestPostVarsSparse : public SimulationTestPostVars
{
protected:
  //--------------------------------------------------------------------------
  // SimulationTest virtuals
  //--------------------------------------------------------------------------
  virtual void Init()
  {
      #define SETUP_THE_C(I)  \
        case I:               \
          allocatesyn##I(10); \
          theC= &Csyn##I;     \
          break;

      // all different delay groups get same connectivity
      for(int i = 0; i < 10; i++)
      {
          // **YUCK** extract correct sparse projection
          SparseProjection *theC;
          switch (i)
          {
            SETUP_THE_C(0)
            SETUP_THE_C(1)
            SETUP_THE_C(2)
            SETUP_THE_C(3)
            SETUP_THE_C(4)
            SETUP_THE_C(5)
            SETUP_THE_C(6)
            SETUP_THE_C(7)
            SETUP_THE_C(8)
            SETUP_THE_C(9)
          };

          // loop through pre-synaptic neurons
          for(int j = 0; j < 10; j++)
          {
              // each pre-synatic neuron gets one target neuron
              unsigned int trg= (j + 1) % 10;
              theC->indInG[j]= j;
              theC->ind[j]= trg;
          }
          theC->indInG[10]= 10;
      }

      // Superclass
      SimulationTestPostVars::Init();

      // for all synapse groups
      for(int i = 0; i < 10; i++)
      {
          // for all synapses
          for(int j = 0; j < 10; j++)
          {
              SetTheW(i, j, 0.0f);
          }
      }
  }
};
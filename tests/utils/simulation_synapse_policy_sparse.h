#pragma once

#include "simulation_synapse_policy_dense.h"

// Standard includes
#include <functional>
#include <numeric>

//----------------------------------------------------------------------------
// SimulationSynapsePolicySparse
//----------------------------------------------------------------------------
class SimulationSynapsePolicySparse : public SimulationSynapsePolicyDense
{
public:
  //----------------------------------------------------------------------------
  // Public API
  //----------------------------------------------------------------------------
  void Init()
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
      SimulationSynapsePolicyDense::Init();

      // for all synapse groups
      for(int i = 0; i < 10; i++)
      {
          // for all synapses
          for(int j = 0; j < 10; j++)
          {
              SetTheW(i, j, 0.0f);
          }
      }
#ifndef CPU_ONLY
      initializeAllSparseArrays();
#endif  // CPU_ONLY
  }

  template<typename UpdateFn, typename StepGeNNFn>
  float Simulate(UpdateFn updateFn, StepGeNNFn stepGeNNFn)
  {
      float err = 0.0f;
      float x[10][10];
      for (int i = 0; i < (int)(20.0f / DT); i++)
      {
        // **YUCK** update global time - this shouldn't be user responsibility
        t = i * DT;

        // for each delay
        for (int d = 0; d < 10; d++)
        {
            // for all pre-synaptic neurons
            for (int j = 0; j < 10; j++)
            {
                float newX;
                if(updateFn(i, d, j, t, newX))
                {
                    x[d][j] = newX;
                }
                else if(i == 0)
                {
                    x[d][j] = 0.0f;
                }
            }

            // Add error for this time step to total
            err += std::inner_product(&x[d][0], &x[d][10],
                                      GetTheW(d),
                                      0.0f,
                                      std::plus<float>(),
                                      [](float a, float b){ return abs(a - b); });
         }

        // Step GeNN
        stepGeNNFn();
      }

      return err;
  }
};

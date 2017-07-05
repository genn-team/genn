// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include DEFINITIONS_HEADER

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_vars.h"
#include "../../utils/simulation_neuron_policy_pre_var.h"
#include "../../utils/simulation_synapse_policy_sparse.h"

//----------------------------------------------------------------------------
// SimulationSynapsePolicy
//----------------------------------------------------------------------------
class SimulationSynapsePolicy : public SimulationSynapsePolicySparse
{
public:
  //----------------------------------------------------------------------------
  // Public API
  //----------------------------------------------------------------------------
  void Init()
  {
      // Superclass
      SimulationSynapsePolicySparse::Init();

      // Create array pointing to thresholds
      m_TheThresh[0] = &threshsyn0;
      m_TheThresh[1] = &threshsyn1;
      m_TheThresh[2] = &threshsyn2;
      m_TheThresh[3] = &threshsyn3;
      m_TheThresh[4] = &threshsyn4;
      m_TheThresh[5] = &threshsyn5;
      m_TheThresh[6] = &threshsyn6;
      m_TheThresh[7] = &threshsyn7;
      m_TheThresh[8] = &threshsyn8;
      m_TheThresh[9] = &threshsyn9;
  }

  template<typename UpdateFn, typename StepGeNNFn>
  float Simulate(UpdateFn updateFn, StepGeNNFn stepGeNNFn)
  {
      const float swapT = 20.0f / 2.0f;
      float theSwap = 20.0f;

      // Reset threshold
      for(int k = 0; k < 10; k++)
      {
          *(m_TheThresh[k]) = 2.0f * (k + 1);
      }

      float err = 0.0f;
      float x[10][10];
      for (int i = 0; i < (int)(20.0f / DT); i++)
      {
          t = i * DT;

          // If swapping point has been reached
          if(abs(t - swapT) < DT)
          {
              // Update threshold
              for(int k = 0; k < 10; k++)
              {
                  *(m_TheThresh[k]) = 2.0f + (3 * k);
              }

              theSwap = t;
          }

          // for each delay
          for (int d = 0; d < 10; d++)
          {
              // for all pre-synaptic neurons
              for (int j = 0; j < 10; j++)
              {
                  float newX;
                  float evntT = t-2*DT-d*DT+5e-5;
                  if(updateFn(i, d, j, t, evntT, (evntT < theSwap), newX))
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

private:
  //--------------------------------------------------------------------------
  // Members
  //--------------------------------------------------------------------------
  float *m_TheThresh[10];
};

// Combine neuron and synapse policies together to build variable-testing fixture
typedef SimulationTestVars<SimulationNeuronPolicyPreVar, SimulationSynapsePolicy> SimTest;


TEST_P(SimTest, AcceptableError)
{
    float err = Simulate(
      [](unsigned int i, unsigned int d, unsigned int j, float t, float evntT, bool beforeSwap, float &newX)
      {
          if(beforeSwap)
          {
              if ((t > d*DT+0.1001) && (fmod(evntT+10*j,(float) (2*(d+1))) < 1e-4))
              {
                  newX = t-2*DT-d*DT+10*j;
                  return true;
              }
          }
          else
          {
              if((t > d*DT+0.1001) && (fmod(evntT+10*j,(float) (2.0f+3*d)) < 1e-4))
              {
                  newX = t-2*DT-d*DT+10*j;
                  return true;
              }
          }

          return false;
      });

  // Check total error is less than some tolerance
  EXPECT_LT(err, 1e-3);
}

#ifndef CPU_ONLY
auto simulatorBackends = ::testing::Values(true, false);
#else
auto simulatorBackends = ::testing::Values(false);
#endif

WRAPPED_INSTANTIATE_TEST_CASE_P(MODEL_NAME,
                                SimTest,
                                simulatorBackends);
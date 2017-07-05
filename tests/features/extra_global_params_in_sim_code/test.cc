// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include DEFINITIONS_HEADER

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test_vars.h"
#include "../../utils/simulation_neuron_policy_pre_var.h"
#include "../../utils/simulation_synapse_policy_none.h"

// Combine neuron and synapse policies together to build variable-testing fixture
typedef SimulationTestVars<SimulationNeuronPolicyPreVar, SimulationSynapsePolicyNone> SimTest;


TEST_P(SimTest, AcceptableError)
{
    float err = Simulate(
      [](unsigned int i, unsigned int j, float t, float &newX)
      {
          if(i > 0)
          {
              newX = (t - DT) + pow(t - DT, 2.0) + (j * 10);
              return true;
          }
          else
          {
            return false;
          }
      });

  // Check total error is less than some tolerance
  EXPECT_LT(err, 2e-2);
}

#ifndef CPU_ONLY
auto simulatorBackends = ::testing::Values(true, false);
#else
auto simulatorBackends = ::testing::Values(false);
#endif

WRAPPED_INSTANTIATE_TEST_CASE_P(MODEL_NAME,
                                SimTest,
                                simulatorBackends);
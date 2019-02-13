//--------------------------------------------------------------------------
/*! \file extra_global_params_in_sim_code/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
// Standard C include
#include <cmath>

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


TEST_F(SimTest, ExtraGlobalParamsInSimCode)
{
    float err = Simulate(
      [](unsigned int i, unsigned int j, float t, float &newX)
      {
          if(i > 0) {
              newX = (t - DT) + std::pow(t - DT, 2.0) + (j * 10);
              return true;
          }
          else {
            return false;
          }
      });

  // Check total error is less than some tolerance
  EXPECT_LT(err, 2e-2);
}

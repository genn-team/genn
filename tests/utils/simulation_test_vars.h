#pragma once

// Test includes
#include "simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestVars
//----------------------------------------------------------------------------
template<typename NeuronPolicy, typename SynapsePolicy>
class SimulationTestVars : public SimulationTest
{
protected:
  //--------------------------------------------------------------------------
  // SimulationTest virtuals
  //--------------------------------------------------------------------------
  virtual void Init()
  {
    m_NeuronPolicy.Init();
    m_SynapsePolicy.Init();
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------
  template<typename UpdateFn>
  float Simulate(UpdateFn update)
  {
      return m_SynapsePolicy.Simulate(update,
                                     [this](){ StepGeNN(); });
  }

private:
  // -------------------------------------------------------------------------
  // Members
  // -------------------------------------------------------------------------
  NeuronPolicy m_NeuronPolicy;
  SynapsePolicy m_SynapsePolicy;
};
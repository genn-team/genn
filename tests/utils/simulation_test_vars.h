#pragma once

// Test includes
#include "simulation_test.h"

#define ASSIGN_ARRAY_VARS(ARRAY_NAME, VAR_PREFIX, COUNT)  \
    for(int i_##__LINE__ = 0; i_##__LINE__ < COUNT; i++)  \
    {                                                     \
        ARRAY_NAME[i_##__LINE__] = VAR_PREFIX##i_##__LINE__; \
    }
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

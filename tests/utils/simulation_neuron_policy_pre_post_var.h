#pragma once

//----------------------------------------------------------------------------
// SimulationNeuronPolicyPrePostVar
//----------------------------------------------------------------------------
class SimulationNeuronPolicyPrePostVar
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void Init()
    {
        // Initialise neuron parameters
        for (int i = 0; i < 10; i++) {
            shiftpre[i] = i * 10.0f;
            shiftpost[i] = i * 10.0f;
        }
    }
};
